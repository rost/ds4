# The SYCL Backend

This is a fourth GPU backend for ds4, alongside Metal, CUDA and ROCm, targeting
Intel GPUs through SYCL 2020 and Intel's DPC++ compiler. It exists to run
DeepSeek V4 Flash on Intel Arc and Battlemage hardware.

Scope is deliberately narrow: **DeepSeek V4 Flash only.** No PRO variant, no
GLM. Entries belonging to those model families are stubbed and stay stubbed.

Build it with `make sycl`. Run its tests with `make test-sycl`.

## Where it sits

ds4 keeps its engine in one large C file and talks to whichever GPU backend was
compiled in through a flat C ABI. No backend knows about any other.

```
                        ds4.c
             (engine: graphs, sessions, sampling)
                          |
                          |  flat C ABI, ~340 entry points
                          |  ds4_gpu.h + ds4_gpu_mgpu.h
          +---------------+---------------+---------------+
          |               |               |               |
     ds4_metal.m     ds4_cuda.cu     ds4_rocm.cu      ds4_sycl.cpp
       (Apple)        (NVIDIA)          (AMD)          (Intel)   <-- this one
```

The ABI is the only contract. `ds4.c` calls an entry, checks its return value,
and falls back or fails. A backend that has not implemented an entry links a
stub that reports failure, and the engine copes.

**The ROCm backend is this port's structural reference, not CUDA.** ROCm and
CUDA diverge in what they implement and how they dispatch, and ROCm is the
closer analogue: it hand-writes its kernels rather than vendoring llama.cpp's
mmq tier, which is the shape this port follows.

## The two files that are not kernels

```
  ds4_sycl.cpp  (571 lines)          ds4_sycl_unavailable.cpp  (243 lines)
  +--------------------------+       +------------------------------+
  | device + queue discovery |       | 64 stubs for entries this    |
  | g_devices, g_current_tier|       | backend does not implement   |
  |                          |       |                              |
  | tensor ABI:              |       | SYCL_UNAVAILABLE_NONZERO_OK  |
  |   alloc / view / free    |       | SYCL_UNAVAILABLE_ZERO_OK     |
  |   read / write / copy    |       | SYCL_UNAVAILABLE_VOID        |
  |   alloc_ptr_on(tier)     |       |                              |
  |                          |       | Each macro returns THAT      |
  | init / cleanup           |       | entry's own failure value.   |
  +--------------------------+       +------------------------------+
              |
              | #include, at the END of the file
              v
  +--------------------------------------------------------------+
  |  sycl/*.hpp   -- one header per subsystem, ~8,500 lines       |
  +--------------------------------------------------------------+
```

The kernel headers are `#include`d at the **end** of `ds4_sycl.cpp`, not the
top. That is deliberate: they reference `g_devices` and `ds4_sycl_queue()`, and
several call ABI entries defined earlier in the same translation unit. One
translation unit, one compile, no device linking.

### Why the stub file is not just `return 0`

**ds4's ABI has no uniform success convention.** Most entries return nonzero
for success. A minority return zero for success (`ds4_gpu_set_current_device`,
`ds4_gpu_args_probe_auto_cuda`). Two use a third convention entirely
(`ds4_gpu_decode_graph_begin`/`_end`: 0 committed, -1 failed).

A stub must return **its own entry's failure value**, or it silently reports
success for work that never happened. The three macros make that choice
explicit at each line instead of hiding it behind a shared `return 0`.

## The subsystems

```
  sycl/ds4_sycl_common.hpp        231   shared helpers, no kernels
    |
    +-- overflow-checked size arithmetic
    +-- model-range bounds checks
    +-- sycl_device_scratch_guard   (RAII for device scratch)
    +-- sycl_stage_host_bytes       (host mmap -> device, see below)
    +-- sycl_block_row_reduce       (work-group tree reduction)
    +-- Q8_0 block decode

  ds4_sycl_output.hpp         233   softmax, swiglu, output projection
  ds4_sycl_embedding.hpp      287   token embedding, Q8_0 and HC
  ds4_sycl_norm_rope.hpp      637   RMS norm family, RoPE, YaRN scaling
  ds4_sycl_compressor.hpp     984   KV compression, APE, ratio-N replay
  ds4_sycl_matmul.hpp         360   dense Q8_0 and F16 matmul
  ds4_sycl_router.hpp         505   MoE expert selection
  ds4_sycl_shared_expert.hpp  640   the always-on expert MLP
  ds4_sycl_fp8_kv.hpp         351   E4M3 KV round trip, raw KV ring store
  ds4_sycl_streaming.hpp    1,177   expert weight cache, LRU + slab pool
  ds4_sycl_moe.hpp          2,352   routed MoE kernels, four quant formats
  ds4_sycl_moe_launch.hpp     712   routed MoE dispatcher
```

Roughly 240 ABI entries are implemented; 64 remain stubbed (62 via the macros
plus 2 hand-written). A stub reachability audit
(`ds4-sycl-stub-reachability-v2.md`) traced every one of them outward through
its enclosing preprocessor and runtime gates and found **none reachable with
consequence on the ordinary Flash decode path**, single-session or batched.

Most of what remains is permanently out of scope: GLM (35), DSpark, PRO, and
CUDA-only paths. The rest is tensor parallelism and a few multi-tier entries.

## How a call flows

Taking a routed-MoE decode as the example, since it touches the most machinery:

```
  ds4.c
    |  ds4_gpu_routed_moe_one_tensor(out, gate, up, mid, down,
    |                                model_map, model_size, ...)
    v
  ds4_sycl_moe_launch.hpp
    |
    |  1. validate arguments          <-- overflow-safe, per-entry polarity
    |  2. build a plan                <-- picks a format path from gate/down type
    |  3. RESOLVE THE WEIGHT POINTER  <-- device-resident? pass through.
    |                                     not cached? stage. See "weights"
    |  4. compact to the SELECTED experts only   <-- 6 of 256, not 256
    |  5. quantise activations to Q8_K
    |  6. dispatch on the format path
    v
        +-------------+-------------+-------------+-------------+
        |  q4k_path   |  iq2_path   |  q2k_path   | mxfp4_path  |
        +-------------+-------------+-------------+-------------+
                             |
                             v
  ds4_sycl_moe.hpp     gate/up kernel -> SwiGLU -> down kernel -> weighted sum
                             |
                             v
                        out tensor (device USM)
```

Each format block in the dispatcher is a clearly delimited `if (path) { ... }`
region. That is a structural decision, not an accident: three separate plans
added format paths to the same 2,000-line function, and the delimiters are what
let them do it without fighting over the same lines.

## The mmap trap

**This is the single most important thing to know before writing a kernel here.**

CUDA and HIP, under unified virtual addressing, let a kernel dereference an
arbitrary host pointer. The model file is mmapped, so their kernels read weights
straight from the mapping.

**SYCL on Level Zero does not.** A kernel handed a raw `model_map`-derived
pointer does not fault. It reads zeros and the launch reports success.

```
   CUDA / HIP                          SYCL / Level Zero
   ----------                          -----------------
   mmap(model) ---------------+        mmap(model)
                              |             |
                              |             |  malloc_device + memcpy
                              |             v
                              |        device USM copy
                              |             |
                              v             v
                          [ kernel ]    [ kernel ]
                          reads real    reads real
                            weights       weights

                                        Skip the copy and the kernel
                                        reads ZEROS, silently, and
                                        reports success.
```

This cost real time once: a MoE decode path returned all-zero output with no
exception raised, and the cause was one unstaged pointer.

### The second trap, one layer lower, and it is worse

Staging is not enough on its own. **A device memcpy whose source is a
file-backed mmap pointer can silently DMA zeros** for pages the CPU has not
faulted in yet. The copy engine reads the mapping before the kernel has
populated it, and reports success.

Every synthetic test builds its fixture in host memory the CPU just wrote, so
every page is resident and the race never fires. It only appears with a real
mmap'd model, and it appears more the larger the model, because fewer of its
pages will ever have been touched. On the real 80 GB Flash model, where most
pages are never CPU-touched by design, large regions would have arrived as
zeros.

`sycl_copy_host_to_device_paged_safe` stages every host-to-device copy through a
resident heap buffer first, forcing the pages to fault in on the CPU before the
DMA reads them. **Never hand an mmap pointer straight to a device copy.** Spec
6y has the details; `tests/test_sycl_gguf_load.c`, which builds a real GGUF on
disk, is the only test that can catch a regression here.

### Weights are device-resident now, so most calls stage nothing

`ds4_gpu_cache_model_range` allocates device memory for a range, copies it once
through the paged-safe helper, and registers it per tier. `sycl_model_range_ptr`
then resolves a request to that device pointer, sub-range containment included,
exactly as CUDA's `cuda_model_range_ptr` does.

All staging funnels through three helpers, and each passes through with no
allocation, no copy and no wait when handed a pointer that is already
device-resident:

| Helper | Used by |
|---|---|
| `sycl_stage_host_bytes` | dense, attention, norm, hc, compressor, fp8 kv |
| `sycl_moe_stage_weights` | routed MoE, full table |
| `sycl_moe_stage_selected_experts` | routed MoE, compacted |

Measured on a real 1.83 GiB GGUF: **94.6 ms and 195 MB staged per layer-eval
before, 18.8 ms and zero bytes after.**

The cache is **bounded** by `ds4_gpu_tier_free_vram`. An 80 GB model does not fit
one 24 GB card, so what fits is cached and the rest stages exactly as before.
Both paths are live and both must stay correct; on a single-GPU development box
the staging fallback is the one that actually runs.

## Testing, and what it cannot see

Every subsystem has a test binary that runs against a **synthetic model**: a
host buffer filled with a deterministic pattern, passed as `model_map`. No
weights are needed and none are downloaded.

```
   tests/test_sycl_<subsystem>.c
     |
     |  build a fake model buffer
     |  compute the expected answer with a scalar CPU oracle,
     |    transcribed from the matching function in ds4.c
     |  call the ABI entry
     |  compare
     v
   tests/test_sycl_harness.h    shared CHECK / CHECK_CLOSE
```

`make test-sycl` runs all of them. Every new test file must be added to that
aggregate target: three separate integrations have silently excluded one, and
a test nothing runs is worse than no test.

**Two things this cannot check.** MXFP4 has no CPU oracle in `ds4.c` at all, so
it is validated against an independently written scalar reference instead, which
is a weaker guarantee. And the same is true of indexed decode attention, where
no CPU function consumes a raw top-k array.

**The blind spot that mattered most.** Every test above calls a `ds4_gpu_*`
entry directly. None of them goes through the engine. So the entire class of
"the engine calls an entry this backend left stubbed to its failure value" was
invisible, and it hid three separate catastrophic defects: no session could be
created, no model could be opened, and no token could be decoded. Each was found
by reading, not by testing.

`tests/test_sycl_session_smoke.c` closes that gap from the other side. It uses
`DS4_TEST_HOOKS` entry points in `ds4.c` to drive the engine's own graph
allocator and then the real `begin_commands`, kernel, `end_commands`,
`synchronize` encode sequence, on a synthetic model with no weights. **Both of
its hooks found a defect the first time they ran.** If you add an engine-facing
entry, extend that file, not just a kernel test.

### Ablation

A passing test proves nothing until you have seen it fail. After a test passes,
the implementation is deliberately broken and the test re-run; if it still
passes, the test is blind and gets strengthened.

This is not ceremony. Seven tests in this project have passed while unable to
catch the defect they existed for, and two live bugs reached the branch behind
that blindness. The failure modes are catalogued in the project spec
(`ds4-sycl-spec.md`, sections 6b through 6n) and include: uninitialised local
memory reading as zero on this hardware, normalisation dividing out a scale-only
error, a reduction whose topology resolves the chosen indices correctly by
accident, a launch too small to expose a race, and rounding ties that resolve
differently on GPU and host.

## Build

```
  make sycl
    |
    +-- ds4.c, ds4_ssd.c, ... -----> cc -DDS4_SYCL_BUILD -> .o
    |
    +-- ds4_sycl.cpp ---------------> icpx -fsycl -iquote . -> ds4_sycl.o
    |     (pulls in every sycl/*.hpp)
    |
    +-- ds4_sycl_unavailable.cpp ---> icpx -fsycl -> .o
    |
    +-- link -fsycl -> ds4, ds4-server, ds4-bench, ds4-eval, ds4-agent
```

Two build details that matter:

**`-iquote .` and never `-I.`** for the `ds4_sycl.o` rule. With `-I.`, the
repository's own `sycl/` directory shadows the oneAPI `<sycl/sycl.hpp>` header
and the build fails in a confusing way. `-iquote` applies only to `"..."`
includes, which is exactly what is wanted. Proven empirically, not assumed.

**`-ffast-math -fno-finite-math-only`.** Fast math is wanted; finite-math is
not, because several kernels rely on Inf and NaN behaving properly. Dropping the
second flag silently elides a guard in the compressor.

The environment needs four oneAPI scripts sourced in the same shell as the
build. Do not use `setvars.sh`:

```sh
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
source /opt/intel/oneapi/umf/latest/env/vars.sh
source /opt/intel/oneapi/tbb/latest/env/vars.sh
source /opt/intel/oneapi/mkl/2025.3/env/vars.sh
```

The fourth is oneMKL, pinned to 2025.3 to match the compiler: `/opt/intel/oneapi/`
also holds 2024.2 and 2026.1, and the build is not tested against either.

**oneMKL is wired in with `-qmkl=sequential`**, on both `SYCL_CFLAGS` and
`SYCL_LDLIBS`. That single flag resolves oneMKL's include paths for compiling
and its link libraries (`mkl_sycl`, `mkl_intel_ilp64`, `mkl_sequential`,
`mkl_core`) for linking; it worked cleanly on the first try, so there was no
need to fall back to explicit `-I$(MKLROOT)/include` plus explicit `-l` flags.
`sequential` rather than the threaded variant because this backend manages its
own parallelism through SYCL, not through MKL's own threading layer.

oneMKL's `oneapi::mkl::blas::gemm_batch` (USM, strided form) is the batched
GEMM prefill attention and one dense matmul entry need; it is column-major,
like cuBLAS, and returns a `sycl::event` rather than ordering work on a
stream. See `sycl/ds4_sycl_common.hpp`'s `sycl_gemm_batch_f32` for the shared
wrapper and `tests/test_sycl_gemm_batch_smoke.c` for a minimal correctness
check against a hand-computed result.

## Device and tier model

```
   ds4_gpu_init()
        |
        v
   g_devices : vector<{ sycl::device, sycl::queue }>
        ^
        |  indexed by "logical tier"
        |
   g_current_tier  ---- ds4_sycl_current_queue()
```

A **tier** is ds4's term for one logical GPU. The engine places layers across
tiers and switches with `ds4_gpu_set_current_device`. Today the SYCL backend
enumerates every device but drives one: multi-tier allocation and cross-device
copies are not implemented yet, and the tier-aware allocators reject any tier
other than the current one rather than silently landing on the wrong device.

That strictness is deliberate. Level Zero has no unified virtual addressing to
paper over a wrong-device pointer, so a clean failure is far better than a
plausible-looking wrong answer.

## Multi-GPU: what exists and what it buys

There are **two** parallelism mechanisms in `ds4.c` and they buy different
things. Confusing them wastes weeks.

```
   PIPELINE PLACEMENT                    EXPERT PARALLELISM
   (multi-tier, capacity)                (cuda_tp_ep, latency)

   layer 0..10   -> GPU 0                token N's MoE work
   layer 11..21  -> GPU 1                  |
   layer 22..32  -> GPU 2                  +--> experts  0..127 -> GPU 0
   layer 33..42  -> GPU 3                  +--> experts 128..255 -> GPU 1
        |                                        |
   one GPU busy at a time                  both busy on the SAME token
   NO single-stream speedup                ~2x, ownership is two-way
   lets an 80 GB model exist               combines to one answer
   on 24 GB cards
```

**Pipeline placement** is gated on `e->multi_tier`, built from a
`ds4_gpu_config` the CLI derives from `--gpu-vram` / `--gpu-devices`. Its
linchpin is `ds4_gpu_tier_free_vram`: `engine_classify_multi_tier` refuses a
config whose budgets are zero, so a wrong answer there makes every later entry
unreachable and the failure looks like something else entirely. It reports a
static ceiling minus self-tracked commitments rather than a live Level Zero
Sysman query, because on this driver Sysman's free-memory reading does not move
under allocation pressure (spec 6p). A number that never decreases is worse than
a conservative constant, because the placement planner believes it.

**Expert parallelism** is gated on `g->cuda_tp_ep`, which needs a placement AND
`--cuda-tensor-parallel`. Note this is NOT the Metal `tp_world == 2` mechanism:
`ds4_engine_tp_bind` is compiled out on any non-Apple build, so every
`tp_world`-gated entry here is a permanent no-op. The reachable path is the CUDA
one, because `e->backend == DS4_BACKEND_CUDA` is true for a SYCL build.

Ownership composes with expert compaction: a rank stages only the experts it
owns AND only those the token selected. Staging too much is a silent 42x
regression that no correctness test catches.

`--cuda-tensor-parallel` is refused at startup unless every routed layer is
`(Q4_K, Q4_K)` or `(IQ2_XXS, Q2_K)`, which are the owned-decode kernels that
exist. Flash is the second pair.

### There is a THIRD mechanism: pipelined prefill, and it is not the distributed mode

`ds4_distributed.c` is TCP sockets with a coordinator and workers. It is for
multiple MACHINES, and the README's 1.38x to 1.85x figures come from two
MacBooks over Thunderbolt. **It is not what you use for multiple GPUs in one
box.**

The local equivalent is built into multi-tier placement.
`metal_graph_build_prefill_stages` (`ds4.c:34132`) walks `g->placement[]` and
groups contiguous same-tier layers into stages.
`metal_graph_prefill_pipeline_stage_major` (around `ds4.c:34205`) then refuses
unless **all** of these hold:

```c
metal_graph_build_prefill_stages(...) && n_stages >= 2   // needs 2+ GPUs
mb_cap != 0 && mb_cap < n_tokens                          // microbatching on
n_mb >= 2                                                 // 2+ microbatches
```

Given those, prefill is microbatched and pipelined across tiers: tier T works on
microbatch N+1 while tier T+1 works on N. Ideal speedup is roughly
`S / (1 + (S-1)/M)` for S stages and M microbatches.

**Every ABI entry that path needs is implemented** (`tier_free_vram`,
`register_model_map_no_copy`, `device_cache_tensors`, the `q8_cache_suppressed`
pair, `tensor_copy_xdev_ordered`). It will not silently decline for a missing
symbol. **It has never executed**, because a single GPU always yields
`n_stages == 1`.

### But it will not help until dispatch is asynchronous

`sycl/*.hpp` holds **213 blocking `wait_and_throw` calls**, roughly one per
kernel launch. For contrast, `ds4_cuda.cu` runs **351 launches behind 25 explicit
syncs**.

The pipeline's wave loop (`ds4.c:34308-34376`) only wins if dispatching to tier
T+1 does not wait for tier T. That holds on CUDA. It does not hold here. As
written, engaging the pipeline on SYCL pays real cross-device copy cost and
recovers none of the overlap: roughly 1.0x, plausibly slightly worse.

**That same synchronous pattern blocks three things**, which is why it matters
more than any single one of them:

1. pipelined prefill, worth roughly 2.5x to 4x for 4 to 7 stages
2. decode graph capture, since `wait()` throws during graph recording
3. about 38 percent of decode wall time, the 3.26 ms non-kernel remainder

### Measuring whether the two tiers actually overlap

Expert parallelism only pays if the home and partner tiers compute at the
same time, and on a real 8-GPU Flash run it does not: prefill goes 3.55 ->
7.10 t/s, but generation goes 4.27 -> 2.73 t/s and has never moved from 2.73
to 2.82 across five code states.

`DS4_SYCL_TIMELINE` (`sycl/ds4_sycl_timeline.hpp`) measures the prior
question rather than guessing at mechanisms. `=1` records a bounded ring of
host timestamps at the tier switch, the owned decode entry and exit, the
cross-device copy, the combine, and the one place the host really blocks
(`ds4_sycl_wait_current_tier`). `=2` adds one `single_task` before and after
each owned decode that publishes its tier's busy flag in a shared host USM
word and snapshots the other tiers'; a marker that saw another tier's flag
set caught that tier mid-decode. Both dump at `ds4_gpu_cleanup`.

**Do not reach for `DS4_SYCL_PROFILE` for this.** It sets
`property::queue::enable_profiling`, and on a full-size run that exhausted
the Level Zero event pool (`UR_RESULT_ERROR_OUT_OF_RESOURCES`, then
`UR_RESULT_ERROR_DEVICE_LOST`, then nearly three hours at 100% CPU without
exiting) at roughly 1,900 kernel launches per decode token. It also cannot
answer this question even when it survives: `ds4_sycl_profile_record` calls
`get_profiling_info`, which blocks on an incomplete event and so silently
restores every host wait this backend spent months removing. The timeline
allocates no events at all.

### The handoff, and the thing to check first on real hardware

`ds4_gpu_tensor_copy_xdev` (`sycl/ds4_sycl_mgpu.hpp:443-497`) does a real
single-hop peer-to-peer `queue.memcpy` when `g_gpu_peer_ok[src][dst]` was
validated at init by a byte-exact round-trip probe. **Otherwise it falls back to
a host bounce**, device to host then host to device, two hops.

**That decision has never run off-diagonal.** Whether a B60's dual-die
behind-a-PCIe-switch topology gets true peer access is unknown, and it is 32 MiB
per 512-token microbatch boundary at Flash's shape. If it bounces, the pipeline
economics change materially. **Check this before benchmarking anything
multi-GPU**, or you will be measuring a hidden round trip.

## Performance: where the time goes

Two structural costs dominated, and one is fixed.

**Weight staging, fixed twice over.** Routed MoE used to stage all 256 experts
per layer to read 6, which was 72.6 GiB of host-to-device traffic per token
across Flash's 43 routed layers. Compaction cut that 42x. Device residency then
removed staging entirely for cached ranges.

**Queue drains, under investigation.** `sycl/*.hpp` holds around 208
`wait_and_throw` calls: most ABI entries drain the queue after launching. The
ABI already has command batching (`ds4_gpu_begin_commands` /
`end_commands` / `synchronize`), but `begin_commands` is `{ return 1; }` and
every entry orders itself instead, so batching exists and is unused.

The drains were necessary while every call owned staging scratch that had to be
freed on return. Device residency removes the scratch on cached paths, which is
what makes deferring them possible. Whether that is actually where the time goes
is being measured rather than assumed: a real GGUF layer takes 18.8 ms against
roughly 0.16 ms of memory traffic, so something costs 100x, and it is worth
knowing what before optimising it.

**Do not quote a tokens-per-second figure derived from bandwidth arithmetic.**
The measurement above contradicts it, and tuning has never been done.

## Known gaps

Ordered by how likely each is to bite on first contact with real hardware.

* **The compressor and indexer path has never run through the engine.** Flash's
  compress ratios are 0 for layers 0 and 1 and then 4 or 128 alternating, so
  **41 of 43 layers use it** and 2 do not. Every `DS4_TEST_HOOKS` entry does
  `memset(g_ds4_compress_ratios, 0, ...)`, so every engine-level test runs the
  configuration only those 2 layers have. This is the largest untested surface
  and it subsumes the mixed prefill path, where long-context prefill and its own
  oneMKL GEMM live.
* **Nothing has run against real weights.** The engine's real decode path does
  run end to end on synthetic weights: a complete Flash layer, then a complete
  token through the output head to logits, over a 4-layer shape covering both
  hash-routed and gate-routed layers, bit-identical across 65 repeats. A real
  GGUF opens through `ds4_engine_open`. What has never happened is a token from
  the real 80 GB model.
* **Multi-GPU is written but has never run on more than one card.** Tier
  switching, per-tier allocation, cross-device copies, placement classification,
  the per-device weight cache and expert-parallel decode are all implemented and
  none has seen a second device. The peer-access byte-validation protocol in
  particular has never run off-diagonal; it exists because CUDA's peer-access
  reporting lied on real hardware, and the target cards are two dies behind a
  PCIe switch.
* **Physical device ids and logical tiers are two different index spaces**, and
  the backend translates between them at the one ABI boundary that speaks
  physical ids. `--gpu-devices` values live in `g_gpu[tier].device_id`;
  everything inside this backend (`g_devices`, the queues, every per-tier cache)
  is addressed by tier. `ds4_gpu_device_cache_tensors` and its ranges'
  `target_device` carry physical ids, matching CUDA, and now go through
  `sycl_tier_for_device_id` before anything tier-indexed is touched. A
  non-contiguous list is therefore correct, including
  `--gpu-devices 0,2,4,6,1,3,5,7`, the ordering `--cuda-tensor-parallel` wants
  since it pairs tier `i` with tier `i + n_gpus/2`. It previously staged each
  tier's weights through another tier's queue and into another tier's cache
  slab. What has run on real hardware here is the one-tier, non-zero-physical-id
  case (`tests/test_sycl_placement.c`); the genuinely multi-card ordering is
  asserted by a test that skips on a single-GPU box.
* **`--gpu-vram auto` cannot be trusted on Intel.** It reads Level Zero Sysman,
  which on this driver reports total memory as free. Pass explicit per-device
  budgets. `ds4_gpu_tier_free_vram` deliberately does not use that query.
* **Performance is untuned and we are ~100x off bandwidth-bound.** oneMKL's
  `compute_mode` and the GEMM-versus-kernel dispatch thresholds are at their
  ROCm-literal defaults, no A770 tuning result should be trusted for Battlemage,
  and the per-call queue drain described above is unresolved.
* **The streaming expert cache is implemented but not wired in.** Its three
  lookup hooks are hardcoded false. This matters much less than it once did:
  compaction and device residency between them removed most of what it would
  have saved.
* **Decode graph capture is unavailable.** The extension works on this stack,
  but `wait()` throws during graph recording and the per-entry waits are not all
  gone yet. See the record in `ds4_sycl_unavailable.cpp`.
* **Metal tensor parallelism (`tp_world == 2`) can never run here.** Its bind
  function is compiled out on non-Apple builds. Entries gated on it are
  permanent no-ops; the reachable mechanism is `cuda_tp_ep`.
* **MXFP4 owned decode is not implemented.** Flash does not use MXFP4, and
  CUDA's path there uses a different kernel family entirely.
