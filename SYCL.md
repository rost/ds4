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
  | device + queue discovery |       | 90 stubs for entries this    |
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

Roughly 190 ABI entries are implemented; 90 remain stubbed (87 via the macros
plus 3 hand-written). A stub reachability audit
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
    |  3. STAGE WEIGHTS TO DEVICE     <-- mandatory, see "the mmap trap"
    |  4. quantise activations to Q8_K
    |  5. dispatch on the format path
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

Every subsystem stages. `sycl_stage_host_bytes` in the common header does it
and returns an RAII guard. Where only a scalar is needed, it is read on the host
before launch instead.

This cost real time once: a MoE decode path returned all-zero output with no
exception raised, and the cause was one unstaged pointer.

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

## Known gaps

* **Multi-GPU is written but unverified.** Tier switching, per-tier allocation,
  cross-device copies and per-tier streaming caches are all implemented, but
  this development machine has one GPU. In particular the peer-access
  byte-validation protocol has never run off-diagonal. That protocol exists
  because CUDA's peer-access reporting lied on real hardware, and the target
  cards are two dies behind a PCIe switch, so it is the most likely thing to
  bite on first contact with the real machine.
* **The streaming expert cache is implemented but not wired in.** Routed MoE's
  three lookup hooks are hardcoded false, so the cache, its eviction and its
  per-tier state are correct, tested, and currently unused. This matters much
  less than it once did: routed MoE now stages only the experts a call actually
  selects rather than the whole per-layer table, which cut host-to-device
  traffic by about 42x (72.6 GiB to roughly 1.7 GiB per token at Flash's shape).
* **Tensor parallelism is not implemented.** `--cuda-tensor-parallel` is refused
  at startup on this backend. A single token stream therefore uses one GPU at a
  time; multi-GPU placement gives capacity, not single-stream speed.
* **No end-to-end run against real weights** has happened. That said, the
  engine's own decode path now runs end to end on synthetic weights: a complete
  Flash decoder layer, then a complete token through the output head to logits,
  over a 4-layer shape covering both hash-routed and gate-routed layers, with
  bit-identical output across 65 repeats. A real GGUF also opens through
  `ds4_engine_open`. What has never happened is a token from the real 80 GB
  model.
* **Performance is untuned.** Weight staging is per call with no cross-call
  cache, oneMKL's `compute_mode` and the GEMM-versus-kernel dispatch
  thresholds (the mixed-prefill tiled path's 4 GiB cap among them) are all at
  their ROCm-literal defaults, and no tuning result from the development A770
  should be trusted for the Battlemage target.
