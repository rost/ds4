#pragma once

/* Optional SYCL command-graph capture for the per-token command batch.
 *
 * Why this exists: a decode token on this backend dispatches on the order
 * of three thousand kernels, most of them small enough that the cost of
 * submitting them dominates the cost of running them (measured on a real
 * DeepSeek V4 Flash run: matmul_f16 at 0.025 ms/call and
 * matmul_f16_convert at 0.022 ms/call, 13589 calls each). The CUDA
 * reference dispatches roughly a tenth as many per token.
 * sycl_ext_oneapi_graph exists for exactly this shape: record a run of
 * commands once and hand the whole run to the driver as a single
 * submission, so the per-submission cost is paid once for the run instead
 * of once per kernel.
 *
 * Measured on an Intel Arc A770 (Level Zero, oneAPI 2025.3) with 3000
 * launch-bound kernels, comparing direct submission against a graph:
 * 59.2 ms eager, 19.3 ms recording and finalizing a fresh graph every
 * iteration, 7.9 ms replaying an already-finalized graph. Re-recording
 * every time is the mode this file implements, because a decode token's
 * kernel arguments (KV position above all) change from token to token and
 * this device reports ext_oneapi_limited_graph rather than
 * ext_oneapi_graph, meaning finalized graphs cannot be updated in place.
 *
 * The capture boundary is the command-batch lifecycle this backend
 * already exposes (sycl/ds4_sycl_commands.hpp): ds4.c brackets the whole
 * per-token encode in ds4_gpu_begin_commands / ds4_gpu_end_commands
 * (metal_graph_eval_token_raw_swa), so that pair is already the exact
 * "here is a run of commands, here is where it must have completed"
 * statement a graph needs. No new boundary is invented here.
 *
 * FLUSH ON DEMAND is the design's load-bearing idea. A queue in the
 * recording state throws on queue::wait_and_throw and on
 * event::wait_and_throw (verified directly against this oneAPI install),
 * and this backend waits in roughly a hundred and seventy places, several
 * of which are genuinely load-bearing rather than removable -- the
 * host-to-device staging copy reuses one host bounce buffer across chunks
 * and must wait before overwriting it. Rather than requiring every one of
 * those waits to be deleted before any capture is possible, a wait routed
 * through sycl_batch_wait below FLUSHES the batch: the graph recorded so
 * far is finalized, submitted, and waited on, and recording resumes into a
 * fresh graph. Because every tier's queue is in_order, waiting for the
 * whole batch is strictly stronger than waiting for any single event
 * inside it, so substituting a flush for a wait can only over-synchronize,
 * never under-synchronize. That makes capture correct by construction at
 * every wait site instead of correct only once all of them are gone, and
 * it degrades gracefully: a region that waits often simply produces many
 * small graphs and performs like the un-captured path.
 *
 * The direct consequence, worth stating plainly because it bounds what
 * this feature can currently deliver: while the model-range cache misses,
 * every weight-staging call waits per chunk, so batches stay small and
 * capture wins little. Capture pays off in proportion to how far apart the
 * remaining waits are.
 *
 * DEFERRED FREES are the other half of correctness. sycl_device_scratch_
 * guard frees device scratch in its destructor without waiting, which is
 * safe today only because the kernels using that scratch have already
 * completed by the time the guard goes out of scope. Under capture they
 * have not: a recorded command does not run until the graph is submitted.
 * So while a batch is recording, a guard hands its pointer to the batch
 * instead of freeing it, and the batch frees it after the submission it
 * belongs to has completed. Without this, capture is a use-after-free.
 *
 * Off by default. DS4_SYCL_GRAPH=1 in the environment turns it on, and
 * ds4_sycl_test_graph_enable exists so a test can drive both paths in one
 * process. */

#include "ds4_sycl_common.hpp"

#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <optional>
#include <vector>

namespace {

namespace sycl_graph_ext = sycl::ext::oneapi::experimental;

/* Device scratch whose free was deferred past the submission that uses
 * it. The queue is carried alongside the pointer because a batch may span
 * allocations made against more than one tier's queue. */
struct sycl_deferred_free {
    sycl::queue *q;
    void        *p;
};

/* The run of commands currently being recorded, if any.
 *
 * Exactly one of these exists (g_sycl_graph_batch below). It is not a
 * per-tier object: the batch records the queue it began on and every
 * flush drains that one queue, matching the single-tier scope of the
 * ds4_gpu_begin_commands / ds4_gpu_end_commands pair it is driven by. */
class sycl_graph_batch {
public:
    /* Whether capture is armed at all. Read from the environment once and
     * cached, so a per-kernel call site can consult it without paying for
     * a getenv, and so a test that flips the runtime switch below is not
     * fighting the environment. */
    bool armed() const { return armed_; }

    void set_armed(bool on) { armed_ = on; }

    bool recording() const { return graph_.has_value(); }

    /* Diagnostics for tests and for reporting how well capture is doing:
     * how many separate graph submissions the batch needed (one plus the
     * number of flushes forced by a wait), how many nodes went into them
     * in total, and how many the last one held. Submissions above one
     * means something inside the captured region waited, and
     * nodes / submissions is the average run length capture achieved --
     * the number that decides whether capture can pay for itself at all,
     * since a graph of one or two nodes costs more to finalize than the
     * submissions it saves. */
    uint64_t submissions() const { return submissions_; }
    uint64_t total_nodes() const { return total_nodes_; }
    uint64_t last_node_count() const { return last_node_count_; }

    /* Begins a new batch on `q`, zeroing the diagnostics so they describe
     * this batch. Returns false only when recording could not be started,
     * in which case the caller proceeds un-captured, which is always
     * valid. */
    bool begin(sycl::queue &q) {
        if (!open(q)) return false;
        submissions_ = 0;
        total_nodes_ = 0;
        last_node_count_ = 0;
        return true;
    }

    /* Reopens recording after the batch was drained mid-token by
     * ds4_gpu_flush_commands. Distinct from begin() only in keeping the
     * diagnostics running, because the split flush is a division inside
     * one token's batch and not the start of a new one -- ds4.c splits a
     * decode token into two or three command buffers
     * (metal_graph_encode_decode_token_raw_swa) and only the first would
     * ever be captured if a split ended capture for the rest of the
     * token. */
    bool resume(sycl::queue &q) { return open(q); }

    /* Ends the batch: submits whatever is still recorded and waits for it.
     * Safe to call with no batch in progress, which is what makes it usable
     * from the error paths that call ds4_gpu_synchronize without knowing
     * whether a batch was ever opened. */
    bool end() {
        if (!recording()) return true;
        const bool ok = submit_recorded();
        graph_.reset();
        q_ = nullptr;
        return ok;
    }

    /* Submits the recorded run, waits for it, and resumes recording into a
     * fresh graph. This is what a wait inside a captured region becomes. */
    bool flush() {
        if (!recording()) return true;
        sycl::queue *q = q_;
        if (!submit_recorded()) {
            graph_.reset();
            q_ = nullptr;
            return false;
        }
        graph_.reset();
        try {
            graph_.emplace(q->get_context(), q->get_device());
            graph_->begin_recording(*q);
            return true;
        } catch (const sycl::exception &e) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "command-graph recording failed to resume: %s\n", e.what());
            graph_.reset();
            q_ = nullptr;
            return false;
        }
    }

    /* Takes ownership of scratch whose free must wait for the recorded
     * commands that reference it. Called only while recording. */
    void defer_free(sycl::queue &q, void *p) {
        if (!p) return;
        deferred_.push_back(sycl_deferred_free{&q, p});
    }

    /* Drops any recording without submitting it, and frees the scratch it
     * was holding. For teardown only: the commands recorded so far are
     * discarded, so the work they represented never happens. */
    void abandon() {
        if (recording()) {
            try {
                graph_->end_recording(*q_);
            } catch (const sycl::exception &) {
                /* Nothing useful to do while tearing down. */
            }
        }
        graph_.reset();
        q_ = nullptr;
        drain_deferred();
        submissions_ = 0;
        total_nodes_ = 0;
        last_node_count_ = 0;
    }

private:
    /* Starts recording `q`, whether for a fresh batch or a resumed one. A
     * recording already in progress is flushed first rather than nested:
     * the graph extension has no notion of a nested recording, and
     * flushing preserves the ordering the caller asked for. */
    bool open(sycl::queue &q) {
        if (!armed_) return false;
        if (recording()) {
            if (!flush()) return false;
        }
        if (!q.get_device().has(sycl::aspect::ext_oneapi_limited_graph) &&
            !q.get_device().has(sycl::aspect::ext_oneapi_graph)) {
            /* Disarm rather than retry per token: device capability does
             * not change within a process, so one report is enough. */
            armed_ = false;
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "command-graph capture unavailable on this device, disabled\n");
            return false;
        }
        if (!warm_blas(q)) {
            armed_ = false;
            return false;
        }
        try {
            graph_.emplace(q.get_context(), q.get_device());
            graph_->begin_recording(q);
            q_ = &q;
            return true;
        } catch (const sycl::exception &e) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "command-graph recording failed to start: %s\n", e.what());
            graph_.reset();
            q_ = nullptr;
            armed_ = false;
            return false;
        }
    }

    /* oneMKL's first BLAS call in a process performs a lazy initialization
     * that waits on its own event internally. A wait is illegal on a queue
     * in the recording state, so if that first call happens inside a
     * recording it throws ("wait method cannot be used for an event
     * associated with a command graph") and takes the enclosing kernel
     * entry down with it. Measured on this install: the initialization is
     * one-time and process-global, not per-shape -- once any BLAS call has
     * completed outside a recording, every later call of any shape records
     * cleanly. So a single tiny GEMM here, once, before the first
     * recording ever starts, is enough to make the whole BLAS surface
     * graph-safe. Costs nothing when capture is off, because nothing
     * calls it. */
    bool warm_blas(sycl::queue &q) {
        if (blas_warm_) return true;
        float *scratch = nullptr;
        try {
            scratch = sycl::malloc_device<float>(3, q);
            if (!scratch) return false;
            q.memset(scratch, 0, 3 * sizeof(float)).wait_and_throw();
            sycl_gemm_batch_f32(q, oneapi::mkl::transpose::nontrans,
                                oneapi::mkl::transpose::nontrans,
                                1, 1, 1, 1.0f,
                                scratch, 1, 0,
                                scratch + 1, 1, 0,
                                0.0f,
                                scratch + 2, 1, 1,
                                1);
            q.wait_and_throw();
            sycl::free(scratch, q);
            blas_warm_ = true;
            return true;
        } catch (const std::exception &e) {
            /* std::exception, not sycl::exception: oneMKL's argument
             * errors are on a different branch of the hierarchy, the same
             * reason ds4_gpu_matmul_f16_tensor catches broadly. */
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "command-graph BLAS warm-up failed, capture disabled: %s\n",
                    e.what());
            if (scratch) {
                try {
                    sycl::free(scratch, q);
                } catch (const sycl::exception &) {
                    /* Already reporting a failure; nothing to add. */
                }
            }
            return false;
        }
    }

    /* Ends recording, finalizes, submits, waits, then releases the scratch
     * the submitted commands were using. The wait is what makes the
     * deferred frees safe, and is also exactly the semantics of the wait
     * this submission is standing in for. */
    bool submit_recorded() {
        try {
            last_node_count_ = graph_->get_nodes().size();
            graph_->end_recording(*q_);
            total_nodes_ += last_node_count_;
            if (last_node_count_ != 0) {
                sycl_graph_ext::command_graph<sycl_graph_ext::graph_state::executable>
                        exec = graph_->finalize();
                q_->ext_oneapi_graph(exec);
                submissions_++;
            }
            q_->wait_and_throw();
            drain_deferred();
            return true;
        } catch (const sycl::exception &e) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "command-graph submission failed: %s\n", e.what());
            drain_deferred();
            return false;
        }
    }

    void drain_deferred() {
        for (const sycl_deferred_free &d : deferred_) {
            try {
                sycl::free(d.p, *d.q);
            } catch (const sycl::exception &e) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX
                        "deferred device scratch free failed: %s\n", e.what());
            }
        }
        deferred_.clear();
    }

    std::optional<sycl_graph_ext::command_graph<
            sycl_graph_ext::graph_state::modifiable>> graph_;
    sycl::queue                     *q_ = nullptr;
    std::vector<sycl_deferred_free>  deferred_;
    bool                             armed_ = getenv("DS4_SYCL_GRAPH") != nullptr;
    bool                             blas_warm_ = false;
    uint64_t                         submissions_ = 0;
    uint64_t                         total_nodes_ = 0;
    uint64_t                         last_node_count_ = 0;
};

sycl_graph_batch g_sycl_graph_batch;

}  /* namespace */

/* The two hooks ds4_sycl_common.hpp forward-declares so that
 * sycl_device_scratch_guard and sycl_batch_wait, both defined there
 * alongside the rest of the shared kernel-entry infrastructure, can reach
 * the batch without that header having to know how a batch is built. */
static bool sycl_graph_batch_recording(void) {
    return g_sycl_graph_batch.recording();
}

static void sycl_graph_batch_defer_free(sycl::queue &q, void *p) {
    g_sycl_graph_batch.defer_free(q, p);
}

static bool sycl_graph_batch_flush(void) {
    return g_sycl_graph_batch.flush();
}

/* Drops an unsubmitted recording before g_devices.clear() destroys the
 * queue it was recorded against. Forward-declared in ds4_sycl.cpp so
 * ds4_gpu_cleanup can call it, matching sycl_readback_teardown. */
static void sycl_graph_teardown(void) {
    g_sycl_graph_batch.abandon();
}

/* Test-only switch, mirroring ds4_sycl_test_profile_enable: lets one test
 * process measure both the captured and un-captured paths without
 * re-execing itself with a different environment. */
extern "C" void ds4_sycl_test_graph_enable(int enable) {
    g_sycl_graph_batch.set_armed(enable != 0);
}

extern "C" int ds4_sycl_test_graph_available(void) {
    if (g_devices.empty()) return 0;
    const sycl::device &d = g_devices[0].dev;
    return d.has(sycl::aspect::ext_oneapi_graph) ||
           d.has(sycl::aspect::ext_oneapi_limited_graph) ? 1 : 0;
}

/* How many graph submissions the batch that just ended required. One means
 * the whole captured region went to the device as a single submission;
 * more means a wait inside the region forced that many flushes. Zero means
 * nothing was captured. */
extern "C" uint64_t ds4_sycl_test_graph_submissions(void) {
    return g_sycl_graph_batch.submissions();
}

/* Total nodes recorded across every submission of the batch that just
 * ended. Divided by the submission count this gives the average number of
 * commands capture managed to group into one submission. */
extern "C" uint64_t ds4_sycl_test_graph_total_nodes(void) {
    return g_sycl_graph_batch.total_nodes();
}

extern "C" uint64_t ds4_sycl_test_graph_last_node_count(void) {
    return g_sycl_graph_batch.last_node_count();
}
