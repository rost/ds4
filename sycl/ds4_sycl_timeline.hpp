#pragma once

/* Per-tier timeline recorder: did two tiers ever run at the same time?
 *
 * Expert parallelism (--cuda-tensor-parallel) splits the 256 routed
 * experts across a home tier and a partner tier, so each routed layer's
 * expert work is issued twice, once per tier. That only pays if the two
 * tiers compute concurrently. Measured on a real 8-GPU Flash decode it
 * does not pay: prefill 3.55 -> 7.10 t/s, but generation 4.27 -> 2.73
 * t/s, and generation has sat at 2.73 to 2.82 t/s across five code states
 * and at both 64 and 200 generated tokens. Five candidate mechanisms have
 * been killed by controlled experiment (blocking cross-device copies,
 * cross-device copy cost at all under DS4_FORCE_HOST_BOUNCE=1, peer read
 * versus copy, the eleven post-launch drains in ds4_sycl_moe_owned.hpp,
 * and the per-layer readback of the six selected expert ids), so the
 * remaining question is not "which mechanism" but the prior one: do the
 * two tiers overlap at all.
 *
 * This answers that, and it is deliberately NOT built on
 * property::queue::enable_profiling. That property is what
 * DS4_SYCL_PROFILE turns on, and on a full-size run it died:
 * "embed_token_hc failed: UR_RESULT_ERROR_OUT_OF_RESOURCES" then
 * "synchronize failed: UR_RESULT_ERROR_DEVICE_LOST", after which the
 * process spun at 100% CPU for nearly three hours without exiting. A
 * decode token issues roughly 1,900 kernel launches and a 16-token run
 * about 30,000, and the Level Zero event pool ran dry. Worse for this
 * measurement specifically, ds4_sycl_profile_record calls
 * get_profiling_info, which BLOCKS on an incomplete event: enabling
 * profiling silently reintroduces every host wait recently removed from
 * this backend, and so cannot measure whether removing them produced any
 * overlap.
 *
 * Two levels, both bounded by construction:
 *
 *   DS4_SYCL_TIMELINE=1  host submission timeline. A fixed-size ring of
 *                        records, one steady_clock reading at each of a
 *                        small number of chosen points. No events, no
 *                        allocations after init, no waits. This answers
 *                        "does the host issue home and partner back to
 *                        back, or does it serialise on something".
 *
 *   DS4_SYCL_TIMELINE=2  adds device-side busy markers: one single_task
 *                        submitted on the tier's own queue before and
 *                        after the owned decode. Every tier's queue is
 *                        in_order, so a tier's begin marker runs before
 *                        that tier's MoE kernels and its end marker after
 *                        them, which makes the marker a real device-side
 *                        observation rather than a submission-time one.
 *                        Each marker publishes its own tier's busy flag
 *                        in a shared host USM word and then snapshots
 *                        every other tier's flag. A begin marker that
 *                        sees another tier's flag set observed that tier
 *                        mid-decode: the two really did overlap.
 *
 * The marker deliberately does NOT use a shared atomic counter, which was
 * the first design. No Intel GPU tested here reports
 * usm_atomic_host_allocations (the A770 reports 0 for both the host and
 * the shared variant), so cross-device atomics on one shared word are not
 * available and a ticket order built on them would be fiction. One writer
 * per word plus a snapshot read needs no atomicity at all: the only thing
 * it needs is that a write one device's kernel completed is visible to a
 * later kernel on another device, and that is validated at init by
 * sycl_timeline_probe_visibility below rather than assumed, for the same
 * reason the peer-access path validates itself byte-exactly instead of
 * trusting the driver's report.
 *
 * Level 2 costs two extra one-item kernels per owned decode per tier,
 * about 172 launches per token against the roughly 1,900 a token already
 * issues, and allocates exactly one host USM buffer for the whole run.
 * There is no per-kernel event anywhere, which is the specific thing that
 * exhausted the event pool.
 *
 * When the env var is unset every record point is one bool test on a
 * static that was read from the environment once at init.
 */

#include "ds4_gpu_mgpu.h"
#include "ds4_sycl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

enum sycl_timeline_phase {
    kSyclTimelineTierSwitch = 0,
    kSyclTimelineTierSwitchFenced,
    kSyclTimelineOwnedEnter,
    kSyclTimelineOwnedExit,
    kSyclTimelineXdevEnter,
    kSyclTimelineXdevExit,
    kSyclTimelineCombineEnter,
    kSyclTimelineCombineExit,
    kSyclTimelineWaitEnter,
    kSyclTimelineWaitExit,
    kSyclTimelinePhaseCount
};

static const char *const kSyclTimelinePhaseNames[kSyclTimelinePhaseCount] = {
    "tier_switch",   "tier_switch_fenced", "owned_enter",   "owned_exit",
    "xdev_enter",    "xdev_exit",          "combine_enter", "combine_exit",
    "wait_enter",    "wait_exit",
};

struct sycl_timeline_entry {
    uint64_t ns;     /* steady_clock, relative to ds4_gpu_init_multi */
    uint64_t seq;    /* monotonic, so a ring wrap is visible in the dump */
    int32_t  tier;
    int32_t  peer;   /* the other tier of a copy or a switch, else -1 */
    uint32_t phase;
    uint32_t pad;
};

/* 64 Ki records is 2 MiB and covers a 16-token expert-parallel decode
 * with room to spare: 43 routed layers x 2 tiers x roughly 6 records is
 * about 520 records per token. Chosen to be comfortably above what the
 * runs this exists for produce, rather than exactly at it, so a longer
 * run degrades to "the oldest records were overwritten" (reported at the
 * dump) instead of to a silent truncation. */
static constexpr uint32_t kSyclTimelineCapacity = 1u << 16;

/* Device marker buffer layout, in uint32_t words:
 *
 *   [0                     .. DS4_MAX_GPUS)   per-tier busy flag
 *   [DS4_MAX_GPUS          .. 2*DS4_MAX_GPUS) init-time visibility probe
 *   [2*DS4_MAX_GPUS        .. )               per-tier marker ring
 *
 * Every word has exactly one writer, which is what lets this work with no
 * atomics: a tier writes only its own flag and only its own ring slot,
 * and reads the other tiers' flags. 8192 markers per tier is 2 per owned
 * decode x 43 routed layers x 95 tokens, comfortably past the 16-token
 * runs this exists for; the whole allocation is 2 MiB and is made once. */
static constexpr uint32_t kSyclTimelineMarkersPerTier = 1u << 13;
static constexpr uint32_t kSyclTimelineMarkerWords = 4u;
static constexpr uint32_t kSyclTimelineFlagBase = 0u;
static constexpr uint32_t kSyclTimelineProbeBase = (uint32_t)DS4_MAX_GPUS;
static constexpr uint32_t kSyclTimelineMarkerBase = 2u * (uint32_t)DS4_MAX_GPUS;
static constexpr uint32_t kSyclTimelineMarkerWritten = 0x80000000u;

/* Level: 0 off, 1 host submission timeline, 2 adds device markers. Read
 * once from DS4_SYCL_TIMELINE at init so no record point calls getenv. */
static int      g_sycl_timeline_level = 0;
static bool     g_sycl_timeline_on    = false;
static uint32_t g_sycl_timeline_print = 200;

static sycl_timeline_entry *g_sycl_timeline_ring = nullptr;
static std::atomic<uint64_t> g_sycl_timeline_seq{0};
static uint64_t              g_sycl_timeline_base_ns = 0;

/* Device marker state. The busy flags and every tier's marker ring live
 * in ONE host USM allocation shared by all of them, which is why level 2
 * requires that every device sit in the same sycl::context
 * (ds4_sycl_build_devices builds one context per SYCL platform, normally
 * one for the whole box) and report usm_host_allocations. Both are
 * checked at init, and so is cross-device visibility of that allocation;
 * failing any of the three drops to the host-only timeline with a logged
 * reason rather than producing markers that cannot be trusted.
 *
 * g_sycl_timeline_marks is the per-tier issue count, kept on the host so
 * the kernel needs no device-side counter. */
static uint32_t              *g_sycl_timeline_dev = nullptr;
static std::vector<sycl::context> g_sycl_timeline_dev_ctx;
static uint32_t                   g_sycl_timeline_marks[DS4_MAX_GPUS] = {0};
static int                        g_sycl_timeline_tiers = 0;

static inline uint64_t sycl_timeline_now_ns(void) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
}

/* Returns the sequence number this record took, so a caller that also
 * emits device markers can stamp them with the host record they belong
 * to rather than with whatever the counter happens to read later. Zero
 * when recording is off, which no caller acts on: the device marker path
 * is gated on the level, not on the sequence number. */
static uint64_t sycl_timeline_record(uint32_t phase, int tier, int peer) {
    if (!g_sycl_timeline_on) return 0;
    const uint64_t seq  = g_sycl_timeline_seq.fetch_add(1, std::memory_order_relaxed);
    sycl_timeline_entry *r = &g_sycl_timeline_ring[seq & (kSyclTimelineCapacity - 1u)];
    r->ns    = sycl_timeline_now_ns() - g_sycl_timeline_base_ns;
    r->seq   = seq;
    r->tier  = tier;
    r->peer  = peer;
    r->phase = phase;
    r->pad   = 0;
    return seq;
}

/* One single_task that publishes this tier's busy flag and then snapshots
 * every other tier's, stamping {host seq, phase, own flag, observed busy
 * mask} into this tier's own ring slot. Nothing here waits, allocates, or
 * creates a profiling event; the returned sycl::event is dropped on
 * purpose. The ring slot index is computed on the host, so the kernel
 * needs no device-side counter and therefore no atomic.
 *
 * The accesses are through a volatile pointer so the compiler cannot sink
 * the flag store below the snapshot loads or fold a load of another
 * tier's flag into a value it thinks it already knows. That is a compiler
 * barrier, not a hardware one: whether one device's completed write is
 * actually visible to another device's later kernel is a property of the
 * driver, and is checked at init instead of assumed. */
static void sycl_timeline_device_mark(sycl::queue &q, uint32_t phase, int tier,
                                      uint64_t host_seq) {
    if (g_sycl_timeline_level < 2 || !g_sycl_timeline_dev) return;
    if (tier < 0 || tier >= DS4_MAX_GPUS) return;
    uint32_t      *buf   = g_sycl_timeline_dev;
    const uint32_t idx   = g_sycl_timeline_marks[tier]++;
    const uint32_t slot  = idx & (kSyclTimelineMarkersPerTier - 1u);
    const uint32_t self  = (idx << 1) | (phase == kSyclTimelineOwnedEnter ? 1u : 0u);
    const uint32_t ph    = phase;
    const uint32_t hseq  = (uint32_t)host_seq;
    const int      tiers = g_sycl_timeline_tiers;
    const int      t     = tier;
    try {
        q.single_task([=]() {
            volatile uint32_t *vb = buf;
            vb[kSyclTimelineFlagBase + (uint32_t)t] = self;
            uint32_t mask = 0;
            for (int j = 0; j < tiers; j++) {
                if (j == t) continue;
                if (vb[kSyclTimelineFlagBase + (uint32_t)j] & 1u) mask |= 1u << j;
            }
            const uint32_t base = kSyclTimelineMarkerBase +
                                  ((uint32_t)t * kSyclTimelineMarkersPerTier + slot) *
                                          kSyclTimelineMarkerWords;
            vb[base + 0] = hseq;
            vb[base + 1] = ph;
            vb[base + 2] = self;
            vb[base + 3] = mask | kSyclTimelineMarkerWritten;
        });
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "timeline: device marker failed, "
                "dropping to host-only: %s\n", e.what());
        g_sycl_timeline_level = 1;
    }
}

/* Does a write one device's kernel completed become visible to a later
 * kernel on another device, through the shared host USM allocation the
 * markers use? Every conclusion level 2 draws depends on it, and nothing
 * in SYCL promises it without a dependency between the two commands, so
 * it is measured here rather than assumed. Tier 0 writes a per-tier
 * token; each other tier copies what it reads into its own probe slot;
 * the host compares. Init-time only, a handful of one-item kernels, and
 * it is the one place in this header that waits on a queue.
 *
 * Same discipline as sycl_validate_peer_pair (sycl/ds4_sycl_mgpu.hpp),
 * which byte-validates peer access instead of trusting the driver's
 * report, and for the same reason: a diagnostic that silently reports "no
 * overlap" because a flag was never visible is worse than no diagnostic. */
static bool sycl_timeline_probe_visibility(std::vector<ds4_sycl_device> &devices) {
    uint32_t *buf = g_sycl_timeline_dev;
    for (size_t t = 1; t < devices.size(); t++) {
        const uint32_t token = 0xA5A50000u | (uint32_t)t;
        buf[kSyclTimelineProbeBase + t] = 0;
        try {
            devices[0].queue.single_task([=]() {
                volatile uint32_t *vb = buf;
                vb[kSyclTimelineProbeBase] = token;
            });
            devices[0].queue.wait_and_throw();
            devices[t].queue.single_task([=]() {
                volatile uint32_t *vb = buf;
                vb[kSyclTimelineProbeBase + (uint32_t)t] = vb[kSyclTimelineProbeBase];
            });
            devices[t].queue.wait_and_throw();
        } catch (const sycl::exception &e) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "timeline: cross-device visibility probe for tier %zu threw: %s\n",
                    t, e.what());
            return false;
        }
        if (buf[kSyclTimelineProbeBase + t] != token) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "timeline: tier %zu could not see tier 0's write to the shared "
                    "marker buffer (read 0x%08x, wanted 0x%08x); device markers would "
                    "report no overlap whether or not one happened\n",
                    t, buf[kSyclTimelineProbeBase + t], token);
            return false;
        }
    }
    return true;
}

/* Brackets one tier's owned decode, host record and device marker both,
 * on every exit path this dispatcher has: the ordinary one, the early
 * return taken when none of the six selected experts belongs to this
 * rank, four validation failures inside the try block, and an exception.
 * An unmatched enter would leave that tier's busy flag set for the rest
 * of the run, and every later marker on every other tier would then
 * report an overlap that never happened, so pairing has to come from
 * construction rather than from remembering to record an exit at six
 * places.
 *
 * The device marker is armed separately from the host record because the
 * queue is only resolved inside the try block, after the host record has
 * already been taken. A span that never reaches device_begin (the device
 * list went empty, or resolving the queue threw) emits no device marker
 * at either end, which keeps the pairing the overlap walk relies on. */
struct sycl_timeline_owned_span {
    sycl::queue *q;
    int          tier;
    uint64_t     enter_seq;
    explicit sycl_timeline_owned_span(int t)
            : q(nullptr), tier(t),
              enter_seq(sycl_timeline_record(kSyclTimelineOwnedEnter, t, -1)) {}
    void device_begin(sycl::queue &queue) {
        q = &queue;
        sycl_timeline_device_mark(queue, kSyclTimelineOwnedEnter, tier, enter_seq);
    }
    ~sycl_timeline_owned_span() {
        if (q) sycl_timeline_device_mark(*q, kSyclTimelineOwnedExit, tier, enter_seq);
        sycl_timeline_record(kSyclTimelineOwnedExit, tier, -1);
    }
    sycl_timeline_owned_span(const sycl_timeline_owned_span &) = delete;
    sycl_timeline_owned_span &operator=(const sycl_timeline_owned_span &) = delete;
};

static void sycl_timeline_teardown(void);

/* Called from ds4_gpu_init_multi once g_devices is built. `devices` is
 * g_devices; taking it as a parameter keeps this header free of that
 * global, which is defined in ds4_sycl.cpp's anonymous namespace after
 * this header is included. */
static void sycl_timeline_init(std::vector<ds4_sycl_device> &devices) {
    /* A second ds4_gpu_init_multi without an intervening ds4_gpu_cleanup
     * is legal (the engine retries a failed multi-tier startup), and it
     * would otherwise leak the previous run's marker buffer and leave its
     * ticket counter mid-stream. */
    sycl_timeline_teardown();
    const char *env = getenv("DS4_SYCL_TIMELINE");
    if (!env || env[0] == '0' || env[0] == '\0') return;
    g_sycl_timeline_level = atoi(env);
    if (g_sycl_timeline_level < 1) g_sycl_timeline_level = 1;

    const char *print_env = getenv("DS4_SYCL_TIMELINE_PRINT");
    if (print_env) g_sycl_timeline_print = (uint32_t)strtoul(print_env, nullptr, 10);

    if (!g_sycl_timeline_ring) {
        g_sycl_timeline_ring = (sycl_timeline_entry *)calloc(
                kSyclTimelineCapacity, sizeof(sycl_timeline_entry));
        if (!g_sycl_timeline_ring) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "timeline: ring allocation failed\n");
            g_sycl_timeline_level = 0;
            return;
        }
    }
    g_sycl_timeline_seq.store(0, std::memory_order_relaxed);
    g_sycl_timeline_base_ns = sycl_timeline_now_ns();

    g_sycl_timeline_tiers = (int)devices.size();
    if (g_sycl_timeline_tiers > DS4_MAX_GPUS) g_sycl_timeline_tiers = DS4_MAX_GPUS;
    memset(g_sycl_timeline_marks, 0, sizeof(g_sycl_timeline_marks));

    if (g_sycl_timeline_level >= 2 && !devices.empty()) {
        /* One shared allocation means one shared context: a box whose
         * devices span two SYCL platforms gets two contexts from
         * ds4_sycl_build_devices, and a host USM allocation made on one
         * of them is not addressable from the other. */
        bool ok = true;
        for (ds4_sycl_device &d : devices) {
            if (d.queue.get_context() != devices[0].queue.get_context()) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX "timeline: devices span more than "
                        "one context, device markers unavailable; host timeline only\n");
                ok = false;
                break;
            }
            if (!d.dev.has(sycl::aspect::usm_host_allocations)) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX "timeline: device \"%s\" lacks "
                        "usm_host_allocations, device markers unavailable; "
                        "host timeline only\n",
                        d.dev.get_info<sycl::info::device::name>().c_str());
                ok = false;
                break;
            }
        }
        const size_t words = kSyclTimelineMarkerBase +
                             (size_t)DS4_MAX_GPUS * kSyclTimelineMarkersPerTier *
                                     kSyclTimelineMarkerWords;
        if (ok) {
            try {
                g_sycl_timeline_dev = sycl::malloc_host<uint32_t>(words, devices[0].queue);
            } catch (const sycl::exception &e) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX "timeline: device marker buffer "
                        "allocation failed: %s\n", e.what());
                g_sycl_timeline_dev = nullptr;
            }
            if (g_sycl_timeline_dev) {
                memset(g_sycl_timeline_dev, 0, words * sizeof(uint32_t));
                g_sycl_timeline_dev_ctx.clear();
                g_sycl_timeline_dev_ctx.push_back(devices[0].queue.get_context());
            } else {
                ok = false;
            }
        }
        if (ok && !sycl_timeline_probe_visibility(devices)) ok = false;
        if (!ok) {
            g_sycl_timeline_level = 1;
            if (g_sycl_timeline_dev) {
                try {
                    sycl::free(g_sycl_timeline_dev, devices[0].queue.get_context());
                } catch (const sycl::exception &) { /* nothing left to recover */ }
                g_sycl_timeline_dev = nullptr;
                g_sycl_timeline_dev_ctx.clear();
            }
        } else {
            memset(g_sycl_timeline_dev, 0, words * sizeof(uint32_t));
        }
    } else if (g_sycl_timeline_level >= 2) {
        g_sycl_timeline_level = 1;
    }

    g_sycl_timeline_on = true;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "timeline: level %d, %u host record capacity%s\n",
            g_sycl_timeline_level, kSyclTimelineCapacity,
            g_sycl_timeline_level >= 2 ? ", device markers on" : "");
}

namespace {

struct sycl_timeline_device_marker {
    uint32_t host_seq;
    uint32_t phase;
    uint32_t tier;
    uint32_t index;   /* this tier's own marker count when it was issued */
    uint32_t others;  /* bitmask of tiers observed mid-decode */
};

/* Every marker slot a tier actually wrote, in the order that tier issued
 * them. Slots carry a written sentinel so a run that never filled the
 * ring does not report zeroed slots as markers, and a tier that wrapped
 * past kSyclTimelineMarkersPerTier reports the surviving tail. */
std::vector<sycl_timeline_device_marker> sycl_timeline_device_markers(uint32_t *wrapped_out) {
    std::vector<sycl_timeline_device_marker> out;
    *wrapped_out = 0;
    if (!g_sycl_timeline_dev) return out;
    for (int t = 0; t < g_sycl_timeline_tiers; t++) {
        const uint32_t issued = g_sycl_timeline_marks[t];
        if (issued > kSyclTimelineMarkersPerTier) *wrapped_out += 1;
        const uint32_t live  = issued < kSyclTimelineMarkersPerTier
                                       ? issued
                                       : kSyclTimelineMarkersPerTier;
        const uint32_t first = issued - live;
        for (uint32_t i = 0; i < live; i++) {
            const uint32_t slot = (first + i) & (kSyclTimelineMarkersPerTier - 1u);
            const uint32_t *rec = g_sycl_timeline_dev + kSyclTimelineMarkerBase +
                                  ((uint32_t)t * kSyclTimelineMarkersPerTier + slot) *
                                          kSyclTimelineMarkerWords;
            if (!(rec[3] & kSyclTimelineMarkerWritten)) continue;
            sycl_timeline_device_marker m;
            m.host_seq = rec[0];
            m.phase    = rec[1];
            m.tier     = (uint32_t)t;
            m.index    = rec[2] >> 1;
            m.others   = rec[3] & ~kSyclTimelineMarkerWritten;
            out.push_back(m);
        }
    }
    /* Sorted so the dump can interleave markers with the host records
     * they were stamped against. A tier's own enter and exit marker share
     * one host seq (both belong to the same owned span), so tier and the
     * tier's own issue index break the tie and an exit never prints
     * before its enter. */
    std::sort(out.begin(), out.end(),
              [](const sycl_timeline_device_marker &a,
                 const sycl_timeline_device_marker &b) {
                  if (a.host_seq != b.host_seq) return a.host_seq < b.host_seq;
                  if (a.tier != b.tier) return a.tier < b.tier;
                  return a.index < b.index;
              });
    return out;
}

/* The headline answer. A begin marker runs on the tier's own in_order
 * queue immediately before that tier's expert kernels and an end marker
 * immediately after them, so a marker that observed another tier's flag
 * set saw that tier genuinely mid-decode. Any such observation is direct
 * evidence the two tiers were busy at the same time; none at all, over
 * hundreds of routed layers, is direct evidence they were not. */
void sycl_timeline_report_device_overlap(void) {
    uint32_t wrapped = 0;
    std::vector<sycl_timeline_device_marker> marks = sycl_timeline_device_markers(&wrapped);
    if (marks.empty()) return;

    uint32_t enters[DS4_MAX_GPUS] = {0}, exits[DS4_MAX_GPUS] = {0};
    uint32_t enter_saw[DS4_MAX_GPUS] = {0}, exit_saw[DS4_MAX_GPUS] = {0};
    uint32_t total_saw = 0;
    for (const sycl_timeline_device_marker &m : marks) {
        if (m.tier >= (uint32_t)DS4_MAX_GPUS) continue;
        if (m.phase == kSyclTimelineOwnedEnter) {
            enters[m.tier]++;
            if (m.others) { enter_saw[m.tier]++; total_saw++; }
        } else if (m.phase == kSyclTimelineOwnedExit) {
            exits[m.tier]++;
            if (m.others) { exit_saw[m.tier]++; total_saw++; }
        }
    }

    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "timeline: device markers: %zu recorded%s\n", marks.size(),
            wrapped ? ", RING WRAPPED on at least one tier (raise "
                      "kSyclTimelineMarkersPerTier or shorten the run)"
                    : "");
    fprintf(stderr, DS4_GPU_LOG_PREFIX "timeline: %4s %8s %14s %8s %14s\n", "tier",
            "enters", "saw_other_busy", "exits", "saw_other_busy");
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        if (!enters[t] && !exits[t]) continue;
        fprintf(stderr, DS4_GPU_LOG_PREFIX "timeline: %4d %8u %14u %8u %14u\n", t, enters[t],
                enter_saw[t], exits[t], exit_saw[t]);
    }
    if (g_sycl_timeline_tiers < 2) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "timeline: OVERLAP: one tier configured, nothing to overlap; the marker "
                "mechanism itself recorded %zu markers\n",
                marks.size());
        return;
    }
    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "timeline: OVERLAP: %u of %zu device markers saw another tier mid-decode -> "
            "%s\n",
            total_saw, marks.size(),
            total_saw ? "TIERS DID OVERLAP" : "TIERS DID NOT OVERLAP");
}

void sycl_timeline_report_host(const std::vector<sycl_timeline_entry> &recs) {
    struct tier_stats {
        uint64_t owned_ns = 0, wait_ns = 0, xdev_ns = 0, combine_ns = 0;
        uint32_t owned_calls = 0, wait_calls = 0, xdev_calls = 0, combine_calls = 0;
        uint64_t owned_open_ns = 0, wait_open_ns = 0, xdev_open_ns = 0, combine_open_ns = 0;
        bool owned_open = false, wait_open = false, xdev_open = false, combine_open = false;
    } stats[DS4_MAX_GPUS];

    /* Time from one tier's owned exit to the next tier's owned enter: the
     * host-side gap expert parallelism needs to be near zero, since a
     * partner half that has not been issued yet cannot be overlapping
     * anything. */
    uint64_t gap_ns = 0, gap_max_ns = 0;
    uint32_t gaps = 0;
    bool     have_exit = false;
    uint64_t last_exit_ns = 0;
    int      last_exit_tier = -1;

    for (const sycl_timeline_entry &r : recs) {
        if (r.tier < 0 || r.tier >= DS4_MAX_GPUS) continue;
        tier_stats &s = stats[r.tier];
        switch (r.phase) {
        case kSyclTimelineOwnedEnter:
            s.owned_open = true;
            s.owned_open_ns = r.ns;
            if (have_exit && last_exit_tier != r.tier) {
                const uint64_t g = r.ns > last_exit_ns ? r.ns - last_exit_ns : 0;
                gap_ns += g;
                if (g > gap_max_ns) gap_max_ns = g;
                gaps++;
            }
            break;
        case kSyclTimelineOwnedExit:
            if (s.owned_open) {
                s.owned_ns += r.ns - s.owned_open_ns;
                s.owned_calls++;
                s.owned_open = false;
            }
            have_exit = true;
            last_exit_ns = r.ns;
            last_exit_tier = r.tier;
            break;
        case kSyclTimelineWaitEnter: s.wait_open = true; s.wait_open_ns = r.ns; break;
        case kSyclTimelineWaitExit:
            if (s.wait_open) {
                s.wait_ns += r.ns - s.wait_open_ns;
                s.wait_calls++;
                s.wait_open = false;
            }
            break;
        case kSyclTimelineXdevEnter: s.xdev_open = true; s.xdev_open_ns = r.ns; break;
        case kSyclTimelineXdevExit:
            if (s.xdev_open) {
                s.xdev_ns += r.ns - s.xdev_open_ns;
                s.xdev_calls++;
                s.xdev_open = false;
            }
            break;
        case kSyclTimelineCombineEnter: s.combine_open = true; s.combine_open_ns = r.ns; break;
        case kSyclTimelineCombineExit:
            if (s.combine_open) {
                s.combine_ns += r.ns - s.combine_open_ns;
                s.combine_calls++;
                s.combine_open = false;
            }
            break;
        default: break;
        }
    }

    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "timeline: host time inside each entry, microseconds\n");
    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "timeline: %4s %8s %12s %8s %12s %8s %12s %8s %12s\n",
            "tier", "owned", "owned_us", "waits", "wait_us", "xdev", "xdev_us", "comb",
            "comb_us");
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        const tier_stats &s = stats[t];
        if (!s.owned_calls && !s.wait_calls && !s.xdev_calls && !s.combine_calls) continue;
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "timeline: %4d %8u %12.1f %8u %12.1f %8u %12.1f %8u %12.1f\n",
                t, s.owned_calls, s.owned_ns / 1000.0, s.wait_calls, s.wait_ns / 1000.0,
                s.xdev_calls, s.xdev_ns / 1000.0, s.combine_calls, s.combine_ns / 1000.0);
    }
    if (gaps) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "timeline: host gap from one tier's owned exit to the other tier's owned "
                "enter: mean %.1f us, max %.1f us over %u handoffs\n",
                gap_ns / 1000.0 / gaps, gap_max_ns / 1000.0, gaps);
    }
}

}  /* namespace */

/* Dumped from ds4_gpu_cleanup, alongside the other teardown fan-outs, and
 * before g_devices.clear() destroys the queue the device marker buffer was
 * allocated on. */
static void sycl_timeline_dump(void) {
    if (!g_sycl_timeline_on) return;
    const uint64_t total = g_sycl_timeline_seq.load(std::memory_order_relaxed);
    const uint64_t live  = total < kSyclTimelineCapacity ? total : kSyclTimelineCapacity;
    const uint64_t first = total - live;

    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "timeline: %llu host records, %llu surviving in a %u-record ring\n",
            (unsigned long long)total, (unsigned long long)live, kSyclTimelineCapacity);

    std::vector<sycl_timeline_entry> recs;
    recs.reserve((size_t)live);
    for (uint64_t i = 0; i < live; i++) {
        recs.push_back(g_sycl_timeline_ring[(first + i) & (kSyclTimelineCapacity - 1u)]);
    }

    sycl_timeline_report_host(recs);
    sycl_timeline_report_device_overlap();

    /* The raw window starts at the first owned decode, which is where the
     * expert-parallel handoff begins; everything before it is model load
     * and the first token's attention, which answers nothing here. */
    size_t start = 0;
    for (size_t i = 0; i < recs.size(); i++) {
        if (recs[i].phase == kSyclTimelineOwnedEnter) { start = i; break; }
    }
    const size_t want = g_sycl_timeline_print == 0 ? recs.size() : g_sycl_timeline_print;
    const size_t end  = start + want < recs.size() ? start + want : recs.size();
    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "timeline: raw records %zu..%zu (DS4_SYCL_TIMELINE_PRINT=0 for all)\n",
            start, end);
    fprintf(stderr, DS4_GPU_LOG_PREFIX "timeline: %14s %6s %5s %5s  %s\n", "us", "seq",
            "tier", "peer", "event");

    /* Device markers are interleaved into the host window by the host seq
     * they were stamped with at submission, so one layer's four lines --
     * partner enter, partner device marker, home enter, home device
     * marker -- read together instead of in two tables the reader has to
     * join by hand. A device line's "saw" column is the whole point: a
     * nonzero mask there is the other tier caught mid-decode. */
    uint32_t wrapped = 0;
    std::vector<sycl_timeline_device_marker> marks = sycl_timeline_device_markers(&wrapped);
    size_t mi = 0;
    while (mi < marks.size() && start < recs.size() && marks[mi].host_seq < recs[start].seq) {
        mi++;
    }
    for (size_t i = start; i < end; i++) {
        const sycl_timeline_entry &r = recs[i];
        fprintf(stderr, DS4_GPU_LOG_PREFIX "timeline: %14.3f %6llu %5d %5d  %s\n",
                r.ns / 1000.0, (unsigned long long)r.seq, r.tier, r.peer,
                r.phase < kSyclTimelinePhaseCount ? kSyclTimelinePhaseNames[r.phase]
                                                  : "?");
        while (mi < marks.size() && marks[mi].host_seq <= r.seq) {
            const sycl_timeline_device_marker &m = marks[mi];
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "timeline: %14s %6u %5u %5s  device %s, saw busy tiers 0x%x\n",
                    "", m.host_seq, m.tier, "",
                    m.phase == kSyclTimelineOwnedEnter ? "enter" : "exit", m.others);
            mi++;
        }
    }
}

/* Frees the host USM marker buffer through the context it was allocated
 * on, for the same reason every other teardown fan-out in ds4_gpu_cleanup
 * runs before g_devices.clear(): that clear destroys the queues, and a
 * free afterwards has no live context to free through (spec 6g). */
static void sycl_timeline_teardown(void) {
    if (g_sycl_timeline_dev && !g_sycl_timeline_dev_ctx.empty()) {
        try {
            sycl::free(g_sycl_timeline_dev, g_sycl_timeline_dev_ctx[0]);
        } catch (const sycl::exception &e) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "timeline: marker buffer free failed: %s\n",
                    e.what());
        }
    }
    g_sycl_timeline_dev = nullptr;
    g_sycl_timeline_dev_ctx.clear();
    memset(g_sycl_timeline_marks, 0, sizeof(g_sycl_timeline_marks));
    g_sycl_timeline_tiers = 0;
    free(g_sycl_timeline_ring);
    g_sycl_timeline_ring  = nullptr;
    g_sycl_timeline_on    = false;
    g_sycl_timeline_level = 0;
    g_sycl_timeline_seq.store(0, std::memory_order_relaxed);
}
