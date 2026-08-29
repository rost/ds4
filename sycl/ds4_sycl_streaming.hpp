#pragma once

/* Resident SSD-streaming expert cache: fixed-size-class slab pooling with
 * LRU eviction and a dedicated-allocation fallback for experts outside the
 * established size class.
 *
 * Ported from rocm/ds4_rocm_runtime.cuh:106-170 (struct shapes),
 * :1417-1710 (find / evict / make-room / slab grow / slot acquire / alloc)
 * and :2680-2891 (cuda_stream_resident_seed_experts, backing
 * ds4_gpu_stream_expert_cache_seed_experts).  The slab pool exists because
 * a per-miss device-allocate-and-free round trip was a large fraction of
 * real decode time (rocm/ds4_rocm_runtime.cuh:124-130): hundreds of ~12 MiB
 * allocations and frees per generated token, each paying page-table cost.
 *
 * ROCm gates cuda_stream_resident_seed_experts and
 * ds4_gpu_stream_expert_cache_configured_count on a global
 * g_ssd_streaming_mode flag that only ds4_gpu_set_ssd_streaming (a later
 * task's entry) ever sets true.  Porting that gate here would make this
 * task's own tests unexercisable, so streaming is treated as
 * unconditionally enabled; a later task retrofits the gate once it
 * introduces the flag.
 *
 * Two things ROCm does are intentionally NOT ported:
 *   - evict_past_layers_first (env-var gated): out of scope, plain
 *     LRU-by-last_used only.
 *   - the live free-VRAM query via cudaMemGetInfo: SYCL/Level Zero has no
 *     portable free-memory query without Sysman.  sycl_stream_vram_ceiling
 *     below substitutes a static ceiling (total device memory minus a
 *     fixed reserve), which is strictly more conservative than a live
 *     query (it never overestimates headroom, and only gets safer as VRAM
 *     fragments).  Level Zero Sysman is what a later plan could use to
 *     restore a live check. */

#include "ds4_sycl_common.hpp"

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

static constexpr uint32_t DS4_SYCL_STREAM_MAX_N_EXPERT = 384u;
static constexpr uint32_t DS4_SYCL_STREAM_N_EXPERT_USED = 8u;

struct sycl_stream_resident_expert {
    const void *model_map;
    uint32_t    layer;
    int32_t     expert;
    uint64_t    gate_expert_bytes;
    uint64_t    down_expert_bytes;
    uint64_t    gate_offset;
    uint64_t    up_offset;
    uint64_t    down_offset;
    char       *base;
    char       *gate;
    char       *up;
    char       *down;
    uint64_t    bytes;
    uint64_t    last_used;
    bool        pooled;
};

struct sycl_stream_expert_slab {
    char    *base;
    uint64_t bytes;
};

struct sycl_stream_resident_key {
    const void *model_map;
    uint32_t    layer;
    int32_t     expert;
    uint64_t    gate_offset;
    uint64_t    up_offset;
    uint64_t    down_offset;
    uint64_t    gate_expert_bytes;
    uint64_t    down_expert_bytes;

    bool operator==(const sycl_stream_resident_key &o) const {
        return model_map == o.model_map && layer == o.layer &&
               expert == o.expert && gate_offset == o.gate_offset &&
               up_offset == o.up_offset && down_offset == o.down_offset &&
               gate_expert_bytes == o.gate_expert_bytes &&
               down_expert_bytes == o.down_expert_bytes;
    }
};

struct sycl_stream_resident_key_hash {
    size_t operator()(const sycl_stream_resident_key &k) const {
        uint64_t h = (uint64_t)(uintptr_t)k.model_map;
        auto mix = [&h](uint64_t v) {
            h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };
        mix((uint64_t)k.layer);
        mix((uint64_t)(uint32_t)k.expert);
        mix(k.gate_offset);
        mix(k.up_offset);
        mix(k.down_offset);
        mix(k.gate_expert_bytes);
        mix(k.down_expert_bytes);
        return (size_t)h;
    }
};

static std::vector<sycl_stream_resident_expert> g_sycl_stream_resident;
static std::unordered_map<sycl_stream_resident_key, size_t, sycl_stream_resident_key_hash>
        g_sycl_stream_resident_index;
static uint64_t      g_sycl_stream_resident_bytes = 0;
static uint64_t      g_sycl_stream_resident_clock = 0;
static std::vector<sycl_stream_expert_slab> g_sycl_stream_slabs;
static std::vector<char *>  g_sycl_stream_free_slots;
static uint64_t      g_sycl_stream_slot_bytes = 0;
static uint32_t      g_sycl_stream_slot_count = 0;
static uint32_t      g_sycl_stream_cache_budget = 0;

static sycl_stream_resident_key sycl_stream_make_key(
        const void *model_map, uint32_t layer, int32_t expert,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    return sycl_stream_resident_key{model_map,        layer,
                                    expert,           gate_offset,
                                    up_offset,        down_offset,
                                    gate_expert_bytes, down_expert_bytes};
}

static sycl_stream_resident_key sycl_stream_entry_key(const sycl_stream_resident_expert &e) {
    return sycl_stream_make_key(e.model_map, e.layer, e.expert, e.gate_offset,
                                e.up_offset, e.down_offset, e.gate_expert_bytes,
                                e.down_expert_bytes);
}

/* Non-mutating lookup: the caller bumps last_used on a hit (see
 * seed_experts below), so a pure lookup with no intent to use the entry
 * stays side-effect-free. */
static int sycl_stream_resident_find(
        const void *model_map, uint32_t layer, int32_t expert,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    const auto key = sycl_stream_make_key(model_map, layer, expert, gate_offset,
                                          up_offset, down_offset,
                                          gate_expert_bytes, down_expert_bytes);
    const auto it = g_sycl_stream_resident_index.find(key);
    if (it != g_sycl_stream_resident_index.end() &&
        it->second < g_sycl_stream_resident.size()) {
        return (int)it->second;
    }
    return -1;
}

/* An entry is protected from eviction only while it belongs to the same
 * layer as the admission in progress AND is one of that admission's own
 * chosen expert ids: this stops a seed batch from evicting its own
 * sibling members while making room for each other. */
static bool sycl_stream_is_protected(const sycl_stream_resident_expert &e, uint32_t layer,
                              const int32_t *protected_ids, uint32_t n_protected) {
    if (!protected_ids || e.layer != layer) return false;
    for (uint32_t i = 0; i < n_protected; i++) {
        if (e.expert == protected_ids[i]) return true;
    }
    return false;
}

static bool sycl_stream_evict_at(sycl::queue &q, size_t idx) {
    if (idx >= g_sycl_stream_resident.size()) return false;
    sycl_stream_resident_expert &e = g_sycl_stream_resident[idx];
    const auto evicted_key = sycl_stream_entry_key(e);
    if (e.base) {
        if (e.pooled) {
            g_sycl_stream_free_slots.push_back(e.base);
        } else {
            try {
                sycl::free(e.base, q);
            } catch (const sycl::exception &ex) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX
                        "streaming cache evict free failed: %s\n", ex.what());
            }
        }
    }
    g_sycl_stream_resident_bytes = g_sycl_stream_resident_bytes >= e.bytes
                                       ? g_sycl_stream_resident_bytes - e.bytes
                                       : 0;
    g_sycl_stream_resident_index.erase(evicted_key);
    /* Swap-remove keeps this O(1): move the last element into the evicted
     * slot, then fix up its index entry, rather than a vector erase. */
    const size_t last = g_sycl_stream_resident.size() - 1u;
    if (idx != last) {
        g_sycl_stream_resident[idx] = g_sycl_stream_resident[last];
        g_sycl_stream_resident_index[sycl_stream_entry_key(g_sycl_stream_resident[idx])] = idx;
    }
    g_sycl_stream_resident.pop_back();
    return true;
}

static bool sycl_stream_evict_one(sycl::queue &q, uint32_t layer,
                           const int32_t *protected_ids, uint32_t n_protected) {
    size_t   victim = (size_t)-1;
    uint64_t oldest = UINT64_MAX;
    for (size_t i = 0; i < g_sycl_stream_resident.size(); i++) {
        const sycl_stream_resident_expert &e = g_sycl_stream_resident[i];
        if (sycl_stream_is_protected(e, layer, protected_ids, n_protected)) continue;
        if (e.last_used < oldest) {
            oldest = e.last_used;
            victim = i;
        }
    }
    if (victim == (size_t)-1) return false;
    return sycl_stream_evict_at(q, victim);
}

static uint64_t sycl_stream_vram_ceiling(sycl::queue &q) {
    const uint64_t total =
            q.get_device().get_info<sycl::info::device::global_mem_size>();
    /* ROCm's own default free-VRAM reserve is 16 GiB (see
     * cuda_stream_resident_free_reserve_bytes).  A literal 16 GiB reserve
     * would leave zero headroom on a 16 GB dev box, so clamp the reserve
     * to at most a quarter of total memory. */
    const uint64_t max_reserve = 16ull * 1024 * 1024 * 1024;
    const uint64_t reserve = std::min(max_reserve, total / 4);
    return total > reserve ? total - reserve : 0;
}

static bool sycl_stream_resident_make_room(sycl::queue &q, uint64_t bytes, uint32_t layer,
                                    const int32_t *protected_ids, uint32_t n_protected) {
    while (g_sycl_stream_resident.size() >= g_sycl_stream_cache_budget) {
        if (!sycl_stream_evict_one(q, layer, protected_ids, n_protected)) return false;
    }
    const uint64_t ceiling = sycl_stream_vram_ceiling(q);
    while (g_sycl_stream_resident_bytes > ceiling ||
           bytes > ceiling - g_sycl_stream_resident_bytes) {
        if (!sycl_stream_evict_one(q, layer, protected_ids, n_protected)) return false;
    }
    return true;
}

static bool sycl_stream_expert_slab_grow(sycl::queue &q, uint64_t slot_bytes) {
    constexpr uint64_t slab_target_bytes = 1024ull * 1024 * 1024;
    if (g_sycl_stream_slot_count >= g_sycl_stream_cache_budget) return false;
    uint32_t slab_slots = slot_bytes >= slab_target_bytes
                              ? 1u
                              : (uint32_t)(slab_target_bytes / slot_bytes);
    const uint32_t want = g_sycl_stream_cache_budget - g_sycl_stream_slot_count;
    if (slab_slots > want) slab_slots = want;

    while (slab_slots != 0) {
        uint64_t slab_bytes = 0;
        if (!sycl_u64_mul_checked(slab_slots, slot_bytes, &slab_bytes)) {
            slab_slots >>= 1u;
            continue;
        }
        const uint64_t ceiling = sycl_stream_vram_ceiling(q);
        if (g_sycl_stream_resident_bytes > ceiling ||
            slab_bytes > ceiling - g_sycl_stream_resident_bytes) {
            slab_slots >>= 1u;
            continue;
        }
        void *base = sycl::malloc_device((size_t)slab_bytes, q);
        if (base == nullptr) {
            slab_slots >>= 1u;
            continue;
        }
        g_sycl_stream_slabs.push_back({(char *)base, slab_bytes});
        g_sycl_stream_free_slots.reserve(g_sycl_stream_free_slots.size() + slab_slots);
        for (uint32_t i = 0; i < slab_slots; i++) {
            g_sycl_stream_free_slots.push_back((char *)base + (uint64_t)i * slot_bytes);
        }
        g_sycl_stream_slot_count += slab_slots;
        return true;
    }
    return false;
}

/* The resident cache's size class is fixed by the first admission.  A
 * later call whose bytes differ returns nullptr for a genuine size-class
 * mismatch (not exhaustion), which the caller (alloc) uses to decide
 * whether to fall back to a dedicated allocation. */
static char *sycl_stream_expert_slot_acquire(sycl::queue &q, uint64_t bytes, uint32_t layer,
                                      const int32_t *protected_ids, uint32_t n_protected) {
    if (g_sycl_stream_slot_bytes == 0) g_sycl_stream_slot_bytes = bytes;
    if (bytes != g_sycl_stream_slot_bytes) return nullptr;
    for (;;) {
        if (!g_sycl_stream_free_slots.empty()) {
            char *slot = g_sycl_stream_free_slots.back();
            g_sycl_stream_free_slots.pop_back();
            return slot;
        }
        if (g_sycl_stream_slot_count < g_sycl_stream_cache_budget &&
            sycl_stream_expert_slab_grow(q, bytes)) {
            continue;
        }
        if (!sycl_stream_evict_one(q, layer, protected_ids, n_protected)) return nullptr;
    }
}

static int sycl_stream_resident_alloc(sycl::queue &q, const void *model_map, uint32_t layer,
                               int32_t expert, const int32_t *protected_ids,
                               uint32_t n_protected, uint64_t gate_offset,
                               uint64_t up_offset, uint64_t down_offset,
                               uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    if (g_sycl_stream_cache_budget == 0) return -1;

    uint64_t gate_pair = 0;
    if (!sycl_u64_mul_checked(2u, gate_expert_bytes, &gate_pair) ||
        gate_pair > UINT64_MAX - down_expert_bytes) {
        return -1;
    }
    const uint64_t bytes = gate_pair + down_expert_bytes;

    char *base = sycl_stream_expert_slot_acquire(q, bytes, layer, protected_ids,
                                                 n_protected);
    bool pooled = base != nullptr;
    if (!pooled && bytes == g_sycl_stream_slot_bytes) {
        /* Size class matches but the pool is genuinely exhausted (no free
         * slot, no room to grow, nothing evictable): a dedicated
         * allocation would just duplicate the pool, so fail outright
         * instead. */
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "streaming expert cache cannot reserve a %.2f MiB slot for "
                "layer=%u expert=%d\n",
                (double)bytes / 1048576.0, layer, expert);
        return -1;
    }
    if (!pooled) {
        /* Genuine size-class mismatch: fall back to a dedicated
         * allocation, matching rocm/ds4_rocm_runtime.cuh:1678-1710. */
        if (!sycl_stream_resident_make_room(q, bytes, layer, protected_ids,
                                            n_protected)) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "streaming expert cache cannot keep %.2f MiB for layer=%u "
                    "expert=%d\n",
                    (double)bytes / 1048576.0, layer, expert);
            return -1;
        }
        base = (char *)sycl::malloc_device((size_t)bytes, q);
        while (base == nullptr &&
               sycl_stream_evict_one(q, layer, protected_ids, n_protected)) {
            base = (char *)sycl::malloc_device((size_t)bytes, q);
        }
        if (base == nullptr) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "streaming expert cache allocation failed for layer=%u "
                    "expert=%d (%.2f MiB)\n",
                    layer, expert, (double)bytes / 1048576.0);
            return -1;
        }
    }

    sycl_stream_resident_expert e{};
    e.model_map          = model_map;
    e.layer              = layer;
    e.expert             = expert;
    e.gate_expert_bytes  = gate_expert_bytes;
    e.down_expert_bytes  = down_expert_bytes;
    e.gate_offset        = gate_offset;
    e.up_offset          = up_offset;
    e.down_offset        = down_offset;
    e.base               = base;
    e.gate               = base;
    e.up                 = base + gate_expert_bytes;
    e.down               = base + 2u * gate_expert_bytes;
    e.bytes              = bytes;
    e.last_used          = ++g_sycl_stream_resident_clock;
    e.pooled             = pooled;
    g_sycl_stream_resident.push_back(e);
    g_sycl_stream_resident_index[sycl_stream_entry_key(e)] =
            g_sycl_stream_resident.size() - 1u;
    g_sycl_stream_resident_bytes += bytes;
    return (int)g_sycl_stream_resident.size() - 1;
}

static void sycl_stream_resident_release_all(sycl::queue &q) {
    for (sycl_stream_resident_expert &e : g_sycl_stream_resident) {
        if (e.base && !e.pooled) {
            try {
                sycl::free(e.base, q);
            } catch (const sycl::exception &ex) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX
                        "streaming cache release free failed: %s\n", ex.what());
            }
        }
    }
    g_sycl_stream_resident.clear();
    g_sycl_stream_resident_index.clear();
    g_sycl_stream_resident_bytes = 0;
    g_sycl_stream_resident_clock = 0;
    for (sycl_stream_expert_slab &slab : g_sycl_stream_slabs) {
        if (slab.base) {
            try {
                sycl::free(slab.base, q);
            } catch (const sycl::exception &ex) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX
                        "streaming cache release slab free failed: %s\n", ex.what());
            }
        }
    }
    g_sycl_stream_slabs.clear();
    g_sycl_stream_free_slots.clear();
    g_sycl_stream_slot_bytes = 0;
    g_sycl_stream_slot_count = 0;
}

/* Deduplicates expert_ids by expert value (keeping the highest-priority
 * entry for each), selects the top seed_cap distinct experts by priority,
 * and returns them sorted ASCENDING by priority (lowest first): the
 * caller processes in that order so the highest-priority expert ends up
 * with the freshest last_used clock and is therefore least likely to be
 * evicted first.  ROCm builds this with an insertion sort into a
 * fixed-size array (rocm/ds4_rocm_runtime.cuh:2744-2765); that exact
 * shape is not load-bearing, only the "top seed_cap by priority, deduped
 * by expert id" result is. */
static std::vector<uint32_t> sycl_stream_select_chosen_ascending(
        const int32_t *expert_ids, const uint32_t *expert_priorities,
        uint32_t n_experts, uint32_t n_total_expert, uint32_t seed_cap) {
    std::vector<bool>     seen(n_total_expert, false);
    std::vector<uint32_t> best_index(n_total_expert);
    std::vector<uint32_t> best_priority(n_total_expert);
    for (uint32_t i = 0; i < n_experts; i++) {
        const uint32_t expert   = (uint32_t)expert_ids[i];
        const uint32_t priority = expert_priorities ? expert_priorities[i]
                                                     : (n_experts - i);
        if (!seen[expert] || priority > best_priority[expert]) {
            seen[expert]          = true;
            best_index[expert]    = i;
            best_priority[expert] = priority;
        }
    }

    std::vector<std::pair<uint32_t, uint32_t>> candidates;  /* (priority, index) */
    for (uint32_t expert = 0; expert < n_total_expert; expert++) {
        if (seen[expert]) candidates.push_back({best_priority[expert], best_index[expert]});
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const auto &a, const auto &b) { return a.first > b.first; });
    if (candidates.size() > seed_cap) candidates.resize(seed_cap);

    std::vector<uint32_t> ascending;
    ascending.reserve(candidates.size());
    for (auto it = candidates.rbegin(); it != candidates.rend(); ++it) {
        ascending.push_back(it->second);
    }
    return ascending;
}

static int sycl_stream_resident_seed_experts(
        const void *model_map, uint64_t model_size, uint32_t layer,
        const int32_t *expert_ids, const uint32_t *expert_priorities,
        uint32_t n_experts, uint32_t n_total_expert, uint64_t gate_offset,
        uint64_t up_offset, uint64_t down_offset, uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!model_map || !expert_ids || n_experts == 0 || n_total_expert == 0 ||
        n_total_expert > DS4_SYCL_STREAM_MAX_N_EXPERT || gate_expert_bytes == 0 ||
        down_expert_bytes == 0) {
        return 0;
    }
    /* budget == 0 is the one true zero-work-succeeds case here, distinct
     * from the malformed-input checks above which return 0 (see the
     * ds4_gpu_stream_expert_cache_seed_experts wrapper for the null-table
     * case, which fails before this function is ever called). */
    if (g_sycl_stream_cache_budget == 0) return 1;

    for (uint32_t i = 0; i < n_experts; i++) {
        const int32_t expert_i = expert_ids[i];
        if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "streaming seed expert id %d outside 0..%u (layer=%u)\n",
                    expert_i, n_total_expert, layer);
            return 0;
        }
    }

    uint64_t gate_bytes = 0, down_bytes = 0;
    if (!sycl_u64_mul_checked(n_total_expert, gate_expert_bytes, &gate_bytes) ||
        !sycl_u64_mul_checked(n_total_expert, down_expert_bytes, &down_bytes) ||
        !sycl_model_range_fits(model_size, gate_offset, gate_bytes) ||
        !sycl_model_range_fits(model_size, up_offset, gate_bytes) ||
        !sycl_model_range_fits(model_size, down_offset, down_bytes)) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "streaming seed expert range outside model map\n");
        return 0;
    }

    const uint32_t seed_cap =
            std::min({n_experts, g_sycl_stream_cache_budget, DS4_SYCL_STREAM_MAX_N_EXPERT});
    if (seed_cap == 0) return 1;

    const std::vector<uint32_t> chosen = sycl_stream_select_chosen_ascending(
            expert_ids, expert_priorities, n_experts, n_total_expert, seed_cap);
    const uint32_t chosen_count = (uint32_t)chosen.size();
    if (chosen_count == 0) return 1;

    std::vector<int32_t> protected_ids(chosen_count);
    for (uint32_t i = 0; i < chosen_count; i++) protected_ids[i] = expert_ids[chosen[i]];

    sycl::queue &q  = ds4_sycl_current_queue();
    bool         ok = true;
    for (uint32_t i = 0; ok && i < chosen_count; i++) {
        const int32_t expert_i = expert_ids[chosen[i]];
        int idx = sycl_stream_resident_find(model_map, layer, expert_i, gate_offset,
                                            up_offset, down_offset, gate_expert_bytes,
                                            down_expert_bytes);
        if (idx >= 0) {
            g_sycl_stream_resident[(size_t)idx].last_used = ++g_sycl_stream_resident_clock;
            continue;
        }

        idx = sycl_stream_resident_alloc(q, model_map, layer, expert_i,
                                         protected_ids.data(), chosen_count,
                                         gate_offset, up_offset, down_offset,
                                         gate_expert_bytes, down_expert_bytes);
        if (idx < 0) {
            ok = false;
            break;
        }

        const uint64_t expert = (uint64_t)(uint32_t)expert_i;
        uint64_t       gate_rel = 0, down_rel = 0;
        if (!sycl_u64_mul_checked(expert, gate_expert_bytes, &gate_rel) ||
            !sycl_u64_mul_checked(expert, down_expert_bytes, &down_rel)) {
            ok = false;
            break;
        }

        sycl_stream_resident_expert &entry = g_sycl_stream_resident[(size_t)idx];
        try {
            /* Mechanism A: direct mmap-to-device copy, synchronous in
             * effect via the single wait after the batch. UP uses the
             * GATE-sized stride, per rocm/ds4_rocm_runtime.cuh:2825. */
            q.memcpy(entry.gate, (const char *)model_map + gate_offset + gate_rel,
                     gate_expert_bytes);
            q.memcpy(entry.up, (const char *)model_map + up_offset + gate_rel,
                     gate_expert_bytes);
            q.memcpy(entry.down, (const char *)model_map + down_offset + down_rel,
                     down_expert_bytes);
            q.wait_and_throw();
        } catch (const sycl::exception &ex) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming seed upload failed: %s\n",
                    ex.what());
            ok = false;
            break;
        }
    }

    if (!ok) {
        /* Fail-safe: release the whole resident cache rather than leave a
         * partially loaded entry behind, matching
         * rocm/ds4_rocm_runtime.cuh:2894-2905.  This is reported as a
         * no-op success (1) to the caller, not a hard failure: it is
         * distinct from the malformed-input checks above, which do
         * return 0. */
        sycl_stream_resident_release_all(q);
        return 1;
    }
    return 1;
}

/* Selected-expert scratch cache: one reusable compacted buffer holding
 * the current decode step's up-to-DS4_SYCL_STREAM_N_EXPERT_USED selected
 * experts' gate/up/down bytes, back to back in slot order.  Grow-only
 * capacity, matching rocm/ds4_rocm_runtime.cuh:82-104 and
 * cuda_stream_selected_ensure_buffers (:1036-1074), minus the
 * pinned-host-staging and device-side pointer-array pieces this port
 * does not need: see the "always resolve synchronously" note on
 * sycl_stream_selected_load below for why. */
struct sycl_stream_selected_cache {
    int         loaded = 0;
    const void *model_map = nullptr;
    uint32_t    layer = 0;
    uint32_t    n_total_expert = 0;
    uint32_t    n_selected = 0;
    uint64_t    gate_expert_bytes = 0;
    uint64_t    down_expert_bytes = 0;
    int32_t     selected_ids[DS4_SYCL_STREAM_N_EXPERT_USED] = {};
    char       *gate = nullptr;
    char       *up = nullptr;
    char       *down = nullptr;
    uint64_t    gate_capacity = 0;
    uint64_t    down_capacity = 0;
};

static sycl_stream_selected_cache g_sycl_stream_selected;

/* Resident-cache hit/miss counters, incremented only by the two
 * admission loops below (selected load and batch prepare): a hit is a
 * sycl_stream_resident_find success (no mmap-to-device copy needed), a
 * miss is a sycl_stream_resident_alloc call (a genuine admission that
 * does copy).  Test-only instrumentation proving the "a resident hit
 * short-circuits re-copying" property; not part of the ABI. */
static uint64_t g_sycl_stream_hits = 0;
static uint64_t g_sycl_stream_misses = 0;

static void sycl_stream_selected_release(sycl::queue &q) {
    if (g_sycl_stream_selected.gate) {
        try { sycl::free(g_sycl_stream_selected.gate, q); }
        catch (const sycl::exception &ex) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "streaming selected cache release failed: %s\n", ex.what());
        }
    }
    if (g_sycl_stream_selected.up) {
        try { sycl::free(g_sycl_stream_selected.up, q); }
        catch (const sycl::exception &ex) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "streaming selected cache release failed: %s\n", ex.what());
        }
    }
    if (g_sycl_stream_selected.down) {
        try { sycl::free(g_sycl_stream_selected.down, q); }
        catch (const sycl::exception &ex) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "streaming selected cache release failed: %s\n", ex.what());
        }
    }
    g_sycl_stream_selected = sycl_stream_selected_cache{};
}

static bool sycl_stream_selected_ensure_buffers(sycl::queue &q, uint64_t gate_bytes,
                                                uint64_t down_bytes) {
    if (gate_bytes == 0 || down_bytes == 0) return false;
    if (g_sycl_stream_selected.gate_capacity < gate_bytes) {
        if (g_sycl_stream_selected.gate) sycl::free(g_sycl_stream_selected.gate, q);
        if (g_sycl_stream_selected.up) sycl::free(g_sycl_stream_selected.up, q);
        g_sycl_stream_selected.gate = (char *)sycl::malloc_device((size_t)gate_bytes, q);
        g_sycl_stream_selected.up = (char *)sycl::malloc_device((size_t)gate_bytes, q);
        g_sycl_stream_selected.gate_capacity = 0;
        if (!g_sycl_stream_selected.gate || !g_sycl_stream_selected.up) return false;
        g_sycl_stream_selected.gate_capacity = gate_bytes;
    }
    if (g_sycl_stream_selected.down_capacity < down_bytes) {
        if (g_sycl_stream_selected.down) sycl::free(g_sycl_stream_selected.down, q);
        g_sycl_stream_selected.down = (char *)sycl::malloc_device((size_t)down_bytes, q);
        g_sycl_stream_selected.down_capacity = 0;
        if (!g_sycl_stream_selected.down) return false;
        g_sycl_stream_selected.down_capacity = down_bytes;
    }
    return true;
}

/* Backs both ds4_gpu_stream_expert_cache_begin_selected_load and
 * ds4_gpu_stream_expert_cache_seed_selected (ROCm's cuda_stream_selected_load,
 * rocm/ds4_rocm_runtime.cuh:4041 onward).
 *
 * ROCm defers finishing this call (leaving g_stream_selected_pending.active
 * set, building device-side pointer arrays instead of a compacted buffer)
 * when every requested id is already resident, as a copy-avoidance
 * optimisation consumed only by rocm/ds4_rocm_moe_launch.cuh's MoE-launch
 * kernels (cuda_stream_selected_apply/_apply_split), entirely out of
 * scope here.  That pending state IS set on a Flash-reachable path in
 * ROCm (ds4.c's metal_graph_decode_cuda_selected_load, guarded
 * "!DS4_ROCM_BUILD && !DS4_NO_GPU && !__APPLE__", calls begin_selected_load
 * and never resolves it afterward), so "assume it is always inactive"
 * would be wrong to assert blindly; the resolution here is to never
 * create the pending state in the first place: this function always
 * fully resolves before returning, unconditionally building the
 * compacted buffer for every selected slot regardless of which were
 * already resident.  This is a strict behavioural superset of ROCm
 * (correct in every case ROCm is correct, at the cost of one avoidable
 * device-to-device copy in the all-resident case, which has no consumer
 * here to notice). */
static int sycl_stream_selected_load(
        const void *model_map, uint64_t model_size, uint32_t layer,
        const int32_t *selected_ids, uint32_t n_total_expert, uint32_t n_selected,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    if (!model_map || !selected_ids || n_total_expert == 0 ||
        n_total_expert > DS4_SYCL_STREAM_MAX_N_EXPERT || n_selected == 0 ||
        n_selected > DS4_SYCL_STREAM_N_EXPERT_USED || gate_expert_bytes == 0 ||
        down_expert_bytes == 0) {
        return 0;
    }

    for (uint32_t i = 0; i < n_selected; i++) {
        if (selected_ids[i] < 0 || (uint32_t)selected_ids[i] >= n_total_expert) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "streaming selected expert id %d outside 0..%u (layer=%u)\n",
                    selected_ids[i], n_total_expert, layer);
            return 0;
        }
    }

    uint64_t gate_bytes = 0, down_bytes = 0;
    if (!sycl_u64_mul_checked(n_selected, gate_expert_bytes, &gate_bytes) ||
        !sycl_u64_mul_checked(n_selected, down_expert_bytes, &down_bytes)) {
        return 0;
    }
    for (uint32_t i = 0; i < n_selected; i++) {
        const uint64_t expert = (uint64_t)(uint32_t)selected_ids[i];
        uint64_t       gate_rel = 0, down_rel = 0;
        if (!sycl_u64_mul_checked(expert, gate_expert_bytes, &gate_rel) ||
            !sycl_u64_mul_checked(expert, down_expert_bytes, &down_rel) ||
            !sycl_model_range_fits(model_size, gate_offset + gate_rel, gate_expert_bytes) ||
            !sycl_model_range_fits(model_size, up_offset + gate_rel, gate_expert_bytes) ||
            !sycl_model_range_fits(model_size, down_offset + down_rel, down_expert_bytes)) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "streaming selected expert offset outside model map\n");
            return 0;
        }
    }

    sycl::queue &q = ds4_sycl_current_queue();
    if (!sycl_stream_selected_ensure_buffers(q, gate_bytes, down_bytes)) return 0;

    g_sycl_stream_selected.model_map = model_map;
    g_sycl_stream_selected.layer = layer;
    g_sycl_stream_selected.n_total_expert = n_total_expert;
    g_sycl_stream_selected.n_selected = n_selected;
    g_sycl_stream_selected.gate_expert_bytes = gate_expert_bytes;
    g_sycl_stream_selected.down_expert_bytes = down_expert_bytes;
    for (uint32_t i = 0; i < n_selected; i++) {
        g_sycl_stream_selected.selected_ids[i] = selected_ids[i];
    }

    for (uint32_t i = 0; i < n_selected; i++) {
        const int32_t expert_i = selected_ids[i];
        int idx = sycl_stream_resident_find(model_map, layer, expert_i, gate_offset,
                                            up_offset, down_offset, gate_expert_bytes,
                                            down_expert_bytes);
        if (idx >= 0) {
            g_sycl_stream_resident[(size_t)idx].last_used = ++g_sycl_stream_resident_clock;
            g_sycl_stream_hits++;
            continue;
        }

        idx = sycl_stream_resident_alloc(q, model_map, layer, expert_i, selected_ids,
                                         n_selected, gate_offset, up_offset, down_offset,
                                         gate_expert_bytes, down_expert_bytes);
        if (idx < 0) {
            sycl_stream_resident_release_all(q);
            return 0;
        }
        g_sycl_stream_misses++;

        const uint64_t expert = (uint64_t)(uint32_t)expert_i;
        uint64_t       gate_rel = 0, down_rel = 0;
        if (!sycl_u64_mul_checked(expert, gate_expert_bytes, &gate_rel) ||
            !sycl_u64_mul_checked(expert, down_expert_bytes, &down_rel)) {
            sycl_stream_resident_release_all(q);
            return 0;
        }

        sycl_stream_resident_expert &entry = g_sycl_stream_resident[(size_t)idx];
        try {
            q.memcpy(entry.gate, (const char *)model_map + gate_offset + gate_rel,
                     gate_expert_bytes);
            q.memcpy(entry.up, (const char *)model_map + up_offset + gate_rel,
                     gate_expert_bytes);
            q.memcpy(entry.down, (const char *)model_map + down_offset + down_rel,
                     down_expert_bytes);
            q.wait_and_throw();
        } catch (const sycl::exception &ex) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected upload failed: %s\n",
                    ex.what());
            sycl_stream_resident_release_all(q);
            return 0;
        }
    }

    /* Unconditionally compact every slot (whether freshly admitted or
     * already resident) into the scratch buffer, and bump last_used
     * again for the ones that were only just looked up above, matching
     * rocm/ds4_rocm_runtime.cuh:2966's clock bump inside
     * cuda_stream_selected_compact_mask. */
    try {
        for (uint32_t i = 0; i < n_selected; i++) {
            const int idx = sycl_stream_resident_find(
                    model_map, layer, selected_ids[i], gate_offset, up_offset,
                    down_offset, gate_expert_bytes, down_expert_bytes);
            if (idx < 0) {
                sycl_stream_resident_release_all(q);
                return 0;
            }
            sycl_stream_resident_expert &entry = g_sycl_stream_resident[(size_t)idx];
            entry.last_used = ++g_sycl_stream_resident_clock;
            q.memcpy(g_sycl_stream_selected.gate + (uint64_t)i * gate_expert_bytes,
                     entry.gate, gate_expert_bytes);
            q.memcpy(g_sycl_stream_selected.up + (uint64_t)i * gate_expert_bytes,
                     entry.up, gate_expert_bytes);
            q.memcpy(g_sycl_stream_selected.down + (uint64_t)i * down_expert_bytes,
                     entry.down, down_expert_bytes);
        }
        q.wait_and_throw();
    } catch (const sycl::exception &ex) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected compact failed: %s\n",
                ex.what());
        sycl_stream_resident_release_all(q);
        return 0;
    }

    g_sycl_stream_selected.loaded = 1;
    return 1;
}

/* Pure host-side dedup: builds the map[expert] -> first-seen unique slot
 * table and the flat compact_ids[token*n_selected] mapping, matching
 * rocm/ds4_rocm_runtime.cuh:3109 onward's host-side loop (no device calls
 * in this step, ports unchanged).  Bails out if a distinct
 * (DS4_SYCL_STREAM_MAX_N_EXPERT + 1)th expert would be needed. */
static bool sycl_stream_batch_dedup(const int32_t *ids, uint32_t n_tokens,
                                    uint32_t n_selected, uint32_t n_total_expert,
                                    std::vector<int32_t> &unique_ids_out,
                                    std::vector<uint32_t> &compact_ids_out) {
    uint64_t n_ids64 = (uint64_t)n_tokens * (uint64_t)n_selected;
    std::vector<int32_t> slot_of(n_total_expert, -1);
    unique_ids_out.clear();
    compact_ids_out.assign((size_t)n_ids64, 0u);
    for (uint64_t i = 0; i < n_ids64; i++) {
        const int32_t expert = ids[i];
        if (expert < 0 || (uint32_t)expert >= n_total_expert) return false;
        int32_t slot = slot_of[(uint32_t)expert];
        if (slot < 0) {
            if (unique_ids_out.size() >= DS4_SYCL_STREAM_MAX_N_EXPERT) return false;
            slot = (int32_t)unique_ids_out.size();
            slot_of[(uint32_t)expert] = slot;
            unique_ids_out.push_back(expert);
        }
        compact_ids_out[i] = (uint32_t)slot;
    }
    return !unique_ids_out.empty();
}

/* Backs ds4_gpu_stream_expert_cache_prepare_selected_batch (ROCm's
 * cuda_stream_batch_selected_prepare_from_host, rocm/ds4_rocm_runtime.cuh:3109
 * onward).  Does not build or upload the device-side pointer-array /
 * pair_missing tables ROCm uploads for its MoE-launch consumer: no
 * compute kernel here reads them.  What is tested here
 * is the host-side dedup mapping (exposed via a test-only hook) and that
 * every unique expert ends up resident with the right bytes. */
static int sycl_stream_batch_selected_prepare(
        const void *model_map, uint64_t model_size, uint32_t layer,
        const int32_t *ids, uint32_t n_tokens, uint32_t n_total_expert,
        uint32_t n_selected, uint64_t gate_offset, uint64_t up_offset,
        uint64_t down_offset, uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    if (!model_map || !ids || n_tokens <= 1 || n_total_expert == 0 ||
        n_total_expert > DS4_SYCL_STREAM_MAX_N_EXPERT || n_selected == 0 ||
        n_selected > DS4_SYCL_STREAM_N_EXPERT_USED || gate_expert_bytes == 0 ||
        down_expert_bytes == 0) {
        return 0;
    }

    std::vector<int32_t>  unique_ids;
    std::vector<uint32_t> compact_ids;
    if (!sycl_stream_batch_dedup(ids, n_tokens, n_selected, n_total_expert, unique_ids,
                                compact_ids)) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "streaming batch selected dedup failed (layer=%u)\n", layer);
        return 0;
    }

    for (int32_t expert_i : unique_ids) {
        const uint64_t expert = (uint64_t)(uint32_t)expert_i;
        uint64_t       gate_rel = 0, down_rel = 0;
        if (!sycl_u64_mul_checked(expert, gate_expert_bytes, &gate_rel) ||
            !sycl_u64_mul_checked(expert, down_expert_bytes, &down_rel) ||
            !sycl_model_range_fits(model_size, gate_offset + gate_rel, gate_expert_bytes) ||
            !sycl_model_range_fits(model_size, up_offset + gate_rel, gate_expert_bytes) ||
            !sycl_model_range_fits(model_size, down_offset + down_rel, down_expert_bytes)) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "streaming batch selected expert offset outside model map\n");
            return 0;
        }
    }

    sycl::queue &q = ds4_sycl_current_queue();
    for (int32_t expert_i : unique_ids) {
        int idx = sycl_stream_resident_find(model_map, layer, expert_i, gate_offset,
                                            up_offset, down_offset, gate_expert_bytes,
                                            down_expert_bytes);
        if (idx >= 0) {
            g_sycl_stream_resident[(size_t)idx].last_used = ++g_sycl_stream_resident_clock;
            g_sycl_stream_hits++;
            continue;
        }

        idx = sycl_stream_resident_alloc(q, model_map, layer, expert_i,
                                         unique_ids.data(), (uint32_t)unique_ids.size(),
                                         gate_offset, up_offset, down_offset,
                                         gate_expert_bytes, down_expert_bytes);
        /* No release-whole-cache-on-failure here: ROCm's own failure path
         * at this exact point (rocm/ds4_rocm_runtime.cuh:3288-3306) just
         * breaks out and returns 0, unlike begin_selected_load's failure
         * path, which does release.  Matched deliberately, not an
         * oversight. */
        if (idx < 0) return 0;
        g_sycl_stream_misses++;

        const uint64_t expert = (uint64_t)(uint32_t)expert_i;
        uint64_t       gate_rel = 0, down_rel = 0;
        if (!sycl_u64_mul_checked(expert, gate_expert_bytes, &gate_rel) ||
            !sycl_u64_mul_checked(expert, down_expert_bytes, &down_rel)) {
            return 0;
        }

        sycl_stream_resident_expert &entry = g_sycl_stream_resident[(size_t)idx];
        try {
            q.memcpy(entry.gate, (const char *)model_map + gate_offset + gate_rel,
                     gate_expert_bytes);
            q.memcpy(entry.up, (const char *)model_map + up_offset + gate_rel,
                     gate_expert_bytes);
            q.memcpy(entry.down, (const char *)model_map + down_offset + down_rel,
                     down_expert_bytes);
            q.wait_and_throw();
        } catch (const sycl::exception &ex) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming batch selected upload failed: %s\n",
                    ex.what());
            return 0;
        }
    }
    return 1;
}

extern "C" void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    g_sycl_stream_cache_budget = experts;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_configured_count(void) {
    return g_sycl_stream_cache_budget;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_current_count(void) {
    return (uint32_t)g_sycl_stream_resident.size();
}

/* Ignores both byte arguments, exactly like ROCm
 * (rocm/ds4_rocm_current_api_compat.cuh:170): this is not an oversight,
 * the resident cache's admission count is purely budget-driven. */
extern "C" uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    return ds4_gpu_stream_expert_cache_configured_count();
}

extern "C" int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table, const int32_t *expert_ids,
        const uint32_t *expert_priorities, uint32_t n_experts) {
    /* A null table is a hard failure, matching
     * rocm/ds4_rocm_current_api_compat.cuh:293 ("if (!table) return 0;"),
     * checked ahead of cuda_stream_resident_seed_experts; ds4.c's caller
     * treats a 0 result as failure.  budget == 0 is the only
     * zero-work-succeeds path (handled inside
     * sycl_stream_resident_seed_experts above). */
    if (!table) return 0;
    return sycl_stream_resident_seed_experts(
            table->model_map, table->model_size, table->layer, expert_ids,
            expert_priorities, n_experts, table->n_total_expert, table->gate_offset,
            table->up_offset, table->down_offset, table->gate_expert_bytes,
            table->down_expert_bytes);
}

extern "C" int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table, const int32_t *selected_ids,
        uint32_t n_selected) {
    /* Null table: hard failure, matching
     * rocm/ds4_rocm_current_api_compat.cuh:199-215 ("if (!table) return
     * 0;"); ds4.c's callers check "== 0" as failure. */
    if (!table) return 0;
    return sycl_stream_selected_load(
            table->model_map, table->model_size, table->layer, selected_ids,
            table->n_total_expert, n_selected, table->gate_offset, table->up_offset,
            table->down_offset, table->gate_expert_bytes, table->down_expert_bytes);
}

/* ROCm's seed_selected wrapper is begin_selected_load followed by
 * cuda_stream_selected_finish_pending_missing(0)
 * (rocm/ds4_rocm_current_api_compat.cuh:178-197), and that finish step is
 * a no-op whenever nothing is left pending.  Since
 * sycl_stream_selected_load never leaves anything pending (see its
 * comment above), the two entries are structurally identical here; this
 * is a correct consequence of that design, not a missed distinction. */
extern "C" int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table, const int32_t *selected_ids,
        uint32_t n_selected) {
    return ds4_gpu_stream_expert_cache_begin_selected_load(table, selected_ids,
                                                           n_selected);
}

extern "C" int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table, const int32_t *selected_ids,
        uint32_t n_tokens, uint32_t n_selected) {
    /* Null table: hard failure, matching
     * rocm/ds4_rocm_current_api_compat.cuh:217-246 ("if (!table) return
     * 0;"); ds4.c's callers check "!= 0" as success. */
    if (!table) return 0;
    return sycl_stream_batch_selected_prepare(
            table->model_map, table->model_size, table->layer, selected_ids,
            n_tokens, table->n_total_expert, n_selected, table->gate_offset,
            table->up_offset, table->down_offset, table->gate_expert_bytes,
            table->down_expert_bytes);
}

/* Test-only side doors.  None of the entries above expose the resident
 * cache's or the selected cache's device-side bytes, the hit/miss
 * counters, the batch dedup mapping, or a way to reset all of it between
 * test cases.  Forward-declared only in tests/test_sycl_streaming.c,
 * never added to ds4_gpu.h. */
extern "C" int ds4_sycl_stream_test_read_resident(uint32_t layer, int32_t expert,
                                                  void *gate_out, void *up_out,
                                                  void *down_out) {
    for (const sycl_stream_resident_expert &e : g_sycl_stream_resident) {
        if (e.layer != layer || e.expert != expert) continue;
        sycl::queue &q = ds4_sycl_current_queue();
        try {
            if (gate_out) q.memcpy(gate_out, e.gate, e.gate_expert_bytes);
            if (up_out) q.memcpy(up_out, e.up, e.gate_expert_bytes);
            if (down_out) q.memcpy(down_out, e.down, e.down_expert_bytes);
            q.wait_and_throw();
        } catch (const sycl::exception &ex) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "stream test readback failed: %s\n",
                    ex.what());
            return 0;
        }
        return 1;
    }
    return 0;
}

extern "C" int ds4_sycl_stream_test_read_selected(uint32_t slot, void *gate_out,
                                                  void *up_out, void *down_out) {
    if (!g_sycl_stream_selected.loaded || slot >= g_sycl_stream_selected.n_selected) {
        return 0;
    }
    sycl::queue &q = ds4_sycl_current_queue();
    const uint64_t gate_bytes = g_sycl_stream_selected.gate_expert_bytes;
    const uint64_t down_bytes = g_sycl_stream_selected.down_expert_bytes;
    try {
        if (gate_out) {
            q.memcpy(gate_out, g_sycl_stream_selected.gate + (uint64_t)slot * gate_bytes,
                     gate_bytes);
        }
        if (up_out) {
            q.memcpy(up_out, g_sycl_stream_selected.up + (uint64_t)slot * gate_bytes,
                     gate_bytes);
        }
        if (down_out) {
            q.memcpy(down_out, g_sycl_stream_selected.down + (uint64_t)slot * down_bytes,
                     down_bytes);
        }
        q.wait_and_throw();
    } catch (const sycl::exception &ex) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "stream test selected readback failed: %s\n",
                ex.what());
        return 0;
    }
    return 1;
}

extern "C" void ds4_sycl_stream_test_hit_miss(uint64_t *hits, uint64_t *misses) {
    if (hits) *hits = g_sycl_stream_hits;
    if (misses) *misses = g_sycl_stream_misses;
}

extern "C" int ds4_sycl_stream_test_batch_dedup(
        const int32_t *ids, uint32_t n_tokens, uint32_t n_selected,
        uint32_t n_total_expert, int32_t *unique_ids_out, uint32_t *unique_count_out,
        uint32_t *compact_ids_out) {
    if (!ids || !unique_ids_out || !unique_count_out || !compact_ids_out) return 0;
    std::vector<int32_t>  unique_ids;
    std::vector<uint32_t> compact_ids;
    if (!sycl_stream_batch_dedup(ids, n_tokens, n_selected, n_total_expert, unique_ids,
                                compact_ids)) {
        return 0;
    }
    for (size_t i = 0; i < unique_ids.size(); i++) unique_ids_out[i] = unique_ids[i];
    *unique_count_out = (uint32_t)unique_ids.size();
    for (size_t i = 0; i < compact_ids.size(); i++) compact_ids_out[i] = compact_ids[i];
    return 1;
}

extern "C" void ds4_sycl_stream_test_reset(void) {
    if (!g_devices.empty()) {
        sycl::queue &q = ds4_sycl_current_queue();
        sycl_stream_resident_release_all(q);
        sycl_stream_selected_release(q);
    }
    g_sycl_stream_cache_budget = 0;
    g_sycl_stream_hits = 0;
    g_sycl_stream_misses = 0;
}
