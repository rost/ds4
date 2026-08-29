/* Tests for the SYCL expert-streaming caches: the resident cache's LRU
 * eviction, slab pool and dedicated-allocation fallback, seed_experts
 * priority selection, the selected-cache scratch buffer (begin_selected_load
 * / seed_selected) and the batch-selected cache's host-side deduplication
 * (prepare_selected_batch).  Self-contained (does not use
 * tests/test_sycl_harness.h, used by other tests in this suite).  Needs no model
 * file: the "model" is a plain host byte array standing in for an mmapped
 * weights file. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s\n", (msg));                           \
            return 1;                                                       \
        }                                                                   \
    } while (0)

/* Test-only side doors into sycl/ds4_sycl_streaming.hpp.  None of the
 * streaming-cache ABI entries expose a resident entry's bytes, the selected
 * scratch buffer's bytes, the resident-hit/miss counts, the batch dedup
 * mapping, or a way to reset cache state between test cases. */
extern int  ds4_sycl_stream_test_read_resident(uint32_t layer, int32_t expert,
                                               void *gate_out, void *up_out,
                                               void *down_out);
extern void ds4_sycl_stream_test_reset(void);
extern int  ds4_sycl_stream_test_read_selected(uint32_t slot, void *gate_out,
                                               void *up_out, void *down_out);
extern void ds4_sycl_stream_test_hit_miss(uint64_t *hits, uint64_t *misses);
extern int  ds4_sycl_stream_test_batch_dedup(const int32_t *ids, uint32_t n_tokens,
                                             uint32_t n_selected,
                                             uint32_t n_total_expert,
                                             int32_t *unique_ids_out,
                                             uint32_t *unique_count_out,
                                             uint32_t *compact_ids_out);

enum { GATE_TAG = 0, UP_TAG = 101, DOWN_TAG = 202 };

/* Deterministic, section-distinguishing fill: byte[i] = (layer*31 +
 * expert*7 + i) & 0xff for the gate section,
 * extended with a per-section tag for up/down so a copy that reads the
 * wrong source section (e.g. gate content into the up slot) shows up as a
 * content mismatch instead of silently matching, since gate and up share
 * the same local index range within their own sections. */
static unsigned char pattern_byte(uint32_t layer, int32_t expert, uint32_t i,
                                  unsigned tag) {
    return (unsigned char)((layer * 31u + (uint32_t)expert * 7u + i + tag) & 0xffu);
}

/* Lays out one layer's worth of [gate][up][down] sections back to back in
 * `model`, each n_total_expert * its per-expert byte count, and fills
 * every expert's slice with pattern_byte. */
static void fill_model(unsigned char *model, uint32_t layer,
                       uint32_t n_total_expert, uint64_t gate_expert_bytes,
                       uint64_t down_expert_bytes, uint64_t gate_offset,
                       uint64_t up_offset, uint64_t down_offset) {
    for (uint32_t e = 0; e < n_total_expert; e++) {
        for (uint64_t i = 0; i < gate_expert_bytes; i++) {
            model[gate_offset + e * gate_expert_bytes + i] =
                pattern_byte(layer, (int32_t)e, (uint32_t)i, GATE_TAG);
            model[up_offset + e * gate_expert_bytes + i] =
                pattern_byte(layer, (int32_t)e, (uint32_t)i, UP_TAG);
        }
        for (uint64_t i = 0; i < down_expert_bytes; i++) {
            model[down_offset + e * down_expert_bytes + i] =
                pattern_byte(layer, (int32_t)e, (uint32_t)i, DOWN_TAG);
        }
    }
}

static int expert_resident_matches(unsigned char *model, uint32_t layer,
                                   int32_t expert, uint64_t gate_offset,
                                   uint64_t up_offset, uint64_t down_offset,
                                   uint64_t gate_expert_bytes,
                                   uint64_t down_expert_bytes) {
    unsigned char *gate = (unsigned char *)malloc(gate_expert_bytes);
    unsigned char *up = (unsigned char *)malloc(gate_expert_bytes);
    unsigned char *down = (unsigned char *)malloc(down_expert_bytes);
    int ok = gate && up && down &&
             ds4_sycl_stream_test_read_resident(layer, expert, gate, up, down);
    if (ok) {
        ok = memcmp(gate,
                    model + gate_offset + (uint64_t)expert * gate_expert_bytes,
                    gate_expert_bytes) == 0 &&
             memcmp(up, model + up_offset + (uint64_t)expert * gate_expert_bytes,
                    gate_expert_bytes) == 0 &&
             memcmp(down,
                    model + down_offset + (uint64_t)expert * down_expert_bytes,
                    down_expert_bytes) == 0;
    }
    free(gate);
    free(up);
    free(down);
    return ok;
}

static int expert_absent(uint32_t layer, int32_t expert) {
    return !ds4_sycl_stream_test_read_resident(layer, expert, NULL, NULL, NULL);
}

/* Seeding fewer experts than the budget: every one must be readable back
 * byte-for-byte, and current_count must equal what was actually seeded. */
static int test_seed_fewer_than_budget(void) {
    ds4_sycl_stream_test_reset();
    enum { N_EXPERT = 8, GATE_BYTES = 64, DOWN_BYTES = 48, LAYER = 0 };
    unsigned char model[N_EXPERT * (2 * GATE_BYTES + DOWN_BYTES)];
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = N_EXPERT * GATE_BYTES;
    const uint64_t down_offset = 2u * N_EXPERT * GATE_BYTES;
    fill_model(model, LAYER, N_EXPERT, GATE_BYTES, DOWN_BYTES, gate_offset,
              up_offset, down_offset);

    ds4_gpu_set_streaming_expert_cache_budget(5);
    ds4_gpu_stream_expert_table table = {
        model, sizeof(model), LAYER, N_EXPERT, gate_offset, up_offset,
        down_offset, GATE_BYTES, DOWN_BYTES};
    int32_t ids[] = {1, 4};
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, ids, NULL, 2) != 0,
          "seed_fewer: seed_experts failed");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 2,
          "seed_fewer: current_count should equal experts seeded");
    CHECK(expert_resident_matches(model, LAYER, 1, gate_offset, up_offset,
                                  down_offset, GATE_BYTES, DOWN_BYTES),
          "seed_fewer: expert 1 content mismatch");
    CHECK(expert_resident_matches(model, LAYER, 4, gate_offset, up_offset,
                                  down_offset, GATE_BYTES, DOWN_BYTES),
          "seed_fewer: expert 4 content mismatch");
    fprintf(stderr, "  test_seed_fewer_than_budget OK\n");
    return 0;
}

/* Seeding more experts than the budget across several calls: the resident
 * count must never exceed the budget, and the least-recently-used expert
 * (from an earlier call, not protected by the current call's batch) must
 * be the one evicted. */
static int test_lru_eviction_across_calls(void) {
    ds4_sycl_stream_test_reset();
    enum { N_EXPERT = 8, GATE_BYTES = 32, DOWN_BYTES = 32, LAYER = 0, BUDGET = 3 };
    unsigned char model[N_EXPERT * (2 * GATE_BYTES + DOWN_BYTES)];
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = N_EXPERT * GATE_BYTES;
    const uint64_t down_offset = 2u * N_EXPERT * GATE_BYTES;
    fill_model(model, LAYER, N_EXPERT, GATE_BYTES, DOWN_BYTES, gate_offset,
              up_offset, down_offset);

    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    ds4_gpu_stream_expert_table table = {
        model, sizeof(model), LAYER, N_EXPERT, gate_offset, up_offset,
        down_offset, GATE_BYTES, DOWN_BYTES};

    int32_t first[] = {0, 1, 2};
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, first, NULL, 3) != 0,
          "lru: first seed failed");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == BUDGET,
          "lru: current_count should reach budget");

    /* Default priority is (n_experts - i): expert 2 (last in the input
     * list) has the LOWEST default priority, and processing runs in
     * ASCENDING priority order, so expert 2 is processed FIRST within
     * that batch and ends up with the smallest last_used.  It must be the
     * one evicted when a fourth, unrelated expert forces room; expert 0
     * (highest default priority, processed last, freshest last_used) must
     * survive the longest. */
    int32_t second[] = {3};
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, second, NULL, 1) != 0,
          "lru: second seed failed");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == BUDGET,
          "lru: current_count must never exceed budget");
    CHECK(expert_absent(LAYER, 2), "lru: expert 2 should have been evicted");
    CHECK(expert_resident_matches(model, LAYER, 0, gate_offset, up_offset,
                                  down_offset, GATE_BYTES, DOWN_BYTES),
          "lru: expert 0 should still be resident");
    CHECK(expert_resident_matches(model, LAYER, 1, gate_offset, up_offset,
                                  down_offset, GATE_BYTES, DOWN_BYTES),
          "lru: expert 1 should still be resident");
    CHECK(expert_resident_matches(model, LAYER, 3, gate_offset, up_offset,
                                  down_offset, GATE_BYTES, DOWN_BYTES),
          "lru: expert 3 should be resident");
    fprintf(stderr, "  test_lru_eviction_across_calls OK\n");
    return 0;
}

/* seed_experts must pick the top `budget` distinct experts by priority
 * (dropping the rest outright, not seeding then evicting them), and must
 * process the chosen set in ASCENDING priority order so the
 * highest-priority expert ends up with the freshest last_used and
 * therefore survives a later eviction that a lower-priority sibling from
 * the same batch does not. */
static int test_priority_ordering(void) {
    ds4_sycl_stream_test_reset();
    enum { N_EXPERT = 8, GATE_BYTES = 32, DOWN_BYTES = 16, LAYER = 0, BUDGET = 2 };
    unsigned char model[N_EXPERT * (2 * GATE_BYTES + DOWN_BYTES)];
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = N_EXPERT * GATE_BYTES;
    const uint64_t down_offset = 2u * N_EXPERT * GATE_BYTES;
    fill_model(model, LAYER, N_EXPERT, GATE_BYTES, DOWN_BYTES, gate_offset,
              up_offset, down_offset);

    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    ds4_gpu_stream_expert_table table = {
        model, sizeof(model), LAYER, N_EXPERT, gate_offset, up_offset,
        down_offset, GATE_BYTES, DOWN_BYTES};

    int32_t ids[] = {5, 6, 7};
    uint32_t priorities[] = {1, 100, 2};
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, ids, priorities, 3) != 0,
          "priority: seed_experts failed");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == BUDGET,
          "priority: current_count should equal budget");
    CHECK(expert_absent(LAYER, 5),
          "priority: lowest-priority expert 5 should never have been chosen");
    CHECK(expert_resident_matches(model, LAYER, 6, gate_offset, up_offset,
                                  down_offset, GATE_BYTES, DOWN_BYTES),
          "priority: expert 6 content mismatch");
    CHECK(expert_resident_matches(model, LAYER, 7, gate_offset, up_offset,
                                  down_offset, GATE_BYTES, DOWN_BYTES),
          "priority: expert 7 content mismatch");

    /* Force one eviction: expert 7 (priority 2, processed before expert 6
     * within the batch) has the older last_used and must go, leaving the
     * higher-priority expert 6 resident. */
    int32_t third[] = {2};
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, third, NULL, 1) != 0,
          "priority: third seed failed");
    CHECK(expert_absent(LAYER, 7),
          "priority: lower-priority sibling should be evicted first");
    CHECK(expert_resident_matches(model, LAYER, 6, gate_offset, up_offset,
                                  down_offset, GATE_BYTES, DOWN_BYTES),
          "priority: higher-priority sibling should survive");
    fprintf(stderr, "  test_priority_ordering OK\n");
    return 0;
}

/* A budget of 0 is an explicit no-op success: nothing gets cached. */
static int test_budget_zero_is_noop_success(void) {
    ds4_sycl_stream_test_reset();
    enum { N_EXPERT = 4, GATE_BYTES = 16, DOWN_BYTES = 16, LAYER = 0 };
    unsigned char model[N_EXPERT * (2 * GATE_BYTES + DOWN_BYTES)];
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = N_EXPERT * GATE_BYTES;
    const uint64_t down_offset = 2u * N_EXPERT * GATE_BYTES;
    fill_model(model, LAYER, N_EXPERT, GATE_BYTES, DOWN_BYTES, gate_offset,
              up_offset, down_offset);

    ds4_gpu_set_streaming_expert_cache_budget(0);
    CHECK(ds4_gpu_stream_expert_cache_configured_count() == 0,
          "budget_zero: configured_count should reflect budget 0");
    ds4_gpu_stream_expert_table table = {
        model, sizeof(model), LAYER, N_EXPERT, gate_offset, up_offset,
        down_offset, GATE_BYTES, DOWN_BYTES};
    int32_t ids[] = {0, 1};
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, ids, NULL, 2) != 0,
          "budget_zero: seed_experts with budget 0 must report success");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 0,
          "budget_zero: nothing should be cached");
    fprintf(stderr, "  test_budget_zero_is_noop_success OK\n");
    return 0;
}

/* This corrects an earlier expectation, verified directly
 * against source: rocm/ds4_rocm_current_api_compat.cuh's
 * ds4_gpu_stream_expert_cache_seed_experts checks "if (!table) return 0;"
 * before ever calling into the resident-cache algorithm, and ds4.c's
 * caller treats a 0 result as failure.  A null table is FAILURE, not the
 * "zero work, still succeeds" case (that's budget == 0, tested above). Do
 * not "fix" this back to expecting success. */
static int test_null_table_is_failure(void) {
    ds4_sycl_stream_test_reset();
    ds4_gpu_set_streaming_expert_cache_budget(4);
    int32_t ids[] = {0};
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(NULL, ids, NULL, 1) == 0,
          "null_table: a null table must return failure (0)");
    fprintf(stderr, "  test_null_table_is_failure OK\n");
    return 0;
}

/* Malformed, non-null-table inputs that must all fail outright: null
 * expert_ids, zero experts, zero total experts, more than the 384-expert
 * ceiling, and either per-expert byte count being zero. */
static int test_malformed_inputs_fail(void) {
    ds4_sycl_stream_test_reset();
    enum { N_EXPERT = 4, GATE_BYTES = 16, DOWN_BYTES = 16, LAYER = 0 };
    unsigned char model[N_EXPERT * (2 * GATE_BYTES + DOWN_BYTES)];
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = N_EXPERT * GATE_BYTES;
    const uint64_t down_offset = 2u * N_EXPERT * GATE_BYTES;
    fill_model(model, LAYER, N_EXPERT, GATE_BYTES, DOWN_BYTES, gate_offset,
              up_offset, down_offset);
    ds4_gpu_set_streaming_expert_cache_budget(4);

    ds4_gpu_stream_expert_table table = {
        model, sizeof(model), LAYER, N_EXPERT, gate_offset, up_offset,
        down_offset, GATE_BYTES, DOWN_BYTES};
    int32_t ids[] = {0};

    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, NULL, NULL, 1) == 0,
          "malformed: null expert_ids must fail");
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, ids, NULL, 0) == 0,
          "malformed: n_experts == 0 must fail");

    ds4_gpu_stream_expert_table zero_total = table;
    zero_total.n_total_expert = 0;
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&zero_total, ids, NULL, 1) == 0,
          "malformed: n_total_expert == 0 must fail");

    ds4_gpu_stream_expert_table too_many = table;
    too_many.n_total_expert = 385;
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&too_many, ids, NULL, 1) == 0,
          "malformed: n_total_expert > 384 must fail");

    ds4_gpu_stream_expert_table zero_gate = table;
    zero_gate.gate_expert_bytes = 0;
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&zero_gate, ids, NULL, 1) == 0,
          "malformed: gate_expert_bytes == 0 must fail");

    ds4_gpu_stream_expert_table zero_down = table;
    zero_down.down_expert_bytes = 0;
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&zero_down, ids, NULL, 1) == 0,
          "malformed: down_expert_bytes == 0 must fail");

    int32_t bad_id[] = {(int32_t)N_EXPERT};
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, bad_id, NULL, 1) == 0,
          "malformed: an out-of-range expert id must fail");

    ds4_gpu_stream_expert_table bad_range = table;
    bad_range.gate_offset = (uint64_t)sizeof(model) - 1;
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&bad_range, ids, NULL, 1) == 0,
          "malformed: an out-of-model gate range must fail");

    fprintf(stderr, "  test_malformed_inputs_fail OK\n");
    return 0;
}

/* An expert whose combined (2*gate + down) byte size differs from the
 * size class already fixed by an earlier admission must fall back to a
 * dedicated allocation rather than a pooled slot, and must still be
 * byte-correct. */
static int test_dedicated_allocation_fallback(void) {
    ds4_sycl_stream_test_reset();
    enum { LAYER = 0, BUDGET = 4 };
    enum { N_A = 4, GATE_A = 32, DOWN_A = 16 };
    enum { N_B = 2, GATE_B = 8, DOWN_B = 4 };
    unsigned char model_a[N_A * (2 * GATE_A + DOWN_A)];
    unsigned char model_b[N_B * (2 * GATE_B + DOWN_B)];
    const uint64_t gate_offset_a = 0;
    const uint64_t up_offset_a = N_A * GATE_A;
    const uint64_t down_offset_a = 2u * N_A * GATE_A;
    fill_model(model_a, LAYER, N_A, GATE_A, DOWN_A, gate_offset_a, up_offset_a,
              down_offset_a);
    const uint64_t gate_offset_b = 0;
    const uint64_t up_offset_b = N_B * GATE_B;
    const uint64_t down_offset_b = 2u * N_B * GATE_B;
    fill_model(model_b, LAYER, N_B, GATE_B, DOWN_B, gate_offset_b, up_offset_b,
              down_offset_b);

    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);

    /* Fix the slab size class at 2*GATE_A + DOWN_A. */
    ds4_gpu_stream_expert_table table_a = {
        model_a, sizeof(model_a), LAYER, N_A, gate_offset_a, up_offset_a,
        down_offset_a, GATE_A, DOWN_A};
    int32_t ids_a[] = {0};
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table_a, ids_a, NULL, 1) != 0,
          "dedicated: fixing the size class failed");

    /* model_b's per-expert byte count is a different size class, and its
     * layer/expert key collides with model_a's expert 0 unless the model
     * pointer participates in the key (it does: all eight key fields).
     * Use a distinct expert id to also sanity-check that two different
     * model maps at the "same" layer/expert never alias. */
    ds4_gpu_stream_expert_table table_b = {
        model_b, sizeof(model_b), LAYER, N_B, gate_offset_b, up_offset_b,
        down_offset_b, GATE_B, DOWN_B};
    int32_t ids_b[] = {1};
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table_b, ids_b, NULL, 1) != 0,
          "dedicated: mismatched-size seed failed");
    CHECK(expert_resident_matches(model_b, LAYER, 1, gate_offset_b, up_offset_b,
                                  down_offset_b, GATE_B, DOWN_B),
          "dedicated: mismatched-size expert content wrong");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 2,
          "dedicated: both size classes should be resident");
    fprintf(stderr, "  test_dedicated_allocation_fallback OK\n");
    return 0;
}

/* Re-seeding an expert that is already resident must bump its recency
 * without growing the cache. */
static int test_reseed_resident_does_not_grow(void) {
    ds4_sycl_stream_test_reset();
    enum { N_EXPERT = 4, GATE_BYTES = 16, DOWN_BYTES = 16, LAYER = 0, BUDGET = 3 };
    unsigned char model[N_EXPERT * (2 * GATE_BYTES + DOWN_BYTES)];
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = N_EXPERT * GATE_BYTES;
    const uint64_t down_offset = 2u * N_EXPERT * GATE_BYTES;
    fill_model(model, LAYER, N_EXPERT, GATE_BYTES, DOWN_BYTES, gate_offset,
              up_offset, down_offset);

    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    ds4_gpu_stream_expert_table table = {
        model, sizeof(model), LAYER, N_EXPERT, gate_offset, up_offset,
        down_offset, GATE_BYTES, DOWN_BYTES};

    int32_t ids[] = {2};
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, ids, NULL, 1) != 0,
          "reseed: first seed failed");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 1,
          "reseed: unexpected count after first seed");
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, ids, NULL, 1) != 0,
          "reseed: second seed failed");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 1,
          "reseed: current_count must not grow on a resident hit");
    CHECK(expert_resident_matches(model, LAYER, 2, gate_offset, up_offset,
                                  down_offset, GATE_BYTES, DOWN_BYTES),
          "reseed: content mismatch after re-seed");
    fprintf(stderr, "  test_reseed_resident_does_not_grow OK\n");
    return 0;
}

/* budget_for_expert_size ignores both byte arguments and delegates to
 * configured_count, matching rocm/ds4_rocm_current_api_compat.cuh:170. */
static int test_budget_for_expert_size_ignores_bytes(void) {
    ds4_sycl_stream_test_reset();
    ds4_gpu_set_streaming_expert_cache_budget(7);
    CHECK(ds4_gpu_stream_expert_cache_budget_for_expert_size(1, 1) == 7,
          "budget_for_size: should equal configured_count regardless of bytes");
    CHECK(ds4_gpu_stream_expert_cache_budget_for_expert_size(1u << 20, 1u << 20) == 7,
          "budget_for_size: byte arguments must not change the result");
    fprintf(stderr, "  test_budget_for_expert_size_ignores_bytes OK\n");
    return 0;
}

/* ---- Selected and batch-selected scratch caches ---- */

static int selected_slot_matches(unsigned char *model, int32_t expert,
                                 uint32_t slot, uint64_t gate_offset,
                                 uint64_t up_offset, uint64_t down_offset,
                                 uint64_t gate_expert_bytes,
                                 uint64_t down_expert_bytes) {
    unsigned char *gate = (unsigned char *)malloc(gate_expert_bytes);
    unsigned char *up = (unsigned char *)malloc(gate_expert_bytes);
    unsigned char *down = (unsigned char *)malloc(down_expert_bytes);
    int ok = gate && up && down &&
             ds4_sycl_stream_test_read_selected(slot, gate, up, down);
    if (ok) {
        ok = memcmp(gate, model + gate_offset + (uint64_t)expert * gate_expert_bytes,
                    gate_expert_bytes) == 0 &&
             memcmp(up, model + up_offset + (uint64_t)expert * gate_expert_bytes,
                    gate_expert_bytes) == 0 &&
             memcmp(down, model + down_offset + (uint64_t)expert * down_expert_bytes,
                    down_expert_bytes) == 0;
    }
    free(gate);
    free(up);
    free(down);
    return ok;
}

/* begin_selected_load must copy every selected expert's bytes into the
 * compacted per-slot scratch buffer, byte-for-byte, in selection order
 * (not sorted, not deduplicated: the selected cache always holds exactly
 * n_selected slots, one per input id, even if some ids repeat). */
static int test_begin_selected_load_byte_content(void) {
    ds4_sycl_stream_test_reset();
    enum { N_EXPERT = 8, GATE_BYTES = 24, DOWN_BYTES = 16, LAYER = 0, BUDGET = 8 };
    unsigned char model[N_EXPERT * (2 * GATE_BYTES + DOWN_BYTES)];
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = N_EXPERT * GATE_BYTES;
    const uint64_t down_offset = 2u * N_EXPERT * GATE_BYTES;
    fill_model(model, LAYER, N_EXPERT, GATE_BYTES, DOWN_BYTES, gate_offset,
              up_offset, down_offset);

    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    ds4_gpu_stream_expert_table table = {
        model, sizeof(model), LAYER, N_EXPERT, gate_offset, up_offset,
        down_offset, GATE_BYTES, DOWN_BYTES};
    int32_t selected[] = {3, 1, 5};
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, selected, 3) != 0,
          "begin_selected_load: call failed");
    CHECK(selected_slot_matches(model, 3, 0, gate_offset, up_offset, down_offset,
                                GATE_BYTES, DOWN_BYTES),
          "begin_selected_load: slot 0 (expert 3) content mismatch");
    CHECK(selected_slot_matches(model, 1, 1, gate_offset, up_offset, down_offset,
                                GATE_BYTES, DOWN_BYTES),
          "begin_selected_load: slot 1 (expert 1) content mismatch");
    CHECK(selected_slot_matches(model, 5, 2, gate_offset, up_offset, down_offset,
                                GATE_BYTES, DOWN_BYTES),
          "begin_selected_load: slot 2 (expert 5) content mismatch");
    fprintf(stderr, "  test_begin_selected_load_byte_content OK\n");
    return 0;
}

/* A selected id already resident in the resident cache must not trigger a
 * new mmap-to-device admission copy: only the genuinely new id should
 * count as a miss.  Proven with an internal counter, not by timing. */
static int test_resident_hit_short_circuit(void) {
    ds4_sycl_stream_test_reset();
    enum { N_EXPERT = 8, GATE_BYTES = 16, DOWN_BYTES = 16, LAYER = 0, BUDGET = 8 };
    unsigned char model[N_EXPERT * (2 * GATE_BYTES + DOWN_BYTES)];
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = N_EXPERT * GATE_BYTES;
    const uint64_t down_offset = 2u * N_EXPERT * GATE_BYTES;
    fill_model(model, LAYER, N_EXPERT, GATE_BYTES, DOWN_BYTES, gate_offset,
              up_offset, down_offset);

    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    ds4_gpu_stream_expert_table table = {
        model, sizeof(model), LAYER, N_EXPERT, gate_offset, up_offset,
        down_offset, GATE_BYTES, DOWN_BYTES};

    /* Pre-seed via seed_experts, which does not touch these
     * counters: they are scoped to this entry's own admission loops
     * (begin_selected_load / prepare_selected_batch) so a hit/miss here
     * unambiguously reflects THIS entry's resident-cache-hit short
     * circuit, not seed_experts'. */
    int32_t pre[] = {2};
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, pre, NULL, 1) != 0,
          "resident_hit: pre-seed failed");

    uint64_t hits = 0, misses = 0;
    ds4_sycl_stream_test_hit_miss(&hits, &misses);
    CHECK(hits == 0 && misses == 0,
          "resident_hit: counters must still be zero before begin_selected_load runs");

    int32_t selected[] = {2, 6};
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, selected, 2) != 0,
          "resident_hit: begin_selected_load failed");
    ds4_sycl_stream_test_hit_miss(&hits, &misses);
    CHECK(hits == 1, "resident_hit: already-resident expert 2 should register one hit");
    CHECK(misses == 1, "resident_hit: new expert 6 should register exactly one new miss");
    CHECK(selected_slot_matches(model, 2, 0, gate_offset, up_offset, down_offset,
                                GATE_BYTES, DOWN_BYTES),
          "resident_hit: slot 0 (expert 2, resident hit) content mismatch");
    CHECK(selected_slot_matches(model, 6, 1, gate_offset, up_offset, down_offset,
                                GATE_BYTES, DOWN_BYTES),
          "resident_hit: slot 1 (expert 6, fresh admission) content mismatch");
    fprintf(stderr, "  test_resident_hit_short_circuit OK\n");
    return 0;
}

/* Independent CPU reimplementation of "first unique slot per expert,
 * assigned in first-seen order", used as the oracle prepare_selected_batch
 * must match. */
static void cpu_batch_dedup_oracle(const int32_t *ids, uint32_t n_tokens,
                                   uint32_t n_selected, int32_t *unique_ids_out,
                                   uint32_t *unique_count_out,
                                   uint32_t *compact_ids_out) {
    int32_t  slot_of[512];
    uint32_t unique_count = 0;
    for (uint32_t i = 0; i < 512; i++) slot_of[i] = -1;
    for (uint32_t i = 0; i < n_tokens * n_selected; i++) {
        const int32_t expert = ids[i];
        if (slot_of[expert] < 0) {
            slot_of[expert] = (int32_t)unique_count;
            unique_ids_out[unique_count] = expert;
            unique_count++;
        }
        compact_ids_out[i] = (uint32_t)slot_of[expert];
    }
    *unique_count_out = unique_count;
}

/* Dedup correctness: heavy overlap between tokens, one token whose whole
 * selection set duplicates an earlier token's exactly, and one token
 * whose n_selected slots are all the same expert repeated.  Compares the
 * implementation's dedup mapping against an independent CPU oracle,
 * rather than only against hand-computed constants. */
static int test_prepare_selected_batch_dedup(void) {
    ds4_sycl_stream_test_reset();
    enum { N_EXPERT = 8, GATE_BYTES = 8, DOWN_BYTES = 8, LAYER = 0, BUDGET = 8,
           N_TOKENS = 4, N_SELECTED = 3 };
    unsigned char model[N_EXPERT * (2 * GATE_BYTES + DOWN_BYTES)];
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = N_EXPERT * GATE_BYTES;
    const uint64_t down_offset = 2u * N_EXPERT * GATE_BYTES;
    fill_model(model, LAYER, N_EXPERT, GATE_BYTES, DOWN_BYTES, gate_offset,
              up_offset, down_offset);
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    ds4_gpu_stream_expert_table table = {
        model, sizeof(model), LAYER, N_EXPERT, gate_offset, up_offset,
        down_offset, GATE_BYTES, DOWN_BYTES};

    /* token0: {1,2,3} (3 new uniques). token1: {2,3,4} (1 new: 4).
     * token2: {1,1,1} (all duplicates of token0's expert 1, no new
     * uniques). token3: {1,2,3} (whole set duplicates token0's exactly,
     * no new uniques). */
    int32_t ids[N_TOKENS * N_SELECTED] = {
        1, 2, 3,
        2, 3, 4,
        1, 1, 1,
        1, 2, 3};

    int32_t  oracle_unique[N_EXPERT];
    uint32_t oracle_count = 0;
    uint32_t oracle_compact[N_TOKENS * N_SELECTED];
    cpu_batch_dedup_oracle(ids, N_TOKENS, N_SELECTED, oracle_unique, &oracle_count,
                          oracle_compact);
    CHECK(oracle_count == 4, "batch_dedup: oracle itself miscomputed (test bug)");

    int32_t  got_unique[N_EXPERT];
    uint32_t got_count = 0;
    uint32_t got_compact[N_TOKENS * N_SELECTED];
    CHECK(ds4_sycl_stream_test_batch_dedup(ids, N_TOKENS, N_SELECTED, N_EXPERT,
                                          got_unique, &got_count, got_compact) != 0,
          "batch_dedup: dedup hook failed");
    CHECK(got_count == oracle_count, "batch_dedup: unique_count mismatch vs oracle");
    for (uint32_t i = 0; i < oracle_count; i++) {
        CHECK(got_unique[i] == oracle_unique[i], "batch_dedup: unique_ids mismatch vs oracle");
    }
    for (uint32_t i = 0; i < N_TOKENS * N_SELECTED; i++) {
        CHECK(got_compact[i] == oracle_compact[i], "batch_dedup: compact_ids mismatch vs oracle");
    }

    CHECK(ds4_gpu_stream_expert_cache_prepare_selected_batch(&table, ids, N_TOKENS,
                                                             N_SELECTED) != 0,
          "batch_dedup: prepare_selected_batch failed");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == oracle_count,
          "batch_dedup: resident cache should hold exactly the unique experts");
    CHECK(expert_resident_matches(model, LAYER, 4, gate_offset, up_offset, down_offset,
                                  GATE_BYTES, DOWN_BYTES),
          "batch_dedup: expert 4 content mismatch");
    fprintf(stderr, "  test_prepare_selected_batch_dedup OK\n");
    return 0;
}

/* A batch touching exactly DS4_SYCL_STREAM_MAX_N_EXPERT (384) distinct
 * experts, the ceiling n_total_expert itself enforces, must still resolve
 * correctly with no off-by-one in the unique-slot bookkeeping. */
static int test_batch_near_ceiling(void) {
    ds4_sycl_stream_test_reset();
    enum { N_EXPERT = 384, GATE_BYTES = 4, DOWN_BYTES = 4, LAYER = 0, BUDGET = 400,
           N_SELECTED = 8, N_TOKENS = N_EXPERT / N_SELECTED };
    static unsigned char model[N_EXPERT * (2 * GATE_BYTES + DOWN_BYTES)];
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = N_EXPERT * GATE_BYTES;
    const uint64_t down_offset = 2u * N_EXPERT * GATE_BYTES;
    fill_model(model, LAYER, N_EXPERT, GATE_BYTES, DOWN_BYTES, gate_offset,
              up_offset, down_offset);
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    ds4_gpu_stream_expert_table table = {
        model, sizeof(model), LAYER, N_EXPERT, gate_offset, up_offset,
        down_offset, GATE_BYTES, DOWN_BYTES};

    static int32_t ids[N_TOKENS * N_SELECTED];
    for (uint32_t i = 0; i < N_TOKENS * N_SELECTED; i++) ids[i] = (int32_t)i;

    CHECK(ds4_gpu_stream_expert_cache_prepare_selected_batch(&table, ids, N_TOKENS,
                                                             N_SELECTED) != 0,
          "batch_ceiling: prepare_selected_batch failed at 384 unique experts");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == N_EXPERT,
          "batch_ceiling: all 384 distinct experts should be resident");
    CHECK(expert_resident_matches(model, LAYER, 383, gate_offset, up_offset, down_offset,
                                  GATE_BYTES, DOWN_BYTES),
          "batch_ceiling: last expert (383) content mismatch");
    fprintf(stderr, "  test_batch_near_ceiling OK\n");
    return 0;
}

/* Malformed inputs to the three batch-selected entries must fail (0), matching
 * the same null-table/malformed-argument polarity verified above:
 * rocm/ds4_rocm_current_api_compat.cuh's begin_selected_load, seed_selected
 * and prepare_selected_batch wrappers all read "if (!table) return 0;". */
static int test_task2_malformed_inputs_fail(void) {
    ds4_sycl_stream_test_reset();
    enum { N_EXPERT = 8, GATE_BYTES = 8, DOWN_BYTES = 8, LAYER = 0, BUDGET = 8 };
    unsigned char model[N_EXPERT * (2 * GATE_BYTES + DOWN_BYTES)];
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = N_EXPERT * GATE_BYTES;
    const uint64_t down_offset = 2u * N_EXPERT * GATE_BYTES;
    fill_model(model, LAYER, N_EXPERT, GATE_BYTES, DOWN_BYTES, gate_offset,
              up_offset, down_offset);
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    ds4_gpu_stream_expert_table table = {
        model, sizeof(model), LAYER, N_EXPERT, gate_offset, up_offset,
        down_offset, GATE_BYTES, DOWN_BYTES};
    int32_t selected[] = {0, 1};
    int32_t batch_ids[] = {0, 1, 2, 3};

    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(NULL, selected, 2) == 0,
          "task2_malformed: null table (begin_selected_load) should fail");
    CHECK(ds4_gpu_stream_expert_cache_seed_selected(NULL, selected, 2) == 0,
          "task2_malformed: null table (seed_selected) should fail");
    CHECK(ds4_gpu_stream_expert_cache_prepare_selected_batch(NULL, batch_ids, 2, 2) == 0,
          "task2_malformed: null table (prepare_selected_batch) should fail");

    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, NULL, 2) == 0,
          "task2_malformed: null selected_ids should fail");
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, selected, 0) == 0,
          "task2_malformed: n_selected == 0 should fail");
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, selected, 9) == 0,
          "task2_malformed: n_selected > 8 should fail");
    int32_t out_of_range[] = {0, (int32_t)N_EXPERT};
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, out_of_range, 2) == 0,
          "task2_malformed: out-of-range selected id should fail");

    CHECK(ds4_gpu_stream_expert_cache_prepare_selected_batch(&table, batch_ids, 1, 2) == 0,
          "task2_malformed: n_tokens <= 1 should fail");
    int32_t bad_batch_ids[] = {0, (int32_t)N_EXPERT, 1, 2};
    CHECK(ds4_gpu_stream_expert_cache_prepare_selected_batch(&table, bad_batch_ids, 2, 2) == 0,
          "task2_malformed: out-of-range batch expert id should fail");

    fprintf(stderr, "  test_task2_malformed_inputs_fail OK\n");
    return 0;
}

/* ---- Lifecycle, teardown, trivial entries ---- */

/* set_ssd_streaming(false) must release the whole resident cache, and
 * set_ssd_streaming(true) again on an already-empty cache must be a
 * harmless no-op (ROCm's ds4_gpu_set_ssd_streaming releases
 * unconditionally on every call, not only when disabling; see
 * rocm/ds4_rocm_current_api_compat.cuh:119-126). */
static int test_set_ssd_streaming_false_releases_resident(void) {
    ds4_sycl_stream_test_reset();
    enum { N_EXPERT = 4, GATE_BYTES = 16, DOWN_BYTES = 16, LAYER = 0, BUDGET = 4 };
    unsigned char model[N_EXPERT * (2 * GATE_BYTES + DOWN_BYTES)];
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = N_EXPERT * GATE_BYTES;
    const uint64_t down_offset = 2u * N_EXPERT * GATE_BYTES;
    fill_model(model, LAYER, N_EXPERT, GATE_BYTES, DOWN_BYTES, gate_offset,
              up_offset, down_offset);
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    ds4_gpu_stream_expert_table table = {
        model, sizeof(model), LAYER, N_EXPERT, gate_offset, up_offset,
        down_offset, GATE_BYTES, DOWN_BYTES};

    ds4_gpu_set_ssd_streaming(true);
    int32_t ids[] = {0, 1};
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, ids, NULL, 2) != 0,
          "ssd_streaming: seed failed");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 2,
          "ssd_streaming: seed should have populated the cache");

    ds4_gpu_set_ssd_streaming(false);
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 0,
          "ssd_streaming: disabling streaming must release the resident cache");

    /* Re-enabling on an already-empty cache is a harmless no-op: no crash,
     * current_count stays 0, and a subsequent seed still works. */
    ds4_gpu_set_ssd_streaming(true);
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 0,
          "ssd_streaming: re-enabling an empty cache should not populate it");
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, ids, NULL, 2) != 0,
          "ssd_streaming: seed after re-enable failed");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 2,
          "ssd_streaming: full seed/teardown/re-seed cycle left the cache inconsistent");
    fprintf(stderr, "  test_set_ssd_streaming_false_releases_resident OK\n");
    return 0;
}

/* recommended_working_set_size must be a plausible nonzero value (this
 * A770 has 16 GiB) while a device is initialised, and exactly 0 (the
 * "unavailable" value ds4.c's auto-cache configuration checks for,
 * ds4.c:55374-55379) once the device is torn down.  Re-initialises
 * afterward so later tests in this binary are unaffected. */
static int test_recommended_working_set_size(void) {
    const uint64_t with_device = ds4_gpu_recommended_working_set_size();
    CHECK(with_device > (1ull << 30),
          "working_set_size: expected a plausible (>1 GiB) value with a device present");

    ds4_gpu_cleanup();
    CHECK(ds4_gpu_recommended_working_set_size() == 0,
          "working_set_size: expected 0 (unavailable) with no device initialised");
    CHECK(ds4_gpu_init() != 0, "working_set_size: re-init after teardown failed");
    fprintf(stderr, "  test_recommended_working_set_size OK\n");
    return 0;
}

/* The three trivial no-op entries must be callable and change nothing
 * observable: none of them has an ABI-visible state to read back, so this
 * only proves they do not crash and do not disturb unrelated cache state. */
static int test_trivial_noop_entries_callable(void) {
    ds4_sycl_stream_test_reset();
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_streaming_expert_cache_budget(3);

    ds4_gpu_set_glm_streaming_prefill_full_layer(true);
    ds4_gpu_set_glm_streaming_prefill_full_layer(false);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(12345);
    ds4_gpu_stream_expert_cache_reset_route_hotness();

    CHECK(ds4_gpu_stream_expert_cache_configured_count() == 3,
          "trivial_noop: the three no-ops must not disturb the configured budget");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 0,
          "trivial_noop: the three no-ops must not disturb the resident count");
    fprintf(stderr, "  test_trivial_noop_entries_callable OK\n");
    return 0;
}

/* Streaming mode off must be a no-op SUCCESS for seed_experts and
 * begin_selected_load (rocm/ds4_rocm_runtime.cuh's
 * cuda_stream_resident_seed_experts and cuda_stream_selected_load both
 * check "if (!g_ssd_streaming_mode) return 1;" before any other
 * validation), matching every other entry's zero-work-succeeds convention. */
static int test_mode_off_seed_and_selected_are_noop_success(void) {
    ds4_sycl_stream_test_reset();
    enum { N_EXPERT = 4, GATE_BYTES = 8, DOWN_BYTES = 8, LAYER = 0, BUDGET = 4 };
    unsigned char model[N_EXPERT * (2 * GATE_BYTES + DOWN_BYTES)];
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = N_EXPERT * GATE_BYTES;
    const uint64_t down_offset = 2u * N_EXPERT * GATE_BYTES;
    fill_model(model, LAYER, N_EXPERT, GATE_BYTES, DOWN_BYTES, gate_offset,
              up_offset, down_offset);
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    ds4_gpu_stream_expert_table table = {
        model, sizeof(model), LAYER, N_EXPERT, gate_offset, up_offset,
        down_offset, GATE_BYTES, DOWN_BYTES};

    ds4_gpu_set_ssd_streaming(false);
    int32_t ids[] = {0, 1};
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, ids, NULL, 2) != 0,
          "mode_off: seed_experts must succeed as a no-op while streaming is disabled");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 0,
          "mode_off: seed_experts must not admit anything while streaming is disabled");
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, ids, 2) != 0,
          "mode_off: begin_selected_load must succeed as a no-op while streaming is disabled");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 0,
          "mode_off: begin_selected_load must not admit anything while streaming is disabled");
    fprintf(stderr, "  test_mode_off_seed_and_selected_are_noop_success OK\n");
    return 0;
}

/* prepare_selected_batch has the OPPOSITE polarity: ROCm folds
 * "!g_ssd_streaming_mode" into the same condition as every malformed-input
 * check in cuda_stream_batch_selected_prepare_from_host, all of which
 * return 0.  Streaming disabled is therefore a FAILURE for this one
 * entry, not a no-op success; this is a real, verified asymmetry, not an
 * inconsistency to "fix" toward the other two entries. */
static int test_mode_off_prepare_selected_batch_fails(void) {
    ds4_sycl_stream_test_reset();
    enum { N_EXPERT = 4, GATE_BYTES = 8, DOWN_BYTES = 8, LAYER = 0, BUDGET = 4 };
    unsigned char model[N_EXPERT * (2 * GATE_BYTES + DOWN_BYTES)];
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = N_EXPERT * GATE_BYTES;
    const uint64_t down_offset = 2u * N_EXPERT * GATE_BYTES;
    fill_model(model, LAYER, N_EXPERT, GATE_BYTES, DOWN_BYTES, gate_offset,
              up_offset, down_offset);
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    ds4_gpu_stream_expert_table table = {
        model, sizeof(model), LAYER, N_EXPERT, gate_offset, up_offset,
        down_offset, GATE_BYTES, DOWN_BYTES};

    ds4_gpu_set_ssd_streaming(false);
    int32_t batch_ids[] = {0, 1, 2, 3};
    CHECK(ds4_gpu_stream_expert_cache_prepare_selected_batch(&table, batch_ids, 2, 2) == 0,
          "mode_off: prepare_selected_batch must fail (not no-op succeed) while disabled");
    fprintf(stderr, "  test_mode_off_prepare_selected_batch_fails OK\n");
    return 0;
}

/* ---- Per-tier cache state ----------------------------------------------
 *
 * This machine has one physical GPU, so these use ds4_gpu_init_multi with
 * BOTH device_indices pointing at index 0: ds4_gpu_init_multi validates
 * only that each requested index is < the enumerated device count, not
 * that requested indices are distinct, and ds4_sycl_build_devices builds
 * one real sycl::queue per requested entry regardless of duplicates. The
 * result is two logical tiers, each with its own queue sharing one
 * context, backed by the same physical card -- exactly the technique
 * tests/test_sycl_mgpu.c's test_copy_xdev_same_device already relies on
 * to exercise multi-tier plumbing on single-GPU hardware. This proves the
 * per-tier array's own indexing and isolation for real (a genuine host-side
 * data-structure property); it cannot and does not claim anything about
 * real cross-die memory behaviour, which needs a second physical device
 * (the B60 fleet). */

static int start_two_logical_tiers(void) {
    ds4_gpu_cleanup();
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 0;
    if (ds4_gpu_init_multi(&cfg) == 0) return 0;
    ds4_gpu_set_ssd_streaming(true);
    return 1;
}

static int test_two_tiers_resident_cache_isolated(void) {
    CHECK(start_two_logical_tiers(), "init_multi for two logical tiers");

    enum { N_EXPERT = 4, GATE_BYTES = 8, DOWN_BYTES = 8, LAYER = 0, BUDGET = 4 };
    unsigned char model[N_EXPERT * (2 * GATE_BYTES + DOWN_BYTES)];
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = N_EXPERT * GATE_BYTES;
    const uint64_t down_offset = 2u * N_EXPERT * GATE_BYTES;
    fill_model(model, LAYER, N_EXPERT, GATE_BYTES, DOWN_BYTES, gate_offset,
              up_offset, down_offset);
    ds4_gpu_stream_expert_table table = {
        model, sizeof(model), LAYER, N_EXPERT, gate_offset, up_offset,
        down_offset, GATE_BYTES, DOWN_BYTES};

    CHECK(ds4_gpu_set_current_device(0) == 0, "select tier 0");
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    int32_t ids0[] = {0, 1};
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, ids0, NULL, 2) != 0,
          "seed on tier 0");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 2,
          "tier 0 resident count after its own seed");
    CHECK(expert_resident_matches(model, LAYER, 0, gate_offset, up_offset, down_offset,
                                  GATE_BYTES, DOWN_BYTES),
          "tier 0 resident expert 0 content");

    CHECK(ds4_gpu_set_current_device(1) == 0, "select tier 1");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 0,
          "tier 1 must start with an empty resident cache, not tier 0's");
    CHECK(!ds4_sycl_stream_test_read_resident(LAYER, 0, NULL, NULL, NULL),
          "tier 1 must not see tier 0's resident expert 0");
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    int32_t ids1[] = {2, 3};
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, ids1, NULL, 2) != 0,
          "seed on tier 1");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 2,
          "tier 1 resident count after its own seed");

    CHECK(ds4_gpu_set_current_device(0) == 0, "back to tier 0");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 2,
          "tier 0's own cache must be undisturbed by tier 1's seed");
    CHECK(!ds4_sycl_stream_test_read_resident(LAYER, 2, NULL, NULL, NULL),
          "tier 0 must not see tier 1's resident expert 2");
    CHECK(expert_resident_matches(model, LAYER, 0, gate_offset, up_offset, down_offset,
                                  GATE_BYTES, DOWN_BYTES),
          "tier 0 resident expert 0 still intact after tier 1 activity");

    ds4_gpu_cleanup();
    fprintf(stderr, "  test_two_tiers_resident_cache_isolated OK\n");
    return 0;
}

static int test_teardown_frees_every_tier(void) {
    CHECK(start_two_logical_tiers(), "init_multi for teardown test");

    enum { N_EXPERT = 4, GATE_BYTES = 8, DOWN_BYTES = 8, LAYER = 0, BUDGET = 4 };
    unsigned char model[N_EXPERT * (2 * GATE_BYTES + DOWN_BYTES)];
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = N_EXPERT * GATE_BYTES;
    const uint64_t down_offset = 2u * N_EXPERT * GATE_BYTES;
    fill_model(model, LAYER, N_EXPERT, GATE_BYTES, DOWN_BYTES, gate_offset,
              up_offset, down_offset);
    ds4_gpu_stream_expert_table table = {
        model, sizeof(model), LAYER, N_EXPERT, gate_offset, up_offset,
        down_offset, GATE_BYTES, DOWN_BYTES};
    int32_t ids[] = {0, 1};

    CHECK(ds4_gpu_set_current_device(0) == 0, "select tier 0");
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, ids, NULL, 2) != 0,
          "seed tier 0");

    CHECK(ds4_gpu_set_current_device(1) == 0, "select tier 1");
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, ids, NULL, 2) != 0,
          "seed tier 1");

    /* Toggling streaming mode must release BOTH tiers' resident caches,
     * not just whichever tier happened to be current at the call site
     * (the bug an earlier single-queue teardown had once a second tier
     * could exist). Still positioned on tier 1 here, so check it first. */
    ds4_gpu_set_ssd_streaming(false);
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 0,
          "tier 1 must be emptied by set_ssd_streaming(false)");
    CHECK(ds4_gpu_set_current_device(0) == 0, "back to tier 0");
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 0,
          "tier 0 must be emptied by set_ssd_streaming(false), not just tier 1");

    ds4_gpu_cleanup();
    fprintf(stderr, "  test_teardown_frees_every_tier OK\n");
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_seed_fewer_than_budget() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_lru_eviction_across_calls() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_priority_ordering() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_budget_zero_is_noop_success() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_null_table_is_failure() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_malformed_inputs_fail() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dedicated_allocation_fallback() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_reseed_resident_does_not_grow() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_budget_for_expert_size_ignores_bytes() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_begin_selected_load_byte_content() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_resident_hit_short_circuit() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_prepare_selected_batch_dedup() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_batch_near_ceiling() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_task2_malformed_inputs_fail() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_set_ssd_streaming_false_releases_resident() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_recommended_working_set_size() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_trivial_noop_entries_callable() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_mode_off_seed_and_selected_are_noop_success() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_mode_off_prepare_selected_batch_fails() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_two_tiers_resident_cache_isolated() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_teardown_frees_every_tier() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_streaming OK\n");
    return 0;
}
