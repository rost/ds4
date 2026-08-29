/* Correctness tests for SYCL compute kernels, validated against scalar
 * oracles implemented here.  The ds4.c CPU references are static and
 * cannot be linked, so each oracle reimplements the documented formula
 * with the ds4.c line number cited.  Needs no model file.
 *
 * This file covers the embedding-lookup kernels: single-token and
 * batched-token lookup over the F16 and Q8_0 embedding table formats. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include "test_sycl_harness.h"

#include <stdio.h>
#include <string.h>

/* Oracle for the F16 path: embed_token_f16 (ds4.c:6695-6711) composed with
 * hc_from_plain_embedding (ds4.c:9890-9894).  The GPU fuses the two into
 * one launch, so all n_hc output rows must be identical copies of the one
 * looked-up embedding row. */
static int test_embed_f16(void) {
    enum { N_VOCAB = 8, N_EMBD = 64, N_HC = 4, TOKEN = 5 };
    unsigned short table[N_VOCAB * N_EMBD];
    float          want[N_EMBD];
    float          got[N_HC * N_EMBD];

    /* Build an F16 table by round-tripping through the SYCL half type is
     * not available in a C test, so store bit patterns for exactly
     * representable small integers: sign 0, exponent 15 (bias), mantissa 0
     * gives 1.0 (0x3C00); successive integers follow the standard layout.
     * Using exactly representable values keeps the oracle exact. */
    static const unsigned short kHalfOne[8] = {
        0x0000, /* 0.0 */ 0x3C00, /* 1.0 */ 0x4000, /* 2.0 */ 0x4200, /* 3.0 */
        0x4400, /* 4.0 */ 0x4500, /* 5.0 */ 0x4600, /* 6.0 */ 0x4700  /* 7.0 */
    };
    static const float kHalfOneF[8] = {0,1,2,3,4,5,6,7};

    for (int t = 0; t < N_VOCAB; t++) {
        for (int e = 0; e < N_EMBD; e++) {
            table[t * N_EMBD + e] = kHalfOne[(t + e) % 8];
        }
    }
    for (int e = 0; e < N_EMBD; e++) want[e] = kHalfOneF[(TOKEN + e) % 8];

    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(to != NULL, "embed_f16: allocation failed");

    CHECK(ds4_gpu_embed_token_hc_tensor(to, table, sizeof(table), 0,
                                        N_VOCAB, TOKEN, N_EMBD, N_HC) != 0,
          "embed_f16: call");
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0, "embed_f16: read");

    for (int h = 0; h < N_HC; h++) {
        for (int e = 0; e < N_EMBD; e++) {
            CHECK_CLOSE(got[h * N_EMBD + e], want[e], 1e-3,
                        "embed_f16: value mismatch");
        }
    }
    /* Every HC row must be an identical copy: this is what proves the
     * fused broadcast, and a per-row lookup bug would pass the check
     * above while failing this one. */
    for (int h = 1; h < N_HC; h++) {
        CHECK(memcmp(&got[0], &got[h * N_EMBD], N_EMBD * sizeof(float)) == 0,
              "embed_f16: hc rows are not identical copies");
    }

    /* A weight_offset past the end of the model buffer must be rejected. */
    CHECK(ds4_gpu_embed_token_hc_tensor(to, table, sizeof(table),
                                        sizeof(table), N_VOCAB, TOKEN,
                                        N_EMBD, N_HC) == 0,
          "embed_f16: out-of-range weight offset must be rejected");

    /* token >= n_vocab is rejected at the single-token entry, matching
     * rocm/ds4_rocm_embedding_launch.cuh:2 `token >= n_vocab ... return 0`.
     * Note this differs from the BATCHED entry, where out-of-range ids in
     * the token tensor are silently clamped to 0 rather than rejected. */
    CHECK(ds4_gpu_embed_token_hc_tensor(to, table, sizeof(table), 0,
                                        N_VOCAB, N_VOCAB, N_EMBD, N_HC) == 0,
          "embed_f16: token >= n_vocab must be rejected");

    ds4_gpu_tensor_free(to);
    fprintf(stderr, "  test_embed_f16 OK\n");
    return 0;
}

/* Oracle for the Q8_0 path: embed_token_q8_0 (ds4.c:6713-6737).  Layout is
 * blocks of 32 values, 34 bytes each: a 2-byte F16 scale then 32 int8.
 * Built by hand here so the layout assumption is explicit rather than
 * inherited from the implementation under test. */
static int test_embed_q8_0(void) {
    enum { N_VOCAB = 4, N_EMBD = 64, BLOCKS = N_EMBD / 32, TOKEN = 2 };
    const size_t row_bytes = (size_t)BLOCKS * 34;
    unsigned char table[N_VOCAB * BLOCKS * 34];
    float         want[N_EMBD];
    float         got[N_EMBD];

    memset(table, 0, sizeof(table));
    for (int t = 0; t < N_VOCAB; t++) {
        for (int b = 0; b < BLOCKS; b++) {
            unsigned char *blk = &table[(size_t)t * row_bytes + (size_t)b * 34];
            /* F16 1.0 is 0x3C00, so the scale is exactly one and the
             * dequantised value equals the stored int8 exactly. */
            blk[0] = 0x00; blk[1] = 0x3C;
            for (int i = 0; i < 32; i++) {
                blk[2 + i] = (unsigned char)(signed char)((t + b + i) % 15 - 7);
            }
        }
    }
    for (int e = 0; e < N_EMBD; e++) {
        int b = e / 32, i = e % 32;
        want[e] = (float)(signed char)((TOKEN + b + i) % 15 - 7);
    }

    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(to != NULL, "embed_q8_0: allocation failed");
    CHECK(ds4_gpu_embed_token_q8_0_tensor(to, table, sizeof(table), 0,
                                          N_VOCAB, TOKEN, N_EMBD) != 0,
          "embed_q8_0: call");
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0, "embed_q8_0: read");
    for (int e = 0; e < N_EMBD; e++) {
        /* Scale is exactly 1.0, so this comparison is exact, not tolerant. */
        CHECK(got[e] == want[e], "embed_q8_0: value mismatch");
    }

    ds4_gpu_tensor_free(to);
    fprintf(stderr, "  test_embed_q8_0 OK\n");
    return 0;
}

/* Oracle for the batched Q8_0 path: embed_token_q8_0 (ds4.c:6713-6737)
 * applied per token.  This exercises the block-stride arithmetic
 * (block = e/32, offset = e%32, block stride 34 bytes) AND the per-token
 * row offset (tok * blocks * 34) TOGETHER: test_embed_q8_0 above exercises
 * the block arithmetic with only one token, and test_embed_tokens_hc_clamp
 * exercises the per-token row offset with only the F16 path.  Neither
 * alone would catch an error that appears only when both combine, which is
 * exactly what this test is for.  It also covers the same id-clamping
 * behaviour as test_embed_tokens_hc_clamp, on the Q8_0 path. */
static int test_embed_tokens_q8_0(void) {
    enum { N_VOCAB = 5, N_EMBD = 64, BLOCKS = N_EMBD / 32, N_TOKENS = 5 };
    const size_t row_bytes = (size_t)BLOCKS * 34;
    unsigned char table[N_VOCAB * BLOCKS * 34];
    float         want[N_TOKENS * N_EMBD];
    float         got[N_TOKENS * N_EMBD];
    /* Three distinct real tokens (1, 3, 2) plus one id past n_vocab and one
     * negative id, both of which must clamp to token 0's row. */
    int32_t       tokens[N_TOKENS] = { 1, 3, 2, N_VOCAB + 3, -1 };

    memset(table, 0, sizeof(table));
    for (int t = 0; t < N_VOCAB; t++) {
        for (int b = 0; b < BLOCKS; b++) {
            unsigned char *blk = &table[(size_t)t * row_bytes + (size_t)b * 34];
            /* F16 1.0 is 0x3C00, so the scale is exactly one and the
             * dequantised value equals the stored int8 exactly. */
            blk[0] = 0x00; blk[1] = 0x3C;
            for (int i = 0; i < 32; i++) {
                /* Depends on both t and b, so a wrong row offset or a
                 * wrong block offset each produce a visibly wrong value
                 * rather than a coincidentally matching one. */
                blk[2 + i] = (unsigned char)(signed char)((t + b + i) % 15 - 7);
            }
        }
    }

    for (int ti = 0; ti < N_TOKENS; ti++) {
        int32_t  raw = tokens[ti];
        uint32_t tok = raw < 0 ? 0u : (uint32_t)raw;
        if (tok >= N_VOCAB) tok = 0u;
        for (int e = 0; e < N_EMBD; e++) {
            int b = e / 32, i = e % 32;
            want[ti * N_EMBD + e] =
                    (float)(signed char)(((int)tok + b + i) % 15 - 7);
        }
    }

    ds4_gpu_tensor *tt = ds4_gpu_tensor_alloc(sizeof(tokens));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tt != NULL && to != NULL, "embed_tokens_q8_0: allocation failed");
    CHECK(ds4_gpu_tensor_write(tt, 0, tokens, sizeof(tokens)) != 0,
          "embed_tokens_q8_0: write tokens");

    CHECK(ds4_gpu_embed_tokens_q8_0_tensor(to, tt, table, sizeof(table), 0,
                                           N_VOCAB, N_TOKENS, N_EMBD) != 0,
          "embed_tokens_q8_0: call");
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0,
          "embed_tokens_q8_0: read");

    for (int i = 0; i < N_TOKENS * N_EMBD; i++) {
        /* Scale is exactly 1.0, so this comparison is exact, not tolerant. */
        CHECK(got[i] == want[i], "embed_tokens_q8_0: value mismatch");
    }

    ds4_gpu_tensor_free(tt);
    ds4_gpu_tensor_free(to);
    fprintf(stderr, "  test_embed_tokens_q8_0 OK\n");
    return 0;
}

/* Batched clamping test: an id of N_VOCAB + 3 and an id of -1 must both
 * produce token 0's embedding rather than failing.  Specified at
 * rocm/ds4_rocm_common.cuh:65-83 and load-bearing, not a bug to fix. */
static int test_embed_tokens_hc_clamp(void) {
    enum { N_VOCAB = 5, N_EMBD = 16, N_HC = 2, N_TOKENS = 2 };
    static const unsigned short kHalfOne[8] = {
        0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700
    };
    static const float kHalfOneF[8] = {0,1,2,3,4,5,6,7};

    unsigned short table[N_VOCAB * N_EMBD];
    float          want0[N_EMBD];
    float          got[N_TOKENS * N_HC * N_EMBD];
    int32_t        tokens[N_TOKENS] = { N_VOCAB + 3, -1 };

    for (int t = 0; t < N_VOCAB; t++) {
        for (int e = 0; e < N_EMBD; e++) {
            table[t * N_EMBD + e] = kHalfOne[(t + e) % 8];
        }
    }
    for (int e = 0; e < N_EMBD; e++) want0[e] = kHalfOneF[(0 + e) % 8];

    ds4_gpu_tensor *tt = ds4_gpu_tensor_alloc(sizeof(tokens));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tt != NULL && to != NULL, "embed_tokens_clamp: allocation failed");
    CHECK(ds4_gpu_tensor_write(tt, 0, tokens, sizeof(tokens)) != 0,
          "embed_tokens_clamp: write tokens");

    CHECK(ds4_gpu_embed_tokens_hc_tensor(to, tt, table, sizeof(table), 0,
                                         N_VOCAB, N_TOKENS, N_EMBD, N_HC) != 0,
          "embed_tokens_clamp: call");
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0,
          "embed_tokens_clamp: read");

    for (int t = 0; t < N_TOKENS; t++) {
        for (int h = 0; h < N_HC; h++) {
            for (int e = 0; e < N_EMBD; e++) {
                size_t idx = ((size_t)t * N_HC + (size_t)h) * N_EMBD + (size_t)e;
                CHECK_CLOSE(got[idx], want0[e], 1e-3,
                            "embed_tokens_clamp: value mismatch");
            }
        }
    }

    ds4_gpu_tensor_free(tt);
    ds4_gpu_tensor_free(to);
    fprintf(stderr, "  test_embed_tokens_hc_clamp OK\n");
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_embed_f16() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_embed_q8_0() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_embed_tokens_q8_0() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_embed_tokens_hc_clamp() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_embedding OK\n");
    return 0;
}
