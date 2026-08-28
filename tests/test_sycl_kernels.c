/* Correctness tests for SYCL compute kernels, validated against scalar
 * oracles implemented here.  The ds4.c CPU references are static and
 * cannot be linked, so each oracle reimplements the documented formula
 * with the ds4.c line number cited.  Needs no model file. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <math.h>
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

/* Kernels accumulate in a different order than the scalar oracle, so
 * exact equality is not required for reductions.  Elementwise ops should
 * still match to within a tight tolerance. */
#define CHECK_CLOSE(got, want, tol, msg)                                    \
    do {                                                                    \
        double d_ = fabs((double)(got) - (double)(want));                   \
        if (!(d_ <= (tol))) {                                               \
            fprintf(stderr, "FAIL: %s (got %.9g want %.9g delta %.3g)\n",   \
                    (msg), (double)(got), (double)(want), d_);              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static int test_add(void) {
    enum { N = 1024 };
    float a[N], b[N], c[N], got[N];
    for (int i = 0; i < N; i++) {
        a[i] = (float)i * 0.5f;
        b[i] = (float)(N - i) * 0.25f;
        c[i] = (float)(i % 7) - 3.0f;
    }

    ds4_gpu_tensor *ta = ds4_gpu_tensor_alloc(sizeof(a));
    ds4_gpu_tensor *tb = ds4_gpu_tensor_alloc(sizeof(b));
    ds4_gpu_tensor *tc = ds4_gpu_tensor_alloc(sizeof(c));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(ta && tb && tc && to, "add: allocation failed");

    CHECK(ds4_gpu_tensor_write(ta, 0, a, sizeof(a)) != 0, "add: write a");
    CHECK(ds4_gpu_tensor_write(tb, 0, b, sizeof(b)) != 0, "add: write b");
    CHECK(ds4_gpu_tensor_write(tc, 0, c, sizeof(c)) != 0, "add: write c");

    CHECK(ds4_gpu_add_tensor(to, ta, tb, N) != 0, "add: ds4_gpu_add_tensor");
    memset(got, 0, sizeof(got));
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0, "add: read");
    for (int i = 0; i < N; i++) {
        /* Oracle: out = a + b, ds4.c:11631 and the other residual sites. */
        CHECK_CLOSE(got[i], a[i] + b[i], 1e-6, "add: value mismatch");
    }

    CHECK(ds4_gpu_add3_tensor(to, ta, tb, tc, N) != 0, "add3: ds4_gpu_add3_tensor");
    memset(got, 0, sizeof(got));
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0, "add3: read");
    for (int i = 0; i < N; i++) {
        /* Oracle: out = a + b + c.  NOTE: ds4 has no CPU implementation of
         * a three-term add; every CPU site does two sequential two-term
         * adds, and ds4_gpu_add3_tensor's only caller is the GLM decode
         * path (ds4.c:42284), which this project does not support.  This
         * is a synthetic-only oracle by necessity, recorded deliberately
         * rather than skipped. */
        CHECK_CLOSE(got[i], a[i] + b[i] + c[i], 1e-6, "add3: value mismatch");
    }

    /* Zero-length work is SUCCESS and returns nonzero, matching
     * rocm/ds4_rocm_misc_launch.cuh:3 `if (n == 0u) return 1;`. */
    CHECK(ds4_gpu_add_tensor(to, ta, tb, 0) != 0, "add: n=0 must succeed");

    /* Undersized output must be rejected, returning 0. */
    ds4_gpu_tensor *small = ds4_gpu_tensor_alloc(16);
    CHECK(small != NULL, "add: small allocation failed");
    CHECK(ds4_gpu_add_tensor(small, ta, tb, N) == 0,
          "add: undersized output must be rejected");
    ds4_gpu_tensor_free(small);

    ds4_gpu_tensor_free(ta);
    ds4_gpu_tensor_free(tb);
    ds4_gpu_tensor_free(tc);
    ds4_gpu_tensor_free(to);
    fprintf(stderr, "  test_add OK\n");
    return 0;
}

/* Oracle mirrors swiglu at ds4.c:10556-10567 with silu at ds4.c:10546.
 * The GPU kernel additionally multiplies by `weight`, which the CPU
 * function has no parameter for; we implement it faithfully rather than
 * assuming callers always pass 1.0. */
static float oracle_swiglu(float g, float u, float clamp, float weight) {
    if (clamp > 1e-6f) {
        if (u >  clamp) u =  clamp;
        if (u < -clamp) u = -clamp;
        if (g >  clamp) g =  clamp;
    }
    float s = g / (1.0f + expf(-g));
    return s * u * weight;
}

static int test_swiglu(void) {
    enum { N = 512 };
    float gate[N], up[N], got[N];
    for (int i = 0; i < N; i++) {
        gate[i] = ((float)i / (float)N) * 8.0f - 4.0f;
        up[i]   = ((float)(N - i) / (float)N) * 6.0f - 3.0f;
    }

    ds4_gpu_tensor *tg = ds4_gpu_tensor_alloc(sizeof(gate));
    ds4_gpu_tensor *tu = ds4_gpu_tensor_alloc(sizeof(up));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tg && tu && to, "swiglu: allocation failed");
    CHECK(ds4_gpu_tensor_write(tg, 0, gate, sizeof(gate)) != 0, "swiglu: write gate");
    CHECK(ds4_gpu_tensor_write(tu, 0, up, sizeof(up)) != 0, "swiglu: write up");

    /* Unclamped, unit weight. */
    CHECK(ds4_gpu_swiglu_tensor(to, tg, tu, N, 0.0f, 1.0f) != 0, "swiglu: call");
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0, "swiglu: read");
    for (int i = 0; i < N; i++) {
        CHECK_CLOSE(got[i], oracle_swiglu(gate[i], up[i], 0.0f, 1.0f), 1e-5,
                    "swiglu: unclamped mismatch");
    }

    /* Clamped, non-unit weight.  A weight other than 1.0 is what proves
     * the parameter is actually applied rather than ignored. */
    CHECK(ds4_gpu_swiglu_tensor(to, tg, tu, N, 2.0f, 0.75f) != 0, "swiglu: clamped call");
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0, "swiglu: clamped read");
    for (int i = 0; i < N; i++) {
        CHECK_CLOSE(got[i], oracle_swiglu(gate[i], up[i], 2.0f, 0.75f), 1e-5,
                    "swiglu: clamped mismatch");
    }

    CHECK(ds4_gpu_swiglu_tensor(to, tg, tu, 0, 0.0f, 1.0f) != 0,
          "swiglu: n=0 must succeed");

    ds4_gpu_tensor_free(tg);
    ds4_gpu_tensor_free(tu);
    ds4_gpu_tensor_free(to);
    fprintf(stderr, "  test_swiglu OK\n");
    return 0;
}

/* Oracle mirrors the gate loop in output_hc_head_one, ds4.c:14018-14020.
 * IMPORTANT NUMERICAL NOTE: ds4's CPU path uses sigmoid_stable
 * (ds4.c:10419-10427), which branches on the sign of x to avoid exp()
 * overflow.  The GPU kernels on every backend use the unstable
 * 1/(1+exp(-z)) form.  The two are mathematically equal and differ only
 * for large-magnitude negative z.  This oracle uses the STABLE form and
 * the test inputs are kept in a bounded range so the difference cannot
 * appear.  If a future change feeds large negative values here and this
 * test starts failing, that is a pre-existing CPU/GPU divergence, not a
 * SYCL regression. */
static float oracle_sigmoid_stable(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float e = expf(x);
    return e / (1.0f + e);
}

static int test_output_hc_weights(void) {
    enum { N_HC = 4, N_TOK = 64, N = N_HC * N_TOK };
    const float eps = 0.01f;
    const size_t row_bytes = (size_t)N_HC * sizeof(float);
    float pre[N], got[N];
    for (int i = 0; i < N; i++) pre[i] = ((float)i / (float)N) * 4.0f - 2.0f;

    /* Model buffer: one f32 scale, then n_hc f32 base values. */
    float model[1 + N_HC];
    model[0] = 1.25f;
    for (int h = 0; h < N_HC; h++) model[1 + h] = (float)h * 0.5f - 0.75f;

    ds4_gpu_tensor *tp = ds4_gpu_tensor_alloc(sizeof(pre));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tp && to, "hc_weights: allocation failed");
    CHECK(ds4_gpu_tensor_write(tp, 0, pre, sizeof(pre)) != 0, "hc_weights: write pre");

    CHECK(ds4_gpu_output_hc_weights_tensor(to, tp, model, sizeof(model),
                                           0, sizeof(float), N_HC, eps) != 0,
          "hc_weights: call");
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0, "hc_weights: read");
    for (int i = 0; i < N; i++) {
        float z = pre[i] * model[0] + model[1 + (i % N_HC)];
        CHECK_CLOSE(got[i], oracle_sigmoid_stable(z) + eps, 1e-5,
                    "hc_weights: value mismatch");
    }

    /* A scale offset past the end of the model buffer must be rejected. */
    CHECK(ds4_gpu_output_hc_weights_tensor(to, tp, model, sizeof(model),
                                           sizeof(model), sizeof(float),
                                           N_HC, eps) == 0,
          "hc_weights: out-of-range scale offset must be rejected");

    /* An out tensor whose byte size is not an exact multiple of
     * n_hc * sizeof(float) must be rejected: it cannot hold a whole
     * number of HC rows.  rocm/ds4_rocm_hc_output_launch.cuh:213-218 is
     * the authority for this guard. */
    ds4_gpu_tensor *to_unaligned = ds4_gpu_tensor_alloc(row_bytes + 4);
    CHECK(to_unaligned != NULL, "hc_weights: unaligned allocation failed");
    CHECK(ds4_gpu_output_hc_weights_tensor(to_unaligned, tp, model,
                                           sizeof(model), 0, sizeof(float),
                                           N_HC, eps) == 0,
          "hc_weights: non-row-aligned out size must be rejected");
    ds4_gpu_tensor_free(to_unaligned);

    /* A pre tensor smaller than out must be rejected: there is not
     * enough input to cover every output element. */
    ds4_gpu_tensor *tp_small = ds4_gpu_tensor_alloc(sizeof(pre) / 2);
    CHECK(tp_small != NULL, "hc_weights: small pre allocation failed");
    CHECK(ds4_gpu_output_hc_weights_tensor(to, tp_small, model, sizeof(model),
                                           0, sizeof(float), N_HC, eps) == 0,
          "hc_weights: undersized pre must be rejected");
    ds4_gpu_tensor_free(tp_small);

    ds4_gpu_tensor_free(tp);
    ds4_gpu_tensor_free(to);
    fprintf(stderr, "  test_output_hc_weights OK\n");
    return 0;
}

/* Oracle mirrors cpu_directional_steering_project_rows, ds4.c:37054-37074:
 * dot = sum_i x[row][i]*dir[layer][i]; then x[row][i] -= scale*dot*dir[i].
 * The GPU reduces in tree order and the oracle sums sequentially, so the
 * dot products differ in the last bits; the tolerance reflects that. */
static int test_directional_steering(void) {
    enum { WIDTH = 300, ROWS = 8, LAYER = 2, NLAYER = 4 };
    const float scale = 0.5f;
    float x[ROWS * WIDTH], dir[NLAYER * WIDTH], got[ROWS * WIDTH];
    for (int i = 0; i < ROWS * WIDTH; i++) x[i] = ((float)(i % 17) - 8.0f) * 0.1f;
    for (int i = 0; i < NLAYER * WIDTH; i++) dir[i] = ((float)(i % 11) - 5.0f) * 0.05f;

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *td = ds4_gpu_tensor_alloc(sizeof(dir));
    CHECK(tx && td, "steering: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "steering: write x");
    CHECK(ds4_gpu_tensor_write(td, 0, dir, sizeof(dir)) != 0, "steering: write dir");

    CHECK(ds4_gpu_directional_steering_project_tensor(tx, td, LAYER, WIDTH,
                                                      ROWS, scale) != 0,
          "steering: call");

    /* A zero scale must succeed without modifying x, matching ROCm's
     * early-out at rocm/ds4_rocm_misc_launch.cuh:70. */
    CHECK(ds4_gpu_directional_steering_project_tensor(tx, td, LAYER, WIDTH,
                                                      ROWS, 0.0f) != 0,
          "steering: zero scale must succeed");
    CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0, "steering: read");

    const float *d = &dir[LAYER * WIDTH];
    for (int r = 0; r < ROWS; r++) {
        double dot = 0.0;
        for (int i = 0; i < WIDTH; i++) dot += (double)x[r * WIDTH + i] * d[i];
        for (int i = 0; i < WIDTH; i++) {
            float want = x[r * WIDTH + i] - (float)(scale * dot) * d[i];
            CHECK_CLOSE(got[r * WIDTH + i], want, 1e-4, "steering: value mismatch");
        }
    }

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(td);
    fprintf(stderr, "  test_directional_steering OK\n");
    return 0;
}

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

/* Oracle mirrors rms_norm_no_weight, ds4.c:6754-6760:
 *   scale = 1/sqrt(mean(x^2) + eps);  out[i] = x[i] * scale
 * The CPU accumulates in double and the GPU tree-reduces in float, so
 * expect ULP-level divergence at these row widths, not more. */
static void oracle_rms_norm_plain(float *out, const float *x, int n, float eps) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(sum / (double)n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale;
}

static int test_rms_norm_plain(void) {
    /* WIDTHS deliberately spans below, at, and above the fixed 256 work
     * group.  A width below 256 is the case that catches a port which
     * fails to initialise every local slot. */
    const int widths[] = {1, 17, 255, 256, 257, 1024};
    const float eps = 1e-5f;

    for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
        const int n = widths[wi];
        float *x    = (float *)malloc((size_t)n * sizeof(float));
        float *got  = (float *)malloc((size_t)n * sizeof(float));
        float *want = (float *)malloc((size_t)n * sizeof(float));
        CHECK(x && got && want, "rms_plain: host allocation failed");

        for (int i = 0; i < n; i++) x[i] = ((float)(i % 13) - 6.0f) * 0.25f;
        oracle_rms_norm_plain(want, x, n, eps);

        ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        ds4_gpu_tensor *to = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        CHECK(tx && to, "rms_plain: device allocation failed");
        CHECK(ds4_gpu_tensor_write(tx, 0, x, (uint64_t)n * sizeof(float)) != 0,
              "rms_plain: write");
        CHECK(ds4_gpu_rms_norm_plain_tensor(to, tx, (uint32_t)n, eps) != 0,
              "rms_plain: call");
        CHECK(ds4_gpu_tensor_read(to, 0, got, (uint64_t)n * sizeof(float)) != 0,
              "rms_plain: read");
        for (int i = 0; i < n; i++) {
            CHECK_CLOSE(got[i], want[i], 1e-4, "rms_plain: value mismatch");
        }

        ds4_gpu_tensor_free(tx);
        ds4_gpu_tensor_free(to);
        free(x); free(got); free(want);
    }

    /* Zero length is SUCCESS and returns nonzero, matching
     * rocm/ds4_rocm_norm_rope.cuh:414 `if (n == 0u) return 1;`. */
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(64);
    CHECK(t != NULL, "rms_plain: alloc for n=0 case");
    CHECK(ds4_gpu_rms_norm_plain_tensor(t, t, 0, eps) != 0,
          "rms_plain: n=0 must succeed");
    /* An output too small for n floats must be rejected. */
    CHECK(ds4_gpu_rms_norm_plain_tensor(t, t, 1024, eps) == 0,
          "rms_plain: undersized tensor must be rejected");
    ds4_gpu_tensor_free(t);

    fprintf(stderr, "  test_rms_norm_plain OK\n");
    return 0;
}

static int test_rms_norm_plain_rows(void) {
    enum { N = 300, ROWS = 5 };
    const float eps = 1e-5f;
    float x[N * ROWS], got[N * ROWS], want[N];

    for (int i = 0; i < N * ROWS; i++) x[i] = ((float)(i % 23) - 11.0f) * 0.1f;

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tx && to, "rms_plain_rows: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "rms_plain_rows: write");
    CHECK(ds4_gpu_rms_norm_plain_rows_tensor(to, tx, N, ROWS, eps) != 0,
          "rms_plain_rows: call");
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0, "rms_plain_rows: read");

    /* Each row must be normalised independently.  Rows are given different
     * magnitudes above, so a kernel that normalised across the whole buffer
     * instead of per row would fail here while passing a single-row test. */
    for (int r = 0; r < ROWS; r++) {
        oracle_rms_norm_plain(want, &x[r * N], N, eps);
        for (int i = 0; i < N; i++) {
            CHECK_CLOSE(got[r * N + i], want[i], 1e-4,
                        "rms_plain_rows: value mismatch");
        }
    }

    CHECK(ds4_gpu_rms_norm_plain_rows_tensor(to, tx, N, 0, eps) != 0,
          "rms_plain_rows: rows=0 must succeed");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(to);
    fprintf(stderr, "  test_rms_norm_plain_rows OK\n");
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_add() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_swiglu() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_output_hc_weights() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_directional_steering() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_embed_f16() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_embed_q8_0() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_embed_tokens_q8_0() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_embed_tokens_hc_clamp() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_rms_norm_plain() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_rms_norm_plain_rows() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_kernels OK\n");
    return 0;
}
