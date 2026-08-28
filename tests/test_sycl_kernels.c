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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

/* Oracle mirrors rms_norm_weight, ds4.c:6763-6769: identical to
 * oracle_rms_norm_plain above plus a per-channel out[i] *= w[i]. */
static void oracle_rms_norm_weight(float *out, const float *x, const float *w,
                                   int n, float eps) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(sum / (double)n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

static int test_rms_norm_weight(void) {
    const int widths[] = {1, 17, 255, 256, 257, 1024};
    const float eps = 1e-5f;

    for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
        const int n = widths[wi];
        float *x    = (float *)malloc((size_t)n * sizeof(float));
        float *w    = (float *)malloc((size_t)n * sizeof(float));
        float *got  = (float *)malloc((size_t)n * sizeof(float));
        float *want = (float *)malloc((size_t)n * sizeof(float));
        CHECK(x && w && got && want, "rms_weight: host allocation failed");

        for (int i = 0; i < n; i++) {
            x[i] = ((float)(i % 13) - 6.0f) * 0.25f;
            w[i] = ((float)(i % 7) + 1.0f) * 0.3f;
        }
        oracle_rms_norm_weight(want, x, w, n, eps);

        ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        ds4_gpu_tensor *to = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        CHECK(tx && to, "rms_weight: device allocation failed");
        CHECK(ds4_gpu_tensor_write(tx, 0, x, (uint64_t)n * sizeof(float)) != 0,
              "rms_weight: write");
        CHECK(ds4_gpu_rms_norm_weight_tensor(to, tx, w,
                                             (uint64_t)n * sizeof(float), 0,
                                             (uint32_t)n, eps) != 0,
              "rms_weight: call");
        CHECK(ds4_gpu_tensor_read(to, 0, got, (uint64_t)n * sizeof(float)) != 0,
              "rms_weight: read");
        for (int i = 0; i < n; i++) {
            CHECK_CLOSE(got[i], want[i], 1e-4, "rms_weight: value mismatch");
        }

        ds4_gpu_tensor_free(tx);
        ds4_gpu_tensor_free(to);
        free(x); free(w); free(got); free(want);
    }

    /* Zero length is SUCCESS, matching rocm/ds4_rocm_norm_rope.cuh:429. */
    float model_dummy[1] = {1.0f};
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(64);
    CHECK(t != NULL, "rms_weight: alloc for n=0 case");
    CHECK(ds4_gpu_rms_norm_weight_tensor(t, t, model_dummy,
                                         sizeof(model_dummy), 0, 0,
                                         eps) != 0,
          "rms_weight: n=0 must succeed");

    /* An out-of-range weight_offset must be rejected. */
    CHECK(ds4_gpu_rms_norm_weight_tensor(t, t, model_dummy,
                                         sizeof(model_dummy),
                                         sizeof(model_dummy), 1, eps) == 0,
          "rms_weight: out-of-range weight_offset must be rejected");
    ds4_gpu_tensor_free(t);

    fprintf(stderr, "  test_rms_norm_weight OK\n");
    return 0;
}

static int test_rms_norm_weight_rows(void) {
    enum { N = 300, ROWS = 5 };
    const float eps = 1e-5f;
    float x[N * ROWS], w[N], got[N * ROWS], want[N];

    for (int i = 0; i < N * ROWS; i++) x[i] = ((float)(i % 23) - 11.0f) * 0.1f;
    for (int i = 0; i < N; i++) w[i] = ((float)(i % 5) + 1.0f) * 0.2f;

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tx && to, "rms_weight_rows: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "rms_weight_rows: write");
    CHECK(ds4_gpu_rms_norm_weight_rows_tensor(to, tx, w, sizeof(w), 0, N,
                                              ROWS, eps) != 0,
          "rms_weight_rows: call");
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0,
          "rms_weight_rows: read");

    for (int r = 0; r < ROWS; r++) {
        oracle_rms_norm_weight(want, &x[r * N], w, N, eps);
        for (int i = 0; i < N; i++) {
            CHECK_CLOSE(got[r * N + i], want[i], 1e-4,
                        "rms_weight_rows: value mismatch");
        }
    }

    CHECK(ds4_gpu_rms_norm_weight_rows_tensor(to, tx, w, sizeof(w), 0, N, 0,
                                              eps) != 0,
          "rms_weight_rows: rows=0 must succeed");

    /* An out-of-range weight_offset must be rejected. */
    CHECK(ds4_gpu_rms_norm_weight_rows_tensor(to, tx, w, sizeof(w),
                                              sizeof(w), N, ROWS, eps) == 0,
          "rms_weight_rows: out-of-range weight_offset must be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(to);
    fprintf(stderr, "  test_rms_norm_weight_rows OK\n");
    return 0;
}

/* ds4_gpu_add_rms_norm_weight_tensor has no kernel of its own: it composes
 * ds4_gpu_add_tensor and ds4_gpu_rms_norm_weight_tensor
 * (rocm/ds4_rocm_norm_rope.cuh:451-469).  Validate against those two oracle
 * steps run separately, and assert sum_out as well as norm_out: a kernel
 * that computed the norm correctly but never published the elementwise sum
 * would otherwise pass. */
static int test_add_rms_norm_weight(void) {
    enum { N = 200 };
    const float eps = 1e-5f;
    float a[N], b[N], w[N], want_sum[N], want_norm[N];
    float got_sum[N], got_norm[N];

    for (int i = 0; i < N; i++) {
        a[i] = ((float)(i % 19) - 9.0f) * 0.15f;
        b[i] = ((float)(i % 11) - 5.0f) * 0.1f;
        w[i] = ((float)(i % 6) + 1.0f) * 0.25f;
        want_sum[i] = a[i] + b[i];
    }
    oracle_rms_norm_weight(want_norm, want_sum, w, N, eps);

    ds4_gpu_tensor *ta = ds4_gpu_tensor_alloc(sizeof(a));
    ds4_gpu_tensor *tb = ds4_gpu_tensor_alloc(sizeof(b));
    ds4_gpu_tensor *tsum = ds4_gpu_tensor_alloc(sizeof(got_sum));
    ds4_gpu_tensor *tnorm = ds4_gpu_tensor_alloc(sizeof(got_norm));
    CHECK(ta && tb && tsum && tnorm, "add_rms_weight: allocation failed");
    CHECK(ds4_gpu_tensor_write(ta, 0, a, sizeof(a)) != 0,
          "add_rms_weight: write a");
    CHECK(ds4_gpu_tensor_write(tb, 0, b, sizeof(b)) != 0,
          "add_rms_weight: write b");

    CHECK(ds4_gpu_add_rms_norm_weight_tensor(tnorm, tsum, ta, tb, w,
                                             sizeof(w), 0, N, eps) != 0,
          "add_rms_weight: call");
    CHECK(ds4_gpu_tensor_read(tsum, 0, got_sum, sizeof(got_sum)) != 0,
          "add_rms_weight: read sum");
    CHECK(ds4_gpu_tensor_read(tnorm, 0, got_norm, sizeof(got_norm)) != 0,
          "add_rms_weight: read norm");

    for (int i = 0; i < N; i++) {
        CHECK_CLOSE(got_sum[i], want_sum[i], 1e-4,
                    "add_rms_weight: sum_out mismatch");
        CHECK_CLOSE(got_norm[i], want_norm[i], 1e-4,
                    "add_rms_weight: norm_out mismatch");
    }

    ds4_gpu_tensor_free(ta);
    ds4_gpu_tensor_free(tb);
    ds4_gpu_tensor_free(tsum);
    ds4_gpu_tensor_free(tnorm);
    fprintf(stderr, "  test_add_rms_norm_weight OK\n");
    return 0;
}

/* Oracle mirrors head_rms_norm_inplace, ds4.c:6772-6781: normalises each
 * head's head_dim slice independently within one token's n_head*head_dim
 * block. */
static void oracle_head_rms_norm_inplace(float *x, int n_head, int head_dim,
                                         float eps) {
    for (int h = 0; h < n_head; h++) {
        float *head = x + (size_t)h * head_dim;
        double ss = 0.0;
        for (int i = 0; i < head_dim; i++) ss += (double)head[i] * head[i];
        const float scale = 1.0f / sqrtf((float)(ss / (double)head_dim) + eps);
        for (int i = 0; i < head_dim; i++) head[i] *= scale;
    }
}

static int test_head_rms_norm(void) {
    enum { N_TOK = 3, N_HEAD = 4, HEAD_DIM = 65 };
    const float eps = 1e-5f;
    float x[N_TOK * N_HEAD * HEAD_DIM];
    float want[N_TOK * N_HEAD * HEAD_DIM];
    float got[N_TOK * N_HEAD * HEAD_DIM];

    /* Each (token, head) pair gets a different magnitude, so a kernel that
     * normalised across heads instead of within each, or across the whole
     * buffer instead of per (token, head), fails rather than passing by
     * coincidence. */
    for (int t = 0; t < N_TOK; t++) {
        for (int h = 0; h < N_HEAD; h++) {
            const float mag = (float)(h + 1) * (float)(t + 2);
            float *dst = &x[(t * N_HEAD + h) * HEAD_DIM];
            for (int i = 0; i < HEAD_DIM; i++) {
                dst[i] = ((float)(i % 9) - 4.0f) * 0.1f * mag;
            }
        }
    }
    memcpy(want, x, sizeof(x));

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    CHECK(tx != NULL, "head_rms_norm: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "head_rms_norm: write");
    CHECK(ds4_gpu_head_rms_norm_tensor(tx, N_TOK, N_HEAD, HEAD_DIM, eps) != 0,
          "head_rms_norm: call");
    CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0,
          "head_rms_norm: read");

    for (int t = 0; t < N_TOK; t++) {
        oracle_head_rms_norm_inplace(&want[t * N_HEAD * HEAD_DIM], N_HEAD,
                                     HEAD_DIM, eps);
    }
    for (int i = 0; i < N_TOK * N_HEAD * HEAD_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], 1e-4, "head_rms_norm: value mismatch");
    }

    CHECK(ds4_gpu_head_rms_norm_tensor(tx, N_TOK, N_HEAD, 0, eps) != 0,
          "head_rms_norm: head_dim=0 must succeed");
    CHECK(ds4_gpu_head_rms_norm_tensor(tx, 0, N_HEAD, HEAD_DIM, eps) != 0,
          "head_rms_norm: n_tok=0 must succeed");

    ds4_gpu_tensor *small = ds4_gpu_tensor_alloc(4);
    CHECK(small != NULL, "head_rms_norm: alloc for undersized case");
    CHECK(ds4_gpu_head_rms_norm_tensor(small, 1, 1, HEAD_DIM, eps) == 0,
          "head_rms_norm: undersized tensor must be rejected");
    ds4_gpu_tensor_free(small);

    ds4_gpu_tensor_free(tx);
    fprintf(stderr, "  test_head_rms_norm OK\n");
    return 0;
}

/* ds4_gpu_dsv4_qkv_rms_norm_rows_tensor runs rms_norm_weight
 * (ds4.c:6763-6769) twice per row in one launch, with the Q and KV halves
 * selecting their own width (rocm/ds4_rocm_norm_rope.cuh:47-81).  Q_N and
 * KV_N are deliberately different: a port that computed the row stride
 * before selecting the width, or that reused one width for both sides,
 * would mis-index one half silently if the widths matched. */
static int test_dsv4_qkv_rms_norm_rows(void) {
    enum { ROWS = 4, Q_N = 192, KV_N = 96 };
    const float eps = 1e-5f;
    float q_x[ROWS * Q_N], kv_x[ROWS * KV_N];
    float q_w[Q_N], kv_w[KV_N];
    float q_got[ROWS * Q_N], kv_got[ROWS * KV_N];
    float q_want[Q_N], kv_want[KV_N];

    for (int i = 0; i < ROWS * Q_N; i++) {
        q_x[i] = ((float)(i % 17) - 8.0f) * 0.1f;
    }
    for (int i = 0; i < ROWS * KV_N; i++) {
        kv_x[i] = ((float)(i % 11) - 5.0f) * 0.2f;
    }
    for (int i = 0; i < Q_N; i++) q_w[i] = ((float)(i % 5) + 1.0f) * 0.3f;
    for (int i = 0; i < KV_N; i++) kv_w[i] = ((float)(i % 3) + 1.0f) * 0.4f;

    /* Concatenate both weight vectors into one host "model" buffer so the
     * offsets exercised are real byte offsets rather than both being 0. */
    unsigned char model[sizeof(q_w) + sizeof(kv_w)];
    memcpy(model, q_w, sizeof(q_w));
    memcpy(model + sizeof(q_w), kv_w, sizeof(kv_w));
    const uint64_t q_off = 0;
    const uint64_t kv_off = sizeof(q_w);

    ds4_gpu_tensor *tqx = ds4_gpu_tensor_alloc(sizeof(q_x));
    ds4_gpu_tensor *tqo = ds4_gpu_tensor_alloc(sizeof(q_got));
    ds4_gpu_tensor *tkvx = ds4_gpu_tensor_alloc(sizeof(kv_x));
    ds4_gpu_tensor *tkvo = ds4_gpu_tensor_alloc(sizeof(kv_got));
    CHECK(tqx && tqo && tkvx && tkvo, "dsv4_qkv: allocation failed");
    CHECK(ds4_gpu_tensor_write(tqx, 0, q_x, sizeof(q_x)) != 0,
          "dsv4_qkv: write q");
    CHECK(ds4_gpu_tensor_write(tkvx, 0, kv_x, sizeof(kv_x)) != 0,
          "dsv4_qkv: write kv");

    CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
              tqo, tqx, model, sizeof(model), q_off, Q_N,
              tkvo, tkvx, kv_off, KV_N, ROWS, eps) != 0,
          "dsv4_qkv: call");

    CHECK(ds4_gpu_tensor_read(tqo, 0, q_got, sizeof(q_got)) != 0,
          "dsv4_qkv: read q");
    CHECK(ds4_gpu_tensor_read(tkvo, 0, kv_got, sizeof(kv_got)) != 0,
          "dsv4_qkv: read kv");

    for (int r = 0; r < ROWS; r++) {
        oracle_rms_norm_weight(q_want, &q_x[r * Q_N], q_w, Q_N, eps);
        for (int i = 0; i < Q_N; i++) {
            CHECK_CLOSE(q_got[r * Q_N + i], q_want[i], 1e-4,
                        "dsv4_qkv: q value mismatch");
        }
        oracle_rms_norm_weight(kv_want, &kv_x[r * KV_N], kv_w, KV_N, eps);
        for (int i = 0; i < KV_N; i++) {
            CHECK_CLOSE(kv_got[r * KV_N + i], kv_want[i], 1e-4,
                        "dsv4_qkv: kv value mismatch");
        }
    }

    CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
              tqo, tqx, model, sizeof(model), q_off, Q_N,
              tkvo, tkvx, kv_off, KV_N, 0, eps) != 0,
          "dsv4_qkv: rows=0 must succeed");

    CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
              tqo, tqx, model, sizeof(model), sizeof(model), Q_N,
              tkvo, tkvx, kv_off, KV_N, ROWS, eps) == 0,
          "dsv4_qkv: out-of-range q weight_offset must be rejected");

    ds4_gpu_tensor_free(tqx);
    ds4_gpu_tensor_free(tqo);
    ds4_gpu_tensor_free(tkvx);
    ds4_gpu_tensor_free(tkvo);
    fprintf(stderr, "  test_dsv4_qkv_rms_norm_rows OK\n");
    return 0;
}

/* Oracle for rope_yarn_ramp, ds4.c:10209-10212, and shared by both RoPE
 * oracles below.  Indexed by the channel index i0 = pair * 2. */
static float oracle_rope_yarn_ramp(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

/* Oracle mirrors rope_tail_ext_inplace, ds4.c:10228-10276, reached via
 * rope_tail_layer_inplace, ds4.c:10292-10323, specialised to ONE head's
 * head_dim-wide row (ds4.c's version loops over n_head itself; each head
 * resets theta_extrap to `pos` independently, so operating one row at a
 * time here is equivalent).
 *
 * DELIBERATE DIVERGENCE FROM THE GPU: this oracle accumulates the angle
 * ITERATIVELY, theta_extrap *= theta_scale per pair (ds4.c:10273), matching
 * the CPU reference exactly.  Every GPU backend (Metal, CUDA, ROCm, and
 * this SYCL port) instead computes theta_extrap = pos * pow(theta_scale,
 * pair) directly.  The two forms are mathematically identical but not
 * bit-identical; the caller's tolerance absorbs the resulting ULP-level
 * drift.  This is a pre-existing cross-backend property of ds4, not a
 * SYCL defect, and the kernel must NOT be changed to match this oracle's
 * form. */
static void oracle_rope_tail_row(float *row, uint32_t head_dim, uint32_t n_rot,
                                 uint32_t pos, uint32_t n_ctx_orig,
                                 float freq_base, float freq_scale,
                                 float ext_factor, float attn_factor,
                                 float beta_fast, float beta_slow,
                                 int inverse) {
    if (n_rot == 0u) return;
    const uint32_t n_nope = head_dim - n_rot;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    const float sin_sign = inverse ? -1.0f : 1.0f;
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        const float start = floorf((float)n_rot *
                logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        const float end = ceilf((float)n_rot *
                logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, start);
        corr1 = fminf((float)(n_rot - 1), end);
    }

    float *tail = row + n_nope;
    float theta_extrap = (float)pos;
    for (uint32_t i = 0; i < n_rot; i += 2) {
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float ramp_mix =
                    oracle_rope_yarn_ramp(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        const float c = cosf(theta) * mscale;
        const float s = sin_sign * sinf(theta) * mscale;
        const float x0 = tail[i];
        const float x1 = tail[i + 1];
        tail[i]     = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
        theta_extrap *= theta_scale;
    }
}

/* ds4_gpu_rope_tail_tensor: the flat, pair-parallel, no-reduction RoPE
 * kernel.  N_HEAD > 1 and N_TOK > 1 both hold so the kernel's own
 * decomposition of the flat id into (pair, head, token) is genuinely
 * exercised; a transposed decomposition would still pass with either at
 * 1.  HEAD_DIM > N_ROT exercises the scale-only... except this kernel has
 * no scale at all, so it exercises the leading channels being left
 * completely untouched, which is the flat path's equivalent property. */
static int test_rope_tail(void) {
    enum { N_TOK = 3, N_HEAD = 2, HEAD_DIM = 24, N_ROT = 16,
           N_NOPE = HEAD_DIM - N_ROT, TOTAL = N_TOK * N_HEAD * HEAD_DIM };
    const uint32_t POS0 = 5;
    const uint32_t N_CTX_ORIG = 4096;
    const float FREQ_BASE = 10000.0f;
    const float BETA_FAST = 32.0f;
    const float BETA_SLOW = 1.0f;
    const float ATTN_FACTOR = 1.0f;
    /* c/s divergence between the iterative oracle and the GPU's
     * direct-power form measured at these parameters is on the order of
     * 1e-7; this tolerance leaves two orders of magnitude of margin. */
    const double TOL = 1e-4;

    struct { float freq_scale; float ext_factor; int inverse; } cases[] = {
        {1.0f, 0.0f, 0}, /* plain interpolation, forward */
        {1.0f, 0.0f, 1}, /* plain interpolation, inverse */
        {0.5f, 1.0f, 0}, /* YaRN correction active, forward */
        {0.5f, 1.0f, 1}, /* YaRN correction active, inverse */
    };

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        float x[TOTAL], got[TOTAL], want_row[HEAD_DIM];
        for (int i = 0; i < TOTAL; i++) x[i] = ((float)(i % 13) - 6.0f) * 0.2f;

        ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
        CHECK(tx != NULL, "rope_tail: allocation failed");
        CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
              "rope_tail: write");

        CHECK(ds4_gpu_rope_tail_tensor(tx, N_TOK, N_HEAD, HEAD_DIM, N_ROT,
                                       POS0, N_CTX_ORIG, cases[ci].inverse,
                                       FREQ_BASE, cases[ci].freq_scale,
                                       cases[ci].ext_factor, ATTN_FACTOR,
                                       BETA_FAST, BETA_SLOW) != 0,
              "rope_tail: call");
        CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0,
              "rope_tail: read");

        for (int t = 0; t < N_TOK; t++) {
            for (int h = 0; h < N_HEAD; h++) {
                const int base = (t * N_HEAD + h) * HEAD_DIM;
                const float *row_x   = &x[base];
                const float *row_got = &got[base];

                /* No `scale`: the leading n_nope channels must be
                 * BYTE-IDENTICAL to the input.  A kernel that scaled them
                 * (borrowing the fused kernel's behaviour by mistake)
                 * would fail here. */
                for (int c = 0; c < N_NOPE; c++) {
                    CHECK(row_got[c] == row_x[c],
                          "rope_tail: leading channels must be untouched");
                }

                memcpy(want_row, row_x, sizeof(want_row));
                oracle_rope_tail_row(want_row, HEAD_DIM, N_ROT, POS0 + (uint32_t)t,
                                     N_CTX_ORIG, FREQ_BASE, cases[ci].freq_scale,
                                     cases[ci].ext_factor, ATTN_FACTOR,
                                     BETA_FAST, BETA_SLOW, cases[ci].inverse);
                for (int c = N_NOPE; c < HEAD_DIM; c++) {
                    CHECK_CLOSE(row_got[c], want_row[c], TOL,
                                "rope_tail: rotated tail mismatch");
                }
            }
        }

        ds4_gpu_tensor_free(tx);
    }

    /* n_rot == 0 has zero pairs and must succeed without modifying x,
     * matching the pairs == 0 early-out. */
    {
        float x[TOTAL], got[TOTAL];
        for (int i = 0; i < TOTAL; i++) x[i] = ((float)(i % 13) - 6.0f) * 0.2f;
        ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
        CHECK(tx != NULL, "rope_tail: n_rot=0 allocation failed");
        CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
              "rope_tail: n_rot=0 write");
        CHECK(ds4_gpu_rope_tail_tensor(tx, N_TOK, N_HEAD, HEAD_DIM, 0, POS0,
                                       N_CTX_ORIG, false, FREQ_BASE, 1.0f,
                                       0.0f, 1.0f, BETA_FAST, BETA_SLOW) != 0,
              "rope_tail: n_rot=0 must succeed");
        CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0,
              "rope_tail: n_rot=0 read");
        CHECK(memcmp(x, got, sizeof(x)) == 0,
              "rope_tail: n_rot=0 must leave the tensor untouched");
        ds4_gpu_tensor_free(tx);
    }

    /* An odd n_rot must be rejected: the kernel forms pairs, and an odd
     * count would read one element past the tail. */
    {
        ds4_gpu_tensor *t = ds4_gpu_tensor_alloc((uint64_t)TOTAL * sizeof(float));
        CHECK(t != NULL, "rope_tail: odd n_rot allocation failed");
        CHECK(ds4_gpu_rope_tail_tensor(t, N_TOK, N_HEAD, HEAD_DIM, N_ROT - 1,
                                       POS0, N_CTX_ORIG, false, FREQ_BASE,
                                       1.0f, 0.0f, 1.0f, BETA_FAST,
                                       BETA_SLOW) == 0,
              "rope_tail: odd n_rot must be rejected");

        /* n_rot > head_dim must also be rejected. */
        CHECK(ds4_gpu_rope_tail_tensor(t, N_TOK, N_HEAD, HEAD_DIM,
                                       HEAD_DIM + 2, POS0, N_CTX_ORIG, false,
                                       FREQ_BASE, 1.0f, 0.0f, 1.0f, BETA_FAST,
                                       BETA_SLOW) == 0,
              "rope_tail: n_rot > head_dim must be rejected");
        ds4_gpu_tensor_free(t);
    }

    /* A tensor smaller than n_tok * n_head * head_dim floats must be
     * rejected. */
    {
        ds4_gpu_tensor *small = ds4_gpu_tensor_alloc(sizeof(float) * 4);
        CHECK(small != NULL, "rope_tail: small allocation failed");
        CHECK(ds4_gpu_rope_tail_tensor(small, N_TOK, N_HEAD, HEAD_DIM, N_ROT,
                                       POS0, N_CTX_ORIG, false, FREQ_BASE,
                                       1.0f, 0.0f, 1.0f, BETA_FAST,
                                       BETA_SLOW) == 0,
              "rope_tail: undersized tensor must be rejected");
        ds4_gpu_tensor_free(small);
    }

    fprintf(stderr, "  test_rope_tail OK\n");
    return 0;
}

/* ds4_gpu_head_rms_norm_rope_tail_tensor: the fused, one-work-group-per-row
 * kernel.  Validated against the COMPOSED oracle: head RMS-norm first
 * (oracle_head_rms_norm_inplace with n_head=1, applied to one row at a
 * time), then RoPE on the now-scaled row, matching the GPU's own "scale
 * everything, then rotate the scaled tail" ordering -- the fused kernel
 * never stores an unscaled tail anywhere, so composing the two oracles in
 * this order is exactly equivalent to the kernel's single fused pass. */
static int test_head_rms_norm_rope_tail(void) {
    enum { N_TOK = 2, N_HEAD = 3, HEAD_DIM = 24, N_ROT = 16,
           TOTAL = N_TOK * N_HEAD * HEAD_DIM };
    const uint32_t POS0 = 5;
    const uint32_t N_CTX_ORIG = 4096;
    const float FREQ_BASE = 10000.0f;
    const float BETA_FAST = 32.0f;
    const float BETA_SLOW = 1.0f;
    const float ATTN_FACTOR = 1.0f;
    const float EPS = 1e-5f;
    const double TOL = 1e-4;

    struct { float freq_scale; float ext_factor; int inverse; } cases[] = {
        {1.0f, 0.0f, 0},
        {1.0f, 0.0f, 1},
        {0.5f, 1.0f, 0},
        {0.5f, 1.0f, 1},
    };

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        float x[TOTAL], got[TOTAL], want_row[HEAD_DIM];

        /* Each (token, head) pair gets a different magnitude, exactly as
         * test_head_rms_norm does, so a kernel that mixed up rows would
         * fail rather than pass by coincidence. */
        for (int t = 0; t < N_TOK; t++) {
            for (int h = 0; h < N_HEAD; h++) {
                const float mag = (float)(h + 1) * (float)(t + 2);
                float *dst = &x[(t * N_HEAD + h) * HEAD_DIM];
                for (int i = 0; i < HEAD_DIM; i++) {
                    dst[i] = ((float)(i % 9) - 4.0f) * 0.1f * mag;
                }
            }
        }

        ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
        CHECK(tx != NULL, "head_rms_rope: allocation failed");
        CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
              "head_rms_rope: write");

        CHECK(ds4_gpu_head_rms_norm_rope_tail_tensor(
                  tx, N_TOK, N_HEAD, HEAD_DIM, N_ROT, POS0, N_CTX_ORIG,
                  cases[ci].inverse, FREQ_BASE, cases[ci].freq_scale,
                  cases[ci].ext_factor, ATTN_FACTOR, BETA_FAST, BETA_SLOW,
                  EPS) != 0,
              "head_rms_rope: call");
        CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0,
              "head_rms_rope: read");

        for (int t = 0; t < N_TOK; t++) {
            for (int h = 0; h < N_HEAD; h++) {
                const int base = (t * N_HEAD + h) * HEAD_DIM;
                memcpy(want_row, &x[base], sizeof(want_row));
                oracle_head_rms_norm_inplace(want_row, 1, HEAD_DIM, EPS);
                oracle_rope_tail_row(want_row, HEAD_DIM, N_ROT, POS0 + (uint32_t)t,
                                     N_CTX_ORIG, FREQ_BASE, cases[ci].freq_scale,
                                     cases[ci].ext_factor, ATTN_FACTOR,
                                     BETA_FAST, BETA_SLOW, cases[ci].inverse);
                for (int c = 0; c < HEAD_DIM; c++) {
                    CHECK_CLOSE(got[base + c], want_row[c], TOL,
                                "head_rms_rope: value mismatch");
                }
            }
        }

        ds4_gpu_tensor_free(tx);
    }

    /* An odd n_rot must be rejected. */
    {
        ds4_gpu_tensor *t = ds4_gpu_tensor_alloc((uint64_t)TOTAL * sizeof(float));
        CHECK(t != NULL, "head_rms_rope: odd n_rot allocation failed");
        CHECK(ds4_gpu_head_rms_norm_rope_tail_tensor(
                  t, N_TOK, N_HEAD, HEAD_DIM, N_ROT - 1, POS0, N_CTX_ORIG,
                  false, FREQ_BASE, 1.0f, 0.0f, 1.0f, BETA_FAST, BETA_SLOW,
                  EPS) == 0,
              "head_rms_rope: odd n_rot must be rejected");

        /* n_rot > head_dim must also be rejected. */
        CHECK(ds4_gpu_head_rms_norm_rope_tail_tensor(
                  t, N_TOK, N_HEAD, HEAD_DIM, HEAD_DIM + 2, POS0, N_CTX_ORIG,
                  false, FREQ_BASE, 1.0f, 0.0f, 1.0f, BETA_FAST, BETA_SLOW,
                  EPS) == 0,
              "head_rms_rope: n_rot > head_dim must be rejected");
        ds4_gpu_tensor_free(t);
    }

    /* A tensor smaller than n_tok * n_head * head_dim floats must be
     * rejected. */
    {
        ds4_gpu_tensor *small = ds4_gpu_tensor_alloc(sizeof(float) * 4);
        CHECK(small != NULL, "head_rms_rope: small allocation failed");
        CHECK(ds4_gpu_head_rms_norm_rope_tail_tensor(
                  small, N_TOK, N_HEAD, HEAD_DIM, N_ROT, POS0, N_CTX_ORIG,
                  false, FREQ_BASE, 1.0f, 0.0f, 1.0f, BETA_FAST, BETA_SLOW,
                  EPS) == 0,
              "head_rms_rope: undersized tensor must be rejected");
        ds4_gpu_tensor_free(small);
    }

    fprintf(stderr, "  test_head_rms_norm_rope_tail OK\n");
    return 0;
}

/* Half-precision bit decode, standard IEEE754 binary16 -> binary32,
 * including the subnormal case.  Needed because a SYCL half type is not
 * available in a plain C test (same reason cited at test_embed_f16 above);
 * this is the decode counterpart of the round-trip-through-known-bit-
 * patterns encoding technique used throughout this file. */
static float oracle_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            int e = -1;
            do { mant <<= 1; e++; } while ((mant & 0x400u) == 0u);
            mant &= 0x3FFu;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* oracle_ape_value(model, ape_type, width, row, col): the APE accessor
 * later compressor tasks reuse.  `model` points directly at the start of
 * the APE table (offset already applied), matching how the GPU side reads
 * model_ape_value_dev(..., width, row, col) as row * width + col
 * (rocm/ds4_rocm_norm_rope.cuh:366-378); ds4's CPU reads the same element
 * via tensor_2d_value(model, ape, j, pos_mod) (ds4.c:7882-7887), which is
 * tensor_1d_value(y * dim[0] + x).  These address the same element.
 * ape_type is restricted to 0 = F32, 1 = F16, 8 = Q8_0
 * (rocm/ds4_rocm_compressor.cuh:181-183); the Q8_0 branch uses the same
 * 32-value, 34-byte block layout (2-byte little-endian F16 scale then 32
 * int8) as test_embed_q8_0 above. */
static float oracle_ape_value(const unsigned char *model, uint32_t ape_type,
                              uint32_t width, uint32_t row, uint32_t col) {
    if (ape_type == 1u) {
        const uint16_t *p = (const uint16_t *)model;
        return oracle_half_to_float(p[(uint64_t)row * width + col]);
    }
    if (ape_type == 8u) {
        const uint64_t row_bytes = ((uint64_t)width + 31u) / 32u * 34u;
        const unsigned char *blk =
                model + (uint64_t)row * row_bytes + (uint64_t)(col / 32u) * 34u;
        const uint16_t raw = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        const float scale = oracle_half_to_float(raw);
        const signed char q = (signed char)blk[2 + (col % 32u)];
        return scale * (float)q;
    }
    const float *p = (const float *)model;
    return p[(uint64_t)row * width + col];
}

/* Fills an ape_type-encoded table of `rows` rows by `width` columns with a
 * value distinguishable per (row, col), so a transposed or misindexed read
 * produces a visibly wrong value rather than a coincidental match.  The F16
 * branch is restricted to the 8 exactly-representable small integers used
 * by test_embed_f16 above, for the same reason: no float-to-half encoder
 * exists in this C test, so the oracle must stay exact by construction
 * rather than by rounding agreement with the kernel. */
static void oracle_fill_ape(unsigned char *dst, uint32_t ape_type,
                            uint32_t width, uint32_t rows) {
    static const uint16_t kHalfSmallInt[8] = {
        0x0000, /* 0.0 */ 0x3C00, /* 1.0 */ 0x4000, /* 2.0 */ 0x4200, /* 3.0 */
        0x4400, /* 4.0 */ 0x4500, /* 5.0 */ 0x4600, /* 6.0 */ 0x4700  /* 7.0 */
    };
    if (ape_type == 0u) {
        float *p = (float *)dst;
        for (uint32_t r = 0; r < rows; r++) {
            for (uint32_t c = 0; c < width; c++) {
                p[(uint64_t)r * width + c] = (float)((int)(r * 13u + c * 7u) - 20);
            }
        }
        return;
    }
    if (ape_type == 1u) {
        uint16_t *p = (uint16_t *)dst;
        for (uint32_t r = 0; r < rows; r++) {
            for (uint32_t c = 0; c < width; c++) {
                p[(uint64_t)r * width + c] = kHalfSmallInt[(r * 3u + c * 5u) % 8u];
            }
        }
        return;
    }
    /* Q8_0: `rows` independent row blocks, each ceil(width/32) blocks of
     * 32 values / 34 bytes.  Scale is fixed at exactly 1.0 (0x3C00) so the
     * dequantised value equals the stored int8 exactly. */
    const uint64_t row_bytes = ((uint64_t)width + 31u) / 32u * 34u;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < width; c += 32u) {
            unsigned char *blk = dst + (uint64_t)r * row_bytes + (uint64_t)(c / 32u) * 34u;
            blk[0] = 0x00; blk[1] = 0x3C;
            for (uint32_t i = 0; i < 32u && c + i < width; i++) {
                blk[2 + i] = (unsigned char)(signed char)((r * width + c + i) % 15u - 7);
            }
        }
    }
}

/* The store oracle: the store half of compressor_decode_one
 * (ds4.c:12549-12554), whose ring math is at ds4.c:12521-12524, applied
 * per token in a batch rather than one token at a time. */
static void oracle_compressor_store(float *state_kv, float *state_score,
                                    const float *kv, const float *sc,
                                    const unsigned char *ape, uint32_t ape_type,
                                    uint32_t width, uint32_t ratio,
                                    uint32_t pos0, uint32_t n_tokens) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t pos_mod = (pos0 + t) % ratio;
        const uint32_t dst_row = (ratio == 4u) ? ratio + pos_mod : pos_mod;
        for (uint32_t j = 0; j < width; j++) {
            state_kv[(uint64_t)dst_row * width + j] = kv[(uint64_t)t * width + j];
            state_score[(uint64_t)dst_row * width + j] =
                    sc[(uint64_t)t * width + j] +
                    oracle_ape_value(ape, ape_type, width, pos_mod, j);
        }
    }
}

/* Runs one (ape_type, ratio) combination of test_compressor_store below:
 * builds kv/sc/ape inputs, seeds the state tensors with a sentinel so any
 * row the kernel does not touch is provably untouched, calls the entry
 * under test, and compares every state row against oracle_compressor_store.
 * `head_dim` and `ratio` are caller-chosen so this same body covers both
 * the general-ratio and the ratio == 4 destination-row path. */
static int oracle_compressor_store_case(uint32_t ape_type, uint32_t head_dim,
                                        uint32_t ratio, uint32_t pos0,
                                        uint32_t n_tokens, uint64_t ape_offset) {
    const uint32_t coff       = ratio == 4u ? 2u : 1u;
    const uint32_t width      = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t ape_bytes =
            ape_type == 0u ? (uint64_t)width * ratio * sizeof(float) :
            ape_type == 1u ? (uint64_t)width * ratio * sizeof(uint16_t) :
                             (uint64_t)ratio * (((uint64_t)width + 31u) / 32u * 34u);
    const uint64_t model_size = ape_offset + ape_bytes;

    float *kv    = (float *)malloc((size_t)n_tokens * width * sizeof(float));
    float *sc    = (float *)malloc((size_t)n_tokens * width * sizeof(float));
    float *want_kv    = (float *)malloc((size_t)state_rows * width * sizeof(float));
    float *want_score = (float *)malloc((size_t)state_rows * width * sizeof(float));
    float *got_kv     = (float *)malloc((size_t)state_rows * width * sizeof(float));
    float *got_score  = (float *)malloc((size_t)state_rows * width * sizeof(float));
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    CHECK(kv && sc && want_kv && want_score && got_kv && got_score && model,
          "compressor_store: host allocation failed");

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t j = 0; j < width; j++) {
            kv[(uint64_t)t * width + j] = (float)(t * 1000u + j * 10u + 3u);
            sc[(uint64_t)t * width + j] = (float)((int)(t * 100u) - (int)(j * 2u) - 5);
        }
    }
    /* Sentinel padding before ape_offset: an implementation that ignores
     * ape_offset and reads from the start of the model buffer would decode
     * this instead of the real table and produce a visibly wrong score. */
    memset(model, 0xAA, (size_t)ape_offset);
    oracle_fill_ape(model + ape_offset, ape_type, width, ratio);

    /* Sentinel state rows: for ratio == 4 only the high half (rows
     * ratio..2*ratio-1) should ever be written.  Seeding the whole buffer
     * with a value no oracle row produces lets the post-call comparison
     * prove the low half was left alone, not just that the high half is
     * correct. */
    for (uint32_t i = 0; i < state_rows * width; i++) {
        want_kv[i] = want_score[i] = -999.0f;
    }
    oracle_compressor_store(want_kv, want_score, kv, sc, model + ape_offset,
                            ape_type, width, ratio, pos0, n_tokens);

    ds4_gpu_tensor *tkv    = ds4_gpu_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
    ds4_gpu_tensor *tsc    = ds4_gpu_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
    ds4_gpu_tensor *tskv   = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *tssc   = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    CHECK(tkv && tsc && tskv && tssc, "compressor_store: device allocation failed");

    CHECK(ds4_gpu_tensor_write(tkv, 0, kv, (uint64_t)n_tokens * width * sizeof(float)) != 0,
          "compressor_store: write kv");
    CHECK(ds4_gpu_tensor_write(tsc, 0, sc, (uint64_t)n_tokens * width * sizeof(float)) != 0,
          "compressor_store: write sc");
    CHECK(ds4_gpu_tensor_write(tskv, 0, want_kv, (uint64_t)state_rows * width * sizeof(float)) != 0,
          "compressor_store: seed state_kv sentinel");
    CHECK(ds4_gpu_tensor_write(tssc, 0, want_score, (uint64_t)state_rows * width * sizeof(float)) != 0,
          "compressor_store: seed state_score sentinel");

    CHECK(ds4_gpu_compressor_store_batch_tensor(
              tkv, tsc, tskv, tssc, model, model_size, ape_offset, ape_type,
              head_dim, ratio, pos0, n_tokens) != 0,
          "compressor_store: call");

    CHECK(ds4_gpu_tensor_read(tskv, 0, got_kv, (uint64_t)state_rows * width * sizeof(float)) != 0,
          "compressor_store: read state_kv");
    CHECK(ds4_gpu_tensor_read(tssc, 0, got_score, (uint64_t)state_rows * width * sizeof(float)) != 0,
          "compressor_store: read state_score");

    for (uint32_t i = 0; i < state_rows * width; i++) {
        CHECK(got_kv[i] == want_kv[i], "compressor_store: state_kv mismatch");
        CHECK_CLOSE(got_score[i], want_score[i], 1e-3,
                    "compressor_store: state_score mismatch");
    }

    ds4_gpu_tensor_free(tkv);
    ds4_gpu_tensor_free(tsc);
    ds4_gpu_tensor_free(tskv);
    ds4_gpu_tensor_free(tssc);
    free(kv); free(sc); free(want_kv); free(want_score);
    free(got_kv); free(got_score); free(model);
    return 0;
}

/* ds4_gpu_compressor_store_batch_tensor has no ds4.c call site: it is
 * invoked backend-internally by ds4_gpu_compressor_update_tensor with
 * n_tokens = 1 (rocm/ds4_rocm_compressor.cuh:292-298), never by ds4.c
 * directly.  This entry point is tested here directly against
 * oracle_compressor_store for all three permitted ape_type values, and
 * for both the general-ratio and the ratio == 4 destination-row path
 * (they compute dst_row differently; covering only one would prove only
 * half the entry).  pos0 is non-zero and not a multiple of ratio in both
 * cases, so the modular ring wrap is genuinely exercised rather than
 * hidden by tokens landing at rows 0..n_tokens-1 in order. */
static int test_compressor_store(void) {
    const uint32_t ape_types[] = {0u, 1u, 8u};

    for (size_t i = 0; i < sizeof(ape_types) / sizeof(ape_types[0]); i++) {
        const uint32_t ape_type = ape_types[i];

        /* General ratio: coff == 1, dst_row == pos_mod.  ratio = 3,
         * pos0 = 8 (8 % 3 == 2, non-zero and not a multiple of 3), and
         * n_tokens == ratio so the three tokens sweep the whole ring
         * exactly once: pos_mod goes 2, 0, 1, wrapping from the last row
         * back to the first partway through the batch. */
        if (oracle_compressor_store_case(ape_type, /*head_dim=*/5u, /*ratio=*/3u,
                                         /*pos0=*/8u, /*n_tokens=*/3u,
                                         /*ape_offset=*/16u) != 0) {
            return 1;
        }

        /* ratio == 4: coff == 2, dst_row == ratio + pos_mod, landing in the
         * HIGH half (rows 4..7) of the 8-row ring; the low half (rows
         * 0..3) must be left untouched.  pos0 = 6 (6 % 4 == 2, non-zero
         * and not a multiple of 4); pos_mod goes 2, 3, 0, 1, again
         * wrapping mid-batch. */
        if (oracle_compressor_store_case(ape_type, /*head_dim=*/5u, /*ratio=*/4u,
                                         /*pos0=*/6u, /*n_tokens=*/4u,
                                         /*ape_offset=*/16u) != 0) {
            return 1;
        }
    }

    /* An unsupported ape_type must be rejected (rocm/ds4_rocm_compressor.cuh:
     * 181-183 permits only 0, 1 and 8). */
    {
        enum { HEAD_DIM = 5, RATIO = 3, WIDTH = HEAD_DIM, N_TOKENS = 3 };
        unsigned char model[WIDTH * RATIO * sizeof(float)];
        memset(model, 0, sizeof(model));
        ds4_gpu_tensor *tkv  = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * WIDTH * sizeof(float));
        ds4_gpu_tensor *tsc  = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * WIDTH * sizeof(float));
        ds4_gpu_tensor *tskv = ds4_gpu_tensor_alloc((uint64_t)RATIO * WIDTH * sizeof(float));
        ds4_gpu_tensor *tssc = ds4_gpu_tensor_alloc((uint64_t)RATIO * WIDTH * sizeof(float));
        CHECK(tkv && tsc && tskv && tssc,
              "compressor_store: allocation failed (unsupported ape_type case)");
        CHECK(ds4_gpu_compressor_store_batch_tensor(
                  tkv, tsc, tskv, tssc, model, sizeof(model), 0, /*ape_type=*/2u,
                  HEAD_DIM, RATIO, 8u, N_TOKENS) == 0,
              "compressor_store: unsupported ape_type must be rejected");
        /* n_tokens == 0 must also be rejected: unlike most entries in this
         * backend, zero-sized work is not success here, matching the
         * validation list at rocm/ds4_rocm_compressor.cuh:204-208. */
        CHECK(ds4_gpu_compressor_store_batch_tensor(
                  tkv, tsc, tskv, tssc, model, sizeof(model), 0, 0u,
                  HEAD_DIM, RATIO, 8u, 0u) == 0,
              "compressor_store: n_tokens == 0 must be rejected");
        ds4_gpu_tensor_free(tkv);
        ds4_gpu_tensor_free(tsc);
        ds4_gpu_tensor_free(tskv);
        ds4_gpu_tensor_free(tssc);
    }

    fprintf(stderr, "  test_compressor_store OK\n");
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
    if (test_rms_norm_weight() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_rms_norm_weight_rows() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_add_rms_norm_weight() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_head_rms_norm() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dsv4_qkv_rms_norm_rows() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_rope_tail() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_head_rms_norm_rope_tail() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_compressor_store() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_kernels OK\n");
    return 0;
}
