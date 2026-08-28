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

/* The pool oracle: compressor_update_pool_kernel
 * (rocm/ds4_rocm_compressor.cuh:121-160), which matches
 * compressor_pool_decode_state (ds4.c:12452-12503) but expressed as an
 * unconditional softmax rather than the CPU's early-out on the sentinel
 * DS4_NEG_INF.  The `max_s > -INFINITY` guard below is NOT part of a
 * byte-for-byte port of the CUDA loop: without it, an all -inf candidate
 * set (the empty-softmax case this file tests) hits the IEEE `inf - inf`
 * indeterminate form and resolves to NaN instead of 0. */
static void oracle_compressor_pool(float *out, const float *state_kv,
                                   const float *state_score,
                                   uint32_t head_dim, uint32_t ratio) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;

    for (uint32_t d = 0; d < head_dim; d++) {
        float vals[16], scores[16];
        uint32_t n = 0;
        float max_s = -INFINITY;

        if (ratio == 4u) {
            for (uint32_t r = 0; r < 4u; r++) {
                vals[n] = state_kv[(uint64_t)r * width + d];
                scores[n] = state_score[(uint64_t)r * width + d];
                if (scores[n] > max_s) max_s = scores[n];
                n++;
            }
            for (uint32_t r = 0; r < 4u; r++) {
                vals[n] = state_kv[(uint64_t)(ratio + r) * width + head_dim + d];
                scores[n] = state_score[(uint64_t)(ratio + r) * width + head_dim + d];
                if (scores[n] > max_s) max_s = scores[n];
                n++;
            }
        } else {
            for (uint32_t r = 0; r < ratio; r++) {
                vals[n] = state_kv[(uint64_t)r * width + d];
                scores[n] = state_score[(uint64_t)r * width + d];
                if (scores[n] > max_s) max_s = scores[n];
                n++;
            }
        }

        float den = 0.0f, acc = 0.0f;
        if (max_s > -INFINITY) {
            for (uint32_t i = 0; i < n; i++) {
                const float w = expf(scores[i] - max_s);
                den += w;
                acc += vals[i] * w;
            }
        }
        out[d] = den != 0.0f ? acc / den : 0.0f;
    }
}

/* Step 1: the pool kernel driven in isolation through
 * ds4_gpu_compressor_update_tensor with state_already_stored = true and
 * n_rot = 0, so the store delegation and RoPE are both bypassed and only
 * pooling plus the RMS weight normalisation run.  The reference calls this
 * "the single most complex kernel in the four subsystems" and recommends
 * proving it before trusting any orchestration around it. */
static int test_compressor_pool(void) {
    const float RMS_EPS = 1e-5f;

    /* A1: general ratio (coff == 1), ratio = 3, head_dim = 4. */
    {
        enum { HEAD_DIM = 4, RATIO = 3, WIDTH = HEAD_DIM, STATE_ROWS = RATIO };
        float state_kv[STATE_ROWS * WIDTH], state_score[STATE_ROWS * WIDTH];
        float norm_w[HEAD_DIM], pool_want[HEAD_DIM], want[HEAD_DIM];

        for (uint32_t r = 0; r < STATE_ROWS; r++) {
            for (uint32_t c = 0; c < WIDTH; c++) {
                state_kv[r * WIDTH + c] = 200.0f + (float)r * 20.0f + (float)c * 3.0f;
                state_score[r * WIDTH + c] = 10.0f + (float)r * 4.0f + (float)c * 0.5f;
            }
        }
        for (uint32_t d = 0; d < HEAD_DIM; d++) norm_w[d] = (float)(d % 3 + 1) * 0.4f;

        oracle_compressor_pool(pool_want, state_kv, state_score, HEAD_DIM, RATIO);
        oracle_rms_norm_weight(want, pool_want, norm_w, HEAD_DIM, RMS_EPS);

        const uint64_t ape_offset = 16;
        const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
        const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
        unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
        CHECK(model != NULL, "compressor_pool: model allocation failed (A1)");
        memcpy(model + norm_offset, norm_w, sizeof(norm_w));

        float kv_cur[WIDTH] = {0}, sc_cur[WIDTH] = {0};
        float comp_sentinel[HEAD_DIM];
        for (uint32_t d = 0; d < HEAD_DIM; d++) comp_sentinel[d] = -999.0f;

        ds4_gpu_tensor *tkv_cur = ds4_gpu_tensor_alloc(sizeof(kv_cur));
        ds4_gpu_tensor *tsc_cur = ds4_gpu_tensor_alloc(sizeof(sc_cur));
        ds4_gpu_tensor *tskv    = ds4_gpu_tensor_alloc(sizeof(state_kv));
        ds4_gpu_tensor *tssc    = ds4_gpu_tensor_alloc(sizeof(state_score));
        ds4_gpu_tensor *tcomp   = ds4_gpu_tensor_alloc(sizeof(comp_sentinel));
        CHECK(tkv_cur && tsc_cur && tskv && tssc && tcomp,
              "compressor_pool: device allocation failed (A1)");

        CHECK(ds4_gpu_tensor_write(tkv_cur, 0, kv_cur, sizeof(kv_cur)) != 0, "compressor_pool: write kv_cur (A1)");
        CHECK(ds4_gpu_tensor_write(tsc_cur, 0, sc_cur, sizeof(sc_cur)) != 0, "compressor_pool: write sc_cur (A1)");
        CHECK(ds4_gpu_tensor_write(tskv, 0, state_kv, sizeof(state_kv)) != 0, "compressor_pool: write state_kv (A1)");
        CHECK(ds4_gpu_tensor_write(tssc, 0, state_score, sizeof(state_score)) != 0, "compressor_pool: write state_score (A1)");
        CHECK(ds4_gpu_tensor_write(tcomp, 0, comp_sentinel, sizeof(comp_sentinel)) != 0, "compressor_pool: write comp sentinel (A1)");

        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp,
                  model, model_size, ape_offset, /*ape_type=*/0u,
                  norm_offset, /*norm_type=*/0u, HEAD_DIM, RATIO,
                  /*pos=*/5u, /*comp_row=*/0u, /*n_rot=*/0u,
                  /*n_ctx_orig=*/4096u, /*freq_base=*/10000.0f, /*freq_scale=*/1.0f,
                  /*ext_factor=*/0.0f, /*attn_factor=*/1.0f, /*beta_fast=*/32.0f,
                  /*beta_slow=*/1.0f, RMS_EPS, /*state_already_stored=*/true,
                  /*decode_one_token=*/false, /*defer_finalize=*/false) != 0,
              "compressor_pool: call (A1)");

        float got[HEAD_DIM], got_kv[STATE_ROWS * WIDTH], got_score[STATE_ROWS * WIDTH];
        CHECK(ds4_gpu_tensor_read(tcomp, 0, got, sizeof(got)) != 0, "compressor_pool: read comp (A1)");
        CHECK(ds4_gpu_tensor_read(tskv, 0, got_kv, sizeof(got_kv)) != 0, "compressor_pool: read state_kv (A1)");
        CHECK(ds4_gpu_tensor_read(tssc, 0, got_score, sizeof(got_score)) != 0, "compressor_pool: read state_score (A1)");
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            CHECK_CLOSE(got[d], want[d], 1e-4, "compressor_pool: emitted value mismatch (A1)");
        }
        /* General ratio never shifts: the ring must be untouched. */
        for (uint32_t i = 0; i < STATE_ROWS * WIDTH; i++) {
            CHECK(got_kv[i] == state_kv[i], "compressor_pool: ring kv mutated (A1)");
            CHECK(got_score[i] == state_score[i], "compressor_pool: ring score mutated (A1)");
        }

        ds4_gpu_tensor_free(tkv_cur); ds4_gpu_tensor_free(tsc_cur);
        ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
        free(model);
    }

    /* A2: ratio == 4, the asymmetric-lane gather.  The low ring rows carry
     * one value range at column d (the low lane) and a POISONED,
     * unmistakably different value range at column head_dim + d (a lane a
     * correct low-row read never touches).  The high ring rows are the
     * mirror image: genuine data at head_dim + d, poison at d.  A port
     * that reads column d for both halves -- the sharpest trap in this
     * task -- would pick up the high rows' poison instead of their
     * genuine high-lane data, producing a visibly different, wrong
     * result. */
    {
        enum { HEAD_DIM = 3, RATIO = 4, WIDTH = HEAD_DIM * 2, STATE_ROWS = RATIO * 2 };
        float state_kv[STATE_ROWS * WIDTH], state_score[STATE_ROWS * WIDTH];

        for (uint32_t r = 0; r < 4u; r++) {
            for (uint32_t c = 0; c < WIDTH; c++) {
                state_kv[r * WIDTH + c] = 100.0f + (float)r * 10.0f + (float)c;
                state_score[r * WIDTH + c] = 5.0f + (float)r + 0.1f * (float)c;
            }
        }
        for (uint32_t r = 0; r < 4u; r++) {
            const uint32_t row = 4u + r;
            for (uint32_t c = 0; c < WIDTH; c++) {
                if (c < HEAD_DIM) {
                    state_kv[row * WIDTH + c] = -500.0f - (float)r - (float)c;
                    state_score[row * WIDTH + c] = -777.0f - (float)r - (float)c;
                } else {
                    const uint32_t d = c - HEAD_DIM;
                    state_kv[row * WIDTH + c] = 700.0f + (float)r * 10.0f + (float)d;
                    state_score[row * WIDTH + c] = 20.0f + (float)r + 0.1f * (float)d;
                }
            }
        }

        float norm_w[HEAD_DIM], pool_want[HEAD_DIM], want[HEAD_DIM];
        for (uint32_t d = 0; d < HEAD_DIM; d++) norm_w[d] = (float)(d + 1) * 0.5f;
        oracle_compressor_pool(pool_want, state_kv, state_score, HEAD_DIM, RATIO);
        oracle_rms_norm_weight(want, pool_want, norm_w, HEAD_DIM, RMS_EPS);

        const uint64_t ape_offset = 16;
        const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
        const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
        unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
        CHECK(model != NULL, "compressor_pool: model allocation failed (A2)");
        memcpy(model + norm_offset, norm_w, sizeof(norm_w));

        float kv_cur[WIDTH] = {0}, sc_cur[WIDTH] = {0};
        float comp_sentinel[HEAD_DIM];
        for (uint32_t d = 0; d < HEAD_DIM; d++) comp_sentinel[d] = -999.0f;

        ds4_gpu_tensor *tkv_cur = ds4_gpu_tensor_alloc(sizeof(kv_cur));
        ds4_gpu_tensor *tsc_cur = ds4_gpu_tensor_alloc(sizeof(sc_cur));
        ds4_gpu_tensor *tskv    = ds4_gpu_tensor_alloc(sizeof(state_kv));
        ds4_gpu_tensor *tssc    = ds4_gpu_tensor_alloc(sizeof(state_score));
        ds4_gpu_tensor *tcomp   = ds4_gpu_tensor_alloc(sizeof(comp_sentinel));
        CHECK(tkv_cur && tsc_cur && tskv && tssc && tcomp,
              "compressor_pool: device allocation failed (A2)");

        CHECK(ds4_gpu_tensor_write(tkv_cur, 0, kv_cur, sizeof(kv_cur)) != 0, "compressor_pool: write kv_cur (A2)");
        CHECK(ds4_gpu_tensor_write(tsc_cur, 0, sc_cur, sizeof(sc_cur)) != 0, "compressor_pool: write sc_cur (A2)");
        CHECK(ds4_gpu_tensor_write(tskv, 0, state_kv, sizeof(state_kv)) != 0, "compressor_pool: write state_kv (A2)");
        CHECK(ds4_gpu_tensor_write(tssc, 0, state_score, sizeof(state_score)) != 0, "compressor_pool: write state_score (A2)");
        CHECK(ds4_gpu_tensor_write(tcomp, 0, comp_sentinel, sizeof(comp_sentinel)) != 0, "compressor_pool: write comp sentinel (A2)");

        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp,
                  model, model_size, ape_offset, /*ape_type=*/0u,
                  norm_offset, /*norm_type=*/0u, HEAD_DIM, RATIO,
                  /*pos=*/7u, /*comp_row=*/0u, /*n_rot=*/0u,
                  /*n_ctx_orig=*/4096u, /*freq_base=*/10000.0f, /*freq_scale=*/1.0f,
                  /*ext_factor=*/0.0f, /*attn_factor=*/1.0f, /*beta_fast=*/32.0f,
                  /*beta_slow=*/1.0f, RMS_EPS, /*state_already_stored=*/true,
                  /*decode_one_token=*/false, /*defer_finalize=*/false) != 0,
              "compressor_pool: call (A2)");

        float got[HEAD_DIM], got_kv[STATE_ROWS * WIDTH], got_score[STATE_ROWS * WIDTH];
        CHECK(ds4_gpu_tensor_read(tcomp, 0, got, sizeof(got)) != 0, "compressor_pool: read comp (A2)");
        CHECK(ds4_gpu_tensor_read(tskv, 0, got_kv, sizeof(got_kv)) != 0, "compressor_pool: read state_kv (A2)");
        CHECK(ds4_gpu_tensor_read(tssc, 0, got_score, sizeof(got_score)) != 0, "compressor_pool: read state_score (A2)");
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            CHECK_CLOSE(got[d], want[d], 1e-4, "compressor_pool: emitted value mismatch (A2, ratio4 lanes)");
        }

        /* ratio == 4 always shifts on emit, even when the store step was
         * skipped: rows 0..3 must become the OLD rows 4..7 verbatim (full
         * width, both lanes, poison columns included), and rows 4..7 must
         * be unchanged (the shift copies the high half onto itself too). */
        for (uint32_t r = 0; r < 4u; r++) {
            for (uint32_t c = 0; c < WIDTH; c++) {
                const float want_kv_v = state_kv[(4u + r) * WIDTH + c];
                const float want_sc_v = state_score[(4u + r) * WIDTH + c];
                CHECK(got_kv[r * WIDTH + c] == want_kv_v, "compressor_pool: shifted low kv mismatch (A2)");
                CHECK(got_score[r * WIDTH + c] == want_sc_v, "compressor_pool: shifted low score mismatch (A2)");
                CHECK(got_kv[(4u + r) * WIDTH + c] == want_kv_v, "compressor_pool: high kv changed by shift (A2)");
                CHECK(got_score[(4u + r) * WIDTH + c] == want_sc_v, "compressor_pool: high score changed by shift (A2)");
            }
        }

        ds4_gpu_tensor_free(tkv_cur); ds4_gpu_tensor_free(tsc_cur);
        ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
        free(model);
    }

    /* A3: every candidate score is -inf.  den must resolve to exactly 0,
     * not NaN; a dropped guard here only manifests at sequence starts,
     * where the ring genuinely holds nothing yet. */
    {
        enum { HEAD_DIM = 2, RATIO = 3, WIDTH = HEAD_DIM, STATE_ROWS = RATIO };
        float state_kv[STATE_ROWS * WIDTH], state_score[STATE_ROWS * WIDTH];
        for (uint32_t i = 0; i < STATE_ROWS * WIDTH; i++) {
            state_kv[i] = 42.0f;
            state_score[i] = -INFINITY;
        }
        float norm_w[HEAD_DIM] = {1.0f, 1.0f};

        const uint64_t ape_offset = 16;
        const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
        const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
        unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
        CHECK(model != NULL, "compressor_pool: model allocation failed (A3)");
        memcpy(model + norm_offset, norm_w, sizeof(norm_w));

        float kv_cur[WIDTH] = {0}, sc_cur[WIDTH] = {0};
        float comp_sentinel[HEAD_DIM] = {-999.0f, -999.0f};

        ds4_gpu_tensor *tkv_cur = ds4_gpu_tensor_alloc(sizeof(kv_cur));
        ds4_gpu_tensor *tsc_cur = ds4_gpu_tensor_alloc(sizeof(sc_cur));
        ds4_gpu_tensor *tskv    = ds4_gpu_tensor_alloc(sizeof(state_kv));
        ds4_gpu_tensor *tssc    = ds4_gpu_tensor_alloc(sizeof(state_score));
        ds4_gpu_tensor *tcomp   = ds4_gpu_tensor_alloc(sizeof(comp_sentinel));
        CHECK(tkv_cur && tsc_cur && tskv && tssc && tcomp,
              "compressor_pool: device allocation failed (A3)");

        CHECK(ds4_gpu_tensor_write(tkv_cur, 0, kv_cur, sizeof(kv_cur)) != 0, "compressor_pool: write kv_cur (A3)");
        CHECK(ds4_gpu_tensor_write(tsc_cur, 0, sc_cur, sizeof(sc_cur)) != 0, "compressor_pool: write sc_cur (A3)");
        CHECK(ds4_gpu_tensor_write(tskv, 0, state_kv, sizeof(state_kv)) != 0, "compressor_pool: write state_kv (A3)");
        CHECK(ds4_gpu_tensor_write(tssc, 0, state_score, sizeof(state_score)) != 0, "compressor_pool: write state_score (A3)");
        CHECK(ds4_gpu_tensor_write(tcomp, 0, comp_sentinel, sizeof(comp_sentinel)) != 0, "compressor_pool: write comp sentinel (A3)");

        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp,
                  model, model_size, ape_offset, /*ape_type=*/0u,
                  norm_offset, /*norm_type=*/0u, HEAD_DIM, RATIO,
                  /*pos=*/2u, /*comp_row=*/0u, /*n_rot=*/0u,
                  /*n_ctx_orig=*/4096u, /*freq_base=*/10000.0f, /*freq_scale=*/1.0f,
                  /*ext_factor=*/0.0f, /*attn_factor=*/1.0f, /*beta_fast=*/32.0f,
                  /*beta_slow=*/1.0f, RMS_EPS, /*state_already_stored=*/true,
                  /*decode_one_token=*/false, /*defer_finalize=*/false) != 0,
              "compressor_pool: call (A3)");

        float got[HEAD_DIM];
        CHECK(ds4_gpu_tensor_read(tcomp, 0, got, sizeof(got)) != 0, "compressor_pool: read comp (A3)");
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            CHECK(got[d] == 0.0f, "compressor_pool: empty softmax must be exactly 0, not NaN (A3)");
        }

        ds4_gpu_tensor_free(tkv_cur); ds4_gpu_tensor_free(tsc_cur);
        ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
        free(model);
    }

    fprintf(stderr, "  test_compressor_pool OK\n");
    return 0;
}

/* Step 2: ds4_gpu_compressor_update_tensor is itself single-token, so it
 * compares 1:1 against compressor_decode_one (ds4.c:12507-12602) with no
 * loop needed.  Covers a non-emit position (store only), an emit boundary
 * (store, pool, RMS weight norm and RoPE tail together), and for ratio ==
 * 4 the post-emit shift that copies the high half of the ring into BOTH
 * halves (rocm/ds4_rocm_compressor.cuh:162-172).  Every case asserts the
 * full state ring after the call, not only the emitted row: a correct
 * emission with a corrupted ring would pass otherwise and break the next
 * token. */
static int test_compressor_update(void) {
    const float RMS_EPS = 1e-5f;
    const uint32_t N_CTX_ORIG = 4096u;
    const float FREQ_BASE = 10000.0f, FREQ_SCALE = 1.0f, EXT_FACTOR = 0.0f;
    const float ATTN_FACTOR = 1.0f, BETA_FAST = 32.0f, BETA_SLOW = 1.0f;

    /* B1: a non-emit position stores the current token into the ring and
     * returns success without touching comp_cache. */
    {
        enum { HEAD_DIM = 4, RATIO = 3, WIDTH = HEAD_DIM, STATE_ROWS = RATIO };
        const uint32_t POS = 3u; /* (3+1) % 3 == 1: not an emit boundary */
        float state_kv[STATE_ROWS * WIDTH], state_score[STATE_ROWS * WIDTH];
        float want_kv[STATE_ROWS * WIDTH], want_score[STATE_ROWS * WIDTH];
        float kv_cur[WIDTH], sc_cur[WIDTH];

        for (uint32_t r = 0; r < STATE_ROWS; r++) {
            for (uint32_t c = 0; c < WIDTH; c++) {
                state_kv[r * WIDTH + c] = 300.0f + (float)r * 20.0f + (float)c * 2.0f;
                state_score[r * WIDTH + c] = 8.0f + (float)r * 3.0f + (float)c * 0.2f;
            }
        }
        memcpy(want_kv, state_kv, sizeof(state_kv));
        memcpy(want_score, state_score, sizeof(state_score));
        for (uint32_t c = 0; c < WIDTH; c++) {
            kv_cur[c] = 777.0f + (float)c;
            sc_cur[c] = 333.0f + (float)c * 2.0f;
        }

        const uint64_t ape_offset = 16;
        const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
        const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
        unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
        CHECK(model != NULL, "compressor_update: model allocation failed (B1)");
        oracle_fill_ape(model + ape_offset, /*ape_type=*/0u, WIDTH, RATIO);
        float norm_w[HEAD_DIM];
        for (uint32_t d = 0; d < HEAD_DIM; d++) norm_w[d] = (float)(d + 1) * 0.25f;
        memcpy(model + norm_offset, norm_w, sizeof(norm_w));

        oracle_compressor_store(want_kv, want_score, kv_cur, sc_cur,
                                model + ape_offset, 0u, WIDTH, RATIO, POS, 1u);

        float comp_sentinel[HEAD_DIM];
        for (uint32_t d = 0; d < HEAD_DIM; d++) comp_sentinel[d] = -999.0f;

        ds4_gpu_tensor *tkv_cur = ds4_gpu_tensor_alloc(sizeof(kv_cur));
        ds4_gpu_tensor *tsc_cur = ds4_gpu_tensor_alloc(sizeof(sc_cur));
        ds4_gpu_tensor *tskv    = ds4_gpu_tensor_alloc(sizeof(state_kv));
        ds4_gpu_tensor *tssc    = ds4_gpu_tensor_alloc(sizeof(state_score));
        ds4_gpu_tensor *tcomp   = ds4_gpu_tensor_alloc(sizeof(comp_sentinel));
        CHECK(tkv_cur && tsc_cur && tskv && tssc && tcomp,
              "compressor_update: device allocation failed (B1)");

        CHECK(ds4_gpu_tensor_write(tkv_cur, 0, kv_cur, sizeof(kv_cur)) != 0, "compressor_update: write kv_cur (B1)");
        CHECK(ds4_gpu_tensor_write(tsc_cur, 0, sc_cur, sizeof(sc_cur)) != 0, "compressor_update: write sc_cur (B1)");
        CHECK(ds4_gpu_tensor_write(tskv, 0, state_kv, sizeof(state_kv)) != 0, "compressor_update: write state_kv (B1)");
        CHECK(ds4_gpu_tensor_write(tssc, 0, state_score, sizeof(state_score)) != 0, "compressor_update: write state_score (B1)");
        CHECK(ds4_gpu_tensor_write(tcomp, 0, comp_sentinel, sizeof(comp_sentinel)) != 0, "compressor_update: write comp sentinel (B1)");

        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp,
                  model, model_size, ape_offset, /*ape_type=*/0u,
                  norm_offset, /*norm_type=*/0u, HEAD_DIM, RATIO,
                  POS, /*comp_row=*/0u, /*n_rot=*/0u,
                  N_CTX_ORIG, FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR,
                  BETA_FAST, BETA_SLOW, RMS_EPS, /*state_already_stored=*/false,
                  /*decode_one_token=*/false, /*defer_finalize=*/false) != 0,
              "compressor_update: call (B1, non-emit)");

        float got_kv[STATE_ROWS * WIDTH], got_score[STATE_ROWS * WIDTH], got_comp[HEAD_DIM];
        CHECK(ds4_gpu_tensor_read(tskv, 0, got_kv, sizeof(got_kv)) != 0, "compressor_update: read state_kv (B1)");
        CHECK(ds4_gpu_tensor_read(tssc, 0, got_score, sizeof(got_score)) != 0, "compressor_update: read state_score (B1)");
        CHECK(ds4_gpu_tensor_read(tcomp, 0, got_comp, sizeof(got_comp)) != 0, "compressor_update: read comp (B1)");

        for (uint32_t i = 0; i < STATE_ROWS * WIDTH; i++) {
            CHECK(got_kv[i] == want_kv[i], "compressor_update: stored kv mismatch (B1)");
            CHECK_CLOSE(got_score[i], want_score[i], 1e-3, "compressor_update: stored score mismatch (B1)");
        }
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            CHECK(got_comp[d] == comp_sentinel[d], "compressor_update: comp_cache must be untouched on non-emit (B1)");
        }

        ds4_gpu_tensor_free(tkv_cur); ds4_gpu_tensor_free(tsc_cur);
        ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
        free(model);
    }

    /* B2: an emit boundary, general ratio.  Store, pool, RMS weight norm
     * and RoPE tail all run and are checked against one combined
     * oracle. */
    {
        enum { HEAD_DIM = 4, RATIO = 3, WIDTH = HEAD_DIM, STATE_ROWS = RATIO,
               N_ROT = 2, COMP_ROW = 1, COMP_ROWS = COMP_ROW + 1 };
        const uint32_t POS = 2u; /* (2+1) % 3 == 0: emit */
        float state_kv[STATE_ROWS * WIDTH], state_score[STATE_ROWS * WIDTH];
        float want_kv[STATE_ROWS * WIDTH], want_score[STATE_ROWS * WIDTH];
        float kv_cur[WIDTH], sc_cur[WIDTH];

        for (uint32_t r = 0; r < STATE_ROWS; r++) {
            for (uint32_t c = 0; c < WIDTH; c++) {
                state_kv[r * WIDTH + c] = 50.0f + (float)r * 15.0f + (float)c;
                state_score[r * WIDTH + c] = 3.0f + (float)r * 2.0f + (float)c * 0.3f;
            }
        }
        memcpy(want_kv, state_kv, sizeof(state_kv));
        memcpy(want_score, state_score, sizeof(state_score));
        for (uint32_t c = 0; c < WIDTH; c++) {
            kv_cur[c] = 900.0f + (float)c * 5.0f;
            sc_cur[c] = 60.0f - (float)c;
        }

        const uint64_t ape_offset = 24;
        const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
        const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
        unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
        CHECK(model != NULL, "compressor_update: model allocation failed (B2)");
        oracle_fill_ape(model + ape_offset, 0u, WIDTH, RATIO);
        float norm_w[HEAD_DIM];
        for (uint32_t d = 0; d < HEAD_DIM; d++) norm_w[d] = (float)(d % 4 + 1) * 0.35f;
        memcpy(model + norm_offset, norm_w, sizeof(norm_w));

        oracle_compressor_store(want_kv, want_score, kv_cur, sc_cur,
                                model + ape_offset, 0u, WIDTH, RATIO, POS, 1u);

        float pool_want[HEAD_DIM], normed_want[HEAD_DIM];
        oracle_compressor_pool(pool_want, want_kv, want_score, HEAD_DIM, RATIO);
        oracle_rms_norm_weight(normed_want, pool_want, norm_w, HEAD_DIM, RMS_EPS);
        const uint32_t comp_pos = POS + 1u - RATIO;
        oracle_rope_tail_row(normed_want, HEAD_DIM, N_ROT, comp_pos, N_CTX_ORIG,
                             FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR,
                             BETA_FAST, BETA_SLOW, /*inverse=*/0);

        float comp_sentinel[COMP_ROWS * HEAD_DIM];
        for (uint32_t i = 0; i < COMP_ROWS * HEAD_DIM; i++) comp_sentinel[i] = -999.0f;

        ds4_gpu_tensor *tkv_cur = ds4_gpu_tensor_alloc(sizeof(kv_cur));
        ds4_gpu_tensor *tsc_cur = ds4_gpu_tensor_alloc(sizeof(sc_cur));
        ds4_gpu_tensor *tskv    = ds4_gpu_tensor_alloc(sizeof(state_kv));
        ds4_gpu_tensor *tssc    = ds4_gpu_tensor_alloc(sizeof(state_score));
        ds4_gpu_tensor *tcomp   = ds4_gpu_tensor_alloc(sizeof(comp_sentinel));
        CHECK(tkv_cur && tsc_cur && tskv && tssc && tcomp,
              "compressor_update: device allocation failed (B2)");

        CHECK(ds4_gpu_tensor_write(tkv_cur, 0, kv_cur, sizeof(kv_cur)) != 0, "compressor_update: write kv_cur (B2)");
        CHECK(ds4_gpu_tensor_write(tsc_cur, 0, sc_cur, sizeof(sc_cur)) != 0, "compressor_update: write sc_cur (B2)");
        CHECK(ds4_gpu_tensor_write(tskv, 0, state_kv, sizeof(state_kv)) != 0, "compressor_update: write state_kv (B2)");
        CHECK(ds4_gpu_tensor_write(tssc, 0, state_score, sizeof(state_score)) != 0, "compressor_update: write state_score (B2)");
        CHECK(ds4_gpu_tensor_write(tcomp, 0, comp_sentinel, sizeof(comp_sentinel)) != 0, "compressor_update: write comp sentinel (B2)");

        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp,
                  model, model_size, ape_offset, /*ape_type=*/0u,
                  norm_offset, /*norm_type=*/0u, HEAD_DIM, RATIO,
                  POS, COMP_ROW, N_ROT,
                  N_CTX_ORIG, FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR,
                  BETA_FAST, BETA_SLOW, RMS_EPS, /*state_already_stored=*/false,
                  /*decode_one_token=*/false, /*defer_finalize=*/false) != 0,
              "compressor_update: call (B2, emit)");

        float got_kv[STATE_ROWS * WIDTH], got_score[STATE_ROWS * WIDTH];
        float got_comp[COMP_ROWS * HEAD_DIM];
        CHECK(ds4_gpu_tensor_read(tskv, 0, got_kv, sizeof(got_kv)) != 0, "compressor_update: read state_kv (B2)");
        CHECK(ds4_gpu_tensor_read(tssc, 0, got_score, sizeof(got_score)) != 0, "compressor_update: read state_score (B2)");
        CHECK(ds4_gpu_tensor_read(tcomp, 0, got_comp, sizeof(got_comp)) != 0, "compressor_update: read comp (B2)");

        for (uint32_t i = 0; i < STATE_ROWS * WIDTH; i++) {
            CHECK(got_kv[i] == want_kv[i], "compressor_update: post-emit ring kv mismatch (B2)");
            CHECK_CLOSE(got_score[i], want_score[i], 1e-3, "compressor_update: post-emit ring score mismatch (B2)");
        }
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            CHECK_CLOSE(got_comp[COMP_ROW * HEAD_DIM + d], normed_want[d], 1e-3,
                       "compressor_update: emitted row mismatch (B2)");
        }
        for (uint32_t i = 0; i < COMP_ROW * HEAD_DIM; i++) {
            CHECK(got_comp[i] == comp_sentinel[i], "compressor_update: rows before comp_row must be untouched (B2)");
        }

        ds4_gpu_tensor_free(tkv_cur); ds4_gpu_tensor_free(tsc_cur);
        ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
        free(model);
    }

    /* B3: ratio == 4, exercising the full store -> pool -> norm -> RoPE
     * chain together with the post-emit shift.  The low ring rows and the
     * high ring rows (and, within the high rows, the low vs. high lane
     * columns) all carry visibly different values so a lane or row swap
     * anywhere in the pipeline produces a detectably wrong answer, and the
     * ring is checked in full afterward, not just the emitted row. */
    {
        enum { HEAD_DIM = 4, RATIO = 4, WIDTH = HEAD_DIM * 2, STATE_ROWS = RATIO * 2,
               N_ROT = 2, COMP_ROW = 0, COMP_ROWS = COMP_ROW + 1 };
        const uint32_t POS = 7u; /* pos % 4 == 3: last slot of the window; (7+1)%4==0: emit */
        float state_kv[STATE_ROWS * WIDTH], state_score[STATE_ROWS * WIDTH];
        float want_kv[STATE_ROWS * WIDTH], want_score[STATE_ROWS * WIDTH];
        float kv_cur[WIDTH], sc_cur[WIDTH];

        /* Low ring (rows 0..3): previous window.  Genuine data at column
         * d < HEAD_DIM; poison at head_dim + d, which a correct low-row
         * read never touches. */
        for (uint32_t r = 0; r < 4u; r++) {
            for (uint32_t c = 0; c < WIDTH; c++) {
                if (c < HEAD_DIM) {
                    state_kv[r * WIDTH + c] = 100.0f + (float)r * 10.0f + (float)c;
                    state_score[r * WIDTH + c] = 5.0f + (float)r + 0.1f * (float)c;
                } else {
                    const uint32_t d = c - HEAD_DIM;
                    state_kv[r * WIDTH + c] = -300.0f - (float)r - (float)d;
                    state_score[r * WIDTH + c] = -666.0f - (float)r - (float)d;
                }
            }
        }
        /* High ring (rows 4..6): the first three tokens of the current
         * window, already stored.  Poison at column d < HEAD_DIM, genuine
         * data at head_dim + d. */
        for (uint32_t r = 0; r < 3u; r++) {
            const uint32_t row = 4u + r;
            for (uint32_t c = 0; c < WIDTH; c++) {
                if (c < HEAD_DIM) {
                    state_kv[row * WIDTH + c] = -500.0f - (float)r - (float)c;
                    state_score[row * WIDTH + c] = -777.0f - (float)r - (float)c;
                } else {
                    const uint32_t d = c - HEAD_DIM;
                    state_kv[row * WIDTH + c] = 700.0f + (float)r * 10.0f + (float)d;
                    state_score[row * WIDTH + c] = 20.0f + (float)r + 0.1f * (float)d;
                }
            }
        }
        /* Row 7 (the fourth slot of the window) is not pre-seeded: this
         * call's store fills it from kv_cur/sc_cur. */
        for (uint32_t c = 0; c < WIDTH; c++) {
            state_kv[7 * WIDTH + c] = 0.0f;
            state_score[7 * WIDTH + c] = 0.0f;
        }
        memcpy(want_kv, state_kv, sizeof(state_kv));
        memcpy(want_score, state_score, sizeof(state_score));
        for (uint32_t c = 0; c < WIDTH; c++) {
            kv_cur[c] = c < HEAD_DIM ? (-800.0f - (float)c) : (750.0f + (float)(c - HEAD_DIM));
            sc_cur[c] = c < HEAD_DIM ? (-888.0f - (float)c) : (25.0f + (float)(c - HEAD_DIM));
        }

        const uint64_t ape_offset = 32;
        const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
        const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
        unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
        CHECK(model != NULL, "compressor_update: model allocation failed (B3)");
        oracle_fill_ape(model + ape_offset, 0u, WIDTH, RATIO);
        float norm_w[HEAD_DIM];
        for (uint32_t d = 0; d < HEAD_DIM; d++) norm_w[d] = (float)(d + 2) * 0.3f;
        memcpy(model + norm_offset, norm_w, sizeof(norm_w));

        oracle_compressor_store(want_kv, want_score, kv_cur, sc_cur,
                                model + ape_offset, 0u, WIDTH, RATIO, POS, 1u);

        float pool_want[HEAD_DIM], normed_want[HEAD_DIM];
        oracle_compressor_pool(pool_want, want_kv, want_score, HEAD_DIM, RATIO);
        oracle_rms_norm_weight(normed_want, pool_want, norm_w, HEAD_DIM, RMS_EPS);
        const uint32_t comp_pos = POS + 1u - RATIO;
        oracle_rope_tail_row(normed_want, HEAD_DIM, N_ROT, comp_pos, N_CTX_ORIG,
                             FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR,
                             BETA_FAST, BETA_SLOW, /*inverse=*/0);

        /* The expected post-shift ring: rows 0..3 become the OLD rows
         * 4..7 (i.e. want_kv/want_score after the store above, before any
         * shift); rows 4..7 stay the same. */
        float want_shifted_kv[STATE_ROWS * WIDTH], want_shifted_score[STATE_ROWS * WIDTH];
        memcpy(want_shifted_kv, want_kv, sizeof(want_kv));
        memcpy(want_shifted_score, want_score, sizeof(want_score));
        for (uint32_t r = 0; r < 4u; r++) {
            memcpy(&want_shifted_kv[r * WIDTH], &want_kv[(4u + r) * WIDTH], WIDTH * sizeof(float));
            memcpy(&want_shifted_score[r * WIDTH], &want_score[(4u + r) * WIDTH], WIDTH * sizeof(float));
        }

        float comp_sentinel[COMP_ROWS * HEAD_DIM];
        for (uint32_t i = 0; i < COMP_ROWS * HEAD_DIM; i++) comp_sentinel[i] = -999.0f;

        ds4_gpu_tensor *tkv_cur = ds4_gpu_tensor_alloc(sizeof(kv_cur));
        ds4_gpu_tensor *tsc_cur = ds4_gpu_tensor_alloc(sizeof(sc_cur));
        ds4_gpu_tensor *tskv    = ds4_gpu_tensor_alloc(sizeof(state_kv));
        ds4_gpu_tensor *tssc    = ds4_gpu_tensor_alloc(sizeof(state_score));
        ds4_gpu_tensor *tcomp   = ds4_gpu_tensor_alloc(sizeof(comp_sentinel));
        CHECK(tkv_cur && tsc_cur && tskv && tssc && tcomp,
              "compressor_update: device allocation failed (B3)");

        CHECK(ds4_gpu_tensor_write(tkv_cur, 0, kv_cur, sizeof(kv_cur)) != 0, "compressor_update: write kv_cur (B3)");
        CHECK(ds4_gpu_tensor_write(tsc_cur, 0, sc_cur, sizeof(sc_cur)) != 0, "compressor_update: write sc_cur (B3)");
        CHECK(ds4_gpu_tensor_write(tskv, 0, state_kv, sizeof(state_kv)) != 0, "compressor_update: write state_kv (B3)");
        CHECK(ds4_gpu_tensor_write(tssc, 0, state_score, sizeof(state_score)) != 0, "compressor_update: write state_score (B3)");
        CHECK(ds4_gpu_tensor_write(tcomp, 0, comp_sentinel, sizeof(comp_sentinel)) != 0, "compressor_update: write comp sentinel (B3)");

        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp,
                  model, model_size, ape_offset, /*ape_type=*/0u,
                  norm_offset, /*norm_type=*/0u, HEAD_DIM, RATIO,
                  POS, COMP_ROW, N_ROT,
                  N_CTX_ORIG, FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR,
                  BETA_FAST, BETA_SLOW, RMS_EPS, /*state_already_stored=*/false,
                  /*decode_one_token=*/false, /*defer_finalize=*/false) != 0,
              "compressor_update: call (B3, ratio4 emit + shift)");

        float got_kv[STATE_ROWS * WIDTH], got_score[STATE_ROWS * WIDTH];
        float got_comp[COMP_ROWS * HEAD_DIM];
        CHECK(ds4_gpu_tensor_read(tskv, 0, got_kv, sizeof(got_kv)) != 0, "compressor_update: read state_kv (B3)");
        CHECK(ds4_gpu_tensor_read(tssc, 0, got_score, sizeof(got_score)) != 0, "compressor_update: read state_score (B3)");
        CHECK(ds4_gpu_tensor_read(tcomp, 0, got_comp, sizeof(got_comp)) != 0, "compressor_update: read comp (B3)");

        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            CHECK_CLOSE(got_comp[COMP_ROW * HEAD_DIM + d], normed_want[d], 1e-3,
                       "compressor_update: emitted row mismatch (B3, ratio4)");
        }
        for (uint32_t i = 0; i < STATE_ROWS * WIDTH; i++) {
            CHECK(got_kv[i] == want_shifted_kv[i], "compressor_update: post-shift ring kv mismatch (B3)");
            CHECK_CLOSE(got_score[i], want_shifted_score[i], 1e-3, "compressor_update: post-shift ring score mismatch (B3)");
        }

        ds4_gpu_tensor_free(tkv_cur); ds4_gpu_tensor_free(tsc_cur);
        ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
        free(model);
    }

    /* Validation: an odd n_rot, n_rot > head_dim, and a non-zero norm_type
     * must all be rejected before any kernel launches
     * (rocm/ds4_rocm_compressor.cuh:272-279). */
    {
        enum { HEAD_DIM = 4, RATIO = 3, WIDTH = HEAD_DIM, STATE_ROWS = RATIO };
        float state_kv[STATE_ROWS * WIDTH] = {0}, state_score[STATE_ROWS * WIDTH] = {0};
        float kv_cur[WIDTH] = {0}, sc_cur[WIDTH] = {0}, comp[HEAD_DIM] = {0};
        const uint64_t ape_offset = 16;
        const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
        const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
        unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
        CHECK(model != NULL, "compressor_update: model allocation failed (validation)");

        ds4_gpu_tensor *tkv_cur = ds4_gpu_tensor_alloc(sizeof(kv_cur));
        ds4_gpu_tensor *tsc_cur = ds4_gpu_tensor_alloc(sizeof(sc_cur));
        ds4_gpu_tensor *tskv    = ds4_gpu_tensor_alloc(sizeof(state_kv));
        ds4_gpu_tensor *tssc    = ds4_gpu_tensor_alloc(sizeof(state_score));
        ds4_gpu_tensor *tcomp   = ds4_gpu_tensor_alloc(sizeof(comp));
        CHECK(tkv_cur && tsc_cur && tskv && tssc && tcomp,
              "compressor_update: device allocation failed (validation)");
        CHECK(ds4_gpu_tensor_write(tskv, 0, state_kv, sizeof(state_kv)) != 0, "compressor_update: write state_kv (validation)");
        CHECK(ds4_gpu_tensor_write(tssc, 0, state_score, sizeof(state_score)) != 0, "compressor_update: write state_score (validation)");

        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp, model, model_size,
                  ape_offset, 0u, norm_offset, 0u, HEAD_DIM, RATIO,
                  /*pos=*/2u, /*comp_row=*/0u, /*n_rot=*/1u, N_CTX_ORIG,
                  FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST,
                  BETA_SLOW, RMS_EPS, true, false, false) == 0,
              "compressor_update: odd n_rot must be rejected");
        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp, model, model_size,
                  ape_offset, 0u, norm_offset, 0u, HEAD_DIM, RATIO,
                  /*pos=*/2u, /*comp_row=*/0u, /*n_rot=*/HEAD_DIM + 2u, N_CTX_ORIG,
                  FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST,
                  BETA_SLOW, RMS_EPS, true, false, false) == 0,
              "compressor_update: n_rot > head_dim must be rejected");
        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp, model, model_size,
                  ape_offset, 0u, norm_offset, /*norm_type=*/1u, HEAD_DIM, RATIO,
                  /*pos=*/2u, /*comp_row=*/0u, /*n_rot=*/0u, N_CTX_ORIG,
                  FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST,
                  BETA_SLOW, RMS_EPS, true, false, false) == 0,
              "compressor_update: non-zero norm_type must be rejected");

        ds4_gpu_tensor_free(tkv_cur); ds4_gpu_tensor_free(tsc_cur);
        ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
        free(model);
    }

    fprintf(stderr, "  test_compressor_update OK\n");
    return 0;
}

/* Mirrors compressor_finish_prefill_state_cpu, ds4.c:12407-12427: after a
 * from-scratch prefill, clear whichever ring rows the trailing partial
 * window's per-token stores never touched, so decode resumes from the same
 * partial-window state the streaming path would have produced.  Rows the
 * tail actually wrote are left alone.  ds4.c's function has no pos0
 * parameter: it is only ever reached with pos0 == 0 (ds4.c:28119's
 * zero_prefix gate is a precondition for calling the batched GPU prefill
 * entry this test drives), so this oracle assumes pos0 == 0 too. */
static void oracle_compressor_finish_prefill(float *state_kv, float *state_score,
                                             uint32_t head_dim, uint32_t ratio,
                                             uint32_t n_tokens) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t rem = n_tokens % ratio;
    const uint32_t clear_start = ratio == 4u ? ratio + rem : rem;
    const uint32_t clear_end = ratio == 4u ? 2u * ratio : ratio;
    for (uint32_t row = clear_start; row < clear_end; row++) {
        for (uint32_t j = 0; j < width; j++) {
            state_kv[(uint64_t)row * width + j] = 0.0f;
            state_score[(uint64_t)row * width + j] = -INFINITY;
        }
    }
}

/* The central deliverable tested here: ds4_gpu_compressor_prefill_tensor has no
 * CPU counterpart with a matching signature.  ds4's own CPU prefill never
 * batches the compressor either: it always loops per token
 * (prefill_layer_major_cpu, ds4.c:13757, reaching compressor_decode_one via
 * ds4.c:13107, :13125, :13342, :13360).  So the oracle here IS that
 * per-token loop, built entirely from the already-validated oracles
 * above (oracle_compressor_store, oracle_compressor_pool,
 * oracle_rms_norm_weight, oracle_rope_tail_row), run once per position from
 * pos0 == 0 to n_tokens - 1 with a fresh state ring (state_kv zeroed,
 * state_score -inf, matching rocm/ds4_rocm_compressor.cuh:384-387), taking
 * the emitted row at every ratio boundary and applying the ratio == 4 shift
 * exactly as oracle_compressor_pool's caller does above.  pos0 == 0 matches
 * the only configuration ds4.c ever calls this entry with (see the finish
 * oracle's comment above), which is what lets the final ring be compared
 * against oracle_compressor_finish_prefill directly.
 *
 * ape_type is fixed at F32 (0): the batched entry's ape/norm handling is
 * identical to the already-covered store and update entries regardless of
 * ape_type, so re-sweeping all three here would not exercise anything new. */
static int run_compressor_prefill_case(uint32_t head_dim, uint32_t ratio,
                                       uint32_t n_tokens, uint32_t n_rot,
                                       const char *tag) {
    const float RMS_EPS = 1e-5f;
    const uint32_t N_CTX_ORIG = 4096u;
    const float FREQ_BASE = 10000.0f, FREQ_SCALE = 1.0f, EXT_FACTOR = 0.0f;
    const float ATTN_FACTOR = 1.0f, BETA_FAST = 32.0f, BETA_SLOW = 1.0f;

    const uint32_t coff       = ratio == 4u ? 2u : 1u;
    const uint32_t width      = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp     = n_tokens / ratio;

    float *kv = (float *)malloc((size_t)n_tokens * width * sizeof(float));
    float *sc = (float *)malloc((size_t)n_tokens * width * sizeof(float));
    float *norm_w = (float *)malloc((size_t)head_dim * sizeof(float));
    float *state_kv = (float *)malloc((size_t)state_rows * width * sizeof(float));
    float *state_score = (float *)malloc((size_t)state_rows * width * sizeof(float));
    float *comp_want = n_comp ? (float *)malloc((size_t)n_comp * head_dim * sizeof(float)) : NULL;
    CHECK(kv && sc && norm_w && state_kv && state_score && (n_comp == 0 || comp_want),
          "compressor_prefill: host allocation failed");

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t j = 0; j < width; j++) {
            kv[(uint64_t)t * width + j] = (float)(t * 37u + j * 11u + 5u) * 0.25f;
            sc[(uint64_t)t * width + j] = (float)((int)(t * 19u) - (int)(j * 3u) - 7);
        }
    }
    for (uint32_t d = 0; d < head_dim; d++) norm_w[d] = (float)(d % 5u + 1u) * 0.2f;

    const uint64_t ape_offset = 16;
    const uint64_t norm_offset = ape_offset + (uint64_t)width * ratio * sizeof(float);
    const uint64_t model_size = norm_offset + (uint64_t)head_dim * sizeof(float);
    unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
    CHECK(model != NULL, "compressor_prefill: model allocation failed");
    oracle_fill_ape(model + ape_offset, /*ape_type=*/0u, width, ratio);
    memcpy(model + norm_offset, norm_w, (size_t)head_dim * sizeof(float));

    /* Step 1: fresh ring, asymmetric init. */
    for (uint32_t i = 0; i < state_rows * width; i++) {
        state_kv[i] = 0.0f;
        state_score[i] = -INFINITY;
    }

    uint32_t emitted = 0;
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        oracle_compressor_store(state_kv, state_score, kv + (uint64_t)pos * width,
                                sc + (uint64_t)pos * width, model + ape_offset,
                                /*ape_type=*/0u, width, ratio, pos, 1u);
        if (((pos + 1u) % ratio) != 0u) continue;

        float pool_want[256], normed_want[256];
        oracle_compressor_pool(pool_want, state_kv, state_score, head_dim, ratio);
        oracle_rms_norm_weight(normed_want, pool_want, norm_w, (int)head_dim, RMS_EPS);
        const uint32_t comp_pos = pos + 1u - ratio;
        oracle_rope_tail_row(normed_want, head_dim, n_rot, comp_pos, N_CTX_ORIG,
                             FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR,
                             BETA_FAST, BETA_SLOW, /*inverse=*/0);
        memcpy(comp_want + (uint64_t)emitted * head_dim, normed_want,
               (size_t)head_dim * sizeof(float));
        emitted++;

        if (ratio == 4u) {
            for (uint32_t r = 0; r < 4u; r++) {
                memcpy(&state_kv[(uint64_t)r * width], &state_kv[(uint64_t)(4u + r) * width],
                       (size_t)width * sizeof(float));
                memcpy(&state_score[(uint64_t)r * width], &state_score[(uint64_t)(4u + r) * width],
                       (size_t)width * sizeof(float));
            }
            for (uint32_t r = 0; r < 4u; r++) {
                memcpy(&state_kv[(uint64_t)(4u + r) * width], &state_kv[(uint64_t)r * width],
                       (size_t)width * sizeof(float));
                memcpy(&state_score[(uint64_t)(4u + r) * width], &state_score[(uint64_t)r * width],
                       (size_t)width * sizeof(float));
            }
        }
    }
    CHECK(emitted == n_comp, "compressor_prefill: oracle emit count mismatch");

    /* Post-prefill state cleanup: clear whatever the tail never wrote. */
    oracle_compressor_finish_prefill(state_kv, state_score, head_dim, ratio, n_tokens);

    ds4_gpu_tensor *tkv  = ds4_gpu_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
    ds4_gpu_tensor *tsc  = ds4_gpu_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
    ds4_gpu_tensor *tskv = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *tssc = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *tcomp = ds4_gpu_tensor_alloc((uint64_t)(n_comp ? n_comp : 1u) * head_dim * sizeof(float));
    CHECK(tkv && tsc && tskv && tssc && tcomp, "compressor_prefill: device allocation failed");

    CHECK(ds4_gpu_tensor_write(tkv, 0, kv, (uint64_t)n_tokens * width * sizeof(float)) != 0,
          "compressor_prefill: write kv");
    CHECK(ds4_gpu_tensor_write(tsc, 0, sc, (uint64_t)n_tokens * width * sizeof(float)) != 0,
          "compressor_prefill: write sc");

    /* Sentinel-seed the state and comp_cache tensors: the entry must
     * reinitialise state itself (step 1), and must leave comp_cache alone
     * beyond row n_comp when n_comp == 0. */
    {
        float *sentinel_state = (float *)malloc((size_t)state_rows * width * sizeof(float));
        CHECK(sentinel_state != NULL, "compressor_prefill: sentinel allocation failed");
        for (uint32_t i = 0; i < state_rows * width; i++) sentinel_state[i] = 123456.0f;
        CHECK(ds4_gpu_tensor_write(tskv, 0, sentinel_state, (uint64_t)state_rows * width * sizeof(float)) != 0,
              "compressor_prefill: seed state_kv sentinel");
        CHECK(ds4_gpu_tensor_write(tssc, 0, sentinel_state, (uint64_t)state_rows * width * sizeof(float)) != 0,
              "compressor_prefill: seed state_score sentinel");
        free(sentinel_state);
    }
    {
        const uint32_t comp_alloc_rows = n_comp ? n_comp : 1u;
        float *sentinel_comp = (float *)malloc((size_t)comp_alloc_rows * head_dim * sizeof(float));
        CHECK(sentinel_comp != NULL, "compressor_prefill: comp sentinel allocation failed");
        for (uint32_t i = 0; i < comp_alloc_rows * head_dim; i++) sentinel_comp[i] = -999.0f;
        CHECK(ds4_gpu_tensor_write(tcomp, 0, sentinel_comp, (uint64_t)comp_alloc_rows * head_dim * sizeof(float)) != 0,
              "compressor_prefill: seed comp sentinel");
        free(sentinel_comp);
    }

    CHECK(ds4_gpu_compressor_prefill_tensor(
              tcomp, tskv, tssc, tkv, tsc,
              model, model_size, ape_offset, /*ape_type=*/0u,
              norm_offset, /*norm_type=*/0u, head_dim, ratio,
              /*pos0=*/0u, n_tokens, n_rot, N_CTX_ORIG,
              /*quantize_fp8=*/false, FREQ_BASE, FREQ_SCALE, EXT_FACTOR,
              ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) != 0,
          "compressor_prefill: call");

    float *got_kv = (float *)malloc((size_t)state_rows * width * sizeof(float));
    float *got_score = (float *)malloc((size_t)state_rows * width * sizeof(float));
    float *got_comp = n_comp ? (float *)malloc((size_t)n_comp * head_dim * sizeof(float)) : NULL;
    CHECK(got_kv && got_score && (n_comp == 0 || got_comp), "compressor_prefill: readback allocation failed");
    CHECK(ds4_gpu_tensor_read(tskv, 0, got_kv, (uint64_t)state_rows * width * sizeof(float)) != 0,
          "compressor_prefill: read state_kv");
    CHECK(ds4_gpu_tensor_read(tssc, 0, got_score, (uint64_t)state_rows * width * sizeof(float)) != 0,
          "compressor_prefill: read state_score");
    if (n_comp) {
        CHECK(ds4_gpu_tensor_read(tcomp, 0, got_comp, (uint64_t)n_comp * head_dim * sizeof(float)) != 0,
              "compressor_prefill: read comp");
    }

    for (uint32_t i = 0; i < n_comp * head_dim; i++) {
        CHECK_CLOSE(got_comp[i], comp_want[i], 1e-2, "compressor_prefill: emitted row mismatch");
    }
    for (uint32_t i = 0; i < state_rows * width; i++) {
        if (state_score[i] == -INFINITY) {
            CHECK(got_score[i] == -INFINITY, "compressor_prefill: cleared score row mismatch");
            CHECK(got_kv[i] == 0.0f, "compressor_prefill: cleared kv row mismatch");
        } else {
            CHECK(got_kv[i] == state_kv[i], "compressor_prefill: final ring kv mismatch");
            CHECK_CLOSE(got_score[i], state_score[i], 1e-3, "compressor_prefill: final ring score mismatch");
        }
    }

    fprintf(stderr, "  run_compressor_prefill_case(%s) OK\n", tag);

    ds4_gpu_tensor_free(tkv); ds4_gpu_tensor_free(tsc);
    ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
    free(kv); free(sc); free(norm_w); free(state_kv); free(state_score);
    free(comp_want); free(got_kv); free(got_score); free(got_comp);
    free(model);
    return 0;
}

/* Step 4's central deliverable: the batched prefill entry, checked against
 * the per-token driver above.  head_dim = 4 keeps N_ROT = 2 valid (n_rot
 * must be even and <= head_dim) while staying small enough for the 256-wide
 * scratch buffers in run_compressor_prefill_case. */
static int test_compressor_prefill(void) {
    /* General ratio: n_tokens = 7, ratio = 3 -> n_comp = 2, rem = 1 != 0,
     * so the trailing partial window's tail-seed branch runs. */
    if (run_compressor_prefill_case(/*head_dim=*/4u, /*ratio=*/3u,
                                    /*n_tokens=*/7u, /*n_rot=*/2u,
                                    "general ratio, rem != 0") != 0) {
        return 1;
    }

    /* ratio == 4: n_tokens = 11 -> n_comp = 2, cutoff = 8 >= ratio (the
     * PREVIOUS-window seed branch runs, pulling window 1's raw tokens
     * rather than window 0's), and rem = 3 != 0 (the CURRENT-window tail
     * seed branch also runs, into rows 4..6, leaving row 7 to be cleared). */
    if (run_compressor_prefill_case(/*head_dim=*/4u, /*ratio=*/4u,
                                    /*n_tokens=*/11u, /*n_rot=*/2u,
                                    "ratio == 4, cutoff >= ratio and rem != 0") != 0) {
        return 1;
    }

    /* n_comp == 0: n_tokens = 2 < ratio = 3.  The entry must seed state and
     * return success without ever touching comp_cache. */
    if (run_compressor_prefill_case(/*head_dim=*/4u, /*ratio=*/3u,
                                    /*n_tokens=*/2u, /*n_rot=*/2u,
                                    "n_comp == 0") != 0) {
        return 1;
    }

    /* Validation and failure-propagation checks. */
    {
        enum { HEAD_DIM = 4, RATIO = 3, WIDTH = HEAD_DIM, STATE_ROWS = RATIO, N_TOKENS = 6 };
        float kv[N_TOKENS * WIDTH] = {0}, sc[N_TOKENS * WIDTH] = {0};
        const uint64_t ape_offset = 16;
        const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
        const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
        unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
        CHECK(model != NULL, "compressor_prefill: model allocation failed (validation)");

        ds4_gpu_tensor *tkv  = ds4_gpu_tensor_alloc(sizeof(kv));
        ds4_gpu_tensor *tsc  = ds4_gpu_tensor_alloc(sizeof(sc));
        ds4_gpu_tensor *tskv = ds4_gpu_tensor_alloc((uint64_t)STATE_ROWS * WIDTH * sizeof(float));
        ds4_gpu_tensor *tssc = ds4_gpu_tensor_alloc((uint64_t)STATE_ROWS * WIDTH * sizeof(float));
        ds4_gpu_tensor *tcomp = ds4_gpu_tensor_alloc((uint64_t)(N_TOKENS / RATIO) * HEAD_DIM * sizeof(float));
        CHECK(tkv && tsc && tskv && tssc && tcomp, "compressor_prefill: device allocation failed (validation)");
        CHECK(ds4_gpu_tensor_write(tkv, 0, kv, sizeof(kv)) != 0, "compressor_prefill: write kv (validation)");
        CHECK(ds4_gpu_tensor_write(tsc, 0, sc, sizeof(sc)) != 0, "compressor_prefill: write sc (validation)");

        /* n_tokens == 0 must be rejected, matching the validation list at
         * rocm/ds4_rocm_compressor.cuh:357. */
        CHECK(ds4_gpu_compressor_prefill_tensor(
                  tcomp, tskv, tssc, tkv, tsc, model, model_size,
                  ape_offset, 0u, norm_offset, 0u, HEAD_DIM, RATIO,
                  /*pos0=*/0u, /*n_tokens=*/0u, /*n_rot=*/0u, 4096u,
                  false, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-5f) == 0,
              "compressor_prefill: n_tokens == 0 must be rejected");

        /* A non-zero norm_type must be rejected before any kernel launches. */
        CHECK(ds4_gpu_compressor_prefill_tensor(
                  tcomp, tskv, tssc, tkv, tsc, model, model_size,
                  ape_offset, 0u, norm_offset, /*norm_type=*/1u, HEAD_DIM, RATIO,
                  /*pos0=*/0u, N_TOKENS, /*n_rot=*/0u, 4096u,
                  false, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-5f) == 0,
              "compressor_prefill: non-zero norm_type must be rejected");

        /* quantize_fp8 = true must propagate ds4_gpu_dsv4_fp8_kv_quantize_tensor's
         * failure: that entry is still stubbed (implemented later), so this
         * must fail rather than silently succeed. */
        CHECK(ds4_gpu_compressor_prefill_tensor(
                  tcomp, tskv, tssc, tkv, tsc, model, model_size,
                  ape_offset, 0u, norm_offset, 0u, HEAD_DIM, RATIO,
                  /*pos0=*/0u, N_TOKENS, /*n_rot=*/0u, 4096u,
                  /*quantize_fp8=*/true, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-5f) == 0,
              "compressor_prefill: quantize_fp8 must propagate the stubbed FP8 entry's failure");

        ds4_gpu_tensor_free(tkv); ds4_gpu_tensor_free(tsc);
        ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
        free(model);
    }

    fprintf(stderr, "  test_compressor_prefill OK\n");
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
    if (test_compressor_pool() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_compressor_update() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_compressor_prefill() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_kernels OK\n");
    return 0;
}
