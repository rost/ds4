/* Shared test harness for the SYCL kernel correctness tests: the CHECK
 * assertion macros, the M_PI fallback, and the oracles genuinely needed
 * by more than one subsystem test file. */

#ifndef DS4_TEST_SYCL_HARNESS_H
#define DS4_TEST_SYCL_HARNESS_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
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

/* Half-precision bit decode, standard IEEE754 binary16 -> binary32,
 * including the subnormal case.  Needed because a SYCL half type is not
 * available in a plain C test (same reason cited at test_embed_f16 in
 * tests/test_sycl_embedding.c); this is the decode counterpart of the
 * round-trip-through-known-bit-
 * patterns encoding technique used throughout this file. */
static inline float oracle_half_to_float(uint16_t h) {
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

/* Oracle mirrors rms_norm_weight, ds4.c:6763-6769: mean-square normalise
 * then apply a per-channel weight, out[i] = x[i] * scale * w[i]. Used by
 * both the norm/rope tests directly and the compressor tests, which
 * validate their fused pooled-RMS-norm-then-RoPE kernels against this same
 * oracle composed with oracle_rope_tail_row below. */
static inline void oracle_rms_norm_weight(float *out, const float *x,
                                          const float *w, int n, float eps) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(sum / (double)n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

/* Oracle for rope_yarn_ramp, ds4.c:10209-10212, and shared by both RoPE
 * oracles below.  Indexed by the channel index i0 = pair * 2. */
static inline float oracle_rope_yarn_ramp(float low, float high, int i0) {
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
 * form.  Used by both the norm/rope tests and the compressor tests, which
 * validate their fused prefill/replay kernels against this same rotation
 * applied to an already-pooled-and-normalised row. */
static inline void oracle_rope_tail_row(float *row, uint32_t head_dim,
                                        uint32_t n_rot, uint32_t pos,
                                        uint32_t n_ctx_orig, float freq_base,
                                        float freq_scale, float ext_factor,
                                        float attn_factor, float beta_fast,
                                        float beta_slow, int inverse) {
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

#endif /* DS4_TEST_SYCL_HARNESS_H */
