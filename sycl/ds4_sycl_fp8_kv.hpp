#pragma once

/* E4M3-style FP8 value table and nearest-rounding dequantiser for the
 * DeepSeek V4 compressed KV cache, ported from the CPU reference
 * dsv4_e4m3fn_value_cpu/dsv4_e4m3fn_dequant_cpu (ds4.c:3196-3235) and
 * cross-checked against the CUDA/ROCm device versions
 * dsv4_e4m3fn_value_dev/dsv4_e4m3fn_dequant_dev
 * (rocm/ds4_rocm_norm_rope.cuh:307-330).  The two device references agree
 * with the CPU lookup table at every exponent: for the non-subnormal
 * branch (exp 1..15) exp_scale[exp] == exp2f((float)exp - 7.0f) exactly,
 * since every table entry is an exact power of two; the subnormal branch
 * (exp == 0) is an explicit special case in both, not part of the table.
 * sycl_e4m3fn_value below follows the exp2f-style device formula rather
 * than reproducing the CPU's 16-entry table, since that agreement holds.
 *
 * The magnitude code is 7 bits: a 4-bit exponent field and a 3-bit
 * mantissa field, packed as (exp << 3) | mant, ranging 0..126 (127 is
 * unused: 448 is the largest representable magnitude, at code 126).
 * sycl_e4m3fn_dequant finds the nearest representable magnitude via
 * binary search, then breaks an exact tie to the even neighbour: ported
 * literally from ds4.c:3228-3232, including the direction check on which
 * of the two tied codes is even, since collapsing it to a plain `<=` or
 * dropping it changes the result for exactly half of all ties.
 *
 * Kept in an anonymous namespace: these are internal helpers for this
 * header's own kernel code (added in a later task), not part of any
 * public API.  The two extern "C" functions below exist only so
 * tests/test_sycl_fp8_kv.c (plain C, no ABI entry point yet for this
 * subsystem) can exercise them from the host. */

#include "ds4_sycl_common.hpp"

namespace {

static inline float sycl_e4m3fn_value(int code) {
    const int exp = (code >> 3) & 0x0f;
    const int mant = code & 0x07;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * sycl::exp2((float)exp - 7.0f);
}

static inline float sycl_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = sycl::fmin(sycl::fabs(x), 448.0f);

    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (sycl_e4m3fn_value(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int best = lo;
    if (best < 126) {
        const float best_diff = sycl::fabs(ax - sycl_e4m3fn_value(best));
        const float next_diff = sycl::fabs(ax - sycl_e4m3fn_value(best + 1));
        if (next_diff < best_diff ||
            (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best++;
        }
    }

    return sign * sycl_e4m3fn_value(best);
}

} // namespace

extern "C" float ds4_sycl_test_e4m3fn_value(int code) {
    return sycl_e4m3fn_value(code);
}

extern "C" float ds4_sycl_test_e4m3fn_dequant(float x) {
    return sycl_e4m3fn_dequant(x);
}
