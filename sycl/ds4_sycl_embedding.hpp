#pragma once

/* Embedding lookup.  ROCm keeps the kernels in ds4_rocm_common.cuh:11-83
 * and the launch glue in ds4_rocm_embedding_launch.cuh:1-151; both are
 * inlined here per entry point.  The ROCm bounds checks are the
 * specification for the validation below.
 *
 * All entries are NONZERO-means-success.
 *
 * These kernels FUSE the table lookup with the hyper-connection broadcast:
 * the output is [n_hc][n_embd] row-major with all n_hc rows identical.
 * ds4's CPU path does this as two steps, embed_token_f16 (ds4.c:6695) then
 * hc_from_plain_embedding (ds4.c:9890). */

#include "ds4_sycl_common.hpp"

extern "C" int ds4_gpu_embed_token_hc_tensor(
        ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t n_vocab, uint32_t token,
        uint32_t n_embd, uint32_t n_hc) {
    /* Guard set copied from rocm/ds4_rocm_embedding_launch.cuh:1-15.
     * Note token >= n_vocab is REJECTED here; the batched entry instead
     * clamps out-of-range ids to 0.  That asymmetry is upstream behaviour. */
    if (!out_hc || !model_map || n_vocab == 0u || token >= n_vocab ||
        n_embd == 0u || n_hc == 0u ||
        (uint64_t)n_embd * n_hc > UINT32_MAX) {
        return 0;
    }

    uint64_t weight_bytes = 0;
    uint64_t out_bytes    = 0;
    if (!sycl_u64_mul3_checked(n_vocab, n_embd, sizeof(uint16_t), &weight_bytes) ||
        !sycl_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !sycl_u64_mul3_checked(n_hc, n_embd, sizeof(float), &out_bytes) ||
        !sycl_tensor_has_bytes(out_hc, out_bytes)) {
        return 0;
    }
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, model_size,
                                            "token_embd");
    if (!wptr) return 0;
    if (g_devices.empty()) return 0;

    const uint64_t n = (uint64_t)n_embd * n_hc;

    try {
        sycl::queue &q = ds4_sycl_queue(out_hc->device_id);

        /* The weight table lives in the host mmap.  Copy only the one row
         * this token needs to the device rather than the whole table. */
        const uint64_t row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
        sycl_device_scratch_guard drow_guard = sycl_stage_host_bytes(
                q, wptr + (uint64_t)token * row_bytes, row_bytes);
        if (!drow_guard.p) return 0;
        uint16_t *drow = (uint16_t *)drow_guard.p;

        float         *o = (float *)out_hc->ptr;
        const uint32_t e_stride = n_embd;
        sycl::event _ds4_prof_ev33 = q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
            const size_t e = gid % e_stride;
            /* Broadcast: e cycles over [0,n_embd) as gid advances, so
             * every one of the n_hc rows receives the same values. */
            o[gid] = (float)sycl::bit_cast<sycl::half>(drow[e]);
        });
        q.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev33);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "embed_token_hc failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

/* Q8_0 row layout and per-element decode are handled by the shared
 * sycl_q8_0_row_bytes_checked and sycl_q8_0_dequant helpers in
 * ds4_sycl_common.hpp.  See rocm/ds4_rocm_common.cuh:19-63 for the
 * authoritative layout they encode. */
static int ds4_sycl_embed_token_hc_q8_0(ds4_gpu_tensor *out_hc,
                                        const void *model_map,
                                        uint64_t model_size,
                                        uint64_t weight_offset,
                                        uint32_t n_vocab, uint32_t token,
                                        uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !model_map || n_vocab == 0u || token >= n_vocab ||
        n_embd == 0u || n_hc == 0u ||
        (uint64_t)n_embd * n_hc > UINT32_MAX) {
        return 0;
    }

    uint64_t       row_bytes = 0;
    uint64_t       weight_bytes = 0;
    uint64_t       out_bytes = 0;
    if (!sycl_q8_0_row_bytes_checked(n_embd, &row_bytes) ||
        !sycl_u64_mul_checked(row_bytes, n_vocab, &weight_bytes) ||
        !sycl_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !sycl_u64_mul3_checked(n_hc, n_embd, sizeof(float), &out_bytes) ||
        !sycl_tensor_has_bytes(out_hc, out_bytes)) {
        return 0;
    }
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, model_size,
                                            "token_embd_q8_0");
    if (!wptr) return 0;
    if (g_devices.empty()) return 0;

    const uint64_t n = (uint64_t)n_embd * n_hc;

    try {
        sycl::queue &q = ds4_sycl_queue(out_hc->device_id);
        sycl_device_scratch_guard drow_guard = sycl_stage_host_bytes(
                q, wptr + (uint64_t)token * row_bytes, row_bytes);
        if (!drow_guard.p) return 0;
        unsigned char *drow = (unsigned char *)drow_guard.p;

        float         *o = (float *)out_hc->ptr;
        const uint32_t e_stride = n_embd;
        sycl::event _ds4_prof_ev34 = q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
            const size_t e = gid % e_stride;
            o[gid] = sycl_q8_0_dequant(drow, (uint32_t)e);
        });
        q.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev34);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "embed_token_hc_q8_0 failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

/* Public wrapper, n_hc = 1.  Mirrors
 * rocm/ds4_rocm_embedding_launch.cuh:115-131. */
extern "C" int ds4_gpu_embed_token_q8_0_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t n_vocab, uint32_t token,
        uint32_t n_embd) {
    return ds4_sycl_embed_token_hc_q8_0(out, model_map, model_size,
                                        weight_offset, n_vocab, token,
                                        n_embd, 1u);
}

/* PERFORMANCE CLIFF, deliberately shipped as a known deferred issue rather
 * than fixed here: ROCm indexes the registered model-map table directly on
 * device, so it never copies the embedding table at all.  SYCL kernels
 * cannot dereference the host mmap that model_map points at, so every
 * lookup must first stage bytes to device memory.  For the single-token
 * entries above that is one row, which is cheap.  The batched entries
 * below instead copy the WHOLE embedding table to device scratch on every
 * call, because a batch can reference arbitrary rows scattered across the
 * table.  That per-call copy is proportional to the entire table size,
 * n_vocab * n_embd * bytes_per_element: for a real DeepSeek V4 model
 * (roughly 130k vocab by 7168 embd, F16) that is on the order of 1.8 GB
 * per call, which is correct but not usable in production. The intended
 * fix is to route these through the device-resident model cache ds4
 * already has for other backends, ds4_gpu_register_model_map_no_copy plus
 * ds4_gpu_lookup_cache, instead of a fresh full-table copy every call.
 * Wiring the batched embedding entries up to that cache belongs to a
 * later plan; this comment is the record that the cliff is known. */
extern "C" int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *tokens_t,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_vocab, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    /* Guard set copied from rocm/ds4_rocm_embedding_launch.cuh, the
     * batched-launcher form: unlike the single-token entry, out-of-range
     * token ids are CLAMPED inside the kernel below, not rejected here. */
    if (!out_hc || !tokens_t || !model_map || n_vocab == 0u ||
        n_tokens == 0u || n_embd == 0u || n_hc == 0u) {
        return 0;
    }

    uint64_t weight_bytes = 0;
    uint64_t n            = 0;
    if (!sycl_u64_mul3_checked(n_vocab, n_embd, sizeof(uint16_t), &weight_bytes) ||
        !sycl_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !sycl_tensor_has_i32(tokens_t, n_tokens) ||
        !sycl_u64_mul_checked(n_tokens, n_hc, &n) ||
        !sycl_u64_mul_checked(n, n_embd, &n) ||
        !sycl_tensor_has_f32(out_hc, n)) {
        return 0;
    }
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, model_size,
                                            "token_embd");
    if (!wptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out_hc->device_id);

        sycl_device_scratch_guard dtab_guard =
                sycl_stage_host_bytes(q, wptr, weight_bytes);
        if (!dtab_guard.p) return 0;
        uint16_t *dtab = (uint16_t *)dtab_guard.p;

        const int32_t  *tok      = (const int32_t *)tokens_t->ptr;
        float          *o        = (float *)out_hc->ptr;
        const uint32_t  e_stride = n_embd;
        const uint32_t  hc       = n_hc;
        const uint32_t  vocab    = n_vocab;
        sycl::event _ds4_prof_ev35 = q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
            const size_t d   = gid % e_stride;
            const size_t t   = (gid / e_stride) / hc;
            int32_t      raw_tok = tok[t];
            uint32_t     tk   = raw_tok < 0 ? 0u : (uint32_t)raw_tok;
            if (tk >= vocab) tk = 0u;
            o[gid] = (float)sycl::bit_cast<sycl::half>(dtab[(size_t)tk * e_stride + d]);
        });
        q.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev35);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "embed_tokens_hc failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

/* Batched Q8_0 helper.  Same performance-cliff caveat as
 * ds4_gpu_embed_tokens_hc_tensor above: the whole quantised table is
 * copied to device scratch on every call. */
static int ds4_sycl_embed_tokens_hc_q8_0(ds4_gpu_tensor *out_hc,
                                         const ds4_gpu_tensor *tokens_t,
                                         const void *model_map,
                                         uint64_t model_size,
                                         uint64_t weight_offset,
                                         uint32_t n_vocab, uint32_t n_tokens,
                                         uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !tokens_t || !model_map || n_vocab == 0u ||
        n_tokens == 0u || n_embd == 0u || n_hc == 0u) {
        return 0;
    }

    uint64_t       row_bytes = 0;
    uint64_t       weight_bytes = 0;
    uint64_t       n = 0;
    if (!sycl_q8_0_row_bytes_checked(n_embd, &row_bytes) ||
        !sycl_u64_mul_checked(row_bytes, n_vocab, &weight_bytes) ||
        !sycl_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !sycl_tensor_has_i32(tokens_t, n_tokens) ||
        !sycl_u64_mul_checked(n_tokens, n_hc, &n) ||
        !sycl_u64_mul_checked(n, n_embd, &n) ||
        !sycl_tensor_has_f32(out_hc, n)) {
        return 0;
    }
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, model_size,
                                            "token_embd_q8_0");
    if (!wptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out_hc->device_id);

        sycl_device_scratch_guard dtab_guard =
                sycl_stage_host_bytes(q, wptr, weight_bytes);
        if (!dtab_guard.p) return 0;
        unsigned char *dtab = (unsigned char *)dtab_guard.p;

        const int32_t  *tok      = (const int32_t *)tokens_t->ptr;
        float          *o        = (float *)out_hc->ptr;
        const uint32_t  e_stride = n_embd;
        const uint32_t  hc       = n_hc;
        const uint32_t  vocab    = n_vocab;
        const uint64_t  rb       = row_bytes;
        sycl::event _ds4_prof_ev36 = q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
            const size_t d   = gid % e_stride;
            const size_t t   = (gid / e_stride) / hc;
            int32_t      raw_tok = tok[t];
            uint32_t     tk   = raw_tok < 0 ? 0u : (uint32_t)raw_tok;
            if (tk >= vocab) tk = 0u;
            o[gid] = sycl_q8_0_dequant(dtab + (size_t)tk * rb, (uint32_t)d);
        });
        q.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev36);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "embed_tokens_hc_q8_0 failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_embed_tokens_q8_0_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens_t,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_vocab, uint32_t n_tokens, uint32_t n_embd) {
    return ds4_sycl_embed_tokens_hc_q8_0(out, tokens_t, model_map, model_size,
                                         weight_offset, n_vocab, n_tokens,
                                         n_embd, 1u);
}
