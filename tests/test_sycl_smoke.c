/* Smoke test for the SYCL backend skeleton: device enumeration, tensor
 * allocation, and host/device round-trip.  Needs no model file. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

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

int ds4_sycl_device_count(void);

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init returned zero");
    CHECK(ds4_sycl_device_count() >= 1, "no SYCL device enumerated");

    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(4096);
    CHECK(t != NULL, "ds4_gpu_tensor_alloc returned NULL");
    CHECK(ds4_gpu_tensor_bytes(t) == 4096, "wrong tensor byte count");
    CHECK(ds4_gpu_tensor_contents(t) != NULL, "tensor has no device pointer");

    ds4_gpu_tensor *v = ds4_gpu_tensor_view(t, 1024, 512);
    CHECK(v != NULL, "ds4_gpu_tensor_view returned NULL");
    CHECK(ds4_gpu_tensor_bytes(v) == 512, "wrong view byte count");
    CHECK((char *)ds4_gpu_tensor_contents(v) ==
          (char *)ds4_gpu_tensor_contents(t) + 1024,
          "view pointer is not base plus offset");
    CHECK(ds4_gpu_tensor_device(t) == 0,
          "allocation was not stamped with tier 0");
    CHECK(ds4_gpu_tensor_device(v) == ds4_gpu_tensor_device(t),
          "view did not inherit the base tensor device");

    /* A view whose offset and length individually fit but whose sum wraps
     * past UINT64_MAX must be rejected, not silently accepted. */
    ds4_gpu_tensor *bad = ds4_gpu_tensor_view(t, 0xFFFFFFFFFFFFFF00ULL, 0x200ULL);
    CHECK(bad == NULL, "view accepted an offset+length that overflows");

    float src[16];
    float dst[16];
    for (int i = 0; i < 16; i++) src[i] = (float)i * 1.5f;
    memset(dst, 0, sizeof(dst));

    CHECK(ds4_gpu_tensor_write(t, 0, src, sizeof(src)) != 0,
          "ds4_gpu_tensor_write failed");
    CHECK(ds4_gpu_tensor_read(t, 0, dst, sizeof(dst)) != 0,
          "ds4_gpu_tensor_read failed");
    CHECK(memcmp(src, dst, sizeof(src)) == 0,
          "round-trip data mismatch");

    ds4_gpu_tensor *t2 = ds4_gpu_tensor_alloc(4096);
    CHECK(t2 != NULL, "second alloc returned NULL");
    CHECK(ds4_gpu_tensor_copy(t2, 0, t, 0, sizeof(src)) != 0,
          "ds4_gpu_tensor_copy failed");
    memset(dst, 0, sizeof(dst));
    CHECK(ds4_gpu_tensor_read(t2, 0, dst, sizeof(dst)) != 0,
          "read after copy failed");
    CHECK(memcmp(src, dst, sizeof(src)) == 0,
          "device to device copy mismatch");

    CHECK(ds4_gpu_tensor_fill_f32(t2, 2.5f, 16) != 0,
          "ds4_gpu_tensor_fill_f32 failed");
    CHECK(ds4_gpu_tensor_read(t2, 0, dst, sizeof(dst)) != 0,
          "read after fill failed");
    for (int i = 0; i < 16; i++) {
        CHECK(dst[i] == 2.5f, "fill_f32 produced wrong value");
    }

    ds4_gpu_tensor_free(t2);

    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(t);

    fprintf(stderr, "  test_sycl_smoke OK (devices=%d)\n",
            ds4_sycl_device_count());
    ds4_gpu_cleanup();
    return 0;
}
