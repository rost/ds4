/* Smoke test for the SYCL backend skeleton: device enumeration, tensor
 * allocation, and host/device round-trip.  Needs no model file. */

#include "ds4_gpu.h"

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
    CHECK(ds4_gpu_init() == 0, "ds4_gpu_init returned nonzero");
    CHECK(ds4_sycl_device_count() >= 1, "no SYCL device enumerated");

    fprintf(stderr, "  test_sycl_smoke OK (devices=%d)\n",
            ds4_sycl_device_count());
    ds4_gpu_cleanup();
    return 0;
}
