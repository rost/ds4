/* Proves oneMKL's strided-batched gemm_batch is wired correctly (build
 * flags, column-major argument order, the stride_a == 0 KV-broadcast case)
 * against a hand-computed result, before any production matmul or
 * attention entry depends on it.  See the block comment on
 * ds4_sycl_test_gemm_batch_smoke (sycl/ds4_sycl_matmul.hpp) for the exact
 * shape and the hand-computed expected values this checks. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <stdio.h>

int ds4_sycl_test_gemm_batch_smoke(void);

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s\n", (msg));                           \
            return 1;                                                       \
        }                                                                   \
    } while (0)

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init returned zero");
    CHECK(ds4_sycl_test_gemm_batch_smoke() != 0, "gemm_batch smoke mismatch");
    fprintf(stderr, "  test_sycl_gemm_batch_smoke OK\n");
    ds4_gpu_cleanup();
    return 0;
}
