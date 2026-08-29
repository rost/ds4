/* Host-side test for ds4_sycl_missing_required_subgroup_width, the pure
 * comparison helper backing the startup sub-group-width guard in
 * ds4_gpu_init (ds4_sycl.cpp).  Exercised here against synthetic
 * supported-widths arrays instead of real device behavior, per spec 6m/6r:
 * a mismatched device silently runs a different sub-group width rather
 * than erroring, so the check itself cannot be exercised by changing what
 * a device reports.  Needs no model file.
 *
 * This is a self-contained file, following this suite's established convention: it defines its own
 * CHECK macro (copied in shape from tests/test_sycl_router.c) rather than
 * sharing a harness, and does not include tests/test_sycl_harness.h or
 * ds4_gpu.h's SYCL-only counterpart ds4_sycl.h. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <stdio.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s\n", (msg));                           \
            return 1;                                                       \
        }                                                                   \
    } while (0)

/* declared with extern "C" linkage in ds4_sycl.cpp; forward-declared here
 * rather than pulling in a SYCL-only header, matching this codebase's
 * existing convention (see how tests/test_sycl_smoke.c forward-declares
 * ds4_sycl_device_count instead of including ds4_sycl.h). */
extern uint32_t ds4_sycl_missing_required_subgroup_width(
        const uint32_t *supported, int n_supported);

int main(void) {
    const uint32_t superset[] = {4, 8, 16, 32, 64};
    CHECK(ds4_sycl_missing_required_subgroup_width(
              superset, 5) == 0,
          "superset of required widths must be accepted");

    const uint32_t exact[] = {8, 16, 32};
    CHECK(ds4_sycl_missing_required_subgroup_width(
              exact, 3) == 0,
          "exact match of required widths must be accepted");

    const uint32_t unordered[] = {32, 8, 16};
    CHECK(ds4_sycl_missing_required_subgroup_width(
              unordered, 3) == 0,
          "required widths out of ascending order must be accepted");

    const uint32_t missing_8[] = {16, 32};
    CHECK(ds4_sycl_missing_required_subgroup_width(
              missing_8, 2) == 8,
          "missing width 8 must be reported");

    const uint32_t missing_16[] = {8, 32};
    CHECK(ds4_sycl_missing_required_subgroup_width(
              missing_16, 2) == 16,
          "missing width 16 must be reported");

    const uint32_t missing_32[] = {8, 16};
    CHECK(ds4_sycl_missing_required_subgroup_width(
              missing_32, 2) == 32,
          "missing width 32 must be reported");

    CHECK(ds4_sycl_missing_required_subgroup_width(
              NULL, 0) == 8,
          "empty supported list must report the first required width");

    fprintf(stderr, "test_sycl_subgroup_guard OK\n");
    return 0;
}
