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

    const uint32_t exact[] = {16, 32};
    CHECK(ds4_sycl_missing_required_subgroup_width(
              exact, 2) == 0,
          "exact match of required widths must be accepted");

    const uint32_t unordered[] = {32, 16};
    CHECK(ds4_sycl_missing_required_subgroup_width(
              unordered, 2) == 0,
          "required widths out of ascending order must be accepted");

    /* The two real devices this backend has run on, by the widths they
     * actually report.  Xe2 is the reason width 8 is not required: the
     * Arc Pro B60 reports {16, 32} and nothing else, so requiring 8 --
     * which this backend did until B60 silicon was first available to
     * test against -- refused every Battlemage card at startup. */
    const uint32_t xe2_b60[] = {16, 32};
    CHECK(ds4_sycl_missing_required_subgroup_width(
              xe2_b60, 2) == 0,
          "Xe2 (Arc Pro B60) reporting {16, 32} must be accepted");

    const uint32_t xe1_a770[] = {8, 16, 32};
    CHECK(ds4_sycl_missing_required_subgroup_width(
              xe1_a770, 3) == 0,
          "Xe1 (A770) reporting {8, 16, 32} must still be accepted");

    const uint32_t missing_16[] = {8, 32};
    CHECK(ds4_sycl_missing_required_subgroup_width(
              missing_16, 2) == 16,
          "missing width 16 must be reported");

    const uint32_t missing_32[] = {8, 16};
    CHECK(ds4_sycl_missing_required_subgroup_width(
              missing_32, 2) == 32,
          "missing width 32 must be reported");

    /* Width 8 on its own satisfies nothing: it is not required, and the
     * two widths that are required are both absent. */
    const uint32_t only_8[] = {8};
    CHECK(ds4_sycl_missing_required_subgroup_width(
              only_8, 1) == 16,
          "a device offering only width 8 must be refused");

    CHECK(ds4_sycl_missing_required_subgroup_width(
              NULL, 0) == 16,
          "empty supported list must report the first required width");

    fprintf(stderr, "test_sycl_subgroup_guard OK\n");
    return 0;
}
