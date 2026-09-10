#include <stdint.h>

int32_t external_matmul_uk_ixj(
                      float * restrict c,
                      const float * restrict a,
                      const float * restrict b,
                      int64_t extent_i,
                      int64_t extent_j,
                      int64_t c_stride_i,
                      int64_t c_stride_j,
                      int64_t a_stride_i,
                      int64_t a_stride_j,
                      int64_t b_stride_i,
                      int64_t b_stride_j)
{
    /* A is invariant along j and B is invariant along i. */
    (void)a_stride_j;
    (void)b_stride_i;

    for (int64_t i = 0; i < extent_i; ++i) {
        float *c_row = c + i * c_stride_i;
#pragma omp simd simdlen(16) aligned(b, c_row : 64)
        for (int64_t j = 0; j < extent_j; ++j) {
            c_row[j * c_stride_j] +=
                a[i * a_stride_i] *
                b[j * b_stride_j];
        }
    }

    return 0;
}
