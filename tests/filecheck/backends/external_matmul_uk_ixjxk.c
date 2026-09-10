#include <stdint.h>

int32_t external_matmul_uk_ixjxk(
    float *restrict c,
    const float *restrict a,
    const float *restrict b,
    int64_t extent_i,
    int64_t extent_j,
    int64_t extent_k,
    int64_t c_stride_i,
    int64_t c_stride_j,
    int64_t c_stride_k,
    int64_t a_stride_i,
    int64_t a_stride_j,
    int64_t a_stride_k,
    int64_t b_stride_i,
    int64_t b_stride_j,
    int64_t b_stride_k)
{
    /* C is invariant along k, A along j, and B along i. */
    (void)c_stride_k;
    (void)a_stride_j;
    (void)b_stride_i;
    for (int64_t i = 0; i < extent_i; ++i) {
        float *c_row = c + i * c_stride_i;
        const float *a_row = a + i * a_stride_i;
        for (int64_t k = 0; k < extent_k; ++k) {
            const float a_value = a_row[k * a_stride_k];
            const float *b_row = b + k * b_stride_k;
#pragma omp simd simdlen(16) aligned(b_row, c_row : 64)
            for (int64_t j = 0; j < extent_j; ++j) {
                c_row[j * c_stride_j] += a_value * b_row[j * b_stride_j];
            }
        }
    }

    return 0;
}
