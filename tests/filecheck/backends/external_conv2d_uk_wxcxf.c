#include <stdint.h>

int32_t external_conv2d_uk_wxcxf(
    float *restrict output,
    const float *restrict input,
    const float *restrict filter,
    int64_t extent_w,
    int64_t extent_c,
    int64_t extent_f,
    int64_t output_stride_w,
    int64_t output_stride_c,
    int64_t output_stride_f,
    int64_t input_stride_w,
    int64_t input_stride_c,
    int64_t input_stride_f,
    int64_t filter_stride_w,
    int64_t filter_stride_c,
    int64_t filter_stride_f)
{
    /* Output is invariant along c, input along f, and filter along w. */
    (void)output_stride_c;
    (void)input_stride_f;
    (void)filter_stride_w;

    for (int64_t w = 0; w < extent_w; ++w) {
        float *output_row = output + w * output_stride_w;
        const float *input_row = input + w * input_stride_w;
        for (int64_t c = 0; c < extent_c; ++c) {
            const float input_value = input_row[c * input_stride_c];
            const float *filter_row = filter + c * filter_stride_c;
#pragma omp simd simdlen(16) aligned(output_row, filter_row : 64)
            for (int64_t f = 0; f < extent_f; ++f) {
                output_row[f * output_stride_f] +=
                    input_value * filter_row[f * filter_stride_f];
            }
        }
    }

    return 0;
}
