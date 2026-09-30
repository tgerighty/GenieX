/* Test-only include wrap: count real flat Q8_0 quantizer calls. */
#pragma once
#define quantize_f32_q8_0_flat_kernel quantize_f32_q8_0_flat_kernel_impl
#include_next "hvx-mm-kernels-flat.h"
#undef quantize_f32_q8_0_flat_kernel

static unsigned ptq1_ffn_flat_quant_calls;

static inline void quantize_f32_q8_0_flat_kernel(
    const uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_row_size,
    size_t dst_row_size
) {
    ptq1_ffn_flat_quant_calls++;
    quantize_f32_q8_0_flat_kernel_impl(src_data, dst_data, tmp_data, ne0, nrows, src_row_size,
                                       dst_row_size);
}
