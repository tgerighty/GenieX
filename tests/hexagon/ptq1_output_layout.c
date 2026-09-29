// SPDX-License-Identifier: BSD-3-Clause
// Build against the patched Prism HTP headers and run with hexagon-sim v75.
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "matmul-ops.h"

static void check_rows(uint32_t rows, uint32_t threads) {
    const uint32_t k = 5120;
    const size_t row_bytes = (size_t) rows * sizeof(float);
    const size_t worker_rows = hex_round_up((rows + threads - 1) / threads, 32);
    const size_t output_bytes = worker_rows * sizeof(float);
    const size_t quant_bytes = htp_mm_round_up(k * sizeof(float), QK_Q8_0_TILED * sizeof(float));
    struct htp_mm_hvx_vtcm_layout layout;

    htp_mm_hvx_vtcm_layout_build(&layout, HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT,
        HTP_TYPE_PTQ1_0, k, 1, threads, row_bytes, 1120, k * sizeof(float), 0, 2,
        false, false, false);

    assert(layout.dst_bytes / threads == MAX(output_bytes, quant_bytes) + HTP_MM_PTQ1_ACT_SCRATCH_SIZE);
    assert(layout.src0_bytes == threads * 2 * (k / 128) * HTP_MM_WEIGHT_TILE_SIZE_PTQ1_0);
    for (uint32_t ith = 0; ith < threads; ++ith) {
        const size_t first = worker_rows * ith;
        const size_t last = MIN(first + worker_rows, rows);
        if (first >= rows) continue;
        assert(last - first <= output_bytes / sizeof(float));
        assert((last - first) * sizeof(float) <= layout.dst_bytes / threads - HTP_MM_PTQ1_ACT_SCRATCH_SIZE);
    }
    if (rows <= 248321) assert(layout.total_bytes <= 8 * 1024 * 1024);
    else assert(layout.total_bytes > 8 * 1024 * 1024);
    if (rows == 248320) printf("PTQ1 output head VTCM %zu bytes\n", layout.total_bytes);
}

static void check_partial_output(uint32_t rows) {
    const uint32_t threads = 8;
    const uint32_t worker_rows = hex_round_up((rows + threads - 1) / threads, 32);
    float output[83];
    assert(rows <= 81 && worker_rows == 32);
    for (unsigned i = 0; i < 83; ++i) output[i] = -1.0f;

    for (uint32_t ith = 0; ith < threads; ++ith) {
        const uint32_t first = worker_rows * ith;
        const uint32_t last = MIN(first + worker_rows, rows);
        if (first >= rows) continue;
        float guarded[34];
        for (unsigned i = 0; i < 34; ++i) guarded[i] = -1.0f;
        float *tmp = guarded + 1;
        for (uint32_t ct = first / 32; ct < (last + 31) / 32; ++ct) {
            const uint32_t valid = MIN(32, rows - ct * 32);
            for (uint32_t r = 0; r < valid; ++r) tmp[ct * 32 - first + r] = (float)(ct * 32 + r + 1);
        }
        memcpy(output + first + 1, tmp, (last - first) * sizeof(float));
        assert(guarded[0] == -1.0f && guarded[33] == -1.0f);
    }
    assert(output[0] == -1.0f && output[rows + 1] == -1.0f);
    for (uint32_t row = 0; row < rows; ++row) assert(output[row + 1] == (float)(row + 1));
}

static void check_prefill_scratch(void) {
    const uint32_t k = 5120;
    const uint32_t threads = 8;
    struct htp_mm_hvx_vtcm_layout layout;
    htp_mm_hvx_vtcm_layout_build(&layout, HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT,
        HTP_TYPE_PTQ1_0, k, 3, threads, 81 * sizeof(float), 1120, k * sizeof(float), 0, 2,
        false, false, false);
    assert(layout.dst_bytes / threads == k * sizeof(float) + HTP_MM_PTQ1_ACT_SCRATCH_SIZE);
    assert(layout.total_bytes <= 8 * 1024 * 1024);

    const uint32_t prefill_ks[] = {5120, 6144};
    const uint32_t prefill_rows[] = {8, 9, 16, 512, 513};
    for (size_t ki = 0; ki < sizeof(prefill_ks) / sizeof(prefill_ks[0]); ++ki) {
        const uint32_t prefill_k = prefill_ks[ki];
        for (size_t i = 0; i < sizeof(prefill_rows) / sizeof(prefill_rows[0]); ++i) {
            const uint32_t rows = prefill_rows[i];
            htp_mm_hvx_vtcm_layout_build(&layout, HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT,
                HTP_TYPE_PTQ1_0, prefill_k, rows, threads, 256 * sizeof(float), 1120,
                prefill_k * sizeof(float), 0, 2, false, false, false);
            const size_t table_bytes = rows > 8 && rows <= 512 ? (size_t) rows * 192 * sizeof(float) : 0;
            assert(layout.dst_bytes / threads == prefill_k * sizeof(float) + table_bytes + HTP_MM_PTQ1_ACT_SCRATCH_SIZE);
            assert(layout.total_bytes <= 8 * 1024 * 1024);
        }
    }

    const uint32_t wide_rows[] = {4, 5, 8, 128, 129};
    for (size_t i = 0; i < sizeof(wide_rows) / sizeof(wide_rows[0]); ++i) {
        const uint32_t rows = wide_rows[i];
        htp_mm_hvx_vtcm_layout_build(&layout, HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT,
            HTP_TYPE_PTQ1_0, 17408, rows, threads, 273 * sizeof(float), 17408 / 128 * 28,
            17408 * sizeof(float), 0, 2, false, false, false);
        const size_t table_bytes = rows > 4 && rows <= 128 ? (size_t) rows * 544 * sizeof(float) : 0;
        assert(layout.dst_bytes / threads == 17408 * sizeof(float) + table_bytes + HTP_MM_PTQ1_ACT_SCRATCH_SIZE);
        assert(layout.total_bytes <= 8 * 1024 * 1024);
    }

    htp_mm_hvx_vtcm_layout_build(&layout, HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT,
        HTP_TYPE_PTQ1_0, 6144, 512, threads, 5120 * sizeof(float), (6144 / 128) * 28,
        6144 * sizeof(float), 5120 * sizeof(float), 2, false, false, false);
    assert(layout.dst_bytes / threads == 6144 * sizeof(float) + 512 * 192 * sizeof(float) + HTP_MM_PTQ1_ACT_SCRATCH_SIZE);
    assert(layout.src2_bytes == 5120 * sizeof(float));
    assert(layout.total_bytes <= 8 * 1024 * 1024);

    htp_mm_hvx_vtcm_layout_build(&layout, HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT,
        HTP_TYPE_PTQ1_0, 17408, 128, threads, 5120 * sizeof(float), 17408 / 128 * 28,
        17408 * sizeof(float), 5120 * sizeof(float), 2, false, false, false);
    assert(layout.dst_bytes / threads == 17408 * sizeof(float) + 128 * 544 * sizeof(float) + HTP_MM_PTQ1_ACT_SCRATCH_SIZE);
    assert(layout.src2_bytes == 5120 * sizeof(float));
    assert(layout.total_bytes <= 8 * 1024 * 1024);

    htp_mm_hvx_vtcm_layout_build(&layout, HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT,
        HTP_TYPE_PTQ1_0, 17408, 310, threads, 5120 * sizeof(float), 17408 / 128 * 28,
        17408 * sizeof(float), 0, 2, false, false, false);
    assert(layout.dst_bytes / threads == 17408 * sizeof(float) + HTP_MM_PTQ1_ACT_SCRATCH_SIZE);
    assert(layout.total_bytes <= 8 * 1024 * 1024);
}

int main(void) {
    const uint32_t coarse_limit = (8 * 1024 * 1024) / sizeof(float);
    assert(coarse_limit == 2097152);
    assert(248320 <= coarse_limit);
    check_rows(1, 8);
    check_rows(33, 8);
    check_rows(248320, 8);
    check_rows(248321, 8);
    check_rows(coarse_limit, 8);
    check_rows(coarse_limit + 1, 8);
    check_rows(20000000, 8);
    check_partial_output(33);
    check_partial_output(81);
    check_prefill_scratch();
    puts("PTQ1 output layout passed");
    return 0;
}
