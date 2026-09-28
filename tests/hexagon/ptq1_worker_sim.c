// SPDX-License-Identifier: BSD-3-Clause
// Include the production worker so the simulator calls hvx_mm_matmul directly.
#include "matmul-ops.c"

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, void *data, unsigned n) {
    (void)q; (void)func; (void)data; (void)n;
    assert(0);
    return false;
}

#ifndef PTQ1_WORKER_K
#define PTQ1_WORKER_K 256
#endif
#ifndef PTQ1_WORKER_M
#define PTQ1_WORKER_M 3
#endif
#ifndef PTQ1_WORKER_REPEATS
#define PTQ1_WORKER_REPEATS 1
#endif
#ifndef PTQ1_WORKER_N
#define PTQ1_WORKER_N 81
#endif
enum { K = PTQ1_WORKER_K, M = PTQ1_WORKER_M, N = PTQ1_WORKER_N,
    ROW_STRIDE = N + 16, TILES = (N + 31) / 32, KB = K / GENIEX_PTQ1_BLOCK_K };

static geniex_ptq1_tile weights[TILES][KB] __attribute__((aligned(128)));
static geniex_ptq1_block blocks[GENIEX_PTQ1_TILE_ROWS * KB];
static float activations[M][K] __attribute__((aligned(128)));
static float biases[M][ROW_STRIDE] __attribute__((aligned(128)));
static float outputs[M][ROW_STRIDE] __attribute__((aligned(128)));
static uint8_t vtcm[524288] __attribute__((aligned(128)));
static uint8_t queue_storage[4096] __attribute__((aligned(128)));

static int scalar_trit(uint8_t packed, unsigned power) {
    return (int)((((uint8_t)(packed * power)) * 3u) >> 8) - 1;
}

static float scalar_dot(unsigned ct, unsigned row, const uint8_t *q8) {
    static const unsigned powers[5] = {1, 3, 9, 27, 81};
    float total = 0.0f;
    for (unsigned kb = 0; kb < KB; ++kb) {
        const geniex_ptq1_tile *w = &weights[ct][kb];
        const int8_t *a = (const int8_t *)q8 + kb * GENIEX_PTQ1_BLOCK_K;
        int sum[4] = {0};
        unsigned k = 0;
        for (unsigned group = 0; group < 3; ++group) {
            for (unsigned n = 0; n < 5; ++n) {
                for (unsigned m = 0; m < 8; ++m, ++k) {
                    sum[k / 32] += scalar_trit(w->qs[group * 8 + m][row], powers[n]) * a[k];
                }
            }
        }
        for (unsigned n = 0; n < 4; ++n) {
            for (unsigned h = 0; h < 2; ++h, ++k) {
                sum[k / 32] += scalar_trit(w->qh[h][row], powers[n]) * a[k];
            }
        }
        const __fp16 *scales = (const __fp16 *)(q8 + K);
        float block_total = 0.0f;
        for (unsigned b = 0; b < 4; ++b) block_total += sum[b] * (float)scales[kb * 4 + b];
        total += geniex_ptq1_half_to_float(w->d[row]) * block_total;
    }
    return total;
}

int main(void) {
    uint32_t seed = 7;
    for (unsigned row = 0; row < GENIEX_PTQ1_TILE_ROWS * KB; ++row) {
        for (unsigned k = 0; k < 24; ++k) {
            seed = seed * 1664525u + 1013904223u;
            blocks[row].qs[k] = seed >> 24;
        }
        for (unsigned k = 0; k < 2; ++k) {
            seed = seed * 1664525u + 1013904223u;
            blocks[row].qh[k] = seed >> 24;
        }
        blocks[row].d = 0x3800 + (row % 3) * 0x400;
    }
    for (unsigned ct = 0; ct < TILES; ++ct) {
        for (unsigned kb = 0; kb < KB; ++kb) {
            geniex_ptq1_pack_tile(&weights[ct][kb], blocks, KB, kb,
                ct + 1 == TILES ? N - ct * GENIEX_PTQ1_TILE_ROWS : GENIEX_PTQ1_TILE_ROWS);
            weights[ct][kb].qs[0][ct % GENIEX_PTQ1_TILE_ROWS] ^= ct + kb + 1;
        }
    }
    for (unsigned ir = 0; ir < M; ++ir) {
        for (unsigned k = 0; k < K; ++k) activations[ir][k] = ((int)(k * (29 + ir) % 255) - 127) / 64.0f;
        for (unsigned n = 0; n < ROW_STRIDE; ++n) {
            biases[ir][n] = ((int)(n % 7) - 3) * 0.125f + ir * 0.25f;
            outputs[ir][n] = 12345.0f;
        }
    }

    struct htp_tensor w = { .data = (uint32_t)(uintptr_t)weights, .type = HTP_TYPE_PTQ1_0,
        .ne = {K, N, 1, 1}, .nb = {28, KB * 28, N * KB * 28, N * KB * 28} };
    struct htp_tensor x = { .data = (uint32_t)(uintptr_t)activations, .type = HTP_TYPE_F32,
        .ne = {K, M, 1, 1}, .nb = {4, K * 4, M * K * 4, M * K * 4} };
    struct htp_tensor b = { .data = (uint32_t)(uintptr_t)biases, .type = HTP_TYPE_F32,
        .ne = {N, M, 1, 1}, .nb = {4, ROW_STRIDE * 4, M * ROW_STRIDE * 4, M * ROW_STRIDE * 4} };
    struct htp_tensor y = { .data = (uint32_t)(uintptr_t)outputs, .type = HTP_TYPE_F32,
        .ne = {N, M, 1, 1}, .nb = {4, ROW_STRIDE * 4, M * ROW_STRIDE * 4, M * ROW_STRIDE * 4} };
    struct htp_context ctx = {0};
    struct htp_ops_context octx = { .ctx = &ctx, .src = {&w, &x, &b}, .dst = &y, .n_threads = 1 };
    struct htp_mm_kernel_params *kparams = (struct htp_mm_kernel_params *)octx.kernel_params;
    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;
    kparams->n_prefetch = 2;
    struct htp_mm_hvx_vtcm_layout layout;
    htp_mm_hvx_vtcm_layout_build(&layout, kparams->kernel_type, w.type, K, M, 1,
        y.nb[1], w.nb[1], htp_mm_q8_0_flat_row_size(K), b.nb[1], 2, false, false, false);
    if (layout.total_bytes + 128 > sizeof(vtcm)) return 1;
    kparams->vtcm_size = layout.total_bytes;
    memset(vtcm, 0xa5, layout.total_bytes + 128);
    ctx.vtcm_base = vtcm;
    ctx.vtcm_size = layout.total_bytes;
    ctx.dma[0] = dma_queue_init(queue_storage, 8, (uintptr_t)(vtcm + layout.off_src0), layout.src0_bytes, &ctx.trace[0]);
    for (unsigned repeat = 0; repeat < PTQ1_WORKER_REPEATS; ++repeat) {
        if (hvx_mm_matmul(&octx) != HTP_STATUS_OK) return 2;
    }
    for (unsigned i = 0; i < 128; ++i) if (vtcm[layout.total_bytes + i] != 0xa5) return 3;
    float checksum = 0.0f;
    for (unsigned ir = 0; ir < M; ++ir) {
        for (unsigned n = N; n < ROW_STRIDE; ++n) if (outputs[ir][n] != 12345.0f) return 4;
        const uint8_t *q8 = vtcm + layout.off_src1 + ir * htp_mm_q8_0_flat_row_size(K);
        const __fp16 *scales = (const __fp16 *)(q8 + K);
        if (!((float)scales[0] > 0.0f && (float)scales[0] < 0.1f)) return 5;
        for (unsigned k = 0; k < K; ++k) {
            const float restored = ((const int8_t *)q8)[k] * (float)scales[k / 32];
            if (!(fabsf(restored - activations[ir][k]) <= 0.03f)) return 7;
        }
        for (unsigned n = 0; n < N; ++n) {
            const float expected = scalar_dot(n / 32, n % 32, q8) + biases[ir][n];
            if (!(fabsf(outputs[ir][n] - expected) <= 0.05f)) {
                printf("mismatch row %u col %u: got %.3f expected %.3f\n", ir, n, outputs[ir][n], expected);
                return 6;
            }
            checksum += outputs[ir][n];
        }
    }
    printf("PTQ1 worker K=%d M=%d repeats=%d checksum %.1f\n", K, M, PTQ1_WORKER_REPEATS, checksum);
    return 0;
}
