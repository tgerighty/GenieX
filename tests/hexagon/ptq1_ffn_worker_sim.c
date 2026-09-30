// SPDX-License-Identifier: BSD-3-Clause
// Compare two bias-free PTQ1 matmuls that share one F32 activation.
// PTQ1_FFN_FUSED=0: two hvx_mm_matmul calls (compiles on PR51).
// PTQ1_FFN_FUSED=1: one op_matmul_ffn call (needs PTQ1 fused support).
#include <string.h>

#ifndef PTQ1_FFN_FUSED
#define PTQ1_FFN_FUSED 0
#endif
#ifndef PTQ1_FFN_COUNT_QUANT
#define PTQ1_FFN_COUNT_QUANT 1
#endif
#ifndef PTQ1_FFN_REJECT_CHECK
#define PTQ1_FFN_REJECT_CHECK 0
#endif
#ifndef PTQ1_WORKER_THREADS
#define PTQ1_WORKER_THREADS 1
#endif

/* When PTQ1_FFN_COUNT_QUANT=1, compile with -I ptq1_ffn_quant_wrap ahead of HTP so the
 * wrap header counts real flat quantizer calls without production hooks. */
#include "matmul-ops.c"

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, void *data, unsigned n) {
    (void)q;
    assert(n == PTQ1_WORKER_THREADS && n > 1);
    // Serial slice coverage only. Worker 0 completes M1 quantization before other slices.
    // This does not test concurrent scheduling or provide an eight-worker speed result.
    for (unsigned i = 0; i < n; ++i) func(n, i, data);
    return true;
}

#ifndef PTQ1_WORKER_K
#define PTQ1_WORKER_K 5120
#endif
#ifndef PTQ1_WORKER_M
#define PTQ1_WORKER_M 1
#endif
#ifndef PTQ1_WORKER_REPEATS
#define PTQ1_WORKER_REPEATS 1
#endif
#ifndef PTQ1_WORKER_N
#define PTQ1_WORKER_N 256
#endif

enum {
    K = PTQ1_WORKER_K,
    M = PTQ1_WORKER_M,
    N = PTQ1_WORKER_N,
    ROW_STRIDE = N + 16,
    TILES = (N + 31) / 32,
    KB = K / GENIEX_PTQ1_BLOCK_K
};

static geniex_ptq1_tile weights_gate[TILES][KB] __attribute__((aligned(128)));
static geniex_ptq1_tile weights_up[TILES][KB] __attribute__((aligned(128)));
static geniex_ptq1_block blocks_gate[GENIEX_PTQ1_TILE_ROWS * KB];
static geniex_ptq1_block blocks_up[GENIEX_PTQ1_TILE_ROWS * KB];
static float activations[M][K] __attribute__((aligned(128)));
static float outputs_gate[M][ROW_STRIDE] __attribute__((aligned(128)));
static float outputs_up[M][ROW_STRIDE] __attribute__((aligned(128)));
/* Fused two rings + full-K cache for K5120 exceeds 524288. */
static uint8_t vtcm[(PTQ1_WORKER_THREADS == 1 ? 1048576 : 8 * 1048576) + 128] __attribute__((aligned(128)));
static uint8_t queue_storage[PTQ1_WORKER_THREADS][8192] __attribute__((aligned(128)));

static int scalar_trit(uint8_t packed, unsigned power) {
    return (int)((((uint8_t)(packed * power)) * 3u) >> 8) - 1;
}

static float scalar_dot(const geniex_ptq1_tile weights[TILES][KB], unsigned ct, unsigned row,
                        const uint8_t *q8) {
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

static void fill_blocks(geniex_ptq1_block *blocks, uint32_t seed_start, unsigned scale_bias) {
    uint32_t seed = seed_start;
    for (unsigned row = 0; row < GENIEX_PTQ1_TILE_ROWS * KB; ++row) {
        for (unsigned k = 0; k < 24; ++k) {
            seed = seed * 1664525u + 1013904223u;
            blocks[row].qs[k] = seed >> 24;
        }
        for (unsigned k = 0; k < 2; ++k) {
            seed = seed * 1664525u + 1013904223u;
            blocks[row].qh[k] = seed >> 24;
        }
        blocks[row].d = 0x3800 + ((row + scale_bias) % 3) * 0x400;
    }
}

static void pack_weights(geniex_ptq1_tile weights[TILES][KB], const geniex_ptq1_block *blocks,
                         unsigned xor_tag) {
    for (unsigned ct = 0; ct < TILES; ++ct) {
        const unsigned valid_rows = ct + 1 == TILES ? N - ct * GENIEX_PTQ1_TILE_ROWS : GENIEX_PTQ1_TILE_ROWS;
        for (unsigned kb = 0; kb < KB; ++kb) {
            geniex_ptq1_pack_tile(&weights[ct][kb], blocks, KB, kb, valid_rows);
            weights[ct][kb].qs[0][ct % valid_rows] ^= xor_tag + ct + kb + 1;
        }
    }
}

static int check_outputs(const geniex_ptq1_tile weights[TILES][KB], float outputs[M][ROW_STRIDE],
                         const uint8_t *q8, const char *label) {
    for (unsigned ir = 0; ir < M; ++ir) {
        for (unsigned n = N; n < ROW_STRIDE; ++n) {
            if (outputs[ir][n] != 12345.0f) {
                printf("%s pad mismatch row %u col %u\n", label, ir, n);
                return 4;
            }
        }
        const __fp16 *scales = (const __fp16 *)(q8 + K);
        if (!((float)scales[0] > 0.0f && (float)scales[0] < 0.1f)) return 5;
        for (unsigned k = 0; k < K; ++k) {
            const float restored = ((const int8_t *)q8)[k] * (float)scales[k / 32];
            if (!(fabsf(restored - activations[ir][k]) <= 0.03f)) return 7;
        }
        for (unsigned n = 0; n < N; ++n) {
            const float expected = scalar_dot(weights, n / 32, n % 32, q8);
            if (!(fabsf(outputs[ir][n] - expected) <= 0.05f)) {
                printf("%s mismatch row %u col %u: got %.3f expected %.3f\n",
                       label, ir, n, outputs[ir][n], expected);
                return 6;
            }
        }
    }
    return 0;
}

int main(void) {
    fill_blocks(blocks_gate, 7u, 0);
    fill_blocks(blocks_up, 11u, 1);
    pack_weights(weights_gate, blocks_gate, 0);
    pack_weights(weights_up, blocks_up, 17);

    for (unsigned ir = 0; ir < M; ++ir) {
        for (unsigned k = 0; k < K; ++k) {
            activations[ir][k] = ((int)(k * (29 + ir) % 255) - 127) / 64.0f;
        }
        for (unsigned n = 0; n < ROW_STRIDE; ++n) {
            outputs_gate[ir][n] = 12345.0f;
            outputs_up[ir][n] = 12345.0f;
        }
    }

    struct htp_tensor w_gate = {
        .data = (uint32_t)(uintptr_t)weights_gate, .type = HTP_TYPE_PTQ1_0,
        .ne = {K, N, 1, 1}, .nb = {28, KB * 28, N * KB * 28, N * KB * 28}
    };
    struct htp_tensor w_up = {
        .data = (uint32_t)(uintptr_t)weights_up, .type = HTP_TYPE_PTQ1_0,
        .ne = {K, N, 1, 1}, .nb = {28, KB * 28, N * KB * 28, N * KB * 28}
    };
    struct htp_tensor x = {
        .data = (uint32_t)(uintptr_t)activations, .type = HTP_TYPE_F32,
        .ne = {K, M, 1, 1}, .nb = {4, K * 4, M * K * 4, M * K * 4}
    };
    struct htp_tensor y_gate = {
        .data = (uint32_t)(uintptr_t)outputs_gate, .type = HTP_TYPE_F32,
        .ne = {N, M, 1, 1}, .nb = {4, ROW_STRIDE * 4, M * ROW_STRIDE * 4, M * ROW_STRIDE * 4}
    };
    struct htp_tensor y_up = {
        .data = (uint32_t)(uintptr_t)outputs_up, .type = HTP_TYPE_F32,
        .ne = {N, M, 1, 1}, .nb = {4, ROW_STRIDE * 4, M * ROW_STRIDE * 4, M * ROW_STRIDE * 4}
    };

    struct htp_context ctx = {0};
    struct htp_ops_context octx = { .ctx = &ctx, .n_threads = PTQ1_WORKER_THREADS };
    struct htp_mm_kernel_params *kparams = (struct htp_mm_kernel_params *)octx.kernel_params;
    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;
    kparams->n_prefetch = 2;

    struct htp_mm_hvx_vtcm_layout layout;
#if PTQ1_FFN_FUSED
    /* Candidate: fused FFN layout with dst row bytes so cache guards match production. */
    htp_mm_hvx_vtcm_layout_build(&layout, kparams->kernel_type, HTP_TYPE_PTQ1_0, K, M, PTQ1_WORKER_THREADS,
        N * sizeof(float), w_gate.nb[1], htp_mm_q8_0_flat_row_size(K), 0, 2, false, false, true);
#else
    /* Baseline: existing single-op layout exactly (bias-free src2 row size 0). */
    htp_mm_hvx_vtcm_layout_build(&layout, kparams->kernel_type, HTP_TYPE_PTQ1_0, K, M, PTQ1_WORKER_THREADS,
        y_gate.nb[1], w_gate.nb[1], htp_mm_q8_0_flat_row_size(K), 0, 2, false, false, false);
#endif
    if (layout.total_bytes + 128 > sizeof(vtcm)) return 1;
    kparams->vtcm_size = layout.total_bytes;
    memset(vtcm, 0xa5, layout.total_bytes + 128);
    ctx.vtcm_base = vtcm;
    ctx.vtcm_size = layout.total_bytes;

    for (unsigned i = 0; i < PTQ1_WORKER_THREADS; ++i) {
        ctx.dma[i] = dma_queue_init(queue_storage[i], 16, (uintptr_t)vtcm,
                                   layout.total_bytes, &ctx.trace[i]);
    }

#if PTQ1_FFN_COUNT_QUANT
    ptq1_ffn_flat_quant_calls = 0;
#endif

#if PTQ1_FFN_FUSED && PTQ1_FFN_REJECT_CHECK
    {
        struct htp_tensor x_bad = x;
        x_bad.ne[0] = (K == 5120) ? 256 : 5120;
        x_bad.ne[1] = (M == 1) ? 3 : 1;
        octx.src[0] = &w_gate;
        octx.src[1] = &x_bad;
        octx.src[2] = &w_up;
        octx.dsts[0] = &y_gate;
        octx.dsts[1] = &y_up;
        const int st = op_matmul_ffn(&octx);
        if (st == HTP_STATUS_OK) {
            puts("PTQ1 FFN reject check failed: unsupported shape returned OK");
            return 9;
        }
        if (st != HTP_STATUS_NO_SUPPORT && st != HTP_STATUS_INVAL_PARAMS &&
            st != HTP_STATUS_VTCM_TOO_SMALL) {
            printf("PTQ1 FFN reject check unexpected status %d\n", st);
            return 9;
        }
        puts("PTQ1 FFN reject check passed");
        return 0;
    }
#endif

    for (unsigned repeat = 0; repeat < PTQ1_WORKER_REPEATS; ++repeat) {
#if PTQ1_FFN_FUSED
        octx.src[0] = &w_gate;
        octx.src[1] = &x;
        octx.src[2] = &w_up;
        octx.dsts[0] = &y_gate;
        octx.dsts[1] = &y_up;
        if (op_matmul_ffn(&octx) != HTP_STATUS_OK) return 2;
#else
        /* Both matmuls inside every baseline repetition for added-call slope. */
        octx.src[0] = &w_gate;
        octx.src[1] = &x;
        octx.src[2] = NULL;
        octx.dst = &y_gate;
        if (hvx_mm_matmul(&octx) != HTP_STATUS_OK) return 2;
        octx.src[0] = &w_up;
        octx.src[1] = &x;
        octx.src[2] = NULL;
        octx.dst = &y_up;
        if (hvx_mm_matmul(&octx) != HTP_STATUS_OK) return 2;
#endif
    }

    for (unsigned i = 0; i < 128; ++i) {
        if (vtcm[layout.total_bytes + i] != 0xa5) return 3;
    }

    const uint8_t *q8 = vtcm + layout.off_src1;
    int rc = check_outputs(weights_gate, outputs_gate, q8, "gate");
    if (rc) return rc;
    rc = check_outputs(weights_up, outputs_up, q8, "up");
    if (rc) return rc;

#if PTQ1_FFN_COUNT_QUANT
    {
        const unsigned expected = PTQ1_FFN_FUSED ? PTQ1_WORKER_REPEATS : (2u * PTQ1_WORKER_REPEATS);
        if (ptq1_ffn_flat_quant_calls != expected) {
            printf("PTQ1 FFN flat quant calls %u expected %u\n",
                   ptq1_ffn_flat_quant_calls, expected);
            return 10;
        }
    }
#endif

    float checksum = 0.0f;
    for (unsigned ir = 0; ir < M; ++ir) {
        for (unsigned n = 0; n < N; ++n) {
            checksum += outputs_gate[ir][n];
            checksum += outputs_up[ir][n];
        }
    }

    uint64_t hash = UINT64_C(14695981039346656037);
    const uint8_t *gate_bytes = (const uint8_t *)outputs_gate;
    const uint8_t *up_bytes = (const uint8_t *)outputs_up;
    for (size_t i = 0; i < sizeof(outputs_gate); ++i) {
        hash = (hash ^ gate_bytes[i]) * UINT64_C(1099511628211);
    }
    for (size_t i = 0; i < sizeof(outputs_up); ++i) {
        hash = (hash ^ up_bytes[i]) * UINT64_C(1099511628211);
    }

    printf("PTQ1 FFN worker K=%d M=%d N=%d repeats=%d checksum %.1f hash %016llx",
           K, M, N, PTQ1_WORKER_REPEATS, checksum, (unsigned long long)hash);
#if PTQ1_FFN_COUNT_QUANT
    printf(" flat_quant_calls %u", ptq1_ffn_flat_quant_calls);
#endif
    printf(" fused=%d worker_slices=%d\n", PTQ1_FFN_FUSED, PTQ1_WORKER_THREADS);
    return 0;
}
