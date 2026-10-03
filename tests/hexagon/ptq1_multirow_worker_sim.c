// SPDX-License-Identifier: BSD-3-Clause
// Functional multirow PTQ1 correctness via production op_matmul.
// Reference only: /tmp/geniex-board-fix-20260930/tests/hexagon/ptq1_worker_sim.c
// (that fixture calls hvx_mm_matmul; this one exercises op_matmul + M1 VTCM).
#include "matmul-ops.c"

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

// Keep production error logging callable in the standalone simulator.
void HAP_debug_v2(int level, const char *file, int line, const char *format, ...) {
    (void) level;
    (void) file;
    (void) line;
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

void HAP_debug(const char *msg, int level, const char *file, int line) {
    (void) level;
    (void) file;
    (void) line;
    puts(msg);
}

// The standalone simulator has no QuRT mutex service. Keep the HMX path loud
// if the dispatcher reaches it; these are link stubs, not lock emulation.
void qurt_mutex_init(qurt_mutex_t *mutex) {
    (void) mutex;
    assert(0 && "unexpected HMX mutex init in PTQ1 worker test");
}
void qurt_mutex_destroy(qurt_mutex_t *mutex) {
    (void) mutex;
    assert(0 && "unexpected HMX mutex destroy in PTQ1 worker test");
}
void qurt_mutex_lock(qurt_mutex_t *mutex) {
    (void) mutex;
    assert(0 && "unexpected HMX mutex lock in PTQ1 worker test");
}
void qurt_mutex_unlock(qurt_mutex_t *mutex) {
    (void) mutex;
    assert(0 && "unexpected HMX mutex unlock in PTQ1 worker test");
}

// The standalone simulator has no QuRT wake service. PTQ1 must never use
// the HMX queue: trap if this unrelated path is selected.
int qurt_futex_wake(void *lock, int n_to_wake) {
    (void) lock;
    (void) n_to_wake;
    assert(0 && "unexpected HMX queue in PTQ1 worker test");
    return -1;
}

// Synchronous multi-worker reuse: run each ith on this thread. For PTQ1 M1
// rows, n_quant_tasks is 1 so idle workers only spin on an already-cleared
// quant barrier (no deadlock under sequential dispatch).
bool work_queue_run_async(work_queue_t q, work_queue_func_t func, void * data, unsigned n) {
    (void) q;
    for (unsigned i = 0; i < n; ++i) {
        func(n, i, data);
    }
    return true;
}

#ifndef PTQ1_WORKER_K
#define PTQ1_WORKER_K 256
#endif
#ifndef PTQ1_WORKER_M
#define PTQ1_WORKER_M 3
#endif
#ifndef PTQ1_WORKER_N
#define PTQ1_WORKER_N 81
#endif
#ifndef PTQ1_WORKER_THREADS
#define PTQ1_WORKER_THREADS 4
#endif
#ifndef PTQ1_WORKER_BIAS
#define PTQ1_WORKER_BIAS 1
#endif
#ifndef PTQ1_WORKER_VTCM_REJECT
#define PTQ1_WORKER_VTCM_REJECT 0
#endif
#ifndef PTQ1_WORKER_BATCH
#define PTQ1_WORKER_BATCH 0
#endif
#ifndef PTQ1_WORKER_PADDED_W
#define PTQ1_WORKER_PADDED_W 0
#endif
#ifndef PTQ1_WORKER_RANDOM_ACT
#define PTQ1_WORKER_RANDOM_ACT 0
#endif
#ifndef PTQ1_WORKER_PREFETCH
#define PTQ1_WORKER_PREFETCH 2
#endif
#ifndef PTQ1_WORKER_Q8_CORRUPT
#define PTQ1_WORKER_Q8_CORRUPT 0
#endif
#if PTQ1_WORKER_PREFETCH != 2 && PTQ1_WORKER_PREFETCH != 4 && PTQ1_WORKER_PREFETCH != 8
#error "PTQ1_WORKER_PREFETCH must be 2, 4, or 8"
#endif
#if PTQ1_WORKER_PREFETCH > 2 && (PTQ1_WORKER_M != 1 || PTQ1_WORKER_BATCH)
#error "prefetch 4 or 8 is only valid for the M1 row path"
#endif

enum {
    K          = PTQ1_WORKER_K,
    M          = PTQ1_WORKER_M,
    N          = PTQ1_WORKER_N,
    NTHREADS   = PTQ1_WORKER_THREADS,
    ROW_STRIDE = N + 16,
    TILES      = (N + 31) / 32,
    KB         = K / GENIEX_PTQ1_BLOCK_K,
    Q8_QUANTS  = ((K + 127) / 128) * 128,
    Q8_SCALES  = (((((K + 31) / 32) * 2) + 127) / 128) * 128,
    Q8_ROW     = Q8_QUANTS + Q8_SCALES,
    QUANT_TMP  = ((K * 4 + 511) / 512) * 512
};

static geniex_ptq1_tile weights[TILES][KB] __attribute__((aligned(128)));
static geniex_ptq1_block blocks[GENIEX_PTQ1_TILE_ROWS * KB];
static float activations[M][K] __attribute__((aligned(128)));
static float activation_copy[M][K];
static float bias_vec[N] __attribute__((aligned(128)));
static float outputs[M][ROW_STRIDE] __attribute__((aligned(128)));
// Hardware-class VTCM budget for M1 layout (actual rows reuse the same blob).
static uint8_t vtcm[8 * 1024 * 1024] __attribute__((aligned(128)));
static uint8_t queue_storage[NTHREADS][4096] __attribute__((aligned(128)));
static uint8_t q8_row[Q8_ROW] __attribute__((aligned(128)));
static uint8_t quant_tmp[QUANT_TMP] __attribute__((aligned(128)));

static int scalar_trit(uint8_t packed, unsigned power) {
    return (int) ((((uint8_t) (packed * power)) * 3u) >> 8) - 1;
}

// Independent Prism 16-then-8 scalar packing reference (not the HVX path).
static float scalar_dot(unsigned ct, unsigned row, const uint8_t * q8) {
    static const unsigned powers[5] = {1, 3, 9, 27, 81};
    float total = 0.0f;
    for (unsigned kb = 0; kb < KB; ++kb) {
        const geniex_ptq1_tile * w = &weights[ct][kb];
        const int8_t * a = (const int8_t *) q8 + kb * GENIEX_PTQ1_BLOCK_K;
        int sum[4] = {0};
        unsigned k = 0;
        for (unsigned stage = 0; stage < 2; ++stage) {
            const unsigned width  = stage == 0 ? 16 : 8;
            const unsigned offset = stage == 0 ? 0 : 16;
            for (unsigned n = 0; n < 5; ++n) {
                for (unsigned m = 0; m < width; ++m, ++k) {
                    sum[k / 32] += scalar_trit(w->qs[offset + m][row], powers[n]) * a[k];
                }
            }
        }
        for (unsigned n = 0; n < 4; ++n) {
            for (unsigned h = 0; h < 2; ++h, ++k) {
                sum[k / 32] += scalar_trit(w->qh[h][row], powers[n]) * a[k];
            }
        }
        const __fp16 * scales = (const __fp16 *) (q8 + K);
        float block_total = 0.0f;
        for (unsigned b = 0; b < 4; ++b) {
            block_total += sum[b] * (float) scales[kb * 4 + b];
        }
        total += geniex_ptq1_half_to_float(w->d[row]) * block_total;
    }
    return total;
}

static uint64_t bytes_hash(const void * p, size_t n) {
    uint64_t hash = UINT64_C(14695981039346656037);
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) {
        hash = (hash ^ b[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

static int tensors_equal(const struct htp_tensor * a, const struct htp_tensor * b) {
    if (a->data != b->data || a->type != b->type) {
        return 0;
    }
    for (int i = 0; i < 4; ++i) {
        if (a->ne[i] != b->ne[i] || a->nb[i] != b->nb[i]) {
            return 0;
        }
    }
    return 1;
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
        const unsigned valid_rows =
            ct + 1 == TILES ? N - ct * GENIEX_PTQ1_TILE_ROWS : GENIEX_PTQ1_TILE_ROWS;
        for (unsigned kb = 0; kb < KB; ++kb) {
            geniex_ptq1_pack_tile(&weights[ct][kb], blocks, KB, kb, valid_rows);
            weights[ct][kb].qs[0][ct % valid_rows] ^= ct + kb + 1;
        }
    }
    seed = 7;
    for (unsigned ir = 0; ir < M; ++ir) {
        for (unsigned k = 0; k < K; ++k) {
#if PTQ1_WORKER_RANDOM_ACT
            seed = seed * 1664525u + 1013904223u;
            activations[ir][k] = (float) (seed >> 8) / 8388608.0f - 1.0f;
#else
            activations[ir][k] = ((int) (k * (29 + ir) % 255) - 127) / 64.0f;
#endif
        }
        for (unsigned n = 0; n < ROW_STRIDE; ++n) {
            outputs[ir][n] = 12345.0f;
        }
    }
    for (unsigned n = 0; n < N; ++n) {
        bias_vec[n] = ((int) (n % 7) - 3) * 0.125f;
    }
    memcpy(activation_copy, activations, sizeof(activations));
    const uint64_t weight_hash = bytes_hash(weights, sizeof(weights));
    const uint64_t act_hash    = bytes_hash(activations, sizeof(activations));
    const uint64_t bias_hash   = bytes_hash(bias_vec, sizeof(bias_vec));

    struct htp_tensor w = {
        .data = (uint32_t) (uintptr_t) weights,
        .type = HTP_TYPE_PTQ1_0,
        .ne   = {K, PTQ1_WORKER_PADDED_W ? TILES * 32 : N, 1, 1},
        .nb   = {28, KB * 28, (PTQ1_WORKER_PADDED_W ? TILES * 32 : N) * KB * 28,
            (PTQ1_WORKER_PADDED_W ? TILES * 32 : N) * KB * 28},
    };
    struct htp_tensor x = {
        .data = (uint32_t) (uintptr_t) activations,
        .type = HTP_TYPE_F32,
        .ne   = {K, M, 1, 1},
        .nb   = {4, K * 4, M * K * 4, M * K * 4},
    };
    struct htp_tensor b = {
        .data = (uint32_t) (uintptr_t) bias_vec,
        .type = HTP_TYPE_F32,
        .ne   = {N, 1, 1, 1},
        .nb   = {4, N * 4, N * 4, N * 4},
    };
    struct htp_tensor y = {
        .data = (uint32_t) (uintptr_t) outputs,
        .type = HTP_TYPE_F32,
        .ne   = {N, M, 1, 1},
        .nb   = {4, ROW_STRIDE * 4, M * ROW_STRIDE * 4, M * ROW_STRIDE * 4},
    };
    const struct htp_tensor w0 = w;
    const struct htp_tensor x0 = x;
    const struct htp_tensor b0 = b;
    const struct htp_tensor y0 = y;

    struct htp_context ctx = {0};
    struct htp_ops_context octx = {
        .ctx       = &ctx,
        .src       = {&w, &x, PTQ1_WORKER_BIAS ? &b : NULL},
        .dst       = &y,
        .n_threads = NTHREADS,
    };
    struct htp_mm_kernel_params * kparams = (struct htp_mm_kernel_params *) octx.kernel_params;
    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;
#if PTQ1_WORKER_BATCH
    // Real four-worker concurrency is covered on the board, not by this stub.
    _Static_assert(NTHREADS == 1, "batch simulator needs one real quant task");
    kparams->kernel_type = HTP_MM_KERNEL_HVX_PTQ1_BATCH;
#endif
    kparams->n_prefetch  = PTQ1_WORKER_PREFETCH;

    // Host-equivalent M1 VTCM layout even when M>1.
    struct htp_mm_hvx_vtcm_layout layout;
    htp_mm_hvx_vtcm_layout_build(
        &layout, kparams->kernel_type, w.type, K, PTQ1_WORKER_BATCH ? M : 1, NTHREADS, y.nb[1], w.nb[1],
        htp_mm_q8_0_flat_row_size(K), PTQ1_WORKER_BIAS ? b.nb[1] : 0,
        PTQ1_WORKER_PREFETCH, false, false, false);
    if (layout.total_bytes + 128 > sizeof(vtcm)) {
        printf("PTQ1 multirow VTCM layout %zu exceeds fixture buffer\n", layout.total_bytes);
        return 1;
    }
    kparams->vtcm_size = layout.total_bytes;
    memset(vtcm, 0xa5, layout.total_bytes + 128);
    ctx.vtcm_base = vtcm;
#if PTQ1_WORKER_VTCM_REJECT
    ctx.vtcm_size = layout.total_bytes - 1;
#else
    ctx.vtcm_size = layout.total_bytes;
#endif
    for (unsigned t = 0; t < NTHREADS; ++t) {
        ctx.dma[t] = dma_queue_init(
            queue_storage[t], PTQ1_WORKER_PREFETCH == 8 ? 16 : 8,
            (uintptr_t) (vtcm + layout.off_src0), layout.src0_bytes, &ctx.trace[t]);
    }

    const int status = op_matmul(&octx);
#if PTQ1_WORKER_VTCM_REJECT
    if (status != HTP_STATUS_VTCM_TOO_SMALL) {
        printf("expected VTCM_TOO_SMALL got %d\n", status);
        return 9;
    }
    puts("PTQ1 multirow VTCM reject passed");
    return 0;
#else
    if (status != HTP_STATUS_OK) {
        printf("op_matmul failed status=%d\n", status);
        return 2;
    }
#endif

#if PTQ1_WORKER_Q8_CORRUPT
    vtcm[layout.off_src1] ^= 1;
#endif
    if ((size_t) layout.off_src1 + (PTQ1_WORKER_BATCH ? M : 1) * Q8_ROW > layout.total_bytes) {
        puts("PTQ1 Q8 rows exceed VTCM layout");
        return 14;
    }
    const unsigned q8_rows = PTQ1_WORKER_BATCH ? M : 1;
    for (unsigned row = 0; row < q8_rows; ++row) {
        const unsigned ir = PTQ1_WORKER_BATCH ? row : M - 1;
        memset(q8_row, 0xa5, sizeof(q8_row));
        quantize_f32_q8_0_flat_kernel((const uint8_t *) activations[ir],
            q8_row, quant_tmp, K, 1, K * 4, Q8_ROW);
        if (memcmp(vtcm + layout.off_src1 + row * Q8_ROW, q8_row, Q8_ROW)) {
            printf("PTQ1 Q8 row byte mismatch row=%u bytes=%u\n", ir, Q8_ROW);
#if PTQ1_WORKER_Q8_CORRUPT
            puts("PTQ1 Q8 corruption control detected");
#endif
            return 15;
        }
    }
    printf("PTQ1 Q8 exact bytes rows=%u bytes_per_row=%u\n", q8_rows, Q8_ROW);
#if PTQ1_WORKER_Q8_CORRUPT
    puts("PTQ1 Q8 corruption control was not detected");
    return 16;
#endif

#if PTQ1_WORKER_SCALE_CACHE_CHECK
    _Static_assert(PTQ1_WORKER_BATCH && K == 5120 && M >= 2 && M <= 31 &&
                   NTHREADS == 1 && TILES >= 4 && !PTQ1_WORKER_VTCM_REJECT,
                   "scale-cache check needs an eligible one-worker batch");
    // Quant scratch contains the last row below the unchanged activation scratch.
    const uint8_t *scratch = vtcm + layout.off_dst;
    const size_t table_end = 256 + (size_t) M * 640;
    if (layout.dst_bytes < HTP_MM_PTQ1_ACT_SCRATCH_SIZE ||
        table_end > layout.dst_bytes - HTP_MM_PTQ1_ACT_SCRATCH_SIZE) {
        puts("PTQ1 scale table exceeds quant scratch");
        return 14;
    }
    float expected_scales[160];
    for (unsigned ir = 0; ir < M; ++ir) {
        quantize_f32_q8_0_flat_kernel((const uint8_t *) activations[ir],
            q8_row, quant_tmp, K, 1, K * 4, Q8_ROW);
        geniex_ptq1_prepare_flat_scales(expected_scales, q8_row, K);
        if (memcmp(scratch + 256 + ir * 640, expected_scales, sizeof(expected_scales))) {
            puts("PTQ1 scale table differs after worker completion");
            return 12;
        }
    }
    if (memcmp(scratch + table_end, (const uint8_t *) activations[M - 1] + table_end,
               K * sizeof(float) - table_end)) {
        puts("PTQ1 gap before activation scratch changed");
        return 13;
    }
    puts("PTQ1 table and scratch-gap checks passed");
#endif
    for (unsigned i = 0; i < 128; ++i) {
        if (vtcm[layout.total_bytes + i] != 0xa5) {
            return 3;
        }
    }
    if (!tensors_equal(&w, &w0) || !tensors_equal(&x, &x0) || !tensors_equal(&y, &y0) ||
        (PTQ1_WORKER_BIAS && !tensors_equal(&b, &b0))) {
        puts("descriptor mutated");
        return 10;
    }
    if (bytes_hash(weights, sizeof(weights)) != weight_hash ||
        bytes_hash(activations, sizeof(activations)) != act_hash ||
        bytes_hash(bias_vec, sizeof(bias_vec)) != bias_hash ||
        memcmp(activations, activation_copy, sizeof(activations)) != 0) {
        puts("input mutated");
        return 11;
    }

    float checksum = 0.0f;
    for (unsigned ir = 0; ir < M; ++ir) {
        for (unsigned n = N; n < ROW_STRIDE; ++n) {
            if (outputs[ir][n] != 12345.0f) {
                return 4;
            }
        }
        // Each row reuses the same M1 VTCM act slot; re-quantize for the oracle.
        quantize_f32_q8_0_flat_kernel(
            (const uint8_t *) activations[ir], q8_row, quant_tmp, K, 1, K * 4, Q8_ROW);
        for (unsigned n = 0; n < N; ++n) {
            const float expected =
                scalar_dot(n / 32, n % 32, q8_row) + (PTQ1_WORKER_BIAS ? bias_vec[n] : 0.0f);
            if (!(fabsf(outputs[ir][n] - expected) <= 0.05f)) {
                printf("mismatch row %u col %u: got %.3f expected %.3f\n", ir, n, outputs[ir][n],
                       expected);
                return 6;
            }
            checksum += outputs[ir][n];
        }
    }

    const uint64_t hash = bytes_hash(outputs, sizeof(outputs));
    printf("PTQ1 multirow op_matmul K=%d M=%d N=%d threads=%d bias=%d checksum %.1f hash %016llx\n",
           K, M, N, NTHREADS, PTQ1_WORKER_BIAS, checksum, (unsigned long long) hash);
    return 0;
}
