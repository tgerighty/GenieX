// SPDX-License-Identifier: BSD-3-Clause
// Product PTQ1 HMX block route with real QuRT, DMA, and HMX queues.
#include "hmx-queue.h"
static bool fixture_observe_hmx_queue_push(hmx_queue_t q, struct hmx_queue_desc d);
#define hmx_queue_push(q, d) fixture_observe_hmx_queue_push((q), (d))
#include "matmul-ops.c"
#undef hmx_queue_push
#include "hexagon_sim_timer.h"
#include "qurt_memory.h"
#include "qurt_hvx.h"
#include "qurt_hmx.h"

#include <stdio.h>
#include <stdarg.h>
#include <stddef.h>
#include <string.h>

static hmx_queue_t observed_queue;
static atomic_uint observed_jobs;
static atomic_uint unexpected_jobs;
#if PTQ1_QUEUE_ERROR_ONLY
static atomic_int fault_mode; // 1: fail one push; 2: queue one real no-op first.
static atomic_uint fault_seen;
#endif
static bool fixture_observe_hmx_queue_push(hmx_queue_t q, struct hmx_queue_desc d) {
#if PTQ1_QUEUE_ERROR_ONLY
    if (q == observed_queue && d.func == ptq1_hmx_dot_job_run) {
        const int mode = atomic_exchange(&fault_mode, 0);
        if (mode) {
            atomic_fetch_add(&fault_seen, 1);
            if (mode == 1) return false;
            if (!hmx_queue_push(q, hmx_queue_make_desc((hmx_queue_func) HMX_QUEUE_NOOP, NULL)))
                return false;
        }
    }
#endif
    const bool pushed = hmx_queue_push(q, d);
    if (q == observed_queue && pushed) {
        if (d.func == ptq1_hmx_dot_job_run && d.data != NULL)
            atomic_fetch_add(&observed_jobs, 1);
        else
            atomic_fetch_add(&unexpected_jobs, 1);
    }
    return pushed;
}

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

int compute_resource_hmx_lock(unsigned int context_id) {
    (void) context_id;
    return qurt_hmx_lock();
}
int compute_resource_hmx_unlock(unsigned int context_id) {
    (void) context_id;
    return qurt_hmx_unlock();
}

#ifndef PTQ1_WORKER_K
#define PTQ1_WORKER_K 5120
#endif
#ifndef PTQ1_WORKER_M
#define PTQ1_WORKER_M 30
#endif
#ifndef PTQ1_WORKER_N
#define PTQ1_WORKER_N 256
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
#ifndef PTQ1_HMX_AUX_REJECT
#define PTQ1_HMX_AUX_REJECT 0
#endif
#ifndef PTQ1_WORKER_BATCH
#define PTQ1_WORKER_BATCH 1
#endif
#ifndef PTQ1_WORKER_PADDED_W
#define PTQ1_WORKER_PADDED_W 0
#endif
#ifndef PTQ1_WORKER_RANDOM_ACT
#define PTQ1_WORKER_RANDOM_ACT 0
#endif
#ifndef PTQ1_BREADTH_CASE
#define PTQ1_BREADTH_CASE 0
#endif
#ifndef PTQ1_BREADTH_EXPECT_HASH
#define PTQ1_BREADTH_EXPECT_HASH UINT64_C(0)
#endif
#ifndef PTQ1_QUEUE_ERROR_ONLY
#define PTQ1_QUEUE_ERROR_ONLY 0
#endif
_Static_assert(PTQ1_BREADTH_CASE >= 0 && PTQ1_BREADTH_CASE <= 4,
               "unsupported breadth case");

enum {
    K          = PTQ1_WORKER_K,
    M          = PTQ1_WORKER_M,
    N          = PTQ1_WORKER_N,
    NTHREADS   = PTQ1_WORKER_THREADS,
    ROW_STRIDE = N + 16,
    TILES      = (N + 31) / 32,
    KB         = K / GENIEX_PTQ1_BLOCK_K
};
_Static_assert(PTQ1_WORKER_BATCH && NTHREADS >= 1 && NTHREADS <= 4 &&
                   (K == 5120 || K == 6144 || K == 17408),
               "HMX width screen requires K5120, K6144, or K17408 true batch and one to four workers");

static geniex_ptq1_tile weights[TILES][KB] __attribute__((aligned(128)));
static geniex_ptq1_block blocks[GENIEX_PTQ1_TILE_ROWS * KB];
static geniex_ptq1_block anchor_blocks[GENIEX_PTQ1_TILE_ROWS * KB];
static float activations[M][K] __attribute__((aligned(128)));
static float activation_copy[M][K];
static float bias_vec[N] __attribute__((aligned(128)));
static float outputs[M][ROW_STRIDE] __attribute__((aligned(128)));
static float warmup_outputs[M][ROW_STRIDE] __attribute__((aligned(128)));
static uint8_t accepted_q8_rows[M][K + ((K / 16 + 127) & ~127u)] __attribute__((aligned(128)));
static uint8_t *vtcm;
static uint8_t queue_storage[NTHREADS][32768] __attribute__((aligned(128)));
static uint8_t alias_storage[NTHREADS][256] __attribute__((aligned(128)));
static uint8_t work_queue_storage[65536] __attribute__((aligned(4096)));
static uint8_t hmx_queue_storage[40960] __attribute__((aligned(128)));
static unsigned char *aux_base;
static size_t aux_bytes;
static uint64_t bytes_hash(const void * p, size_t n) {
    uint64_t hash = UINT64_C(14695981039346656037);
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) {
        hash = (hash ^ b[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

#ifndef PTQ1_EXPECT_DIRECT_PACK
#define PTQ1_EXPECT_DIRECT_PACK 0
#endif

static uint16_t signed_q8_half_bits(int8_t value) {
    if (!value) return 0;
    const unsigned magnitude = value < 0 ? -(int)value : value;
    unsigned leading = 0;
    while (magnitude >> (leading + 1)) ++leading;
    return (uint16_t)((value < 0 ? 0x8000u : 0u) |
                      ((leading + 15) << 10) |
                      ((magnitude << (10 - leading)) & 0x3ffu));
}

static int check_activation_slots(void) {
    static struct {
        uint8_t before[2048];
        struct ptq1_hmx_scratch body;
        uint8_t after[128];
    } guarded;
    struct ptq1_hmx_scratch *scratch = &guarded.body;
    static struct {
        uint8_t before[128], rows[32][17537], after[128];
    } source;
    static const int8_t extremes[] = {-128, -127, -1, 0, 1, 127};
    static const unsigned parts[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 319, 383, 1087};
    _Static_assert(sizeof(*scratch) == 110592, "HMX scratch size changed");
    memset(&source, 0xa5, sizeof(source));
    for (unsigned r = 0; r < 32; ++r)
        for (unsigned p = 0; p < sizeof(parts) / sizeof(parts[0]); ++p)
            for (unsigned k = 0; k < 16; ++k)
                source.rows[r][1 + parts[p] * 16 + k] = (uint8_t)(k < 6 ?
                    extremes[(k + r + p) % 6] : (int8_t)(r * 7 + p * 19 + k * 11));
    const uint64_t source_hash = bytes_hash(&source, sizeof(source));
    for (unsigned rows = 0; rows <= 32; ++rows) {
        for (unsigned p = 0; p < sizeof(parts) / sizeof(parts[0]); ++p) {
            const unsigned part = parts[p];
            const unsigned slot_index = part % PTQ1_HMX_PARTS;
            uint16_t expected[PTQ1_HMX_TILE / 2] = {0};
            for (unsigned r = 0; r < rows; ++r)
                for (unsigned k = 0; k < 16; ++k)
                    expected[(r / 2) * 64 + k * 2 + (r & 1)] = signed_q8_half_bits(
                        (int8_t)source.rows[r][1 + part * 16 + k]);
            memset(&guarded, 0xa5, sizeof(guarded));
            ptq1_hmx_pack_activation(scratch, part, slot_index,
                                     rows ? &source.rows[0][1] : NULL,
                                     sizeof(source.rows[0]), rows);
            const unsigned offset = PTQ1_HMX_ACT_OFF + slot_index * PTQ1_HMX_TILE;
            uint8_t *slot = scratch->vtcm + offset;
            if (memcmp(slot, expected, sizeof(expected))) {
                printf("PTQ1_HMX_SLOT_FAIL rows=%u part=%u slot=%u bytes\n",
                       rows, part, slot_index);
                return 0;
            }
            const uint8_t *guarded_bytes = (const uint8_t *)&guarded;
            const size_t body_start = (const uint8_t *)scratch - guarded_bytes;
            const size_t act_start = body_start + offsetof(struct ptq1_hmx_scratch, act_rows);
            const size_t act_end = body_start + offsetof(struct ptq1_hmx_scratch, decoded);
            for (size_t i = 0; i < sizeof(guarded); ++i)
                if ((i < body_start + offset || i >= body_start + offset + PTQ1_HMX_TILE) &&
                    (PTQ1_EXPECT_DIRECT_PACK || i < act_start || i >= act_end) &&
                    guarded_bytes[i] != 0xa5) {
                    printf("PTQ1_HMX_SLOT_FAIL rows=%u part=%u scratch_offset=%zu\n",
                           rows, part, i);
                    return 0;
                }
            if (rows == 0 && p == 0) {
                slot[0] ^= 1;
                if (!memcmp(slot, expected, sizeof(expected))) return 0;
                slot[0] ^= 1;
                puts("PTQ1_HMX_SLOT_NEGATIVE_CONTROL_PASS");
            }
            printf("PTQ1_HMX_SLOT_BYTES rows=%u part=%u slot=%u bytes=%u hash=%016llx\n",
                   rows, part, slot_index, PTQ1_HMX_TILE,
                   (unsigned long long)bytes_hash(slot, PTQ1_HMX_TILE));
        }
    }
    if (bytes_hash(&source, sizeof(source)) != source_hash) return 0;
    puts("PTQ1_HMX_SLOT_BYTES_PASS cases=429 scratch_bytes=110592 source_unchanged=1");
    return 1;
}

static void set_row_params(struct htp_mm_kernel_params *params,
                           const struct htp_mm_hvx_vtcm_layout *layout, size_t q8_stride) {
    params->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;
    params->src1_row_size = q8_stride;
    params->vtcm_size = layout->total_bytes;
    params->vtcm_src0_size = layout->src0_bytes;
    params->vtcm_src1_size = layout->src1_bytes;
    params->vtcm_src2_size = layout->src2_bytes;
    params->vtcm_src3_size = layout->src3_bytes;
    params->vtcm_dst_size = layout->dst_bytes;
    params->n_prefetch = 2;
    params->n_hmx = 0;
    params->div_ne12_ne1 = init_fastdiv_values(1);
    params->div_ne1 = init_fastdiv_values(1);
    params->div_r2 = init_fastdiv_values(1);
    params->div_r3 = init_fastdiv_values(1);
    params->div_ne11 = init_fastdiv_values(1);
    params->div_n_act_threads = init_fastdiv_values(1);
    params->div_ne00_padded = init_fastdiv_values(1);
}

static int aux_guards_ok(void) {
    // Eight-part arrays use all bytes before the column scales.
    const unsigned offsets[] = {PTQ1_HMX_SCALE_OFF + 512,
                                PTQ1_HMX_SCALE_OFF + 256, PTQ1_HMX_VTCM_BYTES - 128};
    for (unsigned t = 0; t < NTHREADS; ++t) {
        const struct ptq1_hmx_scratch *scratch =
            (const struct ptq1_hmx_scratch *)(aux_base + t * sizeof(*scratch));
        for (unsigned g = 0; g < sizeof(offsets) / sizeof(offsets[0]); ++g)
            for (unsigned i = 0; i < 128; ++i)
                if (scratch->vtcm[offsets[g] + i] != 0xa5) return 0;
    }
    for (unsigned i = 0; i < 128; ++i)
        if (aux_base[aux_bytes + i] != 0xa5) return 0;
    return 1;
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

static int count_different_seed7_tiles(void) {
    uint32_t anchor_seed = 7;
    for (unsigned row = 0; row < GENIEX_PTQ1_TILE_ROWS * KB; ++row) {
        for (unsigned k = 0; k < 24; ++k) {
            anchor_seed = anchor_seed * 1664525u + 1013904223u;
            anchor_blocks[row].qs[k] = anchor_seed >> 24;
        }
        for (unsigned k = 0; k < 2; ++k) {
            anchor_seed = anchor_seed * 1664525u + 1013904223u;
            anchor_blocks[row].qh[k] = anchor_seed >> 24;
        }
        anchor_blocks[row].d = 0x3800 + (row % 3) * 0x400;
    }
    unsigned different = 0;
    for (unsigned ct = 0; ct < TILES; ++ct) {
        const unsigned valid_rows =
            ct + 1 == TILES ? N - ct * GENIEX_PTQ1_TILE_ROWS : GENIEX_PTQ1_TILE_ROWS;
        for (unsigned kb = 0; kb < KB; ++kb) {
            geniex_ptq1_tile anchor_tile;
            geniex_ptq1_pack_tile(&anchor_tile, anchor_blocks, KB, kb, valid_rows);
            anchor_tile.qs[0][ct % valid_rows] ^= ct + kb + 1;
            different += memcmp(&weights[ct][kb], &anchor_tile, sizeof(anchor_tile)) != 0;
        }
    }
    return (int) different;
}

int main(void) {
    if (ptq1_hmx_eligible(5120, 3, 1, 1) ||
        !ptq1_hmx_eligible(5120, 4, 1, 1) ||
        !ptq1_hmx_eligible(5120, 32, 1, 4) ||
        ptq1_hmx_eligible(5120, 33, 1, 1) ||
        ptq1_hmx_eligible(5121, 4, 1, 1) ||
        ptq1_hmx_eligible(6144, 1, 1, 1) ||
        ptq1_hmx_eligible(6144, 3, 1, 1) ||
        !ptq1_hmx_eligible(6144, 4, 1, 1) ||
        !ptq1_hmx_eligible(6144, 32, 1, 4) ||
        ptq1_hmx_eligible(6144, 33, 1, 1) ||
        ptq1_hmx_eligible(6145, 4, 1, 1) ||
        ptq1_hmx_eligible(17407, 4, 1, 1) ||
        ptq1_hmx_eligible(17408, 3, 1, 1) ||
        !ptq1_hmx_eligible(17408, 4, 1, 1) ||
        !ptq1_hmx_eligible(17408, 32, 1, 4) ||
        ptq1_hmx_eligible(17408, 33, 1, 1) ||
        ptq1_hmx_eligible(17409, 4, 1, 1) ||
        ptq1_hmx_eligible(5120, 4, 0, 1) ||
        ptq1_hmx_eligible(5120, 4, 1, 0) ||
        ptq1_hmx_eligible(5120, 4, 1, 5)) {
        puts("PTQ1_HMX_ADMISSION_BOUNDARY_FAIL");
        return 33;
    }
    puts("PTQ1_HMX_ADMISSION_BOUNDARIES_PASS");
    uint32_t seed = PTQ1_BREADTH_CASE == 1 ? 12345u : 7u;
    static const uint16_t breadth_scales[] = {
        0x3555, 0x3bff, 0x0400, 0x0001, 0x7bff, 0xb2ab
    };
    for (unsigned row = 0; row < GENIEX_PTQ1_TILE_ROWS * KB; ++row) {
        for (unsigned k = 0; k < 24; ++k) {
            seed = seed * 1664525u + 1013904223u;
            blocks[row].qs[k] = seed >> 24;
        }
        for (unsigned k = 0; k < 2; ++k) {
            seed = seed * 1664525u + 1013904223u;
            blocks[row].qh[k] = seed >> 24;
        }
        blocks[row].d = PTQ1_BREADTH_CASE == 4 ? breadth_scales[row % 6] :
                        0x3800 + (row % 3) * 0x400;
    }
    for (unsigned ct = 0; ct < TILES; ++ct) {
        const unsigned valid_rows =
            ct + 1 == TILES ? N - ct * GENIEX_PTQ1_TILE_ROWS : GENIEX_PTQ1_TILE_ROWS;
        for (unsigned kb = 0; kb < KB; ++kb) {
            geniex_ptq1_pack_tile(&weights[ct][kb], blocks, KB, kb, valid_rows);
            weights[ct][kb].qs[0][ct % valid_rows] ^= ct + kb + 1;
        }
    }
    if (PTQ1_BREADTH_CASE == 1) {
        const int different = count_different_seed7_tiles();
        printf("PTQ1_BREADTH_SEED_DIFF tiles=%d/%d\n", different, TILES * KB);
        if (different != TILES * KB) return 28;
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
    printf("PTQ1_BREADTH_INPUT case=%d weight_hash=%016llx act_hash=%016llx\n",
           PTQ1_BREADTH_CASE, (unsigned long long) weight_hash,
           (unsigned long long) act_hash);

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
    kparams->kernel_type = HTP_MM_KERNEL_HVX_PTQ1_BATCH;
#endif
    kparams->n_prefetch  = 2;

    // K6144 M1 and M33 use the accepted row fallback and one-row VTCM layout.
    struct htp_mm_hvx_vtcm_layout layout;
    const unsigned layout_rows = M <= 32 ? M : 1;
    if (M > 32 || (K == 6144 && M == 1))
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;
    htp_mm_hvx_vtcm_layout_build(
        &layout, kparams->kernel_type, w.type, K, layout_rows, NTHREADS, y.nb[1], w.nb[1],
        htp_mm_q8_0_flat_row_size(K), PTQ1_WORKER_BIAS ? b.nb[1] : 0, 2, false, false, false);
    const size_t q8_stride = htp_mm_q8_0_flat_row_size(K);
    const size_t expected_q8_stride = K + ((K / 16 + 127) & ~127u);
    if (q8_stride != expected_q8_stride) return 21;
    if (K == 6144 && (M == 1 || M > 32))
        set_row_params(kparams, &layout, q8_stride);
    const size_t flat_bytes = layout_rows * q8_stride;
    if (layout.off_src1 + flat_bytes > layout.total_bytes ||
        flat_bytes > layout.src1_bytes) return 29;
    printf("PTQ1 truebatch layout bytes=%zu q8stride=%zu rows=%d tiles=%d\n",
           layout.total_bytes, q8_stride, M, TILES);
    const size_t hmx_offset = (layout.total_bytes + 2047) & ~(size_t)2047;
    aux_bytes = NTHREADS * sizeof(struct ptq1_hmx_scratch);
    const size_t aux_end = hmx_offset + aux_bytes;
    const size_t allocation = (aux_end + 128 + 4095) & ~(size_t)4095;
    if (allocation > 8 * 1024 * 1024 ||
        dma_queue_sizeof(256) > sizeof(queue_storage[0]) ||
        dma_queue_alias_sizeof() > sizeof(alias_storage[0]) ||
        work_queue_sizeof(NTHREADS, 16, 16384) > sizeof(work_queue_storage) ||
        hmx_queue_sizeof(8, 32768) > sizeof(hmx_queue_storage)) {
        printf("PTQ1 multirow VTCM layout %zu exceeds fixture buffer\n", layout.total_bytes);
        return 1;
    }
    qurt_mem_pool_t pool;
    qurt_mem_region_t region;
    qurt_mem_region_attr_t attr;
    if (qurt_mem_pool_attach("TCM_PHYSPOOL", &pool) != QURT_EOK) return 12;
    qurt_mem_region_attr_init(&attr);
    qurt_mem_region_attr_set_cache_mode(&attr, QURT_MEM_CACHE_NONE_SHARED);
    qurt_mem_region_attr_set_mapping(&attr, QURT_MEM_MAPPING_PHYS_CONTIGUOUS);
    qurt_mem_region_attr_set_physaddr(&attr, 0xd9000000u);
    if (qurt_mem_region_create(&region, allocation, pool, &attr) != QURT_EOK) return 13;
    if (qurt_mem_region_attr_get(region, &attr) != QURT_EOK) return 14;
    unsigned addr;
    qurt_mem_region_attr_get_virtaddr(&attr, &addr);
    vtcm = (uint8_t *)(uintptr_t)addr;
    kparams->vtcm_size = layout.total_bytes;
    uint8_t kernel_params_before[sizeof(octx.kernel_params)];
    memcpy(kernel_params_before, octx.kernel_params, sizeof(kernel_params_before));
    memset(vtcm, 0xa5, allocation);
    aux_base = vtcm + hmx_offset;
    const uint64_t clean_aux_hash = bytes_hash(aux_base, aux_bytes);
    printf("PTQ1 candidate extra VTCM offset=%zu bytes=%zu allocation=%zu\n",
           hmx_offset, aux_bytes, allocation);
    ctx.vtcm_base = vtcm;
#if PTQ1_WORKER_VTCM_REJECT
    ctx.vtcm_size = layout.total_bytes - 1;
#else
    ctx.vtcm_size = PTQ1_HMX_AUX_REJECT ? aux_end - 1 : aux_end;
#endif
    ctx.work_queue = work_queue_init(work_queue_storage, NTHREADS, 16, 16384);
    if (!ctx.work_queue) return 15;
    for (unsigned t = 0; t < NTHREADS; ++t) {
        ctx.dma_cached[t] = dma_queue_init(
            queue_storage[t], 256, (uintptr_t)vtcm, layout.total_bytes, &ctx.trace[t]);
        ctx.dma[t] = dma_queue_alias_init(alias_storage[t], ctx.dma_cached[t], 1);
    }
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) return 16;
    if (!check_activation_slots()) return 39;
    hmx_queue_t hmx = hmx_queue_init(hmx_queue_storage, 8, 32768, 0,
                                     &ctx.trace[HTP_MAX_NTHREADS]);
    if (!hmx) return 22;
    work_queue_wakeup(ctx.work_queue);
    observed_queue = hmx;
    atomic_store(&observed_jobs, 0);
    atomic_store(&unexpected_jobs, 0);

    ctx.hmx_queue = NULL;
    struct htp_mm_hvx_vtcm_layout row_layout;
    htp_mm_hvx_vtcm_layout_build(&row_layout, HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT,
        w.type, K, 1, NTHREADS, y.nb[1], w.nb[1], q8_stride,
        PTQ1_WORKER_BIAS ? b.nb[1] : 0, 2, false, false, false);
    if (K == 6144) {
        if (row_layout.off_src1 + q8_stride > row_layout.total_bytes ||
            q8_stride > row_layout.src1_bytes || row_layout.total_bytes > ctx.vtcm_size)
            return 36;
        for (unsigned r = 0; r < M; ++r) {
            struct htp_tensor row_x = x, row_y = y;
            row_x.ne[1] = row_y.ne[1] = 1;
            row_x.data += r * x.nb[1];
            row_y.data += r * y.nb[1];
            struct htp_ops_context row_octx = octx;
            row_octx.src[1] = &row_x;
            row_octx.dst = &row_y;
            struct htp_mm_kernel_params *row_params =
                (struct htp_mm_kernel_params *)row_octx.kernel_params;
            set_row_params(row_params, &row_layout, q8_stride);
            if (op_matmul(&row_octx) != HTP_STATUS_OK) return 37;
            memcpy(accepted_q8_rows[r], vtcm + row_layout.off_src1, q8_stride);
        }
    } else if (op_matmul(&octx) != HTP_STATUS_OK) return 11;
    if (atomic_load(&observed_jobs) || atomic_load(&unexpected_jobs) ||
        memcmp(octx.kernel_params, kernel_params_before, sizeof(kernel_params_before)) != 0 ||
        bytes_hash(aux_base, aux_bytes) != clean_aux_hash) return 30;
    const int k6144_hmx_route = K == 6144 && ptq1_hmx_eligible(K, M, N, NTHREADS) &&
                                !PTQ1_HMX_AUX_REJECT;
    const size_t q8_report_bytes = K == 6144 && !k6144_hmx_route ? q8_stride : flat_bytes;
    const uint64_t accepted_q8_hash = K == 6144 ?
        bytes_hash(k6144_hmx_route ? (const void *)accepted_q8_rows :
                   (const void *)accepted_q8_rows[M - 1], q8_report_bytes) :
        bytes_hash(vtcm + layout.off_src1, flat_bytes);
    memcpy(warmup_outputs, outputs, sizeof(outputs));
    if (K == 6144) {
        for (unsigned r = 0; r < M; ++r)
            for (unsigned c = 0; c < ROW_STRIDE; ++c) outputs[r][c] = 12345.0f;
        if (op_matmul(&octx) != HTP_STATUS_OK ||
            memcmp(outputs, warmup_outputs, sizeof(outputs)) != 0 ||
            memcmp(vtcm + row_layout.off_src1, accepted_q8_rows[M - 1], q8_stride) != 0 ||
            atomic_load(&observed_jobs) || atomic_load(&unexpected_jobs) ||
            memcmp(octx.kernel_params, kernel_params_before, sizeof(kernel_params_before)) != 0 ||
            bytes_hash(aux_base, aux_bytes) != clean_aux_hash)
            return 38;
        puts("PTQ1_HMX_NO_QUEUE_FALLBACK_PASS output_padding_last_row_q8_exact=1");
    }
    const uint64_t accepted_hash = bytes_hash(outputs, sizeof(outputs));
    const uint64_t expected_hash = K == 5120 && PTQ1_BREADTH_CASE == 0 && M == 30 && N == 64 && NTHREADS == 1 ?
        UINT64_C(0x2744204fe1718231) : (uint64_t) PTQ1_BREADTH_EXPECT_HASH;
    const int pinned = expected_hash != 0;
    if (pinned && accepted_hash != expected_hash) {
        printf("accepted PR63 baseline hash changed: got %016llx expected %016llx\n",
               (unsigned long long) accepted_hash,
               (unsigned long long) expected_hash);
        return 23; // Do not time or claim a same-context comparison.
    }
    printf("PASS accepted baseline hash %016llx before candidate case=%d%s\n",
           (unsigned long long) accepted_hash, PTQ1_BREADTH_CASE,
           pinned ? " pinned" : " discovery");
    ctx.hmx_queue = hmx;
    atomic_store(&observed_jobs, 0);
    atomic_store(&unexpected_jobs, 0);
    for (unsigned r = 0; r < M; ++r)
        for (unsigned c = 0; c < ROW_STRIDE; ++c) outputs[r][c] = 12345.0f;
    const int candidate_status = op_matmul(&octx);
    const unsigned jobs = atomic_load(&observed_jobs);
    const unsigned unexpected = atomic_load(&unexpected_jobs);
    const int eligible = ptq1_hmx_eligible(K, M, N, NTHREADS) && !PTQ1_HMX_AUX_REJECT;
    const unsigned partition_rows = ((N + NTHREADS - 1) / NTHREADS + 31) & ~31u;
    const unsigned active_workers = eligible ? (N + partition_rows - 1) / partition_rows : 0;
    const unsigned expected_jobs = eligible ? TILES * KB : 0;
    printf("PTQ1_HMX_ROUTE jobs=%u expected_jobs=%u active_workers=%u unexpected=%u\n",
           jobs, expected_jobs, active_workers, unexpected);
    const int candidate_guard = aux_guards_ok();
    const uint64_t candidate_q8_hash = bytes_hash(
        K == 6144 && !eligible ? vtcm + row_layout.off_src1 : vtcm + layout.off_src1,
        q8_report_bytes);
    const int candidate_q8_equal = K == 6144 ?
        (eligible ? memcmp(vtcm + layout.off_src1, accepted_q8_rows, flat_bytes) == 0 :
                    memcmp(vtcm + row_layout.off_src1, accepted_q8_rows[M - 1], q8_stride) == 0) :
        accepted_q8_hash == candidate_q8_hash;
    printf("PTQ1_BREADTH_Q8 case=%d bytes=%zu accepted=%016llx candidate=%016llx\n",
           PTQ1_BREADTH_CASE, q8_report_bytes, (unsigned long long) accepted_q8_hash,
           (unsigned long long) candidate_q8_hash);
    if (candidate_status != HTP_STATUS_OK || unexpected || !candidate_guard ||
        jobs != expected_jobs ||
        !candidate_q8_equal ||
        memcmp(octx.kernel_params, kernel_params_before, sizeof(kernel_params_before)) != 0 ||
        memcmp(outputs, warmup_outputs, sizeof(outputs)) != 0) {
        int first = -1;
        uint32_t got = 0, ref = 0;
        for (unsigned i = 0; i < M * ROW_STRIDE; ++i) {
            if (memcmp(&outputs[0][0] + i, &warmup_outputs[0][0] + i, sizeof(float))) {
                first = (int) i;
                memcpy(&got, &outputs[0][0] + i, sizeof(got));
                memcpy(&ref, &warmup_outputs[0][0] + i, sizeof(ref));
                break;
            }
        }
        printf("PTQ1_BREADTH_MISMATCH status=%d guards=%d jobs=%u expected_jobs=%u unexpected=%u q8_equal=%d first=%d row=%d col=%d got=%08x ref=%08x\n",
               candidate_status, candidate_guard,
               jobs, expected_jobs, unexpected,
               candidate_q8_equal, first,
               first < 0 ? -1 : first / ROW_STRIDE,
               first < 0 ? -1 : first % ROW_STRIDE,
               (unsigned) got, (unsigned) ref);
        return 24;
    }
    if (K == 6144 && eligible)
        printf("PTQ1_HMX_Q8_BYTES_EXACT_PASS bytes=%zu\n", flat_bytes);
    uint32_t changed;
    memcpy(&changed, &outputs[0][0], sizeof(changed));
    changed ^= 1u;
    memcpy(&outputs[0][0], &changed, sizeof(changed));
    if (memcmp(outputs, warmup_outputs, sizeof(outputs)) == 0) return 19;
    changed ^= 1u;
    memcpy(&outputs[0][0], &changed, sizeof(changed));
    if (memcmp(outputs, warmup_outputs, sizeof(outputs)) != 0) return 20;
    puts("PASS changed output bit rejected and restored");
    printf("PASS candidate warmup %d active output words, padding, and VTCM guards\n", M * N);
    if (!eligible && bytes_hash(aux_base, aux_bytes) != clean_aux_hash) return 31;
    printf("PTQ1_HMX_AUX_FALLBACK forced=%d untouched=%d\n", PTQ1_HMX_AUX_REJECT,
           bytes_hash(aux_base, aux_bytes) == clean_aux_hash);

    for (size_t i = layout.total_bytes; i < hmx_offset; ++i) {
        if (vtcm[i] != 0xa5) {
            return 3;
        }
    }
    if (memcmp(octx.kernel_params, kernel_params_before, sizeof(kernel_params_before)) != 0 ||
        !tensors_equal(&w, &w0) || !tensors_equal(&x, &x0) || !tensors_equal(&y, &y0) ||
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
        for (unsigned n = 0; n < N; ++n) {
            checksum += outputs[ir][n];
        }
    }

    const uint64_t hash = bytes_hash(outputs, sizeof(outputs));
    printf("PTQ1_BREADTH_%s case=%d K=%d M=%d N=%d threads=%d bias=%d q8stride=%zu checksum %.1f hash=%016llx\n",
           pinned ? "PASS" : "DISCOVERY_MATCH",
           PTQ1_BREADTH_CASE, K, M, N, NTHREADS, PTQ1_WORKER_BIAS,
           q8_stride, checksum, (unsigned long long) hash);

    // Simulator Pcycles include the test-only queue observer. They are not board speed.
    // Timing requires an accepted-control hash for this exact width and shape.
    if (K != 6144 && !PTQ1_QUEUE_ERROR_ONLY && pinned && M == 30 && N == 256 && NTHREADS == 4 && PTQ1_BREADTH_CASE == 0 &&
        PTQ1_WORKER_BIAS && !PTQ1_WORKER_RANDOM_ACT && !PTQ1_HMX_AUX_REJECT) {
    const unsigned order[3][2] = {{0, 1}, {1, 0}, {0, 1}};
    for (unsigned pair = 0; pair < 3; ++pair) {
        for (unsigned turn = 0; turn < 2; ++turn) {
            const unsigned candidate = order[pair][turn];
            ctx.hmx_queue = candidate ? hmx : NULL;
            atomic_store(&observed_jobs, 0);
            atomic_store(&unexpected_jobs, 0);
            for (unsigned r = 0; r < M; ++r)
                for (unsigned c = 0; c < ROW_STRIDE; ++c) outputs[r][c] = 12345.0f;
            const uint64_t aux_before = bytes_hash(aux_base, aux_bytes);
            atomic_signal_fence(memory_order_seq_cst);
            const uint64_t start = hexagon_sim_read_pcycles();
            atomic_signal_fence(memory_order_seq_cst);
            const int timed_status = op_matmul(&octx);
            atomic_signal_fence(memory_order_seq_cst);
            const uint64_t stop = hexagon_sim_read_pcycles();
            atomic_signal_fence(memory_order_seq_cst);
            const unsigned timed_jobs = candidate ? expected_jobs : 0;
            if (timed_status != HTP_STATUS_OK ||
                atomic_load(&observed_jobs) != timed_jobs ||
                atomic_load(&unexpected_jobs) != 0 ||
                memcmp(outputs, warmup_outputs, sizeof(outputs)) != 0 ||
                bytes_hash(vtcm + layout.off_src1, flat_bytes) != accepted_q8_hash ||
                !aux_guards_ok() ||
                (!candidate && bytes_hash(aux_base, aux_bytes) != aux_before) ||
                !tensors_equal(&w, &w0) || !tensors_equal(&x, &x0) || !tensors_equal(&y, &y0) ||
                (PTQ1_WORKER_BIAS && !tensors_equal(&b, &b0)) ||
                bytes_hash(weights, sizeof(weights)) != weight_hash ||
                bytes_hash(activations, sizeof(activations)) != act_hash ||
                bytes_hash(bias_vec, sizeof(bias_vec)) != bias_hash ||
                memcmp(activations, activation_copy, sizeof(activations)) != 0) {
                printf("PTQ1_T4_TIMED_GATE_FAIL pair=%u turn=%u candidate=%u status=%d jobs=%u unexpected=%u\n",
                       pair + 1, turn + 1, candidate, timed_status,
                       atomic_load(&observed_jobs), atomic_load(&unexpected_jobs));
                return 32;
            }
            printf("PTQ1_T4_WORKER_PCYCLES pair=%u turn=%u arm=%s cycles=%llu\n",
                   pair + 1, turn + 1, candidate ? "candidate" : "base",
                   (unsigned long long)(stop - start));
        }
    }
    puts(pinned ? "PTQ1_T4_PAIRED_EXACT_PASS_INSTRUMENTED_NOT_BOARD_SPEED" :
                  "PTQ1_T4_PAIRED_DISCOVERY_INSTRUMENTED_NOT_BOARD_SPEED");
    } else {
        puts("PTQ1_T4_TIMING_SKIPPED_SHAPE_GATE_ONLY");
    }
#if PTQ1_QUEUE_ERROR_ONLY
    for (int mode = 1; mode <= 2; ++mode) {
        for (unsigned r = 0; r < M; ++r)
            for (unsigned c = 0; c < ROW_STRIDE; ++c) outputs[r][c] = 12345.0f;
        atomic_store(&fault_seen, 0);
        atomic_store(&fault_mode, mode);
        const int failed = op_matmul(&octx);
        if (failed != HTP_STATUS_INTERNAL_ERR || atomic_load(&fault_mode) != 0 ||
            atomic_load(&fault_seen) != 1 || !hmx_queue_empty(hmx) || !aux_guards_ok() ||
            memcmp(octx.kernel_params, kernel_params_before, sizeof(kernel_params_before)) != 0)
            return 34;
        printf("PTQ1_QUEUE_ERROR mode=%d status=INTERNAL_ERR consumed=1 drained=1\n", mode);
        for (unsigned r = 0; r < M; ++r)
            for (unsigned c = 0; c < ROW_STRIDE; ++c) outputs[r][c] = 12345.0f;
        if (op_matmul(&octx) != HTP_STATUS_OK || !hmx_queue_empty(hmx) ||
            memcmp(outputs, warmup_outputs, sizeof(outputs)) != 0 ||
            (K == 6144 ? memcmp(vtcm + layout.off_src1, accepted_q8_rows, flat_bytes) != 0 :
                         bytes_hash(vtcm + layout.off_src1, flat_bytes) != accepted_q8_hash) ||
            !aux_guards_ok() ||
            memcmp(octx.kernel_params, kernel_params_before, sizeof(kernel_params_before)) != 0)
            return 35;
        printf("PTQ1_QUEUE_RECOVERY mode=%d status=OK output_q8_guards_exact=1\n", mode);
    }
#endif
    work_queue_suspend(ctx.work_queue);
    hmx_queue_free(hmx);
    work_queue_free(ctx.work_queue);
    qurt_hvx_unlock();
    qurt_mem_region_delete(region);
    return 0;
}
