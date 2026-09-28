// SPDX-License-Identifier: BSD-3-Clause
// Exercise the production HMX FlashAttention path with its real async queue.
#include "flash-attn-ops.c"
#include "qurt_memory.h"
#include "qurt_hvx.h"
#include "qurt_hmx.h"
#include "HAP_perf.h"

#ifndef HMX_FA_KV_LEN
#define HMX_FA_KV_LEN 64
#endif
#ifndef HMX_FA_Q_LEN
#define HMX_FA_Q_LEN 1
#endif
#ifndef HMX_FA_REPEATS
#define HMX_FA_REPEATS 1
#endif
#ifndef HMX_FA_USE_MASK
#define HMX_FA_USE_MASK 0
#endif
#ifndef HMX_FA_PIPELINE
#define HMX_FA_PIPELINE 0
#endif
#ifndef HMX_FA_MASK_PER_HEAD
#define HMX_FA_MASK_PER_HEAD 0
#endif

enum { DK = 256, DV = 256, G = 8, L = HMX_FA_KV_LEN, Q = HMX_FA_Q_LEN,
       OUT_STRIDE = DV + 16, MASK_STRIDE = ((L + 63) / 64) * 64 };

static __fp16 query[G][Q][DK] __attribute__((aligned(128)));
static __fp16 keys[L][DK] __attribute__((aligned(128)));
static __fp16 values[L][DV] __attribute__((aligned(128)));
static __fp16 output[Q][G][OUT_STRIDE] __attribute__((aligned(128)));
#if HMX_FA_USE_MASK
static __fp16 mask_data[HMX_FA_MASK_PER_HEAD ? G : 1][Q][MASK_STRIDE] __attribute__((aligned(128)));
#endif
static uint8_t dma_storage[65536] __attribute__((aligned(128)));
static uint8_t hmx_storage[65536] __attribute__((aligned(128)));

int compute_resource_hmx_lock(unsigned int context_id) {
    (void) context_id;
    int rc = qurt_hmx_lock();
    if (rc != QURT_EOK) abort();
    return rc;
}

int compute_resource_hmx_unlock(unsigned int context_id) {
    (void) context_id;
    return qurt_hmx_unlock();
}

// The operator uses one HVX thread here. A call into a larger worker pool is a test error.
bool work_queue_run_async(work_queue_t q, work_queue_func_t func, void * data, unsigned n) {
    (void) q; (void) func; (void) data; (void) n;
    assert(0);
    return false;
}

static void fill_inputs(void) {
    for (unsigned h = 0; h < G; ++h) {
        for (unsigned t = 0; t < Q; ++t) {
            for (unsigned d = 0; d < DK; ++d) {
                query[h][t][d] = (__fp16) (((int) ((d * 3 + h * 11 + t * 7) % 29) - 14) / 32.0f);
            }
        }
    }
    for (unsigned k = 0; k < L; ++k) {
        for (unsigned d = 0; d < DK; ++d) {
            keys[k][d] = (__fp16) (((int) ((d * 7 + k * 5) % 31) - 15) / 32.0f);
            values[k][d] = (__fp16) (((int) ((d * 11 + k * 3) % 37) - 18) / 32.0f);
        }
    }
#if HMX_FA_USE_MASK
    for (unsigned h = 0; h < (HMX_FA_MASK_PER_HEAD ? G : 1); ++h) {
        for (unsigned t = 0; t < Q; ++t) {
            for (unsigned k = 0; k < L; ++k) {
                mask_data[h][t][k] = k <= L - Q + t ? (__fp16) ((k + h) % 3 == 0 ? -0.25f : 0.0f) : -INFINITY;
            }
        }
    }
#endif
    for (unsigned t = 0; t < Q; ++t) {
        for (unsigned h = 0; h < G; ++h) {
            for (unsigned d = 0; d < OUT_STRIDE; ++d) output[t][h][d] = (__fp16) 123.0f;
        }
    }
}

static bool check_output(void) {
    const float tolerance = 0.001f;
    float max_error = 0.0f;
    float max_expected = 0.0f;
    for (unsigned t = 0; t < Q; ++t) {
        for (unsigned h = 0; h < G; ++h) {
            float max_score = -INFINITY;
            float scores[L];
            for (unsigned k = 0; k < L; ++k) {
                float dot = 0.0f;
                for (unsigned d = 0; d < DK; ++d) dot += (float) query[h][t][d] * (float) keys[k][d];
                scores[k] = dot / 16.0f;
#if HMX_FA_USE_MASK
                scores[k] += (float) mask_data[HMX_FA_MASK_PER_HEAD ? h : 0][t][k];
#endif
                if (scores[k] > max_score) max_score = scores[k];
            }
            float denominator = 0.0f;
            for (unsigned k = 0; k < L; ++k) {
                scores[k] = expf(scores[k] - max_score);
                denominator += scores[k];
            }
            for (unsigned d = 0; d < DV; ++d) {
                float numerator = 0.0f;
                for (unsigned k = 0; k < L; ++k) numerator += scores[k] * (float) values[k][d];
                float expected = numerator / denominator;
                float error = fabsf((float) output[t][h][d] - expected);
                if (error > max_error) max_error = error;
                if (fabsf(expected) > max_expected) max_expected = fabsf(expected);
                if (!(error <= tolerance)) {
                    printf("HMX FA mismatch t=%u h=%u d=%u got=%g expected=%g\n",
                           t, h, d, (float) output[t][h][d], expected);
                    return false;
                }
            }
            for (unsigned d = DV; d < OUT_STRIDE; ++d) {
                if ((float) output[t][h][d] != 123.0f) return false;
            }
        }
    }
    printf("HMX FA scalar max_error=%g max_expected=%g tolerance=%g\n", max_error, max_expected, tolerance);
    return max_expected > tolerance;
}

int main(void) {
    fill_inputs();
    struct htp_tensor q = { .data = (uint32_t) (uintptr_t) query, .type = HTP_TYPE_F16,
        .ne = {DK, Q, G, 1}, .nb = {2, DK * 2, Q * DK * 2, G * Q * DK * 2} };
    struct htp_tensor k = { .data = (uint32_t) (uintptr_t) keys, .type = HTP_TYPE_F16,
        .ne = {DK, L, 1, 1}, .nb = {2, DK * 2, L * DK * 2, L * DK * 2} };
    struct htp_tensor v = { .data = (uint32_t) (uintptr_t) values, .type = HTP_TYPE_F16,
        .ne = {DV, L, 1, 1}, .nb = {2, DV * 2, L * DV * 2, L * DV * 2} };
    struct htp_tensor dst = { .data = (uint32_t) (uintptr_t) output, .type = HTP_TYPE_F16,
        .ne = {DV, G, Q, 1}, .nb = {2, OUT_STRIDE * 2, G * OUT_STRIDE * 2, Q * G * OUT_STRIDE * 2} };
#if HMX_FA_USE_MASK
    struct htp_tensor mask = { .data = (uint32_t) (uintptr_t) mask_data, .type = HTP_TYPE_F16,
        .ne = {L, Q, HMX_FA_MASK_PER_HEAD ? G : 1, 1},
        .nb = {2, MASK_STRIDE * 2, Q * MASK_STRIDE * 2, (HMX_FA_MASK_PER_HEAD ? G : 1) * Q * MASK_STRIDE * 2} };
#endif
    struct htp_context ctx = {0};
    struct htp_ops_context octx = { .ctx = &ctx, .src = {&q, &k, &v,
#if HMX_FA_USE_MASK
        &mask
#endif
        }, .dst = &dst, .n_threads = 1 };
    struct htp_fa_kernel_params * params = (struct htp_fa_kernel_params *) octx.kernel_params;
    const unsigned bc = 64;
    params->kernel_type = HTP_FA_KERNEL_HMX;
    params->n_threads = 1;
    params->Br = 4;
    params->Bc = bc;
    params->n_kv_blocks = (L + bc - 1) / bc;
    params->G = G;
    params->scale = 1.0f / 16.0f;
    params->m0 = params->m1 = 1.0f;
    params->n_head_log2 = G;
    params->broadcast_rk3 = init_fastdiv_values(1);
    params->broadcast_rv3 = init_fastdiv_values(1);
    params->u.hmx.g_br = 32;
    params->u.hmx.pipeline = HMX_FA_PIPELINE;
    params->u.hmx.div_G = init_fastdiv_values(G);
    params->u.hmx.row_buf_stride = hex_align_up(bc * sizeof(__fp16), 256) / 128;
    params->u.hmx.mask_buf_row_stride = hex_align_up(bc * sizeof(__fp16), 128) / sizeof(__fp16);
#if HMX_FA_USE_MASK
    params->u.hmx.mask_broadcast = !HMX_FA_MASK_PER_HEAD;
    params->src3_div2 = init_fastdiv_values(HMX_FA_MASK_PER_HEAD ? G : 1);
    params->src3_div3 = init_fastdiv_values(1);
#endif

    struct hmx_fa_vtcm_layout layout;
    hmx_fa_vtcm_layout_build(&layout, G, DK, DV, params->Br, bc, 1, HMX_FA_PIPELINE, false);
    if (dma_queue_sizeof(64) > sizeof(dma_storage) ||
        hmx_queue_sizeof(8, 32768) > sizeof(hmx_storage)) return 1;
    qurt_mem_pool_t vtcm_pool;
    qurt_mem_region_t vtcm_region;
    qurt_mem_region_attr_t vtcm_attr;
    if (qurt_mem_pool_attach("TCM_PHYSPOOL", &vtcm_pool) != QURT_EOK) return 1;
    qurt_mem_region_attr_init(&vtcm_attr);
    qurt_mem_region_attr_set_cache_mode(&vtcm_attr, QURT_MEM_CACHE_NONE_SHARED);
    qurt_mem_region_attr_set_mapping(&vtcm_attr, QURT_MEM_MAPPING_PHYS_CONTIGUOUS);
    qurt_mem_region_attr_set_physaddr(&vtcm_attr, 0xd9000000u);
    if (qurt_mem_region_create(&vtcm_region, hex_align_up(layout.total_bytes + 128, 4096),
                               vtcm_pool, &vtcm_attr) != QURT_EOK) return 1;
    if (qurt_mem_region_attr_get(vtcm_region, &vtcm_attr) != QURT_EOK) return 1;
    unsigned vtcm_addr;
    unsigned vtcm_phys;
    qurt_mem_region_attr_get_virtaddr(&vtcm_attr, &vtcm_addr);
    qurt_mem_region_attr_get_physaddr(&vtcm_attr, &vtcm_phys);
    uint8_t * vtcm = (uint8_t *) (uintptr_t) vtcm_addr;
    params->vtcm_size = layout.total_bytes;
    ctx.vtcm_base = vtcm;
    ctx.vtcm_size = layout.total_bytes;
    ctx.hmx_enabled = true;
    memset(vtcm, 0xa5, layout.total_bytes + 128);
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) return 1;
    ctx.dma[0] = dma_queue_init(dma_storage, 64, (uintptr_t) vtcm, layout.total_bytes, &ctx.trace[0]);
    ctx.hmx_queue = hmx_queue_init(hmx_storage, 8, 32768, 0, &ctx.trace[HTP_MAX_NTHREADS]);
    if (!ctx.hmx_queue) return 2;
    printf("HMX FA fixture vtcm=%p phys=%08x bytes=%zu k_dma=%zu k_tiles=%zu v_dma=%zu v_tiles=%zu\n",
           vtcm, vtcm_phys, layout.total_bytes, layout.off_k_fp16[0], layout.off_k_tiles[0],
           layout.off_v_fp16[0], layout.off_v_tiles[0]);
    fflush(stdout);
    for (unsigned repeat = 0; repeat < HMX_FA_REPEATS; ++repeat) {
        uint64_t start = HAP_perf_get_pcycles();
        if (hmx_flash_attn_ext(&octx) != HTP_STATUS_OK) return 3;
        printf("HMX FA sample repeat=%u pcycles=%llu\n", repeat,
               (unsigned long long) (HAP_perf_get_pcycles() - start));
    }
    hmx_queue_free(ctx.hmx_queue);
    qurt_hvx_unlock();
    for (unsigned i = 0; i < 128; ++i) if (vtcm[layout.total_bytes + i] != 0xa5) return 4;
    if (!check_output()) return 5;
    uint64_t hash = UINT64_C(14695981039346656037);
    const uint8_t * bytes = (const uint8_t *) output;
    for (size_t i = 0; i < sizeof(output); ++i) hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    printf("HMX FA worker L=%d Q=%d mask=%d per_head=%d pipeline=%d repeats=%d vtcm=%zu hash %016llx\n",
           L, Q, HMX_FA_USE_MASK, HMX_FA_MASK_PER_HEAD, HMX_FA_PIPELINE,
           HMX_FA_REPEATS, layout.total_bytes, (unsigned long long) hash);
    qurt_mem_region_delete(vtcm_region);
    return 0;
}
