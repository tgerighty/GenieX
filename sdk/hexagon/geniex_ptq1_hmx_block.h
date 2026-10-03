// SPDX-License-Identifier: BSD-3-Clause
// PTQ1 HMX block route. Include after the accepted PTQ1 batch worker.
#ifndef GENIEX_PTQ1_HMX_BLOCK_H
#define GENIEX_PTQ1_HMX_BLOCK_H
#include "hmx-utils.h"

enum { PTQ1_HMX_M = 32, PTQ1_HMX_N = 32, PTQ1_HMX_K = 16, PTQ1_HMX_BLOCK_K = 128,
       PTQ1_HMX_PARTS = 8, PTQ1_HMX_TILE = 2048,
       PTQ1_HMX_ACT_OFF = 0, PTQ1_HMX_WT_OFF = 16384, PTQ1_HMX_OUT_OFF = 32768,
       PTQ1_HMX_SCALE_OFF = 49152, PTQ1_HMX_VTCM_BYTES = 65536 };
_Static_assert(PTQ1_HMX_PARTS * PTQ1_HMX_TILE == PTQ1_HMX_WT_OFF - PTQ1_HMX_ACT_OFF &&
               PTQ1_HMX_WT_OFF + PTQ1_HMX_PARTS * PTQ1_HMX_TILE == PTQ1_HMX_OUT_OFF &&
               PTQ1_HMX_OUT_OFF + PTQ1_HMX_PARTS * PTQ1_HMX_TILE == PTQ1_HMX_SCALE_OFF &&
               PTQ1_HMX_SCALE_OFF + 256 <= PTQ1_HMX_VTCM_BYTES,
               "eight-part HMX VTCM slots overlap");

struct ptq1_hmx_scratch {
    uint8_t vtcm[PTQ1_HMX_VTCM_BYTES] __attribute__((aligned(2048)));
    __fp16 act_rows[PTQ1_HMX_M][64] __attribute__((aligned(128)));
    int8_t decoded[PTQ1_HMX_BLOCK_K][PTQ1_HMX_N] __attribute__((aligned(128)));
    int32_t partial[8][PTQ1_HMX_M][PTQ1_HMX_N] __attribute__((aligned(128)));
    float tile[PTQ1_HMX_M][PTQ1_HMX_N] __attribute__((aligned(128)));
};
_Static_assert(sizeof(struct ptq1_hmx_scratch) == 110592, "per-worker auxiliary VTCM size changed");


static int ptq1_hmx_eligible(uint32_t k, uint32_t m, uint32_t n, unsigned nth) {
    return (k == 5120 || k == 6144 || k == 17408) && m >= 4 && m <= PTQ1_HMX_M &&
           n > 0 && nth >= 1 && nth <= 4;
}

static uint8_t *ptq1_hmx_aux_ptr(const struct htp_ops_context *octx, size_t layout_bytes) {
    const size_t aux_bytes = sizeof(struct ptq1_hmx_scratch);
    if (!octx->ctx->hmx_queue || !octx->ctx->vtcm_base ||
        octx->n_threads < 1 || octx->n_threads > 4 || layout_bytes > SIZE_MAX - 2047)
        return NULL;
    const size_t aux_off = (layout_bytes + 2047) & ~(size_t)2047;
    const uintptr_t base = (uintptr_t)octx->ctx->vtcm_base;
    if (aux_off > UINTPTR_MAX - base || ((base + aux_off) & 2047u) ||
        octx->n_threads > (SIZE_MAX - aux_off) / aux_bytes ||
        aux_off + (size_t)octx->n_threads * aux_bytes > octx->ctx->vtcm_size)
        return NULL;
    return (uint8_t *)octx->ctx->vtcm_base + aux_off;
}

struct ptq1_hmx_dot_job {
    __fp16 *out[PTQ1_HMX_PARTS], *act[PTQ1_HMX_PARTS], *wt[PTQ1_HMX_PARTS], *scales;
};
static void ptq1_hmx_dot_job_run(void *arg) {
    struct ptq1_hmx_dot_job *j = arg;
    for (unsigned slot = 0; slot < PTQ1_HMX_PARTS; ++slot)
        core_dot_chunk_fp16_short(j->out[slot], j->act[slot], j->wt[slot],
                                  j->scales, 1, 1, 1);
}

static void ptq1_hmx_decode_tile(struct ptq1_hmx_scratch *scratch, const geniex_ptq1_tile *tile) {
    for (unsigned group = 0; group < 6; ++group) {
        const unsigned base = group * 4;
        const HVX_Vector codes = hvx_vmemu(&tile->qs[base][0]);
        const HVX_VectorPair unpacked = Q6_Wuh_vunpack_Vub(codes);
        HVX_Vector low = Q6_V_lo_W(unpacked), high = Q6_V_hi_W(unpacked);
        const unsigned logical_base = group < 4 ? base : 80 + base - 16;
        for (unsigned p = 0; p < 5; ++p) {
            const HVX_Vector lo_trits = geniex_ptq1_decode_next(&low);
            const HVX_Vector hi_trits = geniex_ptq1_decode_next(&high);
            const HVX_Vector bytes = Q6_Vb_vpacke_VhVh(hi_trits, lo_trits);
            *(HVX_Vector *)&scratch->decoded[logical_base + p * (group < 4 ? 16 : 8)][0] = bytes;
        }
    }
    uint8_t qh_copy[128] __attribute__((aligned(128))) = {0};
    memcpy(qh_copy, tile->qh, 64);
    const HVX_VectorPair unpacked = Q6_Wuh_vunpack_Vub(*(const HVX_Vector *)qh_copy);
    HVX_Vector state = Q6_V_lo_W(unpacked);
    for (unsigned p = 0; p < 4; ++p) {
        const HVX_Vector trits = geniex_ptq1_decode_next(&state);
        hvx_vec_store_u(&scratch->decoded[120 + 2 * p][0], 64,
                        Q6_Vb_vpacke_VhVh(trits, trits));
    }
}

static void ptq1_hmx_pack_activation(struct ptq1_hmx_scratch *scratch,
                                      unsigned part, unsigned slot,
                                      const uint8_t *flat, size_t flat_stride,
                                      unsigned rows) {
    const HVX_VectorPred active = Q6_Q_vsetq_R(2 * PTQ1_HMX_K);
    const HVX_Vector zero = Q6_V_vzero();
    for (unsigned r = 0; r < PTQ1_HMX_M; r += 2) {
        HVX_Vector even = zero, odd = zero;
        if (r < rows) {
            const HVX_Vector bytes = hvx_vmemu(flat + r * flat_stride + part * PTQ1_HMX_K);
            const HVX_VectorPair integers = Q6_Wh_vunpack_Vb(bytes);
            const HVX_Vector halves = Q6_Vhf_equals_Vh(Q6_V_lo_W(integers));
            even = Q6_V_vmux_QVV(active, halves, zero);
        }
        if (r + 1 < rows) {
            const HVX_Vector bytes = hvx_vmemu(flat + (r + 1) * flat_stride + part * PTQ1_HMX_K);
            const HVX_VectorPair integers = Q6_Wh_vunpack_Vb(bytes);
            const HVX_Vector halves = Q6_Vhf_equals_Vh(Q6_V_lo_W(integers));
            odd = Q6_V_vmux_QVV(active, halves, zero);
        }
        const HVX_VectorPair pair = Q6_W_vshuff_VVR(odd, even, -2);
        ((HVX_Vector *)(scratch->vtcm + PTQ1_HMX_ACT_OFF + slot * PTQ1_HMX_TILE))[r / 2] = Q6_V_lo_W(pair);
    }
}

static void ptq1_hmx_pack_weight(struct ptq1_hmx_scratch *scratch, unsigned part, unsigned slot) {
    __fp16 *wt = (__fp16 *)(scratch->vtcm + PTQ1_HMX_WT_OFF + slot * PTQ1_HMX_TILE);
    const unsigned offset = (part % 8) * PTQ1_HMX_K;
    for (unsigned q = 0; q < 4; ++q) {
        const HVX_Vector trits = *(const HVX_Vector *)&scratch->decoded[offset + 4 * q][0];
        const HVX_VectorPair rows = Q6_Wh_vunpack_Vb(trits);
        const HVX_Vector first = Q6_Vhf_equals_Vh(Q6_V_lo_W(rows));
        const HVX_Vector second = Q6_Vhf_equals_Vh(Q6_V_hi_W(rows));
        ((HVX_Vector *)wt)[2 * q] = Q6_Vh_vshuff_Vh(first);
        ((HVX_Vector *)wt)[2 * q + 1] = Q6_Vh_vshuff_Vh(second);
    }
    const HVX_Vector zero = Q6_V_vzero();
    for (unsigned q = 8; q < 16; ++q) ((HVX_Vector *)wt)[q] = zero;
}

static int ptq1_hmx_run_block(struct htp_mm_context *mmctx, struct ptq1_hmx_scratch *scratch,
                              unsigned first_part, const uint8_t *flat, size_t flat_stride,
                              unsigned rows) {
    struct ptq1_hmx_dot_job j = {
        .scales = (__fp16 *)(scratch->vtcm + PTQ1_HMX_SCALE_OFF),
    };
    for (unsigned slot = 0; slot < PTQ1_HMX_PARTS; ++slot) {
        ptq1_hmx_pack_activation(scratch, first_part + slot, slot, flat, flat_stride, rows);
        ptq1_hmx_pack_weight(scratch, first_part + slot, slot);
        j.out[slot] = (__fp16 *)(scratch->vtcm + PTQ1_HMX_OUT_OFF + slot * PTQ1_HMX_TILE);
        j.act[slot] = (__fp16 *)(scratch->vtcm + PTQ1_HMX_ACT_OFF + slot * PTQ1_HMX_TILE);
        j.wt[slot] = (__fp16 *)(scratch->vtcm + PTQ1_HMX_WT_OFF + slot * PTQ1_HMX_TILE);
    }
    qurt_mutex_lock(&mmctx->ptq1_hmx_mutex);
    const bool pushed = hmx_queue_push(mmctx->octx->ctx->hmx_queue,
                                      hmx_queue_make_desc(ptq1_hmx_dot_job_run, &j));
    bool complete = false;
    bool unexpected = false;
    if (pushed) {
        // The descriptor holds &j: keep its stack frame live until this exact
        // job completes, even if an unexpected descriptor was ahead of it.
        for (;;) {
            const struct hmx_queue_desc popped = hmx_queue_pop(mmctx->octx->ctx->hmx_queue);
            if (popped.func == ptq1_hmx_dot_job_run && popped.data == &j) {
                complete = true;
                break;
            }
            unexpected = true;
        }
    }
    qurt_mutex_unlock(&mmctx->ptq1_hmx_mutex);
    if (!complete || unexpected) return 0;
    for (unsigned slot = 0; slot < PTQ1_HMX_PARTS; ++slot) {
        for (unsigned r = 0; r < PTQ1_HMX_M; r += 2) {
            const HVX_Vector row_pair =
                ((const HVX_Vector *)(scratch->vtcm + PTQ1_HMX_OUT_OFF + slot * PTQ1_HMX_TILE))[r / 2];
            const HVX_Vector separate = Q6_Vh_vdeal_Vh(row_pair);
            const HVX_Vector integers = hvx_vec_i16_from_hf_rnd_sat(separate);
            const HVX_VectorPair widened = Q6_Ww_vunpack_Vh(integers);
            *(HVX_Vector *)&scratch->partial[(first_part + slot) % 8][r][0] = Q6_V_lo_W(widened);
            *(HVX_Vector *)&scratch->partial[(first_part + slot) % 8][r + 1][0] = Q6_V_hi_W(widened);
        }
    }
    return 1;
}

static void ptq1_hmx_accumulate_groups(const geniex_ptq1_tile *tile, const float scales[4],
                                        const HVX_Vector groups[4], float *outputs,
                                        unsigned valid_columns) {
    HVX_Vector sum = Q6_V_vzero();
    for (unsigned b = 0; b < 4; ++b) {
        const HVX_Vector integers = Q6_Vsf_equals_Vw(groups[b]);
        uint32_t scale_bits;
        memcpy(&scale_bits, &scales[b], sizeof(scale_bits));
        const HVX_Vector product = Q6_Vqf32_vmpy_VsfVsf(Q6_V_vsplat_R(scale_bits), integers);
        sum = b == 0 ? Q6_Vqf32_vadd_Vqf32Vsf(product, sum)
                     : Q6_Vqf32_vadd_Vqf32Vqf32(product, sum);
    }
    const HVX_Vector sum_sf = Q6_Vsf_equals_Vqf32(sum);
    uint16_t weight_halves[64] __attribute__((aligned(128))) = {0};
    memcpy(weight_halves, tile->d, 32 * sizeof(uint16_t));
    const HVX_VectorPair converted = Q6_Wqf32_vmpy_VhfVhf(
        *(const HVX_Vector *)weight_halves, Q6_Vh_vsplat_R(0x3c00));
    const HVX_Vector lo_sf = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(converted));
    const HVX_Vector hi_sf = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(converted));
    const HVX_Vector weight_sf = Q6_V_lo_W(Q6_W_vshuff_VVR(hi_sf, lo_sf, -4));
    const HVX_Vector weighted = Q6_Vqf32_vmpy_VsfVsf(sum_sf, weight_sf);
    // The accepted dot helper rounds its weighted block to SF before the outer output add.
    HVX_Vector weighted_sf = Q6_Vsf_equals_Vqf32(weighted);
    asm volatile("" : "+v"(weighted_sf));
    if (valid_columns < PTQ1_HMX_N) {
        float partial[PTQ1_HMX_N] __attribute__((aligned(128)));
        *(HVX_Vector *)partial = weighted_sf;
        for (unsigned row = 0; row < valid_columns; ++row) outputs[row] += partial[row];
    } else {
        const HVX_Vector accumulated = Q6_Vqf32_vadd_VsfVsf(weighted_sf, *(const HVX_Vector *)outputs);
        *(HVX_Vector *)outputs = Q6_Vsf_equals_Vqf32(accumulated);
    }
}

static int ptq1_hmx_route_tile(struct htp_mm_context *mmctx, struct ptq1_hmx_scratch *scratch,
                                const geniex_ptq1_tile *tile, const uint8_t *flat,
                                size_t flat_stride, uint32_t k, unsigned rows,
                                unsigned valid_columns) {
    for (unsigned r = 0; r < rows; ++r) memset(scratch->tile[r], 0, sizeof(scratch->tile[r]));
    for (unsigned block = 0; block < k / PTQ1_HMX_BLOCK_K; ++block) {
        ptq1_hmx_decode_tile(scratch, &tile[block]);
        if (!ptq1_hmx_run_block(mmctx, scratch, block * PTQ1_HMX_PARTS, flat,
                                 flat_stride, rows)) return 0;
        for (unsigned r = 0; r < rows; ++r) {
            float scales[4];
            geniex_ptq1_flat_scales(scales, flat + r * flat_stride + k, block);
            HVX_Vector groups[4];
            for (unsigned group = 0; group < 4; ++group) {
                const HVX_Vector left = *(const HVX_Vector *)&scratch->partial[2 * group][r][0];
                const HVX_Vector right = *(const HVX_Vector *)&scratch->partial[2 * group + 1][r][0];
                groups[group] = Q6_Vw_vadd_VwVw(left, right);
                const HVX_Vector magnitude = Q6_Vw_vabs_Vw_sat(groups[group]);
                const HVX_VectorPred too_large = Q6_Q_vcmp_gt_VwVw(magnitude, Q6_V_vsplat_R(4096));
                const HVX_Vector flags = Q6_V_vmux_QVV(too_large, Q6_V_vsplat_R(1), Q6_V_vzero());
                if (hvx_vec_get_i32(hvx_vec_reduce_max_i32(flags))) return 0;
            }
            ptq1_hmx_accumulate_groups(&tile[block], scales, groups, scratch->tile[r], valid_columns);
        }
    }
    return 1;
}

// Same weight DMA and quant barrier as the accepted batch worker; only the dot is replaced.
static void hvx_mm_2d_repacked_ptq1_batch_hmx(unsigned nth, unsigned ith, void *data) {
    (void) nth;
    htp_matmul_preamble;
    struct ptq1_hmx_scratch *scratch = (struct ptq1_hmx_scratch *)
        (mmctx->ptq1_hmx_aux + ith * mmctx->ptq1_hmx_aux_stride);
    const uint32_t first = src0_nrows_per_thread * ith;
    const uint32_t last = MIN(first + src0_nrows_per_thread, ne01);
    const uint32_t ct_end = (last + 31) / 32;
    const uint32_t blocks = ne00 / GENIEX_PTQ1_BLOCK_K;
    const uint32_t tile_bytes = blocks * sizeof(geniex_ptq1_tile);
    uint8_t *weight_buffer = mmctx->vtcm_src0 + ith * mmctx->vtcm_src0_size_per_thread;
    uint32_t push_ct = first / 32;
    if (first < last) {
        const uint32_t count = MIN(last, ne0) > first ? MIN(last, ne0) - first : 0;
        if (src2 && count) {
            dma_queue_push(dma_queue, dma_make_ptr((float *)mmctx->vtcm_src2 + first,
                (const float *)src2->data + first), count * sizeof(float),
                count * sizeof(float), count * sizeof(float), 1);
            dma_queue_pop_nowait(dma_queue);
        }
        for (uint32_t d = 0; d < 2 && push_ct < ct_end; ++d, ++push_ct)
            dma_queue_push(dma_queue, dma_make_ptr(weight_buffer + d * tile_bytes,
                (const uint8_t *)src0->data + push_ct * tile_bytes),
                sizeof(geniex_ptq1_tile), sizeof(geniex_ptq1_tile), sizeof(geniex_ptq1_tile), blocks);
    }
    hvx_mm_run_quant_task(mmctx, ith);
    if (first >= last) return;
    hmx_init_column_scales(scratch->vtcm + PTQ1_HMX_SCALE_OFF, Q6_V_vsplat_R(0x3c00));
    for (uint32_t ct = first / 32; ct < ct_end; ++ct) {
        const uint8_t *w_tile = dma_queue_pop(dma_queue).dst;
        const uint32_t col = ct * 32;
        const uint32_t count = MIN(32, ne0 > col ? ne0 - col : 0);
        if (!w_tile || !ptq1_hmx_route_tile(mmctx, scratch, (const geniex_ptq1_tile *)w_tile,
                         mmctx->vtcm_src1, mmctx->vtcm_src1_stride, ne00, ne11, count)) {
            atomic_store(&mmctx->ptq1_hmx_error, 2);
            // Drain prefetch before reporting a failed operation; never try partial-output fallback.
            while (push_ct > ct + 1) { dma_queue_pop(dma_queue); ++ct; }
            return;
        }
        for (unsigned r = 0; r < ne11; ++r) {
            uint8_t *out = (uint8_t *)dst->data + r * nb1 + col * sizeof(float);
            if (src2) hvx_add_f32_uaa(out, (const uint8_t *)scratch->tile[r],
                                       mmctx->vtcm_src2 + col * sizeof(float), count);
            else hvx_copy_f32_ua(out, (const uint8_t *)scratch->tile[r], count);
        }
        if (push_ct < ct_end) {
            dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile,
                           (const uint8_t *)src0->data + push_ct * tile_bytes),
                           sizeof(geniex_ptq1_tile), sizeof(geniex_ptq1_tile), sizeof(geniex_ptq1_tile), blocks);
            ++push_ct;
        }
    }
}
#endif /* GENIEX_PTQ1_HMX_BLOCK_H */
