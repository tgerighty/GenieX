// SPDX-License-Identifier: BSD-3-Clause
#ifndef GENIEX_PTQ1_HVX_H
#define GENIEX_PTQ1_HVX_H

#include <string.h>

#include "ptq1_tile.h"

#ifdef __hexagon__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

static inline float geniex_ptq1_half_to_float(uint16_t bits) {
    const uint32_t exponent = bits & 0x7c00u;
    if (exponent != 0 && exponent != 0x7c00u) {
        const uint32_t word = ((uint32_t) (bits & 0x8000u) << 16) |
                              (((uint32_t) (bits & 0x7fffu) + 0x1c000u) << 13);
        float value;
        memcpy(&value, &word, sizeof(value));
        return value;
    }
    union {
        uint16_t bits;
        _Float16 value;
    } half = {.bits = bits};
    return (float)half.value;
}

typedef struct {
    HVX_Vector lo;
    HVX_Vector hi;
} geniex_ptq1_act_pair;

typedef struct {
    geniex_ptq1_act_pair qs[3][5][2];
    geniex_ptq1_act_pair qh[4];
} geniex_ptq1_activation;

static inline void geniex_ptq1_prepare_act_pair(
    geniex_ptq1_act_pair *pair, const int8_t *activation, unsigned offset, unsigned count) {
#ifdef PTQ1_SCALAR_PREP
    int16_t lanes[128] __attribute__((aligned(128)));
    for (unsigned m = 0; m < 4; ++m) {
        const int16_t value = m < count ? activation[offset + m] : 0;
        for (unsigned row = 0; row < GENIEX_PTQ1_TILE_ROWS; ++row) lanes[m * GENIEX_PTQ1_TILE_ROWS + row] = value;
    }
    pair->lo = *(const HVX_Vector *)&lanes[0];
    if (count > 2) pair->hi = *(const HVX_Vector *)&lanes[64];
#else
    const HVX_VectorPred first = Q6_Q_vsetq_R(64);
    const HVX_Vector     a0    = Q6_Vh_vsplat_R(activation[offset]);
    const HVX_Vector     a1    = Q6_Vh_vsplat_R(activation[offset + 1]);
    const HVX_Vector     a2    = Q6_Vh_vsplat_R(count > 2 ? activation[offset + 2] : 0);
    const HVX_Vector     a3    = Q6_Vh_vsplat_R(count > 2 ? activation[offset + 3] : 0);
    pair->lo                   = Q6_V_vmux_QVV(first, a0, a1);
    if (count > 2) pair->hi    = Q6_V_vmux_QVV(first, a2, a3);
#endif
}

static inline void geniex_ptq1_prepare_activation(geniex_ptq1_activation *prepared, const int8_t *activation) {
    for (unsigned group = 0; group < 3; ++group) {
        for (unsigned n = 0; n < 5; ++n) {
            for (unsigned chunk = 0; chunk < 2; ++chunk) {
                geniex_ptq1_prepare_act_pair(
                    &prepared->qs[group][n][chunk], activation, group * 40 + n * 8 + chunk * 4, 4);
            }
        }
    }
    for (unsigned n = 0; n < 4; ++n) geniex_ptq1_prepare_act_pair(&prepared->qh[n], activation, 120 + n * 2, 2);
}

static inline HVX_Vector geniex_ptq1_decode_trit(HVX_Vector bytes, unsigned power) {
    const HVX_Vector mask = Q6_Vh_vsplat_R(255);
    const HVX_Vector one  = Q6_Vh_vsplat_R(1);
#ifdef PTQ1_SHIFT_DECODE
    HVX_Vector x = bytes;
    for (unsigned p = power; p > 1; p /= 3) {
        x = Q6_Vh_vadd_VhVh(x, Q6_Vh_vasl_VhR(x, 1));
    }
#else
    HVX_Vector x = power == 1 ? bytes : Q6_Vh_vmpyi_VhVh(bytes, Q6_Vh_vsplat_R(power));
#endif
    x = Q6_V_vand_VV(x, mask);
    x = Q6_Vh_vadd_VhVh(x, Q6_Vh_vasl_VhR(x, 1));
    x = Q6_Vh_vasr_VhR(x, 8);
    return Q6_Vh_vsub_VhVh(x, one);
}

static inline void geniex_ptq1_accumulate(
    HVX_Vector *scaled_lo, HVX_Vector *scaled_hi, const geniex_ptq1_act_pair *activation, HVX_Vector sums[2]) {
    const HVX_Vector one       = Q6_Vh_vsplat_R(1);
    const HVX_Vector mask      = Q6_Vh_vsplat_R(255);
    const HVX_Vector triple_lo = Q6_Vh_vadd_VhVh(*scaled_lo, Q6_Vh_vasl_VhR(*scaled_lo, 1));
    const HVX_Vector triple_hi = Q6_Vh_vadd_VhVh(*scaled_hi, Q6_Vh_vasl_VhR(*scaled_hi, 1));
    const HVX_Vector trit_lo   = Q6_Vh_vsub_VhVh(Q6_Vh_vasr_VhR(triple_lo, 8), one);
    const HVX_Vector trit_hi   = Q6_Vh_vsub_VhVh(Q6_Vh_vasr_VhR(triple_hi, 8), one);
    sums[0]                    = Q6_Vh_vmpyiacc_VhVhVh(sums[0], trit_lo, activation->lo);
    sums[1]                    = Q6_Vh_vmpyiacc_VhVhVh(sums[1], trit_hi, activation->hi);
    *scaled_lo                 = Q6_V_vand_VV(triple_lo, mask);
    *scaled_hi                 = Q6_V_vand_VV(triple_hi, mask);
}

static inline void geniex_ptq1_accumulate_two(
    HVX_Vector *scaled_lo, HVX_Vector *scaled_hi, const geniex_ptq1_act_pair *act0,
    const geniex_ptq1_act_pair *act1, HVX_Vector sums0[2], HVX_Vector sums1[2]) {
    const HVX_Vector one       = Q6_Vh_vsplat_R(1);
    const HVX_Vector mask      = Q6_Vh_vsplat_R(255);
    const HVX_Vector triple_lo = Q6_Vh_vadd_VhVh(*scaled_lo, Q6_Vh_vasl_VhR(*scaled_lo, 1));
    const HVX_Vector triple_hi = Q6_Vh_vadd_VhVh(*scaled_hi, Q6_Vh_vasl_VhR(*scaled_hi, 1));
    const HVX_Vector trit_lo   = Q6_Vh_vsub_VhVh(Q6_Vh_vasr_VhR(triple_lo, 8), one);
    const HVX_Vector trit_hi   = Q6_Vh_vsub_VhVh(Q6_Vh_vasr_VhR(triple_hi, 8), one);
    sums0[0] = Q6_Vh_vmpyiacc_VhVhVh(sums0[0], trit_lo, act0->lo);
    sums0[1] = Q6_Vh_vmpyiacc_VhVhVh(sums0[1], trit_hi, act0->hi);
    sums1[0] = Q6_Vh_vmpyiacc_VhVhVh(sums1[0], trit_lo, act1->lo);
    sums1[1] = Q6_Vh_vmpyiacc_VhVhVh(sums1[1], trit_hi, act1->hi);
    *scaled_lo = Q6_V_vand_VV(triple_lo, mask);
    *scaled_hi = Q6_V_vand_VV(triple_hi, mask);
}

static inline void geniex_ptq1_dot_tile(
    const geniex_ptq1_tile *tile, const geniex_ptq1_activation *activation, const float scales[4], float *outputs) {
    static const unsigned powers[5] = {1, 3, 9, 27, 81};
    HVX_Vector            acc[4][2];
#ifdef PTQ1_SCALAR_REDUCE
    int16_t lanes[4][2][2 * GENIEX_PTQ1_TILE_ROWS] __attribute__((aligned(128)));
#else
    int16_t partial[4][2 * GENIEX_PTQ1_TILE_ROWS] __attribute__((aligned(128)));
#endif

    for (unsigned b = 0; b < 4; ++b) {
        for (unsigned m = 0; m < 2; ++m) acc[b][m] = Q6_V_vzero();
    }

#pragma clang loop unroll(full)
    for (unsigned group = 0; group < 3; ++group) {
#pragma clang loop unroll(full)
        for (unsigned m = 0; m < 8; m += 4) {
            const HVX_VectorPair products = Q6_Wuh_vunpack_Vub(*(const HVX_UVector *)tile->qs[group * 8 + m]);
            HVX_Vector scaled_lo = Q6_V_lo_W(products);
            HVX_Vector scaled_hi = Q6_V_hi_W(products);
#pragma clang loop unroll(full)
            for (unsigned n = 0; n < 5; ++n) {
                const unsigned scale_index = (group * 40 + n * 8 + m) / 32;
                geniex_ptq1_accumulate(
                    &scaled_lo, &scaled_hi, &activation->qs[group][n][m / 4], acc[scale_index]);
            }
        }
    }
    const HVX_VectorPair qh_products = Q6_Wuh_vunpack_Vub(*(const HVX_UVector *)tile->qh[0]);
#pragma clang loop unroll(full)
    for (unsigned n = 0; n < 4; ++n) {
        const HVX_Vector trit = geniex_ptq1_decode_trit(Q6_V_lo_W(qh_products), powers[n]);
        acc[3][0] = Q6_Vh_vmpyiacc_VhVhVh(acc[3][0], trit, activation->qh[n].lo);
    }
#ifdef PTQ1_SCALAR_REDUCE
    for (unsigned b = 0; b < 4; ++b) {
        for (unsigned m = 0; m < 2; ++m) *(HVX_Vector *)lanes[b][m] = acc[b][m];
    }
#else
    for (unsigned b = 0; b < 4; ++b) {
        const HVX_Vector combined = Q6_Vh_vadd_VhVh(acc[b][0], acc[b][1]);
        *(HVX_Vector *)partial[b] = Q6_Vh_vadd_VhVh(combined, Q6_V_vror_VR(combined, 64));
    }
#endif
#pragma clang loop unroll_count(4)
    for (unsigned row = 0; row < GENIEX_PTQ1_TILE_ROWS; ++row) {
#ifdef PTQ1_SCALAR_REDUCE
        const unsigned lane = row;
#endif
        float sum = 0.0f;
        for (unsigned b = 0; b < 4; ++b) {
#ifdef PTQ1_SCALAR_REDUCE
            const int32_t value = lanes[b][0][lane] + lanes[b][0][32 + lane] +
                                  lanes[b][1][lane] + lanes[b][1][32 + lane];
            sum += scales[b] * value;
#else
            sum += scales[b] * partial[b][row];
#endif
        }
        outputs[row] = geniex_ptq1_half_to_float(tile->d[row]) * sum;
    }
}

static inline void geniex_ptq1_dot_tile_two(
    const geniex_ptq1_tile *tile, const geniex_ptq1_activation *act0,
    const geniex_ptq1_activation *act1, const float scales0[4], const float scales1[4],
    float *outputs0, float *outputs1, unsigned valid_rows) {
    static const unsigned powers[4] = {1, 3, 9, 27};
    HVX_Vector acc0[4][2], acc1[4][2];
    int16_t partial0[4][2 * GENIEX_PTQ1_TILE_ROWS] __attribute__((aligned(128)));
    int16_t partial1[4][2 * GENIEX_PTQ1_TILE_ROWS] __attribute__((aligned(128)));

    for (unsigned b = 0; b < 4; ++b) {
        for (unsigned m = 0; m < 2; ++m) acc0[b][m] = acc1[b][m] = Q6_V_vzero();
    }
#pragma clang loop unroll(full)
    for (unsigned group = 0; group < 3; ++group) {
#pragma clang loop unroll(full)
        for (unsigned m = 0; m < 8; m += 4) {
            const HVX_VectorPair products = Q6_Wuh_vunpack_Vub(*(const HVX_UVector *)tile->qs[group * 8 + m]);
            HVX_Vector scaled_lo = Q6_V_lo_W(products);
            HVX_Vector scaled_hi = Q6_V_hi_W(products);
#pragma clang loop unroll(full)
            for (unsigned n = 0; n < 5; ++n) {
                const unsigned scale_index = (group * 40 + n * 8 + m) / 32;
                geniex_ptq1_accumulate_two(&scaled_lo, &scaled_hi, &act0->qs[group][n][m / 4],
                    &act1->qs[group][n][m / 4], acc0[scale_index], acc1[scale_index]);
            }
        }
    }
    const HVX_VectorPair qh_products = Q6_Wuh_vunpack_Vub(*(const HVX_UVector *)tile->qh[0]);
#pragma clang loop unroll(full)
    for (unsigned n = 0; n < 4; ++n) {
        const HVX_Vector trit = geniex_ptq1_decode_trit(Q6_V_lo_W(qh_products), powers[n]);
        acc0[3][0] = Q6_Vh_vmpyiacc_VhVhVh(acc0[3][0], trit, act0->qh[n].lo);
        acc1[3][0] = Q6_Vh_vmpyiacc_VhVhVh(acc1[3][0], trit, act1->qh[n].lo);
    }
    for (unsigned b = 0; b < 4; ++b) {
        const HVX_Vector combined0 = Q6_Vh_vadd_VhVh(acc0[b][0], acc0[b][1]);
        const HVX_Vector combined1 = Q6_Vh_vadd_VhVh(acc1[b][0], acc1[b][1]);
        *(HVX_Vector *)partial0[b] = Q6_Vh_vadd_VhVh(combined0, Q6_V_vror_VR(combined0, 64));
        *(HVX_Vector *)partial1[b] = Q6_Vh_vadd_VhVh(combined1, Q6_V_vror_VR(combined1, 64));
    }
#pragma clang loop unroll_count(2)
    for (unsigned row = 0; row < valid_rows; ++row) {
        float sum0 = 0.0f, sum1 = 0.0f;
        for (unsigned b = 0; b < 4; ++b) {
            sum0 += scales0[b] * partial0[b][row];
            sum1 += scales1[b] * partial1[b][row];
        }
        const float weight_scale = geniex_ptq1_half_to_float(tile->d[row]);
        outputs0[row] += weight_scale * sum0;
        outputs1[row] += weight_scale * sum1;
    }
}

// Weights contain one 32-row tile per 128-wide K block. The flat Q8 row
// contains K int8 values, then four FP16 scales per K block.
static inline void geniex_ptq1_flat_scales(float scales[4], const uint8_t *scale_bytes, uint32_t block) {
    for (unsigned b = 0; b < 4; ++b) {
        uint16_t bits;
        memcpy(&bits, scale_bytes + 2 * (block * 4 + b), sizeof(bits));
        scales[b] = geniex_ptq1_half_to_float(bits);
    }
}

static inline void geniex_ptq1_prepare_flat_scales(float *scales, const void *flat_q8, uint32_t k) {
    const uint8_t *scale_bytes = (const uint8_t *)flat_q8 + k;
    for (uint32_t block = 0; block < k / GENIEX_PTQ1_BLOCK_K; ++block)
        geniex_ptq1_flat_scales(scales + 4 * block, scale_bytes, block);
}

static inline void geniex_ptq1_dot_flat_q8(uint32_t k, float *outputs, const geniex_ptq1_tile *weights,
    const void *flat_q8, unsigned valid_rows, geniex_ptq1_activation *scratch) {
    const int8_t  *quants      = (const int8_t *)flat_q8;
    const uint8_t *scale_bytes = (const uint8_t *)flat_q8 + k;
    float          partial[GENIEX_PTQ1_TILE_ROWS];

    assert(k % GENIEX_PTQ1_BLOCK_K == 0);
    assert(valid_rows <= GENIEX_PTQ1_TILE_ROWS);
    for (unsigned row = 0; row < valid_rows; ++row) outputs[row] = 0.0f;
    for (uint32_t block = 0; block < k / GENIEX_PTQ1_BLOCK_K; ++block) {
        float scales[4];
        geniex_ptq1_flat_scales(scales, scale_bytes, block);
        geniex_ptq1_prepare_activation(scratch, quants + block * GENIEX_PTQ1_BLOCK_K);
        geniex_ptq1_dot_tile(&weights[block], scratch, scales, partial);
        for (unsigned row = 0; row < valid_rows; ++row) outputs[row] += partial[row];
    }
}

static inline void geniex_ptq1_dot_pair_flat_q8_scaled(uint32_t k, float *outputs0, float *outputs1,
    const geniex_ptq1_tile *weights0, const geniex_ptq1_tile *weights1, const void *flat_q8, unsigned valid_rows0,
    unsigned valid_rows1, geniex_ptq1_activation *scratch, const float *prepared_scales) {
    const int8_t  *quants      = (const int8_t *)flat_q8;
    const uint8_t *scale_bytes = (const uint8_t *)flat_q8 + k;
    float          partial[GENIEX_PTQ1_TILE_ROWS];

    assert(k % GENIEX_PTQ1_BLOCK_K == 0);
    assert(valid_rows0 <= GENIEX_PTQ1_TILE_ROWS && valid_rows1 <= GENIEX_PTQ1_TILE_ROWS);
    for (unsigned row = 0; row < valid_rows0; ++row) outputs0[row] = 0.0f;
    for (unsigned row = 0; row < valid_rows1; ++row) outputs1[row] = 0.0f;
    for (uint32_t block = 0; block < k / GENIEX_PTQ1_BLOCK_K; ++block) {
        float scales[4];
        if (!prepared_scales) geniex_ptq1_flat_scales(scales, scale_bytes, block);
        geniex_ptq1_prepare_activation(scratch, quants + block * GENIEX_PTQ1_BLOCK_K);
        geniex_ptq1_dot_tile(&weights0[block], scratch, prepared_scales ? prepared_scales + 4 * block : scales, partial);
        for (unsigned row = 0; row < valid_rows0; ++row) outputs0[row] += partial[row];
        geniex_ptq1_dot_tile(&weights1[block], scratch, prepared_scales ? prepared_scales + 4 * block : scales, partial);
        for (unsigned row = 0; row < valid_rows1; ++row) outputs1[row] += partial[row];
    }
}

static inline void geniex_ptq1_dot_pair_flat_q8(uint32_t k, float *outputs0, float *outputs1,
    const geniex_ptq1_tile *weights0, const geniex_ptq1_tile *weights1, const void *flat_q8, unsigned valid_rows0,
    unsigned valid_rows1, geniex_ptq1_activation *scratch) {
    geniex_ptq1_dot_pair_flat_q8_scaled(k, outputs0, outputs1, weights0, weights1, flat_q8,
        valid_rows0, valid_rows1, scratch, NULL);
}

static inline void geniex_ptq1_dot_two_rows_flat_q8(uint32_t k, float *outputs0, float *outputs1,
    const geniex_ptq1_tile *weights, const void *flat_q8_0, const void *flat_q8_1, unsigned valid_rows,
    geniex_ptq1_activation *scratch0, geniex_ptq1_activation *scratch1) {
    const int8_t *quants0 = (const int8_t *)flat_q8_0;
    const int8_t *quants1 = (const int8_t *)flat_q8_1;
    const uint8_t *scale_bytes0 = (const uint8_t *)flat_q8_0 + k;
    const uint8_t *scale_bytes1 = (const uint8_t *)flat_q8_1 + k;
    assert(k % GENIEX_PTQ1_BLOCK_K == 0);
    assert(valid_rows <= GENIEX_PTQ1_TILE_ROWS);
    for (unsigned row = 0; row < valid_rows; ++row) outputs0[row] = outputs1[row] = 0.0f;
    for (uint32_t block = 0; block < k / GENIEX_PTQ1_BLOCK_K; ++block) {
        float scales0[4], scales1[4];
        geniex_ptq1_flat_scales(scales0, scale_bytes0, block);
        geniex_ptq1_flat_scales(scales1, scale_bytes1, block);
        geniex_ptq1_prepare_activation(scratch0, quants0 + block * GENIEX_PTQ1_BLOCK_K);
        geniex_ptq1_prepare_activation(scratch1, quants1 + block * GENIEX_PTQ1_BLOCK_K);
        geniex_ptq1_dot_tile_two(&weights[block], scratch0, scratch1, scales0, scales1,
            outputs0, outputs1, valid_rows);
    }
}

static inline void geniex_ptq1_dot_pair_two_rows_flat_q8_scaled(uint32_t k,
    float *outputs00, float *outputs01, float *outputs10, float *outputs11,
    const geniex_ptq1_tile *weights0, const geniex_ptq1_tile *weights1,
    const void *flat_q8_0, const void *flat_q8_1, unsigned valid_rows0, unsigned valid_rows1,
    geniex_ptq1_activation *scratch0, geniex_ptq1_activation *scratch1,
    const float *prepared_scales0, const float *prepared_scales1) {
    const int8_t *quants0 = (const int8_t *)flat_q8_0;
    const int8_t *quants1 = (const int8_t *)flat_q8_1;
    const uint8_t *scale_bytes0 = (const uint8_t *)flat_q8_0 + k;
    const uint8_t *scale_bytes1 = (const uint8_t *)flat_q8_1 + k;
    assert(k % GENIEX_PTQ1_BLOCK_K == 0);
    assert(valid_rows0 <= GENIEX_PTQ1_TILE_ROWS && valid_rows1 <= GENIEX_PTQ1_TILE_ROWS);
    for (unsigned row = 0; row < valid_rows0; ++row) outputs00[row] = outputs01[row] = 0.0f;
    for (unsigned row = 0; row < valid_rows1; ++row) outputs10[row] = outputs11[row] = 0.0f;
    for (uint32_t block = 0; block < k / GENIEX_PTQ1_BLOCK_K; ++block) {
        float scales0[4], scales1[4];
        if (!prepared_scales0) geniex_ptq1_flat_scales(scales0, scale_bytes0, block);
        if (!prepared_scales1) geniex_ptq1_flat_scales(scales1, scale_bytes1, block);
        geniex_ptq1_prepare_activation(scratch0, quants0 + block * GENIEX_PTQ1_BLOCK_K);
        geniex_ptq1_prepare_activation(scratch1, quants1 + block * GENIEX_PTQ1_BLOCK_K);
        geniex_ptq1_dot_tile_two(&weights0[block], scratch0, scratch1,
            prepared_scales0 ? prepared_scales0 + 4 * block : scales0,
            prepared_scales1 ? prepared_scales1 + 4 * block : scales1,
            outputs00, outputs01, valid_rows0);
        geniex_ptq1_dot_tile_two(&weights1[block], scratch0, scratch1,
            prepared_scales0 ? prepared_scales0 + 4 * block : scales0,
            prepared_scales1 ? prepared_scales1 + 4 * block : scales1,
            outputs10, outputs11, valid_rows1);
    }
}

static inline void geniex_ptq1_dot_pair_two_rows_flat_q8(uint32_t k,
    float *outputs00, float *outputs01, float *outputs10, float *outputs11,
    const geniex_ptq1_tile *weights0, const geniex_ptq1_tile *weights1,
    const void *flat_q8_0, const void *flat_q8_1, unsigned valid_rows0, unsigned valid_rows1,
    geniex_ptq1_activation *scratch0, geniex_ptq1_activation *scratch1) {
    geniex_ptq1_dot_pair_two_rows_flat_q8_scaled(k, outputs00, outputs01, outputs10, outputs11,
        weights0, weights1, flat_q8_0, flat_q8_1, valid_rows0, valid_rows1,
        scratch0, scratch1, NULL, NULL);
}
#endif
#endif
