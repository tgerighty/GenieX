// SPDX-License-Identifier: BSD-3-Clause
#ifndef GENIEX_PTQ1_HVX_H
#define GENIEX_PTQ1_HVX_H

#include <string.h>

#include "ptq1_tile.h"

#ifdef __hexagon__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

static inline float geniex_ptq1_half_to_float(uint16_t bits) {
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
    pair->hi = *(const HVX_Vector *)&lanes[64];
#else
    const HVX_VectorPred first = Q6_Q_vsetq_R(64);
    const HVX_Vector     a0    = Q6_Vh_vsplat_R(activation[offset]);
    const HVX_Vector     a1    = Q6_Vh_vsplat_R(activation[offset + 1]);
    const HVX_Vector     a2    = Q6_Vh_vsplat_R(count > 2 ? activation[offset + 2] : 0);
    const HVX_Vector     a3    = Q6_Vh_vsplat_R(count > 2 ? activation[offset + 3] : 0);
    pair->lo                   = Q6_V_vmux_QVV(first, a0, a1);
    pair->hi                   = Q6_V_vmux_QVV(first, a2, a3);
#endif
}

// Prism stores qs in a 16-byte stage followed by an 8-byte stage.
static inline unsigned geniex_ptq1_qs_activation_index(unsigned group, unsigned power, unsigned lane) {
    return group < 2 ? power * 16 + group * 8 + lane : 80 + power * 8 + lane;
}

static inline void geniex_ptq1_prepare_activation(geniex_ptq1_activation *prepared, const int8_t *activation) {
    for (unsigned group = 0; group < 3; ++group) {
        for (unsigned n = 0; n < 5; ++n) {
            for (unsigned chunk = 0; chunk < 2; ++chunk) {
                geniex_ptq1_prepare_act_pair(
                    &prepared->qs[group][n][chunk], activation,
                    geniex_ptq1_qs_activation_index(group, n, chunk * 4), 4);
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
    HVX_Vector x = Q6_Vh_vmpyi_VhVh(bytes, Q6_Vh_vsplat_R(power));
#endif
    x = Q6_V_vand_VV(x, mask);
    x = Q6_Vh_vadd_VhVh(x, Q6_Vh_vasl_VhR(x, 1));
    x = Q6_Vh_vasr_VhR(x, 8);
    return Q6_Vh_vsub_VhVh(x, one);
}

static inline void geniex_ptq1_accumulate(
    HVX_VectorPair products, unsigned power, const geniex_ptq1_act_pair *activation, HVX_Vector sums[2]) {
    const HVX_Vector     lo       = geniex_ptq1_decode_trit(Q6_V_lo_W(products), power);
    const HVX_Vector     hi       = geniex_ptq1_decode_trit(Q6_V_hi_W(products), power);
    sums[0]                       = Q6_Vh_vmpyiacc_VhVhVh(sums[0], lo, activation->lo);
    sums[1]                       = Q6_Vh_vmpyiacc_VhVhVh(sums[1], hi, activation->hi);
}

static inline void geniex_ptq1_dot_tile(
    const geniex_ptq1_tile *tile, const geniex_ptq1_activation *activation, const float scales[4], float *outputs) {
    static const unsigned powers[5] = {1, 3, 9, 27, 81};
    // Each scale group has 32 ternary products: abs(sum) <= 32 * 128 = 4096.
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
#pragma clang loop unroll(full)
            for (unsigned n = 0; n < 5; ++n) {
                const unsigned scale_index = geniex_ptq1_qs_activation_index(group, n, m) / 32;
                geniex_ptq1_accumulate(products, powers[n], &activation->qs[group][n][m / 4], acc[scale_index]);
            }
        }
    }
    const HVX_VectorPair qh_products = Q6_Wuh_vunpack_Vub(*(const HVX_UVector *)tile->qh[0]);
#pragma clang loop unroll(full)
    for (unsigned n = 0; n < 4; ++n) {
        geniex_ptq1_accumulate(qh_products, powers[n], &activation->qh[n], acc[3]);
    }
#ifdef PTQ1_SCALAR_REDUCE
    for (unsigned b = 0; b < 4; ++b) {
        for (unsigned m = 0; m < 2; ++m) *(HVX_Vector *)lanes[b][m] = acc[b][m];
    }
#else
    for (unsigned b = 0; b < 4; ++b) {
        const HVX_Vector lo = Q6_Vh_vadd_VhVh(acc[b][0], Q6_V_vror_VR(acc[b][0], 64));
        const HVX_Vector hi = Q6_Vh_vadd_VhVh(acc[b][1], Q6_V_vror_VR(acc[b][1], 64));
        *(HVX_Vector *)partial[b] = Q6_Vh_vadd_VhVh(lo, hi);
    }
#endif
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

// Weights contain one 32-row tile per 128-wide K block. The flat Q8 row
// contains K int8 values, then four FP16 scales per K block.
static inline void geniex_ptq1_flat_scales(float scales[4], const uint8_t *scale_bytes, uint32_t block) {
    for (unsigned b = 0; b < 4; ++b) {
        uint16_t bits;
        memcpy(&bits, scale_bytes + 2 * (block * 4 + b), sizeof(bits));
        scales[b] = geniex_ptq1_half_to_float(bits);
    }
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

static inline void geniex_ptq1_dot_pair_flat_q8(uint32_t k, float *outputs0, float *outputs1,
    const geniex_ptq1_tile *weights0, const geniex_ptq1_tile *weights1, const void *flat_q8, unsigned valid_rows0,
    unsigned valid_rows1, geniex_ptq1_activation *scratch) {
    const int8_t  *quants      = (const int8_t *)flat_q8;
    const uint8_t *scale_bytes = (const uint8_t *)flat_q8 + k;
    float          partial[GENIEX_PTQ1_TILE_ROWS];

    assert(k % GENIEX_PTQ1_BLOCK_K == 0);
    assert(valid_rows0 <= GENIEX_PTQ1_TILE_ROWS && valid_rows1 <= GENIEX_PTQ1_TILE_ROWS);
    for (unsigned row = 0; row < valid_rows0; ++row) outputs0[row] = 0.0f;
    for (unsigned row = 0; row < valid_rows1; ++row) outputs1[row] = 0.0f;
    for (uint32_t block = 0; block < k / GENIEX_PTQ1_BLOCK_K; ++block) {
        float scales[4];
        geniex_ptq1_flat_scales(scales, scale_bytes, block);
        geniex_ptq1_prepare_activation(scratch, quants + block * GENIEX_PTQ1_BLOCK_K);
        geniex_ptq1_dot_tile(&weights0[block], scratch, scales, partial);
        for (unsigned row = 0; row < valid_rows0; ++row) outputs0[row] += partial[row];
        geniex_ptq1_dot_tile(&weights1[block], scratch, scales, partial);
        for (unsigned row = 0; row < valid_rows1; ++row) outputs1[row] += partial[row];
    }
}
#endif
#endif
