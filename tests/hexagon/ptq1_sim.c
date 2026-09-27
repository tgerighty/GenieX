// SPDX-License-Identifier: BSD-3-Clause
// Standalone PTQ1_0 unpack check for the VENTUNO Q's Hexagon HTP v75 simulator.
// The packing order follows PrismML's block_ptq1_0 (24 qs bytes, 2 qh bytes).
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ptq1_hvx.h"
#include "ptq1_tile.h"

#ifdef __hexagon__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

#define TILE_ROWS GENIEX_PTQ1_TILE_ROWS
typedef geniex_ptq1_block ptq1_block;
typedef geniex_ptq1_tile  ptq1_tile;

static int trit(uint8_t packed, unsigned power) {
    const uint8_t shifted = (uint8_t)(packed * power);
    return (int)(((unsigned)shifted * 3u) >> 8) - 1;
}

static int dot_with_index(const ptq1_block *block) {
    const unsigned powers[5] = {1, 3, 9, 27, 81};
    int            sum       = 0;
    unsigned       index     = 0;

    for (unsigned group = 0; group < 3; ++group) {
        const unsigned offset = group * 8;
        for (unsigned n = 0; n < 5; ++n) {
            for (unsigned m = 0; m < 8; ++m) {
                sum += trit(block->qs[offset + m], powers[n]) * (int)++index;
            }
        }
    }
    for (unsigned n = 0; n < 4; ++n) {
        for (unsigned h = 0; h < 2; ++h) {
            sum += trit(block->qh[h], powers[n]) * (int)++index;
        }
    }
    return index == 128 ? sum : 0;
}

static int dot_reference(const ptq1_block *block, const int8_t *activation) {
    const unsigned powers[5] = {1, 3, 9, 27, 81};
    int            sum       = 0;
    unsigned       index     = 0;

    for (unsigned group = 0; group < 3; ++group) {
        for (unsigned n = 0; n < 5; ++n) {
            for (unsigned m = 0; m < 8; ++m) {
                sum += trit(block->qs[group * 8 + m], powers[n]) * activation[index++];
            }
        }
    }
    for (unsigned n = 0; n < 4; ++n) {
        for (unsigned h = 0; h < 2; ++h) {
            sum += trit(block->qh[h], powers[n]) * activation[index++];
        }
    }
    return sum;
}

static float half_to_float(uint16_t bits) {
    union {
        uint16_t bits;
        _Float16 value;
    } half = {.bits = bits};
    return (float)half.value;
}

static float dot_reference_scaled(const ptq1_block *block, const int8_t *activation, const float scales[4]) {
    const unsigned powers[5]  = {1, 3, 9, 27, 81};
    int32_t        partial[4] = {0};
    unsigned       index      = 0;

    for (unsigned group = 0; group < 3; ++group) {
        for (unsigned n = 0; n < 5; ++n) {
            for (unsigned m = 0; m < 8; ++m) {
                partial[index / 32] += trit(block->qs[group * 8 + m], powers[n]) * activation[index];
                ++index;
            }
        }
    }
    for (unsigned n = 0; n < 4; ++n) {
        for (unsigned h = 0; h < 2; ++h) {
            partial[3] += trit(block->qh[h], powers[n]) * activation[index++];
        }
    }
    float result = 0.0f;
    for (unsigned b = 0; b < 4; ++b) result += scales[b] * partial[b];
    return half_to_float(block->d) * result;
}

static void dot_tiled(const ptq1_tile *restrict tile, const int8_t *restrict activation, int32_t *restrict sums) {
    const unsigned powers[5] = {1, 3, 9, 27, 81};
    unsigned       index     = 0;

    for (unsigned row = 0; row < TILE_ROWS; ++row) sums[row] = 0;
    for (unsigned group = 0; group < 3; ++group) {
        for (unsigned n = 0; n < 5; ++n) {
            for (unsigned m = 0; m < 8; ++m) {
                const uint8_t *codes = tile->qs[group * 8 + m];
                const int      a     = activation[index++];
                for (unsigned row = 0; row < TILE_ROWS; ++row) {
                    sums[row] += trit(codes[row], powers[n]) * a;
                }
            }
        }
    }
    for (unsigned n = 0; n < 4; ++n) {
        for (unsigned h = 0; h < 2; ++h) {
            const uint8_t *codes = tile->qh[h];
            const int      a     = activation[index++];
            for (unsigned row = 0; row < TILE_ROWS; ++row) {
                sums[row] += trit(codes[row], powers[n]) * a;
            }
        }
    }
}

#ifdef __hexagon__
#if !defined(PTQ1_BENCH) && !defined(PTQ1_BENCH_STREAM) && !defined(PTQ1_BENCH_FULLK)
static int verify_hvx_trits(void) {
    static const unsigned powers[5] = {1, 3, 9, 27, 81};
    int16_t               codes[64] __attribute__((aligned(128)));
    int16_t               decoded[64] __attribute__((aligned(128)));
    for (unsigned code = 0; code < 256; ++code) {
        for (unsigned lane = 0; lane < 64; ++lane) codes[lane] = (int16_t)code;
        const HVX_Vector input = *(const HVX_Vector *)codes;
        for (unsigned p = 0; p < 5; ++p) {
            *(HVX_Vector *)decoded = geniex_ptq1_decode_trit(input, powers[p]);
            for (unsigned lane = 0; lane < 64; ++lane) {
                if (decoded[lane] != trit((uint8_t)code, powers[p])) {
                    printf("PTQ1_0 trit mismatch: code %u power %u lane %u\n", code, powers[p], lane);
                    return 1;
                }
            }
        }
    }
    return 0;
}
#endif

#endif

int main(void) {
    ptq1_block block = {.d = 0x3c00};                     // FP16 scale 1.0
    for (unsigned i = 0; i < 24; ++i) block.qs[i] = 128;  // zero trits
    for (unsigned i = 0; i < 2; ++i) block.qh[i] = 128;
    block.qs[0] = 255;  // +1 at indices 1, 9, 17, 25, 33
    block.qh[0] = 0;    // -1 at indices 121, 123, 125, 127

    const int actual = dot_with_index(&block);
    if (actual != -411 || sizeof(block) != 28) return 1;

    ptq1_block blocks[TILE_ROWS];
    ptq1_tile  tile;
    int8_t     activation[128];
    int32_t    sums[TILE_ROWS];
    uint32_t   seed = 7;

    for (unsigned row = 0; row < TILE_ROWS; ++row) {
        for (unsigned k = 0; k < 24; ++k) {
            seed              = seed * 1664525u + 1013904223u;
            blocks[row].qs[k] = (uint8_t)(seed >> 24);
        }
        for (unsigned k = 0; k < 2; ++k) {
            seed              = seed * 1664525u + 1013904223u;
            blocks[row].qh[k] = (uint8_t)(seed >> 24);
        }
        blocks[row].d = 0x3c00;
    }
    for (unsigned k = 0; k < 128; ++k) activation[k] = (int8_t)((k * 29u) % 255u - 127);

    geniex_ptq1_pack_tile(&tile, blocks, 1, 0, TILE_ROWS);
#if !defined(PTQ1_BENCH) && !defined(PTQ1_BENCH_STREAM) && !defined(PTQ1_BENCH_FULLK)
    ptq1_block roundtrip[TILE_ROWS];
    geniex_ptq1_unpack_tile(roundtrip, &tile, 1, 0, TILE_ROWS);
    if (memcmp(roundtrip, blocks, sizeof(blocks)) != 0) return 1;

    ptq1_block two_blocks[TILE_ROWS][2];
    for (unsigned row = 0; row < TILE_ROWS; ++row) {
        two_blocks[row][0] = blocks[row];
        two_blocks[row][1] = blocks[TILE_ROWS - 1 - row];
    }
    geniex_ptq1_pack_tile(&tile, two_blocks, 2, 1, TILE_ROWS - 1);
    if (tile.d[TILE_ROWS - 1] != 0 || tile.qs[0][TILE_ROWS - 1] != 128 || tile.qh[0][TILE_ROWS - 1] != 128) return 1;
    for (unsigned row = 0; row < TILE_ROWS; ++row) memset(&two_blocks[row][1], 0xa5, sizeof(ptq1_block));
    geniex_ptq1_unpack_tile(two_blocks, &tile, 2, 1, TILE_ROWS - 1);
    for (unsigned row = 0; row < TILE_ROWS - 1; ++row) {
        if (memcmp(&two_blocks[row][1], &blocks[TILE_ROWS - 1 - row], sizeof(ptq1_block)) != 0) return 1;
    }
    if (((const uint8_t *)&two_blocks[TILE_ROWS - 1][1])[0] != 0xa5) return 1;
    geniex_ptq1_pack_tile(&tile, blocks, 1, 0, TILE_ROWS);
#endif
    dot_tiled(&tile, activation, sums);
    const float unit_scales[4] = {1, 1, 1, 1};
    for (unsigned row = 0; row < TILE_ROWS; ++row) {
        if (sums[row] != dot_reference(&blocks[row], activation)) {
            printf("PTQ1_0 tile mismatch at row %u\n", row);
            return 1;
        }
        if (dot_reference_scaled(&blocks[row], activation, unit_scales) != sums[row]) return 1;
    }
#ifdef __hexagon__
    float hvx_outputs[TILE_ROWS];
#ifdef PTQ1_BENCH
    geniex_ptq1_activation prepared;
    float                  checksum  = 0;
    const float            scales[4] = {0.5f, 1.0f, 2.0f, 4.0f};
    for (unsigned i = 0; i < 256; ++i) {
        activation[i % 128] = (int8_t)((int)(i % 255) - 127);
        geniex_ptq1_prepare_activation(&prepared, activation);
        geniex_ptq1_dot_tile(&tile, &prepared, scales, hvx_outputs);
        checksum += hvx_outputs[i % TILE_ROWS];
    }
    printf("PTQ1_0 256 tiles checksum: %.1f\n", checksum);
    return 0;
#elif defined(PTQ1_BENCH_STREAM)
    geniex_ptq1_activation prepared;
    static ptq1_tile       stream[256] __attribute__((aligned(128)));
    float                  checksum  = 0;
    const float            scales[4] = {0.5f, 1.0f, 2.0f, 4.0f};
    for (unsigned i = 0; i < 256; ++i) {
        stream[i] = tile;
        stream[i].qs[0][i % TILE_ROWS] ^= (uint8_t)(i + 1);
    }
    geniex_ptq1_prepare_activation(&prepared, activation);
    for (unsigned i = 0; i < 256; ++i) {
        geniex_ptq1_dot_tile(&stream[i], &prepared, scales, hvx_outputs);
        checksum += hvx_outputs[i % TILE_ROWS];
    }
    printf("PTQ1_0 256 streamed tiles checksum: %.1f\n", checksum);
    return 0;
#elif defined(PTQ1_BENCH_FULLK)
    enum { K = 5120, K_BLOCKS = K / GENIEX_PTQ1_BLOCK_K, WEIGHT_TILES = 32 };
    static geniex_ptq1_tile weights[WEIGHT_TILES][K_BLOCKS] __attribute__((aligned(128)));
    static uint8_t          flat_q8[K + 384] __attribute__((aligned(128)));
    const uint16_t          scale_bits = 0x3c00;
    for (unsigned k = 0; k < K; ++k) flat_q8[k] = (uint8_t)activation[k % 128];
    for (unsigned b = 0; b < K / 32; ++b) memcpy(flat_q8 + K + 2 * b, &scale_bits, sizeof(scale_bits));
    for (unsigned ct = 0; ct < WEIGHT_TILES; ++ct) {
        for (unsigned kb = 0; kb < K_BLOCKS; ++kb) {
            weights[ct][kb] = tile;
            weights[ct][kb].qs[0][ct] ^= (uint8_t)(ct + kb + 1);
        }
    }
    float checksum = 0.0f;
#ifdef PTQ1_BENCH_REUSE
    static geniex_ptq1_activation prepped[K_BLOCKS];
    const float                   unit_scales_fullk[4] = {1, 1, 1, 1};
    float                         partial[TILE_ROWS];
    for (unsigned kb = 0; kb < K_BLOCKS; ++kb) {
        geniex_ptq1_prepare_activation(&prepped[kb], (const int8_t *)flat_q8 + kb * GENIEX_PTQ1_BLOCK_K);
    }
    for (unsigned ct = 0; ct < WEIGHT_TILES; ++ct) {
        for (unsigned row = 0; row < TILE_ROWS; ++row) hvx_outputs[row] = 0.0f;
        for (unsigned kb = 0; kb < K_BLOCKS; ++kb) {
            geniex_ptq1_dot_tile(&weights[ct][kb], &prepped[kb], unit_scales_fullk, partial);
            for (unsigned row = 0; row < TILE_ROWS; ++row) hvx_outputs[row] += partial[row];
        }
        checksum += hvx_outputs[ct];
    }
#elif defined(PTQ1_BENCH_PAIR)
    (void)hvx_outputs;
    geniex_ptq1_activation prepared;
    float                  pair[2][TILE_ROWS];
    for (unsigned ct = 0; ct < WEIGHT_TILES; ct += 2) {
        geniex_ptq1_dot_pair_flat_q8(
            K, pair[0], pair[1], weights[ct], weights[ct + 1], flat_q8, TILE_ROWS, TILE_ROWS, &prepared);
        checksum += pair[0][ct] + pair[1][ct + 1];
    }
#else
    geniex_ptq1_activation prepared;
    for (unsigned ct = 0; ct < WEIGHT_TILES; ++ct) {
        geniex_ptq1_dot_flat_q8(K, hvx_outputs, weights[ct], flat_q8, TILE_ROWS, &prepared);
        checksum += hvx_outputs[ct];
    }
#endif
    printf("PTQ1_0 5120-K 32-tile checksum: %.1f\n", checksum);
    return 0;
#else
    geniex_ptq1_activation prepared;
    if (verify_hvx_trits()) return 1;
    for (unsigned k = 0; k < 128; ++k) {
        int8_t basis[128] = {0};
        basis[k]          = 1;
        geniex_ptq1_prepare_activation(&prepared, basis);
        geniex_ptq1_dot_tile(&tile, &prepared, unit_scales, hvx_outputs);
        for (unsigned row = 0; row < TILE_ROWS; ++row) {
            if (hvx_outputs[row] != dot_reference(&blocks[row], basis)) {
                printf("PTQ1_0 HVX basis mismatch at k %u row %u: got %.0f expected %d\n",
                    k,
                    row,
                    hvx_outputs[row],
                    dot_reference(&blocks[row], basis));
                return 1;
            }
        }
    }
    const float scales[4] = {0.5f, 1.0f, 2.0f, 4.0f};
    for (unsigned row = 0; row < TILE_ROWS; ++row) {
        const uint16_t half_scales[3] = {0x3800, 0x3c00, 0x4000};
        blocks[row].d                 = half_scales[row % 3];
    }
    geniex_ptq1_pack_tile(&tile, blocks, 1, 0, TILE_ROWS);
    geniex_ptq1_prepare_activation(&prepared, activation);
    geniex_ptq1_dot_tile(&tile, &prepared, scales, hvx_outputs);
    for (unsigned row = 0; row < TILE_ROWS; ++row) {
        if (hvx_outputs[row] != dot_reference_scaled(&blocks[row], activation, scales)) {
            printf("PTQ1_0 HVX scale mismatch at row %u\n", row);
            return 1;
        }
    }

    geniex_ptq1_tile two_tiles[2];
    uint8_t          flat_q8[384] __attribute__((aligned(128)));
    const uint16_t   scale_bits[8] = {0x3800, 0x3c00, 0x4000, 0x4200, 0x3400, 0x4400, 0x3c00, 0x3800};
    float            expected_scales[8];
    memcpy(flat_q8, activation, 128);
    for (unsigned k = 0; k < 128; ++k) flat_q8[128 + k] = (uint8_t)((k * 17u) % 255u - 127);
    memcpy(flat_q8 + 256, scale_bits, sizeof(scale_bits));
    for (unsigned b = 0; b < 8; ++b) expected_scales[b] = half_to_float(scale_bits[b]);
    for (unsigned row = 0; row < TILE_ROWS; ++row) {
        two_blocks[row][0] = blocks[row];
        two_blocks[row][1] = blocks[TILE_ROWS - 1 - row];
    }
    geniex_ptq1_pack_tile(&two_tiles[0], two_blocks, 2, 0, TILE_ROWS - 1);
    geniex_ptq1_pack_tile(&two_tiles[1], two_blocks, 2, 1, TILE_ROWS - 1);
    for (unsigned row = 0; row < TILE_ROWS; ++row) hvx_outputs[row] = 12345.0f;
    geniex_ptq1_dot_flat_q8(256, hvx_outputs, two_tiles, flat_q8, TILE_ROWS - 1, &prepared);
    for (unsigned row = 0; row < TILE_ROWS - 1; ++row) {
        const float expected =
            dot_reference_scaled(&two_blocks[row][0], (const int8_t *)flat_q8, expected_scales) +
            dot_reference_scaled(&two_blocks[row][1], (const int8_t *)flat_q8 + 128, expected_scales + 4);
        if (hvx_outputs[row] != expected) {
            printf("PTQ1_0 flat Q8 mismatch at row %u\n", row);
            return 1;
        }
    }
    if (hvx_outputs[TILE_ROWS - 1] != 12345.0f) return 1;

    float pair0[TILE_ROWS];
    float pair1[TILE_ROWS];
    for (unsigned row = 0; row < TILE_ROWS; ++row) pair0[row] = pair1[row] = 12345.0f;
    geniex_ptq1_dot_pair_flat_q8(
        256, pair0, pair1, two_tiles, two_tiles, flat_q8, TILE_ROWS - 1, TILE_ROWS - 2, &prepared);
    for (unsigned row = 0; row < TILE_ROWS - 1; ++row) {
        if (pair0[row] != hvx_outputs[row]) return 1;
        if (row < TILE_ROWS - 2 && pair1[row] != hvx_outputs[row]) return 1;
    }
    if (pair0[TILE_ROWS - 1] != 12345.0f || pair1[TILE_ROWS - 2] != 12345.0f || pair1[TILE_ROWS - 1] != 12345.0f)
        return 1;
#endif
#endif
    printf("PTQ1_0 tile: 32 rows match CPU reference\n");
    return 0;
}
