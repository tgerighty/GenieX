// SPDX-License-Identifier: BSD-3-Clause
#ifndef GENIEX_PTQ1_TILE_H
#define GENIEX_PTQ1_TILE_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GENIEX_PTQ1_TILE_ROWS 32
#define GENIEX_PTQ1_BLOCK_K 128

typedef struct {
    uint8_t  qs[24];
    uint8_t  qh[2];
    uint16_t d;
} geniex_ptq1_block;

typedef struct {
    uint8_t  qs[24][GENIEX_PTQ1_TILE_ROWS];
    uint8_t  qh[2][GENIEX_PTQ1_TILE_ROWS];
    uint16_t d[GENIEX_PTQ1_TILE_ROWS];
} geniex_ptq1_tile;

#ifdef __cplusplus
static_assert(sizeof(geniex_ptq1_block) == 28, "PTQ1_0 block must be 28 bytes");
static_assert(offsetof(geniex_ptq1_block, d) == 26, "PTQ1_0 scale must start at byte 26");
static_assert(sizeof(geniex_ptq1_tile) == 896, "PTQ1_0 tile must be 896 bytes");
#else
_Static_assert(sizeof(geniex_ptq1_block) == 28, "PTQ1_0 block must be 28 bytes");
_Static_assert(offsetof(geniex_ptq1_block, d) == 26, "PTQ1_0 scale must start at byte 26");
_Static_assert(sizeof(geniex_ptq1_tile) == 896, "PTQ1_0 tile must be 896 bytes");
#endif

// src/dst use GGUF row-major blocks; row_stride_blocks is K / 128.
static inline void geniex_ptq1_pack_tile(
    geniex_ptq1_tile *tile, const void *src, size_t row_stride_blocks, size_t block_index, unsigned valid_rows) {
    if (valid_rows > GENIEX_PTQ1_TILE_ROWS) {
        assert(valid_rows <= GENIEX_PTQ1_TILE_ROWS);
        return;
    }
    const uint8_t *blocks = (const uint8_t *)src;
    for (unsigned row = 0; row < GENIEX_PTQ1_TILE_ROWS; ++row) {
        const uint8_t *block =
            row < valid_rows ? blocks + (row * row_stride_blocks + block_index) * sizeof(geniex_ptq1_block) : NULL;
        for (unsigned k = 0; k < 24; ++k) tile->qs[k][row] = block ? block[k] : 128;
        for (unsigned k = 0; k < 2; ++k) tile->qh[k][row] = block ? block[24 + k] : 128;
        tile->d[row] = 0;
        if (block) memcpy(&tile->d[row], block + 26, sizeof(tile->d[row]));
    }
}

static inline void geniex_ptq1_unpack_tile(
    void *dst, const geniex_ptq1_tile *tile, size_t row_stride_blocks, size_t block_index, unsigned valid_rows) {
    if (valid_rows > GENIEX_PTQ1_TILE_ROWS) {
        assert(valid_rows <= GENIEX_PTQ1_TILE_ROWS);
        return;
    }
    uint8_t *blocks = (uint8_t *)dst;
    for (unsigned row = 0; row < valid_rows; ++row) {
        uint8_t *block = blocks + (row * row_stride_blocks + block_index) * sizeof(geniex_ptq1_block);
        for (unsigned k = 0; k < 24; ++k) block[k] = tile->qs[k][row];
        for (unsigned k = 0; k < 2; ++k) block[24 + k] = tile->qh[k][row];
        memcpy(block + 26, &tile->d[row], sizeof(tile->d[row]));
    }
}

#ifdef __hexagon__
// DSP-only dequantization; host code uses the packing helpers above.
static inline float geniex_ptq1_fp16_to_fp32(uint16_t h) {
    union { uint16_t bits; _Float16 value; } half;
    half.bits = h;
    return (float) half.value;
}

static inline int geniex_ptq1_trit(uint8_t packed, unsigned power) {
    const uint8_t shifted = (uint8_t)(packed * (uint8_t)power);
    return (int)(((unsigned)shifted * 3u) >> 8) - 1;
}

// Prism qs traversal is a 16-byte stage then an 8-byte stage (not 8/8/8).
static inline void geniex_ptq1_dequant_tile_row(float *y, const geniex_ptq1_tile *tile, unsigned row) {
    static const unsigned powers[5] = {1, 3, 9, 27, 81};
    const float d = geniex_ptq1_fp16_to_fp32(tile->d[row]);
    unsigned out = 0;

    for (unsigned stage = 0; stage < 2; ++stage) {
        const unsigned width  = stage == 0 ? 16u : 8u;
        const unsigned offset = stage == 0 ? 0u : 16u;
        for (unsigned n = 0; n < 5; ++n) {
            for (unsigned m = 0; m < width; ++m) {
                y[out++] = (float)geniex_ptq1_trit(tile->qs[offset + m][row], powers[n]) * d;
            }
        }
    }
    for (unsigned n = 0; n < 4; ++n) {
        for (unsigned h = 0; h < 2; ++h) {
            y[out++] = (float)geniex_ptq1_trit(tile->qh[h][row], powers[n]) * d;
        }
    }
    assert(out == GENIEX_PTQ1_BLOCK_K);
}

// Repacked layout: tile-row major, then K blocks. tiles points at tile-row ct.
static inline void geniex_ptq1_dequant_repacked_row(
    float *y, const geniex_ptq1_tile *tiles, uint32_t n_k_tiles, unsigned row_in_tile) {
    for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
        geniex_ptq1_dequant_tile_row(y + (size_t)kt * GENIEX_PTQ1_BLOCK_K, &tiles[kt], row_in_tile);
    }
}

#endif  // __hexagon__
#endif  // GENIEX_PTQ1_TILE_H
