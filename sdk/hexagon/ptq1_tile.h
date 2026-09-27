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

#endif
