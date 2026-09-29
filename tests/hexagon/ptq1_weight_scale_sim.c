// SPDX-License-Identifier: BSD-3-Clause
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ptq1_hvx.h"

#ifndef PTQ1_PREP_REPEATS
#define PTQ1_PREP_REPEATS 1
#endif
#ifndef PTQ1_PREP_EXHAUSTIVE
#define PTQ1_PREP_EXHAUSTIVE 0
#endif

enum { GROUPS = 8, KB = 136, ROWS = GENIEX_PTQ1_TILE_ROWS };
static geniex_ptq1_tile weights[GROUPS][KB] __attribute__((aligned(128)));
static float scales[GROUPS][KB * ROWS] __attribute__((aligned(128)));

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float reference_half(uint16_t bits) {
    union { uint16_t bits; _Float16 value; } half = { .bits = bits };
    return (float)half.value;
}

static int check_rows(unsigned offset, unsigned valid_rows) {
    uint8_t storage[2 * sizeof(geniex_ptq1_tile) + 8] __attribute__((aligned(128)));
    float output[2 * ROWS + 2];
    const uint32_t sentinel = 0x4f123456u;
    memset(storage, 0xa5, sizeof(storage));
    geniex_ptq1_tile *tiles = (geniex_ptq1_tile *)(storage + offset);
    for (unsigned block = 0; block < 2; ++block) {
        for (unsigned row = 0; row < ROWS; ++row)
            tiles[block].d[row] = (uint16_t)(0x3400u + ((block + row) % 7u) * 0x80u);
    }
    for (unsigned row = 0; row < 2 * ROWS + 2; ++row)
        memcpy(&output[row], &sentinel, sizeof(sentinel));
    geniex_ptq1_prepare_weight_scales(output + 1, tiles, 256, valid_rows);
    if (float_bits(output[0]) != sentinel || float_bits(output[2 * ROWS + 1]) != sentinel) return 1;
    for (unsigned block = 0; block < 2; ++block) {
        for (unsigned row = 0; row < ROWS; ++row) {
            const uint32_t expected = row < valid_rows ?
                float_bits(reference_half(tiles[block].d[row])) : sentinel;
            if (float_bits(output[1 + block * ROWS + row]) != expected) return 2;
        }
    }
    for (unsigned i = 0; i < offset; ++i) if (storage[i] != 0xa5) return 3;
    for (unsigned i = offset + 2 * sizeof(*tiles); i < sizeof(storage); ++i)
        if (storage[i] != 0xa5) return 4;
    return 0;
}

static int check_all_half_patterns(void) {
    geniex_ptq1_tile tile __attribute__((aligned(128))) = {0};
    float output[ROWS];
    for (unsigned first = 0; first <= UINT16_MAX; first += ROWS) {
        for (unsigned row = 0; row < ROWS; ++row) tile.d[row] = (uint16_t)(first + row);
        geniex_ptq1_prepare_weight_scales(output, &tile, 128, ROWS);
        for (unsigned row = 0; row < ROWS; ++row) {
            if (float_bits(output[row]) != float_bits(reference_half(tile.d[row]))) {
                printf("PTQ1 scale mismatch raw=%04x\n", tile.d[row]);
                return 1;
            }
        }
    }
    const uint16_t special[] = {0, 1, 0x7c00, 0x7e00, 0x8000, 0x8001, 0xfc00, 0xfe00};
    for (unsigned i = 0; i < sizeof(special) / sizeof(special[0]); ++i)
        for (unsigned lane = 0; lane < ROWS; ++lane) {
            for (unsigned row = 0; row < ROWS; ++row) tile.d[row] = 0x3c00;
            tile.d[lane] = special[i];
            geniex_ptq1_prepare_weight_scales(output, &tile, 128, ROWS);
            for (unsigned row = 0; row < ROWS; ++row)
                if (float_bits(output[row]) != float_bits(reference_half(tile.d[row]))) return 2;
        }
    for (unsigned offset = 2; offset <= 4; offset += 2)
        for (unsigned rows = 0; rows <= ROWS; ++rows)
            if (check_rows(offset, rows)) return 3;
    puts("PTQ1 weight-scale exhaustive and boundary checks passed");
    return 0;
}

int main(void) {
#if PTQ1_PREP_EXHAUSTIVE
    if (check_all_half_patterns()) return 1;
#endif
    for (unsigned group = 0; group < GROUPS; ++group)
        for (unsigned block = 0; block < KB; ++block)
            for (unsigned row = 0; row < ROWS; ++row)
                weights[group][block].d[row] = (uint16_t)(0x3400u + ((group + block + row) % 7u) * 0x80u);
    for (unsigned repeat = 0; repeat < PTQ1_PREP_REPEATS; ++repeat) {
        for (unsigned group = 0; group < GROUPS; ++group)
            geniex_ptq1_prepare_weight_scales(scales[group], weights[group], KB * GENIEX_PTQ1_BLOCK_K, ROWS);
        __asm__ volatile("" ::: "memory");
    }
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned group = 0; group < GROUPS; ++group)
        for (unsigned block = 0; block < KB; ++block)
            for (unsigned row = 0; row < ROWS; ++row) {
                const unsigned index = block * ROWS + row;
                const uint32_t actual = float_bits(scales[group][index]);
                const uint32_t expected = float_bits(reference_half(weights[group][block].d[row]));
                if (actual != expected) return 2;
                hash = (hash ^ actual) * UINT64_C(1099511628211);
            }
    printf("PTQ1 weight scales repeats=%d hash %016llx\n", PTQ1_PREP_REPEATS, (unsigned long long)hash);
    return 0;
}
