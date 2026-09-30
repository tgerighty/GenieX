// SPDX-License-Identifier: BSD-3-Clause
// Build with Prism's dma-queue.c and run on hexagon-sim v75.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dma-queue.h"
#include "ptq1_hvx.h"

#ifndef PTQ1_DMA_TILES
#define PTQ1_DMA_TILES 32
#endif
enum { K = 5120, K_BLOCKS = K / GENIEX_PTQ1_BLOCK_K, TILES = PTQ1_DMA_TILES };

static geniex_ptq1_tile weights[TILES][K_BLOCKS] __attribute__((aligned(128)));
static geniex_ptq1_tile scratch_weights[2][K_BLOCKS] __attribute__((aligned(128)));
static uint8_t          flat_q8[K + 384] __attribute__((aligned(128)));
static uint8_t          queue_storage[4096] __attribute__((aligned(128)));
static float            outputs[TILES][GENIEX_PTQ1_TILE_ROWS];

static void fill_input(void) {
    geniex_ptq1_block blocks[GENIEX_PTQ1_TILE_ROWS];
    uint32_t          seed = 7;
    for (unsigned row = 0; row < GENIEX_PTQ1_TILE_ROWS; ++row) {
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
    geniex_ptq1_tile tile;
    geniex_ptq1_pack_tile(&tile, blocks, 1, 0, GENIEX_PTQ1_TILE_ROWS);
    for (unsigned k = 0; k < K; ++k) flat_q8[k] = (uint8_t)((k * 29u % 255u) - 127);
    const uint16_t one = 0x3c00;
    for (unsigned b = 0; b < K / 32; ++b) memcpy(flat_q8 + K + 2 * b, &one, sizeof(one));
    for (unsigned ct = 0; ct < TILES; ++ct) {
        for (unsigned kb = 0; kb < K_BLOCKS; ++kb) {
            weights[ct][kb] = tile;
            weights[ct][kb].qs[0][ct] ^= (uint8_t)(ct + kb + 1);
        }
    }
}

static int push_tile(dma_queue *queue, unsigned tile, void *target) {
    return dma_queue_push(queue,
        dma_make_ptr(target, weights[tile]),
        sizeof(weights[tile][0]),
        sizeof(weights[tile][0]),
        sizeof(weights[tile][0]),
        K_BLOCKS);
}

int main(void) {
    fill_input();
    struct htp_thread_trace trace = {0};
    dma_queue *queue = dma_queue_init(queue_storage, 8, (uintptr_t)scratch_weights, sizeof(scratch_weights), &trace);
    if (!push_tile(queue, 0, scratch_weights[0])) return 1;
    if (TILES > 1 && !push_tile(queue, 1, scratch_weights[1])) return 1;
    unsigned               next = TILES > 1 ? 2 : 1;
    geniex_ptq1_activation prepared;
#ifdef PTQ1_DMA_PAIR
    for (unsigned ct = 0; ct < TILES;) {
        const geniex_ptq1_tile *w0 = dma_queue_pop(queue).dst;
        if (!w0) return 2;
        const geniex_ptq1_tile *w1 = NULL;
        if (ct + 1 < TILES) {
            w1 = dma_queue_pop(queue).dst;
            if (!w1) return 2;
            geniex_ptq1_dot_pair_flat_q8(K,
                outputs[ct],
                outputs[ct + 1],
                w0,
                w1,
                flat_q8,
                GENIEX_PTQ1_TILE_ROWS,
                GENIEX_PTQ1_TILE_ROWS,
                &prepared);
        } else {
            geniex_ptq1_dot_flat_q8(K, outputs[ct], w0, flat_q8, GENIEX_PTQ1_TILE_ROWS, &prepared);
        }
        if (next < TILES && !push_tile(queue, next++, (void *)w0)) return 3;
        if (w1 && next < TILES && !push_tile(queue, next++, (void *)w1)) return 4;
        ct += w1 ? 2 : 1;
    }
#else
    for (unsigned ct = 0; ct < TILES; ++ct) {
        const geniex_ptq1_tile *w = dma_queue_pop(queue).dst;
        if (!w) return 2;
        geniex_ptq1_dot_flat_q8(K, outputs[ct], w, flat_q8, GENIEX_PTQ1_TILE_ROWS, &prepared);
        if (next < TILES && !push_tile(queue, next++, (void *)w)) return 3;
    }
#endif
#ifdef PTQ1_DMA_VERIFY
    for (unsigned ct = 0; ct < TILES; ++ct) {
        float expected[GENIEX_PTQ1_TILE_ROWS];
        geniex_ptq1_dot_flat_q8(K, expected, weights[ct], flat_q8, GENIEX_PTQ1_TILE_ROWS, &prepared);
        if (memcmp(expected, outputs[ct], sizeof(expected)) != 0) return 5;
    }
#endif
    float checksum = 0.0f;
    for (unsigned ct = 0; ct < TILES; ++ct) checksum += outputs[ct][ct];
    printf("PTQ1 DMA checksum %.1f\n", checksum);
    return 0;
}
