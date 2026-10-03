// SPDX-License-Identifier: BSD-3-Clause
// Accepted-baseline bit comparison, not an independent scalar oracle or serving test.
#include <inttypes.h>
#include <stdio.h>
#include "ptq1_hvx.h"

enum { ROWS = GENIEX_PTQ1_TILE_ROWS, MAX_K = 17408, MAX_BLOCKS = MAX_K / 128 };
static geniex_ptq1_block blocks[2][ROWS * MAX_BLOCKS];
static geniex_ptq1_tile weights[2][MAX_BLOCKS] __attribute__((aligned(128)));
static uint8_t flat[MAX_K + MAX_K / 16] __attribute__((aligned(128)));
static geniex_ptq1_activation scratch __attribute__((aligned(128)));
#if defined(GENIEX_PTQ1_QUAD_ORACLE) || defined(GENIEX_PTQ1_QUAD_TEST)
static float prepared_scales[4 * MAX_BLOCKS];
#endif

int main(void) {
    const uint16_t scales[] = {0x3555, 0x3c01, 0xb2ab, 0xbc03};
    const unsigned lengths[] = {256, 5120, 6144, 17408};
    const unsigned tails[] = {3, 31, 32};
    unsigned cases = 0;
#ifdef GENIEX_PTQ1_QUAD_TEST
    unsigned eight_cases = 0;
#endif
    for (unsigned ki = 0; ki < 4; ++ki) {
        const unsigned k = lengths[ki];
        for (unsigned mode = 0; mode < 3; ++mode) {
            for (unsigned i = 0; i < k; ++i)
                flat[i] = mode == 1 ? (uint8_t)-128 : mode == 2 ? 127 : (uint8_t)(i * 29 + 67);
            for (unsigned i = 0; i < k / 32; ++i)
                memcpy(flat + k + 2 * i, &scales[i % 4], sizeof(scales[0]));
            for (unsigned w = 0; w < 2; ++w) {
                for (unsigned row = 0; row < ROWS; ++row) {
                    for (unsigned b = 0; b < k / 128; ++b) {
                        geniex_ptq1_block *block = &blocks[w][row * MAX_BLOCKS + b];
                        for (unsigned i = 0; i < 24; ++i)
                            block->qs[i] = (uint8_t)(i * 32 + row + b * 17 + w * 91);
                        block->qh[0] = (uint8_t)(row * 7 + b * 29 + w * 31);
                        block->qh[1] = (uint8_t)(255 - row * 5 - b * 13 - w * 19);
                        block->d = scales[(row + b + w) % 4];
                    }
                }
                for (unsigned b = 0; b < k / 128; ++b)
                    geniex_ptq1_pack_tile(&weights[w][b], blocks[w], MAX_BLOCKS, b, ROWS);
            }
            for (unsigned ti = 0; ti < 3; ++ti) {
                const unsigned valid = tails[ti];
                float output[2][ROWS + 2];
                for (unsigned w = 0; w < 2; ++w)
                    for (unsigned row = 0; row < ROWS + 2; ++row) output[w][row] = 12345.0f;
                memset(&scratch, 0xa5, sizeof(scratch));
                geniex_ptq1_dot_pair_flat_q8(k, output[0], output[1], weights[0], weights[1],
                    flat, valid, valid, &scratch);
                uint64_t hash = UINT64_C(14695981039346656037);
                printf("PTQ1_BASELINE_BITS_RAW K=%u valid=%u mode=%u bits=", k, valid, mode);
                for (unsigned w = 0; w < 2; ++w) {
                    for (unsigned row = 0; row < ROWS + 2; ++row) {
                        uint32_t bits;
                        memcpy(&bits, &output[w][row], sizeof(bits));
                        if ((bits & 0x7f800000u) == 0x7f800000u ||
                            (row >= valid && output[w][row] != 12345.0f)) return 1;
                        hash = (hash ^ bits) * UINT64_C(1099511628211);
                        printf("%08" PRIx32, bits);
                    }
                }
                printf("\nPTQ1_BASELINE_BITS K=%u valid=%u mode=%u hash=%016" PRIx64 "\n", k, valid, mode, hash);
#if defined(GENIEX_PTQ1_QUAD_ORACLE) || defined(GENIEX_PTQ1_QUAD_TEST)
                const unsigned rows[4] = {valid, 32, valid == 32 ? 32 : mode == 0 ? 0 : 1,
                                          valid == 32 ? 32 : 31};
                for (unsigned cached = 0; cached < 2; ++cached) {
                    float expected[4][ROWS + 2];
#ifdef GENIEX_PTQ1_QUAD_TEST
                    float actual[4][ROWS + 2];
#endif
                    for (unsigned w = 0; w < 4; ++w)
                        for (unsigned row = 0; row < ROWS + 2; ++row)
                            expected[w][row] = 12345.0f;
#ifdef GENIEX_PTQ1_QUAD_TEST
                    for (unsigned w = 0; w < 4; ++w)
                        for (unsigned row = 0; row < ROWS + 2; ++row)
                            actual[w][row] = 12345.0f;
#endif
                    if (cached) geniex_ptq1_prepare_flat_scales(prepared_scales, flat, k);
                    const float *cache = cached ? prepared_scales : NULL;
                    geniex_ptq1_dot_pair_flat_q8_scaled(k, expected[0], expected[1],
                        weights[0], weights[1], flat, rows[0], rows[1], &scratch, cache);
                    geniex_ptq1_dot_pair_flat_q8_scaled(k, expected[2], expected[3],
                        weights[1], weights[0], flat, rows[2], rows[3], &scratch, cache);
                    printf("PTQ1_QUAD_ORACLE_BITS_RAW K=%u valid=%u mode=%u cached=%u bits=",
                        k, valid, mode, cached);
                    for (unsigned w = 0; w < 4; ++w) {
                        for (unsigned row = 0; row < ROWS + 2; ++row) {
                            uint32_t bits;
                            memcpy(&bits, &expected[w][row], sizeof(bits));
                            if ((bits & 0x7f800000u) == 0x7f800000u ||
                                (row >= rows[w] && expected[w][row] != 12345.0f)) return 1;
                            printf("%08" PRIx32, bits);
                        }
                    }
                    printf("\n");
#ifdef GENIEX_PTQ1_QUAD_TEST
                    geniex_ptq1_dot_quad_flat_q8_scaled(k,
                        actual[0], actual[1], actual[2], actual[3],
                        weights[0], weights[1], weights[1], weights[0], flat,
                        rows[0], rows[1], rows[2], rows[3], &scratch, cache);
                    if (memcmp(expected, actual, sizeof(expected))) {
                        fprintf(stderr, "PTQ1_QUAD_BITS_FAIL K=%u valid=%u mode=%u cached=%u\n",
                            k, valid, mode, cached);
                        return 1;
                    }
                    const geniex_ptq1_tile *eight_weights[8] = {
                        weights[0], weights[1], weights[1], weights[0],
                        weights[1], weights[0], weights[0], weights[1]};
                    float eight_expected[8 * ROWS];
                    float eight_actual_storage[8 * ROWS + 2];
                    float *eight_actual = eight_actual_storage + 1;
                    eight_actual_storage[0] = 12345.0f;
                    eight_actual_storage[8 * ROWS + 1] = 12345.0f;
                    for (unsigned pair = 0; pair < 4; ++pair)
                        geniex_ptq1_dot_pair_flat_q8_scaled(k,
                            eight_expected + 2 * pair * ROWS,
                            eight_expected + (2 * pair + 1) * ROWS,
                            eight_weights[2 * pair], eight_weights[2 * pair + 1],
                            flat, ROWS, ROWS, &scratch, cache);
                    geniex_ptq1_dot_eight_flat_q8_scaled(k, eight_actual,
                        eight_weights, flat, &scratch, cache);
                    if (memcmp(eight_expected, eight_actual, sizeof(eight_expected)) ||
                        eight_actual_storage[0] != 12345.0f ||
                        eight_actual_storage[8 * ROWS + 1] != 12345.0f) {
                        fprintf(stderr, "PTQ1_EIGHT_BITS_FAIL K=%u valid=%u mode=%u cached=%u\n",
                            k, valid, mode, cached);
                        return 1;
                    }
                    ++eight_cases;
#endif
                }
#endif
                ++cases;
            }
        }
    }
    printf("PTQ1_BASELINE_BITS_CASES=%u\n", cases);
#if defined(GENIEX_PTQ1_QUAD_ORACLE) || defined(GENIEX_PTQ1_QUAD_TEST)
    printf("PTQ1_QUAD_ORACLE_BITS_CASES=%u\n", cases * 2);
#ifdef GENIEX_PTQ1_QUAD_TEST
    printf("PTQ1_QUAD_BITS_CASES=%u\n", cases * 2);
    printf("PTQ1_EIGHT_BITS_CASES=%u\n", eight_cases);
#endif
#endif
    return 0;
}
