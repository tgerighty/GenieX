// SPDX-License-Identifier: BSD-3-Clause
// v75 actual-worker fixture: links real hexagon/htp/get-rows-ops.c and calls
// op_get_rows. Not a host replica of dispatch (see ptq1_get_rows_sim.c).
// Built with Hexagon SDK / computev75 on Plex; adapters supply sync workers.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "get-rows-ops.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "ptq1_tile.h"
#include "work-queue.h"

static void expect(int cond, const char * msg) {
    if (!cond) {
        printf("FAIL: %s\n", msg);
        exit(1);
    }
}

static uint32_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *)p;
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 16777619u;
    }
    return h;
}

static float oracle_fp16_to_fp32(uint16_t h) {
    const uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t m = mant;
            uint32_t e = 127 - 15 + 1;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x3ffu;
            bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static void oracle_dequant_block(float * y, const geniex_ptq1_block * x) {
    static const unsigned powers[5] = {1, 3, 9, 27, 81};
    const float d = oracle_fp16_to_fp32(x->d);
    unsigned out = 0;
    for (unsigned stage = 0; stage < 2; ++stage) {
        const unsigned width  = stage == 0 ? 16u : 8u;
        const unsigned offset = stage == 0 ? 0u : 16u;
        for (unsigned n = 0; n < 5; ++n) {
            for (unsigned m = 0; m < width; ++m) {
                const uint8_t q = (uint8_t)(x->qs[offset + m] * (uint8_t)powers[n]);
                const int16_t xi = (int16_t)(((uint16_t)q * 3u) >> 8);
                y[out++] = (float)(xi - 1) * d;
            }
        }
    }
    for (unsigned n = 0; n < 4; ++n) {
        for (unsigned h = 0; h < 2; ++h) {
            const uint8_t q = (uint8_t)(x->qh[h] * (uint8_t)powers[n]);
            const int16_t xi = (int16_t)(((uint16_t)q * 3u) >> 8);
            y[out++] = (float)(xi - 1) * d;
        }
    }
}

static uint16_t fp32_to_fp16_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;
        }
        mant |= 0x800000u;
        uint32_t t = mant >> (1 - exp + 13);
        return (uint16_t)(sign | t);
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static void fill_block(geniex_ptq1_block * b, unsigned seed, float scale) {
    for (unsigned i = 0; i < 24; ++i) {
        b->qs[i] = (uint8_t)((seed * 17u + i * 3u) % 243u);
    }
    for (unsigned i = 0; i < 2; ++i) {
        b->qh[i] = (uint8_t)((seed * 11u + i * 5u) % 243u);
    }
    b->d = fp32_to_fp16_bits(scale);
}

static geniex_ptq1_tile * pack_matrix(const geniex_ptq1_block * src, uint32_t nrows, uint32_t n_k_tiles) {
    const uint32_t n_col_tiles = (nrows + GENIEX_PTQ1_TILE_ROWS - 1) / GENIEX_PTQ1_TILE_ROWS;
    geniex_ptq1_tile * dst = (geniex_ptq1_tile *)calloc(n_col_tiles * n_k_tiles, sizeof(geniex_ptq1_tile));
    expect(dst != NULL, "alloc tiles");
    for (uint32_t ct = 0; ct < n_col_tiles; ++ct) {
        const unsigned valid = (unsigned)((nrows - ct * GENIEX_PTQ1_TILE_ROWS) < GENIEX_PTQ1_TILE_ROWS
                                              ? (nrows - ct * GENIEX_PTQ1_TILE_ROWS)
                                              : GENIEX_PTQ1_TILE_ROWS);
        for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
            geniex_ptq1_pack_tile(&dst[ct * n_k_tiles + kt],
                                  src + ct * GENIEX_PTQ1_TILE_ROWS * n_k_tiles,
                                  n_k_tiles, kt, valid);
        }
    }
    return dst;
}

static int floats_close(const float * a, const float * b, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        if (a[i] != b[i] && !(isnan(a[i]) && isnan(b[i]))) {
            if (fabsf(a[i] - b[i]) > 1e-5f * fmaxf(1.0f, fabsf(b[i]))) {
                printf("  mismatch at %u: got %g expected %g\n", i, a[i], b[i]);
                return 0;
            }
        }
    }
    return 1;
}

struct fixture_ctx {
    struct htp_context htp;
    uint8_t            wq_mem[64];
    work_queue_t       wq;
};

static void fixture_ctx_init(struct fixture_ctx * fc, uint32_t n_threads) {
    memset(fc, 0, sizeof(*fc));
    fc->wq = work_queue_init(fc->wq_mem, n_threads, 4, 0);
    fc->htp.worker_pool = fc->wq;
    fc->htp.n_threads   = n_threads;
}

static void set_ptr_field(struct htp_tensor * t, void * p) {
    t->data = (uint32_t)(uintptr_t)p;
}

static void fill_tensor_shape(struct htp_tensor * t, uint32_t type,
                              uint32_t ne0, uint32_t ne1, uint32_t ne2, uint32_t ne3,
                              uint32_t nb0, uint32_t nb1, uint32_t nb2, uint32_t nb3,
                              void * data, uint32_t size) {
    memset(t, 0, sizeof(*t));
    t->type = type;
    t->ne[0] = ne0;
    t->ne[1] = ne1;
    t->ne[2] = ne2;
    t->ne[3] = ne3;
    t->nb[0] = nb0;
    t->nb[1] = nb1;
    t->nb[2] = nb2;
    t->nb[3] = nb3;
    t->size  = size;
    set_ptr_field(t, data);
}

static int call_op_get_rows(struct fixture_ctx * fc,
                            struct htp_tensor * src0,
                            struct htp_tensor * src1,
                            struct htp_tensor * dst,
                            int32_t logical_nrows,
                            uint32_t n_threads) {
    struct htp_ops_context octx;
    memset(&octx, 0, sizeof(octx));
    octx.ctx       = &fc->htp;
    octx.op        = HTP_OP_GET_ROWS;
    octx.src[0]    = src0;
    octx.src[1]    = src1;
    octx.dst       = dst;
    octx.n_threads = n_threads;
    struct htp_get_rows_kernel_params * kp =
        (struct htp_get_rows_kernel_params *)octx.kernel_params;
    kp->logical_nrows = logical_nrows;
    return op_get_rows(&octx);
}

/* Valid gather: logicalN=33 padded to 64, indices 0/31/32/repeated, pad+hash. */
static void test_ptq1_valid(uint32_t k, int use_i32) {
    const uint32_t logical_n = 33;
    const uint32_t padded_n  = 64;
    const uint32_t n_k_tiles = k / GENIEX_PTQ1_BLOCK_K;
    const uint32_t n_threads = 4;

    geniex_ptq1_block * blocks =
        (geniex_ptq1_block *)calloc((size_t)logical_n * n_k_tiles, sizeof(*blocks));
    expect(blocks != NULL, "alloc blocks");
    for (uint32_t r = 0; r < logical_n; ++r) {
        for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
            float scale = (r == 0) ? 0.0f : ((r & 1u) ? -0.25f : (0.5f + 0.01f * (float)r));
            fill_block(&blocks[r * n_k_tiles + kt], r * 1000u + kt, scale);
        }
    }

    geniex_ptq1_tile * tiles = pack_matrix(blocks, logical_n, n_k_tiles);
    const uint32_t n_col_tiles = (logical_n + GENIEX_PTQ1_TILE_ROWS - 1) / GENIEX_PTQ1_TILE_ROWS;
    const size_t tiles_bytes = (size_t)n_col_tiles * n_k_tiles * sizeof(geniex_ptq1_tile);
    const uint32_t hash_before = fnv1a(tiles, tiles_bytes);

    int32_t idx32[] = {0, 31, 32, 0};
    int64_t idx64[] = {0, 31, 32, 0};
    const uint32_t n_idx = 4;
    void * idx_buf = use_i32 ? (void *)idx32 : (void *)idx64;
    const uint32_t idx_nb0 = use_i32 ? sizeof(int32_t) : sizeof(int64_t);
    const uint32_t idx_type = use_i32 ? HTP_TYPE_I32 : HTP_TYPE_I64;

    const size_t dst_nb1 = (size_t)(k + 4) * sizeof(float);
    float * dst_buf = (float *)malloc((size_t)n_idx * dst_nb1);
    float * oracle  = (float *)malloc((size_t)n_idx * dst_nb1);
    expect(dst_buf && oracle, "alloc dst");

    for (uint32_t i = 0; i < n_idx; ++i) {
        float * row = (float *)((uint8_t *)dst_buf + i * dst_nb1);
        float * oro = (float *)((uint8_t *)oracle + i * dst_nb1);
        for (uint32_t j = 0; j < k + 4; ++j) {
            row[j] = 1234.5f;
            oro[j] = 1234.5f;
        }
        const int64_t i01 = use_i32 ? (int64_t)idx32[i] : idx64[i];
        for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
            oracle_dequant_block(oro + kt * GENIEX_PTQ1_BLOCK_K,
                                 &blocks[i01 * n_k_tiles + kt]);
        }
    }

    struct htp_tensor src0, src1, dst;
    fill_tensor_shape(&src0, HTP_TYPE_PTQ1_0, k, padded_n, 1, 1,
                      /*nb0*/sizeof(geniex_ptq1_block),
                      /*nb1*/(uint32_t)(n_k_tiles * sizeof(geniex_ptq1_tile)),
                      0, 0, tiles, (uint32_t)tiles_bytes);
    /* Repacked layout: nb1 is not row-major GGUF; DSP uses tile indexing via K. */
    src0.nb[1] = (uint32_t)((size_t)n_k_tiles * sizeof(geniex_ptq1_tile));

    fill_tensor_shape(&src1, idx_type, n_idx, 1, 1, 1,
                      idx_nb0, idx_nb0 * n_idx, idx_nb0 * n_idx, idx_nb0 * n_idx,
                      idx_buf, idx_nb0 * n_idx);
    fill_tensor_shape(&dst, HTP_TYPE_F32, k, n_idx, 1, 1,
                      sizeof(float), (uint32_t)dst_nb1,
                      (uint32_t)dst_nb1 * n_idx, (uint32_t)dst_nb1 * n_idx,
                      dst_buf, (uint32_t)(n_idx * dst_nb1));

    struct fixture_ctx fc;
    fixture_ctx_init(&fc, n_threads);
    const int rc = call_op_get_rows(&fc, &src0, &src1, &dst, (int32_t)logical_n, n_threads);
    expect(rc == HTP_STATUS_OK, use_i32 ? "PTQ1 I32 valid OK" : "PTQ1 I64 valid OK");

    for (uint32_t i = 0; i < n_idx; ++i) {
        float * got = (float *)((uint8_t *)dst_buf + i * dst_nb1);
        float * exp = (float *)((uint8_t *)oracle + i * dst_nb1);
        char msg[128];
        snprintf(msg, sizeof(msg), "%s K=%u valid idx_slot=%u (incl pad)",
                 use_i32 ? "I32" : "I64", k, i);
        expect(floats_close(got, exp, k + 4), msg);
    }

    const uint32_t hash_after = fnv1a(tiles, tiles_bytes);
    expect(hash_before == hash_after, use_i32 ? "I32 input hash preserved" : "I64 input hash preserved");

    free(oracle);
    free(dst_buf);
    free(tiles);
    free(blocks);
}

/* Invalid indices must return INVAL_PARAMS with zero output writes. */
static void test_ptq1_reject_index(uint32_t k, int64_t bad_index, int use_i32, const char * tag) {
    const uint32_t logical_n = 33;
    const uint32_t padded_n  = 64;
    const uint32_t n_k_tiles = k / GENIEX_PTQ1_BLOCK_K;
    const uint32_t n_threads = 4;

    geniex_ptq1_block * blocks =
        (geniex_ptq1_block *)calloc((size_t)logical_n * n_k_tiles, sizeof(*blocks));
    expect(blocks != NULL, "alloc blocks reject");
    for (uint32_t r = 0; r < logical_n; ++r) {
        for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
            fill_block(&blocks[r * n_k_tiles + kt], r + kt, 0.5f);
        }
    }
    geniex_ptq1_tile * tiles = pack_matrix(blocks, logical_n, n_k_tiles);
    const uint32_t n_col_tiles = (logical_n + GENIEX_PTQ1_TILE_ROWS - 1) / GENIEX_PTQ1_TILE_ROWS;
    const size_t tiles_bytes = (size_t)n_col_tiles * n_k_tiles * sizeof(geniex_ptq1_tile);
    const uint32_t hash_before = fnv1a(tiles, tiles_bytes);

    int32_t idx32[2] = {0, (int32_t)bad_index};
    int64_t idx64[2] = {0, bad_index};
    const uint32_t n_idx = 2;
    void * idx_buf = use_i32 ? (void *)idx32 : (void *)idx64;
    const uint32_t idx_nb0 = use_i32 ? sizeof(int32_t) : sizeof(int64_t);
    const uint32_t idx_type = use_i32 ? HTP_TYPE_I32 : HTP_TYPE_I64;

    const size_t dst_nb1 = (size_t)(k + 4) * sizeof(float);
    float * dst_buf = (float *)malloc((size_t)n_idx * dst_nb1);
    expect(dst_buf != NULL, "alloc dst reject");
    for (uint32_t i = 0; i < n_idx; ++i) {
        float * row = (float *)((uint8_t *)dst_buf + i * dst_nb1);
        for (uint32_t j = 0; j < k + 4; ++j) {
            row[j] = 777.0f;
        }
    }
    const uint32_t dst_hash_before = fnv1a(dst_buf, n_idx * dst_nb1);

    struct htp_tensor src0, src1, dst;
    fill_tensor_shape(&src0, HTP_TYPE_PTQ1_0, k, padded_n, 1, 1,
                      sizeof(geniex_ptq1_block),
                      (uint32_t)(n_k_tiles * sizeof(geniex_ptq1_tile)),
                      0, 0, tiles, (uint32_t)tiles_bytes);
    fill_tensor_shape(&src1, idx_type, n_idx, 1, 1, 1,
                      idx_nb0, idx_nb0 * n_idx, idx_nb0 * n_idx, idx_nb0 * n_idx,
                      idx_buf, idx_nb0 * n_idx);
    fill_tensor_shape(&dst, HTP_TYPE_F32, k, n_idx, 1, 1,
                      sizeof(float), (uint32_t)dst_nb1,
                      (uint32_t)dst_nb1 * n_idx, (uint32_t)dst_nb1 * n_idx,
                      dst_buf, (uint32_t)(n_idx * dst_nb1));

    struct fixture_ctx fc;
    fixture_ctx_init(&fc, n_threads);
    const int rc = call_op_get_rows(&fc, &src0, &src1, &dst, (int32_t)logical_n, n_threads);
    expect(rc == HTP_STATUS_INVAL_PARAMS, tag);

    const uint32_t dst_hash_after = fnv1a(dst_buf, n_idx * dst_nb1);
    expect(dst_hash_before == dst_hash_after, "reject: no output write");
    expect(hash_before == fnv1a(tiles, tiles_bytes), "reject: input hash preserved");

    free(dst_buf);
    free(tiles);
    free(blocks);
}

static void test_shape_and_metadata(uint32_t k) {
    const uint32_t logical_n = 33;
    const uint32_t padded_n  = 64;
    const uint32_t n_k_tiles = k / GENIEX_PTQ1_BLOCK_K;
    const uint32_t n_threads = 4;

    geniex_ptq1_block * blocks =
        (geniex_ptq1_block *)calloc((size_t)logical_n * n_k_tiles, sizeof(*blocks));
    expect(blocks != NULL, "alloc meta blocks");
    for (uint32_t r = 0; r < logical_n; ++r) {
        fill_block(&blocks[r * n_k_tiles], r, 0.25f);
    }
    geniex_ptq1_tile * tiles = pack_matrix(blocks, logical_n, n_k_tiles);
    const uint32_t n_col_tiles = (logical_n + GENIEX_PTQ1_TILE_ROWS - 1) / GENIEX_PTQ1_TILE_ROWS;
    const size_t tiles_bytes = (size_t)n_col_tiles * n_k_tiles * sizeof(geniex_ptq1_tile);

    int32_t idx32[] = {0};
    float dst_buf[128 + 4];
    for (uint32_t j = 0; j < 128 + 4; ++j) {
        dst_buf[j] = 1.0f;
    }

    struct fixture_ctx fc;
    fixture_ctx_init(&fc, n_threads);

    struct htp_tensor src0, src1, dst;

    /* logical_nrows <= 0 */
    fill_tensor_shape(&src0, HTP_TYPE_PTQ1_0, k, padded_n, 1, 1,
                      sizeof(geniex_ptq1_block),
                      (uint32_t)(n_k_tiles * sizeof(geniex_ptq1_tile)),
                      0, 0, tiles, (uint32_t)tiles_bytes);
    fill_tensor_shape(&src1, HTP_TYPE_I32, 1, 1, 1, 1,
                      sizeof(int32_t), sizeof(int32_t), sizeof(int32_t), sizeof(int32_t),
                      idx32, sizeof(idx32));
    fill_tensor_shape(&dst, HTP_TYPE_F32, k, 1, 1, 1,
                      sizeof(float), (uint32_t)((k + 4) * sizeof(float)),
                      (uint32_t)((k + 4) * sizeof(float)), (uint32_t)((k + 4) * sizeof(float)),
                      dst_buf, (uint32_t)((k + 4) * sizeof(float)));
    expect(call_op_get_rows(&fc, &src0, &src1, &dst, 0, n_threads) == HTP_STATUS_INVAL_PARAMS,
           "logical_nrows=0 -> INVAL_PARAMS");
    expect(call_op_get_rows(&fc, &src0, &src1, &dst, -1, n_threads) == HTP_STATUS_INVAL_PARAMS,
           "logical_nrows<0 -> INVAL_PARAMS");
    expect(call_op_get_rows(&fc, &src0, &src1, &dst, (int32_t)(padded_n + 1), n_threads) ==
               HTP_STATUS_INVAL_PARAMS,
           "logical_nrows>ne01 -> INVAL_PARAMS");

    /* zero-size K */
    fill_tensor_shape(&src0, HTP_TYPE_PTQ1_0, 0, padded_n, 1, 1,
                      sizeof(geniex_ptq1_block),
                      (uint32_t)(n_k_tiles * sizeof(geniex_ptq1_tile)),
                      0, 0, tiles, (uint32_t)tiles_bytes);
    fill_tensor_shape(&dst, HTP_TYPE_F32, 0, 1, 1, 1,
                      sizeof(float), sizeof(float), sizeof(float), sizeof(float),
                      dst_buf, sizeof(float));
    expect(call_op_get_rows(&fc, &src0, &src1, &dst, (int32_t)logical_n, n_threads) ==
               HTP_STATUS_NO_SUPPORT,
           "ne00=0 -> NO_SUPPORT");

    /* bad index stride */
    fill_tensor_shape(&src0, HTP_TYPE_PTQ1_0, k, padded_n, 1, 1,
                      sizeof(geniex_ptq1_block),
                      (uint32_t)(n_k_tiles * sizeof(geniex_ptq1_tile)),
                      0, 0, tiles, (uint32_t)tiles_bytes);
    fill_tensor_shape(&src1, HTP_TYPE_I32, 1, 1, 1, 1,
                      8 /* bad */, 8, 8, 8, idx32, 8);
    fill_tensor_shape(&dst, HTP_TYPE_F32, k, 1, 1, 1,
                      sizeof(float), (uint32_t)((k + 4) * sizeof(float)),
                      (uint32_t)((k + 4) * sizeof(float)), (uint32_t)((k + 4) * sizeof(float)),
                      dst_buf, (uint32_t)((k + 4) * sizeof(float)));
    expect(call_op_get_rows(&fc, &src0, &src1, &dst, (int32_t)logical_n, n_threads) ==
               HTP_STATUS_NO_SUPPORT,
           "bad idx stride -> NO_SUPPORT");

    /* batched indices (ne11 != 1) */
    fill_tensor_shape(&src1, HTP_TYPE_I32, 1, 2, 1, 1,
                      sizeof(int32_t), sizeof(int32_t), 2 * sizeof(int32_t), 2 * sizeof(int32_t),
                      idx32, 2 * sizeof(int32_t));
    expect(call_op_get_rows(&fc, &src0, &src1, &dst, (int32_t)logical_n, n_threads) ==
               HTP_STATUS_NO_SUPPORT,
           "batched ne11!=1 -> NO_SUPPORT");

    /* zero index count */
    fill_tensor_shape(&src1, HTP_TYPE_I32, 0, 1, 1, 1,
                      sizeof(int32_t), 0, 0, 0, idx32, 0);
    fill_tensor_shape(&dst, HTP_TYPE_F32, k, 0, 1, 1,
                      sizeof(float), (uint32_t)((k + 4) * sizeof(float)),
                      0, 0, dst_buf, 0);
    expect(call_op_get_rows(&fc, &src0, &src1, &dst, (int32_t)logical_n, n_threads) ==
               HTP_STATUS_NO_SUPPORT,
           "nr=0 -> NO_SUPPORT");

    free(tiles);
    free(blocks);
}

/* F32 GET_ROWS HVX path regression (ne00 < 2048 avoids DMA). */
static void test_f32_get_rows(void) {
    const uint32_t k = 64;
    const uint32_t nrows = 8;
    const uint32_t n_idx = 3;
    const uint32_t n_threads = 4;
    int32_t idx32[] = {0, 3, 7};

    float * src = (float *)malloc((size_t)nrows * k * sizeof(float));
    float * dst = (float *)malloc((size_t)n_idx * (k + 2) * sizeof(float));
    expect(src && dst, "alloc f32");
    for (uint32_t r = 0; r < nrows; ++r) {
        for (uint32_t c = 0; c < k; ++c) {
            src[r * k + c] = (float)(r * 1000 + c);
        }
    }
    const size_t dst_nb1 = (size_t)(k + 2) * sizeof(float);
    for (uint32_t i = 0; i < n_idx; ++i) {
        float * row = (float *)((uint8_t *)dst + i * dst_nb1);
        for (uint32_t j = 0; j < k + 2; ++j) {
            row[j] = -1.0f;
        }
    }

    struct htp_tensor src0, src1, dst_t;
    fill_tensor_shape(&src0, HTP_TYPE_F32, k, nrows, 1, 1,
                      sizeof(float), (uint32_t)(k * sizeof(float)),
                      (uint32_t)(nrows * k * sizeof(float)),
                      (uint32_t)(nrows * k * sizeof(float)),
                      src, (uint32_t)(nrows * k * sizeof(float)));
    fill_tensor_shape(&src1, HTP_TYPE_I32, n_idx, 1, 1, 1,
                      sizeof(int32_t), sizeof(int32_t) * n_idx,
                      sizeof(int32_t) * n_idx, sizeof(int32_t) * n_idx,
                      idx32, sizeof(idx32));
    fill_tensor_shape(&dst_t, HTP_TYPE_F32, k, n_idx, 1, 1,
                      sizeof(float), (uint32_t)dst_nb1,
                      (uint32_t)dst_nb1 * n_idx, (uint32_t)dst_nb1 * n_idx,
                      dst, (uint32_t)(n_idx * dst_nb1));

    struct fixture_ctx fc;
    fixture_ctx_init(&fc, n_threads);
    const int rc = call_op_get_rows(&fc, &src0, &src1, &dst_t, 0, n_threads);
    expect(rc == HTP_STATUS_OK, "F32 GET_ROWS OK");

    for (uint32_t i = 0; i < n_idx; ++i) {
        float * row = (float *)((uint8_t *)dst + i * dst_nb1);
        for (uint32_t c = 0; c < k; ++c) {
            char msg[96];
            snprintf(msg, sizeof(msg), "F32 row %u col %u", i, c);
            expect(row[c] == src[idx32[i] * k + c], msg);
        }
        expect(row[k] == -1.0f && row[k + 1] == -1.0f, "F32 dst pad preserved");
    }

    free(dst);
    free(src);
}

int main(void) {
    const uint32_t ks[] = {128u, 256u, 5120u};
    for (unsigned i = 0; i < 3; ++i) {
        test_ptq1_valid(ks[i], 1);
        test_ptq1_valid(ks[i], 0);
        test_ptq1_reject_index(ks[i], -1, 1, "I32 negative -> INVAL_PARAMS before write");
        test_ptq1_reject_index(ks[i], 33, 1, "I32 logicalN=33 -> INVAL_PARAMS before write");
        test_ptq1_reject_index(ks[i], 63, 1, "I32 padded-tail=63 -> INVAL_PARAMS before write");
        test_ptq1_reject_index(ks[i], -1, 0, "I64 negative -> INVAL_PARAMS before write");
        test_ptq1_reject_index(ks[i], 33, 0, "I64 logicalN=33 -> INVAL_PARAMS before write");
        test_ptq1_reject_index(ks[i], 63, 0, "I64 padded-tail=63 -> INVAL_PARAMS before write");
    }
    test_shape_and_metadata(128);
    test_f32_get_rows();

    printf("v75 GET_ROWS actual-worker fixture passed "
           "(op_get_rows + real get-rows-ops.c, I32/I64, K128/256/5120, "
           "logicalN33 padded64, idx 0/31/32/repeated, reject-before-write, "
           "pad+input hash, shape/metadata, F32 HVX regression)\n");
    return 0;
}
