// Exercise Prism's production SET_ROWS worker through op_set_rows().
#include <stdio.h>
#include "set-rows-ops.c"

#ifndef P13_ROWS
#define P13_ROWS 3
#endif

enum { MAX_WIDTH = 11008, ROWS = P13_ROWS, DST_ROWS = 4, GUARD = 256 };

static _Alignas(128) uint8_t src_storage[ROWS * (MAX_WIDTH * sizeof(float) + 256) + GUARD * 2];
static _Alignas(128) uint8_t dst_storage[DST_ROWS * (MAX_WIDTH * sizeof(_Float16) + 256) + GUARD * 2];
static _Alignas(128) int32_t idx32[ROWS];
static _Alignas(128) int64_t idx64[ROWS];

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, void * data, unsigned int n) {
    (void)q;
    for (unsigned int i = 0; i < n; ++i) func(n, i, data);
    return true;
}

uint64 HAP_perf_qtimer_count_to_us(uint64 count) {
    return count;
}

static int run_case(uint32_t width, uint32_t src_offset, uint32_t dst_offset, bool use_i64,
                    uint32_t repeats, uint32_t src_pad, uint32_t dst_pad) {
    const uint32_t src_row = width * sizeof(float) + src_pad;
    const uint32_t dst_row = width * sizeof(_Float16) + dst_pad;
    uint8_t * src = src_storage + GUARD + src_offset;
    uint8_t * dst = dst_storage + GUARD + dst_offset;
    memset(src_storage, 0xA5, sizeof(src_storage));
    memset(dst_storage, 0x5A, sizeof(dst_storage));

    for (uint32_t r = 0; r < ROWS; ++r) {
        const uint32_t dest_row = r == 0 ? 2 : r == 1 ? 0 : 3;
        idx32[r] = (int32_t)dest_row;
        idx64[r] = (int64_t)dest_row;
        float * row = (float *)(src + r * src_row);
        for (uint32_t j = 0; j < width; ++j) {
            row[j] = (float)((int)((r * 37 + j * 13) % 2047) - 1023) / 37.0f;
        }
    }

    for (uint32_t j = 0; j < width * sizeof(_Float16); ++j) {
        if (dst[dst_row + j] != 0x5A) return 5;
    }

    struct htp_tensor src_tensor = {0}, index_tensor = {0}, dst_tensor = {0};
    src_tensor.data = (uint32_t)(uintptr_t)src;
    src_tensor.type = HTP_TYPE_F32;
    src_tensor.ne[0] = width; src_tensor.ne[1] = ROWS; src_tensor.ne[2] = 1; src_tensor.ne[3] = 1;
    src_tensor.nb[1] = src_row; src_tensor.nb[2] = ROWS * src_row; src_tensor.nb[3] = ROWS * src_row;

    index_tensor.data = (uint32_t)(uintptr_t)(use_i64 ? (void *)idx64 : (void *)idx32);
    index_tensor.type = use_i64 ? HTP_TYPE_I64 : HTP_TYPE_I32;
    index_tensor.ne[0] = ROWS; index_tensor.ne[1] = 1; index_tensor.ne[2] = 1; index_tensor.ne[3] = 1;
    index_tensor.nb[0] = use_i64 ? sizeof(int64_t) : sizeof(int32_t);

    dst_tensor.data = (uint32_t)(uintptr_t)dst;
    dst_tensor.type = HTP_TYPE_F16;
    dst_tensor.ne[0] = width; dst_tensor.ne[1] = DST_ROWS; dst_tensor.ne[2] = 1; dst_tensor.ne[3] = 1;
    dst_tensor.nb[1] = dst_row; dst_tensor.nb[2] = DST_ROWS * dst_row; dst_tensor.nb[3] = DST_ROWS * dst_row;

    struct htp_context context = {0};
    struct htp_ops_context op = {0};
    op.ctx = &context;
    op.src[0] = &src_tensor; op.src[1] = &index_tensor; op.dst = &dst_tensor;
    op.n_threads = 1;
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        if (op_set_rows(&op) != HTP_STATUS_OK) return 1;
    }

    for (uint32_t r = 0; r < ROWS; ++r) {
        const uint32_t out_row = use_i64 ? (uint32_t)idx64[r] : (uint32_t)idx32[r];
        const float * in = (const float *)(src + r * src_row);
        const _Float16 * out = (const _Float16 *)(dst + out_row * dst_row);
        for (uint32_t j = 0; j < width; ++j) {
            if (out[j] != (_Float16)in[j]) return 2;
        }
    }

    for (uint32_t j = 0; j < GUARD; ++j) {
        if (src_storage[j] != 0xA5 || src_storage[sizeof(src_storage) - 1 - j] != 0xA5) return 3;
        if (dst_storage[j] != 0x5A || dst_storage[sizeof(dst_storage) - 1 - j] != 0x5A) return 4;
    }
    return 0;
}

int main(void) {
#ifdef P13_BENCH_WIDTH
    const int rc = run_case(P13_BENCH_WIDTH, 0, 0, false, P13_BENCH_REPEATS,
                            P13_BENCH_PAD, P13_BENCH_PAD);
    if (rc) return rc;
    printf("SET_ROWS bench width=%u rows=%u repeats=%u aligned output and guards passed\n",
           (unsigned)P13_BENCH_WIDTH, (unsigned)P13_ROWS, (unsigned)P13_BENCH_REPEATS);
#else
    const uint32_t widths[] = { 1024, 1279, 1280, 5120, 11008 };
    const uint32_t offsets[][2] = { {0, 0}, {4, 0}, {0, 2}, {4, 2} };
    for (uint32_t w = 0; w < sizeof(widths) / sizeof(widths[0]); ++w) {
        for (uint32_t a = 0; a < sizeof(offsets) / sizeof(offsets[0]); ++a) {
            for (uint32_t i64 = 0; i64 < 2; ++i64) {
                const int rc = run_case(widths[w], offsets[a][0], offsets[a][1], i64 != 0, 1, 128, 128);
                if (rc) {
                    printf("SET_ROWS failed width=%u src_offset=%u dst_offset=%u index=%u rc=%d\n",
                           (unsigned)widths[w], (unsigned)offsets[a][0], (unsigned)offsets[a][1], i64 ? 64 : 32, rc);
                    return rc;
                }
            }
        }
    }
    for (uint32_t a = 0; a < sizeof(offsets) / sizeof(offsets[0]); ++a) {
        for (uint32_t i64 = 0; i64 < 2; ++i64) {
            const int rc = run_case(1, offsets[a][0], offsets[a][1], i64 != 0, 1, 0, 0);
            if (rc) {
                printf("SET_ROWS transposed width=1 failed src_offset=%u dst_offset=%u index=%u rc=%d\n",
                       (unsigned)offsets[a][0], (unsigned)offsets[a][1], i64 ? 64 : 32, rc);
                return rc;
            }
        }
    }
    puts("SET_ROWS F32-to-F16 output and guards passed");
#endif
    return 0;
}
