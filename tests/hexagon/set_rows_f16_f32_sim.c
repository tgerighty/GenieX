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
static uint64_t output_hash = UINT64_C(1469598103934665603);

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, void * data, unsigned int n) {
    (void)q;
    for (unsigned int i = 0; i < n; ++i) func(n, i, data);
    return true;
}

uint64 HAP_perf_qtimer_count_to_us(uint64 count) {
    return count;
}

static float input_value(uint32_t row, uint32_t element) {
    switch (element % 8) {
        case 0: return 0.0f;
        case 1: return -0.0f;
        case 2: return 1.00048828125f;
        case 3: return 1.0004884f;
        case 4: return -1.00048828125f;
        case 5: return -1.0004884f;
        default: return (float)((int)((row * 37 + element * 13) % 2047) - 1023) / 37.0f;
    }
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
            row[j] = input_value(r, j);
        }
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
        const uint16_t * out = (const uint16_t *)(dst + out_row * dst_row);
        for (uint32_t j = 0; j < width; ++j) {
            const float expected = input_value(r, j);
            _Float16 actual_half;
            memcpy(&actual_half, &out[j], sizeof(actual_half));
            const float actual = (float)actual_half;
            const float tolerance = 0.001f * (expected > 1.0f || expected < -1.0f ?
                                               (expected > 0.0f ? expected : -expected) : 1.0f);
            if (actual - expected > tolerance || expected - actual > tolerance) {
                printf("SET_ROWS mismatch row=%u element=%u input=%g got=0x%04x expected=0x%04x\n",
                       (unsigned)r, (unsigned)j, expected, (unsigned)out[j], (unsigned)(_Float16)expected);
                return 2;
            }
            output_hash ^= out[j];
            output_hash *= UINT64_C(1099511628211);
            uint32_t expected_input_bits, actual_input_bits;
            const float expected_input = input_value(r, j);
            memcpy(&expected_input_bits, &expected_input, sizeof(expected_input_bits));
            memcpy(&actual_input_bits, src + r * src_row + j * sizeof(float), sizeof(actual_input_bits));
            if (actual_input_bits != expected_input_bits) return 6;
        }
    }

    const uintptr_t dst_base = (uintptr_t)dst_storage;
    for (uintptr_t pos = 0; pos < sizeof(dst_storage); ++pos) {
        const uintptr_t addr = dst_base + pos;
        bool is_output = false;
        for (uint32_t r = 0; r < ROWS; ++r) {
            const uint32_t out_row = use_i64 ? (uint32_t)idx64[r] : (uint32_t)idx32[r];
            const uintptr_t begin = (uintptr_t)(dst + out_row * dst_row);
            if (addr >= begin && addr < begin + width * sizeof(_Float16)) {
                is_output = true;
                break;
            }
        }
        if (!is_output && dst_storage[pos] != 0x5A) return 4;
    }
    for (uint32_t j = 0; j < sizeof(src_storage); ++j) {
        bool is_input = false;
        for (uint32_t r = 0; r < ROWS; ++r) {
            const uintptr_t begin = (uintptr_t)(src + r * src_row);
            const uintptr_t addr = (uintptr_t)(src_storage + j);
            if (addr >= begin && addr < begin + width * sizeof(float)) {
                is_input = true;
                break;
            }
        }
        if (!is_input && src_storage[j] != 0xA5) return 3;
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
    printf("SET_ROWS output FNV64=0x%016llx\n", (unsigned long long)output_hash);
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
    printf("SET_ROWS output FNV64=0x%016llx\n", (unsigned long long)output_hash);
#endif
    return 0;
}
