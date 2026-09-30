// SPDX-License-Identifier: BSD-3-Clause
// Actual-worker adapters for the v75 GET_ROWS fixture.
// Synchronous work_queue_run_async for N workers. Timing/logger only —
// MUST NOT replace PTQ1 dequant or op_get_rows dispatch.
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>

#include <HAP_perf.h>

#include "work-queue.h"

/* ---- work queue: sync stand-in for QuRT worker pool ---- */

struct work_queue_s {
    unsigned int n_threads;
};

size_t work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t stack_size) {
    (void)n_threads;
    (void)capacity;
    (void)stack_size;
    return sizeof(struct work_queue_s);
}

size_t work_queue_alignof(void) {
    return sizeof(void *);
}

work_queue_t work_queue_init(void * ptr, uint32_t n_threads, uint32_t capacity, uint32_t stack_size) {
    (void)capacity;
    (void)stack_size;
    struct work_queue_s * q = (struct work_queue_s *)ptr;
    if (q) {
        q->n_threads = n_threads ? n_threads : 1;
    }
    return q;
}

void work_queue_free(work_queue_t q) {
    (void)q;
}

void work_queue_wakeup(work_queue_t q) {
    (void)q;
}

void work_queue_suspend(work_queue_t q) {
    (void)q;
}

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, void * data, unsigned int n) {
    (void)q;
    if (!func || n == 0) {
        return false;
    }
    /* Fixture pattern: run ith=0..n-1 synchronously on the caller thread. */
    for (unsigned int i = 0; i < n; ++i) {
        func(n, i, data);
    }
    return true;
}

/* HAP_perf_get_qtimer_count is SDK static-inline (c31:30). External convert: */
uint64 HAP_perf_qtimer_count_to_us(uint64 count) {
    return count / 19ull;
}

/* HAP_debug / FARF logger adapters — no-op; must not affect compute. */
void HAP_debug(const char * message, int level, const char * file, int line) {
    (void)level;
    (void)file;
    (void)line;
    (void)message;
}

void HAP_debug_v2(int level, const char * file, int line, const char * format, ...) {
    (void)level;
    (void)file;
    (void)line;
    (void)format;
}

#ifdef V75_GET_ROWS_NEED_FUTEX_STUB
int qurt_futex_wait(void * addr, int val) {
    (void)addr;
    (void)val;
    return 0;
}
int qurt_futex_wake(void * addr, int n_wake) {
    (void)addr;
    (void)n_wake;
    return 0;
}
#endif
