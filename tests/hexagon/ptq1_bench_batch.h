/* SPDX-License-Identifier: BSD-3-Clause
 * Opt-in same-process PTQ1 batch timing (compile with -DPTQ1_BENCH_BATCH=1).
 * Times the fixture's real operator callback; does not mock or alter production ops.
 */
#pragma once

#ifndef PTQ1_BENCH_BATCH
#define PTQ1_BENCH_BATCH 0
#endif

#if PTQ1_BENCH_BATCH

#include <stdio.h>
#include <hexagon_sim_timer.h>

enum {
    PTQ1_BENCH_BATCH_WARMUP = 1,
    PTQ1_BENCH_BATCH_REPEAT1 = 1,
    PTQ1_BENCH_BATCH_REPEAT4 = 4,
    PTQ1_BENCH_BATCH_OP_CALLS =
        PTQ1_BENCH_BATCH_WARMUP + PTQ1_BENCH_BATCH_REPEAT1 + PTQ1_BENCH_BATCH_REPEAT4
};

typedef int (*ptq1_bench_batch_op_fn)(void *ctx);

/* Cycle reads stay outside the operator loop. op returns 0 on success. */
static int ptq1_bench_batch_measure(ptq1_bench_batch_op_fn op, void *ctx, unsigned repeats,
                                    unsigned long long *cycles_out) {
    unsigned long long t0;
    unsigned long long t1;
    unsigned i;

    t0 = hexagon_sim_read_pcycles();
    for (i = 0; i < repeats; ++i) {
        int st = op(ctx);
        if (st != 0) return st;
    }
    t1 = hexagon_sim_read_pcycles();
    *cycles_out = t1 - t0;
    return 0;
}

static void ptq1_bench_batch_report(unsigned long long repeat1_cycles,
                                    unsigned long long repeat4_cycles) {
    long long slope = ((long long)repeat4_cycles - (long long)repeat1_cycles) / 3ll;
    printf("PTQ1 batch repeat1_cycles=%llu repeat4_cycles=%llu\n",
           repeat1_cycles, repeat4_cycles);
    printf("PTQ1 batch call_slope=%lld ((repeat4-repeat1)/3); "
           "warn: hexagon_sim_read_pcycles reader overhead means this method needs calibration\n",
           slope);
}

#endif /* PTQ1_BENCH_BATCH */
