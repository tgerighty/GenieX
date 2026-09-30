/*
 * Hexagon v75 worker fixture for HTP BF16 x F32 DDR matmul (zero VTCM).
 *
 * Compiles against real hexagon/htp/matmul-ops.c and links dma-queue.c with
 * --gc-sections. Calls op_matmul with HTP_MM_KERNEL_HVX_BF16_F32_DDR.
 *
 * Four synchronous worker adapter callbacks (no RTOS pool / no HMX):
 *   1) work_queue_run       — serial ith=0..n-1 on the calling thread
 *   2) qurt_futex_wake      — abort on unexpected HMX/async futex use
 *   3) HAP_debug_v2         — FARF/logging adapter
 *   4) HAP_debug            — logging adapter
 *
 * Built only when HEXAGON_SDK_ROOT is present (see run_bf16_matmul_tests.sh).
 * Not a native host binary — DSP pointer width / HVX headers required.
 */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "htp-ctx.h"
#include "htp-ops.h"
#include "matmul-ops.h"
#include "work-queue.h"

/* ---- Synchronous worker adapters (required evidence surface) ---- */

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, void * data, unsigned int n) {
    (void) q;
    if (n == 0 || !func) {
        return false;
    }
    for (unsigned int ith = 0; ith < n; ++ith) {
        func(n, ith, data);
    }
    return true;
}

/* Unused work-queue lifecycle stubs (not exercised by BF16 DDR path). */
size_t work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t stack_size) {
    (void) n_threads;
    (void) capacity;
    (void) stack_size;
    return 64;
}
size_t work_queue_alignof(void) { return 8; }
work_queue_t work_queue_init(void * ptr, uint32_t n_threads, uint32_t capacity, uint32_t stack_size) {
    (void) n_threads;
    (void) capacity;
    (void) stack_size;
    return (work_queue_t) ptr;
}
void work_queue_free(work_queue_t q) { (void) q; }
void work_queue_wakeup(work_queue_t q) { (void) q; }
void work_queue_suspend(work_queue_t q) { (void) q; }

/* Trap unexpected HMX / real work-queue futex use. */
int qurt_futex_wake(void * addr, int n_threads) {
    (void) addr;
    (void) n_threads;
    fprintf(stderr, "bf16_f32_matmul_v75_worker: unexpected qurt_futex_wake (HMX/async?)\n");
    abort();
    return 0;
}

void HAP_debug_v2(int level, const char * file, int line, const char * fmt, ...) {
    (void) level;
    (void) file;
    (void) line;
    va_list ap;
    va_start(ap, fmt);
    (void) vfprintf(stderr, fmt, ap);
    va_end(ap);
}

void HAP_debug(const char * msg, int level, const char * file, int line) {
    (void) level;
    (void) file;
    (void) line;
    fprintf(stderr, "%s", msg);
}

/* ---- Independent BF16 reference (uint16<<16, not F16/half) ---- */

static float ref_bf16_to_f32(uint16_t bits) {
    union {
        float    f;
        uint32_t u;
    } u;
    u.u = ((uint32_t) bits) << 16;
    return u.f;
}

static float ref_dot_row(const uint16_t * w, const float * a, uint32_t k) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < k; ++i) {
        sum += ref_bf16_to_f32(w[i]) * a[i];
    }
    return sum;
}

/*
 * Tolerance: production and reference share the same scalar reduction order
 * (left-to-right mul-add of BF16<<16 * F32). On Hexagon scalar FP they should
 * match bit-exact. Allow a tiny absolute eps for any FMA / eval-order variance
 * across toolchains: ~max(1e-5, 32*ulp(|exp|)). For K=5120 with O(1e-2)
 * products, accumulated sum is O(1..10); 1e-3 absolute covers ~K*eps*scale.
 */
static float tol_for(float exp_v, uint32_t k) {
    float a = fabsf(exp_v);
    float ulp_budget = (k >= 1024) ? 1e-3f : 1e-5f;
    float rel = (a > 1.0f) ? (a * 1e-5f) : 0.0f;
    return (ulp_budget > rel) ? ulp_budget : rel;
}

static int g_fails = 0;
static int g_checks = 0;

static void expect_status(const char * name, int got, int want) {
    g_checks++;
    if (got != want) {
        printf("FAIL %s status got=%d want=%d\n", name, got, want);
        g_fails++;
    }
}

static void expect_near(const char * name, float got, float exp, float tol) {
    g_checks++;
    float d = fabsf(got - exp);
    if (!(d <= tol) && !(isnan(got) && isnan(exp))) {
        printf("FAIL %s got=%a exp=%a |d|=%a tol=%a\n", name, got, exp, d, tol);
        g_fails++;
    }
}

static void expect_true(const char * name, int cond) {
    g_checks++;
    if (!cond) {
        printf("FAIL %s\n", name);
        g_fails++;
    }
}

static uint32_t ptr_u32(const void * p) {
    return (uint32_t) (uintptr_t) p;
}

static void fill_tensor(struct htp_tensor * t,
                        uint32_t type,
                        const uint32_t ne[4],
                        const uint32_t nb[4],
                        void * data,
                        uint32_t size) {
    memset(t, 0, sizeof(*t));
    t->type = type;
    t->data = ptr_u32(data);
    t->size = size;
    for (int i = 0; i < 4; ++i) {
        t->ne[i] = ne[i];
        t->nb[i] = nb[i];
    }
}

static void setup_ctx(struct htp_context * ctx, struct htp_ops_context * octx, uint32_t n_threads) {
    memset(ctx, 0, sizeof(*ctx));
    memset(octx, 0, sizeof(*octx));
    ctx->n_threads  = n_threads;
    ctx->vtcm_base  = NULL;
    ctx->vtcm_size  = 0; /* zero VTCM */
    ctx->work_queue = NULL;
    octx->ctx       = ctx;
    octx->n_threads = n_threads;
    octx->op        = HTP_OP_MUL_MAT;
}

static void setup_kparams(struct htp_ops_context * octx, int32_t n_threads) {
    struct htp_mm_kernel_params kp;
    memset(&kp, 0, sizeof(kp));
    kp.kernel_type    = HTP_MM_KERNEL_HVX_BF16_F32_DDR;
    kp.n_hmx          = 0;
    kp.n_threads      = n_threads;
    kp.vtcm_size      = 0;
    kp.vtcm_src0_size = 0;
    kp.vtcm_src1_size = 0;
    kp.vtcm_dst_size  = 0;
    kp.n_prefetch     = 0;
    memcpy(octx->kernel_params, &kp, sizeof(kp));
}

/* Run one BF16 matmul case with padded row strides; verify vs scalar ref. */
static void test_matmul_case(const char * tag,
                             uint32_t K,
                             uint32_t N,
                             uint32_t M,
                             uint32_t w_row_elems,
                             uint32_t a_row_elems,
                             uint32_t d_row_elems,
                             uint32_t n_threads) {
    expect_true("stride_pad_w", w_row_elems >= K);
    expect_true("stride_pad_a", a_row_elems >= K);
    expect_true("stride_pad_d", d_row_elems >= N);

    const size_t w_bytes = (size_t) N * w_row_elems * sizeof(uint16_t);
    const size_t a_bytes = (size_t) M * a_row_elems * sizeof(float);
    const size_t d_bytes = (size_t) M * d_row_elems * sizeof(float);

    uint16_t * w = (uint16_t *) malloc(w_bytes);
    float *    a = (float *) malloc(a_bytes);
    float *    d = (float *) malloc(d_bytes);
    uint16_t * w_snap = (uint16_t *) malloc(w_bytes);
    float *    a_snap = (float *) malloc(a_bytes);
    expect_true("alloc", w && a && d && w_snap && a_snap);
    if (!w || !a || !d || !w_snap || !a_snap) {
        free(w);
        free(a);
        free(d);
        free(w_snap);
        free(a_snap);
        return;
    }

    memset(w, 0xCD, w_bytes);
    memset(a, 0xAB, a_bytes);
    memset(d, 0x5A, d_bytes); /* padding sentinel */

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t k = 0; k < K; ++k) {
            uint16_t bits = (uint16_t) (0x3c00u + (n * 17u + k * 3u) % 0x200u);
            if ((k + n) & 1u) {
                bits |= 0x8000u;
            }
            w[n * w_row_elems + k] = bits;
        }
    }
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t k = 0; k < K; ++k) {
            a[m * a_row_elems + k] =
                ((k & 1u) ? -1.0f : 1.0f) * (1.0f / 64.0f) + (float) m * (1.0f / 256.0f);
        }
    }
    memcpy(w_snap, w, w_bytes);
    memcpy(a_snap, a, a_bytes);

    struct htp_tensor src0, src1, dst;
    uint32_t ne0[4] = { K, N, 1, 1 };
    uint32_t ne1[4] = { K, M, 1, 1 };
    uint32_t ned[4] = { N, M, 1, 1 };
    uint32_t nb0[4] = {
        sizeof(uint16_t),
        (uint32_t) (w_row_elems * sizeof(uint16_t)),
        (uint32_t) (N * w_row_elems * sizeof(uint16_t)),
        (uint32_t) (N * w_row_elems * sizeof(uint16_t)),
    };
    uint32_t nb1[4] = {
        sizeof(float),
        (uint32_t) (a_row_elems * sizeof(float)),
        (uint32_t) (M * a_row_elems * sizeof(float)),
        (uint32_t) (M * a_row_elems * sizeof(float)),
    };
    uint32_t nbd[4] = {
        sizeof(float),
        (uint32_t) (d_row_elems * sizeof(float)),
        (uint32_t) (M * d_row_elems * sizeof(float)),
        (uint32_t) (M * d_row_elems * sizeof(float)),
    };
    fill_tensor(&src0, HTP_TYPE_BF16, ne0, nb0, w, (uint32_t) w_bytes);
    fill_tensor(&src1, HTP_TYPE_F32, ne1, nb1, a, (uint32_t) a_bytes);
    fill_tensor(&dst, HTP_TYPE_F32, ned, nbd, d, (uint32_t) d_bytes);

    struct htp_tensor src0_desc = src0, src1_desc = src1, dst_desc = dst;

    struct htp_context      ctx;
    struct htp_ops_context  octx;
    setup_ctx(&ctx, &octx, n_threads);
    setup_kparams(&octx, (int32_t) n_threads);
    octx.src[0] = &src0;
    octx.src[1] = &src1;
    octx.src[2] = NULL;
    octx.dst    = &dst;

    int st = op_matmul(&octx);
    expect_status(tag, st, HTP_STATUS_OK);

    /* Descriptors unchanged (data ptr / shape / strides). */
    expect_true("desc_src0", memcmp(&src0, &src0_desc, sizeof(src0)) == 0);
    expect_true("desc_src1", memcmp(&src1, &src1_desc, sizeof(src1)) == 0);
    expect_true("desc_dst", memcmp(&dst, &dst_desc, sizeof(dst)) == 0);

    /* Inputs unchanged. */
    expect_true("in_w", memcmp(w, w_snap, w_bytes) == 0);
    expect_true("in_a", memcmp(a, a_snap, a_bytes) == 0);

    for (uint32_t m = 0; m < M; ++m) {
        float * row = d + (size_t) m * d_row_elems;
        for (uint32_t n = 0; n < N; ++n) {
            float exp = ref_dot_row(w + (size_t) n * w_row_elems, a + (size_t) m * a_row_elems, K);
            char  name[96];
            snprintf(name, sizeof(name), "%s m=%u n=%u", tag, m, n);
            expect_near(name, row[n], exp, tol_for(exp, K));
        }
        /* Output padding beyond N must stay sentinel 0x5A pattern. */
        for (uint32_t p = N; p < d_row_elems; ++p) {
            uint32_t bits;
            memcpy(&bits, &row[p], sizeof(bits));
            if (bits != 0x5A5A5A5Au) {
                printf("FAIL %s out_pad m=%u p=%u bits=0x%08x\n", tag, m, p, bits);
                g_fails++;
                g_checks++;
                break;
            }
            g_checks++;
        }
    }

    free(w);
    free(a);
    free(d);
    free(w_snap);
    free(a_snap);
}

static int call_reject(const char * name,
                       void (*mutate)(struct htp_tensor * s0,
                                      struct htp_tensor * s1,
                                      struct htp_tensor * dst,
                                      struct htp_ops_context * octx),
                       int want) {
    enum { K = 8, N = 4, M = 2 };
    uint16_t w[N * K];
    float    a[M * K];
    float    d[M * N];
    memset(w, 0, sizeof(w));
    memset(a, 0, sizeof(a));
    memset(d, 0, sizeof(d));

    struct htp_tensor src0, src1, dst;
    uint32_t ne0[4] = { K, N, 1, 1 };
    uint32_t ne1[4] = { K, M, 1, 1 };
    uint32_t ned[4] = { N, M, 1, 1 };
    uint32_t nb0[4] = { 2, 2 * K, 2 * K * N, 2 * K * N };
    uint32_t nb1[4] = { 4, 4 * K, 4 * K * M, 4 * K * M };
    uint32_t nbd[4] = { 4, 4 * N, 4 * N * M, 4 * N * M };
    fill_tensor(&src0, HTP_TYPE_BF16, ne0, nb0, w, sizeof(w));
    fill_tensor(&src1, HTP_TYPE_F32, ne1, nb1, a, sizeof(a));
    fill_tensor(&dst, HTP_TYPE_F32, ned, nbd, d, sizeof(d));

    struct htp_context     ctx;
    struct htp_ops_context octx;
    setup_ctx(&ctx, &octx, 2);
    setup_kparams(&octx, 2);
    octx.src[0] = &src0;
    octx.src[1] = &src1;
    octx.src[2] = NULL;
    octx.dst    = &dst;

    mutate(&src0, &src1, &dst, &octx);
    int st = op_matmul(&octx);
    expect_status(name, st, want);
    return st;
}

static void mut_wrong_dtype(struct htp_tensor * s0, struct htp_tensor * s1, struct htp_tensor * dst,
                            struct htp_ops_context * octx) {
    (void) s1;
    (void) dst;
    (void) octx;
    s0->type = HTP_TYPE_F16;
}

static void mut_batch(struct htp_tensor * s0, struct htp_tensor * s1, struct htp_tensor * dst,
                      struct htp_ops_context * octx) {
    (void) octx;
    s0->ne[2] = 2;
    s1->ne[2] = 2;
    dst->ne[2] = 2;
}

static void mut_under_stride(struct htp_tensor * s0, struct htp_tensor * s1, struct htp_tensor * dst,
                             struct htp_ops_context * octx) {
    (void) s1;
    (void) dst;
    (void) octx;
    s0->nb[1] = s0->nb[0] * (s0->ne[0] - 1); /* undersized row stride */
}

static void mut_misalign_stride(struct htp_tensor * s0, struct htp_tensor * s1, struct htp_tensor * dst,
                                struct htp_ops_context * octx) {
    (void) s1;
    (void) dst;
    (void) octx;
    s0->nb[1] = s0->nb[0] * s0->ne[0] + 1; /* odd: not multiple of sizeof(uint16_t) */
}

static void mut_zero_threads(struct htp_tensor * s0, struct htp_tensor * s1, struct htp_tensor * dst,
                             struct htp_ops_context * octx) {
    (void) s0;
    (void) s1;
    (void) dst;
    octx->n_threads = 0;
}

static void mut_bias(struct htp_tensor * s0, struct htp_tensor * s1, struct htp_tensor * dst,
                     struct htp_ops_context * octx) {
    (void) s0;
    (void) s1;
    static struct htp_tensor bias;
    memset(&bias, 0, sizeof(bias));
    bias.type  = HTP_TYPE_F32;
    bias.data  = dst->data;
    bias.ne[0] = dst->ne[0];
    bias.ne[1] = 1;
    bias.ne[2] = 1;
    bias.ne[3] = 1;
    bias.nb[0] = sizeof(float);
    bias.nb[1] = sizeof(float) * bias.ne[0];
    octx->src[2] = &bias;
}

static void mut_wrong_shape(struct htp_tensor * s0, struct htp_tensor * s1, struct htp_tensor * dst,
                            struct htp_ops_context * octx) {
    (void) s1;
    (void) octx;
    dst->ne[0] = s0->ne[1] + 1;
}

int main(void) {
    printf("bf16_f32_matmul_v75_worker: start\n");

    /* Primary shapes: K=5120 N=48, M=1 and M=25; padded row strides. */
    test_matmul_case("K5120_N48_M1", 5120, 48, 1,
                     /*w*/ 5120 + 8, /*a*/ 5120 + 16, /*d*/ 48 + 4, /*nth*/ 4);
    test_matmul_case("K5120_N48_M25", 5120, 48, 25,
                     5120 + 8, 5120 + 16, 48 + 4, 4);

    /* Small odd-K / tail-N with padded source, activation, output strides. */
    test_matmul_case("oddK7_tailN5_M3", 7, 5, 3,
                     /*w pad*/ 11, /*a pad*/ 13, /*d pad*/ 9, 2);
    test_matmul_case("oddK513_tailN7_M2", 513, 7, 2,
                     520, 528, 12, 3);

    /* Reject cases */
    call_reject("rej_wrong_dtype", mut_wrong_dtype, HTP_STATUS_NO_SUPPORT);
    call_reject("rej_batch", mut_batch, HTP_STATUS_NO_SUPPORT);
    call_reject("rej_under_stride", mut_under_stride, HTP_STATUS_INVAL_PARAMS);
    call_reject("rej_misalign_stride", mut_misalign_stride, HTP_STATUS_INVAL_PARAMS);
    call_reject("rej_zero_threads", mut_zero_threads, HTP_STATUS_INVAL_PARAMS);
    call_reject("rej_bias", mut_bias, HTP_STATUS_NO_SUPPORT);
    call_reject("rej_wrong_shape", mut_wrong_shape, HTP_STATUS_INVAL_PARAMS);

    if (g_fails) {
        printf("bf16_f32_matmul_v75_worker: %d FAIL / %d checks\n", g_fails, g_checks);
        return 1;
    }
    printf("bf16_f32_matmul_v75_worker: OK (%d checks)\n", g_checks);
    return 0;
}
