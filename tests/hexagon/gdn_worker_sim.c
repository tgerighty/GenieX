// SPDX-License-Identifier: BSD-3-Clause
// Exercise the production GDN dispatcher and worker with the real DMA queue.
#include "gated-delta-net-ops.c"

#include <stdio.h>

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, void * data, unsigned n) {
    (void) q; (void) func; (void) data; (void) n;
    assert(0);
    return false;
}

#ifndef GDN_S
#define GDN_S 128
#endif
#ifndef GDN_H
#define GDN_H 32
#endif
#ifndef GDN_T
#define GDN_T 1
#endif
#ifndef GDN_VECTOR_GATE
#define GDN_VECTOR_GATE 0
#endif
#ifndef GDN_REPEATS
#define GDN_REPEATS 1
#endif
#ifndef GDN_FAULT_OUTPUT_ZERO
#define GDN_FAULT_OUTPUT_ZERO 0
#endif
#ifndef GDN_FAULT_STATE_NO_DELTA
#define GDN_FAULT_STATE_NO_DELTA 0
#endif

enum { S = GDN_S, H = GDN_H, T = GDN_T, MAX_S = 128, MAX_H = 32, MAX_T = 2 };
_Static_assert(S > 0 && S <= MAX_S && H > 0 && H <= MAX_H && T > 0 && T <= MAX_T, "test shape");

static float q_data[MAX_T * MAX_H * MAX_S] __attribute__((aligned(128)));
static float k_data[MAX_T * MAX_H * MAX_S] __attribute__((aligned(128)));
static float v_data[MAX_T * MAX_H * MAX_S] __attribute__((aligned(128)));
static float g_data[MAX_T * MAX_H * MAX_S] __attribute__((aligned(128)));
static float beta_data[MAX_T * MAX_H] __attribute__((aligned(128)));
static float state_in[MAX_H * MAX_S * MAX_S + 32] __attribute__((aligned(128)));
static float dst_data[MAX_T * MAX_H * MAX_S + MAX_H * MAX_S * MAX_S + 32] __attribute__((aligned(128)));
static float ref_state[MAX_H * MAX_S * MAX_S] __attribute__((aligned(128)));
static float ref_output[MAX_T * MAX_H * MAX_S] __attribute__((aligned(128)));
static uint8_t vtcm[2 * MAX_S * MAX_S * sizeof(float) + 128] __attribute__((aligned(128)));
static uint8_t queue_storage[4096] __attribute__((aligned(128)));

static void init_case(void) {
    for (unsigned h = 0; h < H; ++h) {
        for (unsigned r = 0; r < S; ++r) {
            for (unsigned c = 0; c < S; ++c) {
                const unsigned i = h * S * S + r * S + c;
                state_in[i] = ((int) ((i * 17u + 3u) % 29u) - 14) * 0.0002f;
                ref_state[i] = state_in[i];
            }
        }
    }
    for (unsigned t = 0; t < T; ++t) {
        for (unsigned h = 0; h < H; ++h) {
            const unsigned gh = t * H + h;
            beta_data[gh] = 0.25f + (h % 3) * 0.0625f;
            g_data[gh] = -0.2f - (h % 5) * 0.03125f;
            for (unsigned c = 0; c < S; ++c) {
                const unsigned i = gh * S + c;
                q_data[i] = ((int) ((i * 7u + 1u) % 17u) - 8) * 0.001f;
                k_data[i] = ((int) ((i * 11u + 2u) % 19u) - 9) * 0.001f;
                v_data[i] = ((int) ((i * 13u + 4u) % 23u) - 11) * 0.002f;
                if (GDN_VECTOR_GATE) g_data[i] = -0.2f - (c % 7) * 0.015625f;
            }
        }
    }
    for (unsigned i = 0; i < sizeof(dst_data) / sizeof(dst_data[0]); ++i) dst_data[i] = 12345.0f;
    memset(vtcm, 0xa5, sizeof(vtcm));
}

static void scalar_reference(void) {
    const float scale = 1.0f / sqrtf((float) S);
    for (unsigned h = 0; h < H; ++h) {
        for (unsigned t = 0; t < T; ++t) {
            const unsigned base = (t * H + h) * S;
            const float gate = expf(g_data[GDN_VECTOR_GATE ? base : t * H + h]);
            for (unsigned r = 0; r < S; ++r) {
                float * row = ref_state + h * S * S + r * S;
                float dot = 0.0f;
                for (unsigned c = 0; c < S; ++c) {
                    const float mul = GDN_VECTOR_GATE ? expf(g_data[base + c]) : gate;
                    row[c] *= mul;
                    dot += row[c] * k_data[base + c];
                }
                const float delta = (v_data[base + r] - dot) * beta_data[t * H + h];
                float out = 0.0f;
                for (unsigned c = 0; c < S; ++c) {
                    row[c] += k_data[base + c] * delta;
                    out += row[c] * q_data[base + c];
                }
                ref_output[base + r] = out * scale;
            }
        }
    }
}

static uint64_t hash_floats(const float * data, unsigned n, uint64_t hash) {
    for (unsigned i = 0; i < n; ++i) {
        uint32_t bits;
        memcpy(&bits, data + i, sizeof(bits));
        hash = (hash ^ bits) * UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t expected_hash(void) {
    if (S == 128 && H == 32 && T == 1 && !GDN_VECTOR_GATE) return UINT64_C(0x84a4bde95f58714b);
    if (S == 10 && H == 3 && T == 1 && !GDN_VECTOR_GATE) return UINT64_C(0x875fe36b686b76e2);
    if (S == 128 && H == 2 && T == 1 && GDN_VECTOR_GATE) return UINT64_C(0x77fbaf4836454c14);
    if (S == 128 && H == 2 && T == 2 && !GDN_VECTOR_GATE) return UINT64_C(0xb75652c50bf1f9e8);
    return 0;
}

int main(void) {
    init_case();
    if (dma_queue_sizeof(8) > sizeof(queue_storage)) {
        printf("GDN queue storage too small: %zu\n", dma_queue_sizeof(8));
        return 1;
    }
    const size_t state_bytes = (size_t) S * S * sizeof(float);
    const size_t vtcm_bytes = 2 * ((state_bytes + 127) & ~(size_t) 127);
    if (vtcm_bytes + 128 > sizeof(vtcm)) {
        printf("GDN VTCM test storage too small: %zu\n", vtcm_bytes);
        return 2;
    }

    struct htp_tensor q = { .data = (uint32_t) (uintptr_t) q_data, .type = HTP_TYPE_F32,
        .ne = {S, H, T, 1}, .nb = {4, S * 4, S * H * 4, S * H * T * 4} };
    struct htp_tensor k = { .data = (uint32_t) (uintptr_t) k_data, .type = HTP_TYPE_F32,
        .ne = {S, H, T, 1}, .nb = {4, S * 4, S * H * 4, S * H * T * 4} };
    struct htp_tensor v = { .data = (uint32_t) (uintptr_t) v_data, .type = HTP_TYPE_F32,
        .ne = {S, H, T, 1}, .nb = {4, S * 4, S * H * 4, S * H * T * 4} };
    struct htp_tensor g = { .data = (uint32_t) (uintptr_t) g_data, .type = HTP_TYPE_F32,
        .ne = {GDN_VECTOR_GATE ? S : 1, H, T, 1}, .nb = {4, (GDN_VECTOR_GATE ? S : 1) * 4,
            (GDN_VECTOR_GATE ? S : 1) * H * 4, (GDN_VECTOR_GATE ? S : 1) * H * T * 4} };
    struct htp_tensor beta = { .data = (uint32_t) (uintptr_t) beta_data, .type = HTP_TYPE_F32,
        .ne = {1, H, T, 1}, .nb = {4, 4, H * 4, H * T * 4} };
    struct htp_tensor state = { .data = (uint32_t) (uintptr_t) state_in, .type = HTP_TYPE_F32,
        .ne = {S, S, H, 1}, .nb = {4, S * 4, S * S * 4, S * S * H * 4} };
    struct htp_tensor dst = { .data = (uint32_t) (uintptr_t) dst_data, .type = HTP_TYPE_F32,
        .ne = {S * H, T + S, 1, 1}, .nb = {4, S * H * 4, S * H * (T + S) * 4, S * H * (T + S) * 4} };
    struct htp_context ctx = {0};
    struct htp_ops_context octx = { .ctx = &ctx, .op_params = {1},
        .src = {&q, &k, &v, &g, &beta, &state}, .dst = &dst, .n_threads = 1 };
    ctx.vtcm_base = vtcm;
    ctx.vtcm_size = vtcm_bytes;
    ctx.dma[0] = dma_queue_init(queue_storage, 8, (uintptr_t) vtcm, vtcm_bytes, &ctx.trace[0]);
    for (unsigned i = 0; i < GDN_REPEATS; ++i) {
        const int status = op_gated_delta_net(&octx);
        if (status != HTP_STATUS_OK) {
            printf("GDN operation status %d\n", status);
            return 3;
        }
    }
    dma_queue_flush(ctx.dma[0]);
    for (unsigned i = 0; i < 128; ++i) if (vtcm[vtcm_bytes + i] != 0xa5) {
        printf("GDN VTCM guard changed at %u\n", i);
        return 4;
    }
    for (unsigned i = 0; i < 32; ++i) if (dst_data[T * S * H + S * S * H + i] != 12345.0f) {
        printf("GDN output guard changed at %u\n", i);
        return 5;
    }
    for (unsigned i = 0; i < S * S * H; ++i) {
        const float original = ((int) ((i * 17u + 3u) % 29u) - 14) * 0.0002f;
        if (state_in[i] != original) {
            printf("GDN input state changed at %u\n", i);
            return 6;
        }
    }
    scalar_reference();
    if (GDN_FAULT_OUTPUT_ZERO) {
        memset(dst_data, 0, T * S * H * sizeof(float));
    }
    if (GDN_FAULT_STATE_NO_DELTA) {
        if (T != 1 || GDN_VECTOR_GATE) return 9;
        for (unsigned h = 0; h < H; ++h) {
            const float gate = expf(g_data[h]);
            for (unsigned i = 0; i < S * S; ++i) {
                const unsigned off = h * S * S + i;
                dst_data[T * S * H + off] = state_in[off] * gate;
            }
        }
    }
    float max_output_magnitude = 0.0f;
    float max_output_error = 0.0f;
    float max_state_error = 0.0f;
    float max_state_delta = 0.0f;
    float no_delta_gate[MAX_H];
    if (T == 1 && !GDN_VECTOR_GATE) {
        for (unsigned h = 0; h < H; ++h) no_delta_gate[h] = expf(g_data[h]);
    }
    for (unsigned i = 0; i < T * S * H; ++i) {
        const float magnitude = fabsf(ref_output[i]);
        const float error = fabsf(dst_data[i] - ref_output[i]);
        if (magnitude > max_output_magnitude) max_output_magnitude = magnitude;
        if (error > max_output_error) max_output_error = error;
        if (!(error <= 0.000001f)) {
            printf("output mismatch %u: %.8f != %.8f\n", i, dst_data[i], ref_output[i]);
            return 7;
        }
    }
    for (unsigned i = 0; i < S * S * H; ++i) {
        const float error = fabsf(dst_data[T * S * H + i] - ref_state[i]);
        if (error > max_state_error) max_state_error = error;
        if (T == 1 && !GDN_VECTOR_GATE) {
            const float no_delta = state_in[i] * no_delta_gate[i / (S * S)];
            const float delta = fabsf(ref_state[i] - no_delta);
            if (delta > max_state_delta) max_state_delta = delta;
        }
        if (!(error <= 0.000001f)) {
            printf("state mismatch %u: %.8f != %.8f\n", i, dst_data[T * S * H + i], ref_state[i]);
            return 8;
        }
    }
    printf("GDN reference max output %.9g, output error %.9g, state error %.9g, state delta %.9g\n",
        max_output_magnitude, max_output_error, max_state_error, max_state_delta);
    uint64_t hash = hash_floats(dst_data, T * S * H + S * S * H, UINT64_C(1469598103934665603));
    if (expected_hash() != 0 && hash != expected_hash()) {
        printf("GDN raw output/state hash mismatch: %016llx != %016llx\n",
            (unsigned long long) hash, (unsigned long long) expected_hash());
        return 10;
    }
    printf("GDN worker S=%d H=%d T=%d vector=%d repeats=%d hash=%016llx\n",
        S, H, T, GDN_VECTOR_GATE, GDN_REPEATS, (unsigned long long) hash);
    return 0;
}
