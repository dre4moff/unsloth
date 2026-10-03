// sme_attn.c - SME2 "co-attention" partial for the Qwen3.8-27B verify shape (Backburner, 2026-09-23).
//
// One KV head: 48 query rows (8 tokens x GQA 6) x head dim 256 against keys [k0, k1).
// Output: the unnormalised partial (O[48][256], m[48], l[48]) that a log-sum-exp merge takes:
//   O = sum_j exp(s_j - m) v_j,  l = sum_j exp(s_j - m),  m = max_j s_j,  s_j = scale * q . k_j
//
// Split of work inside one thread:
//   NEON (non-streaming): pack K into the "16 keys x d-pair" layout FMOPA wants (fp16 copy or q8_0
//                         dequant), online softmax on S^T (rows in lanes), pack P and V (key-pair zip).
//   SME  (streaming)    : S^T = Q K^T with FMOPA fp16->fp32 (4 ZA tiles = 16 rows x 64 keys),
//                         O_blk = P V with FMOPA fp16->fp32 (4 ZA tiles = 16 rows x 64 d).
//
// K/V cache layout = llama.cpp's with flash attention (not transposed): row per key, the 4 KV heads
// next to each other, head dim contiguous. fp16: 256 halfs/head. q8_0: 8 blocks x 34 B/head.
//
// Build (macOS):  clang -O3 -mcpu=apple-m4 -DSME_BENCH_MAIN sme_attn.c -o sme_attn
// Build (iOS)  :  xcrun --sdk iphoneos clang -O3 -arch arm64 -mcpu=apple-m4 -DSME_BENCH_MAIN \
//                   -miphoneos-version-min=18.0 sme_attn.c -o sme_attn_ios
// Without -DSME_BENCH_MAIN it is a library file: call sme_attn_partial() (see the header comment on it).
#include <arm_neon.h>
#include <arm_sme.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NR 48       // query rows per KV head
#define HD 256      // head dim
#define BK_MAX 1024
#ifndef PFD
#define PFD 16      // prefetch distance in keys
#endif // keys per online-softmax block (multiple of 64)

typedef __fp16 f16;

typedef struct {
    f16   *Qp;               // [128 d-pairs][48 rows][2]
    f16   *Kp;               // [BK/64][128][64 keys][2]
    f16   *Vp;               // [BK/2 key-pairs][256 d][2]
    f16   *Pp;               // [BK/2][48][2]
    float *St;               // [BK][48]  S transposed
    float *Ot;               // [48][256]
} sme_ws;

// ---------------------------------------------------------------- SME (streaming) parts

__arm_locally_streaming __arm_new("za")
static void sme_qk(const f16 *Qp, const f16 *Kp, float *St, int bk) {
    const svbool_t  ph = svptrue_b16();
    const svbool_t  ps = svptrue_b32();
    const svcount_t pc = svptrue_c16();
    for (int g = 0; g < bk / 64; g++) {
        const f16 *kg = Kp + (size_t)g * 128 * 128;
        for (int qt = 0; qt < 3; qt++) {
            svzero_za();
            const f16 *qq = Qp + qt * 32;
            for (int p = 0; p < 128; p++) {
                svfloat16_t   q = svld1_f16(ph, (const float16_t *)(qq + p * 96));
                svfloat16x4_t k = svld1_f16_x4(pc, (const float16_t *)(kg + p * 128));
                svmopa_za32_f16_m(0, ph, ph, q, svget4_f16(k, 0));
                svmopa_za32_f16_m(1, ph, ph, q, svget4_f16(k, 1));
                svmopa_za32_f16_m(2, ph, ph, q, svget4_f16(k, 2));
                svmopa_za32_f16_m(3, ph, ph, q, svget4_f16(k, 3));
            }
            // tile t, column c = key g*64 + t*16 + c; the column holds 16 query rows -> St[key][qt*16..]
            float *so = St + (size_t)g * 64 * NR + qt * 16;
            for (int c = 0; c < 16; c++) {
                svst1_ver_za32(0, c, ps, so + (0 * 16 + c) * NR);
                svst1_ver_za32(1, c, ps, so + (1 * 16 + c) * NR);
                svst1_ver_za32(2, c, ps, so + (2 * 16 + c) * NR);
                svst1_ver_za32(3, c, ps, so + (3 * 16 + c) * NR);
            }
        }
    }
}

__arm_locally_streaming __arm_new("za")
static void sme_pv(const f16 *Pp, const f16 *Vp, float *Ot, int bk) {
    const svbool_t  ph = svptrue_b16();
    const svbool_t  ps = svptrue_b32();
    const svcount_t pc = svptrue_c16();
    for (int qt = 0; qt < 3; qt++) {
        for (int dg = 0; dg < 4; dg++) {
            svzero_za();
            const f16 *pp = Pp + qt * 32;
            const f16 *vv = Vp + dg * 128;
            for (int kp = 0; kp < bk / 2; kp++) {
                svfloat16_t   p = svld1_f16(ph, (const float16_t *)(pp + kp * 96));
                svfloat16x4_t v = svld1_f16_x4(pc, (const float16_t *)(vv + kp * 512));
                svmopa_za32_f16_m(0, ph, ph, p, svget4_f16(v, 0));
                svmopa_za32_f16_m(1, ph, ph, p, svget4_f16(v, 1));
                svmopa_za32_f16_m(2, ph, ph, p, svget4_f16(v, 2));
                svmopa_za32_f16_m(3, ph, ph, p, svget4_f16(v, 3));
            }
            float *oo = Ot + (size_t)qt * 16 * HD + dg * 64;
            for (int r = 0; r < 16; r++) {
                svst1_hor_za32(0, r, ps, oo + r * HD + 0);
                svst1_hor_za32(1, r, ps, oo + r * HD + 16);
                svst1_hor_za32(2, r, ps, oo + r * HD + 32);
                svst1_hor_za32(3, r, ps, oo + r * HD + 48);
            }
        }
    }
}

// ---------------------------------------------------------------- NEON parts

static inline float32x4_t v_expf(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-87.0f));
    float32x4_t t = vmulq_n_f32(x, 1.4426950409f);
    float32x4_t n = vrndnq_f32(t);
    float32x4_t f = vsubq_f32(t, n);                  // 2^f, |f| <= 0.5, Taylor deg 6 (~1e-7)
    float32x4_t p = vdupq_n_f32(1.5403530e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3333558e-3f), p, f);
    p = vfmaq_f32(vdupq_n_f32(9.6181291e-3f), p, f);
    p = vfmaq_f32(vdupq_n_f32(5.5504109e-2f), p, f);
    p = vfmaq_f32(vdupq_n_f32(2.4022651e-1f), p, f);
    p = vfmaq_f32(vdupq_n_f32(6.9314718e-1f), p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, f);
    int32x4_t e = vshlq_n_s32(vcvtq_s32_f32(n), 23);
    return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), e));
}

// 4x4 transpose of 32-bit words
static inline void tr4(uint32x4_t *a, uint32x4_t *b, uint32x4_t *c, uint32x4_t *d) {
    uint32x4_t t0 = vtrn1q_u32(*a, *b), t1 = vtrn2q_u32(*a, *b);
    uint32x4_t t2 = vtrn1q_u32(*c, *d), t3 = vtrn2q_u32(*c, *d);
    *a = vreinterpretq_u32_u64(vtrn1q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
    *b = vreinterpretq_u32_u64(vtrn1q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
    *c = vreinterpretq_u32_u64(vtrn2q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
    *d = vreinterpretq_u32_u64(vtrn2q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
}

// K rows (fp16, head dim contiguous) -> Kp[g][p][64][2]: word (key j, d-pair p) to Kp32[(g*128+p)*64 + jj]
static void pack_k_f16(const uint8_t *K, size_t rs, int bk, uint32_t *Kp32) {
    for (int j = 0; j < bk; j += 4) {
        const uint32_t *r0 = (const uint32_t *)(K + (size_t)(j + 0) * rs);
        const uint32_t *r1 = (const uint32_t *)(K + (size_t)(j + 1) * rs);
        const uint32_t *r2 = (const uint32_t *)(K + (size_t)(j + 2) * rs);
        const uint32_t *r3 = (const uint32_t *)(K + (size_t)(j + 3) * rs);
        uint32_t *o = Kp32 + (size_t)(j / 64) * 128 * 64 + (j % 64);
        for (int i = 0; i < 4; i++) for (int c = 0; c < 512; c += 128) __builtin_prefetch(K + (size_t)(j + PFD + i) * rs + c);
        for (int p = 0; p < 128; p += 4) {
            uint32x4_t a = vld1q_u32(r0 + p), b = vld1q_u32(r1 + p), c = vld1q_u32(r2 + p), d = vld1q_u32(r3 + p);
            tr4(&a, &b, &c, &d);
            vst1q_u32(o + (p + 0) * 64, a); vst1q_u32(o + (p + 1) * 64, b);
            vst1q_u32(o + (p + 2) * 64, c); vst1q_u32(o + (p + 3) * 64, d);
        }
    }
}

// dequant one q8_0 block (2 B fp16 scale + 32 int8) to 32 fp16 (as 4 x float16x8)
static inline void deq_q8(const uint8_t *blk, float16x8_t o[4]) {
    f16 d; memcpy(&d, blk, 2);
    int8x16_t q0 = vld1q_s8((const int8_t *)blk + 2), q1 = vld1q_s8((const int8_t *)blk + 18);
    o[0] = vmulq_n_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(q0))), d);
    o[1] = vmulq_n_f16(vcvtq_f16_s16(vmovl_high_s8(q0)), d);
    o[2] = vmulq_n_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(q1))), d);
    o[3] = vmulq_n_f16(vcvtq_f16_s16(vmovl_high_s8(q1)), d);
}

static void pack_k_q8(const uint8_t *K, size_t rs, int bk, uint32_t *Kp32) {
    for (int j = 0; j < bk; j += 4) {
        uint32_t *o = Kp32 + (size_t)(j / 64) * 128 * 64 + (j % 64);
        for (int i = 0; i < 4; i++) for (int c = 0; c < 272; c += 128) __builtin_prefetch(K + (size_t)(j + PFD + i) * rs + c);
        __builtin_prefetch(K + (size_t)(j + PFD) * rs + 271); __builtin_prefetch(K + (size_t)(j + PFD + 3) * rs + 271);
        for (int b = 0; b < 8; b++) {              // 8 blocks of 32 d = 16 d-pairs each
            float16x8_t x[4][4];
            for (int i = 0; i < 4; i++) deq_q8(K + (size_t)(j + i) * rs + b * 34, x[i]);
            for (int w = 0; w < 4; w++) {          // 4 words-groups of 4 d-pairs
                uint32x4_t a = vreinterpretq_u32_f16(x[0][w]), bb = vreinterpretq_u32_f16(x[1][w]);
                uint32x4_t c = vreinterpretq_u32_f16(x[2][w]), d = vreinterpretq_u32_f16(x[3][w]);
                tr4(&a, &bb, &c, &d);
                int p = b * 16 + w * 4;
                vst1q_u32(o + (p + 0) * 64, a); vst1q_u32(o + (p + 1) * 64, bb);
                vst1q_u32(o + (p + 2) * 64, c); vst1q_u32(o + (p + 3) * 64, d);
            }
        }
    }
}

// V rows -> Vp[kp][d][2] (zip of rows 2kp, 2kp+1)
static void pack_v_f16(const uint8_t *V, size_t rs, int bk, f16 *Vp) {
    for (int kp = 0; kp < bk / 2; kp++) {
        const f16 *a = (const f16 *)(V + (size_t)(2 * kp) * rs), *b = (const f16 *)(V + (size_t)(2 * kp + 1) * rs);
        f16 *o = Vp + (size_t)kp * 512;
        for (int c = 0; c < 512; c += 128) { __builtin_prefetch((const uint8_t *)a + PFD * rs + c); __builtin_prefetch((const uint8_t *)b + PFD * rs + c); }
        for (int d = 0; d < HD; d += 8) {
            float16x8_t x = vld1q_f16(a + d), y = vld1q_f16(b + d);
            vst1q_f16(o + 2 * d, vzip1q_f16(x, y));
            vst1q_f16(o + 2 * d + 8, vzip2q_f16(x, y));
        }
    }
}

static void pack_v_q8(const uint8_t *V, size_t rs, int bk, f16 *Vp) {
    for (int kp = 0; kp < bk / 2; kp++) {
        const uint8_t *a = V + (size_t)(2 * kp) * rs, *b = V + (size_t)(2 * kp + 1) * rs;
        f16 *o = Vp + (size_t)kp * 512;
        for (int c = 0; c < 272; c += 128) { __builtin_prefetch(a + PFD * rs + c); __builtin_prefetch(b + PFD * rs + c); }
        __builtin_prefetch(a + PFD * rs + 271); __builtin_prefetch(b + PFD * rs + 271);
        for (int blk = 0; blk < 8; blk++) {
            float16x8_t x[4], y[4];
            deq_q8(a + blk * 34, x); deq_q8(b + blk * 34, y);
            for (int w = 0; w < 4; w++) {
                int d = blk * 32 + w * 8;
                vst1q_f16(o + 2 * d, vzip1q_f16(x[w], y[w]));
                vst1q_f16(o + 2 * d + 8, vzip2q_f16(x[w], y[w]));
            }
        }
    }
}

// online softmax on one block. St[bk][48] (already scaled). Updates m,l; writes Pp; returns alpha[48].
static void softmax_blk(const float *St, int bk, float *m, float *l, float *alpha, f16 *Pp) {
    float32x4_t mx[12], mn[12], ls[12];
    for (int i = 0; i < 12; i++) mx[i] = vdupq_n_f32(-INFINITY);
    for (int j = 0; j < bk; j++)
        for (int i = 0; i < 12; i++) mx[i] = vmaxq_f32(mx[i], vld1q_f32(St + j * NR + 4 * i));
    for (int i = 0; i < 12; i++) {
        float32x4_t mo = vld1q_f32(m + 4 * i);
        mn[i] = vmaxq_f32(mo, mx[i]);
        vst1q_f32(alpha + 4 * i, v_expf(vsubq_f32(mo, mn[i])));
        vst1q_f32(m + 4 * i, mn[i]);
        ls[i] = vdupq_n_f32(0);
    }
    for (int kp = 0; kp < bk / 2; kp++) {
        const float *s0 = St + (2 * kp) * NR, *s1 = s0 + NR;
        f16 *o = Pp + kp * 96;
        for (int i = 0; i < 12; i++) {
            float32x4_t p0 = v_expf(vsubq_f32(vld1q_f32(s0 + 4 * i), mn[i]));
            float32x4_t p1 = v_expf(vsubq_f32(vld1q_f32(s1 + 4 * i), mn[i]));
            ls[i] = vaddq_f32(ls[i], vaddq_f32(p0, p1));
            float16x4_t h0 = vcvt_f16_f32(p0), h1 = vcvt_f16_f32(p1);
            vst1q_f16(o + 8 * i, vcombine_f16(vzip1_f16(h0, h1), vzip2_f16(h0, h1)));
        }
    }
    for (int i = 0; i < 12; i++) {
        float32x4_t lo = vld1q_f32(l + 4 * i);
        vst1q_f32(l + 4 * i, vfmaq_f32(ls[i], lo, vld1q_f32(alpha + 4 * i)));
    }
}

static void o_update(float *O, const float *Ot, const float *alpha) {
    for (int r = 0; r < NR; r++) {
        float32x4_t a = vdupq_n_f32(alpha[r]);
        float *o = O + r * HD; const float *t = Ot + r * HD;
        for (int d = 0; d < HD; d += 4) vst1q_f32(o + d, vfmaq_f32(vld1q_f32(t + d), vld1q_f32(o + d), a));
    }
}

// ---------------------------------------------------------------- public API

sme_ws *sme_ws_new(void) {
    sme_ws *w = calloc(1, sizeof *w);
    w->Qp = aligned_alloc(128, 128 * NR * 2 * sizeof(f16));
    w->Kp = aligned_alloc(128, (size_t)BK_MAX * HD * sizeof(f16));
    w->Vp = aligned_alloc(128, (size_t)BK_MAX * HD * sizeof(f16));
    w->Pp = aligned_alloc(128, (size_t)BK_MAX * NR * sizeof(f16));
    w->St = aligned_alloc(128, (size_t)BK_MAX * NR * sizeof(float));
    w->Ot = aligned_alloc(128, (size_t)NR * HD * sizeof(float));
    return w;
}

double sme_t_pack, sme_t_qk, sme_t_sm, sme_t_pv, sme_t_upd; // phase timers (only if sme_timing)
int sme_timing = 0;
#include <time.h>
static inline double now_s(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }

// One KV head. Q: fp32 [48][256] (row stride qrs floats). K,V: pointer to key k0's row for this head
// (i.e. base + k0*rs + head_offset), rs = row stride in bytes, is_q8: 0 fp16, 1 q8_0.
// nk = number of keys, multiple of 64. Writes O[48][256] (unnormalised), m[48], l[48].
void sme_attn_partial(sme_ws *w, const float *Q, int qrs, const uint8_t *K, const uint8_t *V, size_t rs,
                      int is_q8, int nk, float scale, int bk, float *O, float *m, float *l) {
    for (int p = 0; p < 128; p++)
        for (int r = 0; r < NR; r++) {
            w->Qp[p * 96 + r * 2 + 0] = (f16)(Q[r * qrs + 2 * p + 0] * scale);
            w->Qp[p * 96 + r * 2 + 1] = (f16)(Q[r * qrs + 2 * p + 1] * scale);
        }
    for (int r = 0; r < NR; r++) { m[r] = -1e30f; l[r] = 0; }
    memset(O, 0, sizeof(float) * NR * HD);
    float alpha[NR];
    for (int j0 = 0; j0 < nk; j0 += bk) {
        int b = nk - j0 < bk ? nk - j0 : bk;
        double t0 = sme_timing ? now_s() : 0;
        if (is_q8) pack_k_q8(K + (size_t)j0 * rs, rs, b, (uint32_t *)w->Kp);
        else       pack_k_f16(K + (size_t)j0 * rs, rs, b, (uint32_t *)w->Kp);
        double t1 = sme_timing ? now_s() : 0;
        sme_qk(w->Qp, w->Kp, w->St, b);
        double t2 = sme_timing ? now_s() : 0;
        softmax_blk(w->St, b, m, l, alpha, w->Pp);
        double t3 = sme_timing ? now_s() : 0;
        if (is_q8) pack_v_q8(V + (size_t)j0 * rs, rs, b, w->Vp);
        else       pack_v_f16(V + (size_t)j0 * rs, rs, b, w->Vp);
        double t4 = sme_timing ? now_s() : 0;
        sme_pv(w->Pp, w->Vp, w->Ot, b);
        double t5 = sme_timing ? now_s() : 0;
        o_update(O, w->Ot, alpha);
        if (sme_timing) {
            double t6 = now_s();
            sme_t_pack += (t1 - t0) + (t4 - t3); sme_t_qk += t2 - t1; sme_t_sm += t3 - t2;
            sme_t_pv += t5 - t4; sme_t_upd += t6 - t5;
        }
    }
}


// ---------------------------------------------------------------- pipelined version: helpers pack, SME thread computes
// One "SME worker" (ideally one per P-cluster) + nh NEON helper threads. Work items = (head, block of bk keys).
// Helper t packs items t, t+nh, ... into a ring of NSLOT slots; the SME worker consumes them in order.
#include <stdatomic.h>
#define NSLOT 8
typedef struct {
    const float *Q; int qrs, qhs;            // Q for head h at Q + h*qhs
    const uint8_t *K, *V; size_t rs, hb;     // head h base = K + h*hb
    int is_q8, nk, bk, h0, h1; float scale;
    float *O, *m, *l;                        // head h at O + h*48*256 etc.
    int nh;
    f16 *Kp[NSLOT], *Vp[NSLOT];
    _Atomic int ready[NSLOT];
    _Atomic int consumed;
    sme_ws *w;
    // softmax offload (sm_off=1): QK(i) -> St[i%2] -> softmax helper -> Pp[i%2], alpha[i%2]; worker does PV(i-1) meanwhile
    int sm_off;
    float *St2[2]; f16 *Pp2[2]; float alpha2[2][NR];
    _Atomic int st_ready, sm_done;
} sme_pipe;

static int pipe_nitems(const sme_pipe *p) { return (p->h1 - p->h0) * ((p->nk + p->bk - 1) / p->bk); }

void sme_pipe_helper(sme_pipe *p, int t) {
    int nb = (p->nk + p->bk - 1) / p->bk, n = pipe_nitems(p);
    for (int i = t; i < n; i += p->nh) {
        int s = i % NSLOT, h = p->h0 + i / nb, j0 = (i % nb) * p->bk;
        int b = p->nk - j0 < p->bk ? p->nk - j0 : p->bk;
        while (atomic_load_explicit(&p->consumed, memory_order_acquire) < i - NSLOT + 1) __builtin_arm_yield();
        const uint8_t *K = p->K + h * p->hb + (size_t)j0 * p->rs, *V = p->V + h * p->hb + (size_t)j0 * p->rs;
        if (p->is_q8) { pack_k_q8(K, p->rs, b, (uint32_t *)p->Kp[s]); pack_v_q8(V, p->rs, b, p->Vp[s]); }
        else          { pack_k_f16(K, p->rs, b, (uint32_t *)p->Kp[s]); pack_v_f16(V, p->rs, b, p->Vp[s]); }
        atomic_store_explicit(&p->ready[s], i + 1, memory_order_release);
    }
}

void sme_pipe_worker(sme_pipe *p) {
    sme_ws *w = p->w;
    int nb = (p->nk + p->bk - 1) / p->bk, n = pipe_nitems(p);
    float alpha[NR];
    for (int i = 0; i < n; i++) {
        int s = i % NSLOT, h = p->h0 + i / nb, j0 = (i % nb) * p->bk;
        int b = p->nk - j0 < p->bk ? p->nk - j0 : p->bk;
        float *O = p->O + h * NR * HD, *m = p->m + h * NR, *l = p->l + h * NR;
        if (j0 == 0) {
            const float *Q = p->Q + h * p->qhs;
            for (int q = 0; q < 128; q++)
                for (int r = 0; r < NR; r++) {
                    w->Qp[q * 96 + r * 2 + 0] = (f16)(Q[r * p->qrs + 2 * q + 0] * p->scale);
                    w->Qp[q * 96 + r * 2 + 1] = (f16)(Q[r * p->qrs + 2 * q + 1] * p->scale);
                }
            for (int r = 0; r < NR; r++) { m[r] = -1e30f; l[r] = 0; }
            memset(O, 0, sizeof(float) * NR * HD);
        }
        while (atomic_load_explicit(&p->ready[s], memory_order_acquire) != i + 1) __builtin_arm_yield();
        sme_qk(w->Qp, p->Kp[s], w->St, b);
        softmax_blk(w->St, b, m, l, alpha, w->Pp);
        sme_pv(w->Pp, p->Vp[s], w->Ot, b);
        atomic_store_explicit(&p->consumed, i + 1, memory_order_release);
        o_update(O, w->Ot, alpha);
    }
}

static void pipe_item(const sme_pipe *p, int i, int *h, int *j0, int *b) {
    int nb = (p->nk + p->bk - 1) / p->bk;
    *h = p->h0 + i / nb; *j0 = (i % nb) * p->bk; *b = p->nk - *j0 < p->bk ? p->nk - *j0 : p->bk;
}

void sme_pipe_softmax_helper(sme_pipe *p) {
    int n = pipe_nitems(p);
    for (int i = 0; i < n; i++) {
        int h, j0, b; pipe_item(p, i, &h, &j0, &b);
        float *m = p->m + h * NR, *l = p->l + h * NR;
        if (j0 == 0) for (int r = 0; r < NR; r++) { m[r] = -1e30f; l[r] = 0; }
        while (atomic_load_explicit(&p->st_ready, memory_order_acquire) < i + 1) __builtin_arm_yield();
        softmax_blk(p->St2[i & 1], b, m, l, p->alpha2[i & 1], p->Pp2[i & 1]);
        atomic_store_explicit(&p->sm_done, i + 1, memory_order_release);
    }
}

void sme_pipe_worker2(sme_pipe *p) {
    sme_ws *w = p->w;
    int n = pipe_nitems(p);
    for (int i = 0; i <= n; i++) {
        if (i < n) {
            int h, j0, b; pipe_item(p, i, &h, &j0, &b);
            if (j0 == 0) {
                const float *Q = p->Q + h * p->qhs;
                for (int q = 0; q < 128; q++)
                    for (int r = 0; r < NR; r++) {
                        w->Qp[q * 96 + r * 2 + 0] = (f16)(Q[r * p->qrs + 2 * q + 0] * p->scale);
                        w->Qp[q * 96 + r * 2 + 1] = (f16)(Q[r * p->qrs + 2 * q + 1] * p->scale);
                    }
                memset(p->O + h * NR * HD, 0, sizeof(float) * NR * HD);
            }
            while (atomic_load_explicit(&p->ready[i % NSLOT], memory_order_acquire) != i + 1) __builtin_arm_yield();
            while (atomic_load_explicit(&p->sm_done, memory_order_acquire) < i - 1) __builtin_arm_yield(); // St[i&1] free
            sme_qk(w->Qp, p->Kp[i % NSLOT], p->St2[i & 1], b);
            atomic_store_explicit(&p->st_ready, i + 1, memory_order_release);
        }
        if (i >= 1) {
            int k = i - 1, h, j0, b; pipe_item(p, k, &h, &j0, &b);
            while (atomic_load_explicit(&p->sm_done, memory_order_acquire) < k + 1) __builtin_arm_yield();
            sme_pv(p->Pp2[k & 1], p->Vp[k % NSLOT], w->Ot, b);
            atomic_store_explicit(&p->consumed, k + 1, memory_order_release);
            o_update(p->O + h * NR * HD, w->Ot, p->alpha2[k & 1]);
        }
    }
}

sme_pipe *sme_pipe_new(int nh) {
    sme_pipe *p = calloc(1, sizeof *p);
    p->nh = nh; p->w = sme_ws_new();
    for (int s = 0; s < NSLOT; s++) {
        p->Kp[s] = aligned_alloc(128, (size_t)BK_MAX * HD * sizeof(f16));
        p->Vp[s] = aligned_alloc(128, (size_t)BK_MAX * HD * sizeof(f16));
    }
    for (int s = 0; s < 2; s++) {
        p->St2[s] = aligned_alloc(128, (size_t)BK_MAX * NR * sizeof(float));
        p->Pp2[s] = aligned_alloc(128, (size_t)BK_MAX * NR * sizeof(f16));
    }
    return p;
}
static void sme_pipe_reset(sme_pipe *p) {
    for (int s = 0; s < NSLOT; s++) atomic_store(&p->ready[s], 0);
    atomic_store(&p->consumed, 0); atomic_store(&p->st_ready, 0); atomic_store(&p->sm_done, 0);
}

// ---------------------------------------------------------------- one-call pipelined partial (phone-attn server)
// All nkv KV heads, keys [0, nk): nw SME workers (heads split between them), each with nh NEON pack helpers.
// pp: nw pipes from sme_pipe_new(nh). Q: [nkv][48][256] fp32 (unscaled). Writes O [nkv][48][256] unnormalised, m, l [nkv][48].
#include <pthread.h>
typedef struct { sme_pipe *p; int role; } sme_pipe_job;
static void *sme_pipe_thr(void *a) {
    sme_pipe_job *j = a;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    if (j->role < 0) sme_pipe_worker(j->p); else sme_pipe_helper(j->p, j->role);
    return NULL;
}
void sme_attn_pipe(sme_pipe **pp, int nw, int nh, const float *Q, const uint8_t *K, const uint8_t *V, size_t rs, size_t hb,
                   int is_q8, int nkv, int nk, float scale, float *O, float *m, float *l) {
    pthread_t th[64]; sme_pipe_job jb[64]; int k = 0;
    for (int w = 0; w < nw; w++) {
        sme_pipe *p = pp[w]; sme_pipe_reset(p);
        p->Q = Q; p->qrs = HD; p->qhs = NR * HD; p->K = K; p->V = V; p->rs = rs; p->hb = hb; p->is_q8 = is_q8;
        p->nk = nk; p->bk = 512; p->h0 = w * nkv / nw; p->h1 = (w + 1) * nkv / nw; p->scale = scale;
        p->O = O; p->m = m; p->l = l; p->nh = nh; p->sm_off = 0;
        for (int r = -1; r < nh; r++) {
            if (w == 0 && r == -1) continue;              // the caller is worker 0
            jb[k] = (sme_pipe_job){ p, r }; pthread_create(&th[k], 0, sme_pipe_thr, &jb[k]); k++;
        }
    }
    sme_pipe_worker(pp[0]);
    for (int t = 0; t < k; t++) pthread_join(th[t], 0);
}

// ---------------------------------------------------------------- compressed-domain int8 QK (q8_0 K, Q quantised per 32-block)
// T[g][b][key 64][48] int32 = per-block partial dot products; the fp32 score needs sum_b dq[r,b]*dk[key,b]*T.
// Qi: [64 quads][48 rows][4] int8 ; Ki: [g][64 quads][64 keys][4] int8
__arm_locally_streaming __arm_new("za")
static void sme_qk_i8blk(const int8_t *Qi, const int8_t *Ki, int32_t *T, int bk) {
    const svbool_t pb = svptrue_b8(), ps = svptrue_b32();
    const svcount_t pc = svptrue_c8();
    for (int g = 0; g < bk / 64; g++)
        for (int qt = 0; qt < 3; qt++)
            for (int b = 0; b < 8; b++) {
                svzero_za();
                for (int qd = 0; qd < 8; qd++) {
                    int quad = b * 8 + qd;
                    svint8_t   q = svld1_s8(pb, Qi + quad * 192 + qt * 64);
                    svint8x4_t k = svld1_s8_x4(pc, Ki + ((size_t)g * 64 + quad) * 256);
                    svmopa_za32_s8_m(0, pb, pb, q, svget4_s8(k, 0));
                    svmopa_za32_s8_m(1, pb, pb, q, svget4_s8(k, 1));
                    svmopa_za32_s8_m(2, pb, pb, q, svget4_s8(k, 2));
                    svmopa_za32_s8_m(3, pb, pb, q, svget4_s8(k, 3));
                }
                int32_t *to = T + (((size_t)g * 8 + b) * 64) * NR + qt * 16;
                for (int c = 0; c < 16; c++) {
                    svst1_ver_za32(0, c, ps, to + (0 * 16 + c) * NR);
                    svst1_ver_za32(1, c, ps, to + (1 * 16 + c) * NR);
                    svst1_ver_za32(2, c, ps, to + (2 * 16 + c) * NR);
                    svst1_ver_za32(3, c, ps, to + (3 * 16 + c) * NR);
                }
            }
}
// NEON: St[key][r] = sum_b dq[b][r] * dk[key][b] * T[g][b][key][r]
static void i8_combine(const int32_t *T, const float *dq /*[8][48]*/, const float *dk /*[bk][8]*/, float *St, int bk) {
    for (int j = 0; j < bk; j++) {
        int g = j / 64, jj = j % 64;
        float32x4_t acc[12];
        for (int i = 0; i < 12; i++) acc[i] = vdupq_n_f32(0);
        for (int b = 0; b < 8; b++) {
            const int32_t *t = T + (((size_t)g * 8 + b) * 64 + jj) * NR;
            float s = dk[j * 8 + b];
            for (int i = 0; i < 12; i++)
                acc[i] = vfmaq_f32(acc[i], vcvtq_f32_s32(vld1q_s32(t + 4 * i)), vmulq_n_f32(vld1q_f32(dq + b * NR + 4 * i), s));
        }
        for (int i = 0; i < 12; i++) vst1q_f32(St + j * NR + 4 * i, acc[i]);
    }
}

// ================================================================ bench
#ifdef SME_BENCH_MAIN
#include <pthread.h>
#include <sys/qos.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>

static double tnow(void) { return now_s(); }

// raw MOPA ceilings (no loads): 4 tiles, dependent chains of length iters
__arm_locally_streaming __arm_new("za")
static void peak_f16(long iters) {
    svbool_t ph = svptrue_b16();
    svfloat16_t a = svdup_f16(1.0e-3), b = svdup_f16(1.0e-3), c = svdup_f16(2e-3), d = svdup_f16(3e-3);
    for (long i = 0; i < iters; i++) {
        svmopa_za32_f16_m(0, ph, ph, a, b); svmopa_za32_f16_m(1, ph, ph, a, c);
        svmopa_za32_f16_m(2, ph, ph, d, b); svmopa_za32_f16_m(3, ph, ph, d, c);
    }
}
__arm_locally_streaming __arm_new("za")
static void peak_f32(long iters) {
    svbool_t ps = svptrue_b32();
    svfloat32_t a = svdup_f32(1e-3f), b = svdup_f32(1e-3f), c = svdup_f32(2e-3f), d = svdup_f32(3e-3f);
    for (long i = 0; i < iters; i++) {
        svmopa_za32_f32_m(0, ps, ps, a, b); svmopa_za32_f32_m(1, ps, ps, a, c);
        svmopa_za32_f32_m(2, ps, ps, d, b); svmopa_za32_f32_m(3, ps, ps, d, c);
    }
}
__arm_locally_streaming __arm_new("za")
static void peak_s8(long iters) {
    svbool_t pb = svptrue_b8();
    svint8_t a = svdup_s8(1), b = svdup_s8(2), c = svdup_s8(3), d = svdup_s8(-1);
    for (long i = 0; i < iters; i++) {
        svmopa_za32_s8_m(0, pb, pb, a, b); svmopa_za32_s8_m(1, pb, pb, a, c);
        svmopa_za32_s8_m(2, pb, pb, d, b); svmopa_za32_s8_m(3, pb, pb, d, c);
    }
}

typedef struct { int kind; long iters; double t; } peak_arg;
static void *peak_thr(void *p) {
    peak_arg *a = p;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    double t0 = tnow();
    if (a->kind == 0) peak_f16(a->iters); else if (a->kind == 1) peak_f32(a->iters); else peak_s8(a->iters);
    a->t = tnow() - t0;
    return NULL;
}

// ---- data
static uint16_t f2h(float x) { f16 h = (f16)x; uint16_t u; memcpy(&u, &h, 2); return u; }
static float frand(uint64_t *s) { *s = *s * 6364136223846793005ULL + 1442695040888963407ULL; return ((*s >> 40) / (float)(1 << 24)) * 2 - 1; }

typedef struct {
    int N, is_q8, bk; size_t rs; uint8_t *K, *V; float *Q; // Q [4][48][256]
    float *O, *m, *l;                                        // per head
} prob;

// minimal pthread barrier for macOS
typedef struct { pthread_mutex_t mu; pthread_cond_t cv; int n, c, gen; } xbar_t;
static int xbar_init(xbar_t *b, void *a, int n) { (void)a; pthread_mutex_init(&b->mu, 0); pthread_cond_init(&b->cv, 0); b->n = n; b->c = 0; b->gen = 0; return 0; }
static int xbar_wait(xbar_t *b) {
    pthread_mutex_lock(&b->mu); int g = b->gen;
    if (++b->c == b->n) { b->gen++; b->c = 0; pthread_cond_broadcast(&b->cv); }
    else while (g == b->gen) pthread_cond_wait(&b->cv, &b->mu);
    pthread_mutex_unlock(&b->mu); return 0;
}

typedef struct { prob *pr; int h0, h1; sme_ws *w; int tag; xbar_t *bar; double t0, t1; } job;
static void *run_job(void *p) {
    job *j = p;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    if (j->tag) { thread_affinity_policy_data_t pol = { j->tag };
                  thread_policy_set(mach_thread_self(), THREAD_AFFINITY_POLICY, (thread_policy_t)&pol, 1); }
    prob *pr = j->pr;
    if (j->bar) xbar_wait(j->bar);
    j->t0 = tnow();
    size_t hb = pr->is_q8 ? 8 * 34 : HD * 2;
    for (int h = j->h0; h < j->h1; h++)
        sme_attn_partial(j->w, pr->Q + h * NR * HD, HD, pr->K + h * hb, pr->V + h * hb, pr->rs, pr->is_q8,
                         pr->N, 1.0f / 16, pr->bk, pr->O + h * NR * HD, pr->m + h * NR, pr->l + h * NR);
    j->t1 = tnow();
    return NULL;
}


typedef struct { sme_pipe *p; int role; int tag; xbar_t *bar; double t0, t1; } pjob;
static void *run_pjob(void *a) {
    pjob *j = a;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    if (j->tag) { thread_affinity_policy_data_t pol = { j->tag };
                  kern_return_t kr = thread_policy_set(mach_thread_self(), THREAD_AFFINITY_POLICY, (thread_policy_t)&pol, 1);
                  static _Atomic int once; if (kr != KERN_SUCCESS && !atomic_exchange(&once, 1)) fprintf(stderr, "affinity tag: kr=%d (not supported)\n", kr); }
    xbar_wait(j->bar);
    j->t0 = tnow();
    if (j->role == -1) { if (j->p->sm_off) sme_pipe_worker2(j->p); else sme_pipe_worker(j->p); }
    else if (j->role == -2) sme_pipe_softmax_helper(j->p);
    else sme_pipe_helper(j->p, j->role);
    j->t1 = tnow();
    return NULL;
}
// nw SME workers (heads split), nh helpers each
static int sm_off = 0;
static double run_pipe(prob *pr, int nw, int nh, sme_pipe **pp, int tag) {
    pthread_t th[64]; pjob jb[64]; xbar_t bar; int nt = nw * (1 + nh + (sm_off ? 1 : 0)); xbar_init(&bar, 0, nt);
    int k = 0;
    for (int w = 0; w < nw; w++) {
        sme_pipe *p = pp[w]; sme_pipe_reset(p);
        p->Q = pr->Q; p->qrs = HD; p->qhs = NR * HD; p->K = pr->K; p->V = pr->V; p->rs = pr->rs;
        p->hb = pr->is_q8 ? 8 * 34 : HD * 2; p->is_q8 = pr->is_q8; p->nk = pr->N; p->bk = pr->bk;
        p->h0 = w * 4 / nw; p->h1 = (w + 1) * 4 / nw; p->scale = 1.0f / 16; p->O = pr->O; p->m = pr->m; p->l = pr->l; p->nh = nh;
        p->sm_off = sm_off;
        for (int r = sm_off ? -2 : -1; r < nh; r++) { jb[k] = (pjob){ p, r, tag ? w + 1 : 0, &bar, 0, 0 }; pthread_create(&th[k], 0, run_pjob, &jb[k]); k++; }
    }
    double a = 1e30, b = 0;
    for (int t = 0; t < nt; t++) { pthread_join(th[t], 0); if (jb[t].t0 < a) a = jb[t].t0; if (jb[t].t1 > b) b = jb[t].t1; }
    return b - a;
}

static double run(prob *pr, int nthr, sme_ws **ws, int tag) {
    pthread_t th[8]; job jb[8]; xbar_t bar; xbar_init(&bar, 0, nthr);
    for (int t = 0; t < nthr; t++) {
        jb[t] = (job){ pr, t * 4 / nthr, (t + 1) * 4 / nthr, ws[t], tag ? t + 1 : 0, &bar, 0, 0 };
        pthread_create(&th[t], 0, run_job, &jb[t]);
    }
    double a = 1e30, b = 0;
    for (int t = 0; t < nthr; t++) { pthread_join(th[t], 0); if (jb[t].t0 < a) a = jb[t].t0; if (jb[t].t1 > b) b = jb[t].t1; }
    return b - a;
}

// fp32 reference for head h; returns max |O/l - ref| / max|ref|
static void reference(prob *pr, int h, double *err_o, double *err_m, double *err_l) {
    int N = pr->N; float *s = malloc(sizeof(float) * N); float *kf = malloc(sizeof(float) * HD), *o = malloc(sizeof(float) * HD);
    double eo = 0, em = 0, el = 0;
    for (int r = 0; r < NR; r++) {
        const float *q = pr->Q + (h * NR + r) * HD;
        float mx = -INFINITY;
        for (int j = 0; j < N; j++) {
            const uint8_t *row = pr->K + (size_t)j * pr->rs;
            double acc = 0;
            if (pr->is_q8) {
                const uint8_t *hb = row + h * 8 * 34;
                for (int b = 0; b < 8; b++) { f16 d; memcpy(&d, hb + b * 34, 2);
                    for (int i = 0; i < 32; i++) acc += (double)q[b * 32 + i] * ((int8_t)hb[b * 34 + 2 + i] * (float)d); }
            } else { const f16 *k = (const f16 *)(row + h * HD * 2); for (int i = 0; i < HD; i++) acc += (double)q[i] * (float)k[i]; }
            s[j] = acc / 16.0; if (s[j] > mx) mx = s[j];
        }
        double L = 0; double oacc[HD] = {0};
        for (int j = 0; j < N; j++) {
            double p = exp(s[j] - mx); L += p;
            const uint8_t *row = pr->V + (size_t)j * pr->rs;
            if (pr->is_q8) { const uint8_t *hb = row + h * 8 * 34;
                for (int b = 0; b < 8; b++) { f16 d; memcpy(&d, hb + b * 34, 2);
                    for (int i = 0; i < 32; i++) oacc[b * 32 + i] += p * ((int8_t)hb[b * 34 + 2 + i] * (float)d); } }
            else { const f16 *v = (const f16 *)(row + h * HD * 2); for (int i = 0; i < HD; i++) oacc[i] += p * (float)v[i]; }
        }
        double omax = 0; for (int i = 0; i < HD; i++) if (fabs(oacc[i] / L) > omax) omax = fabs(oacc[i] / L);
        float km = pr->m[h * NR + r], kl = pr->l[h * NR + r];
        // kernel l is relative to its own m; bring to the reference's m
        double klr = kl * exp(km - mx);
        for (int i = 0; i < HD; i++) {
            double e = fabs(pr->O[(h * NR + r) * HD + i] / kl - oacc[i] / L) / omax; if (e > eo) eo = e; }
        if (fabs(km - mx) > em) em = fabs(km - mx);
        if (fabs(klr - L) / L > el) el = fabs(klr - L) / L;
    }
    *err_o = eo; *err_m = em; *err_l = el; free(s); free(kf); free(o);
}

static void fill(prob *pr, uint64_t seed, float kscale) {
    uint64_t s = seed; int N = pr->N;
    for (int i = 0; i < 4 * NR * HD; i++) pr->Q[i] = frand(&s) * 1.7f;
    for (int which = 0; which < 2; which++) {
        uint8_t *X = which ? pr->V : pr->K;
        for (int j = 0; j < N; j++) for (int h = 0; h < 4; h++) {
            float x[HD]; for (int i = 0; i < HD; i++) x[i] = frand(&s) * (which ? 1.0f : kscale);
            if (!pr->is_q8) { uint16_t *o = (uint16_t *)(X + (size_t)j * pr->rs + h * HD * 2); for (int i = 0; i < HD; i++) o[i] = f2h(x[i]); }
            else { uint8_t *o = X + (size_t)j * pr->rs + h * 8 * 34;
                for (int b = 0; b < 8; b++) { float am = 0; for (int i = 0; i < 32; i++) if (fabsf(x[b * 32 + i]) > am) am = fabsf(x[b * 32 + i]);
                    float d = am / 127; uint16_t dh = f2h(d); memcpy(o + b * 34, &dh, 2); float id = d ? 1 / d : 0;
                    for (int i = 0; i < 32; i++) o[b * 34 + 2 + i] = (uint8_t)(int8_t)lrintf(x[b * 32 + i] * id); } }
        }
    }
}

#include <sys/sysctl.h>
// 1 if this CPU can run the kernels: FEAT_SME2 (SME instructions SIGILL without it) and a 512-bit streaming
// vector length (the tile layout assumes 16 fp32 lanes, as on the M4: hw.optional.arm.sme_max_svl_b = 64).
int sme2_available(void) {
    int v = 0, svl = 0; size_t n = sizeof v, n2 = sizeof svl;
    if (sysctlbyname("hw.optional.arm.FEAT_SME2", &v, &n, NULL, 0) != 0 || !v) return 0;
    if (sysctlbyname("hw.optional.arm.sme_max_svl_b", &svl, &n2, NULL, 0) != 0) return 0;
    return svl == 64;
}

// Same as the CLI; callable from Swift (iOS app) as sme_bench(argc, argv). Output goes to stdout.
int sme_bench(int argc, char **argv) {
    if (!sme2_available()) { printf("FEAT_SME2 = 0 or streaming vector length != 512 bit (sysctl hw.optional.arm.sme_max_svl_b): not running\n"); return 2; }
    const char *mode = argc > 1 ? argv[1] : "attn";
    if (!strcmp(mode, "qkcmp")) {     // in-cache QK only: fp16 FMOPA vs int8 SMOPA per-block (+ NEON combine)
        int bk = 512, it = argc > 2 ? atoi(argv[2]) : 2000;
        f16 *Qp = aligned_alloc(128, 128 * 96 * 2), *Kp = aligned_alloc(128, bk * 512);
        int8_t *Qi = aligned_alloc(128, 64 * 192), *Ki = aligned_alloc(128, bk * 256);
        int32_t *T = aligned_alloc(128, (size_t)bk * 8 * NR * 4); float *St = aligned_alloc(128, bk * NR * 4);
        float *dq = malloc(8 * NR * 4), *dk = malloc(bk * 8 * 4);
        for (int i = 0; i < 128 * 96; i++) Qp[i] = (f16)(0.01f * (i % 7)); for (int i = 0; i < bk * 256; i++) Kp[i] = (f16)(0.01f * (i % 5));
        for (int i = 0; i < 64 * 192; i++) Qi[i] = i % 11 - 5; for (int i = 0; i < bk * 256; i++) Ki[i] = i % 13 - 6;
        for (int i = 0; i < 8 * NR; i++) dq[i] = 0.01f; for (int i = 0; i < bk * 8; i++) dk[i] = 0.02f;
        double fl = (double)NR * bk * HD * 2 * it;
        for (int rep = 0; rep < 3; rep++) {
            double t0 = tnow(); for (int i = 0; i < it; i++) sme_qk(Qp, Kp, St, bk);
            double t1 = tnow(); for (int i = 0; i < it; i++) sme_qk_i8blk(Qi, Ki, T, bk);
            double t2 = tnow(); for (int i = 0; i < it; i++) i8_combine(T, dq, dk, St, bk);
            double t3 = tnow();
            printf("QK 48x256x%d, %d iters: fp16 FMOPA %.1f ns/key (%.0f GFLOPS) | int8 SMOPA per-block %.1f ns/key (%.0f GOPS eff) | NEON scale-combine %.1f ns/key\n",
                   bk, it, (t1 - t0) / it / bk * 1e9, fl / (t1 - t0) / 1e9, (t2 - t1) / it / bk * 1e9, fl / (t2 - t1) / 1e9, (t3 - t2) / it / bk * 1e9);
        }
        return 0;
    }
    if (!strcmp(mode, "peak")) {        // peak <kind 0=f16w 1=f32 2=s8> <nthreads>
        int kind = argc > 2 ? atoi(argv[2]) : 0, nt = argc > 3 ? atoi(argv[3]) : 1;
        long iters = 20000000; double flop_per = kind == 0 ? 1024 : kind == 1 ? 512 : 2048;
        for (int rep = 0; rep < 3; rep++) {
            pthread_t th[16]; peak_arg a[16];
            double t0 = tnow();
            for (int t = 0; t < nt; t++) { a[t] = (peak_arg){ kind, iters, 0 }; pthread_create(&th[t], 0, peak_thr, &a[t]); }
            for (int t = 0; t < nt; t++) pthread_join(th[t], 0);
            double wall = tnow() - t0, sum = 0;
            for (int t = 0; t < nt; t++) sum += 4.0 * iters * flop_per / a[t].t;
            printf("peak kind=%s threads=%d wall=%.3fs  per-thread-sum=%.0f GFLOPS  aggregate(wall)=%.0f GFLOPS\n",
                   kind == 0 ? "f16->f32" : kind == 1 ? "f32" : "s8->s32", nt, wall, sum / 1e9, nt * 4.0 * iters * flop_per / wall / 1e9);
        }
        return 0;
    }
    // attn <N> <q8 0/1> <threads> <reps> [bk] [check 0/1] [tag 0/1] [timing 0/1] [kscale]
    int N = argc > 2 ? atoi(argv[2]) : 16384, is_q8 = argc > 3 ? atoi(argv[3]) : 0, nthr = argc > 4 ? atoi(argv[4]) : 1;
    int reps = argc > 5 ? atoi(argv[5]) : 5, bk = argc > 6 ? atoi(argv[6]) : 512, check = argc > 7 ? atoi(argv[7]) : 0;
    int tag = argc > 8 ? atoi(argv[8]) : 0; sme_timing = argc > 9 ? atoi(argv[9]) : 0;
    float kscale = argc > 10 ? atof(argv[10]) : 1.0f;
    prob pr = { N, is_q8, bk, is_q8 ? 4 * 8 * 34 : 4 * HD * 2 };
    pr.K = aligned_alloc(16384, (pr.rs * N + 16383) / 16384 * 16384); pr.V = aligned_alloc(16384, (pr.rs * N + 16383) / 16384 * 16384);
    pr.Q = malloc(sizeof(float) * 4 * NR * HD); pr.O = malloc(sizeof(float) * 4 * NR * HD);
    pr.m = malloc(sizeof(float) * 4 * NR); pr.l = malloc(sizeof(float) * 4 * NR);
    fill(&pr, 12345, kscale);
    sme_ws *ws[8]; for (int t = 0; t < 8; t++) ws[t] = sme_ws_new();
    double flop = 4.0 * NR * (double)N * HD * 4;   // 4 heads, QK + PV, 2 flop/MAC
    int nhelp = argc > 11 ? atoi(argv[11]) : 0;   // >0: pipelined, nthr SME workers with nhelp helpers each
    sm_off = argc > 12 ? atoi(argv[12]) : 0;       // 1: + one softmax helper per worker
    sme_pipe *pp[8]; if (nhelp) for (int t = 0; t < nthr; t++) pp[t] = sme_pipe_new(nhelp);
    double best = 1e30, times[64];
    for (int r = 0; r < reps; r++) {   // reps > 64: only the first 64 are listed (used as a CPU load loop)
        double t = nhelp ? run_pipe(&pr, nthr, nhelp, pp, tag) : run(&pr, nthr, ws, tag); if (t < best) best = t;
        if (r < 64) times[r] = t;
    }
    printf("attn N=%d kv=%s sme_threads=%d pack_helpers/worker=%d softmax_helper=%d bk=%d tag=%d  times_ms:", N, is_q8 ? "q8_0" : "f16", nthr, nhelp, sm_off, bk, tag);
    for (int r = 0; r < reps && r < 64; r++) printf(" %.2f", times[r] * 1e3);
    printf("\n  best %.3f ms/layer(4 heads) = %.0f GFLOPS\n", best * 1e3, flop / best / 1e9);
    if (sme_timing) printf("  phases (sum over reps, all threads) ms: pack %.1f  qk %.1f  softmax %.1f  pv %.1f  oupd %.1f\n",
                           sme_t_pack * 1e3, sme_t_qk * 1e3, sme_t_sm * 1e3, sme_t_pv * 1e3, sme_t_upd * 1e3);
    if (check) {
        double wo = 0, wm = 0, wl = 0;
        for (int h = 0; h < 4; h++) { double eo, em, el; reference(&pr, h, &eo, &em, &el); if (eo > wo) wo = eo; if (em > wm) wm = em; if (el > wl) wl = el; }
        printf("  check vs fp64-accum reference: max|O/l - ref|/max|ref| = %.2e   max|m-ref| = %.2e   max rel l err = %.2e\n", wo, wm, wl);
    }
    return 0;
}
#ifndef SME_BENCH_NO_MAIN
int main(int argc, char **argv) { return sme_bench(argc, argv); }
#endif
#endif
