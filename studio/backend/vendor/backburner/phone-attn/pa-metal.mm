// pa-metal.mm - phone-attn GPU engine: partial attention over phone-held q8_0 pages on the phone GPU (Metal).
// Included into ios/Sidecar/Sidecar/RPCBridge.mm after phone-attn.h (one ObjC++ translation unit).
//
// Kernel pa_attn_q8: one threadgroup = one KV head x one chunk of keys; 6 simdgroups, simdgroup s owns query rows
// s*8..s*8+7 of that head (48 rows = 8 tokens x GQA 6). Per 32-key block: dequant K into threadgroup memory,
// S = Q K^T (simdgroup 8x8 matmuls), online softmax (4 lanes per row), O = diag(alpha) O + P V.
// Writes the chunk's unnormalised (O, m, l). Kernel pa_merge folds all chunks into one partial per row.
#import <Metal/Metal.h>
#include <mach/mach_time.h>

namespace pa {

static const char * k_pa_metal_src = R"MSL(
#include <metal_stdlib>
using namespace metal;

#define NR 48
#define HD 256
#define BK 32
#define NSG 6

struct pa_args { uint rs; uint hb; uint nk; uint chunk; uint slot0; uint nkv; };

kernel void pa_attn_q8(
        device const half  * Q   [[buffer(0)]],   // [nkv][48][256], f16(q*scale)
        device const uchar * K   [[buffer(1)]],   // page: key j, head h at K + j*rs + h*hb (q8_0: 8 x (f16 d, 32 x int8))
        device const uchar * V   [[buffer(2)]],
        device float       * Op  [[buffer(3)]],   // [slot][nkv*48][256]
        device float       * ML  [[buffer(4)]],   // [slot][nkv*48][2]
        constant pa_args   & a   [[buffer(5)]],
        threadgroup half   * sh  [[threadgroup(0)]],
        uint2  tg   [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    // 12 simdgroups: pair rb = sg/2 owns query rows rb*8..rb*8+7; hf = sg%2 owns head dims hf*128..hf*128+127
    // (its half of Q stays in registers; it computes the partial scores over those dims and its half of O)
    const uint c = tg.x, h = tg.y;
    const uint j0 = c * a.chunk, j1 = min(j0 + a.chunk, a.nk);
    const int rb = sg / 2, hf = sg % 2;

    threadgroup half  * KV = sh;                                                   // [BK][HD]
    threadgroup float * S  = (threadgroup float *) (sh + BK*HD) + rb*8*BK;         // [8][BK] per pair
    threadgroup half  * P  = (threadgroup half *) ((threadgroup float *) (sh + BK*HD) + NSG*8*BK) + rb*8*BK;
    threadgroup float * D  = (threadgroup float *) ((threadgroup half *) ((threadgroup float *) (sh + BK*HD) + NSG*8*BK) + NSG*8*BK) + rb*64;

    simdgroup_half8x8 q[16];
    device const half * Qh = Q + ((size_t) h*NR + rb*8)*HD + hf*128;
    for (int k = 0; k < 16; k++) simdgroup_load(q[k], Qh + k*8, HD);
    simdgroup_float8x8 o[16];
    for (int k = 0; k < 16; k++) o[k] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const int row = lane / 4, part = lane % 4;       // 4 lanes per query row, 8 keys each
    float m_row = -INFINITY, l_row = 0.0f;
    if (hf == 0) for (int i = lane; i < 64; i += 32) D[i] = 0.0f;

    for (uint jb = j0; jb < j1; jb += BK) {
        // ---- K block -> half (256 items, 384 threads)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < BK*8; i += NSG*2*32) {
            const uint key = i / 8, b = i % 8;
            device const uchar * blk = K + (size_t) (jb + key)*a.rs + h*a.hb + b*34;
            const half d = *((device const half *) blk);
            device const packed_char4 * qs = (device const packed_char4 *) (blk + 2);
            threadgroup half4 * dst = (threadgroup half4 *) (KV + key*HD + b*32);
            for (int t = 0; t < 8; t++) dst[t] = d * half4(char4(qs[t]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- partial S over this simdgroup's 128 dims, all BK keys
        simdgroup_float8x8 s[BK/8];
        for (int t = 0; t < BK/8; t++) {
            s[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int k = 0; k < 16; k++) {
                simdgroup_half8x8 kt;
                simdgroup_load(kt, KV + (t*8)*HD + hf*128 + k*8, HD, ulong2(0, 0), true);   // [8 dims][8 keys]
                simdgroup_multiply_accumulate(s[t], q[k], kt, s[t]);
            }
        }
        if (hf == 1) for (int t = 0; t < BK/8; t++) simdgroup_store(s[t], S + t*8, BK);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (hf == 0) {
            const simdgroup_float8x8 eye = simdgroup_float8x8(1.0f);   // diagonal matrix
            for (int t = 0; t < BK/8; t++) {
                simdgroup_float8x8 o2; simdgroup_load(o2, S + t*8, BK);
                simdgroup_multiply_accumulate(s[t], eye, o2, s[t]);   // s += o2
                simdgroup_store(s[t], S + t*8, BK);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- online softmax (both simdgroups of a pair track the same m, l; hf 0 writes P and alpha)
        float sv[8]; float mx = -INFINITY;
        for (int i = 0; i < 8; i++) { sv[i] = S[row*BK + part*8 + i]; mx = max(mx, sv[i]); }
        mx = max(mx, simd_shuffle_xor(mx, 1)); mx = max(mx, simd_shuffle_xor(mx, 2));
        const float m_new = max(m_row, mx), alpha = exp(m_row - m_new);
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) {
            const float p = exp(sv[i] - m_new); sum += p;
            if (hf == 0) P[row*BK + part*8 + i] = (half) p;
        }
        sum += simd_shuffle_xor(sum, 1); sum += simd_shuffle_xor(sum, 2);
        l_row = l_row*alpha + sum; m_row = m_new;
        if (hf == 0 && part == 0) D[row*8 + row] = alpha;
        // ---- V block -> half (barrier: everyone is done with K and S; P and alpha are written)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < BK*8; i += NSG*2*32) {
            const uint key = i / 8, b = i % 8;
            device const uchar * blk = V + (size_t) (jb + key)*a.rs + h*a.hb + b*34;
            const half d = *((device const half *) blk);
            device const packed_char4 * qs = (device const packed_char4 *) (blk + 2);
            threadgroup half4 * dst = (threadgroup half4 *) (KV + key*HD + b*32);
            for (int t = 0; t < 8; t++) dst[t] = d * half4(char4(qs[t]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dm; simdgroup_load(dm, D, 8);
        for (int k = 0; k < 16; k++) simdgroup_multiply(o[k], dm, o[k]);
        // ---- O[:, hf half] += P V
        simdgroup_half8x8 pm[BK/8];
        for (int kk = 0; kk < BK/8; kk++) simdgroup_load(pm[kk], P + kk*8, BK);
        for (int n = 0; n < 16; n++) {
            for (int kk = 0; kk < BK/8; kk++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, KV + (kk*8)*HD + hf*128 + n*8, HD);
                simdgroup_multiply_accumulate(o[n], pm[kk], vm, o[n]);
            }
        }
    }
    const uint slot = a.slot0 + c;
    const size_t r0 = (size_t) slot*a.nkv*NR + (size_t) h*NR + rb*8;
    for (int k = 0; k < 16; k++) simdgroup_store(o[k], Op + r0*HD + hf*128 + k*8, HD);
    if (part == 0 && hf == 0) { ML[(r0 + row)*2 + 0] = m_row; ML[(r0 + row)*2 + 1] = l_row; }
}

// one threadgroup per row (nkv*48), 256 threads = head dim
// keep-warm (PA_GPU_WARM_US): a trivial dispatch between decode calls so the GPU is still clocked up when the next ATTN arrives
kernel void pa_warm(device uint * x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] += 1u; }

kernel void pa_merge(
        device const float * Op [[buffer(0)]],
        device const float * ML [[buffer(1)]],
        device float       * O  [[buffer(2)]],
        device float       * MLo[[buffer(3)]],
        constant uint4     & a  [[buffer(4)]],   // (n_slots, nkv, rows per head in the partials, 48)
        uint   r [[threadgroup_position_in_grid]],
        ushort d [[thread_position_in_threadgroup]]) {
    const uint h = r / a.w, i = r % a.w;           // output row r = head h, row i
    float M = -INFINITY;
    for (uint c = 0; c < a.x; c++) M = max(M, ML[(((size_t) c*a.y + h)*a.z + i)*2]);
    float L = 0.0f, acc = 0.0f;
    for (uint c = 0; c < a.x; c++) {
        const size_t sr = ((size_t) c*a.y + h)*a.z + i;
        const float mc = ML[sr*2];
        if (mc == -INFINITY) continue;
        const float w = exp(mc - M);
        L += w*ML[sr*2 + 1];
        acc += w*Op[sr*HD + d];
    }
    O[(size_t) r*HD + d] = acc;
    if (d == 0) { MLo[r*2] = M; MLo[r*2 + 1] = L; }
}
)MSL";


static const char * k_pa_na_src = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#define HD  256
#define MR  64    // query rows per KV head: 48 real (8 tokens x GQA 6) + 16 zero rows
#define NB  64    // keys per block
#ifndef PA_NSG
#define PA_NSG 4
#endif
#define NSG PA_NSG
#ifndef PA_RELAXED
#define PA_RELAXED false
#endif

struct pa_na_args { uint rsh; uint nk; uint chunk; uint slot0; uint nkv; };   // rsh: K/V row stride in halfs (nkv*256)
struct pa_naq_args { uint rs; uint hb; uint nk; uint chunk; uint slot0; uint nkv; uint q4; };

// one threadgroup = one KV head x one chunk of keys; K, V are f16 pages read directly by the neural accelerators
kernel void pa_attn_na(
        device const half  * Q   [[buffer(0)]],   // [nkv][64][256] f16(q*scale), rows >= 48 zero
        device const half  * K   [[buffer(1)]],   // key j, head h at K + j*rsh + h*256
        device const half  * V   [[buffer(2)]],
        device float       * Op  [[buffer(3)]],   // [slot][nkv][64][256]
        device float       * ML  [[buffer(4)]],   // [slot][nkv][64][2]
        constant pa_na_args & a  [[buffer(5)]],
        threadgroup char   * shm [[threadgroup(0)]],
        uint2  tg  [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const uint c = tg.x, h = tg.y;
    const uint j0 = c * a.chunk, j1 = min(j0 + a.chunk, a.nk);
    threadgroup float * S  = (threadgroup float *) shm;          // [MR][NB]
    threadgroup half  * P  = (threadgroup half *) (S + MR*NB);   // [MR][NB]
    threadgroup float * AL = (threadgroup float *) (P + MR*NB);  // [MR]

    constexpr auto dS = matmul2d_descriptor(MR, NB, HD, false, true, PA_RELAXED, matmul2d_descriptor::mode::multiply);
    constexpr auto dO = matmul2d_descriptor(MR, HD, NB, false, false, PA_RELAXED, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS, execution_simdgroups<NSG>> mmS;
    matmul2d<dO, execution_simdgroups<NSG>> mmO;

    auto tQ = tensor((device half *) (Q + (size_t) h*MR*HD), dextents<int32_t, 2>(HD, MR), array<int, 2>({1, HD}));
    auto tS = tensor(S, dextents<int32_t, 2>(NB, MR), array<int, 2>({1, NB}));
    auto tP = tensor(P, dextents<int32_t, 2>(NB, MR), array<int, 2>({1, NB}));
    auto tK0 = tensor((device half *) (K + (size_t) j0*a.rsh + h*HD), dextents<int32_t, 2>(HD, NB), array<int, 2>({1, (int) a.rsh}));
    auto tV0 = tensor((device half *) (V + (size_t) j0*a.rsh + h*HD), dextents<int32_t, 2>(HD, NB), array<int, 2>({1, (int) a.rsh}));

    auto oT = mmO.get_destination_cooperative_tensor<decltype(tP), decltype(tV0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < oT.get_capacity(); ++i) { if (oT.is_valid_element(i)) oT[i] = 0.0f; }

    constexpr int TPR = NSG*32/MR, KPT = NB/TPR;   // threads per row, keys per thread
    const int row = tid / TPR, hf = tid % TPR;
    float m_row = -INFINITY, l_row = 0.0f;

    for (uint jb = j0; jb < j1; jb += NB) {
        auto tK = tensor((device half *) (K + (size_t) jb*a.rsh + h*HD), dextents<int32_t, 2>(HD, NB), array<int, 2>({1, (int) a.rsh}));
        auto sT = mmS.get_destination_cooperative_tensor<decltype(tQ), decltype(tK), float>();
        mmS.run(tQ, tK, sT);
        sT.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float mx = -INFINITY;
        for (int i = 0; i < KPT; i++) mx = max(mx, S[row*NB + hf*KPT + i]);
        for (int o = 1; o < TPR; o <<= 1) mx = max(mx, simd_shuffle_xor(mx, o));
        const float m_new = max(m_row, mx), alpha = exp(m_row - m_new);
        float sum = 0.0f;
        for (int i = 0; i < KPT; i++) {
            const float p = exp(S[row*NB + hf*KPT + i] - m_new);
            sum += p;
            P[row*NB + hf*KPT + i] = (half) p;
        }
        for (int o = 1; o < TPR; o <<= 1) sum += simd_shuffle_xor(sum, o);
        l_row = l_row*alpha + sum; m_row = m_new;
        if (hf == 0) AL[row] = alpha;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint16_t i = 0; i < oT.get_capacity(); ++i) {
            if (oT.is_valid_element(i)) { auto ids = oT.get_multidimensional_index(i); oT[i] *= AL[ids[1]]; }
        }
        auto tV = tensor((device half *) (V + (size_t) jb*a.rsh + h*HD), dextents<int32_t, 2>(HD, NB), array<int, 2>({1, (int) a.rsh}));
        mmO.run(tP, tV, oT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const size_t r0 = ((size_t) (a.slot0 + c)*a.nkv + h)*MR;
    auto tO = tensor(Op + r0*HD, dextents<int32_t, 2>(HD, MR), array<int, 2>({1, HD}));
    oT.store(tO);
    if (hf == 0) { ML[(r0 + row)*2 + 0] = m_row; ML[(r0 + row)*2 + 1] = l_row; }
}

// quantized pages (q8_0 or q4_0), dequantized per 32-key block into threadgroup memory, then fed to the neural accelerators
kernel void pa_attn_naq(
        device const half  * Q   [[buffer(0)]],   // [nkv][64][256] f16(q*scale), rows >= 48 zero
        device const uchar * K   [[buffer(1)]],   // key j, head h at K + j*rs + h*hb
        device const uchar * V   [[buffer(2)]],
        device float       * Op  [[buffer(3)]],   // [slot][nkv][64][256]
        device float       * ML  [[buffer(4)]],   // [slot][nkv][64][2]
        constant pa_naq_args & a [[buffer(5)]],
        threadgroup char   * shm [[threadgroup(0)]],
        uint2  tg  [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NBQ = 32;
    const uint c = tg.x, h = tg.y;
    const uint j0 = c * a.chunk, j1 = min(j0 + a.chunk, a.nk);
    threadgroup half  * KV = (threadgroup half *) shm;           // [NBQ][HD] K block, then V block
    threadgroup float * S  = (threadgroup float *) (KV + NBQ*HD); // [MR][NBQ]
    threadgroup half  * P  = (threadgroup half *) (S + MR*NBQ);   // [MR][NBQ]
    threadgroup float * AL = (threadgroup float *) (P + MR*NBQ);  // [MR]

    constexpr auto dS = matmul2d_descriptor(MR, NBQ, HD, false, true, PA_RELAXED, matmul2d_descriptor::mode::multiply);
    constexpr auto dO = matmul2d_descriptor(MR, HD, NBQ, false, false, PA_RELAXED, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS, execution_simdgroups<NSG>> mmS;
    matmul2d<dO, execution_simdgroups<NSG>> mmO;

    auto tQ  = tensor((device half *) (Q + (size_t) h*MR*HD), dextents<int32_t, 2>(HD, MR), array<int, 2>({1, HD}));
    auto tKV = tensor(KV, dextents<int32_t, 2>(HD, NBQ), array<int, 2>({1, HD}));
    auto tS  = tensor(S, dextents<int32_t, 2>(NBQ, MR), array<int, 2>({1, NBQ}));
    auto tP  = tensor(P, dextents<int32_t, 2>(NBQ, MR), array<int, 2>({1, NBQ}));

    auto oT = mmO.get_destination_cooperative_tensor<decltype(tP), decltype(tKV), float>();
    #pragma unroll
    for (uint16_t i = 0; i < oT.get_capacity(); ++i) { if (oT.is_valid_element(i)) oT[i] = 0.0f; }

    constexpr int TPR = NSG*32/MR, KPT = NBQ/TPR;
    const int row = tid / TPR, hf = tid % TPR;
    float m_row = -INFINITY, l_row = 0.0f;
    const uint nblk = HD/32;   // 32-value quant blocks per head row

    for (uint jb = j0; jb < j1; jb += NBQ) {
        // ---- dequantize the K block
        for (uint i = tid; i < NBQ*nblk; i += NSG*32) {
            const uint key = i / nblk, b = i % nblk;
            threadgroup half4 * dst = (threadgroup half4 *) (KV + key*HD + b*32);
            if (a.q4) {
                device const uchar * blk = K + (size_t) (jb + key)*a.rs + h*a.hb + b*18;
                const half d = *((device const half *) blk);
                for (int t = 0; t < 4; t++) {           // q4_0: byte k holds values k (low nibble) and k+16 (high nibble)
                    half4 lo, hi;
                    for (int u = 0; u < 4; u++) { const uchar q = blk[2 + t*4 + u]; lo[u] = (half) ((int) (q & 15) - 8); hi[u] = (half) ((int) (q >> 4) - 8); }
                    dst[t] = d*lo; dst[t + 4] = d*hi;
                }
            } else {
                device const uchar * blk = K + (size_t) (jb + key)*a.rs + h*a.hb + b*34;
                const half d = *((device const half *) blk);
                device const packed_char4 * qs = (device const packed_char4 *) (blk + 2);
                for (int t = 0; t < 8; t++) dst[t] = d * half4(char4(qs[t]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sT = mmS.get_destination_cooperative_tensor<decltype(tQ), decltype(tKV), float>();
        mmS.run(tQ, tKV, sT);
        sT.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float mx = -INFINITY;
        for (int i = 0; i < KPT; i++) mx = max(mx, S[row*NBQ + hf*KPT + i]);
        for (int o = 1; o < TPR; o <<= 1) mx = max(mx, simd_shuffle_xor(mx, o));
        const float m_new = max(m_row, mx), alpha = exp(m_row - m_new);
        float sum = 0.0f;
        for (int i = 0; i < KPT; i++) {
            const float p = exp(S[row*NBQ + hf*KPT + i] - m_new);
            sum += p;
            P[row*NBQ + hf*KPT + i] = (half) p;
        }
        for (int o = 1; o < TPR; o <<= 1) sum += simd_shuffle_xor(sum, o);
        l_row = l_row*alpha + sum; m_row = m_new;
        if (hf == 0) AL[row] = alpha;
        // ---- dequantize the V block into the same buffer (the S matmul is done with K)
        for (uint i = tid; i < NBQ*nblk; i += NSG*32) {
            const uint key = i / nblk, b = i % nblk;
            threadgroup half4 * dst = (threadgroup half4 *) (KV + key*HD + b*32);
            if (a.q4) {
                device const uchar * blk = V + (size_t) (jb + key)*a.rs + h*a.hb + b*18;
                const half d = *((device const half *) blk);
                for (int t = 0; t < 4; t++) {
                    half4 lo, hi;
                    for (int u = 0; u < 4; u++) { const uchar q = blk[2 + t*4 + u]; lo[u] = (half) ((int) (q & 15) - 8); hi[u] = (half) ((int) (q >> 4) - 8); }
                    dst[t] = d*lo; dst[t + 4] = d*hi;
                }
            } else {
                device const uchar * blk = V + (size_t) (jb + key)*a.rs + h*a.hb + b*34;
                const half d = *((device const half *) blk);
                device const packed_char4 * qs = (device const packed_char4 *) (blk + 2);
                for (int t = 0; t < 8; t++) dst[t] = d * half4(char4(qs[t]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint16_t i = 0; i < oT.get_capacity(); ++i) {
            if (oT.is_valid_element(i)) { auto ids = oT.get_multidimensional_index(i); oT[i] *= AL[ids[1]]; }
        }
        mmO.run(tP, tKV, oT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const size_t r0 = ((size_t) (a.slot0 + c)*a.nkv + h)*MR;
    auto tO = tensor(Op + r0*HD, dextents<int32_t, 2>(HD, MR), array<int, 2>({1, HD}));
    oT.store(tO);
    if (hf == 0) { ML[(r0 + row)*2 + 0] = m_row; ML[(r0 + row)*2 + 1] = l_row; }
}
// ---- prefill (ATTN_BIG) kernel: dense rows. Decode calls never use it.
// The ng groups' query rows of one KV head sit back to back (row g*48 + i; zero rows pad to qr), so a threadgroup's BMR rows are
// all real (the per-group kernels pad 48 -> 64) and one dequantized K/V block feeds BMR rows. grid: (qr/BMR, nkv, chunks).
#ifndef PA_BMR
#define PA_BMR 64
#endif
#ifndef PA_BNB
#define PA_BNB 32
#endif
#ifndef PA_BNSG
#define PA_BNSG 4
#endif
struct pa_big_args { uint rs; uint hb; uint nk; uint chunk; uint slot0; uint nkv; uint qr; uint q; };   // q: 0 f16, 1 q8_0, 2 q4_0

static inline void pa_big_load(threadgroup half * KV, device const uchar * X, constant pa_big_args & a, uint h, uint jb, uint j1, ushort tid) {
    constexpr uint nblk = HD/32;
    for (uint i = tid; i < PA_BNB*nblk; i += PA_BNSG*32) {
        const uint key = i / nblk, b = i % nblk;
        threadgroup half4 * dst = (threadgroup half4 *) (KV + key*HD + b*32);
        if (jb + key >= j1) { for (int t = 0; t < 8; t++) dst[t] = half4(0.0h); continue; }
        device const uchar * row = X + (size_t) (jb + key)*a.rs + h*a.hb;
        if (a.q == 2) {
            device const uchar * blk = row + b*18;
            const half d = *((device const half *) blk);
            for (int t = 0; t < 4; t++) {
                const uchar4 q = *((device const packed_uchar4 *) (blk + 2 + t*4));
                dst[t]     = d*(half4(q & uchar4(15)) - 8.0h);
                dst[t + 4] = d*(half4(q >> uchar4(4)) - 8.0h);
            }
        } else if (a.q == 1) {
            device const uchar * blk = row + b*34;
            const half d = *((device const half *) blk);
            device const packed_char4 * qs = (device const packed_char4 *) (blk + 2);
            for (int t = 0; t < 8; t++) dst[t] = d * half4(char4(qs[t]));
        } else {
            device const half4 * src = (device const half4 *) (row + b*64);
            for (int t = 0; t < 8; t++) dst[t] = src[t];
        }
    }
}

kernel void pa_attn_big(
        device const half  * Q   [[buffer(0)]],   // [nkv][qr][256] f16(q*scale), dense rows
        device const uchar * K   [[buffer(1)]],
        device const uchar * V   [[buffer(2)]],
        device float       * Op  [[buffer(3)]],   // [slot][nkv][qr][256]
        device float       * ML  [[buffer(4)]],   // [slot][nkv][qr][2]
        constant pa_big_args & a [[buffer(5)]],
        device const uint  * live [[buffer(6)]],  // 0: a cancelled pre-armed call, do nothing
        threadgroup char   * shm [[threadgroup(0)]],
        uint3  tg  [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    if (live[0] == 0) return;
    constexpr int MRB = PA_BMR, NBB = PA_BNB, NT = PA_BNSG*32;
    const uint t = tg.x, h = tg.y, c = tg.z;
    const uint j0 = c * a.chunk, j1 = min(j0 + a.chunk, a.nk);
    threadgroup half  * KV = (threadgroup half *) shm;            // [NBB][HD]
    threadgroup float * S  = (threadgroup float *) (KV + NBB*HD);  // [MRB][NBB]
    threadgroup half  * P  = (threadgroup half *) (S + MRB*NBB);   // [MRB][NBB]
    threadgroup float * AL = (threadgroup float *) (P + MRB*NBB);  // [MRB]

    constexpr auto dS = matmul2d_descriptor(MRB, NBB, HD, false, true, PA_RELAXED, matmul2d_descriptor::mode::multiply);
    constexpr auto dO = matmul2d_descriptor(MRB, HD, NBB, false, false, PA_RELAXED, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS, execution_simdgroups<PA_BNSG>> mmS;
    matmul2d<dO, execution_simdgroups<PA_BNSG>> mmO;

    auto tQ  = tensor((device half *) (Q + ((size_t) h*a.qr + t*MRB)*HD), dextents<int32_t, 2>(HD, MRB), array<int, 2>({1, HD}));
    auto tKV = tensor(KV, dextents<int32_t, 2>(HD, NBB), array<int, 2>({1, HD}));
    auto tS  = tensor(S, dextents<int32_t, 2>(NBB, MRB), array<int, 2>({1, NBB}));
    auto tP  = tensor(P, dextents<int32_t, 2>(NBB, MRB), array<int, 2>({1, NBB}));

    auto oT = mmO.get_destination_cooperative_tensor<decltype(tP), decltype(tKV), float>();
    #pragma unroll
    for (uint16_t i = 0; i < oT.get_capacity(); ++i) { if (oT.is_valid_element(i)) oT[i] = 0.0f; }

    // RPT rows per thread when threads < rows, else TPR threads per row
    constexpr int TPR = NT >= MRB ? NT/MRB : 1, RPT = NT >= MRB ? 1 : MRB/NT, KPT = NBB/TPR;
    float m_row[RPT], l_row[RPT];
    for (int r = 0; r < RPT; r++) { m_row[r] = -INFINITY; l_row[r] = 0.0f; }

    for (uint jb = j0; jb < j1; jb += NBB) {
        pa_big_load(KV, K, a, h, jb, j1, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sT = mmS.get_destination_cooperative_tensor<decltype(tQ), decltype(tKV), float>();
        mmS.run(tQ, tKV, sT);
        sT.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int nvalid = (int) min((uint) NBB, j1 - jb);
        for (int r = 0; r < RPT; r++) {
            const int row = TPR > 1 ? tid / TPR : tid + r*NT, hf = TPR > 1 ? tid % TPR : 0;
            float mx = -INFINITY;
            for (int i = 0; i < KPT; i++) { const int k = hf*KPT + i; if (k < nvalid) mx = max(mx, S[row*NBB + k]); }
            for (int o = 1; o < TPR; o <<= 1) mx = max(mx, simd_shuffle_xor(mx, o));
            const float m_new = max(m_row[r], mx), alpha = exp(m_row[r] - m_new);
            float sum = 0.0f;
            for (int i = 0; i < KPT; i++) {
                const int k = hf*KPT + i;
                const float p = k < nvalid ? exp(S[row*NBB + k] - m_new) : 0.0f;
                sum += p;
                P[row*NBB + k] = (half) p;
            }
            for (int o = 1; o < TPR; o <<= 1) sum += simd_shuffle_xor(sum, o);
            l_row[r] = l_row[r]*alpha + sum; m_row[r] = m_new;
            if (hf == 0) AL[row] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // KV is overwritten with V; S is read
        pa_big_load(KV, V, a, h, jb, j1, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint16_t i = 0; i < oT.get_capacity(); ++i) {
            if (oT.is_valid_element(i)) { auto ids = oT.get_multidimensional_index(i); oT[i] *= AL[ids[1]]; }
        }
        mmO.run(tP, tKV, oT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const size_t r0 = ((size_t) (a.slot0 + c)*a.nkv + h)*a.qr + t*MRB;
    auto tO = tensor(Op + r0*HD, dextents<int32_t, 2>(HD, MRB), array<int, 2>({1, HD}));
    oT.store(tO);
    for (int r = 0; r < RPT; r++) {
        const int row = TPR > 1 ? tid / TPR : tid + r*NT, hf = TPR > 1 ? tid % TPR : 0;
        if (hf == 0) { ML[(r0 + row)*2 + 0] = m_row[r]; ML[(r0 + row)*2 + 1] = l_row[r]; }
    }
}

// folds the big kernel's chunks: output row R = group g, head h, row i (the per-group layout end() returns)
kernel void pa_merge_big(
        device const float * Op [[buffer(0)]],
        device const float * ML [[buffer(1)]],
        device float       * O  [[buffer(2)]],
        device float       * MLo[[buffer(3)]],
        constant uint4     & a  [[buffer(4)]],   // (n_slots, nkv, qr, fold (bit 0: O / MLo hold earlier pages' result) |
                                                 //  ntok << 8: one group packed with only its ntok tokens' rows, g*ntok + t)
        device const uint  * live [[buffer(5)]],
        uint   R [[threadgroup_position_in_grid]],
        ushort d [[thread_position_in_threadgroup]]) {
    if (live[0] == 0) return;
    const uint rows = a.y*48, g = R / rows, h = (R % rows) / 48, i = R % 48, nt = a.w >> 8;
    if (nt && i % 8 >= nt) {   // a token the call doesn't have: no keys
        O[(size_t) R*HD + d] = 0.0f;
        if (d == 0) { MLo[R*2] = -INFINITY; MLo[R*2 + 1] = 0.0f; }
        return;
    }
    const uint dr = nt ? (i / 8)*nt + i % 8 : g*48 + i;
    const float M0 = (a.w & 1) ? MLo[R*2] : -INFINITY;
    float M = M0;
    for (uint c = 0; c < a.x; c++) M = max(M, ML[(((size_t) c*a.y + h)*a.z + dr)*2]);
    float L = 0.0f, acc = 0.0f;
    if (M0 != -INFINITY) { const float w = exp(M0 - M); L = w*MLo[R*2 + 1]; acc = w*O[(size_t) R*HD + d]; }
    threadgroup_barrier(mem_flags::mem_device);   // every thread has read MLo[R] before thread 0 overwrites it
    for (uint c = 0; c < a.x; c++) {
        const size_t sr = ((size_t) c*a.y + h)*a.z + dr;
        const float mc = ML[sr*2];
        if (mc == -INFINITY) continue;
        const float w = exp(mc - M);
        L += w*ML[sr*2 + 1];
        acc += w*Op[sr*HD + d];
    }
    O[(size_t) R*HD + d] = acc;
    if (d == 0) { MLo[R*2] = M; MLo[R*2 + 1] = L; }
}
)MSL";

struct pa_args_host { uint32_t rs, hb, nk, chunk, slot0, nkv; };
struct pa_na_args_host { uint32_t rsh, nk, chunk, slot0, nkv; };
struct pa_naq_args_host { uint32_t rs, hb, nk, chunk, slot0, nkv, q4; };
struct pa_big_args_host { uint32_t rs, hb, nk, chunk, slot0, nkv, qr, q; };

class metal_engine : public engine {
public:
    std::string error;
    double last_exec_ms = 0, last_sched_ms = 0;   // GPU execution time vs the rest of commit -> completed

    bool init() {
        dev_ = MTLCreateSystemDefaultDevice();
        if (!dev_) { error = "no Metal device"; return false; }
        queue_ = [dev_ newCommandQueue];
        NSError * e = nil;
        id<MTLLibrary> lib = [dev_ newLibraryWithSource:[NSString stringWithUTF8String:k_pa_metal_src] options:nil error:&e];
        if (!lib) { error = std::string("Metal compile: ") + (e ? e.localizedDescription.UTF8String : "?"); return false; }
        pso_attn_  = [dev_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pa_attn_q8"] error:&e];
        pso_merge_ = [dev_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pa_merge"] error:&e];
        pso_warm_  = [dev_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pa_warm"] error:&e];
        if (!pso_attn_ || !pso_merge_) { error = std::string("Metal pipeline: ") + (e ? e.localizedDescription.UTF8String : "?"); return false; }
        // neural-accelerator kernel (Metal 4 tensor ops): optional, used for f16 pages
        if (@available(iOS 26.0, macOS 26.0, *)) {
            if (getenv("PA_NA_DISABLE") == nullptr) {
                pso_na_ = na_pipeline(0);
            }
        }
        return true;
    }
    std::string na_error;
    bool has_na() const { return pso_na_ != nil; }
    uint32_t na_variant_ = 0, na_nsg_ = 4;
    // variant bits: 1 = relaxed-precision tensor matmuls, 2 = 8 simdgroups per threadgroup
    id<MTLComputePipelineState> na_pipeline(uint32_t v) {
        if (@available(iOS 26.0, macOS 26.0, *)) {
            MTLCompileOptions * opt = [MTLCompileOptions new];
            opt.languageVersion = MTLLanguageVersion4_0;
            opt.preprocessorMacros = @{ @"PA_RELAXED" : (v & 1) ? @"true" : @"false", @"PA_NSG" : (v & 2) ? @"8" : @"4" };
            NSError * e2 = nil;
            id<MTLLibrary> lib2 = [dev_ newLibraryWithSource:na_source() options:opt error:&e2];
            id<MTLComputePipelineState> p = nil;
            if (lib2) p = [dev_ newComputePipelineStateWithFunction:[lib2 newFunctionWithName:@"pa_attn_na"] error:&e2];
            if (lib2 && p) pso_naq_ = [dev_ newComputePipelineStateWithFunction:[lib2 newFunctionWithName:@"pa_attn_naq"] error:&e2];
            if (p && !pso_naq_) na_error = std::string("NAQ kernel: ") + (e2 ? e2.localizedDescription.UTF8String : "?");
            if (!p) na_error = std::string("NA kernel v") + std::to_string(v) + ": " + (e2 ? e2.localizedDescription.UTF8String : "?");
            else { na_variant_ = v; na_nsg_ = (v & 2) ? 8 : 4; }
            return p;
        }
        return nil;
    }

    // PA_NA_SRC=file (relative: the app's Documents): the NA kernels' source from a file instead of the built-in copy, so a
    // kernel change can be tried on the phone with phone-push.py + relaunch, no reinstall
    NSString * na_source() {
        if (const char * f = getenv("PA_NA_SRC")) {
            NSString * path = [NSString stringWithUTF8String:f];
            if (![path hasPrefix:@"/"]) path = [[NSHomeDirectory() stringByAppendingPathComponent:@"Documents"] stringByAppendingPathComponent:path];
            NSString * src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
            if (src.length > 0) { na_src_note_ = "src=file"; return src; }
            na_src_note_ = "src=file-missing";
        }
        return [NSString stringWithUTF8String:k_pa_na_src];
    }
    std::string na_src_note_ = "src=builtin";
    // the prefill kernel (pa_attn_big): PA_BIG_CFG="rows,keys,simdgroups" per threadgroup (default 32,64,8; 44 KB of threadgroup memory: A19 allows it)
    // two instances: [0] prefill calls (ng >= 2), [1] one-group calls (decode / verify rounds: 48 rows, keys split finer)
    struct big_kernel {
        uint32_t mr, nb, nsg;
        id<MTLComputePipelineState> pso = nil, merge = nil;
        bool tried = false;
    };
    // A19 Pro q4_0. [0] 256 tokens x 16k keys: 32x64x8 35.0 ms (32x32x8 39.8; 64x32x4 59.8; old per-group path 76.1).
    // [1] one group x 75.5k keys, GPU only: 16x64x8 6.26 ms (32x64x8 9.40; old per-group kernel 10.95)
    big_kernel big_[2] = { { 32, 64, 8 }, { 16, 64, 8 } };
    std::string big_error;
    big_kernel * big_pipeline(uint32_t variant, int ng) {
        big_kernel & k = big_[ng == 1 ? 1 : 0];
        if (k.tried) return k.pso ? &k : nullptr;
        k.tried = true;
        if (@available(iOS 26.0, macOS 26.0, *)) {
            if (const char * c = getenv(ng == 1 ? "PA_BIG_CFG1" : "PA_BIG_CFG")) sscanf(c, "%u,%u,%u", &k.mr, &k.nb, &k.nsg);
            // a GPU with less threadgroup memory (the Mac's 32 KB) takes the largest key block that fits
            while (k.nb > 16 && k.nb*HD*2 + k.mr*k.nb*4 + k.mr*k.nb*2 + k.mr*4 > dev_.maxThreadgroupMemoryLength) k.nb /= 2;
            MTLCompileOptions * opt = [MTLCompileOptions new];
            opt.languageVersion = MTLLanguageVersion4_0;
            opt.preprocessorMacros = @{ @"PA_RELAXED" : (variant & 1) ? @"true" : @"false", @"PA_BMR" : @(k.mr).stringValue,
                                        @"PA_BNB" : @(k.nb).stringValue, @"PA_BNSG" : @(k.nsg).stringValue };
            NSError * e = nil;
            id<MTLLibrary> lib = [dev_ newLibraryWithSource:na_source() options:opt error:&e];
            if (lib) k.pso = [dev_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pa_attn_big"] error:&e];
            if (lib && k.pso) k.merge = [dev_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pa_merge_big"] error:&e];
            if (!k.merge) { k.pso = nil; big_error = std::string("big kernel: ") + (e ? e.localizedDescription.UTF8String : "?"); }
        }
        return k.pso ? &k : nullptr;
    }

    // ATTN_BIG on the prefill kernel: all ng groups in one dispatch per page, dense rows, big key chunks (the rows already give
    // the GPU enough threadgroups), one merge. Same outputs as the per-group path.
    // geometry of one ATTN / ATTN_BIG on the dense-row kernel
    struct big_geom { uint32_t nkv, rows, nt, real, qr, chunk, max_slots, n_slots; size_t slot_b; };
    big_geom geom(const big_kernel & K, const uint32_t * nkeys, int n_pages, const config_req & cfg, int ng, int ntok) const {
        big_geom g;
        g.nkv = cfg.n_head_kv; g.rows = g.nkv * NR;
        // one group with fewer than 8 tokens (decode, short verify rounds): pack only the real rows (GQA member g, token t ->
        // g*nt + t), so a 1-token call is one 16-row tile per head instead of three
        g.nt = (ng == 1 && ntok >= 1 && ntok < 8) ? (uint32_t) ntok : 0;
        g.real = g.nt ? 6 * g.nt : (uint32_t) ng * NR;
        g.qr = (g.real + K.mr - 1) / K.mr * K.mr;
        static const size_t tmp_cap = (size_t) (getenv("PA_GPU_TMP_MB") ? atoi(getenv("PA_GPU_TMP_MB")) : 96) << 20;
        // keys per threadgroup: a prefill ubatch has rows enough to fill the GPU; one group needs the keys split finer
        static const uint32_t chunk_big = getenv("PA_BIG_CHUNK") ? (uint32_t) atoi(getenv("PA_BIG_CHUNK")) : 4096;
        static const uint32_t chunk_one = getenv("PA_BIG_CHUNK1") ? (uint32_t) atoi(getenv("PA_BIG_CHUNK1")) : 1024;
        g.chunk = std::max(K.nb, (ng > 1 ? chunk_big : chunk_one) / K.nb * K.nb);
        // pages go in batches whose partials fit PA_GPU_TMP_MB; each batch's merge folds into the previous batches' result
        g.slot_b = (size_t) g.nkv * g.qr * HD * 4;
        g.max_slots = (uint32_t) std::max<size_t>(1, tmp_cap / g.slot_b);
        g.n_slots = 0;
        for (int i = 0; i < n_pages; i++) g.n_slots += (nkeys[i] + g.chunk - 1) / g.chunk;
        return g;
    }
    void ensure_big(const big_geom & g, const uint32_t * nkeys, int n_pages, int ng) {
        uint32_t top = 0;
        for (int i = 0, cur = 0; i < n_pages; i++) {
            const uint32_t nch = (nkeys[i] + g.chunk - 1) / g.chunk;
            cur = (cur > 0 && cur + nch > g.max_slots) ? nch : cur + nch;
            top = std::max(top, (uint32_t) cur);
        }
        ensure(q_, (size_t) g.nkv * g.qr * HD * 2);
        ensure(op_, (size_t) top * g.slot_b);
        ensure(ml_, (size_t) top * g.nkv * g.qr * 2 * 4);
        ensure(o_, (size_t) ng * g.rows * HD * 4);
        ensure(mlo_, (size_t) ng * g.rows * 2 * 4);
        if (!live_) { live_ = [dev_ newBufferWithLength:16 options:MTLResourceStorageModeShared]; *(uint32_t *) live_.contents = 1; }
    }
    void pack_q(const big_geom & g, const uint16_t * Qs, int ng) {
        uint16_t * qd = (uint16_t *) q_.contents;
        for (uint32_t h = 0; h < g.nkv; h++) {
            if (g.nt) {
                for (uint32_t k = 0; k < 6; k++)
                    memcpy(qd + ((size_t) h * g.qr + k * g.nt) * HD, Qs + ((size_t) h * NR + k * 8) * HD, (size_t) g.nt * HD * 2);
            } else {
                for (int k = 0; k < ng; k++)
                    memcpy(qd + ((size_t) h * g.qr + (size_t) k * NR) * HD, Qs + ((size_t) k * g.rows + (size_t) h * NR) * HD, (size_t) NR * HD * 2);
            }
            memset(qd + ((size_t) h * g.qr + g.real) * HD, 0, (size_t) (g.qr - g.real) * HD * 2);
        }
    }
    void encode_big(id<MTLCommandBuffer> cb, const big_kernel & K, const big_geom & g, const page * pages, const uint32_t * nkeys,
                    int n_pages, const config_req & cfg, int ng, bool na) {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        const NSUInteger tgm = K.nb*HD*2 + K.mr*K.nb*4 + K.mr*K.nb*2 + K.mr*4;
        uint32_t fold = 0;
        for (int i0 = 0; i0 < n_pages; ) {
            [enc setComputePipelineState:K.pso];
            [enc setThreadgroupMemoryLength:(tgm + 15) / 16 * 16 atIndex:0];
            [enc setBuffer:q_ offset:0 atIndex:0];
            [enc setBuffer:op_ offset:0 atIndex:3];
            [enc setBuffer:ml_ offset:0 atIndex:4];
            [enc setBuffer:live_ offset:0 atIndex:6];
            uint32_t slot = 0;
            int i = i0;
            for (; i < n_pages; i++) {
                const uint32_t nch = (nkeys[i] + g.chunk - 1) / g.chunk;
                if (slot > 0 && slot + nch > g.max_slots) break;
                if (nch == 0) continue;
                [enc setBuffer:(__bridge id<MTLBuffer>) pages[i].hk offset:0 atIndex:1];
                [enc setBuffer:(__bridge id<MTLBuffer>) pages[i].hv offset:0 atIndex:2];
                pa_big_args_host a = { cfg.rs, na ? (uint32_t) (HD * 2) : cfg.hb, nkeys[i], g.chunk, slot, g.nkv, g.qr, na ? 0u : (cfg.is_q8 == 2 ? 2u : 1u) };
                [enc setBytes:&a length:sizeof a atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(g.qr / K.mr, g.nkv, nch) threadsPerThreadgroup:MTLSizeMake(K.nsg * 32, 1, 1)];
                slot += nch;
            }
            i0 = i;
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:K.merge];
            uint32_t mg[4] = { slot, g.nkv, g.qr, fold | (g.nt << 8) };
            [enc setBytes:mg length:sizeof mg atIndex:4];
            [enc setBuffer:op_ offset:0 atIndex:0];
            [enc setBuffer:ml_ offset:0 atIndex:1];
            [enc setBuffer:o_ offset:0 atIndex:2];
            [enc setBuffer:mlo_ offset:0 atIndex:3];
            [enc setBuffer:live_ offset:0 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger) ng * g.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(HD, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];   // the next batch reuses op_ / ml_ and folds into o_
            fold = 1;
        }
        [enc endEncoding];
    }

    // Pre-armed decode calls (PA_GPU_ARM=1; default off until measured on the phone): after a one-group call the server arms the next layer's call with the
    // same split; its command buffer is encoded and committed right away, parked on an event. When that call comes, only Q
    // is copied in and the event signalled: the ~1 ms between commit and GPU start moved off the critical path. A call that
    // doesn't match (another layer, token count or split) cancels it (live = 0: its kernels return at once).
    struct armed_t {
        id<MTLCommandBuffer> cb = nil;
        std::vector<void *> hk;
        std::vector<uint32_t> nkeys;
        uint32_t rs = 0, hb = 0, is_q8 = 0, nkv = 0;
        int ntok = 0;
        uint64_t v = 0;
    } armed_;
    id<MTLSharedEvent> arm_ev_ = nil;
    uint64_t arm_v_ = 0, n_arm_fired_ = 0, n_arm_missed_ = 0;
    bool arm_match(const page * pages, const uint32_t * nkeys, int n_pages, const config_req & cfg, int ntok) const {
        if (!armed_.cb || (int) armed_.hk.size() != n_pages || armed_.ntok != ntok || armed_.rs != cfg.rs || armed_.hb != cfg.hb ||
            armed_.is_q8 != cfg.is_q8 || armed_.nkv != cfg.n_head_kv) return false;
        for (int i = 0; i < n_pages; i++) if (armed_.hk[i] != pages[i].hk || armed_.nkeys[i] != nkeys[i]) return false;
        return true;
    }
    void arm_cancel() {
        if (!armed_.cb) return;
        *(uint32_t *) live_.contents = 0;
        arm_ev_.signaledValue = armed_.v;
        [armed_.cb waitUntilCompleted];   // its kernels return at once; the next call's buffers are then free
        *(uint32_t *) live_.contents = 1;
        armed_ = armed_t();
        n_arm_missed_++;
    }
    void arm(const page * pages, const uint32_t * nkeys, int n_pages, const config_req & cfg, int ntok) override {
      @autoreleasepool {
        static const bool on = getenv("PA_GPU_ARM") && atoi(getenv("PA_GPU_ARM")) != 0;   // off: Mac loopback, event wake 0.58 ms vs commit 0.06 (phone unmeasured)
        if (!on || cb_ || n_pages <= 0) return;
        arm_cancel();
        const bool na = cfg.is_q8 == 0;
        big_kernel * K = big_pipeline(cfg.gpu_variant & 1, 1);
        if (!K) return;
        if (!arm_ev_) arm_ev_ = [dev_ newSharedEvent];
        const big_geom g = geom(*K, nkeys, n_pages, cfg, 1, ntok);
        ensure_big(g, nkeys, n_pages, 1);
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        armed_.v = ++arm_v_;
        [cb encodeWaitForEvent:arm_ev_ value:armed_.v];
        encode_big(cb, *K, g, pages, nkeys, n_pages, cfg, 1, na);
        [cb commit];
        armed_.cb = cb; armed_.ntok = ntok; armed_.rs = cfg.rs; armed_.hb = cfg.hb; armed_.is_q8 = cfg.is_q8; armed_.nkv = cfg.n_head_kv;
        armed_.hk.resize(n_pages); armed_.nkeys.assign(nkeys, nkeys + n_pages);
        for (int i = 0; i < n_pages; i++) armed_.hk[i] = pages[i].hk;
      }
    }

    // ATTN_BIG on the prefill kernel: all ng groups in one dispatch per page, dense rows, big key chunks (the rows already give
    // the GPU enough threadgroups), one merge. Same outputs as the per-group path.
    bool begin_big(const big_kernel & K, const page * pages, const uint32_t * nkeys, int n_pages, const uint16_t * Qs, const config_req & cfg, int ng, bool na) {
        const int ntok = ng == 1 ? ntok_hint : 8;
        const big_geom g = geom(K, nkeys, n_pages, cfg, ng, ntok);
        rows_ = g.rows; ng_ = ng; last_big_chunk_ = g.chunk; last_big_slots_ = g.n_slots;
        if (ng == 1 && arm_match(pages, nkeys, n_pages, cfg, ntok)) {
            pack_q(g, Qs, ng);
            cb_ = armed_.cb;
            const uint64_t v = armed_.v;
            armed_ = armed_t();
            t0_ = std::chrono::steady_clock::now();
            t_commit_ = host_s();
            arm_ev_.signaledValue = v;
            n_arm_fired_++; n_big_++;
            return true;
        }
        arm_cancel();
        ensure_big(g, nkeys, n_pages, ng);
        pack_q(g, Qs, ng);
        cb_ = [queue_ commandBuffer];
        encode_big(cb_, K, g, pages, nkeys, n_pages, cfg, ng, na);
        t0_ = std::chrono::steady_clock::now();
        t_commit_ = host_s();
        [cb_ commit];
        n_big_++;
        return true;
    }
    uint32_t last_big_chunk_ = 0, last_big_slots_ = 0;
    double t_commit_ = 0, lat_start_ = 0, lat_end_ = 0;
    uint64_t n_lat_ = 0;
    static double host_s() {   // the timebase of MTLCommandBuffer.GPUStartTime / GPUEndTime
        static mach_timebase_info_data_t tb = [] { mach_timebase_info_data_t t; mach_timebase_info(&t); return t; }();
        return (double) mach_absolute_time() * tb.numer / tb.denom * 1e-9;
    }
    uint64_t n_big_ = 0;

    bool alloc_page(size_t bytes, page & p) override {
      @autoreleasepool {
        id<MTLBuffer> k = [dev_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> v = [dev_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!k || !v) return false;
        p.k = (uint8_t *) k.contents; p.v = (uint8_t *) v.contents;
        p.hk = (__bridge_retained void *) k; p.hv = (__bridge_retained void *) v;
        return true;
      }
    }
    void free_page(page & p) override {
        if (p.hk) { id<MTLBuffer> k = (__bridge_transfer id<MTLBuffer>) p.hk; (void) k; }
        if (p.hv) { id<MTLBuffer> v = (__bridge_transfer id<MTLBuffer>) p.hv; (void) v; }
        p = page();
    }

    // ng groups (ATTN_BIG: a prefill ubatch, 8 tokens per group) go into ONE command buffer: the per-call launch/wake cost is paid
    // once, not ng times. Groups run in batches whose partials fit PA_GPU_TMP_MB (default 96) of scratch; Qs holds ng groups
    // back to back ([ng][nkv][48][256]), end() returns ng x rows.
    bool begin(const page * pages, const uint32_t * nkeys, int n_pages, const uint16_t * Qs, const config_req & cfg, int ng = 1) override {
      @autoreleasepool {   // the server thread has no pool: without this every command buffer (and the pages it references) leaks
        const uint32_t nkv = cfg.n_head_kv, rows = nkv * NR;
        const bool na  = cfg.is_q8 == 0;                // f16 pages -> neural-accelerator kernel
        const bool naq = cfg.is_q8 == 2 || (cfg.is_q8 == 1 && (cfg.gpu_variant & 4));   // q4_0 / q8_0 pages -> dequant + NA
        // prefill calls (ng >= 2, any page type) take the dense-row kernel unless PA_BIG_KERNEL=0; decode (ng == 1) never does
        static const bool big_on = !getenv("PA_BIG_KERNEL") || atoi(getenv("PA_BIG_KERNEL")) != 0;
        // one-group calls (decode, verify rounds) too, with their real rows only: 140k, 2 ANE pages: 147.2 ms/token and 232.3 ms
        // per 8-token round vs 165.4 / 245.0 on the per-group kernel (48 rows padded to 64). PA_BIG_MIN_NG=2: that old path.
        static const int big_min = getenv("PA_BIG_MIN_NG") ? std::max(1, atoi(getenv("PA_BIG_MIN_NG"))) : 1;
        if (ng >= big_min && big_on) {
            if (big_kernel * K = big_pipeline(cfg.gpu_variant & 1, ng)) return begin_big(*K, pages, nkeys, n_pages, Qs, cfg, ng, na);
        }
        arm_cancel();   // the per-group kernels below share its buffers
        if (naq && pso_na_ && (cfg.gpu_variant & 3) != na_variant_) { id<MTLComputePipelineState> p = na_pipeline(cfg.gpu_variant & 3); if (p) pso_na_ = p; }
        if (naq && !pso_naq_) { error = "quantized NA kernel unavailable: " + na_error; return false; }
        if (na && pso_na_ && (cfg.gpu_variant & 3) != na_variant_) { id<MTLComputePipelineState> p = na_pipeline(cfg.gpu_variant & 3); if (p) pso_na_ = p; }
        if (na && !pso_na_) { error = "f16 pages need the NA kernel: " + na_error; return false; }
        const uint32_t rph = (na || naq) ? 64 : NR;     // rows per head in the partials
        const uint32_t chunk = cfg.gpu_chunk ? cfg.gpu_chunk : 1024;
        uint32_t n_slots = 0;
        for (int i = 0; i < n_pages; i++) n_slots += (nkeys[i] + chunk - 1) / chunk;
        const size_t op_g = (size_t) n_slots * nkv * rph * HD * 4, ml_g = (size_t) n_slots * nkv * rph * 2 * 4;   // per group
        static const size_t tmp_cap = (size_t) (getenv("PA_GPU_TMP_MB") ? atoi(getenv("PA_GPU_TMP_MB")) : 96) << 20;
        const int B = std::max(1, std::min(ng, (int) (tmp_cap / std::max<size_t>(op_g, 1))));   // groups per batch
        const size_t qg = (na || naq) ? (size_t) nkv * 64 * HD * 2 : (size_t) rows * HD * 2;   // Q bytes per group
        ensure(q_, qg * ng);
        ensure(op_, op_g * B);
        ensure(ml_, ml_g * B);
        ensure(o_, (size_t) ng * rows * HD * 4);
        ensure(mlo_, (size_t) ng * rows * 2 * 4);
        for (int g = 0; g < ng; g++) {
            const uint16_t * Qg = Qs + (size_t) g * rows * HD;
            uint8_t * qd = (uint8_t *) q_.contents + g * qg;
            if (na || naq) {
                memset(qd, 0, qg);
                for (uint32_t h = 0; h < nkv; h++) memcpy((uint16_t *) qd + (size_t) h * 64 * HD, Qg + (size_t) h * NR * HD, (size_t) NR * HD * 2);
            } else {
                memcpy(qd, Qg, (size_t) rows * HD * 2);
            }
        }
        rows_ = rows; ng_ = ng;

        cb_ = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb_ computeCommandEncoder];
        for (int b0 = 0; b0 < ng; b0 += B) {
            const int nb = std::min(B, ng - b0);
            if (naq) {
                [enc setComputePipelineState:pso_naq_];
                [enc setThreadgroupMemoryLength:(32*256*2 + 64*32*4 + 64*32*2 + 64*4) atIndex:0];
            } else if (na) {
                [enc setComputePipelineState:pso_na_];
                [enc setThreadgroupMemoryLength:(64*64*4 + 64*64*2 + 64*4) atIndex:0];
            } else {
                [enc setComputePipelineState:pso_attn_];
                const NSUInteger tgmem = 32*256*2 + 6*8*32*4 + 6*8*32*2 + 6*64*4;
                [enc setThreadgroupMemoryLength:(tgmem + 15) / 16 * 16 atIndex:0];
            }
            [enc setBuffer:op_ offset:0 atIndex:3];
            [enc setBuffer:ml_ offset:0 atIndex:4];
            for (int j = 0; j < nb; j++) {
                [enc setBuffer:q_ offset:(b0 + j) * qg atIndex:0];
                uint32_t slot = (uint32_t) j * n_slots;   // group j's partials: slots [j*n_slots, (j+1)*n_slots)
                for (int i = 0; i < n_pages; i++) {
                    const uint32_t nch = (nkeys[i] + chunk - 1) / chunk;
                    [enc setBuffer:(__bridge id<MTLBuffer>) pages[i].hk offset:0 atIndex:1];
                    [enc setBuffer:(__bridge id<MTLBuffer>) pages[i].hv offset:0 atIndex:2];
                    if (naq) {
                        pa_naq_args_host a = { cfg.rs, cfg.hb, nkeys[i], chunk, slot, nkv, cfg.is_q8 == 2 ? 1u : 0u };
                        [enc setBytes:&a length:sizeof a atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(nch, nkv, 1) threadsPerThreadgroup:MTLSizeMake(na_nsg_*32, 1, 1)];
                    } else if (na) {
                        pa_na_args_host a = { cfg.rs / 2, nkeys[i], chunk, slot, nkv };
                        [enc setBytes:&a length:sizeof a atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(nch, nkv, 1) threadsPerThreadgroup:MTLSizeMake(na_nsg_*32, 1, 1)];
                    } else {
                        pa_args_host a = { cfg.rs, cfg.hb, nkeys[i], chunk, slot, nkv };
                        [enc setBytes:&a length:sizeof a atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(nch, nkv, 1) threadsPerThreadgroup:MTLSizeMake(12*32, 1, 1)];
                    }
                    slot += nch;
                }
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:pso_merge_];
            uint32_t mg[4] = { n_slots, nkv, rph, NR };
            [enc setBytes:mg length:sizeof mg atIndex:4];
            for (int j = 0; j < nb; j++) {
                const int g = b0 + j;
                [enc setBuffer:op_ offset:j * op_g atIndex:0];
                [enc setBuffer:ml_ offset:j * ml_g atIndex:1];
                [enc setBuffer:o_ offset:(size_t) g * rows * HD * 4 atIndex:2];
                [enc setBuffer:mlo_ offset:(size_t) g * rows * 2 * 4 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(HD, 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];   // the next batch reuses op_ / ml_
        }
        [enc endEncoding];
        t0_ = std::chrono::steady_clock::now();
        t_commit_ = host_s();
        [cb_ commit];
        return true;
      }
    }

    double end(float * O, float * m, float * l) override {
      @autoreleasepool {
        [cb_ waitUntilCompleted];
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
        if (cb_.status != MTLCommandBufferStatusCompleted) {
            for (uint32_t r = 0; r < (uint32_t) ng_ * rows_; r++) { m[r] = -INFINITY; l[r] = 0; }
            error = "GPU command buffer failed";
            cb_ = nil;
            return -1;
        }
        last_exec_ms = (cb_.GPUEndTime - cb_.GPUStartTime) * 1e3;
        last_sched_ms = ms - last_exec_ms;
        // where the non-execution time goes (one-group calls): commit -> GPU start, GPU end -> this thread awake
        if (ng_ == 1) {
            const double st = (cb_.GPUStartTime - t_commit_) * 1e3, en = (host_s() - cb_.GPUEndTime) * 1e3;
            lat_start_ = n_lat_ ? 0.9 * lat_start_ + 0.1 * st : st;
            lat_end_   = n_lat_ ? 0.9 * lat_end_ + 0.1 * en : en;
            n_lat_++;
        }
        memcpy(O, o_.contents, (size_t) ng_ * rows_ * HD * 4);
        const float * ml = (const float *) mlo_.contents;
        for (uint32_t r = 0; r < (uint32_t) ng_ * rows_; r++) { m[r] = ml[2*r]; l[r] = ml[2*r + 1]; }
        cb_ = nil;
        return exec_only ? last_exec_ms : ms;
      }
    }
    // between calls (the server's hot spin): a 32-thread dispatch, not waited for. Skipped while a call's command buffer is out.
    void warm() override {
      @autoreleasepool {
        if (cb_ || armed_.cb || !pso_warm_) return;
        ensure(warm_buf_, 256);
        id<MTLCommandBuffer> cb = [queue_ commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso_warm_];
        [enc setBuffer:warm_buf_ offset:0 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cb commit];
        n_warm++;
      }
    }
    uint64_t n_warm = 0;
    double exec_ms() const override { return last_exec_ms; }
    bool exec_only = getenv("PA_GPU_EXEC_ONLY") != nullptr;
    std::string describe() override {
        char b[640]; snprintf(b, sizeof b, "last_gpu_exec_ms=%.3f last_gpu_launch_ms=%.3f na=%d variant=%u warms=%llu %s big=%u,%u,%u calls=%llu chunk=%u slots=%u %s %s",
            last_exec_ms, last_sched_ms, has_na() ? 1 : 0, na_variant_, (unsigned long long) n_warm, na_src_note_.c_str(), big_[0].mr, big_[0].nb, big_[0].nsg,
            (unsigned long long) n_big_, last_big_chunk_, last_big_slots_, na_error.c_str(), big_error.c_str());
        char l[192]; snprintf(l, sizeof l, " decode GPU latency: commit->start %.3f ms, end->wake %.3f ms (EMA, %llu calls), armed calls fired %llu missed %llu",
            lat_start_, lat_end_, (unsigned long long) n_lat_, (unsigned long long) n_arm_fired_, (unsigned long long) n_arm_missed_);
        return std::string(b) + l;
    }

private:
    id<MTLDevice> dev_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLComputePipelineState> pso_attn_ = nil, pso_merge_ = nil, pso_na_ = nil, pso_naq_ = nil, pso_warm_ = nil;
    id<MTLBuffer> warm_buf_ = nil;
    id<MTLBuffer> q_ = nil, op_ = nil, ml_ = nil, o_ = nil, mlo_ = nil, live_ = nil;
    id<MTLCommandBuffer> cb_ = nil;
    uint32_t rows_ = 0;
    int ng_ = 1;
    std::chrono::steady_clock::time_point t0_;

    void ensure(id<MTLBuffer> __strong & b, size_t bytes) {
        if (b && b.length >= bytes) return;
        b = [dev_ newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
    }
};

} // namespace pa
