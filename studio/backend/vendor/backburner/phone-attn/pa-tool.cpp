// pa-tool: the phone-attn loopback server and test client (macOS).
//   pa-tool serve [port]                       run the phone side on this Mac's SME2 (the "loopback phone")
//   pa-tool test HOST [port] [nkeys] [iters]   fill 16 layers with random q8_0 K/V, check ATTN against an fp64
//                                              reference (layer 0), then time ATTN round trips
// Build: phone-attn/build.sh
#include "phone-attn.h"
#ifdef __OBJC__
#include "pa-ane.mm"   // built as Objective-C++ (build.sh): the ANE page engine, PA_ANE_TEMPLATE=dir.mlmodelc
#include "pa-metal.mm"  // PA_GPU=1: the phone's GPU engine on this Mac's GPU (kernel correctness without a phone)
#endif

#include <algorithm>
#include <cstdlib>
#include <random>

static uint64_t g_s = 0x9e3779b97f4a7c15ull;
static float frand() { g_s = g_s * 6364136223846793005ull + 1442695040888963407ull; return ((g_s >> 40) / (float) (1 << 24)) * 2 - 1; }

// one q8_0 row set: nkv heads x 8 blocks x (f16 d + 32 int8)
static void make_q8_row(uint8_t * row, int nkv, float amp) {
    for (int h = 0; h < nkv; h++) for (int b = 0; b < 8; b++) {
        float x[32], am = 0; for (int i = 0; i < 32; i++) { x[i] = frand() * amp; am = std::max(am, fabsf(x[i])); }
        float d = am / 127; uint16_t dh = pa::f2h(d); uint8_t * o = row + h * 272 + b * 34;
        memcpy(o, &dh, 2); float id = d ? 1 / pa::h2f(dh) : 0;
        for (int i = 0; i < 32; i++) o[2 + i] = (uint8_t) (int8_t) lrintf(std::max(-127.f, std::min(127.f, x[i] * id)));
    }
}
static int g_q4 = 0;   // 1: rows are q4_0 (4 heads x 8 blocks x 18 B)
static void make_q4_row(uint8_t * row, int nkv, float amp) {
    for (int h = 0; h < nkv; h++) for (int b = 0; b < 8; b++) {
        float x[32], am = 0, mx = 0; for (int i = 0; i < 32; i++) { x[i] = frand() * amp; if (fabsf(x[i]) > am) { am = fabsf(x[i]); mx = x[i]; } }
        float d = mx / -8; uint16_t dh = pa::f2h(d); uint8_t * o = row + h * 144 + b * 18;
        memcpy(o, &dh, 2); float id = d ? 1 / pa::h2f(dh) : 0;
        for (int i = 0; i < 16; i++) {
            int lo = std::min(15, (int) (x[i] * id + 8.5f)), hi = std::min(15, (int) (x[i + 16] * id + 8.5f));
            lo = std::max(0, lo); hi = std::max(0, hi);
            o[2 + i] = (uint8_t) (lo | (hi << 4));
        }
    }
}
static float deq(const uint8_t * row, int h, int d) {
    if (g_q4) {
        const uint8_t * blk = row + h * 144 + (d / 32) * 18;
        uint16_t dh; memcpy(&dh, blk, 2);
        const int i = d % 32; const uint8_t q = blk[2 + (i % 16)];
        return pa::h2f(dh) * (float) ((i < 16 ? (q & 15) : (q >> 4)) - 8);
    }
    const uint8_t * blk = row + h * 272 + (d / 32) * 34;
    uint16_t dh; memcpy(&dh, blk, 2);
    return pa::h2f(dh) * (float) (int8_t) blk[2 + d % 32];
}

int main(int argc, char ** argv) {
    std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "serve") {
        int port = argc > 2 ? atoi(argv[2]) : pa::DEFAULT_PORT;
        int thr = argc > 3 ? atoi(argv[3]) : 2;
        pa::status st;
        pa::engine * gpu = nullptr;
#ifdef __OBJC__
        if (getenv("PA_GPU") && atoi(getenv("PA_GPU"))) {
            auto * m = new pa::metal_engine();
            if (!m->init()) { fprintf(stderr, "GPU engine off: %s\n", m->error.c_str()); delete m; } else gpu = m;
        }
#endif
        pa::server srv(&st, [](const std::string & s) { fprintf(stderr, "%s\n", s.c_str()); }, thr, gpu);
#ifdef __OBJC__
        // PA_ANE_TEMPLATE: old keys on this Mac's Neural Engine (bit-identical to the phone's for these models): the accuracy gate
        if (const char * t = getenv("PA_ANE_TEMPLATE")) {
            auto * ane = new pa::ane_engine();
            const size_t mb = getenv("PA_ANE_MB") ? (size_t) atoll(getenv("PA_ANE_MB")) : 4096;
            if (!ane->init(t, std::string(getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp") + "/pa-anekv", mb)) {
                fprintf(stderr, "ANE engine off: %s\n", ane->error.c_str()); delete ane;
            } else {
                fprintf(stderr, "ANE page engine: %u keys per page, budget %zu MB\n", ane->page_keys(), mb);
                srv.set_page_engine(ane);
            }
        }
#endif
        fprintf(stderr, "%s\n", srv.serve(port).c_str());
        return 1;
    }
    if (mode == "stats" && argc >= 3) {   // the peer's STATS without touching its state (no CONFIG)
        pa::client c;
        if (!c.connect(argv[2], argc > 3 ? atoi(argv[3]) : pa::DEFAULT_PORT)) { fprintf(stderr, "connect: %s\n", c.last_err.c_str()); return 1; }
        pa::hello_rep hr; if (!c.hello(hr)) { fprintf(stderr, "hello: %s\n", c.last_err.c_str()); return 1; }
        printf("%s\n", c.stats().c_str());
        return 0;
    }
    if (mode != "test" || argc < 3) {
        fprintf(stderr, "usage: pa-tool serve [port] [threads] | pa-tool test HOST [port] [nkeys] [iters] [layers] [sme_workers] [helpers] [gpu_permille] [gpu_chunk]\n");
        return 2;
    }
    const std::string host = argv[2];
    const int port = argc > 3 ? atoi(argv[3]) : pa::DEFAULT_PORT;
    const int N = argc > 4 ? atoi(argv[4]) : 16384;
    const int iters = argc > 5 ? atoi(argv[5]) : 20;
    const int n_layer = argc > 6 ? atoi(argv[6]) : 16;
    g_q4 = getenv("PA_Q4") ? atoi(getenv("PA_Q4")) : 0;   // PA_Q4=1: send q4_0 rows
    const int nkv = 4, rs = g_q4 ? nkv * 144 : nkv * 272, hb = g_q4 ? 144 : 272;
    const float scale = 1.0f / 16;

    pa::client c;
    if (!c.connect(host, port)) { fprintf(stderr, "connect: %s\n", c.last_err.c_str()); return 1; }
    pa::hello_rep hr;
    if (!c.hello(hr)) { fprintf(stderr, "hello: %s\n", c.last_err.c_str()); return 1; }
    printf("peer: version %u sme2 %u '%s'\n", hr.version, hr.sme2, hr.device);
    const int sw = argc > 7 ? atoi(argv[7]) : 1, sh = argc > 8 ? atoi(argv[8]) : 1;   // SME workers, pack helpers each (0 = plain)
    const int gpm = argc > 9 ? atoi(argv[9]) : 0, gch = argc > 10 ? atoi(argv[10]) : 1024; // GPU share of pages (permille), keys per GPU threadgroup
    const int f16 = argc > 11 ? atoi(argv[11]) : 0;   // phone stores K/V as f16 (needed by the NA kernel)
    const int gv = argc > 12 ? atoi(argv[12]) : 0;    // NA kernel variant bits (1 relaxed precision, 2 eight simdgroups)
    pa::config_req cfg = { (uint32_t) n_layer, (uint32_t) nkv, (uint32_t) rs, (uint32_t) hb, g_q4 ? 2u : 1u, (uint32_t) sw, (uint32_t) sh, (uint32_t) gpm, (uint32_t) gch, (uint32_t) f16, (uint32_t) gv };
    printf("sme workers %d, pack helpers each %d, gpu share %d/1000, gpu chunk %d, phone f16 %d\n", sw, sh, gpm, gch, f16);
    if (!c.config(cfg)) { fprintf(stderr, "config: %s\n", c.last_err.c_str()); return 1; }

    // layer 0's K/V kept for the reference; the other layers get fresh random pages (sent, not kept)
    std::vector<uint8_t> K0((size_t) N * rs), V0((size_t) N * rs);
    for (int j = 0; j < N; j++) {
        if (g_q4) { make_q4_row(&K0[(size_t) j * rs], nkv, 1.5f); make_q4_row(&V0[(size_t) j * rs], nkv, 1.0f); }
        else      { make_q8_row(&K0[(size_t) j * rs], nkv, 1.5f); make_q8_row(&V0[(size_t) j * rs], nkv, 1.0f); }
    }
    const int page = 4096;
    auto t0 = std::chrono::steady_clock::now();
    size_t sent = 0;
    for (int L = 0; L < n_layer; L++) {
        for (int p0 = 0; p0 < N; p0 += page) {
            const int n = std::min(page, N - p0);
            if (!c.append(L, p0, n, &K0[(size_t) p0 * rs], &V0[(size_t) p0 * rs], rs)) { fprintf(stderr, "append layer %d key %d: %s\n", L, p0, c.last_err.c_str()); return 1; }
            sent += 2ull * n * rs;
        }
    }
    const double ta = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("append: %d layers x %d keys = %.1f MB in %.2f s (%.0f MB/s)\n", n_layer, N, sent / 1e6, ta, sent / 1e6 / ta);

    const size_t qn = (size_t) nkv * pa::NR * pa::HD;
    std::vector<uint16_t> Q(qn), O(qn);
    std::vector<float> lse(qn / pa::HD);
    for (size_t i = 0; i < qn; i++) Q[i] = pa::f2h(frand() * 1.7f);
    // PA_NTOK=k (1..8): ATTN calls carry k tokens (a decode step / short verify round): rows t >= k (row r = g*8 + t) are zero
    const int ntok = getenv("PA_NTOK") ? std::max(1, std::min(8, atoi(getenv("PA_NTOK")))) : 8;
    for (size_t i = 0; i < qn; i++) if ((i / pa::HD) % 8 >= (size_t) ntok) Q[i] = 0;
    pa::attn_rep rep;
    // PA_WAIT_MS: let the peer's background work (ANE page builds) finish first; PA_CHECKS=k: check k calls in a row (an ANE
    // page's first call is uncentered, later ones use the previous call's max)
    if (getenv("PA_WAIT_MS")) std::this_thread::sleep_for(std::chrono::milliseconds(atoi(getenv("PA_WAIT_MS"))));
    const int n_checks = getenv("PA_CHECKS") ? std::max(1, atoi(getenv("PA_CHECKS"))) : 1;
    for (int ci = 0; ci < n_checks; ci++) {
    if (!c.attn(0, ntok, 0, scale, Q.data(), qn, O.data(), lse.data(), rep)) { fprintf(stderr, "attn: %s\n", c.last_err.c_str()); return 1; }

    // PA_SKIP_REF=1 skips the O(N) fp64 oracle when probing capacity at 200k+ keys.
    // Keep the default oracle for correctness gates at smaller sizes.
    if (!getenv("PA_SKIP_REF") || atoi(getenv("PA_SKIP_REF")) == 0) {
        double eo = 0, el = 0;
        std::vector<double> s(N), kf((size_t) N * pa::HD), acc(pa::HD);
        for (int h = 0; h < nkv; h++) {
            for (int j = 0; j < N; j++) for (int d = 0; d < pa::HD; d++) kf[(size_t) j * pa::HD + d] = deq(&K0[(size_t) j * rs], h, d);
            for (int r = 0; r < pa::NR; r += ntok < 8 ? 1 : 5) {
                if (r % 8 >= ntok) continue;
                const uint16_t * q = &Q[((size_t) h * pa::NR + r) * pa::HD];
                double mx = -1e300;
                for (int j = 0; j < N; j++) {
                    double a = 0; for (int d = 0; d < pa::HD; d++) a += (double) pa::h2f(pa::f2h(pa::h2f(q[d]) * scale)) * kf[(size_t) j * pa::HD + d];
                    s[j] = a; mx = std::max(mx, a);
                }
                double L = 0; std::fill(acc.begin(), acc.end(), 0.0);
                for (int j = 0; j < N; j++) { double p = exp(s[j] - mx); L += p; for (int d = 0; d < pa::HD; d++) acc[d] += p * deq(&V0[(size_t) j * rs], h, d); }
                double omax = 1e-30; for (int d = 0; d < pa::HD; d++) omax = std::max(omax, fabs(acc[d] / L));
                for (int d = 0; d < pa::HD; d++) eo = std::max(eo, fabs(pa::h2f(O[((size_t) h * pa::NR + r) * pa::HD + d]) - acc[d] / L) / omax);
                el = std::max(el, fabs(lse[h * pa::NR + r] - (mx + log(L))));
            }
        }
        printf("check (layer 0, %d keys, %s): max|O - ref|/max|ref| = %.2e  max|lse - ref| = %.2e  -> %s\n",
               N, ntok < 8 ? "every real row" : "every 5th row", eo, el, (eo < 5e-3 && el < 5e-3) ? "PASS" : "FAIL");
    } else {
        printf("check: skipped fp64 reference (capacity/timing only)\n");
    }

    printf("  (check call: %u of %u pages on the GPU, gpu %.2f ms, sme %.2f ms, phone %.2f ms)\n", rep.gpu_pages, rep.pages, rep.gpu_ms, rep.sme_ms, rep.phone_ms);
    }
    printf("peer stats: %s\n", c.stats().c_str());

    // PA_SECONDS=S: sustained mode. Back-to-back ATTN calls (layers cycling, like a decode round) for S seconds,
    // one line per PA_WINDOW seconds (default 10) with that window's p50s and the phone's thermal state.
    // Burst numbers (the first window) overstate a phone; plan with the last windows.
    if (getenv("PA_SECONDS")) {
        const double secs = atof(getenv("PA_SECONDS")), win = getenv("PA_WINDOW") ? atof(getenv("PA_WINDOW")) : 10;
        const auto T0 = std::chrono::steady_clock::now();
        auto el = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count(); };
        auto p50 = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
        int L = 0;
        for (double w0 = 0; w0 < secs; w0 += win) {
            std::vector<double> rt, ph, gm, sm;
            while (el() < std::min(w0 + win, secs)) {
                auto a = std::chrono::steady_clock::now();
                if (!c.attn(L, ntok, 0, scale, Q.data(), qn, O.data(), lse.data(), rep)) { fprintf(stderr, "attn: %s\n", c.last_err.c_str()); return 1; }
                rt.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
                ph.push_back(rep.phone_ms); gm.push_back(rep.gpu_ms); sm.push_back(rep.sme_ms);
                L = (L + 1) % n_layer;
            }
            const std::string st = c.stats();
            const size_t tp = st.find("thermal=");
            printf("t=%5.0fs calls %4zu | round trip p50 %6.2f ms | phone p50 %6.2f (gpu %6.2f, sme %6.2f) | link+pack %5.2f | %s\n",
                   std::min(w0 + win, secs), rt.size(), p50(rt), p50(ph), p50(gm), p50(sm), p50(rt) - p50(ph),
                   tp == std::string::npos ? "thermal=?" : st.substr(tp, 9).c_str());
            fflush(stdout);
        }
        printf("peer stats: %s\n", c.stats().c_str());
        if (!c.config(cfg)) { fprintf(stderr, "release phone KV: %s\n", c.last_err.c_str()); return 1; }
        return 0;
    }

    // PA_BIG_TOK=T: ATTN_BIG calls of T tokens (a prefill ubatch: T/8 groups), timed; then every group checked against a plain
    // ATTN call with the same Q (the proven one-group path): max |O diff| / max |O| and max |lse diff|
    if (getenv("PA_BIG_TOK")) {
        const int nt = atoi(getenv("PA_BIG_TOK")), ng = (nt + 7) / 8;
        std::vector<uint16_t> Qb((size_t) ng * qn), Ob((size_t) ng * qn), O1(qn);
        std::vector<float> lb((size_t) ng * qn / pa::HD), l1(qn / pa::HD);
        for (auto & x : Qb) x = pa::f2h(frand() * 1.7f);
        std::vector<double> bt, bp;
        const int bgap = getenv("PA_GAP_US") ? atoi(getenv("PA_GAP_US")) : 0;   // Mac work between two layers' calls
        for (int it = 0; it < iters; it++) {
            if (bgap > 0) std::this_thread::sleep_for(std::chrono::microseconds(bgap));
            auto a = std::chrono::steady_clock::now();
            if (!c.attn_big(it % n_layer, nt, 0, scale, Qb.data(), Qb.size(), Ob.data(), lb.data(), rep)) { fprintf(stderr, "attn_big: %s\n", c.last_err.c_str()); return 1; }
            bt.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count()); bp.push_back(rep.phone_ms);
        }
        std::sort(bt.begin(), bt.end()); std::sort(bp.begin(), bp.end());
        printf("ATTN_BIG %d tokens (%d groups) x %d keys, %d calls: round trip p50 %.1f ms (min %.1f) | phone p50 %.1f ms = %.2f ms per group\n",
               nt, ng, N, iters, bt[iters / 2], bt[0], bp[iters / 2], bp[iters / 2] / ng);
        const int L = (iters - 1) % n_layer;
        if (!c.attn_big(L, nt, 0, scale, Qb.data(), Qb.size(), Ob.data(), lb.data(), rep)) return 1;
        double eo = 0, om = 1e-30, el = 0;
        for (int g = 0; g < ng; g++) {
            if (!c.attn(L, 8, 0, scale, Qb.data() + (size_t) g * qn, qn, O1.data(), l1.data(), rep)) return 1;
            for (size_t i = 0; i < qn; i++) { eo = std::max(eo, (double) fabsf(pa::h2f(Ob[(size_t) g * qn + i]) - pa::h2f(O1[i]))); om = std::max(om, (double) fabsf(pa::h2f(O1[i]))); }
            for (size_t r = 0; r < qn / pa::HD; r++) el = std::max(el, (double) fabsf(lb[(size_t) g * qn / pa::HD + r] - l1[r]));
        }
        printf("  big vs one-group calls: max|O diff|/max|O| %.2e, max|lse diff| %.2e -> %s\n", eo / om, el, (eo / om < 5e-3 && el < 5e-3) ? "PASS" : "FAIL");
        printf("peer stats: %s\n", c.stats().c_str());
        if (getenv("PA_KEEP") && atoi(getenv("PA_KEEP"))) return 0;
        if (!c.config(cfg)) { fprintf(stderr, "release phone KV: %s\n", c.last_err.c_str()); return 1; }
        return 0;
    }

    std::vector<double> rt, ph, gm, sm;
    // PA_GAP_US=N: idle N us between calls (a real decode leaves ~4-8 ms of Mac work between two phone layers)
    const int gap_us = getenv("PA_GAP_US") ? atoi(getenv("PA_GAP_US")) : 0;
    for (int it = 0; it < iters; it++) {
        const int L = it % n_layer;
        // PA_GAP_SPIN=1: spin through the gap like the Mac's phone thread does in a real decode (instead of sleeping)
        static const bool spin = getenv("PA_GAP_SPIN") && atoi(getenv("PA_GAP_SPIN"));
        if (gap_us > 0 && spin) { const auto g0 = std::chrono::steady_clock::now(); while (std::chrono::steady_clock::now() - g0 < std::chrono::microseconds(gap_us)) __builtin_arm_yield(); }
        else if (gap_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
        auto a = std::chrono::steady_clock::now();
        if (!c.attn(L, ntok, 0, scale, Q.data(), qn, O.data(), lse.data(), rep)) { fprintf(stderr, "attn: %s\n", c.last_err.c_str()); return 1; }
        rt.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
        ph.push_back(rep.phone_ms); gm.push_back(rep.gpu_ms); sm.push_back(rep.sme_ms);
    }
    std::sort(rt.begin(), rt.end()); std::sort(ph.begin(), ph.end()); std::sort(gm.begin(), gm.end()); std::sort(sm.begin(), sm.end());
    printf("ATTN %d keys x 192 rows, %d calls: round trip p50 %.2f ms (min %.2f, p90 %.2f) | phone compute p50 %.2f ms | link+pack %.2f ms\n",
           N, iters, rt[iters / 2], rt[0], rt[iters * 9 / 10], ph[iters / 2], rt[iters / 2] - ph[iters / 2]);
    printf("  gpu p50 %.2f ms, sme p50 %.2f ms (%u of %u pages on the GPU)\n", gm[iters / 2], sm[iters / 2], rep.gpu_pages, rep.pages);
    printf("peer stats: %s\n", c.stats().c_str());
    // The test owns these pages. Free them before returning so the next shape (or the
    // inference server) sees the phone's actual available memory.
    if (getenv("PA_KEEP") && atoi(getenv("PA_KEEP"))) return 0;   // PA_KEEP=1: leave the keys (and ANE pages) on the peer
    if (!c.config(cfg)) { fprintf(stderr, "release phone KV: %s\n", c.last_err.c_str()); return 1; }
    return 0;
}
