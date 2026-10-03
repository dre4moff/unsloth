#!/usr/bin/env python3
"""ANE old-KV accuracy on REAL attention inputs (dumped by dump-qkv from a saved session).
For each layer: the old keys are cut into pages of PAGE keys; every page becomes one CoreML model shaped like build.py's
(all 4 KV heads, K^T and V as conv weights, R = 6 GQA heads x verify rows), run on the Mac (CPU or Mac ANE). Compared with an
fp64 reference computed from the SAME dequantized q8_0 K/V the Mac GPU reads:
  page  : O/l per query row, max|O/l - ref| / max|ref| (phone-kv's metric), and the log-sum-exp lse = m + ln l (abs error)
  full  : the pages merged (log-sum-exp) with the rest (old keys past the last full page + the new rows' keys, fp64), vs the
          fp64 full attention; next to it the GPU kernel's own output (dumped) vs the same reference.
Variants (weights are quantized here, explicitly, as constexpr_blockwise_shift_scale = what linear_quantize_weights emits):
  fp16      K, V fp16
  int8ch    K int8 one scale per key (256 dims), V int8 one scale per head-dim channel across all keys (= build.py int8ch)
  k8v16     K int8 per key, V fp16
  int8vs    int8ch, but V rows divided by their per-key max s_j first and P multiplied by s_j in the model (P_j s_j . V_j/s_j)
  q8blk     the exact q8_0 data: int8 + one fp16 scale per 32 values (K: blocks along the conv's input axis, V: along its output
            axis), i.e. zero extra quantization error
  check_real.py DUMP_DIR [--layers 0,5,10,15] [--page 16384] [--variants ...] [--units cpu|ne] [--pscale C | --norm]
--pscale C (default 256): the model normalizes P = e/l and multiplies by C before the P.V conv, and returns C.O/l. Without it
the Mac ANE's P.V conv drops the fp16-subnormal products (softmax weights < 6.1e-5: 30-85% of the keys at 51k, measured),
page error 2.6-9.8e-3 -> 1.4-2.1e-3 on layer 0. --pscale 0 = the old unnormalized O; --norm = O/l without the scale-up.
--center twopass|oracle|prevrow: scores come out as q.k - c from the conv itself (extra input channel -c, extra all-ones key
column), so the scores near the row max are rounded near 0 (fp16 step 0.008 at |s|~16 -> <=0.0005 below 1). c = the model's own
first-pass row max (twopass: a second K conv), the exact row max (oracle: upper bound), or the neighbouring verify row's max
(prevrow: stands in for the previous decode step's returned m, which a real engine would pass for free).
--phone IP: run every page on the phone's Neural Engine instead (Sidecar :50061 `pred`, 2026-09-25 build): the model is compiled
here, pushed to Documents/anekv-chk/ (3 reused slots, ~200 MB), loaded with `load .. ne`, fed the real q (and c).
Measured 2026-09-25: on the Mac CPU the fp16 path is ~30% off (fp16 accumulation); CoreML runs pages < ~4k keys on the CPU."""
import argparse, json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

ap = argparse.ArgumentParser()
ap.add_argument("dump"); ap.add_argument("--layers", default="0,5,10,15"); ap.add_argument("--page", type=int, default=16384)
ap.add_argument("--variants", default="fp16,q8blk,int8ch,int8vs"); ap.add_argument("--units", default="cpu", choices=["cpu", "ne"])
ap.add_argument("--norm", action="store_true"); ap.add_argument("--pscale", type=float, default=256);
ap.add_argument("--vchunk", type=int, default=1, help="split the P.V conv into C groups of keys (fp16 accumulation test)");
ap.add_argument("--phone", default="", help="run the pages on this phone's ANE (Sidecar :50061)");
ap.add_argument("--center", default="none", choices=["none", "twopass", "oracle", "prevrow"]); ap.add_argument("--stats", action="store_true", help="print K/V value statistics")
a = ap.parse_args()
H, D, G = 4, 256, 6          # KV heads, head dim, query heads per KV head
CU = ct.ComputeUnit.CPU_AND_NE if a.units == "ne" else ct.ComputeUnit.CPU_ONLY


raw_q8 = {}


def load(i):
    meta = json.load(open(f"{a.dump}/L{i}.json"))
    def raw(k): return np.fromfile(f"{a.dump}/L{i}_{k}.bin", dtype=np.uint8)
    def f32(k):
        t = meta[k]; b = raw(k)
        return np.lib.stride_tricks.as_strided(b.view(np.float32), shape=t["ne"][::-1], strides=t["nb"][::-1]).astype(np.float64)
    def q80(k):                                   # -> [heads, n_kv, 256] exact dequant (fp16 d * int8)
        t = meta[k]; b = raw(k); ne, nb = t["ne"], t["nb"]
        rows = np.lib.stride_tricks.as_strided(b, shape=(ne[2], ne[1], 272), strides=(nb[2], nb[1], 1))
        blk = rows.reshape(ne[2], ne[1], 8, 34)
        d = blk[..., :2].copy().view(np.float16).astype(np.float32)          # [h, n, 8, 1]
        q = blk[..., 2:].copy().view(np.int8).astype(np.float32)             # [h, n, 8, 32]
        raw_q8[k] = (q.reshape(ne[2], ne[1], 256).astype(np.int8), d[..., 0].astype(np.float16))   # [h, n, 256], [h, n, 8]
        return (d * q).reshape(ne[2], ne[1], 256)
    mt = meta["mask"]; mb_ = raw("mask")
    mask = np.lib.stride_tricks.as_strided(mb_.view(np.float16), shape=(mt["ne"][1], mt["ne"][0]),
                                           strides=(mt["nb"][1], mt["nb"][0])).astype(np.float64)   # [rows, n_kv]
    q = f32("q")[0]                                # ne [256, T, 24] -> [24, T, 256]
    out = f32("out")[0]                            # ne [256, 24, T] -> [T, 24, 256]
    return meta["scale"], q, q80("k"), q80("v"), mask, out


def attend(qs, K, V, bias=None):
    """fp64: qs [R, D] (scale folded), K/V [N, D] -> O/l [R, D], m [R], l [R], unnormalized O [R, D]"""
    s = qs @ K.T
    if bias is not None: s = s + bias
    m = s.max(1); e = np.exp(s - m[:, None]); l = e.sum(1); o = e @ V
    return o / l[:, None], m, l, o


def q8ch(w):
    """int8 symmetric, one scale per row of w (axis 0). -> int8 data, fp16 scale [rows, 1]"""
    s = (np.abs(w).max(1, keepdims=True) / 127.0).astype(np.float16).astype(np.float32)
    s[s == 0] = 1
    return np.clip(np.round(w / s), -127, 127).astype(np.int8), s.astype(np.float16)


def weight(w, quant):   # w [out, in] float32 -> MIL const, fp16 or int8-per-output-channel constexpr
    if not quant:
        return w.astype(np.float16).reshape(*w.shape, 1, 1)
    qd, sc = q8ch(w)
    return mb.constexpr_blockwise_shift_scale(data=qd.reshape(*w.shape, 1, 1), scale=sc.reshape(-1, 1, 1, 1))


def build(Ks, Vs, variant, R, raw=None):
    kq = variant in ("int8ch", "k8v16", "int8vs"); vq = variant in ("int8ch", "int8vs")
    vs = variant == "int8vs"; ps = 0 if a.norm else a.pscale

    ext = a.center in ("oracle", "prevrow")
    specs = [mb.TensorSpec(shape=(1, H * D, 1, R), dtype=types.fp16)] + ([mb.TensorSpec(shape=(1, H, 1, R), dtype=types.fp16)] if ext else [])

    def body(q, c):
        os_, ms = [], []
        for h in range(H):
            K, V = Ks[h], Vs[h]; N = K.shape[0]
            qh = mb.slice_by_index(x=q, begin=[0, h * D, 0, 0], end=[1, (h + 1) * D, 1, R])
            if variant == "q8blk":   # exact q8_0: K [N, D] blocks of 32 along D (conv input axis); V^T [D, N] blocks along D (output axis)
                (kq8, kd), (vq8, vd) = raw[h]
                wk = mb.constexpr_blockwise_shift_scale(data=kq8.reshape(N, D, 1, 1), scale=kd.reshape(N, D // 32, 1, 1))
                wv = mb.constexpr_blockwise_shift_scale(data=np.ascontiguousarray(vq8.T).reshape(D, N, 1, 1),
                                                        scale=np.ascontiguousarray(vd.T).reshape(D // 32, N, 1, 1))
            else:
                wk = weight(K, kq)
            s = mb.conv(x=qh, weight=wk)                                              # [1, N, 1, R]
            if a.center != "none":
                ch = mb.reduce_max(x=s, axes=[1], keep_dims=True) if a.center == "twopass" else \
                     mb.slice_by_index(x=c, begin=[0, h, 0, 0], end=[1, h + 1, 1, R])
                qa = mb.concat(values=[qh, mb.mul(x=ch, y=np.float16(-1))], axis=1)       # [1, D+1, 1, R]
                s = mb.conv(x=qa, weight=weight(np.concatenate([K, np.ones((N, 1), np.float32)], 1), kq))   # q.k - c
            m = mb.reduce_max(x=s, axes=[1], keep_dims=True)
            e = mb.exp(x=mb.sub(x=s, y=m))
            l = mb.reduce_sum(x=e, axes=[1], keep_dims=True)
            if ps:
                e = mb.mul(x=mb.real_div(x=e, y=l), y=np.float16(ps))
            if vs:
                sj = np.abs(V).max(1); sj[sj == 0] = 1
                sj = sj.astype(np.float16).astype(np.float32)
                e = mb.mul(x=e, y=sj.astype(np.float16).reshape(1, N, 1, 1))
                V = V / sj[:, None]
            if a.vchunk > 1:   # P.V as C grouped convs over N/C keys each, then the C partial outputs added
                C = a.vchunk; wv = V.reshape(C, N // C, D).transpose(0, 2, 1).reshape(C * D, N // C)
                o = mb.conv(x=e, weight=weight(np.ascontiguousarray(wv), vq), groups=C)               # [1, C*D, 1, R]
                o = mb.reshape(x=mb.reduce_sum(x=mb.reshape(x=o, shape=[1, C, D, R]), axes=[1]), shape=[1, D, 1, R])
            else:
                o = mb.conv(x=e, weight=wv if variant == "q8blk" else weight(np.ascontiguousarray(V.T), vq))   # [1, D, 1, R]
            if a.norm:
                o = mb.real_div(x=o, y=l)
            os_.append(o); ms += [m, l] + ([ch] if a.center == "twopass" else [])
        return mb.concat(values=os_, axis=1, name="o"), mb.concat(values=ms, axis=1, name="ml")

    if ext:
        prog = mb.program(input_specs=specs, opset_version=ct.target.iOS18)(lambda q, c: body(q, c))
    else:
        prog = mb.program(input_specs=specs, opset_version=ct.target.iOS18)(lambda q: body(q, None))
    return ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                      compute_precision=ct.precision.FLOAT16, compute_units=CU)


def phone_predict(mdl, slot, feed):
    """compile mdl, push it to the phone, run one `pred` there -> {name: np.float32 array}"""
    import base64, socket, subprocess, tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    tmp = tempfile.mkdtemp(prefix="anekv-chk-")
    pkg = os.path.join(tmp, f"chk_{slot}.mlpackage"); mdl.save(pkg)
    subprocess.run(["xcrun", "coremlcompiler", "compile", pkg, tmp], check=True, stdout=subprocess.DEVNULL)
    mc = os.path.join(tmp, f"chk_{slot}.mlmodelc")
    subprocess.run([sys.executable, os.path.join(here, "../../scripts/phone-push.py"), a.phone, mc, f"anekv-chk/chk_{slot}.mlmodelc"],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["rm", "-rf", tmp])
    # Sidecar's :50061 serves one connection at a time: connect per page, or the next page's push (its own connection) waits forever
    c = socket.create_connection((a.phone, 50061), timeout=600); _ph = c.makefile("rw")
    def cmd(line):
        _ph.write(line + "\n"); _ph.flush(); r = json.loads(_ph.readline())
        if "error" in r: sys.exit(f"phone: {line.split()[0]}: {r['error']}")
        return r
    cmd(f"load chk{slot} anekv-chk/chk_{slot}.mlmodelc ne")
    r = cmd(f"pred chk{slot} " + " ".join(f"{k}={base64.b64encode(v.astype(np.float16).tobytes()).decode()}" for k, v in feed.items()))
    cmd(f"unload chk{slot}")
    _ph.close(); c.close()
    return {k: np.frombuffer(base64.b64decode(v["b64"]), np.float32).reshape(v["shape"]) for k, v in r["outputs"].items()}


def rowerr(got, ref):   # [R, D] each -> per-row max|got-ref| / max|ref|
    return np.abs(got - ref).max(1) / np.abs(ref).max(1)


variants = a.variants.split(",")
print(f"{'PHONE ' + a.phone + ' ANE' if a.phone else 'units=' + a.units} page={a.page} norm={a.norm} pscale={a.pscale} center={a.center} dump={a.dump}")
print(f"{'layer':>5} {'variant':8} | {'page O rel err: max / median':>28} | {'page lse err max':>16} | {'full O rel err max / med':>25} | {'GPU full max / med':>19} | inf/nan")
summ = {v: [] for v in variants}
for li in [int(x) for x in a.layers.split(",")]:
    scale, q, K, V, mask, gpu = load(li)
    T = q.shape[1]; nkv = K.shape[1]
    vis0 = mask[0] == 0
    n_old = int(vis0.sum()) - 1                                  # row 0 sees the old keys + itself
    assert vis0[:n_old].all() and not vis0[n_old + T:].any()
    npg = n_old // a.page
    # GPU vs fp64 full, and the fp64 full reference per (query head, row)
    ref_full = np.zeros((24, T, D)); gerr = []
    for h in range(24):
        g = h // G
        o, _, _, _ = attend(q[h] * scale, K[g], V[g], mask)
        ref_full[h] = o
        gerr.append(rowerr(gpu[:, h, :], o))
    gerr = np.concatenate(gerr)
    if a.stats:
        for g in range(H):
            Vo = V[g, :n_old]; Ko = K[g, :n_old]
            stepd = np.abs(Vo).max(0) / 127; rms = np.sqrt((Vo ** 2).mean(0))
            s = (q[g * G:(g + 1) * G] * scale).reshape(-1, D) @ Ko.T
            print(f"  L{li} kvh{g}: |V| max {np.abs(Vo).max():.2f} rms {np.sqrt((Vo**2).mean()):.3f}; per-channel int8 step / channel rms:"
                  f" median {np.median(stepd / rms):.3f} max {np.max(stepd / rms):.3f}; |K| max {np.abs(Ko).max():.2f};"
                  f" scores: std {s.std():.2f} max-mean {np.mean(s.max(1) - s.mean(1)):.1f}")
    for var in variants:
        R = G * T
        # fp64 pieces: per KV head, the rows are (query head, token) in that order
        Qg = [(q[g * G:(g + 1) * G] * scale).reshape(R, D) for g in range(H)]
        perr, lerr, bad = [], [], 0
        pages = []   # per page: o_unnorm [H, R, D], m [H, R], l [H, R]
        t0 = time.time()
        for p in range(npg):
            sl = slice(p * a.page, (p + 1) * a.page)
            raw = [((raw_q8["k"][0][g, sl], raw_q8["k"][1][g, sl]), (raw_q8["v"][0][g, sl], raw_q8["v"][1][g, sl])) for g in range(H)]
            mdl = build([K[g, sl] for g in range(H)], [V[g, sl] for g in range(H)], var, R, raw)
            qin = np.concatenate(Qg, 1).T.reshape(1, H * D, 1, R).astype(np.float16)  # [1, H*D, 1, R]
            feed = {"q": qin}
            if a.center in ("oracle", "prevrow"):
                cm = np.stack([(Qg[g] @ K[g, sl].T).max(1) for g in range(H)])        # [H, R] exact row max, rows (head, token)
                if a.center == "prevrow":                                             # token t gets token t-1's max (t=0: t=1's)
                    cm = cm.reshape(H, G, T); cm = np.concatenate([cm[:, :, 1:2], cm[:, :, :-1]], 2).reshape(H, R)
                cin = cm.astype(np.float16); feed["c"] = cin.reshape(1, H, 1, R)
            out = phone_predict(mdl, p, feed) if a.phone else mdl.predict(feed)
            o = out["o"].astype(np.float64); ml = out["ml"].astype(np.float64)
            nm = 3 if a.center == "twopass" else 2
            bad += int((~np.isfinite(o)).sum() + (~np.isfinite(ml)).sum())
            oo = np.zeros((H, R, D)); mm = np.zeros((H, R)); ll = np.zeros((H, R))
            for g in range(H):
                og = o[0, g * D:(g + 1) * D, 0, :].T; m_ = ml[0, nm * g, 0, :]; l_ = ml[0, nm * g + 1, 0, :]
                if a.center == "twopass": m_ = m_ + ml[0, nm * g + 2, 0, :]
                elif a.center != "none": m_ = m_ + cin[g].astype(np.float64)
                on = og if a.norm else og / a.pscale if a.pscale else og / l_[:, None]
                ref, rm, rl, _ = attend(Qg[g], K[g, sl], V[g, sl])
                perr.append(rowerr(on, ref))
                lerr.append(np.abs((m_ + np.log(l_)) - (rm + np.log(rl))))
                oo[g] = on * l_[:, None]; mm[g] = m_; ll[g] = l_
            pages.append((oo, mm, ll))
        # merge pages + fp64 rest (old keys past the pages, and the new keys under the mask)
        ferr = []
        for g in range(H):
            rest = slice(npg * a.page, nkv)
            bias = np.repeat(mask[None, :, rest], G, 0).reshape(R, -1)   # rows (head, token): same mask for every head
            _, rm, rl, ro = attend(Qg[g], K[g, rest], V[g, rest], bias)
            ms_ = [rm] + [pg[1][g] for pg in pages]; M = np.max(ms_, 0)
            L = rl * np.exp(rm - M); O = ro * np.exp(rm - M)[:, None]
            for oo, mm, ll in pages:
                w = np.exp(mm[g] - M); L += ll[g] * w; O += oo[g] * w[:, None]
            ferr.append(rowerr(O / L[:, None], ref_full[g * G:(g + 1) * G].reshape(R, D)))
        perr = np.concatenate(perr); lerr = np.concatenate(lerr); ferr = np.concatenate(ferr)
        summ[var].append((perr.max(), lerr.max(), ferr.max(), gerr.max()))
        print(f"{li:>5} {var:8} | {perr.max():12.2e} / {np.median(perr):9.2e}     | {lerr.max():16.2e} | {ferr.max():11.2e} / {np.median(ferr):9.2e}"
              f"   | {gerr.max():8.2e} / {np.median(gerr):8.2e} | {bad}   ({time.time() - t0:.0f} s, {npg} pages)", flush=True)
print("worst over layers (page O, page lse, full O, GPU full):")
for v, r in summ.items():
    if r: print(f"  {v:8} " + "  ".join(f"{x:.2e}" for x in np.max(r, 0)))
