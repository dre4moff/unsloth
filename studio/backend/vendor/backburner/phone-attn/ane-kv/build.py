#!/usr/bin/env python3
"""Old KV as Neural Engine weights: one CoreML model = one layer's page of N frozen keys, all 4 KV heads.
Input  q  [1, 4*256, 1, R] fp16 (R query rows per KV head: 6 GQA heads x verify width; scale already folded in)
Output o  [1, 4*256, 1, R] unnormalized P.V, and ml [1, 8, 1, R] (per head: running max m, sum l) for the log-sum-exp merge.
K^T and V are the conv weights (static), so S = Q K^T and O = P V are the ANE's favourite op: 1x1 conv with constant weights.
  build.py OUT_DIR --keys N --rows R --wq {fp16,int8ch,int8blk32} [--pfix] [--center none|const|twopass] [--check]
--pfix: P = e/l scaled by 256 before the P.V conv (the ANE drops fp16-subnormal products otherwise; check_real.py), o = 256 O/l.
--center: scores as q.k - c from the conv (257 input channels): const = c baked in as a constant (same ops as the real
  engine's c input), twopass = c = the model's own first-pass row max, input = c is a second input [1, H, 1, R] (the real
  engine: the previous step's m; Sidecar's timing now feeds every input).
Accuracy on real KV: check_real.py (int8ch fails; fp16 + --pfix + --center is the accurate candidate).
Writes OUT_DIR/kv_N{N}_R{R}_{wq}.mlpackage and, via coremlcompiler, the .mlmodelc Sidecar loads (`ane load NAME FILE ne`)."""
import argparse, os, subprocess, sys, time
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--keys", type=int, default=4096); ap.add_argument("--rows", type=int, default=30)
ap.add_argument("--wq", default="int8ch", choices=["fp16", "int8ch", "int8blk32"]); ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--pfix", action="store_true"); ap.add_argument("--center", default="none", choices=["none", "const", "twopass", "input"])
ap.add_argument("--check", action="store_true", help="run on the Mac (CPU) and compare with numpy")
a = ap.parse_args()
N, R, H, D = a.keys, a.rows, a.heads, 256
rng = np.random.default_rng(1)
K = (rng.standard_normal((H, N, D)) * 1.5).astype(np.float16)
V = rng.standard_normal((H, N, D)).astype(np.float16)

specs = [mb.TensorSpec(shape=(1, H * D, 1, R), dtype=types.fp16)]
if a.center == "input":
    specs.append(mb.TensorSpec(shape=(1, H, 1, R), dtype=types.fp16))

def body(q, cin):
    os_, ms = [], []
    for h in range(H):
        qh = mb.slice_by_index(x=q, begin=[0, h * D, 0, 0], end=[1, (h + 1) * D, 1, R])
        if a.center == "none" or a.center == "twopass":
            s = mb.conv(x=qh, weight=K[h].reshape(N, D, 1, 1))                  # [1, N, 1, R]
        if a.center != "none":
            c = (mb.reduce_max(x=s, axes=[1], keep_dims=True) if a.center == "twopass" else
                 mb.slice_by_index(x=cin, begin=[0, h, 0, 0], end=[1, h + 1, 1, R]) if a.center == "input" else
                 np.full((1, 1, 1, R), 12.0, np.float16))
            qa = mb.concat(values=[qh, mb.mul(x=c, y=np.float16(-1))], axis=1)
            s = mb.conv(x=qa, weight=np.concatenate([K[h], np.ones((N, 1), np.float16)], 1).reshape(N, D + 1, 1, 1))
        m = mb.reduce_max(x=s, axes=[1], keep_dims=True)                          # [1, 1, 1, R]
        e = mb.exp(x=mb.sub(x=s, y=m))
        l = mb.reduce_sum(x=e, axes=[1], keep_dims=True)
        if a.pfix:
            e = mb.mul(x=mb.real_div(x=e, y=l), y=np.float16(256))
        o = mb.conv(x=e, weight=np.ascontiguousarray(V[h].T).reshape(D, N, 1, 1))  # [1, D, 1, R]
        os_.append(o); ms += [m, l]
    return mb.concat(values=os_, axis=1, name="o"), mb.concat(values=ms, axis=1, name="ml")

if a.center == "input":
    prog = mb.program(input_specs=specs, opset_version=ct.target.iOS18)(lambda q, c: body(q, c))
else:
    prog = mb.program(input_specs=specs, opset_version=ct.target.iOS18)(lambda q: body(q, None))

t0 = time.time()
m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16)
if a.wq != "fp16":
    from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights
    kw = dict(mode="linear_symmetric", dtype="int8", weight_threshold=0)
    kw.update(granularity="per_channel") if a.wq == "int8ch" else kw.update(granularity="per_block", block_size=32)
    m = linear_quantize_weights(m, OptimizationConfig(global_config=OpLinearQuantizerConfig(**kw)))
t_conv = time.time() - t0
os.makedirs(a.out, exist_ok=True)
name = f"kv_N{N}_R{R}_{a.wq}" + ("_pfix" if a.pfix else "") + ("" if a.center == "none" else "_c" + a.center)
pkg = os.path.join(a.out, name + ".mlpackage"); m.save(pkg)
t0 = time.time()
subprocess.run(["xcrun", "coremlcompiler", "compile", pkg, a.out], check=True, stdout=subprocess.DEVNULL)
t_comp = time.time() - t0
mb_w = H * N * D * 2 * (1 if a.wq != "fp16" else 2) / 1e6
print(f"{name}: weights {mb_w:.1f} MB, convert+quantize {t_conv:.1f} s, coremlcompiler {t_comp:.1f} s -> {a.out}/{name}.mlmodelc")

if a.check:
    q = (rng.standard_normal((1, H * D, 1, R)) * 0.1).astype(np.float16)
    cu = ct.ComputeUnit.CPU_AND_NE if os.environ.get("ANEKV_NE") else ct.ComputeUnit.CPU_ONLY
    mc = ct.models.MLModel(pkg, compute_units=cu)
    feed = {"q": q}
    if a.center == "input": feed["c"] = np.zeros((1, H, 1, R), np.float16)
    out = mc.predict(feed); o, ml = out["o"], out["ml"]
    err = 0
    for h in range(H):
        qh = q[0, h * D:(h + 1) * D, 0, :].astype(np.float64)                  # [D, R]
        s = K[h].astype(np.float64) @ qh                                          # [N, R]
        mm = s.max(0); e = np.exp(s - mm); ll = e.sum(0); oo = V[h].astype(np.float64).T @ e / ll
        got = o[0, h * D:(h + 1) * D, 0, :] / (256.0 if a.pfix else ml[0, 2 * h + 1, 0, :])
        err = max(err, float(np.abs(got - oo).max() / np.abs(oo).max()))
    assert a.center != "twopass", "--check reads ml as (m, l) pairs"
    print(f"check vs fp64 ({cu.name}, {a.wq}): max rel err {err:.2e}")
