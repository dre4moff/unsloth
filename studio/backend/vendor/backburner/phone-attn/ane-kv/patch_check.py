#!/usr/bin/env python3
"""M1 of docs/ANE.md: can a page model be made by rewriting a TEMPLATE .mlmodelc's weights/weight.bin (what the
phone would do) instead of a fresh coremltools build? Builds template A (random K/V), fresh B (other K/V), patches a copy of A
with B's K'/V^T at the blob offsets named in model.mil, then on the Mac ANE (bit-identical to the A18's) compares patched vs
fresh B outputs bit for bit on the same inputs. Also checks the patched weight.bin equals B's byte for byte.
  /usr/local/bin/python3 patch_check.py [--keys 16384] [--rows 48]"""
import argparse, os, re, shutil, struct, subprocess, tempfile
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

ap = argparse.ArgumentParser(); ap.add_argument("--keys", type=int, default=16384); ap.add_argument("--rows", type=int, default=48)
a = ap.parse_args()
N, R, H, D = a.keys, a.rows, 4, 256


def compiled(K, V, out, name):
    """the engine's page model (fp16, P fix, c input) with these K, V [H, N, D] -> path of the .mlmodelc"""
    def body(q, cin):
        os_, ms = [], []
        for h in range(H):
            qh = mb.slice_by_index(x=q, begin=[0, h * D, 0, 0], end=[1, (h + 1) * D, 1, R])
            c = mb.slice_by_index(x=cin, begin=[0, h, 0, 0], end=[1, h + 1, 1, R])
            qa = mb.concat(values=[qh, mb.mul(x=c, y=np.float16(-1))], axis=1)
            s = mb.conv(x=qa, weight=np.concatenate([K[h], np.ones((N, 1), np.float16)], 1).reshape(N, D + 1, 1, 1))
            m = mb.reduce_max(x=s, axes=[1], keep_dims=True)
            e = mb.exp(x=mb.sub(x=s, y=m))
            l = mb.reduce_sum(x=e, axes=[1], keep_dims=True)
            p = mb.mul(x=mb.real_div(x=e, y=l), y=np.float16(256))
            os_.append(mb.conv(x=p, weight=np.ascontiguousarray(V[h].T).reshape(D, N, 1, 1)))
            ms += [m, l]
        return mb.concat(values=os_, axis=1, name="o"), mb.concat(values=ms, axis=1, name="ml")
    specs = [mb.TensorSpec(shape=(1, H * D, 1, R), dtype=types.fp16), mb.TensorSpec(shape=(1, H, 1, R), dtype=types.fp16)]
    prog = mb.program(input_specs=specs, opset_version=ct.target.iOS18)(lambda q, c: body(q, c))
    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16)
    pkg = os.path.join(out, name + ".mlpackage"); m.save(pkg)
    subprocess.run(["xcrun", "coremlcompiler", "compile", pkg, out], check=True, stdout=subprocess.DEVNULL)
    return os.path.join(out, name + ".mlmodelc")


def patch(template, dst, K, V):
    """copy template -> dst and write K' = [K | 1] and V^T of every head into weight.bin at model.mil's offsets"""
    shutil.copytree(template, dst)
    mil = open(os.path.join(dst, "model.mil")).read()
    blobs = {n: int(o) for n, o in re.findall(r"(conv_\d+_weight_0) = const\(\)\[.*?offset = uint64\((\d+)\)", mil)}
    assert len(blobs) == 2 * H, blobs
    with open(os.path.join(dst, "weights", "weight.bin"), "r+b") as f:
        for h in range(H):
            for idx, arr in ((2 * h, np.concatenate([K[h], np.ones((N, 1), np.float16)], 1)), (2 * h + 1, np.ascontiguousarray(V[h].T))):
                off = blobs[f"conv_{idx}_weight_0"]
                f.seek(off); sentinel, dtype, size, data_off = struct.unpack("<IIQQ", f.read(24))
                assert sentinel == 0xDEADBEEF and dtype == 1 and size == arr.nbytes, (hex(sentinel), dtype, size, arr.nbytes)
                f.seek(data_off); f.write(arr.astype(np.float16).tobytes())
    return dst


rng = np.random.default_rng(7)
KA, VA = (rng.standard_normal((H, N, D)) * 1.5).astype(np.float16), rng.standard_normal((H, N, D)).astype(np.float16)
KB, VB = (rng.standard_normal((H, N, D)) * 1.5).astype(np.float16), rng.standard_normal((H, N, D)).astype(np.float16)
tmp = tempfile.mkdtemp(prefix="anekv-patch-")
try:
    A = compiled(KA, VA, tmp, "tmplA"); B = compiled(KB, VB, tmp, "freshB")
    P = patch(A, os.path.join(tmp, "patchedB.mlmodelc"), KB, VB)
    same_bin = open(os.path.join(P, "weights/weight.bin"), "rb").read() == open(os.path.join(B, "weights/weight.bin"), "rb").read()
    print(f"patched weight.bin == fresh B weight.bin: {same_bin}")
    q = (rng.standard_normal((1, H * D, 1, R)) * 0.1).astype(np.float16)
    c = (rng.standard_normal((1, H, 1, R)) * 2).astype(np.float16)
    outs = {}
    for tag, path in (("template A", A), ("fresh B", B), ("patched B", P)):
        mdl = ct.models.CompiledMLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        outs[tag] = mdl.predict({"q": q, "c": c})
    for k in ("o", "ml"):
        pb, fb, ta = outs["patched B"][k], outs["fresh B"][k], outs["template A"][k]
        print(f"{k}: patched vs fresh B bit-identical: {np.array_equal(pb.view(np.uint8), fb.view(np.uint8))}"
              f"   (max |diff| {float(np.abs(pb.astype(np.float64) - fb).max()):.3g}; vs template A max |diff| {float(np.abs(pb.astype(np.float64) - ta).max()):.3g})")
finally:
    shutil.rmtree(tmp)
