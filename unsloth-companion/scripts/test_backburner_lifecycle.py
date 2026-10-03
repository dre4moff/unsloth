#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Execute the device lifecycle adapter on macOS using real loopback clients.

Requires build_backburner_runtime.sh first. Only the host address, sandbox home,
and iOS-only memory-budget query are replaced. No GGUF/ANE model is loaded here;
this verifies cancellation/ports/environment, not physical iPhone performance.
"""
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    repo = Path(__file__).resolve().parents[2]
    work = Path(os.environ.get("UNSLOTH_BACKBURNER_BUILD_ROOT", "/private/tmp/unsloth-backburner-runtime-build"))
    native, engine = work / "native", work / "llama.cpp"
    source = native / "ios/Backburner/Sidecar/RPCBridge.mm"
    with tempfile.TemporaryDirectory(prefix="unsloth-backburner-lifecycle-") as home:
        text = source.read_text()
        start = text.index('+ (NSString *)cableAddress {')
        end = text.index('\n+ (NSDictionary', start)
        text = text[:start] + '+ (NSString *)cableAddress { return @"127.0.0.1"; }\n' + text[end:]
        text = text.replace('NSHomeDirectory()', '@"' + home + '"')
        text = text.replace('os_proc_available_memory()', '(size_t(8ull << 30))')
        test_source = source.with_name("RPCBridge.test.mm")
        test_source.write_text(text)
        names = "sme_ws_new sme_attn_partial sme_pipe_new sme_attn_pipe sme_pipe_worker sme_pipe_worker2 sme_pipe_helper sme_pipe_softmax_helper sme_pipe_controller sme_timing sme_t_pack sme_t_pv sme_t_qk sme_t_sm sme_t_upd sme2_available sme_bench".split()
        macros = [f"-D{symbol}=bb_{symbol}" for symbol in names]
        sme = work / "test-sme.o"
        subprocess.run(["xcrun", "clang", "-c", "-O3", "-mcpu=apple-a18", "-DSME_BENCH_MAIN", "-DSME_BENCH_NO_MAIN", *macros,
                        str(repo / "studio/backend/vendor/backburner/scripts/sme/sme_attn.c"), "-o", str(sme)], check=True)
        output = work / "lifecycle-test"
        subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc",
                        f"-I{engine / 'include'}", f"-I{engine / 'ggml/include'}", f"-I{native}", *macros,
                        str(test_source), str(repo / "unsloth-companion/tests/BackburnerLifecycleHarness.mm"), str(sme),
                        *map(str, sorted((work / "mac").rglob("*.a"))),
                        "-framework", "Foundation", "-framework", "Metal", "-framework", "Accelerate",
                        "-framework", "CoreML", "-framework", "IOKit", "-o", str(output)], check=True)
        subprocess.run([str(output)], check=True, timeout=180)


if __name__ == "__main__":
    main()
