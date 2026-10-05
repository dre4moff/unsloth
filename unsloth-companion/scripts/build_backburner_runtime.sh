#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# MIT upstream engine by StayLameBro; Studio adapter under Apache-2.0.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
vendor="$repo_root/studio/backend/vendor/backburner"
work_root="${UNSLOTH_BACKBURNER_BUILD_ROOT:-/private/tmp/unsloth-backburner-runtime-build}"
engine="$work_root/llama.cpp"
commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["engine_commit"])' "$vendor/UPSTREAM.json")"
mkdir -p "$work_root"
if [ ! -e "$engine/.git" ]; then
    git init "$engine"
    git -C "$engine" remote add origin https://github.com/StayLameBro/backburner-llama.cpp.git
fi
git -C "$engine" fetch --depth 1 origin "$commit"
git -C "$engine" checkout --detach "$commit"
test "$(git -C "$engine" rev-parse HEAD)" = "$commit"
# Backburner 0.0.4 includes bounded remote-worker shutdown upstream.
# Build its engine unchanged; both sides use the same v4 protocol header.
cmp "$vendor/phone-attn/phone-attn.h" "$engine/ggml/src/ggml-metal/phone-attn.h"
source_map="-ffile-prefix-map=$repo_root=unsloth -ffile-prefix-map=$work_root=backburner"
cmake -S "$engine" -B "$work_root/mac" -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_UI_GZIP=OFF -DLLAMA_OPENSSL=OFF \
    -DGGML_NATIVE=OFF -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
    -DCMAKE_C_FLAGS="$source_map" -DCMAKE_CXX_FLAGS="$source_map" -DCMAKE_OBJC_FLAGS="$source_map"
cmake --build "$work_root/mac" --target llama-server llama-quantize -j "${UNSLOTH_BACKBURNER_BUILD_JOBS:-6}"
mkdir -p "$vendor/runtime/bin"
cp "$work_root/mac/bin/llama-server" "$work_root/mac/bin/llama-quantize" "$vendor/runtime/bin/"
codesign -f -s - "$vendor/runtime/bin/llama-server"
codesign -f -s - "$vendor/runtime/bin/llama-quantize"
if rg -a -l -F "${HOME:?}" "$vendor/runtime/bin"; then
    echo "Backburner Mac binaries contain a build-machine home path." >&2; exit 1
fi
if [ "${1:-}" = "--mac-only" ]; then
    python3 - "$vendor" <<'MAC_MANIFEST'
import hashlib, json, sys
from pathlib import Path
vendor = Path(sys.argv[1]); root = vendor / 'runtime'
manifest = json.loads((root / 'MANIFEST.json').read_text())
manifest['engineCommit'] = json.loads((vendor / 'UPSTREAM.json').read_text())['engine_commit']
manifest.pop('macLifecyclePatch', None)
for path in (root / 'bin').iterdir():
    if path.is_file(): manifest['files'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
MAC_MANIFEST
    exit 0
fi

native="$work_root/native"
python3 "$repo_root/unsloth-companion/scripts/adapt_backburner_ios.py" "$native" "$engine"
cmake -S "$engine" -B "$work_root/ios" -G Xcode \
    -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
    -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=18.6 \
    -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_COMMON=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_OPENSSL=OFF \
    -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_METAL_TARGET_OS=ios \
    -DGGML_BLAS_DEFAULT=ON -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DGGML_RPC=ON \
    -DGGML_RPC_RDMA=OFF -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
    -DCMAKE_C_FLAGS="$source_map" -DCMAKE_CXX_FLAGS="$source_map" -DCMAKE_OBJC_FLAGS="$source_map"
cmake --build "$work_root/ios" --config Release -j 6 -- -quiet
sdkroot="$(xcrun --sdk iphoneos --show-sdk-path)"
phone_sme=(sme_ws_new sme_attn_partial sme_pipe_new sme_attn_pipe sme_pipe_worker sme_pipe_worker2 sme_pipe_helper sme_pipe_softmax_helper sme_pipe_controller sme_timing sme_t_pack sme_t_pv sme_t_qk sme_t_sm sme_t_upd sme2_available sme_bench)
sme_names=()
for symbol in "${phone_sme[@]}"; do sme_names+=("-D${symbol}=bb_${symbol}"); done
xcrun -sdk iphoneos clang -c -O3 -isysroot "$sdkroot" -target arm64-apple-ios18.6 \
    -mcpu=apple-a18 -DSME_BENCH_MAIN -DSME_BENCH_NO_MAIN \
    "-ffile-prefix-map=$repo_root=unsloth" \
    "${sme_names[@]}" \
    "$vendor/scripts/sme/sme_attn.c" -o "$work_root/sme.o"
xcrun -sdk iphoneos clang++ -c -std=c++17 -O3 -fobjc-arc -arch arm64 \
    -mios-version-min=18.6 -isysroot "$sdkroot" \
    -I"$engine/include" -I"$engine/ggml/include" \
    "-ffile-prefix-map=$work_root=backburner" \
    "${sme_names[@]}" \
    "$native/ios/Backburner/Sidecar/RPCBridge.mm" -o "$work_root/bridge.o"
framework="$work_root/Backburner.framework"
mkdir -p "$framework/Headers" "$framework/Modules"
cp "$native/ios/Backburner/Sidecar/RPCBridge.h" "$framework/Headers/Backburner.h"
cat > "$framework/Modules/module.modulemap" <<'MAP'
framework module Backburner { umbrella header "Backburner.h" export * }
MAP
cat > "$framework/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?><plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>Backburner</string>
<key>CFBundleIdentifier</key><string>com.OvenTeam.BackburnerRuntime</string>
<key>CFBundleName</key><string>Backburner</string>
<key>CFBundlePackageType</key><string>FMWK</string>
<key>CFBundleShortVersionString</key><string>1.0</string>
<key>CFBundleVersion</key><string>1</string>
<key>MinimumOSVersion</key><string>18.6</string>
<key>CFBundleSupportedPlatforms</key><array><string>iPhoneOS</string></array>
</dict></plist>
PLIST
cat > "$work_root/exports.txt" <<'EXPORTS'
_OBJC_CLASS_$_SidecarRPC
_OBJC_METACLASS_$_SidecarRPC
EXPORTS
libraries=()
while IFS= read -r library; do libraries+=("$library"); done < <(find "$work_root/ios" -path '*/Release-iphoneos/*.a' -type f)
xcrun libtool -static -o "$work_root/combined.a" "${libraries[@]}"
xcrun -sdk iphoneos clang++ -dynamiclib -arch arm64 -mios-version-min=18.6 \
    -isysroot "$sdkroot" "$work_root/bridge.o" "$work_root/sme.o" \
    -Wl,-force_load,"$work_root/combined.a" -Wl,-exported_symbols_list,"$work_root/exports.txt" \
    -framework Foundation -framework Metal -framework Accelerate -framework CoreML -framework Security \
    -install_name @rpath/Backburner.framework/Backburner -o "$framework/Backburner"
xcrun strip -S -x "$framework/Backburner"
if rg -a -l -F "${HOME:?}" "$framework/Backburner"; then
    echo "Backburner iPhone framework contains a build-machine home path." >&2; exit 1
fi
sim_framework="$work_root/simulator/Backburner.framework"
mkdir -p "$sim_framework/Headers" "$sim_framework/Modules"
cp "$framework/Headers/Backburner.h" "$sim_framework/Headers/"
cp "$framework/Modules/module.modulemap" "$sim_framework/Modules/"
cp "$framework/Info.plist" "$sim_framework/Info.plist"
/usr/libexec/PlistBuddy -c 'Set :CFBundleSupportedPlatforms:0 iPhoneSimulator' "$sim_framework/Info.plist"
# The simulator has no iPhone USB network/GPU/ANE. This explicit unavailable
# adapter keeps unit/UI tests runnable; it never pretends to accelerate.
cat > "$work_root/simulator.mm" <<'SIM'
#import "Backburner.h"
@implementation SidecarRPC
+ (void)beginServices {}
+ (void)endServices {}
+ (BOOL)servicesRunning { return NO; }
+ (NSString *)cableAddress { return @""; }
+ (NSString *)deviceModel { return @"Simulator"; }
+ (NSData *)wifiKey { return nil; }
+ (void)setWifiTunnelStatus:(NSString *)status { (void)status; }
+ (NSString *)wifiAddress { return @""; }
+ (NSDictionary *)metalStats { return @{}; }
+ (NSDictionary *)memoryStats { return @{}; }
+ (NSDictionary *)linkStats { return @{}; }
+ (NSDictionary *)tailStatus { return @{ @"state": @"unavailable" }; }
+ (NSString *)startHost:(NSString *)host port:(int)port cacheDir:(NSString *)dir { return @"Requires a physical iPhone"; }
+ (NSString *)startTailPort:(int)port modelPath:(NSString *)path { return @"Requires a physical iPhone"; }
@end
@implementation SidecarRPC (ANE)
+ (void)startANEBenchPort:(int)port {}
+ (void)startPhoneAttnPort:(int)port {}
+ (NSDictionary *)phoneAttnStatus { return @{}; }
+ (NSDictionary *)macStatus { return @{}; }
+ (NSString *)envNote { return @"Simulator: acceleration unavailable"; }
@end
SIM
xcrun -sdk iphonesimulator clang++ -dynamiclib -fobjc-arc -arch arm64 -arch x86_64 \
    -mios-simulator-version-min=18.6 -isysroot "$(xcrun --sdk iphonesimulator --show-sdk-path)" \
    -I"$framework/Headers" "$work_root/simulator.mm" -framework Foundation \
    -install_name @rpath/Backburner.framework/Backburner -o "$sim_framework/Backburner"
output="$repo_root/unsloth-companion/Unsloth Companion/Unsloth Companion/Vendor/Backburner.xcframework"
if [ -d "$output" ]; then /usr/bin/trash "$output"; fi
xcodebuild -create-xcframework -framework "$framework" -framework "$sim_framework" -output "$output"
if [ "${1:-}" != "--reuse-template" ]; then
    coreml_python="${UNSLOTH_BACKBURNER_COREML_PYTHON:-python3}"
    "$coreml_python" "$vendor/phone-attn/ane-kv/build.py" "$work_root/ane-template" \
        --keys 16384 --rows 48 --wq fp16 --pfix --center input
    mkdir -p "$vendor/runtime/anekv"
    ditto "$work_root/ane-template/kv_N16384_R48_fp16_pfix_cinput.mlmodelc" "$vendor/runtime/anekv/tmpl16k.mlmodelc"
else
    # The template generator is unchanged in 0.0.4. Verify every existing byte.
    python3 - "$vendor/runtime" <<'TEMPLATE'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
for rel, digest in json.loads((root / 'MANIFEST.json').read_text())['files'].items():
    if rel.startswith('anekv/') and hashlib.sha256((root / rel).read_bytes()).hexdigest() != digest:
        raise SystemExit(f'ANE template integrity check failed: {rel}')
TEMPLATE
fi
python3 - "$vendor" <<'MANIFEST'
import hashlib, json, sys
from pathlib import Path
vendor = Path(sys.argv[1]); root = vendor / 'runtime'
meta = json.loads((vendor / 'UPSTREAM.json').read_text())
manifest = {'engineCommit': meta['engine_commit'],
            'template': 'Original kv_N16384_R48_fp16_pfix_cinput; coremltools 9.0 / numpy 2.2.6',
            'files': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'MANIFEST.json'}}
(root / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
MANIFEST
echo "Built isolated Backburner runtime at $commit"
