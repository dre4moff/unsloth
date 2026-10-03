#!/usr/bin/env bash
# Builds pa-tool (macOS) and the iOS object of the SME kernel that the app links.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; ROOT="$D/.."
mkdir -p "$D/build"
nice -n 15 xcrun clang -c -O3 -mcpu=apple-m4 -DSME_BENCH_MAIN -DSME_BENCH_NO_MAIN "$ROOT/scripts/sme/sme_attn.c" -o "$D/build/sme_attn_mac.o"
nice -n 15 xcrun clang++ -fobjc-arc -std=c++17 -O2 -mcpu=apple-m4 -x objective-c++ "$D/pa-tool.cpp" -x none "$D/build/sme_attn_mac.o" \
  -framework Foundation -framework CoreML -framework Metal -o "$D/build/pa-tool"
SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
nice -n 15 xcrun -sdk iphoneos clang -c -O3 -isysroot "$SDK" -target arm64-apple-ios16.4 -mcpu=apple-a18 \
  -DSME_BENCH_MAIN -DSME_BENCH_NO_MAIN "$ROOT/scripts/sme/sme_attn.c" -o "$ROOT/ios/Backburner/Sidecar/sme_attn_ios.o"
echo "built $D/build/pa-tool and ios/Backburner/Sidecar/sme_attn_ios.o"
