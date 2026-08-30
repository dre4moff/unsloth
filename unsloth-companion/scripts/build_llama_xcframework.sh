#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$ROOT_DIR/Unsloth Companion/Unsloth Companion"
OUT_DIR="$PROJECT_DIR/Vendor"
PINNED_COMMIT="3173a56471c1753650cd806694145ffd6dcace67"
WORK_DIR="${TMPDIR:-/tmp}/unsloth-companion-llama-$PINNED_COMMIT"
SRC_DIR="$WORK_DIR/llama.cpp"
IOS_MIN="18.6"

for tool in git cmake xcrun; do
  command -v "$tool" >/dev/null 2>&1 || { echo "Missing required tool: $tool" >&2; exit 1; }
done

mkdir -p "$WORK_DIR" "$OUT_DIR"
if [[ ! -d "$SRC_DIR/.git" ]]; then
  mkdir -p "$SRC_DIR"
  git -C "$SRC_DIR" init
  git -C "$SRC_DIR" remote add origin https://github.com/ggml-org/llama.cpp
fi
git -C "$SRC_DIR" fetch --depth 1 origin "$PINNED_COMMIT"
git -C "$SRC_DIR" checkout --detach --force "$PINNED_COMMIT"
test "$(git -C "$SRC_DIR" rev-parse HEAD)" = "$PINNED_COMMIT"

COMMON_ARGS=(
  -DBUILD_SHARED_LIBS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_TOOLS=ON
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_SERVER=OFF
  -DGGML_METAL=ON
  -DGGML_METAL_EMBED_LIBRARY=ON
  -DGGML_BLAS_DEFAULT=ON
  -DGGML_METAL_USE_BF16=ON
  -DGGML_NATIVE=OFF
  -DGGML_OPENMP=OFF
  -DLLAMA_OPENSSL=OFF
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY=
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO
)

cmake -S "$SRC_DIR" -B "$SRC_DIR/build-ios-sim" -G Xcode \
  "${COMMON_ARGS[@]}" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_MIN" \
  -DIOS=ON \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphonesimulator \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
  -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=iphonesimulator

cmake --build "$SRC_DIR/build-ios-sim" --config Release \
  --target llama mtmd vendor-hash ggml ggml-base ggml-cpu ggml-metal ggml-blas -- -quiet

cmake -S "$SRC_DIR" -B "$SRC_DIR/build-ios-device" -G Xcode \
  "${COMMON_ARGS[@]}" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_MIN" \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=iphoneos

cmake --build "$SRC_DIR/build-ios-device" --config Release \
  --target llama mtmd vendor-hash ggml ggml-base ggml-cpu ggml-metal ggml-blas -- -quiet

setup_framework() {
  local build_dir="$1"
  local framework="$build_dir/framework/llama.framework"
  rm -rf "$framework"
  mkdir -p "$framework/Headers" "$framework/Modules"
  cp "$SRC_DIR/include/llama.h" "$framework/Headers/"
  cp "$SRC_DIR/ggml/include/ggml.h" "$framework/Headers/"
  cp "$SRC_DIR/ggml/include/ggml-opt.h" "$framework/Headers/"
  cp "$SRC_DIR/ggml/include/ggml-alloc.h" "$framework/Headers/"
  cp "$SRC_DIR/ggml/include/ggml-backend.h" "$framework/Headers/"
  cp "$SRC_DIR/ggml/include/ggml-metal.h" "$framework/Headers/"
  cp "$SRC_DIR/ggml/include/ggml-cpu.h" "$framework/Headers/"
  cp "$SRC_DIR/ggml/include/ggml-blas.h" "$framework/Headers/"
  cp "$SRC_DIR/ggml/include/gguf.h" "$framework/Headers/"
  cp "$SRC_DIR/tools/mtmd/mtmd.h" "$framework/Headers/"
  cp "$SRC_DIR/tools/mtmd/mtmd-helper.h" "$framework/Headers/"
  cat > "$framework/Modules/module.modulemap" <<'MAP'
framework module llama {
  header "llama.h"
  header "ggml.h"
  header "ggml-opt.h"
  header "ggml-alloc.h"
  header "ggml-backend.h"
  header "ggml-metal.h"
  header "ggml-cpu.h"
  header "ggml-blas.h"
  header "gguf.h"
  header "mtmd.h"
  header "mtmd-helper.h"
  link "c++"
  link framework "Accelerate"
  link framework "Metal"
  link framework "Foundation"
  export *
}
MAP
  cat > "$framework/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleDevelopmentRegion</key><string>en</string>
<key>CFBundleExecutable</key><string>llama</string>
<key>CFBundleIdentifier</key><string>org.ggml.llama</string>
<key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
<key>CFBundleName</key><string>llama</string>
<key>CFBundlePackageType</key><string>FMWK</string>
<key>CFBundleShortVersionString</key><string>1.0</string>
<key>CFBundleVersion</key><string>1</string>
<key>MinimumOSVersion</key><string>$IOS_MIN</string>
<key>CFBundleSupportedPlatforms</key><array><string>iPhoneOS</string></array>
<key>UIDeviceFamily</key><array><integer>1</integer></array>
</dict></plist>
PLIST
}

combine_libraries() {
  local build_dir="$1"
  local release_dir="$2"
  local sdk="$3"
  local archs="$4"
  local minimum_flag="$5"
  local combined="$build_dir/combined.a"
  local output="$build_dir/framework/llama.framework/llama"
  local libraries=(
    "$build_dir/src/$release_dir/libllama.a"
    "$build_dir/tools/mtmd/$release_dir/libmtmd.a"
    "$build_dir/ggml/src/$release_dir/libggml.a"
    "$build_dir/ggml/src/$release_dir/libggml-base.a"
    "$build_dir/ggml/src/$release_dir/libggml-cpu.a"
    "$build_dir/ggml/src/ggml-metal/$release_dir/libggml-metal.a"
    "$build_dir/ggml/src/ggml-blas/$release_dir/libggml-blas.a"
    "$build_dir/vendor/hash/$release_dir/libvendor-hash.a"
  )
  xcrun libtool -static -o "$combined" "${libraries[@]}"
  local arch_flags=()
  for arch in $archs; do arch_flags+=("-arch" "$arch"); done
  xcrun -sdk "$sdk" clang++ -dynamiclib \
    -isysroot "$(xcrun --sdk "$sdk" --show-sdk-path)" \
    "${arch_flags[@]}" "$minimum_flag" \
    -Wl,-force_load,"$combined" \
    -framework Foundation -framework Metal -framework Accelerate \
    -install_name @rpath/llama.framework/llama -o "$output"
  mkdir -p "$build_dir/dSYMs"
  xcrun dsymutil "$output" -o "$build_dir/dSYMs/llama.dSYM"
  xcrun strip -S "$output"
  rm -f "$combined"
}

setup_framework "$SRC_DIR/build-ios-sim"
setup_framework "$SRC_DIR/build-ios-device"
combine_libraries "$SRC_DIR/build-ios-sim" Release-iphonesimulator iphonesimulator "arm64 x86_64" "-mios-simulator-version-min=$IOS_MIN"
combine_libraries "$SRC_DIR/build-ios-device" Release-iphoneos iphoneos arm64 "-mios-version-min=$IOS_MIN"

rm -rf "$OUT_DIR/llama.xcframework"
xcrun xcodebuild -create-xcframework \
  -framework "$SRC_DIR/build-ios-sim/framework/llama.framework" \
  -debug-symbols "$SRC_DIR/build-ios-sim/dSYMs/llama.dSYM" \
  -framework "$SRC_DIR/build-ios-device/framework/llama.framework" \
  -debug-symbols "$SRC_DIR/build-ios-device/dSYMs/llama.dSYM" \
  -output "$OUT_DIR/llama.xcframework"

test -f "$OUT_DIR/llama.xcframework/Info.plist"
echo "Built llama.xcframework from $PINNED_COMMIT at $OUT_DIR/llama.xcframework"
