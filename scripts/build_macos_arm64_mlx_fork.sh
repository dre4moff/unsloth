#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
app_version="0.1.800-mlx.3"
backend_version="2026.8.18+mlxcompaction3"
rust_toolchain="1.89.0"
wheel_name="unsloth-${backend_version}-py3-none-any.whl"
resource_dir="$repo_root/studio/src-tauri/resources/backend"
resource_wheel="$resource_dir/$wheel_name"

if [ "$(uname -m)" != "arm64" ]; then
    echo "This release script must run on Apple Silicon." >&2
    exit 1
fi

for required in uv npm rustup shasum unzip rg lipo vtool plutil codesign hdiutil; do
    if ! command -v "$required" >/dev/null 2>&1; then
        echo "Missing required build tool: $required" >&2
        exit 1
    fi
done

if ! grep -Fq "version = \"$app_version\"" "$repo_root/studio/src-tauri/Cargo.toml"; then
    echo "Cargo app version does not match $app_version" >&2
    exit 1
fi
if ! grep -Fq "__version__ = \"$backend_version\"" "$repo_root/unsloth/_version.py"; then
    echo "Python backend version does not match $backend_version" >&2
    exit 1
fi
if ! grep -Fq "STUDIO_RELEASE_VERSION = \"v$app_version\"" \
    "$repo_root/studio/backend/utils/_studio_release_build.py"; then
    echo "Studio release stamp does not match v$app_version" >&2
    exit 1
fi

echo "Installing pinned JavaScript dependencies..."
npm ci --prefix "$repo_root/studio" --no-audit --no-fund
npm ci --prefix "$repo_root/studio/frontend" --no-audit --no-fund

echo "Building the frontend that will be included in both the app and backend wheel..."
npm run build --prefix "$repo_root/studio/frontend"

wheel_workspace="$(mktemp -d /tmp/unsloth-mlx-wheel.XXXXXX)"
cleanup() {
    case "$wheel_workspace" in
        /tmp/unsloth-mlx-wheel.*) rm -rf "$wheel_workspace" ;;
    esac
}
trap cleanup EXIT

echo "Building the fork backend wheel..."
if ! uv build --wheel --out-dir "$wheel_workspace" "$repo_root" \
    >"$wheel_workspace/build.log" 2>&1; then
    tail -n 100 "$wheel_workspace/build.log" >&2
    exit 1
fi

built_wheel="$wheel_workspace/$wheel_name"
if [ ! -f "$built_wheel" ]; then
    echo "Expected backend wheel was not produced: $wheel_name" >&2
    find "$wheel_workspace" -maxdepth 1 -type f -print >&2
    exit 1
fi
if ! unzip -p "$built_wheel" studio/backend/core/inference/orchestrator.py \
    | grep -Fq "def compact_chat_context"; then
    echo "Backend wheel does not contain MLX context compaction." >&2
    exit 1
fi
if ! unzip -p "$built_wheel" studio/backend/requirements/studio.txt \
    | grep -Fxq "psutil==7.2.2"; then
    echo "Backend wheel does not contain the required psutil runtime pin." >&2
    exit 1
fi
if ! unzip -p "$built_wheel" studio/backend/utils/_studio_release_build.py \
    | grep -Fq "STUDIO_RELEASE_VERSION = \"v$app_version\""; then
    echo "Backend wheel release stamp does not match v$app_version." >&2
    exit 1
fi

mkdir -p "$resource_dir"
# Tauri bundles the whole resource directory. Keep exactly one backend wheel or
# successive fork builds silently carry every older backend inside the app.
while IFS= read -r stale_wheel; do
    if [ "$stale_wheel" != "$resource_wheel" ]; then
        rm -f -- "$stale_wheel"
    fi
done < <(find "$resource_dir" -maxdepth 1 -type f -name 'unsloth-*.whl' -print)
rm -f -- "$resource_wheel"
install -m 0644 "$built_wheel" "$resource_wheel"
wheel_sha256="$(shasum -a 256 "$resource_wheel" | awk '{print $1}')"

echo "Preparing the Apple Silicon Rust target..."
rustup target add --toolchain "$rust_toolchain" aarch64-apple-darwin
rust_sysroot="$(rustc +"$rust_toolchain" --print sysroot)"
rust_lld_dir="$rust_sysroot/lib/rustlib/aarch64-apple-darwin/bin/gcc-ld"
if [ ! -x "$rust_lld_dir/ld64.lld" ]; then
    echo "The pinned Rust toolchain does not provide ld64.lld." >&2
    exit 1
fi
rustc_wrapper="$repo_root/scripts/rustc_macos_lld_wrapper.sh"
linker_wrapper="$repo_root/scripts/macos_lld_linker.sh"
if [ ! -x "$rustc_wrapper" ] || [ ! -x "$linker_wrapper" ]; then
    echo "The macOS Rust/LLD wrapper scripts are not executable." >&2
    exit 1
fi

echo "Building the same-identity app and DMG..."
(
    cd "$repo_root/studio"
    export MACOSX_DEPLOYMENT_TARGET=12.0
    export RUSTUP_TOOLCHAIN="$rust_toolchain"
    export PATH="$rust_lld_dir:$PATH"
    export RUSTC_WRAPPER="$rustc_wrapper"
    export UNSLOTH_MACOS_LLD_LINKER="$linker_wrapper"
    export RUSTFLAGS="${RUSTFLAGS:+${RUSTFLAGS} }--remap-path-prefix=${HOME:?}=/source"
    export UNSLOTH_DESKTOP_BACKEND_VERSION="$backend_version"
    export UNSLOTH_BUNDLED_BACKEND_EXACT_VERSION="$backend_version"
    export UNSLOTH_BUNDLED_BACKEND_WHEEL="$wheel_name"
    export UNSLOTH_BUNDLED_BACKEND_SHA256="$wheel_sha256"
    export UNSLOTH_DISABLE_DESKTOP_UPDATES=1
    npx tauri build -v \
        --target aarch64-apple-darwin \
        --bundles app,dmg
)

app_path="$repo_root/studio/src-tauri/target/aarch64-apple-darwin/release/bundle/macos/Unsloth.app"
app_binary="$app_path/Contents/MacOS/unsloth-studio"
dmg_path="$repo_root/studio/src-tauri/target/aarch64-apple-darwin/release/bundle/dmg/Unsloth_${app_version}_aarch64.dmg"

echo "Verifying release metadata and integrity..."
if [ "$(lipo -archs "$app_binary")" != "arm64" ]; then
    echo "Release binary is not arm64-only." >&2
    exit 1
fi
if ! vtool -show-build "$app_binary" | grep -Eq 'minos +12\.0'; then
    echo "Release binary does not target macOS 12.0." >&2
    exit 1
fi
if [ "$(plutil -extract LSMinimumSystemVersion raw "$app_path/Contents/Info.plist")" != "12.0" ]; then
    echo "Release Info.plist does not require macOS 12.0." >&2
    exit 1
fi
if [ "$(plutil -extract CFBundleIdentifier raw "$app_path/Contents/Info.plist")" != "ai.unsloth.studio" ]; then
    echo "Release bundle identifier does not match the official app." >&2
    exit 1
fi
if [ "$(plutil -extract CFBundleDisplayName raw "$app_path/Contents/Info.plist")" != "Unsloth" ]; then
    echo "Release display name does not match the official app." >&2
    exit 1
fi
if rg -a -l -F "$HOME" "$app_path" >/dev/null; then
    echo "Release app contains a local home-directory path." >&2
    exit 1
fi
if [ "$(find "$app_path/Contents/Resources/backend" -maxdepth 1 -type f -name 'unsloth-*.whl' | wc -l | tr -d ' ')" != "1" ]; then
    echo "Release app must contain exactly one backend wheel." >&2
    exit 1
fi
codesign --verify --deep --strict --verbose=2 "$app_path"
hdiutil verify "$dmg_path"

echo "Bundled backend SHA-256: $wheel_sha256"
echo "App: $app_path"
echo "DMG: $dmg_path"
