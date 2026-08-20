#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
app_version="0.1.800-mlx.1"
backend_version="2026.8.18+mlxcompaction1"
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

mkdir -p "$resource_dir"
rm -f "$resource_wheel"
install -m 0644 "$built_wheel" "$resource_wheel"
wheel_sha256="$(shasum -a 256 "$resource_wheel" | awk '{print $1}')"

echo "Preparing the Apple Silicon Rust target..."
rustup target add aarch64-apple-darwin

echo "Building the fork-safe app and DMG..."
(
    cd "$repo_root/studio"
    export MACOSX_DEPLOYMENT_TARGET=12.0
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

app_path="$repo_root/studio/src-tauri/target/aarch64-apple-darwin/release/bundle/macos/Unsloth MLX Context.app"
app_binary="$app_path/Contents/MacOS/unsloth-studio"
dmg_path="$repo_root/studio/src-tauri/target/aarch64-apple-darwin/release/bundle/dmg/Unsloth MLX Context_${app_version}_aarch64.dmg"

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
if rg -a -l -F "$HOME" "$app_path" >/dev/null; then
    echo "Release app contains a local home-directory path." >&2
    exit 1
fi
codesign --verify --deep --strict --verbose=2 "$app_path"
hdiutil verify "$dmg_path"

echo "Bundled backend SHA-256: $wheel_sha256"
echo "App: $app_path"
echo "DMG: $dmg_path"
