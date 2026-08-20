# Unofficial Apple Silicon fork build

This branch packages the MLX automatic-context-compaction work with the official
macOS identity: **Unsloth**, bundle identifier `ai.unsloth.studio`, and deep-link
scheme `unsloth`. It therefore reuses the official app-support data and the existing
`~/.unsloth` model/backend cache without creating a second copy. Installing it in
Applications replaces the official bundle at the same path; the version number is
the visible distinction.

The build also closes an important desktop packaging gap: the official Tauri wrapper
normally installs the released `unsloth` package from PyPI on first launch. This fork
instead builds a wheel from the current revision, embeds it in the app, verifies its
SHA-256 before installation, and requires that exact backend version at preflight.
Existing managed installs are repaired directly from that wheel instead of first
running the generic network updater. The shipped UI and backend therefore implement
the same source revision.

The desktop-launched backend retains the explicit Metal-context behaviour shipped by
the official 2026.8.18 backend: llama.cpp receives the requested context with `--fit`
enabled instead of the later source-only guard refusing it before launch. The Studio
runtime requirements also include `psutil`, so memory reporting is available in a
fresh or repaired managed environment.

## Build

Run on Apple Silicon macOS:

```bash
./scripts/build_macos_arm64_mlx_fork.sh
```

The script installs the pinned JavaScript dependencies, builds the frontend, creates
the local backend wheel, embeds it as a Tauri resource, targets macOS 12.0 or newer,
uses the crate's minimum supported Rust 1.89 toolchain, and produces `.app` and `.dmg`
bundles for `aarch64-apple-darwin`.

The build selects the `ld64.lld` bundled with that Rust toolchain. This avoids a
known beta Apple-linker defect that can emit proc-macro dylibs whose `LINKEDIT`
string table is rejected by the running macOS dynamic loader.

Generated backend wheels under `studio/src-tauri/resources/backend/` are ignored by
Git. They are reproducible build inputs rather than source files.

## Update and signing policy

Automatic desktop updates are disabled in this unofficial build. This prevents the
fork from silently replacing itself with an official upstream binary. Fork releases
are installed manually from the GitHub Releases page.

Local artifacts are ad-hoc signed when no Apple Developer ID certificate is available.
Ad-hoc signing provides bundle integrity but is not Apple notarization; Gatekeeper may
require the user to approve the first launch manually.
