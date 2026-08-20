# Unofficial Apple Silicon fork build

This branch packages the MLX automatic-context-compaction work as a separate
macOS application named **Unsloth MLX Context**. Its bundle identifier and deep-link
scheme differ from the official app, so installing it does not replace the official
Unsloth Desktop bundle.

The build also closes an important desktop packaging gap: the official Tauri wrapper
normally installs the released `unsloth` package from PyPI on first launch. This fork
instead builds a wheel from the current revision, embeds it in the app, verifies its
SHA-256 before installation, and requires that exact backend version at preflight.
The shipped UI and backend therefore implement the same source revision.

## Build

Run on Apple Silicon macOS:

```bash
./scripts/build_macos_arm64_mlx_fork.sh
```

The script installs the pinned JavaScript dependencies, builds the frontend, creates
the local backend wheel, embeds it as a Tauri resource, targets macOS 12.0 or newer,
and produces `.app` and `.dmg` bundles for `aarch64-apple-darwin`.

Generated backend wheels under `studio/src-tauri/resources/backend/` are ignored by
Git. They are reproducible build inputs rather than source files.

## Update and signing policy

Automatic desktop updates are disabled in this unofficial build. This prevents the
fork from silently replacing itself with an official upstream binary. Fork releases
are installed manually from the GitHub Releases page.

Local artifacts are ad-hoc signed when no Apple Developer ID certificate is available.
Ad-hoc signing provides bundle integrity but is not Apple notarization; Gatekeeper may
require the user to approve the first launch manually.
