#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "rustc wrapper did not receive the compiler path." >&2
    exit 1
fi

real_rustc="$1"
shift
: "${UNSLOTH_MACOS_LLD_LINKER:?Missing Unsloth macOS linker wrapper path}"

exec "$real_rustc" "$@" -C "linker=$UNSLOTH_MACOS_LLD_LINKER"
