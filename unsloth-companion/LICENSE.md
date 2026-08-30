# Licensing map

This repository intentionally keeps the licensing boundaries visible:

- The standalone iPhone Companion source, protocol, documentation, design assets,
  and build scripts under `unsloth-companion/` follow the parent repository's
  [Apache License 2.0](../LICENSE), except for third-party material listed below.
- Desktop Studio integration files under `studio/` that carry
  `SPDX-License-Identifier: AGPL-3.0-only` are licensed under
  [AGPL-3.0-only](../studio/LICENSE.AGPL-3.0).
- The bundled `llama.xcframework` and its headers are derived from llama.cpp and
  retain the upstream MIT license notice in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- No model weights are distributed. Model catalog entries identify their upstream
  repositories and licenses; users must review those terms before downloading.

When extracting components into another application, keep the applicable license,
copyright, and third-party notices with the copied files. This summary is a map of
the repository, not legal advice.
