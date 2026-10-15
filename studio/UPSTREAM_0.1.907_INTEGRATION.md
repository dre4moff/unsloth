# Unsloth 0.1.907 / MLX 43 integration

This release merges the complete upstream `v0.1.907-beta` tree at
`133a8076d10aa109f812f85a953636b3488926f6` into the MLX 42 fork at
`f64be5a899427cb2d560550a476cf876b7544b59`. The shared ancestor is
`ba16c4e9479b8725704862a93a3671b8f8b9dc05`; the upstream side contains 3,310
commits after it. The desktop version is `0.1.907-mlx.43`, and the bundled Python
backend retains upstream's `2026.10.5` base with the fork's local version suffix.

## Upstream implementations adopted

- Local GGUF and MLX compaction, exact worker token counting, branch and checkpoint
  matching, instruction pins, media-aware counting, archive recall, `/compact`,
  compaction markers, API provider compaction, and saved context-meter restoration.
- Browser tools and permissions, chat and history performance, model inventory and
  library controls, image/video/training updates, and all other changes in the tag.
- Installer cache selection and MLX installation, provider reasoning settings,
  credential transactions, desktop lifecycle and process ownership checks.

The old fork `remote_context.py` and context-usage restoration helper were removed.
Their policy tests are covered by upstream's current compaction and context tests.
The obsolete installer-cache helper and partial MLX-phase tests were replaced by
upstream's full cache-selector and MLX installer coverage.

## Fork behavior retained

- Isolated Backburner 0.0.4 / iPhone Speed, wired discovery, the 16 GiB profile,
  selected target and DFlash2 draft, prepared-model identity, separate runtime/KV
  caches, and compaction against admitted total Mac+iPhone context.
- Companion 0.0.5, pairing, per-chat enablement, asynchronous submit/status/collect/wait,
  queued work, compatible worker selection, cancellation, and private credentials.
- Managed Kaggle TPU lifecycle, token storage, model and context capabilities,
  reasoning controls, remote inference with tools executed on the Mac.
- Verified external model relocation, shared-blob handling, disconnected-disk errors,
  restore and identical-cache reconciliation, and exclusion of concurrent loads.
- Visible turn plans and bounded recovery after productive tool actions. Internal
  plan events do not consume the ordinary tool budget. The unlimited UI default
  and an explicitly selected finite cap keep their existing meanings.
- Same app identity, exact bundled backend installation and repair, runtime integrity
  checks, and disabled automatic replacement by upstream desktop binaries.

## Integration safeguards

The private execution checkpoint annotates only the outgoing model prompt. The
semantic conversation retains the message objects used by upstream's branch,
anchor, compaction and recall machinery. Token counting prices the same annotation
that generation receives. Its size scales with the loaded window. Media content
parts remain intact; annotations never replace audio or images with text.

Backburner uses upstream's spawn and healthy-publication lifecycle guards. A
shutdown racing with a Speed load cannot publish a new healthy server. Prepared
weights and the native iPhone engine are unchanged. Companion 0.0.5 is redistributed
byte for byte from the previous verified release.

Tauri 2.12 requires Rust 1.90 or newer. The macOS builder uses installed Rust 1.98.1,
with macOS 12 as the desktop deployment target. The Speed engine still requires
macOS 14; Companion requires iOS 18.6.

Validation and artifact digests are recorded in `release/MLX43-VERIFICATION.json`.
Host tests and packaging checks do not establish new physical iPhone, live Kaggle,
CUDA/Windows/Linux or throughput acceptance.
