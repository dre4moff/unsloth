# Move a downloaded model to another disk

Open the model's row menu and choose **Move to another disk…** (**Sposta su un altro disco…** in Italian). Unload that model, choose an existing folder on the destination disk, then confirm. Studio creates an `Unsloth Models/hub` library inside that folder. This moves the complete Hugging Face repository, including every downloaded quantization, revision and support file within that repository. Separate companion repositories and externally managed Ollama/LM Studio models are not moved by this action.

The dialog reports copying and verification progress and supports cancellation before finalization. It can be closed and reopened while the app stays running. New downloads continue to use the existing default download folder; that preference remains in Settings → Resources. The same action can move a model back to an internal disk.

When weights remain resident in RAM/VRAM, slower storage mainly affects loading. Memory-mapped weights, on-demand tensors/experts, disk offload and memory pressure can also affect generation speed. Model format alone does not establish residency. Keep the disk connected during use. See the [llama.cpp memory-mapping implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-mmap.cpp) for the distinction between mapped and locked memory.

## Transfer and recovery

- Same-filesystem transfers use a rename where internal links remain valid.
- Cross-filesystem transfers copy into a hidden staging directory, flush file writes and verify SHA-256 before publishing the destination and removing the source.
- Relative and absolute links inside a repository are preserved/rebased. On filesystems such as exFAT without symlinks, snapshots are materialized and the additional required space is checked.
- Insufficient space, copy/verification failure or cancellation before finalization leaves the original model intact. A cleanup failure after publication retains the verified destination and reports the remaining original path.
- A forced app termination can leave a `.unsloth-move-*` staging directory on the destination. The original is retained until commit; after commit the verified destination is registered. Transfers do not automatically resume after process termination.
- Managed downloads/updates/deletions cannot overlap the move. Chat loads use the inference lifecycle gate; media loads and training starts are excluded during the filesystem transaction. Already-loaded models and active training block relocation.
- Relocated roots persist separately from the limited recent-folder history. Old repo identifiers and saved local paths resolve to the relocated model. An unavailable disk produces a reconnect error instead of silently downloading the model again.

The implementation is covered by filesystem fault-injection, settings, background-job and cache-snapshot tests. A physical external-drive transfer and actual model inference from that drive still require device-level acceptance. The changes do not themselves export or install a new desktop application.

## Validation for this change

- 57 targeted backend tests passed across `test_model_relocation.py`, `test_hf_cache_settings.py` and `test_model_cache_snapshot.py`.
- Frontend production build and full typecheck passed; modified Python files passed Ruff.
- The additional `test_cache_case_resolution.py` check has four existing failures, reproduced with the unchanged cache-settings module from the base commit. They are outside this change.
