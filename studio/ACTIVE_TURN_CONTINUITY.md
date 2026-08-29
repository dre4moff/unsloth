# Active turn continuity

An active turn starts when the user sends one prompt and ends when Studio returns the final
answer, the user cancels, or the run fails. A single active turn can contain many model passes,
tool calls, and context compactions.

## Problem

Context fitting already runs before every local-model pass, so the prompt can be compacted more
than once during the same response. That alone does not guarantee continuity. A model can lose
its exact execution point after old tool exchanges leave the window, emit another sentence about
what it intends to do, and then stop without either calling the tool or giving a final answer.

The old recovery allowance also covered the whole response rather than the current productive
phase. If it had already been used earlier, a later post-tool stall could be accepted as the end
of the turn. The safetensors/MLX loop did not recover post-tool stalls at all.

## Design

Studio now creates a private turn checkpoint when a Studio chat run begins:

- The objective is copied deterministically from the newest user message. No extra model call is
  needed.
- Each tool that actually runs adds a bounded action row containing its name, selected arguments,
  outcome, and a short result excerpt.
- Raw model reasoning is never persisted or reinjected.
- The checkpoint is mirrored while the response is active under
  `$UNSLOTH_STUDIO_HOME/share/active-turns/` with mode `0600`, then removed when the generator
  finishes. Files left by a hard process crash expire automatically.
- Before any work has happened, the objective remains only in the protected newest user turn and
  the private file; Studio does not waste context by duplicating it in the system prompt.
- Once the turn has operational state, a bounded `<active_turn_checkpoint>` block is injected into
  the system message. Both GGUF and MLX context fitters preserve system/developer messages, so the
  block survives every later compaction in that response.

The block labels the objective and tool excerpts as untrusted data, escapes its own delimiters,
and tells the model to resume from completed work rather than repeat it.

## Loop behavior

- A real tool result resets the post-tool stall allowance because it starts a new productive
  phase.
- A plan-without-action is nudged to call a tool or provide the complete final answer.
- Exact repeated stalls are retried only up to the existing bounded recovery limit. Duplicate
  tool calls and consecutive no-progress guards remain in force, so this is not an unbounded
  empty loop.
- Tool calls, rather than hidden recovery model passes, spend the user-facing tool-call budget.
- Studio's UI defaults the existing tool-call slider to **Max** for new/unset settings. A user who
  selects a finite cap still gets that cap.
- RAG auto-injection and denied approval flows keep their existing no-reprompt behavior.

## Thinking display

The thinking renderer is unchanged. Repeated text observed in the failing runs was present in
separate model generations, not duplicated by the UI. The action checkpoint and per-phase stall
recovery address that loss of execution continuity without storing or fabricating hidden
chain-of-thought.

## Regression coverage

Backend tests cover:

- two context compactions in one response followed by a final answer;
- post-tool recovery in both GGUF and safetensors/MLX loops;
- recovery becoming available again after each new real tool action;
- final answers with intent-like lead-ins not being re-prompted;
- private checkpoint creation, delimiter neutralization, action injection, and cleanup;
- preservation of RAG and approval behavior.
