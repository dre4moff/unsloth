# SPDX-License-Identifier: AGPL-3.0-only
"""Use Studio's existing checkpoint/archive policy with a remote token estimate."""
from core.inference import llama_cpp as policy
from core.inference.context_window import prompt_budget


class RemoteContextFitter:
    def __init__(self, messages, *, thread_id, branch_message_ids, context_length, max_tokens, count_tokens, tools_enabled):
        self.branch = list(messages)
        self.thread_id = thread_id
        self.branch_message_ids = branch_message_ids
        self.context_length = context_length
        self.max_tokens = max_tokens
        self.count_tokens = count_tokens
        self.tools_enabled = tools_enabled
        self.anchors = set()
        self.recalled = False
        self.boundary_applied = False

    def fit(self, messages, tools):
        count = lambda msgs: self.count_tokens(msgs, tools)
        before = list(messages)
        fitted, truncation = policy._fit_with_instruction_pins(
            messages, context_length=self.context_length, max_tokens=self.max_tokens,
            count_tokens=count, anchor_ids=self.anchors,
            keeps_boundary=policy._keeps_compaction_boundary(self.thread_id),
            can_reset=policy._can_reset_epoch(
                self.thread_id, self.tools_enabled,
                tools_withheld=policy._memory_tool_withheld(self.thread_id, tools),
            ),
            reserve_tokens=policy._conversation_recall_reserve(self.thread_id),
            sticky_dropped=(0 if self.boundary_applied else policy._sticky_compaction_boundary(
                self.thread_id, self.branch, self.branch_message_ids,
            )),
        )
        self.boundary_applied = True
        events = []
        if truncation and truncation.get('fits'):
            recalled = policy._archive_and_recall(
                fitted, before, thread_id=self.thread_id, style='inline',
                recall_done=self.recalled,
                force_recall=bool(truncation.get('checkpoint_started', True)),
                recall_budget_tokens=max(0, prompt_budget(self.context_length, self.max_tokens) - count(fitted)),
                count_tokens=count, branch_messages=self.branch + [m for m in before if all(m is not old for old in self.branch)],
            )
            fitted = recalled['conversation']
            self.recalled = self.recalled or recalled['recalled']
            self.anchors.update(id(m) for m in recalled['anchored'])
            events.extend(recalled['events'])
            truncation = {**truncation, **recalled['counts'],
                          'boundary_messages': policy._branch_boundary(fitted, self.branch),
                          'boundary_anchor': policy._branch_boundary_anchor(fitted, self.branch)}
        return fitted, truncation, events
