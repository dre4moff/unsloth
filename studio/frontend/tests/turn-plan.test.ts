// SPDX-License-Identifier: AGPL-3.0-only

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const adapter = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);
const thread = readFileSync(
  new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
  "utf8",
);
const card = readFileSync(
  new URL("../src/components/assistant-ui/turn-plan.tsx", import.meta.url),
  "utf8",
);

test("local Studio tool turns opt into visible planning", () => {
  assert.match(adapter, /enable_tools: true,\s+turn_planning: true,/);
});

test("streamed plans are persisted in assistant metadata", () => {
  assert.match(adapter, /let turnPlan: OpenAIChatChunk\["turn_plan"\]/);
  assert.match(adapter, /if \(chunk\.turn_plan\)/);
  assert.match(adapter, /turnPlan = chunk\.turn_plan/);
  assert.match(adapter, /custom: liveCustom\(\)/);
  assert.match(adapter, /contextTruncation,\s+turnPlan,/);
});

test("normal assistant messages render the persisted checklist", () => {
  assert.match(thread, /custom\?\.turnPlan/);
  assert.match(thread, /<TurnPlanCard plan=\{turnPlan\}/);
  assert.match(card, /aria-label="Execution plan"/);
  assert.match(card, /Step \{shownStep\} \/ \{plan\.steps\.length\}/);
  assert.match(card, /item\.status === "completed"/);
  assert.match(card, /item\.status === "in_progress"/);
});
