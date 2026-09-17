// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const dialog = readFileSync(
  new URL("../src/features/chat/chat-providers-dialog.tsx", import.meta.url),
  "utf8",
);
const adapter = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);

test("managed Kaggle TPU does not require a user-supplied inference API key", () => {
  assert.match(
    dialog,
    /const showApiKeyField =\s*!usesOAuth && !isKaggleTPU && !customPresetSkipsApiKeyField\(providerType\);/,
  );
  assert.match(
    dialog,
    /!isCustomProvider &&\s*!isKaggleTPU &&\s*selectedRegistryEntry\?\.auth_kind !== "chatgpt_oauth" &&\s*!apiKey\.trim\(\)/,
  );
  assert.match(
    dialog,
    /!isEditingCustomProvider &&\s*!isEditingOAuthProvider &&\s*existing\.providerType !== "kaggle_tpu" &&\s*credentialEdit\.action === "missing"/,
  );
});

test("managed Kaggle TPU bundles its launcher and asks only for the Kaggle token", () => {
  assert.match(dialog, /Kaggle API token/);
  assert.match(dialog, /The launcher is included with Studio\./);
  assert.match(dialog, /kaggleApiToken: isKaggleTPU \? kaggleApiToken\.trim\(\) : undefined/);
  assert.match(dialog, /hasKaggleApiToken: created\.has_kaggle_api_token/);
  assert.doesNotMatch(dialog, /id="kaggle-tpu-lab-path"/);
  assert.doesNotMatch(dialog, /placeholder="\/path\/to\/kaggle-tpu-lab"/);
});

test("managed Kaggle TPU starts automatically and waits for its generated endpoint", () => {
  assert.match(dialog, /await startKaggleTPU\(created\.id\)/);
  assert.match(adapter, /await ensureKaggleTPUReady\(externalProvider\.id, abortSignal, externalProvider\.managedConfig\?\.auto_start === true\)/);
  assert.match(adapter, /externalProvider\?\.providerType === "kaggle_tpu"/);
  assert.match(
    adapter,
    /!externalProviderUsesOAuth &&\s*!externalProviderIsKaggleTPU &&\s*!externalProviderIsCustom/,
  );
});
