// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChatProvidersSettings, useExternalProvidersStore } from "@/features/chat";
import { IPhoneCompanionSection } from "../components/iphone-companion-section";

export function ConnectionsTab() {
  const providers = useExternalProvidersStore((s) => s.providers);
  const setProviders = useExternalProvidersStore((s) => s.setProviders);

  return (
    <div className="space-y-6">
      <IPhoneCompanionSection />
      <ChatProvidersSettings providers={providers} onProvidersChange={setProviders} />
    </div>
  );
}
