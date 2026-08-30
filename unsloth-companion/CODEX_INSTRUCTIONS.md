# Regole di manutenzione - Unsloth Companion v1

## Contratto prodotto

- Il Mac resta orchestratore e fallback.
- L'iPhone è un co-processore locale, non una chat.
- La v1 usa soltanto Bonjour + WSS sulla LAN.
- USB, cloud inference, Windows/Linux, chat/browser AiryWay e Stable Diffusion restano fuori scope.
- AiryWay locale è riferimento in sola lettura e non una dipendenza runtime.

## Regole non negoziabili

1. Non introdurre URL modello non verificati o revisioni mobili.
2. Non consentire download modello nel simulatore.
3. Non accettare task prima di pairing e challenge validi.
4. Non scrivere payload nei log.
5. Non aggirare il lifecycle iOS con audio silenzioso, localizzazione o background mode impropri.
6. Non lasciare cache task o temporanei dopo un terminale.
7. Non eliminare blob GGUF ancora referenziati.
8. Non riassegnare automaticamente una cancellazione esplicita dell'utente.
9. Non suddividere una generazione singola o la KV cache tra device.
10. Non segnare una verifica come superata senza evidenza effettiva.

## Verifica richiesta per modifiche

- Swift unit test e UI test interessati.
- `scripts/test_desktop_companion.py` per protocollo/backend.
- typecheck, lint mirato, parità i18n e build frontend per la UI Desktop.
- build `iphoneos` per runtime o storage.
- controllo che l'IPA resti arm64 e non firmata.
- `codesign --verify`, metadata arm64/macOS 12 e `hdiutil verify` per la release Desktop.

## Release

Gli artefatti restano in `release/`. L'IPA unsigned deve essere firmata prima dell'installazione. La release pubblica del prototipo è autorizzata sul fork `dre4moff/unsloth`; pubblicazioni future richiedono una nuova decisione esplicita del maintainer.
