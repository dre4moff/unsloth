# Piano esecutivo v1 - consuntivo

## Ordine completato

1. Inventario dei progetti e isolamento di AiryWay come riferimento in sola lettura.
2. Protocollo JSON Schema v1 condiviso e tipi Swift/Python.
3. Storage manager unico, budget, journal, deduplicazione e recovery.
4. `llama.xcframework` riproducibile e runtime `llama.cpp`/`mtmd`.
5. Discovery Bonjour, WSS, pairing P-256, SAS e autenticazione delle sessioni.
6. Task runtime con lease, heartbeat, cancellazione, shard e media chunked.
7. Pipeline testo, OCR, immagine, video, audio e DSP.
8. UI iPhone e Desktop con auto-best, Multi-iPhone, storage e attivita.
9. Test Swift, UI, Python e frontend.
10. Packaging IPA unsigned, app macOS arm64 e DMG.

## Vincoli applicati

- Nessun modello viene scaricato nel simulatore iOS.
- Un solo download modello+mmproj puo essere attivo o in pausa.
- Nessun payload sensibile viene scritto nei log Companion.
- Un task esplicitamente cancellato dall'utente non viene riassegnato automaticamente.
- Lo switch globale OFF drena i device e lascia proseguire il percorso Mac.
- USB, cloud inference, Windows/Linux, chat/browser AiryWay e Stable Diffusion sono fuori scope v1.

## Gate di collaudo

L'implementazione e conclusa. I test fisici elencati in `TEST_PLAN.md` devono essere eseguiti su dispositivi sbloccati e con profilo sviluppatore autorizzato prima di considerare chiuso il gate di rilascio hardware.
