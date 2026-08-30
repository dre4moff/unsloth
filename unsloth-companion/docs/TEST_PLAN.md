# Piano di test e accettazione

## Automatico - superato

- Protocollo Swift/Python, date ISO-8601, pairing P-256/SAS, replay e WSS TLS reale.
- Task, lease, heartbeat, checksum risultato, cancellazione esplicita, device disable e risultati tardivi.
- Storage: cache protetta, 100 task senza residui, blob GGUF condivisi, tombstone e installazione interrotta.
- Registry: cinque profili, revisioni a 40 caratteri, dimensioni e SHA-256 validi.
- Simulatore: avvio download modello impossibile e controllo Download disabilitato.
- DSP: pitch, RMS e clipping su fixture nota.
- UI: onboarding, inglese, italiano, navigazione Storage/Models e assenza di alert inattesi.
- Desktop: auto-best, Multi-iPhone, attivita corrente, cancellazione, route e orchestration.
- Frontend: typecheck, lint mirato, parita i18n e build produzione.
- Packaging: `iphoneos` arm64 non firmato; macOS arm64 12+, firma ad hoc e DMG valido.

## Fisico - procedura obbligatoria

### iPhone 17 Pro Max

1. Autorizzare il profilo sviluppatore e avviare l'app firmata.
2. Per ciascuno dei cinque profili: download, checksum, load, probe `mtmd` e smoke task.
3. Provare classificazione, summary, compression, extraction, verification, reranking e planning.
4. Provare OCR, immagine, video, audio e DSP con file reali.
5. Provare lock automatico/manuale, Home, cambio app, chiamata, terminazione, perdita Wi-Fi e riconnessione.
6. Eseguire almeno 50 task e 20 GiB cumulativi di media; cache+tmp+log devono restare entro 64 MiB al netto di modelli e cronologia dichiarata.

L'iPhone 14 Pro e escluso dal collaudo su richiesta dell'utente. Auto-best, Multi-iPhone, distribuzione e riassegnazione restano coperti dai test automatici del manager Desktop.

## Criteri di rilascio

- Zero test falliti e zero task duplicati o persi.
- Download, checksum, caricamento e smoke test superati per tutti i profili.
- Nessuna directory task residua dopo cleanup o crash recovery.
- Dimensioni UI coerenti con il filesystem.
- IPA, app e DMG corrispondenti ai checksum pubblicati localmente.
- La release pubblica deve dichiararsi prototipo, distinguere i test automatici dai gate fisici residui e pubblicare i checksum degli artefatti.
