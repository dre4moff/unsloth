# Architettura Unsloth Companion v1

## Topologia

```text
Unsloth Desktop macOS arm64
  UI impostazioni e attivita
  CompanionManager / task routing / auto-best / Multi-iPhone
  server Bonjour + WSS autenticato
                  |
        LAN locale, protocollo v1
                  |
uno o piu iPhone
  discovery / pairing / lifecycle
  task coordinator / storage manager
  llama.cpp + mtmd / Vision / AVFoundation / Accelerate
```

Il Mac è sempre l'orchestratore. Un task non divisibile viene eseguito da un solo iPhone; task distinti e shard indipendenti possono essere distribuiti in parallelo. Una generazione singola e la sua KV cache non vengono suddivise.

## iPhone

- `CompanionAppModel`: bootstrap, modello UI e operazioni utente.
- `CompanionServiceModel`: state machine `offline -> discovering -> pairing -> ready -> leased -> running -> draining -> suspended`.
- `CompanionDiscoveryService` e `CompanionTransport`: Bonjour, Network.framework, TLS/WSS e backpressure.
- `CompanionIdentityStore`: identità P-256, Secure Enclave quando disponibile, Keychain e pairing revocabili.
- `CompanionTaskCoordinator`: ammissione, lease, blob, progressi, cancellazione e risultato terminale unico.
- `ModelRuntimeActor`: load/unload GGUF, Metal/CPU, probe `mtmd`, grammar JSON e streaming token.
- `TaskPipelineActor`: testo, OCR, immagini, video e audio.
- `AudioDSPAnalyzer`: BPM, LUFS, peak/true peak, RMS, pitch YIN, spettro, centroide, transienti e clipping.
- `StorageBudgetManager`: unico proprietario dei file dell'app, protezione dei file in uso, journal e recovery.

## Desktop

- `core.companion.manager`: listener WSS, Bonjour, pairing, heartbeat, selezione device e task.
- `core.companion.models`: contratto Pydantic del protocollo e tipi di stato pubblici.
- `core.companion.security`: identità P-256 e certificato TLS locale.
- `routes.companion`: status, settings, pairing, rename, enable, revoke e cancel.
- orchestratore esistente: delega della compressione contesto quando un iPhone idoneo è pronto; altrimenti conserva il percorso Mac.
- frontend Connections: switch globale, auto-best, Multi-iPhone, stato device e cancellazione dell'attività corrente.

## Routing

In modalità automatica il punteggio considera connessione autenticata, stato, capability, batteria, Low Power Mode, termica, latenza, coda e storage. In modalità multipla sono candidati soltanto gli iPhone selezionati dall'utente.

Una disconnessione riassegna soltanto task divisibili o shard mancanti. L'output provvisorio di un task non divisibile viene scartato. Una cancellazione esplicita dell'utente termina senza riassegnazione; la disattivazione globale drena i device e lascia proseguire il percorso Mac.

## Storage

```text
Application Support/UnslothCompanion/
  Models/Blobs             GGUF content-addressed
  Models/Manifests         riferimenti modello -> blob
  Staging/Downloads        download attivo o in pausa
  Activities               cronologia limitata
  Logs                     massimo 20 MiB
  Resume                   massimo 10 MiB
Library/Caches/UnslothCompanion/Tasks/<taskID>
tmp/UnslothCompanion
Keychain                    identita e pairing iPhone
```

Modelli e staging sono esclusi dai backup. Le eliminazioni passano da rename atomico a `.deleting-*`; il bootstrap completa eliminazioni e installazioni interrotte. Un blob condiviso viene eliminato soltanto quando nessun manifest lo riferisce.

## Lifecycle

L'idle timer è disabilitato solo in foreground quando il servizio è pronto, leased o running. Su inactive non vengono ammessi nuovi task; su background/lock vengono inviati draining e cancellazione, salvata l'attività, eliminate le directory task e chiusa la sessione. `beginBackgroundTask` copre soltanto notifica e cleanup. Heartbeat e lease usano intervalli rispettivamente di 5 e 30 secondi; tre heartbeat mancati chiudono la sessione.

## Scope trasporto

La v1 implementa soltanto LAN Bonjour + WSS. USB e trasporti cloud non sono presenti.
