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
- `TaskPipelineActor`: testo, OCR, immagini, video e audio; per `subagent` tratta
  un payload `text`-only come obiettivo eseguibile, separa il contesto quando esiste
  una `instruction`, usa prompt senza slot imitabili, controllo letterale esatto e
  rifiuto di eco quasi letterali anche annidate in JSON/placeholder con un solo retry correttivo;
  interrompe durante lo streaming i loop meta ripetuti di self-correction.
- `AudioDSPAnalyzer`: BPM, LUFS, peak/true peak, RMS, pitch YIN, spettro, centroide, transienti e clipping.
- `StorageBudgetManager`: unico proprietario dei file dell'app, protezione dei file in uso, journal e recovery.

## Desktop

- `core.companion.manager`: listener WSS, Bonjour, pairing, heartbeat, selezione device,
  esecuzione in background e mailbox dei risultati isolata per chat.
- `core.companion.models`: contratto Pydantic del protocollo e tipi di stato pubblici.
- `core.companion.security`: identità P-256 e certificato TLS locale.
- `routes.companion`: status, settings, pairing, rename, enable, revoke e cancel.
- orchestratore esistente: delega della compressione contesto quando un iPhone idoneo è pronto; altrimenti conserva il percorso Mac. La branch attiva viaggia con i suoi ID messaggio come metadato privato della richiesta, cosi il checkpoint viene ripristinato dalla riga esatta anche se reasoning e tool call producono piu messaggi wire.
- frontend Connections: switch globale, auto-best, Multi-iPhone, stato device e cancellazione dell'attività corrente.

Il tool Desktop usa quattro azioni. `submit` registra il job e restituisce subito il suo
ID, senza attendere l'inferenza dell'iPhone; `status` legge uno snapshot senza attesa;
`collect` preleva i risultati terminali gia pronti; `wait` e la sola barriera di join,
usata quando il Mac ha terminato ogni lavoro indipendente e necessita degli esiti. Il job
non dipende dal ciclo di vita del turno Mac che lo ha avviato. I job pronti vengono
segnalati con un breve aggiornamento di stato tra i passaggi, ma non interrompono una
generazione Mac in corso e non causano polling attivo. Lo schema completo del tool e gia
presente e resta stabile tra telefono libero e occupato: conteggi e mailbox non invalidano
il prefisso KV. Il prompt vieta di cercarlo nel terminale o tramite chiamate volutamente invalide.

Il catalogo comunica al modello principale le capacita compatibili per ogni `kind`; gli
aggiornamenti di runtime comunicano stato e mailbox soltanto nel suffisso. Un batch `items` usa tutti gli iPhone
liberi compatibili anche in routing automatico; la modalita Multi-iPhone limita invece il
pool al sottoinsieme selezionato dall'utente. Il modello deve fare fork, continuare il proprio
ramo sul Mac e fare join soltanto alla fine: non puo descrivere il parallelismo come concettuale.
Ogni runtime iPhone esegue un task per volta: con un solo telefono gli item eccedenti e i
`submit` separati restano nella coda Desktop e vengono avviati uno alla volta, mentre con piu
telefoni vengono consumati da slot distinti. L'iPhone libera runtime e lease prima di inviare `task_completed`; il Desktop
ritenta inoltre il solo `runtime_error` transitorio per interoperare con le IPA precedenti.

Per il modello GGUF del Mac, la finestra e la compattazione coincidono con le release `.7`-`.11`
sia prima sia dopo ogni checkpoint. Non esiste un target post-checkpoint separato, una soglia 10K
o un override prefill: il limite configurato e il budget risposta originale determinano da soli
quando riprodurre il checkpoint o iniziare una nuova epoca. Il ledger variabile vive nel suffisso
utente e il catalogo tool ordinario resta stabile, lasciando a llama-server la normale selezione LRU.

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
