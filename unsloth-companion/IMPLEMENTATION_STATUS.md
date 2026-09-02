# Unsloth Companion v1 - stato di implementazione

Ultimo aggiornamento: 2026-09-02

Il codice previsto dal piano esecutivo v1 e implementato. Questo registro distingue l'implementazione dai collaudi che richiedono interazione fisica con gli iPhone: nessun risultato fisico non eseguito viene indicato come superato.

| Area | Stato | Verifica effettiva |
| --- | --- | --- |
| Xcode iOS 18.6, iPhone only, IT/EN | Completato | Build simulator e `iphoneos`; 219 chiavi per lingua, parita completa |
| AiryWay di riferimento | Completato | Copia locale ispezionata e lasciata invariata |
| Protocollo JSON v1 | Completato | Round-trip Swift/Python, fixture, replay, lease, shard e risultati tardivi |
| Registry modelli | Completato | 5 revisioni e 10 artefatti bloccati; copie incorporata/esterna identiche |
| Download modelli | Completato | Il file temporaneo di CFNetwork viene acquisito sincronicamente nel `tmp` gestito dall'app prima del passaggio asincrono; il claim viene sempre rimosso o spostato nello staging |
| Storage e cache | Completato | Cleanup protetto/journaled, 100 task senza residui, tombstone crash, blob condivisi |
| Runtime llama.cpp/mtmd | Completato | Framework device/simulator, Metal/CPU, grammar JSON, streaming e cancellazione compilati |
| Discovery, pairing e WSS | Completato | Il TXT Bonjour annuncia hostname certificato e porta (IP separato solo diagnostico), evitando l'abort WebSocket di Network.framework sui target IP; il listener usa il contesto TLS server originale, isolato dalla patch client-only di truststore; pairing P-256/SAS disponibile per verifica manuale |
| Pipeline testo/media/DSP | Completato, collaudo fisico riservato all'utente | Pipeline OCR, image, video, audio, DSP e task testuali compilate per `iphoneos`; `subagent` interpreta `text` come obiettivo quando `instruction` manca, oppure separa correttamente contesto e obiettivo quando sono entrambi presenti. Le richieste di stringa esatta usano prompt minimo e confronto byte-per-byte; eco quasi letterali anche dentro JSON e placeholder vengono ritentati una volta con il contratto di formato completo e poi rifiutati. Testo libero in `result.text` per default, JSON solo con `result_schema` |
| UI iPhone e accessibilita | Completato | 3 UI test firmati: onboarding, EN, IT, assenza alert; download disabilitato nel simulatore |
| Desktop auto-best/Multi-iPhone | Completato | 18 test Python di protocollo/manager, inclusa la coda fra submit separati con un solo iPhone; typecheck, lint mirato, i18n e build frontend superati |
| Companion nella chat Desktop | Completato, collaudo fisico riservato all'utente | Lo schema completo e stabile espone capacita per `kind` senza incorporare conteggi volatili nel prefisso KV. Spiega esplicitamente che `submit` resta valido quando l'unico iPhone e occupato: job e batch separati restano queued e proseguono uno alla volta; con piu iPhone gli slot consumano la coda in parallelo. Il prompt vieta di cercare lo schema in file/terminale o tramite errori deliberati; la forzatura esplicita riconosce anche `compaion` e formulazioni di test/riprova. Il tetto scelto dal Mac resta 8192-16384 per task, salvo clamp fisico del contesto del telefono. Il modello puo delegare proattivamente anche senza richiesta esplicita; `submit` resta non bloccante e il Mac assorbe il `runtime_error` transitorio delle IPA precedenti |
| Layout chat Desktop | Completato | La card Companion e gerarchica: job padre e una card figlia per ogni subtask, con stato, ID, iPhone assegnato, durata, token, errore e output separato. `Plan` e un tool indipendente e persistente nel menu `+`, conservato anche dai prompt in coda. Il piano dell'ultimo turno resta ancorato sopra il composer mentre la generazione e attiva e scompare appena il turno termina |
| Compattazione contesto e latenza Mac | Completato nel codice, collaudo conversazione riservato all'utente | `context_window.py` e `checkpoint.py` coincidono con la release `.11`: stessa politica `.7`-`.11` prima e dopo ogni checkpoint, senza protezione post-checkpoint, target 20%, soglia 10K o override prefill. Ledger del turno nel suffisso utente e catalogo tool stabile evitano di riscrivere il prefisso ordinario; gli ID della branch ripristinano il checkpoint esatto senza entrare nel prompt |
| App/DMG macOS | Completato e pubblicato | Build `2026.8.19+mlxcompaction8.companion20` / `0.1.800-mlx.20`: arm64, macOS 12, metadata, firma ad hoc, wheel Companion incorporata, compattazione `.7`-`.11` esatta, coda globale, tool Plan indipendente e validazione difensiva dell'output |
| IPA unsigned | Completato e pubblicato | Artefatto iOS 0.0.1 build 8 arm64/iOS 18.6. Libera runtime e stato prima del completamento, conserva il budget Mac 8k-16k con clamp fisico, esegue l'obiettivo reale e interrompe loop meta/self-correction oltre a rifiutare eco, placeholder e stringhe letterali inesatte; `llama.framework` incorporato, nessuna firma/provisioning |
| Pulizia simulatore | Completato | iPhone 17 Pro simulato azzerato e spento; container, cache e tracce Companion assenti; nessun modello scaricato |
| Build firmata iPhone | Completato | Profilo Xcode generato e build `iphoneos` firmata con team configurato |
| Collaudo iPhone 17 Pro Max | Parziale | Build firmata installata e avviata; trasferimento reale Fast su 5G verificato fino a 61,8 MB, pausa/ripresa e persistenza verificate, preferenza cellulare mantenuta attiva; nessun modello completo ancora sottoposto a smoke test |
| Smoke test dei 5 modelli e stress 20 GiB | Non eseguito | Nessun modello e stato scaricato nel simulatore; il gate resta fisico e non e simulato |

## Verifiche storiche e build corrente

Per `companion20` non sono stati eseguiti nuovi test automatici, simulatori o fisici. La verifica corrente comprende build frontend/DMG di produzione, build iPhoneOS unsigned 8 e integrita degli artefatti finali. I 18 test Python Companion, 388 test safetensors/selezione strumenti e 297 test GGUF/contesto appartengono alla precedente candidata `companion18` e restano solo evidenza storica. I 14 test Swift unitari e 3 UI test iOS firmati restano evidenza storica precedente alla build 8. Il protocollo resta v1.

- 14 test Swift unitari.
- 3 test UI iOS firmati.
- 18 test Python del Companion Desktop, inclusi coda globale, mailbox asincrona e WSS TLS reale in loopback.
- 220 test del resolver `llama.cpp`, inclusi fallback macOS pre/post Tahoe.
- TypeScript typecheck, ESLint mirato, parita i18n e build Vite.
- Build Release `iphoneos`, build fisica firmata e build macOS arm64.
- Packaging IPA riproducibile con symbol table locale rimossa e controllo automatico dei percorsi della macchina di compilazione.
- Backend Desktop verificato su `/api/health`; OpenAPI contiene status, impostazioni, dispositivi, pairing e cancellazione task Companion.
- Pannello verificato dal vivo in `Impostazioni -> iPhone e connessioni`, con switch attivo, auto-best predefinito e Multi-iPhone selezionabile.

## Gate fisico residuo

Per chiudere il collaudo di rilascio occorre eseguire manualmente sull'iPhone 17 Pro Max un task chat realmente delegato al modello gia installato, quindi download/checksum/load/smoke dei cinque profili, lifecycle lock/Home/rete e stress continuativo. Il test reale non e stato avviato dall'agente su richiesta dell'utente. L'iPhone 14 Pro e escluso dai test su richiesta dell'utente. Questo e un limite di collaudo esterno, non una funzione assente dal codice.
