# Unsloth Companion v1 - stato di implementazione

Ultimo aggiornamento: 2026-08-31

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
| Pipeline testo/media/DSP | Completato | Pipeline OCR, image, video, audio, DSP e task testuali compilate per `iphoneos`; `subagent` usa il contesto visibile fornito dal Mac, non riformula l'obiettivo delegato e restituisce testo libero in `result.text` per default, usando JSON solo quando viene richiesto `result_schema` |
| UI iPhone e accessibilita | Completato | 3 UI test firmati: onboarding, EN, IT, assenza alert; download disabilitato nel simulatore |
| Desktop auto-best/Multi-iPhone | Completato | 13 test Python di protocollo/manager; typecheck, lint mirato, i18n e build frontend superati |
| Companion nella chat Desktop | Completato, collaudo fisico riservato all'utente | Ogni passaggio del modello vede worker connessi/idonei/liberi/occupati e capacita parallela per `kind`. `submit` restituisce solo accettazione e job ID; il modello fa fan-out, continua il ramo Mac e usa `collect` o il `wait` limitato soltanto come join finale. I cambi mailbox vengono iniettati tra i passaggi e il prompt vieta di descrivere il tool come sincrono o il parallelismo come concettuale |
| Layout chat Desktop | Completato | Il piano dell'ultimo turno e ancorato sopra il composer e rimosso dalla cronologia; dopo Stop un piano cancellato o rimasto attivo viene rimosso e nessun indicatore continua ad animarsi; il contesto resta visibile nelle finestre strette |
| App/DMG macOS | Completato | Build `2026.8.18+mlxcompaction8.companion11` / `0.1.800-mlx.11`: arm64, macOS 12, metadata, firma ad hoc, privacy rete locale e servizio Bonjour dichiarati, wheel Companion incorporata e `hdiutil verify` |
| IPA unsigned | Completato | Artefatto iOS rigenerato come 0.0.1 build 3 arm64/iOS 18.6 con contesto runtime adattivo 4k/8k/16k, task `subagent`, `llama.framework` incorporato, nessuna firma/provisioning o percorso locale e ZIP integro |
| Pulizia simulatore | Completato | iPhone 17 Pro simulato azzerato e spento; container, cache e tracce Companion assenti; nessun modello scaricato |
| Build firmata iPhone | Completato | Profilo Xcode generato e build `iphoneos` firmata con team configurato |
| Collaudo iPhone 17 Pro Max | Parziale | Build firmata installata e avviata; trasferimento reale Fast su 5G verificato fino a 61,8 MB, pausa/ripresa e persistenza verificate, preferenza cellulare mantenuta attiva; nessun modello completo ancora sottoposto a smoke test |
| Smoke test dei 5 modelli e stress 20 GiB | Non eseguito | Nessun modello e stato scaricato nel simulatore; il gate resta fisico e non e simulato |

## Test automatici superati

Per `companion11` non sono stati eseguiti test automatici ne collaudi fisici, come richiesto dall'utente. La build e le verifiche di integrita dell'artefatto finale non vengono indicate come test comportamentali. I 15 test Python di `companion10`, i 13 test Swift unitari e i 3 UI test iOS firmati restano esclusivamente evidenza storica delle release precedenti. Protocollo e codice Swift non sono cambiati; l'IPA unsigned viene riutilizzata byte per byte.

- 13 test Swift unitari.
- 3 test UI iOS firmati.
- 15 test Python del Companion Desktop, inclusi mailbox asincrona e WSS TLS reale in loopback.
- 220 test del resolver `llama.cpp`, inclusi fallback macOS pre/post Tahoe.
- TypeScript typecheck, ESLint mirato, parita i18n e build Vite.
- Build Release `iphoneos`, build fisica firmata e build macOS arm64.
- Packaging IPA riproducibile con symbol table locale rimossa e controllo automatico dei percorsi della macchina di compilazione.
- Backend Desktop verificato su `/api/health`; OpenAPI contiene status, impostazioni, dispositivi, pairing e cancellazione task Companion.
- Pannello verificato dal vivo in `Impostazioni -> iPhone e connessioni`, con switch attivo, auto-best predefinito e Multi-iPhone selezionabile.

## Gate fisico residuo

Per chiudere il collaudo di rilascio occorre eseguire manualmente sull'iPhone 17 Pro Max un task chat realmente delegato al modello gia installato, quindi download/checksum/load/smoke dei cinque profili, lifecycle lock/Home/rete e stress continuativo. Il test reale non e stato avviato dall'agente su richiesta dell'utente. L'iPhone 14 Pro e escluso dai test su richiesta dell'utente. Questo e un limite di collaudo esterno, non una funzione assente dal codice.
