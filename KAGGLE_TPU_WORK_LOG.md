# Aggiornamento mlx.29 — esecuzione automatica e diagnosi dal sito (6 settembre 2026)

- Verifica attuale con credenziale salvata: quota circa 19,7 ore disponibili; l'ultima operazione interattiva è terminata/cancellata.
- Prova minima privata con gli stessi metadati di avvio upstream: `machine_shape=TpuV5E8`, Internet attivo, stato QUEUED. Batch diagnostico cancellato via API prima della prova successiva.
- L'utente ha effettuato l'accesso nel browser. Avvio diretto dall'editor Kaggle sul notebook diagnostico: messaggio visibile `You are #20 in the queue`. Questa è una verifica indipendente della coda esterna attuale; non una deduzione dal messaggio Studio.
- Difetti individuati: la sessione interattiva richiedeva `Tpu1VmV38`; `GetKernelSessionStatus` legge il batch salvato e non l'operazione interattiva; dopo chiusura dell'app durante allocazione il reconnect non inviava il codice del server; un'operazione terminata poteva restare apparentemente in allocazione.
- Nuovo Start ripristina il flusso batch del repository originale con richiesta esplicita v5e-8 e server completo già consegnato a Kaggle. L'esecuzione può partire senza il processo locale quando arriva il turno. Reconnect/Stop compatibili con le precedenti sessioni interattive; nessuna duplicazione automatica di richieste pendenti.
- 54 test backend mirati superati, inclusi invio completo prima dell'attesa, stato persistito prima della richiesta, assenza di chiavi in chiaro, ripresa di batch in coda e operazioni interattive terminate.
- Build mlx.29 completata; app e DMG verificati (arm64, macOS 12+, firma strict ad-hoc, integrità DMG, app montata identica, backend kaggletpu9 corrispondente ai sorgenti). Copia aggiornata in `/Applications/Unsloth.app` senza avviarla; precedente copia conservata nel Cestino. Evidenze in `release/MLX29-VERIFICATION.json`.
- Su richiesta dell'utente, nessuna attesa ulteriore della coda e nessun test reale del modello. Prova in coda avanzata fino alla posizione 14. Stop disabilitato nell'editor durante l'avvio: eliminato via API il solo notebook temporaneo `unsloth-tpu-diagnosis-0906`; risposta senza errori, editor verificato in stato off, scheda chiusa. Nessun nuovo avvio TPU.
- READY/generazione reale non verificati. La coda esterna è stata riprodotta; le correzioni locali e il pacchetto non garantiscono disponibilità immediata di hardware Kaggle.

---

# Aggiornamento mlx.28 — stato reale coda/sessione Kaggle (5 settembre 2026)

- La mlx.27 poteva mostrare per minuti la stessa frase `Kaggle is allocating the interactive TPU session` anche se Kaggle esponeva separatamente lo stato worker della sessione.
- Aggiunta lettura read-only di `GetKernelSessionStatus`: `QUEUED` mostra minuti trascorsi, `RUNNING` distingue l'assegnazione hardware dalla preparazione Jupyter, errori/cancellazioni terminali non restano mascherati come allocazione.
- Aggiunta lettura della quota TPU per mostrare le ore residue quando l'endpoint quota le fornisce; nessuna nuova sessione viene creata da questi controlli.
- Il tentativo osservato durante questa diagnosi è stato poi arrestato tramite l'endpoint Stop dell'app; nessun nuovo Start viene eseguito automaticamente in questa lavorazione.
- Versione app/backend: `0.1.800-mlx.28` / `2026.8.19+mlxcompaction8.companion20.kaggletpu8`.
- Build/DMG e pulizia completati: `release/Unsloth_0.1.800-mlx.28_aarch64.dmg`, SHA-256 `b67d7703b22689ccd0687f45ae288f770692604d83e46d5ab095be5b5c97e7ab`. App `arm64`, macOS 12+, firma ad-hoc strict valida; DMG integro e app montata identica alla copia `release`. Wheel `kaggletpu8` senza cache o percorsi home locali. Circa 2,9 GB di output rigenerabile (`build`, `node_modules`, `dist`, target Rust, egg-info e cache Python/pytest) spostati nel Cestino; zero cache/junk rilevati dopo la pulizia. Sorgenti, `.git`, `release/` e dati `~/.unsloth` conservati.

---

# Aggiornamento mlx.27 — attesa allocazione Kaggle interattiva (5 settembre 2026)

- Verificato dal vivo che un `CreateKernelSession` valido può restare in allocazione oltre 15 minuti: token e account risultano autenticati, notebook bootstrap `COMPLETE`, ma Kaggle mantiene la richiesta remota in `PROVISIONING`.
- Rimosso il timeout locale predefinito dall'allocazione interattiva: Studio segue la stessa operazione finché Kaggle la completa oppure l'utente preme Stop. Nessun retry/duplicato automatico viene creato per il solo trascorrere del tempo.
- I timeout finiti restano disponibili solo per test o chiamanti che li impostano esplicitamente.
- Versione app/backend: `0.1.800-mlx.27` / `2026.8.19+mlxcompaction8.companion20.kaggletpu7`.
- Build DMG e verifica artefatto completati: `release/Unsloth_0.1.800-mlx.27_aarch64.dmg`, SHA-256 `6a2a1f45e3246fbcd1826c8ad494b3879d8e57d7bb4344dbe1d1b95ba5d35460`. App arm64/macOS 12+, firma ad-hoc strict, DMG integro; app montata identica alla copia release. Wheel `kaggletpu7`: 3.435 file sorgente verificati, zero mismatch/cache/percorso home.
- Pulizia finale completata: circa 3 GB di `build`, `node_modules`, `dist`, target Rust, wheel di staging, cache Python/pytest e `.DS_Store` spostati nel Cestino. Verifica finale: zero cache generate residue fuori da `release/`, firma strict e DMG ancora validi, checksum mlx.27 confermato. Sorgenti, `.git`, `release/` e dati `~/.unsloth` conservati.

---

# Aggiornamento mlx.26 — sessione Kaggle interattiva (5 settembre 2026)

- Il launcher Studio passa dal commit batch TPU (`kaggle kernels push`) al percorso interattivo usato dall'editor Kaggle: bootstrap privato CPU una sola volta, quindi `CreateKernelSession` e Jupyter autenticato.
- L'identificatore documentato dall'SDK per l'endpoint interattivo è `Tpu1VmV38`; il preflight dentro la VM continua a richiedere otto device TPU reali prima di installare/avviare vLLM, quindi non esiste fallback CPU.
- Il kernel upstream Qwen/vLLM resta invariato; configurazione e preflight Studio vengono eseguiti nella sessione Jupyter.
- Stop usa `kernelSessionId` e conserva il notebook bootstrap per Start successivi. `tokenizedJupyterUrl` e token Jupyter restano solo in memoria e non vengono salvati nel file stato.
- Dipendenza `websockets>=15.0.1` resa esplicita nel runtime Studio. Versione app/backend: `0.1.800-mlx.26` / `2026.8.19+mlxcompaction8.companion20.kaggletpu6`.
- La prova live `CreateKernelSession` finale è stata bloccata dalla revisione automatica dell'azione esterna; l'implementazione viene quindi validata localmente con regressioni, packaging e controlli artefatto. Nessuna pubblicazione GitHub.
- Validazione finale mlx.26: 57 test backend/install e 21 test frontend mirati superati; build frontend produzione riuscita; wheel `kaggletpu6` verificato contro 3.435 file sorgente senza mismatch, cache o percorsi home. App arm64/macOS 12+, firma ad-hoc strict e DMG integri; app montata dalla DMG identica alla copia `release`.
- Pulizia finale completata: circa 2,9 GB di build/target/node_modules/dist/cache spostati nel Cestino; zero `__pycache__` e `.DS_Store` residui fuori dall'app release. Sorgenti, `.git`, release, wheel backend tracciato e dati `~/.unsloth` conservati. La cancellazione del notebook Kaggle temporaneo di diagnosi è stata bloccata dalla revisione automatica prima della chiamata esterna.

---

# Aggiornamento mlx.25 — coda Kaggle e monitor (5 settembre, sera)

- Stato verificato alle 18:13 UTC: `KernelWorkerStatus.QUEUED`, nessun log di esecuzione, avvio richiesto alle 17:55:58 UTC. L'attesa era nella coda Kaggle, non nella compilazione del modello.
- L'utente ha completato la verifica d'identità e conferma TPU v5e-8 selezionabile nell'editor. La precedente indisponibilità dell'account non è più una premessa valida.
- Stop registrato alle 18:14:19 UTC; notebook successivamente non accessibile. La causa esatta del messaggio generico non è recuperabile dai log conservati. Non attribuita con certezza a rete o account.
- Individuata e corretta una fragilità del codice: TimeoutExpired nelle letture status/log interrompeva il launcher con il messaggio generico. Ora letture limitate a 15 s, retry automatico per errori transitori di stato, distacco DISCONNECTED dopo cinque letture fallite senza fermare/duplicare il notebook.
- Minuti in coda visibili a ogni aggiornamento; errore di Kaggle prima dell'avvio distinto dagli errori d'installazione/compilazione. Payload notifiche validati.
- Nessun nuovo notebook o tentativo TPU avviato in questa lavorazione. Le prove della nuova gestione sono locali; READY reale ancora non verificato.
- Versione app/backend: `0.1.800-mlx.25` / `2026.8.19+mlxcompaction8.companion20.kaggletpu5`. App/DMG esportati e verificati: firma strict ad-hoc, arm64/macOS 12+, integrità DMG, checksum e corrispondenza dei sorgenti nel wheel. 50 test backend e 3 frontend superati; TypeScript/build frontend riusciti. Cache e file generati spostati nel Cestino. Nessuna pubblicazione GitHub.

---

# Kaggle TPU — aggiornamento locale mlx.24 (5 settembre 2026)

## Diagnosi corrente

- Fonte ufficiale verificata: https://github.com/ARahim3/kaggle-tpu-lab, HEAD `10897e5799c0d911d5c81b4f2f932619f02620bd`. Il kernel vendorizzato coincide byte per byte con la fonte.
- Riprodotto dal vivo con la chiave salvata: Internet funzionante, runtime installato in 49 s, patch MTP applicata, cache corretta e pesi montati. Il server termina con `Insufficient devices for 2D mesh: found 1, expected 8`.
- Verifica minima con richiesta esplicita `TpuV5E8`, sia script sia notebook: nessun device TPU nella VM e JAX restituisce `CpuDevice(id=0)`. Stesso risultato tramite creazione di sessione interattiva ufficiale e interrogazione Jupyter autenticata.
- L'utente conferma verifica telefonica completata ma TPU non disponibile nell'editor Kaggle. Il blocco hardware esterno resta: il token non può rendere disponibile un acceleratore non assegnato da Kaggle.
- Il monitor sovrascriveva la diagnosi della fase server con una scansione dell'intero log: la parola “install” iniziale produceva il falso errore di installazione. Le innocue righe GCE metadata potevano anche essere confuse con guasti DNS.

## Correzioni mlx.24

- Allocazione esplicita `TpuV5E8` via CLI ufficiale.
- Preflight dei device e della rete prima di installare dipendenze; dopo l'installazione, verifica in subprocess del backend JAX TPU e degli otto device. Nessun fallback CPU.
- Un solo nuovo tentativo automatico per i fallimenti di preflight, sullo stesso notebook privato, dopo aver accertato la fine della sessione. Nessun retry di token/quota o sessioni dallo stato incerto.
- Identificatore di tentativo nei log: le righe di una versione precedente non possono far fallire o dichiarare pronto un nuovo avvio.
- Errori riferiti alla fase effettiva, con riconoscimento della mancata TPU e memoria esaurita, senza credenziali. Minuti trascorsi mostrati durante la compilazione.
- Il tunnel autenticato appena creato viene ricontrollato per assorbire errori transitori prima di dichiarare READY.
- Kernel e patch MTP ufficiali conservati; verifiche aggiunte da un adattatore separato con controllo dei punti di inserimento.
- Versioni desktop/backend incrementate insieme: `0.1.800-mlx.24` / `2026.8.19+mlxcompaction8.companion20.kaggletpu4`.

## Verifiche e consegna

- 44 test backend mirati superati; 3 test frontend del contratto Kaggle superati; TypeScript e build frontend completati.
- Preflight CPU, controllo degli otto device JAX, limite del retry, log di versioni precedenti, root cause e readiness tunnel verificati con regressioni automatiche.
- Prove reali: runtime e guasto di assegnazione TPU riprodotti, sia batch sia interattivo. Il successivo tentativo con il nuovo preflight è stato fermato mentre era in coda, dopo la richiesta di concludere rapidamente. Nessun READY o generazione reale: TPU non disponibile nell'account, confermato dall'utente.
- Sessione Jupyter diagnostica cancellata tramite API; notebook diagnostici arrestati ed eliminati. Preferenze e chiave della connessione originale conservate.
- App e DMG mlx.24 verificati: arm64, macOS 12.0+, firma ad-hoc strict valida (non notarizzata), integrità DMG e checksum di tutti gli artefatti superati. Backend kaggletpu4: 3.337 file confrontati con i sorgenti/asset; nessuna cache nel wheel; kernel ufficiale invariato.
- Spostati nel Cestino circa 2,9 GB di cache, dipendenze di build, target Rust, dist, wheel di staging e file temporanei della lavorazione. Conservati sorgenti, `.git`, release e dati/credenziali utente. Nessuna pubblicazione GitHub.
- Artefatti: `release/Unsloth.app`, `release/Unsloth_0.1.800-mlx.24_aarch64.dmg`; evidenze in `release/MLX24-VERIFICATION.json`.

---

# Kaggle TPU — lavoro locale, versione mlx.23

## Diagnosi verificata

- La precedente mlx.22 includeva token cifrato e launcher; il notebook era stato creato.
- Il log Kaggle del tentativo del 4 settembre 2026 mostra errori DNS nella VM, installazione di uv/vLLM fallita e notifiche ntfy irraggiungibili. Internet e TPU risultavano abilitati nei metadati restituiti dall'API.
- Il monitor non leggeva i log Kaggle: poteva rimanere in provisioning senza spiegare l'errore.
- Il recupero leggeva solo gli ultimi otto eventi e perdeva READY; lo Stop ignorava l'esito del comando Kaggle; la chat poteva riavviare sessioni senza Auto-start.
- La compattazione remota veniva eseguita prima del primo giro, senza ricontrollare i successivi risultati degli strumenti. Il comando reasoning Off non arrivava al chat template vLLM.

## Modifiche

- Eventi JSON nel launcher incluso, con fallback sui log ufficiali Kaggle e diagnostica senza credenziali.
- Verifica autenticata `/v1/models` prima di READY; recupero della sessione esistente e controlli del tunnel.
- Stato scritto atomicamente con permessi 0600; notebook privati e distinti per connessione; arresto dei processi locali e verifica dello Stop remoto.
- Connessione salvata prima dell'avvio, progresso visibile, retry senza duplicati. Auto-start richiesto per riavviare da chat; il salvataggio iniziale avvia la nuova connessione.
- Controllo del contesto prima di ogni giro del ciclo strumenti esistente; reasoning Off inoltrato al template vLLM.
- Piano opzionale e registro delle azioni del turno collegati al ciclo remoto esistente. Gli aggiornamenti del piano non consumano il budget degli strumenti.
- Streaming degli errori normalizzato senza riportare eccezioni o credenziali del server remoto.
- Versioni desktop e backend aggiornate insieme. Push GitHub disabilitati e nessuna pubblicazione prevista.
- Cache Python escluse dal wheel, con controllo nella procedura di build.

## Verifiche completate

- **471 test backend superati**: credenziali, provider, ciclo strumenti, lifecycle Kaggle, contesto/checkpoint, regressioni MLX e compattazione OpenAI/Anthropic. Alcune asserzioni preesistenti sono state allineate al comportamento corrente del piano/registro; il controllo dei file aperti usa psutil anche su macOS.
- **4.099 test frontend superati**, TypeScript typecheck e build di produzione riusciti.
- L'integrazione HTTP usa un server OpenAI-compatible di prova reale su localhost: invia una chiamata Python frammentata, Studio esegue Python localmente e crea `hello.py` nella directory temporanea, il server riceve il risultato e restituisce la risposta finale. Verificati autenticazione Bearer, discovery, reasoning, usage, tool choice e compattazione tra i giri degli strumenti.
- Nuovo tentativo Kaggle reale con il token già salvato, Fast start e keepalive limitato a 60 minuti: stessa mancata risoluzione DNS nella VM per ntfy/cloudflared/pip. Nessun READY raggiunto. Le preferenze salvate sono state conservate.
- Notebook di prova privato arrestato tramite il launcher; stato **STOPPED** verificato. Il vecchio notebook preesistente non è stato modificato.
- App/DMG **0.1.800-mlx.23**, backend **2026.8.19+mlxcompaction8.companion20.kaggletpu3**. Architettura arm64, requisito macOS 12.0, firma ad-hoc valida (non notarizzata).
- Verificati firma strict, integrità DMG, contenuto del DMG montato identico all'app in `release`, checksum, un solo wheel e corrispondenza di **1.373 file** con sorgenti/asset. Kernel TPU identico al commit upstream incluso; assenza di cache e percorsi home locali nel wheel.
- Copia aggiornata anche in `/Applications/Unsloth.app`; rilevata l'installazione del backend `kaggletpu3`. Nessun ulteriore test di avvio/interfaccia dopo la richiesta finale dell'utente di fermare i test.

## Matrice delle funzionalità

| Area | Stato locale | Limite della verifica |
| --- | --- | --- |
| Nuova connessione Kaggle con solo token | Integrata; launcher e CLI inclusi/gestiti dall'app | Il notebook parte, ma la sua rete impedisce il caricamento |
| Start, stato, Stop, recupero sessione | Coperti da test e controlli API reali disponibili | READY/reconnect a tunnel vivo non verificabili sul TPU |
| OpenAI Compatible manuale | Configurazione, discovery e modello manuale disponibili | Server HTTP di prova, non ogni prodotto compatibile |
| Streaming, reasoning, tool fragments, immagini | Coperti da test del trasporto/capabilities | Generazione/vision reale TPU non verificata |
| Python, MCP, web, filesystem, Companion | Rimangono nel catalogo/esecutore locale esistente | Python provato via HTTP; nessun nuovo test fisico iPhone |
| Contesto, archivio/recall, checkpoint, piano | Percorso esistente riusato a ogni giro remoto | Token stimati senza tokenizer remoto; test automatici |
| vLLM, XLA, MTP e ottimizzazioni TPU | Kernel upstream conservato | Nessun benchmark TPU completato |
| Fallback | Nessun cambio automatico e silenzioso di modello | La scelta di un altro modello rimane esplicita |

## Pulizia finale

- Su richiesta finale dell'utente, nessun ulteriore test.
- Spostati nel Cestino circa 3,4 GB di dipendenze di build, target Rust, dist, wheel di staging, cache Python/pytest, file temporanei e log di questa lavorazione.
- Rimossi anche i residui dei test temporanei. Conservati sorgenti, `.git`, artefatti in `release`, app installata e tutti i dati in `~/.unsloth`.
- Risultati dei controlli artefatti già completati conservati in `release/MLX23-VERIFICATION.json`. Nessuna pubblicazione GitHub.

## Blocco esterno da risolvere

Il test di accettazione reale resta aperto: Kaggle dichiara Internet e TPU abilitati, ma la VM non risolve i domini necessari. Occorre ripristinare l'accesso Internet della sessione/account Kaggle, quindi riprovare Start e verificare chat, strumenti, immagini e compattazione sul modello reale. Il solo token corretto non può risolvere questo guasto della VM.

Le verifiche simulate e la build non sostituiscono l'accettazione sul TPU reale.
