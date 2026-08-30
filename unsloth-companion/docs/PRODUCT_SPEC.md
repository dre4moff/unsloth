# Specifica prodotto v1

## Definizione

Unsloth Companion è un'app iPhone nativa che esegue lavoro locale delegato da Unsloth Desktop. Non è una seconda chat: il Mac mantiene conversazione, memoria, orchestrazione e fallback.

## Funzioni

### Testo

- classificazione;
- riassunto;
- compressione contesto;
- estrazione strutturata;
- verifica;
- reranking;
- pianificazione leggera.

### Media

- OCR con Vision;
- comprensione immagini con `mtmd`;
- campionamento video adattivo, OCR per frame e sintesi temporale;
- trascrizione/analisi audio quando il probe `mtmd` la supporta;
- DSP deterministico per BPM, LUFS, peak, true peak, RMS, pitch, spettro, centroide, transienti e clipping.

Fotocamera e microfono non possono essere attivati da remoto. La v1 elabora file trasferiti dal Mac o importati localmente; l'accettazione di file originali richiede la policy `raw_media` e lo switch locale dedicato.

## Modelli incorporati nel catalogo

- Fast: Unsloth Gemma 4 E2B QAT Mobile.
- Balanced: Unsloth Gemma 4 E4B QAT Mobile.
- Deep: Unsloth Gemma 4 E4B QAT Q4.
- Community E2B: Mobius Heretic.
- Community E4B: HauhauCS Aggressive.

Ogni voce contiene repository, revisione immutabile, file, dimensione e SHA-256 per modello e `mmproj`. Il download usa Hugging Face alla revisione bloccata, senza URL fittizi.

## Device

La modalità predefinita sceglie automaticamente l'iPhone migliore disponibile. La modalità Multi-iPhone permette di selezionare più device e distribuire task distinti, frame, chunk, batch o shard indipendenti. Capability, modello, batteria, termica, coda, latenza e storage sono visibili sul Mac.

## Continuità

- schermata Guardia Companion quasi nera dopo inattività;
- schermo attivo solo durante disponibilità in foreground;
- draining su lock/background, Low Power Mode, batteria sotto 10%, termica seria o memory warning;
- retry connessione a 0,5/1/2/5 secondi;
- lease 30 secondi e heartbeat ogni 5 secondi;
- cleanup immediato al termine, errore o cancellazione.

## Storage e controllo utente

La schermata Storage mostra modelli, download, staging, cache task, temporanei, attività, log, resume metadata, riserva e spazio libero. Offre pulizia selettiva, cancellazione cronologia/log/download parziali, eliminazione singola o totale dei modelli e reset completo con doppia conferma. I file di un task attivo sono protetti; l'utente può cancellare il task e poi pulire.

## Privacy e scope

Il funzionamento core è locale, senza telemetria cloud. USB, cloud inference, Windows/Linux, chat/browser AiryWay e Stable Diffusion non fanno parte della v1.
