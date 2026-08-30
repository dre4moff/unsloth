# Note sulle sorgenti

Verifica: 2026-08-30.

## AiryWay

La copia locale `../AiryWay` e stata usata in sola lettura per confrontare integrazione `llama.cpp`, gestione GGUF, download e coordinamento memoria. Unsloth Companion ha componenti propri e non dipende da AiryWay a runtime. Chat, browser e Stable Diffusion non sono stati importati.

## Unsloth Desktop

L'integrazione estende il fork locale senza sostituire l'orchestratore esistente:

- backend Companion in `studio/backend/core/companion`;
- route locali in `studio/backend/routes/companion.py`;
- aggancio della compressione contesto in `studio/backend/core/inference/orchestrator.py`;
- impostazioni e gestione task in `studio/frontend/src/features/settings`.

## Dipendenza nativa

`llama.xcframework` deriva da `ggml-org/llama.cpp` al commit bloccato `3173a56471c1753650cd806694145ffd6dcace67`, con slice iOS device e simulator, Metal, Accelerate e `mtmd`.

## Rete e scope

La v1 usa Bonjour/Network.framework e WSS sulla LAN locale. USB, cloud inference e host Windows/Linux sono esclusi dalla v1.
