# UX Unsloth Companion v1

## Onboarding

Spiega elaborazione locale, pairing sicuro, continuità in foreground e cleanup. Al termine avvia la discovery se lo switch Companion è attivo.

## Dashboard

- switch `Use this iPhone`;
- stato della state machine e Mac connesso;
- task corrente con cancellazione;
- SAS e conferma/rifiuto pairing;
- elenco Mac Bonjour disponibili;
- modello caricato, Metal, Vision, Audio e context size.

## Modelli

Mostra i cinque profili bloccati, dimensione totale, stato download, pausa/ripresa/annulla e cancella, import GGUF+mmproj, load/unload ed eliminazione. Nel simulatore il pulsante Download è disabilitato e non può avviare una sessione di rete modello. L'eliminazione totale richiede doppia conferma.

## Attività

Elenca stato, tipo, orari, durata e token. Permette cancellazione del task corrente, eliminazione della singola voce e pulizia della cronologia.

## Storage

Mostra tutte le categorie gestite, riserva e spazio libero. Ogni azione indica lo spazio recuperabile; reset e modelli richiedono doppia conferma. I file protetti vengono saltati e `Cancel task and clean` offre il percorso esplicito.

## Impostazioni

- schermo attivo e Guardia Companion;
- policy media e consenso raw;
- Automatic/CPU/Metal;
- Mac associati e revoca;
- versione app, protocollo e commit runtime.

## Desktop

Nella scheda Connections, `iPhone Companion` include switch globale, modalità miglior iPhone automatica o Multi-iPhone, selezione, rename, enable, revoke, pairing, metriche e task corrente cancellabile.

## Accessibilità

Le viste usano componenti SwiftUI nativi, Dynamic Type, label VoiceOver, contrasto di sistema, materiali che rispettano Reduce Transparency e animazioni brevi compatibili con Reduce Motion. Tutte le stringhe visibili sono presenti in inglese e italiano.
