# Sicurezza e privacy v1

## Confine di fiducia

Bonjour serve esclusivamente alla discovery. Nessun task o blob viene accettato prima dell'autenticazione della sessione.

## Pairing

1. L'iPhone invia identità e chiave pubblica P-256.
2. Il Mac invia identità, chiave, nonce casuale e fingerprint SHA-256 del certificato WSS.
3. L'iPhone verifica il certificato osservato, firma il nonce e mostra il SAS a sei cifre.
4. Mac e iPhone richiedono entrambi conferma locale dello stesso SAS.
5. Il secondo `pairing_confirm` lega firme e identità; soltanto allora la coppia viene persistita.

Le offerte scadono dopo 60 secondi e la conferma dopo 5 minuti. I pairing possono essere revocati da entrambi i lati.

## Identità e sessione

Su iPhone la chiave usa Secure Enclave quando disponibile, con rappresentazione persistita nel Keychain `ThisDeviceOnly`; il fallback software resta nel Keychain. Sul Mac chiave e certificato locali hanno permessi `0600` in una directory privata. Le sessioni successive usano challenge firmate P-256 e certificate pinning.

Gli envelope fuori versione, con timestamp oltre 60 secondi, duplicati o malformati vengono rifiutati. Il backend conserva una finestra limitata di message ID per impedire replay e verifica lo SHA-256 dei risultati canonici.

## Dati

- WSS per tutti i messaggi e blob.
- Nessun payload nei log Companion.
- Camera e microfono non vengono attivati da remoto.
- `raw_media` richiede abilitazione locale esplicita.
- File task eliminati immediatamente dopo completamento, errore o cancellazione.
- Nessuna telemetria cloud del Companion.
