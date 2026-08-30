# Pipeline multimodale v1

## Policy

- `semantic_only`: nessun blob binario trasferito; risultato testuale/JSON.
- `derived_media`: accetta media derivati o ridotti necessari al task e restituisce artefatti semantici.
- `raw_media`: accetta file originali soltanto se lo switch locale `Allow raw media` è attivo.

Ogni blob dichiara task, nome sicuro, MIME type, dimensione e SHA-256. I chunk sono al massimo 1 MiB e vengono scritti incrementando il file, senza accumulare l'intero trasferimento in RAM.

## Immagini e OCR

Vision riconosce il testo. Per la comprensione visuale l'immagine viene ridimensionata incrementalmente fino a 1536 pixel, codificata JPEG e passata a `mtmd`; OCR e inferenza confluiscono nel risultato tipizzato.

## Video

AVFoundation calcola la durata e campiona tra 3 e 24 frame in funzione della lunghezza. I frame rispettano la rotazione, sono limitati a 1280 pixel, ricevono OCR con timestamp e vengono elaborati in ordine temporale dal runtime.

## Audio

AVFoundation decodifica PCM. I task audio-capable passano il payload a `mtmd`; il task DSP processa il flusso in locale con Accelerate/vDSP e restituisce misure numeriche codificate in JSON.

## Cancellazione e cleanup

La cancellazione interrompe inferenza per token, media pipeline e blob aperti. Handle e runtime vengono rilasciati, la directory `Tasks/<taskID>` è eliminata tramite journal e solo dopo viene emesso il messaggio terminale di cancellazione o errore.
