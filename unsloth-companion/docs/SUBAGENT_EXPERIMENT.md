# Report — Test tool `iphone_companion` come subagente

> Aggiornamento 2026-08-31, release `companion10`: il comportamento descritto sotto appartiene alle corse precedenti ai fix. Oltre alle correzioni di prompt/output di `companion9`, il tool ora usa un contratto asincrono: `submit` restituisce subito un job ID, il Mac continua il proprio lavoro e recupera in seguito il risultato con `collect`, senza polling bloccante. Nessun nuovo test fisico del subagente e stato eseguito durante questo intervento, come richiesto.

**Data:** 2026-08-30
**Ambiente:** macOS, host locale redatto prima della pubblicazione
**Tool sotto test:** `iphone_companion` (kind `subagent`)
**Obiettivo del test:** far restituire al subagente la frase esatta `test ready`

---

## 1. Cosa si è verificato

Sono state eseguite due invocazioni del tool `iphone_companion` con `kind=subagent`:

| Corsa | Task ID | Durata | Token gen. | Testo restituito |
|-------|---------|--------|------------|------------------|
| 1     | `a8a50131-d6c5-43a9-8f9f-5438005c4cd8` | ~63 s | 1024 | `{"` |
| 2    | `2d18bae2-7230-43be-a21b-1af6d1bb6259` | ~63 s | 1024 | `{"` |

**Sintomo comune:** in entrambe le corse il campo `result.text` è tornato come `{"`, cioè l'inizio di un oggetto JSON troncato, invece della stringa grezza `test ready` richiesta.

---

## 2. Errori osservati

1. **Output non conforme:** la frase richiesta (`test ready`) non è mai tornata pulita; è sempre `{"`.
2. **Troncamento:** il testo restituito è un prefisso di JSON (`{"`), non un documento completo né la stringa grezza.
3. **Determinismo sospetto:** le due corse hanno restituito *esattamente* lo stesso testo (`{"`) e token-generati identici (1024) con durate quasi uguali (~63 s). Sembra che il subagente generi lavoro autonomo ma il *risultato* venga serializzato/interrotto allo stesso punto.
4. **Wrapper JSON inaspettato:** nonostante l'istruzione "scrivi solo la stringa, senza JSON", il risultato resta in forma di oggetto JSON.

---

## 3. Possibili cause

- **Serializzazione del risultato troncata:** il campo `result.text` che arriva al caller è esso stesso un JSON parzialmente serializzato (`{"text":"{\"` → si vede solo `{"`). Il vero output del subagente potrebbe essere più lungo ma il caller riceve solo la testa.
- **Truncation a 1024 token:** `tokensGenerated` è sempre 1024, valore di soglia. Se il subagente ha generato il JSON completo e poi il wrapper ha tagliato a 1024 token, il testo visibile è solo l'inizio.
- **Formato di risposta non forzato:** il subagente (LLM locale) potrebbe rispondere sempre in forma di oggetto JSON per convenzione, ignorando l'istruzione "solo stringa grezza".
- **Mismatch instruction/risultato:** l'istruzione passa al subagente ma il *contenitore* della risposta è comunque un oggetto `{...}`; il caller legge solo la prima chiave.
- **Possibile bug di parsing** lato caller: si legge `result.text` come stringa ma il valore è un oggetto, quindi viene mostrato come `{"`.

---

## 4. Soluzioni / mitigazioni

1. **Forzare output grezzo:** riformulare l'istruzione come `Rispondi con solo: test ready` e chiedere esplicitamente *nessun* JSON, virgolette o wrapper.
2. **Chiedere campo `text` grezzo:** se il contratto del tool lo permette, chiedere che `result` restituisca il campo `text` come stringa pura invece di un oggetto JSON.
3. **Aumentare `maximum_tokens`:** alzare il limite (es. 2048/4096) per evitare che il wrapper tagli a 1024 token e mostri solo l'inizio del JSON.
4. **Usare `result_schema`:** fornire uno schema JSON che descriva il supporto atteso (es. `{ "text": "test ready" }`) così il subagente restituisce il campo corretto e il caller lo legge correttamente.
5. **Verificare il parsing lato caller:** controllare che il campo `result.text` venga letto come stringa e non come oggetto; se è oggetto, estrarre `.text` esplicitamente.
6. **Riprova con instruction ancora più stringente:** es. `"La risposta finale è esattamente: test ready. Nient'altro."`
7. **Se persiste:** trattare come bug di serializzazione del wrapper e segnalare che il testo grezzo non sta arrivando al caller.

---

## 5. Conclusione

- ✅ Il tool `iphone_companion` con `kind=subagent` **funziona**: parte, ragiona in modo autonomo (~63 s), genera token (1024) e torna con un JSON.
- ⚠️ Il **wrapper di serializzazione/risultato** non sta restituendo la frase pulita che gli si chiede: il campo `result.text` torna come `{"` (JSON troncato), non come la stringa grezza `test ready`.
- 📌 Prossimo passo consigliato: riprovare con `result_schema` esplicito e/o `maximum_tokens` più alto, così vediamo se il testo grezzo arriva pulito.
