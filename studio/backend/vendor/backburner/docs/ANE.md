# The Neural Engine (ANE): what we measured

Three places the ANE could help. One is built and in use; two were measured and set aside. All numbers measured on this
project's hardware (MacBook Pro M4 Pro 24 GB; iPhone 16 Pro Max A18 Pro; iPhone 17 Pro Max A19 Pro), Qwen3.8-27B.

## 1. iPhone ANE for the old context: built, in use

Past 64k tokens the iPhone holds the oldest part of the KV cache and computes attention over it. Old keys and values never
change, so each 16,384-key page of a layer is compiled into an ANE model with the keys and values as its fixed weights; each
attention call sends only the queries. Code: `phone-attn/pa-ane.mm`, template builder `phone-attn/ane-kv/build.py`, `scripts/serve.sh`
pushes it the first time it sees the phone (`scripts/phone-ane.sh IP` by hand) and prints "ANE pages on"; the engine is on
whenever the template is on the phone.

- **Accuracy:** with fp16 page weights the model gives the same output as the exact path on a real 51k-token session (33 of
  33 tokens identical, same KL). int8 page weights are not accurate enough and are not used. The A18 ANE gives bit-identical
  results to the Mac ANE.
- **Speed (A18):** 1.85 ms per 16k-key page for all 4 KV heads, 0.113 ms per 1k keys (0.099 sustained over 20 s). That is 2-3x
  the phone's SME2 unit (0.20-0.27) and GPU (0.31-0.37).
- **Building a page:** the phone patches one template model's weight file in place (checked: byte-identical to a fresh build,
  bit-identical outputs). First load ~0.74 s per page on the A18, including the on-device ANE compile; ~1.1 s per page in
  140k runs.
- **Effect at 140k** (75.5k keys on the phone), time per generated token in our 140k phone-KV test: 279 ms with no ANE pages, 208 ms with 2
  pages per layer, 176 ms with 3. The ANE takes a share of each call (22% of the keys for 1-token calls, 43% for 8-token
  calls); trying to learn the split online failed because an idle ANE clocks down.
- **Not used for reading prompts:** prompt reading sends large calls that the phone GPU's dense kernel handles; page builds run
  in the background and serve generation afterwards.

## 2. iPhone ANE for reading prompts: measured, not built

The idea: run each tail layer's feed-forward block on the ANE and the rest on the phone GPU.

- **Fast alone:** the 27B feed-forward block (5120 -> 17408 -> 5120) on the A18 ANE runs at 6.7 TFLOPS with int8 weights
  (9.9 with int8 activations too), about 5x what the phone GPU gets on the same layers.
- **They don't add up on the phone:** with the ANE busy, the GPU's tail ran 1.5-2.2x slower and the ANE lost 35-40%. Both
  together did only 1.1-1.25x the work of one. So it would have to take turns, not run side by side.
- **Memory:** ~270 MB of app memory per int8 layer, so about 16 layers fit next to the tail.
- **Projection, not built:** A18 split prefill from 1.08x to about 1.3x. Open question first: int8 accuracy on real weights.

## 3. Mac ANE: measured, not used

- **Raw compute adds:** the M4 Pro ANE held 7.5 TFLOPS on a feed-forward block while the GPU held 6.8, at the same time.
- **Real prompt reading got slower:** llama.cpp prefill with the ANE busy lost 1-16% (median 9.6%).
- **Memory is the wall:** an fp16 feed-forward layer on the ANE costs ~1.17 GB of system memory; int4 doesn't help (~0.5 GB per
  layer). Next to the 27B model on a 24 GB Mac, only a handful of layers fit.
- **Writing:** 26% slower with the ANE active (it shares memory bandwidth with the GPU).

## Where the ANE matters next

The phone ANE's job is the context the Mac can't hold. On this 24 GB Mac that starts past ~200k tokens; on a 16 GB Mac it
starts much earlier. Those are the next tests.
