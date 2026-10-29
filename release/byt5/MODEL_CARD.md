---
language:
- en
- km
license: other
license_name: not-yet-licensed
base_model: google/byt5-small
library_name: transformers
pipeline_tag: text2text-generation
tags:
- transliteration
- khmer
- lora
- byt5
- research-preview
---

# khmer-translit-byt5 0.2.0: English word → Khmer phonetic spelling (research preview)

> **Research preview with unverified phonetic quality.** Not published and not licensed. Publication is blocked by
> open data-terms questions (`NOTICES.md`). All measured numbers are agreement with web-mined, unreviewed
> references, not phonetic accuracy.

## Model details
- **Task:** one English word or name → Khmer script approximating its English (General American) pronunciation.
  Sound, not meaning. Example outputs: `london` → ឡុនដុន, `taxi` → តាក់ស៊ី, `doctor` → ដុកទ័រ.
- **Architecture:** `google/byt5-small` (revision `68377bdc18a2ffec8a0533fef03b1c513a4dd49d`, frozen; base weights
  sha256 `5c5aaf56…931a`) with LoRA adapters (r 16, α 32) on attention q/k/v/o and FFN wi_0/wi_1/wo.
- **Files:**
  - adapter: `adapter.safetensors`, sha256 `d514a51f1d31014f5100fa763ed1894c0c53782d1c581fdfdd938a7b164388f8`;
  - `adapter_config.json`: base pin, hashes, decoding.
- **Checkpoint:** experiment E08, run `rs_s0` (seed 0, update 3,000; checkpoint sha256 `d5da8c29…`).
- **Code:** `khmer_translit.byt5` (package `khmer-translit` 0.2.0, extra `[byt5]`). It loads with plain PyTorch and
  Transformers, without PEFT.
- **Developed by:** the khmer-transliterator project (one owner), on a laptop CPU.

## Intended use
- **For:** drafting Khmer phonetic spellings of English loanwords and names for a person to check (glossaries,
  text-to-speech front-ends).
- **Not for:** translation, sentences, romanised Khmer chat, other source languages, or unchecked publication of
  spellings. See `LIMITATIONS.md`.

## How to use
```bash
pip install "khmer-translit[byt5] @ file://…/khmer_translit-0.2.0-py3-none-any.whl"
hf download google/byt5-small --revision 68377bdc18a2ffec8a0533fef03b1c513a4dd49d --local-dir byt5-small
khmer-translit predict --backend byt5 --model adapter --base byt5-small --offline --word london --top-k 4 --json
```

```python
from khmer_translit.byt5 import ByT5Transliterator
tl = ByT5Transliterator.from_pretrained("adapter", base="byt5-small", offline=True)
tl.predict("london").text          # 'ឡុនដុន'
```

- **Input:** lower-cased word plus EOS, at most 32 UTF-8 bytes. Longer inputs are refused, never truncated.
- **Decoding:**
  - single answer: greedy, at most 121 new bytes;
  - candidates: beam search, width max(k, 4), `early_stopping="never"`, `length_penalty=0`.
  - E09 found that `early_stopping=True` truncates byte-level beams.
- **Scores** are summed log-probabilities for ranking, **not calibrated confidence**.
- **No fallback:** the byt5 backend never silently uses another model. The historical E02 model stays the CLI
  default.

## Training data
5,148 English → Khmer pairs over 3,778 words (frozen arm R+S, sha256 `03a4988f…`). Details and terms are in
`DATA_PROVENANCE.md`.

| Source | Rows |
|---|---|
| Brackets mined from FineWeb-2 `khm_Khmr` and Khmer Wikipedia, automatic consonant screen | 2,255 |
| Mined pairs recovered from that screen | 1,292 |
| Spellings attested in the ALT corpus (CC BY 4.0) | 1,217 |
| Unverified Google AI Mode output | 384 |

No reviewer-authored labels exist. All evaluation and test words are filtered out of training by word family.

## Training procedure
- AdamW, learning rate 1e-3, warm-up 100, linear decay to 0, batch 32, 3,000 updates.
- 5.2 h on 6 CPU threads; one run (seed 0).
- Checkpoint selection: dev greedy exact match, then CER, then the earlier update.

## Evaluation
**Track 1 (historical mined-reference agreement).** This is the only track with numbers. The references are
web-mined spellings that passed an automatic consonant screen and were never reviewed. The eval split has been
examined many times (E03–E10), so read these as exploratory agreement, not accuracy.

| Split (items) | Decoding | Exact match | Exact match among 4 candidates | CER |
|---|---|---|---|---|
| dev (253) | greedy | 54.2% | — | 0.151 |
| dev (253) | beam 4, canonical | 55.4% | 82.8% | 0.148 |
| eval (228, 222 families) | greedy | 52.4% | — | 0.161 |
| eval (228) | beam 4, canonical | 53.3% | 79.1% | 0.163 |

- **Uncertainty:** a single training run. On eval, the E10 exploratory candidate A_s0 − this model is +0.6 points
  [95% CI −4.5, +5.9] (paired family bootstrap). Differences of a few points are within noise at this size.
- **Clean outputs:** no invalid-orthography, empty, repeated or truncated outputs on dev, eval or the 120-word
  confirmation set.
- **Typical disagreements with references** (eval, greedy):

  | Kind | Count |
  |---|---|
  | vowels only | 53 |
  | consonants and vowels | 18 |
  | consonants changed | 12 |
  | consonant dropped | 10 |
  | consonant added | 8 |
  | marks only | 7 |

- **NEWS metrics:** the NEWS shared-task metrics (W18-2408: ACC, mean F-score, MRR, MAP_ref) assume reviewed,
  possibly multiple references. Here they would measure agreement with one mined spelling, so only ACC and
  ACC-at-4 are reported.
- **Track 2 (AI-assisted exploratory judgments):** none are counted.
- **Track 3 (independent human phonetic review):** **not collected.** A blinded 60-word batch is prepared (see
  `BENCHMARK_CARD.md`).

## Compute and runtime
Laptop CPU, 6 threads, `HF_HUB_OFFLINE=1` with a local base:

| Measure | Value |
|---|---|
| Load time | 6.1 s |
| Single word, greedy (median / max) | 0.58 s / 0.95 s |
| Single word, 4 candidates (median) | 0.94 s |
| Batch of 276 words, greedy / 4 candidates | 20 s / 44 s |
| Peak RSS | about 2.0 GB |

## Limitations, risks and licence
- Unverified phonetic quality and vowel errors; see `LIMITATIONS.md`.
- **Base model:** Apache 2.0. That does not settle the training data's terms.
- **Blockers:** the Google AI Mode rows face a restrictive terms clause; Wikipedia's CC BY-SA share-alike question
  is open; no code licence has been chosen. See `NOTICES.md`.
- **Attribution:** FineWeb-2 (ODC-By), Khmer Wikipedia (CC BY-SA 4.0), ALT (NICT, CC BY 4.0; Riza et al. 2016).

## Citation
Base model: Xue et al. 2022, "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models", TACL 10
(2022.tacl-1.17).
