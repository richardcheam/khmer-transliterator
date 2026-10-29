---
language:
- en
- km
library_name: numpy
pipeline_tag: text2text-generation
tags:
- transliteration
- khmer
- english
- research-preview
datasets:
- HuggingFaceFW/fineweb-2
---

# khmer-translit E02 letters model: English word → Khmer phonetic spelling (research preview)

> **Research preview. Phonetic quality has not been independently validated. Reported exact-match results measure
> agreement with the specified reference spellings.**

This is **E02** (`letters_s0`), a small model trained from scratch and run with NumPy. It is **not** the newer
ByT5-based model (`khmer-translit-byt5`, E08/E10), which is a different architecture, a different training set and
a separate bundle.

**Publication status: unresolved.** No licence has been chosen for the code or the weights, so this card has no
`license` field. See `NOTICES.md`, `LICENSING_DECISION.md` and `PROVENANCE_AUDIT.md`.

## Model details
- **Task:** one English word or name, read as English (US), written phonetically in Khmer script. Sound, not
  meaning. Example: `coffee` → ខូហ្វ៊ី.
- **Architecture:** character-level Transformer encoder–decoder trained from random initialisation.
  - d 192, 3 encoder + 3 decoder layers, 4 heads, feed-forward 512, GELU, learned positions;
  - 2,609,169 parameters;
  - input: `<task:en>` + the letters a–z (157-token source vocabulary); output: 78 Khmer characters.
- **Weights:** `model.npz`, sha256 `8901050b1093cf90f60523c7941d6c12dbd8571d71698e37abcf874ba437904b`, 10.5 MB,
  float32 arrays plus `__meta__` JSON. Exported from torch checkpoint `199df48c…d848`.
- **Inference implementation:** `khmer_translit.english.Transliterator` in the `khmer-translit` package (0.2.0).
  - Pure NumPy, beam search width 4.
  - No lookup lists, dictionary or pronunciation resource.
- **Not a Transformers model:** `AutoModel.from_pretrained` does not load it. Use the package (below).

## Uses
- **Intended:** drafting Khmer phonetic spellings of English loanwords and names for a person to check (glossaries,
  text-to-speech front-ends), offline on a CPU.
- **Out of scope:**
  - translation;
  - sentences or phrases;
  - romanised Khmer chat;
  - Khmer input;
  - names to be read in their original language;
  - publishing spellings without a Khmer-literate check.

## How to use
Install the wheel; NumPy is its only dependency. No torch, transformers or PEFT is needed.

```bash
python -m venv .venv && . .venv/bin/activate
pip install khmer_translit-0.2.0-py3-none-any.whl          # from this bundle's wheel/ folder
khmer-translit predict --model model.npz --word coffee --top-k 3 --json
printf 'computer\nJohnson\n' | khmer-translit predict --model model.npz --input - --jsonl
```

```python
from khmer_translit.english import Transliterator

tl = Transliterator.from_pretrained("model.npz")      # load once
print(tl.predict("coffee").text)                       # ខូហ្វ៊ី
for r in tl.predict_many(["computer", "", "new york"], top_k=3):
    print(r.input, r.status, [c.text for c in r.candidates])
```

- **Input:** NFKC, lower case, accents removed, apostrophes and hyphens dropped; then letters a–z only, at most 32.
- **Refusals:** empty input, several words, other characters (digits, Khmer) and more than 32 letters return a
  `status` (`empty`, `unsupported_multiword`, `unsupported_characters`, `too_long`) with no candidates. Input is
  never truncated.
- **Candidates:** at most 8 (`--top-k`).
- **`log_score`** is a beam log-probability for ranking, not a probability of being correct.
- **CLI exit codes:** 0 when all inputs succeed, 1 when any input was refused, 2 for usage errors and missing or
  corrupt models.

## Training
- **Data:** 9,735 English–Khmer word pairs (plus 296 train-val pairs for loss tie-breaks), mined from bracketed forms
  in Khmer web text:
  - FineWeb-2 `khm_Khmr` (9,807 rows);
  - Khmer Wikipedia, dump 2026-10-01 (693 rows; 469 rows occur in both).
- **Filters:** a word is kept only if it has a CMUdict entry, is not romanised Khmer, passes a noise filter and is
  unrelated to every evaluation word.
- **Not used for training:**
  - no Chhunneng examples;
  - no LLM-generated labels;
  - no owner-authored labels;
  - no pretrained or auxiliary initialisation.
- **Indirect influences:**
  - the mining aligner and filter used a Chhunneng-trained model;
  - checkpoint selection used owner-verified spellings, 17 of 86 of them LLM-suggested.

  Details and hashes are in `PROVENANCE_AUDIT.md`.
- **Procedure:**
  - AdamW, learning rate 1e-3 with 500 warm-up steps then cosine decay, weight decay 0.01, label smoothing 0.1,
    batch 128, gradient clip 1.0;
  - 40 epochs; epoch 27 selected by dev agreement on 86 items;
  - seed 0; 0.25 h on a laptop CPU (FP32).

## Evaluation (historical, `en-phonetic-v1`)
The metric is exact agreement with the owner's single listed spelling per word. Those references are owner-verified
loanword-list spellings, not independently reviewed under the phonetic policy, so the benchmark is **unqualified**.
Correct-sounding variants count as wrong.

| Track | Items | Exact match (raw beam-4 top-1, family macro) | Reference spelling among top 4 | Mean CER |
|---|---|---|---|---|
| Final primary | 165 (162 families) | 16.7% | 40.4% | 0.353 |
| Final flagged (ambiguous / possible translation) | 26 | 19.2% | 38.5% | 0.315 |
| Exploratory proxy (LLM spellings, owner spot check) | 400 | 30.1% | 51.3% | 0.343 |
| Dev (selection, greedy) | 86 | 20.9% | — | — |

- **Seed spread:** letters seeds 0/1/2 give 16.7 / 15.1 / 16.0% on the primary track.
- **Phone-input models** were +1.65 points [−1.65, +4.94], not adopted.
- **These numbers are not** phonetic accuracy, production validation or a general benchmark result.
- **Report:** `experiments/E02-english-phonetic/FINAL_REPORT.md` in the source repository.

## Requirements and cost
- **Software:** Python ≥ 3.10, NumPy ≥ 2 (tested 2.2.6). CPU only. No network access is needed after installation.
- **Measured** on an Intel Core i7-8750H laptop (12 logical CPUs), in a fresh process with sockets disabled
  (`evidence/install_verify.json`):

  | Measure | Value |
  |---|---|
  | Load time | 0.15 s |
  | Single answer, one word at a time (p50 / p95) | 17 ms / 30 ms |
  | Top-4 candidates (p50) | 19 ms |
  | Batches of 16 | 13 ms per word |
  | Peak RSS | 65 MiB |

## Limitations
- One reference spelling per word, and no independent review: quality is unverified.
- **Web-mined training pairs:**
  - some are conventional or translated Khmer forms rather than sound renderings;
  - names may follow their original language;
  - vowel errors were not screened.
- **One reading per word:** US-English readings are implicit. Ambiguous words (*record*, *live*) get one reading
  chosen by the model.

## Permissions and attribution
Training text derives from Khmer Wikipedia (CC BY-SA 4.0 / GFDL) and FineWeb-2 (ODC-By 1.0, also subject to
Common Crawl's Terms of Use). Whether ShareAlike reaches the weights, and what rights in the underlying web pages
imply, is unresolved. Attribution text and the full source table are in `NOTICES.md`.
