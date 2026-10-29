---
language:
- en
- km
license: cc-by-nc-sa-4.0
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

# en-km-transliterator

**English word → Khmer phonetic spelling.** Give the model one English word or name and it writes how the word
*sounds* in Khmer script. It does not translate.

```
coffee   → ខូហ្វ៊ី      computer → កំពូទែរ      Johnson → ចនសុន
```

> **Research preview. Phonetic quality has not been independently validated. Reported exact-match results measure
> agreement with the specified reference spellings.**

## Model
- **Type:** a small character-level Transformer encoder–decoder (sequence to sequence), trained from scratch. It
  has no pretrained base model.
- **Size:** 2.6 million parameters. d_model 192, 3 encoder and 3 decoder layers, 4 attention heads, feed-forward
  512.
- **Vocabulary:** input letters a–z; output 78 Khmer characters.
- **File:** `model.npz` (10.5 MB), NumPy arrays plus a JSON metadata entry.
  sha256 `8901050b1093cf90f60523c7941d6c12dbd8571d71698e37abcf874ba437904b`.
- **Runtime:** CPU only, through the `khmer-translit` Python package (included here as a wheel). Inference uses
  NumPy only, with beam search of width 4.
- **Not a Transformers checkpoint:** `AutoModel.from_pretrained` cannot load it. Use the package as shown below.

## Quick start
```bash
hf download <user>/en-km-transliterator --local-dir en-km-transliterator
cd en-km-transliterator
python -m venv .venv && . .venv/bin/activate
pip install khmer_translit-0.2.0-py3-none-any.whl          # installs khmer-translit + numpy only

khmer-translit predict --model model.npz --word coffee --top-k 3 --json
printf 'computer\nJohnson\nLondon\n' | khmer-translit predict --model model.npz --input - --jsonl
```

```python
from khmer_translit.english import Transliterator

tl = Transliterator.from_pretrained("model.npz")       # load once, reuse
print(tl.predict("coffee").text)                        # ខូហ្វ៊ី
for r in tl.predict_many(["computer", "", "new york"], top_k=3):
    print(r.input, r.status, [c.text for c in r.candidates])
```

## Inputs and outputs
- **Input:** a single English word or name, at most 32 letters.
  - It is normalised before use: Unicode NFKC, lower case, accents removed, apostrophes and hyphens dropped. So
    `O'Neil` becomes `oneil`, and `Wi-Fi` becomes `wifi`.
- **Refused inputs:** the following return a status and no output. The model never truncates or guesses:
  - empty text (`empty`);
  - several words (`unsupported_multiword`);
  - digits, Khmer or other characters (`unsupported_characters`);
  - more than 32 letters (`too_long`).
- **Output:** up to 8 candidate spellings, best first, each with a `log_score`. The score is a model
  log-probability for ranking candidates. It is **not** a calibrated confidence.
- **Command line:**
  - one JSON line per input line, in the same order;
  - exit code 0 when all inputs succeed, 1 when some were refused, 2 for usage errors or a missing or corrupt model
    file;
  - `--sha256 <hash>` refuses a model file with a different hash.

## Intended use
- **Good for:** drafting Khmer spellings of English loanwords and names for a person to review. Examples are
  glossaries and text-to-speech front-ends.
- **Not for:**
  - translation;
  - sentences;
  - romanised Khmer ("sok sabay");
  - names to be read in their original language rather than English;
  - publishing spellings without a check by a Khmer reader.

## Training data
- **Size:** 9,735 English–Khmer word pairs, with 296 more held back for validation loss.
- **Origin:** mined automatically from Khmer web text where a Khmer spelling is followed by the English word in
  brackets, e.g. "ចនសុន (Johnson)".
- **Sources:**
  - [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), Khmer subset `khm_Khmr` (one shard):
    9,807 pairs;
  - Khmer Wikipedia, dump of 2026-10-01: 693 pairs.
  - 469 pairs occur in both, so the source counts add up to more than the total.
- **Filters:** a pair is kept only if the English word is in the CMU Pronouncing Dictionary, does not look like
  romanised Khmer, passes a noise filter, and is unrelated to every word used for evaluation.
- **No synthetic data:** no LLM-generated spellings and no hand-written labels were used for training.
- **Selection aid:** an older in-house transliteration model helped choose and align the mined pairs. Its outputs
  were never used as training labels.
- **Training procedure:**
  - AdamW, learning rate 1e-3 with 500 warm-up steps then cosine decay, weight decay 0.01, label smoothing 0.1,
    batch 128;
  - 40 epochs; the best epoch (27) was chosen on a separate set of 86 reviewed words;
  - about 15 minutes on a laptop CPU.

## Evaluation
| Test set | Words | Top answer matches the reference | Reference among the top 4 |
|---|---|---|---|
| English loanwords and names, reference spellings checked by the author | 165 | **16.7%** | 40.4% |
| Same kind of words, spellings suggested by an LLM and spot-checked | 400 | 30.1% | 51.3% |

How to read these numbers:
- **Exact agreement with one spelling:** each word has a single reference spelling. A spelling that sounds right
  but is written differently counts as wrong, so these numbers understate usefulness.
- **Agreement, not accuracy:** the references were not independently reviewed. These numbers measure agreement
  with those spellings, not phonetic accuracy, and they are not a general benchmark result.
- **Run-to-run variation:** retraining with other random seeds gave 15.1% and 16.0% on the first test.

## Requirements and speed
- **Software:** Python 3.10–3.12 and NumPy ≥ 2 (tested with 2.2.6). No GPU, no PyTorch and no network access are
  needed after installation.
- **Measured** on an Intel Core i7-8750H laptop, in a fresh process:

  | Measure | Value |
  |---|---|
  | Load time | 0.15 s |
  | One word (median / 95th percentile) | 17 ms / 30 ms |
  | Batches of 16 | 13 ms per word |
  | Peak memory | 65 MB |

## Limitations
- **Unverified quality:** phonetic quality has not been independently checked.
- **Pronunciation:** spellings follow a US English reading. Words with two pronunciations (*record*, *live*) get
  one, chosen implicitly.
- **Web-mined data:** some training spellings are conventional Khmer forms or translations rather than sound
  renderings, and some names follow their original language. The model can reproduce these habits.
- **Word length:** long words (9 letters or more) agree less often with the references.

## Licence
- **Model weights:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Non-commercial use
  only, with attribution; share adaptations under the same licence. See `LICENSE`.
- **`khmer-translit` package (the wheel):**
  [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0). The licence is included
  in the wheel.
- **Copyright:** 2026 Richard (khmer-transliterator project).

## Data attribution
Training pairs were mined from Khmer Wikipedia (dump of 2026-10-01; text available under CC BY-SA 4.0,
https://km.wikipedia.org) and from FineWeb-2 (HuggingFaceFW/fineweb-2, `khm_Khmr`). Contains information from
FineWeb-2, which is made available under the ODC Attribution License (ODC-By 1.0); its use is also subject to
Common Crawl's Terms of Use. Data selection used the CMU Pronouncing Dictionary and the Google `language-resources`
Khmer lexicon (CC BY 4.0, Copyright 2018 Google Inc.).

Rights in the original web pages remain with their authors. The ODC-By licence covers the FineWeb-2 database, not
the individual pages. Whether Wikipedia's ShareAlike condition applies to model weights is legally unsettled. If
you represent a source and have concerns, please open a discussion on this repository.
