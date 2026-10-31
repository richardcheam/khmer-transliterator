---
language:
- en
- km
license: cc-by-nc-sa-4.0
base_model: google/byt5-small
base_model_relation: adapter
library_name: khmer-translit
inference: false
tags:
- transliteration
- khmer
- english
- lora
- byt5
- adapter
---

# en-km-phonetic-byt5-lora

**English word → Khmer phonetic spelling.** Give the model one English word or name and it writes how the word
*sounds* in Khmer script. It does not translate.

```
coffee → ខូហ្វី    london → ឡុនដុន    taxi → តាក់ស៊ី    doctor → ដុកទ័រ
```

> **Research preview.** Quality was judged by one person affiliated with the project, not independently validated.

**This is a LoRA adapter, not a standalone model.** It runs on top of
[`google/byt5-small`](https://huggingface.co/google/byt5-small) and needs the `khmer-translit` Python package. For
a small standalone model without PyTorch, see
[`ric912/en-km-phonetic-transliterator`](https://huggingface.co/ric912/en-km-phonetic-transliterator).

## Model
- **Base:** `google/byt5-small` at revision `68377bdc18a2ffec8a0533fef03b1c513a4dd49d` (300M parameters), frozen.
  It reads and writes raw UTF-8 bytes, so it needs no vocabulary.
- **Adapter:** LoRA, rank 16, alpha 32, on every attention (q, k, v, o) and feed-forward (wi_0, wi_1, wo) layer:
  6.3 million parameters (256 tensors).
- **Files:**
  - `adapter.safetensors` (25 MB), sha256 `698a23ae1526d3da8f3b2bd00f0177799a939d2005c165569fcba7d58e76cc3f`;
  - `adapter_config.json`: the base pin, adapter layout and decoding settings.
- **Not a PEFT or Transformers checkpoint:** `PeftModel` and `AutoModel` cannot load it, and there is no hosted
  inference widget. Load it with the package as shown below.
- **Base check:** loading refuses a base model with a different revision or different weights.

## Quick start
```bash
# 1. The package, with CPU-only PyTorch (about 870 MB installed)
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  "khmer-translit[byt5] @ https://huggingface.co/ric912/en-km-phonetic-transliterator/resolve/v0.2.0/khmer_translit-0.2.0-py3-none-any.whl"
# with uv: uv pip install --torch-backend cpu "khmer-translit[byt5] @ <same URL>"

# 2. The base model at the pinned revision (about 1.2 GB, once)
hf download google/byt5-small --revision 68377bdc18a2ffec8a0533fef03b1c513a4dd49d --local-dir byt5-small

# 3. This adapter
hf download ric912/en-km-phonetic-byt5-lora --local-dir byt5-adapter

# 4. Predict (works offline)
khmer-translit predict --backend byt5 --model byt5-adapter --base byt5-small --offline --word coffee
khmer-translit predict --backend byt5 --model byt5-adapter --base byt5-small --offline --word london --top-k 4 --json
printf 'computer\nJohnson\nLondon\n' | \
  khmer-translit predict --backend byt5 --model byt5-adapter --base byt5-small --offline --input - --jsonl
```

```python
from khmer_translit.byt5 import ByT5Transliterator

tl = ByT5Transliterator.from_pretrained("byt5-adapter", base="byt5-small", offline=True, threads=6)  # load once
print(tl.predict("coffee").text)                              # ខូហ្វី
print([c.text for c in tl.predict("london", top_k=4).candidates])
for r in tl.predict_many(["taxi", "", "new york"]):           # input order kept
    print(r.input, r.status, r.text)
```

The wheel's sha256 is `1bc8242a63d5385c86e772357e9fa3af15ecce3d3f6ccd5105ae42322a02a47d`. `--backend byt5` is
required, because the package's default backend is the small standalone model.

## Inputs and outputs
- **Input:** a single English word or name, at most 32 letters.
  - It is normalised before use: Unicode NFKC, lower case, accents removed, apostrophes and hyphens dropped. So
    `O'Neil` becomes `oneil`, and `Wi-Fi` becomes `wifi`.
- **Refused inputs:** the following return a status and no output. The model never truncates or guesses:
  - empty text (`empty`);
  - several words (`unsupported_multiword`);
  - digits, Khmer or other characters (`unsupported_characters`);
  - more than 32 letters (`too_long`).
- **Output:**
  - by default, one answer by greedy decoding;
  - with `--top-k N`, up to N different spellings, best first, from beam search (width at least 4).
- **Scores:** each candidate has a `log_score`, a model log-probability for ranking. It is **not** a calibrated
  confidence.
- **Command line:** one JSON line per input line, in the same order. A missing or corrupt adapter, or a missing
  base model, exits with code 2.

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
- **Size:** 4,764 English–Khmer word pairs covering 3,585 English words.
- **Sources:**

  | Source | Pairs |
  |---|---|
  | Mined from Khmer web text: a Khmer spelling followed by the English word in brackets, e.g. "ចនសុន (Johnson)", from [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (`khm_Khmr`) and Khmer Wikipedia | 2,255 |
  | More mined pairs of the same kind, kept by documented rules after the first screen wrongly rejected them | 1,292 |
  | Names and loanwords whose Khmer spelling occurs word for word in the human translations of the [ALT Parallel Corpus](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/) (NICT, CC BY 4.0), passed through the same screen | 1,217 |

- **Filters:** an automatic screen keeps a pair only if its Khmer consonants fit the English word's pronunciation
  (from the CMU Pronouncing Dictionary). Every word used for evaluation, and its family (plurals and other forms),
  is kept out of training.
- **No synthetic data:** no LLM-generated spellings and no hand-written labels were used for training. Nobody
  reviewed the training spellings.
- **Selection aid:** an older in-house transliteration model helped choose and align the mined pairs. Its outputs
  were never used as training labels.

## Training procedure
- AdamW, learning rate 1e-3 with 100 warm-up steps then linear decay, no weight decay, gradient clipping 1.0,
  batch 32, 3,000 steps;
- LoRA dropout 0.05; one run, seed 0;
- 4.6 hours on a laptop CPU (6 threads).

## Evaluation
**Main result: human judgment on two separate samples of 120 words.** Each sample was drawn fresh and kept away
from training. It holds 40 everyday words, 40 words with irregular English spelling and 40 names.
- **How it was judged:** one person affiliated with the project judged each spelling as acceptable, not
  acceptable or uncertain. Spellings were shown blind, without the model's name, and mixed with other models'
  outputs. Acceptable means it reads as the word's US English pronunciation.

| Sample | Acceptable | Everyday | Irregular spelling | Names | Uncertain |
|---|---|---|---|---|---|
| Sample 1 | **77/120 (64.2%)** | 27/40 | 14/40 | 36/40 | 3 |
| Sample 2 | **90/120 (75.0%)** | 34/40 | 20/40 | 36/40 | 1 |

How to read these numbers:
- **Same model, two samples:** these are two samples of the same model, not two versions. They differ because of
  the words drawn and judging on different days. With 120 words, each figure is uncertain by about ±8 points.
- **No single headline figure:** the two samples are not pooled into one number.
- **Not typical text:** the samples over-represent irregular spellings and names, so they do not describe
  everyday English text.
- **Uncertain answers:** answers judged "uncertain" are counted as not acceptable.
- **Agreement with web spellings:** on 228 other test words, the top answer matches the mined web spelling
  exactly for 52.3% (character error rate 0.163). Those spellings were not reviewed, and many acceptable answers
  are written differently, so this measures agreement, not accuracy.
- **Alternatives (`--top-k 4`):** on 23 hard, irregularly spelled words, the default answer was acceptable for 8.
  For 15, at least one of the 4 beam candidates was acceptable. So showing a person several candidates can help
  for hard words. This is a small diagnostic, not an accuracy figure.

## Requirements and speed
- **Software:** Python 3.10–3.12, PyTorch ≥ 2.14 (CPU is enough), transformers ≥ 5.19, safetensors ≥ 0.8.
  Tested with torch 2.14.1+cpu and transformers 5.20.2. No network access is needed after the downloads.
- **Measured** on an Intel Core i7-8750H laptop, 6 threads:

  | Measure | Value |
  |---|---|
  | Load time | 5.5–7 s |
  | One word, default answer (median) | 0.55 s |
  | One word, 4 candidates (median) | 0.82 s |
  | Batch throughput, default answer | 14 words/s |
  | Peak memory | 2.0 GB |

## Limitations
- **Unverified quality:** quality was judged by one non-independent person, under one reading of each word. One
  training run; run-to-run variation is not measured.
- **Irregular spellings:** these are the weakest group (35–50% acceptable in the two samples). Silent letters and
  letter-by-letter readings occur.
- **Vowels:** most errors are vowel choices.
- **Pronunciation:** spellings follow a US English reading. Words with two pronunciations (*record*, *live*) get
  one, chosen implicitly.
- **Web-mined data:** some training spellings are conventional Khmer forms or translations rather than sound
  renderings, and some names follow their original language. The model can reproduce these habits.
- **Established loanwords:** the model writes the English sound, not established Khmer forms, so `coffee` gives
  ខូហ្វី rather than the usual កាហ្វេ.
- **Speed:** much slower than the small standalone model. It is meant for drafting, not for high-volume
  real-time use on a CPU.

## Licence
- **Adapter weights:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Non-commercial use
  only, with attribution; share adaptations under the same licence. See `LICENSE`.
- **Base model:** `google/byt5-small` is Google's, under Apache 2.0, and is not included here.
- **`khmer-translit` package (the wheel):**
  [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0). The licence is included
  in the wheel.
- **Copyright:** 2026 Richard Cheam (khmer-transliterator project).

## Data attribution
- **Training pairs:**
  - mined from Khmer Wikipedia (text available under CC BY-SA 4.0, https://km.wikipedia.org) and from FineWeb-2
    (HuggingFaceFW/fineweb-2, `khm_Khmr`);
  - Khmer spellings attested in the ALT Parallel Corpus (NICT, CC BY 4.0; Riza et al. 2016, "Introduction of the
    Asian Language Treebank", Oriental COCOSDA).
- **FineWeb-2:** contains information from FineWeb-2, which is made available under the ODC Attribution License
  (ODC-By 1.0); its use is also subject to Common Crawl's Terms of Use.
- **Data selection:** used the CMU Pronouncing Dictionary and the Google `language-resources` Khmer lexicon
  (CC BY 4.0, Copyright 2018 Google Inc.).

Rights in the original web pages remain with their authors. The ODC-By licence covers the FineWeb-2 database, not
the individual pages. Whether Wikipedia's ShareAlike condition applies to model weights is legally unsettled. If
you represent a source and have concerns, please open a discussion on this repository.

## Citation
Base model: Xue et al. 2022, "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models",
*Transactions of the ACL* 10:291–306.
