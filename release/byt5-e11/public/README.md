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

> **Research preview.** Quality has not been independently validated.

This is a **LoRA adapter** for [`google/byt5-small`](https://huggingface.co/google/byt5-small), not a standalone
model. It runs with the `khmer-translit` Python package on a normal CPU. It cannot be loaded with `AutoModel` or
PEFT. For a much smaller and faster model without PyTorch, see
[`ric912/en-km-phonetic-transliterator`](https://huggingface.co/ric912/en-km-phonetic-transliterator).

## Quick start
```bash
# 1. Install the package with CPU-only PyTorch
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  "khmer-translit[byt5] @ https://huggingface.co/ric912/en-km-phonetic-transliterator/resolve/v0.2.0/khmer_translit-0.2.0-py3-none-any.whl"

# 2. Download the base model (about 1.2 GB, once) and this adapter
hf download google/byt5-small --revision 68377bdc18a2ffec8a0533fef03b1c513a4dd49d --local-dir byt5-small
hf download ric912/en-km-phonetic-byt5-lora --local-dir byt5-adapter

# 3. Predict (works offline)
khmer-translit predict --backend byt5 --model byt5-adapter --base byt5-small --offline --word coffee
khmer-translit predict --backend byt5 --model byt5-adapter --base byt5-small --offline --word london --top-k 4 --json
printf 'computer\nJohnson\nLondon\n' | \
  khmer-translit predict --backend byt5 --model byt5-adapter --base byt5-small --offline --input - --jsonl
```

```python
from khmer_translit.byt5 import ByT5Transliterator

tl = ByT5Transliterator.from_pretrained("byt5-adapter", base="byt5-small", offline=True)  # load once
print(tl.predict("coffee").text)                                  # ខូហ្វី
print([c.text for c in tl.predict("london", top_k=4).candidates])  # 4 alternative spellings
```

## Inputs and outputs
- **Input:** one English word or name, up to 32 letters. Case, accents, apostrophes and hyphens are ignored
  (`O'Neil` → `oneil`).
- **Refused:** empty text, several words, digits or non-Latin characters, and longer words. These return a status
  instead of a guess.
- **Output:** one spelling by default. With `--top-k N` you get up to N alternatives, best first. Their scores rank
  the alternatives and are not confidence values.

## Intended use
- **Good for:** drafting Khmer spellings of English loanwords and names for a person to review, for example in
  glossaries or text-to-speech.
- **Not for:** translation, sentences, romanised Khmer ("sok sabay"), names read in their original language, or
  publishing spellings without a check by a Khmer reader.

## Training data
- **Size:** 4,764 English–Khmer word pairs (3,585 English words).
- **Web pairs:** most pairs were mined from Khmer web text where a Khmer spelling is followed by the English word in
  brackets, e.g. "ចនសុន (Johnson)". The sources were [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
  (Khmer subset) and Khmer Wikipedia.
- **ALT pairs:** the rest are names and loanwords spelled in the human Khmer translations of the
  [ALT Parallel Corpus](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/).
- **Checks:** pairs were checked automatically against the English pronunciation. No LLM-generated or hand-written
  spellings were used.

## Evaluation
Two separate samples of 120 unseen words were judged by a person. Each sample held 40 everyday words, 40 words
with irregular English spelling and 40 names. The judge was affiliated with the project and saw the spellings
without knowing which model wrote them.

| Sample | Acceptable | Everyday | Irregular spelling | Names |
|---|---|---|---|---|
| 1 | **64%** (77/120) | 27/40 | 14/40 | 36/40 |
| 2 | **75%** (90/120) | 34/40 | 20/40 | 36/40 |

- **Sampling uncertainty:** the two samples differ by chance; each figure is uncertain by about ±8 points.
- **Composition:** the samples contain more hard words and names than everyday text does.
- **Alternatives help:** for hard words, one of the `--top-k 4` alternatives is often acceptable when the first
  answer is not.

## Speed
On a laptop CPU (Intel i7-8750H):
- **Load time:** about 6 s.
- **One word:** about 0.5 s.
- **Batches:** about 14 words per second.
- **Memory:** about 2 GB.

## Limitations
- **Irregular spellings:** these are the weakest group. Silent letters are sometimes pronounced, and some words
  are read letter by letter.
- **Vowels:** most errors are in vowels.
- **Pronunciation:** spellings follow a US English reading. Words with two pronunciations (*record*, *live*) get
  one.
- **No established forms:** the model writes the English sound, not established Khmer forms (`coffee` gives
  ខូហ្វី, not the usual កាហ្វេ).
- **Web habits:** it can repeat habits of the web data, such as names spelled the way their original language
  reads them.

## Licence
- **Adapter weights:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Non-commercial use
  only, with attribution; share adaptations under the same licence. See `LICENSE`.
- **Base model:** `google/byt5-small` belongs to Google, under Apache 2.0, and is not included here.
- **`khmer-translit` package:**
  [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0).
- **Copyright:** 2026 Richard Cheam.

## Data attribution
- **Wikipedia and FineWeb-2:** training pairs were mined from Khmer Wikipedia (text under CC BY-SA 4.0,
  https://km.wikipedia.org) and FineWeb-2 (HuggingFaceFW/fineweb-2). This contains information from FineWeb-2,
  which is made available under the ODC Attribution License (ODC-By 1.0). Its use is also subject to Common
  Crawl's Terms of Use.
- **ALT:** some Khmer spellings come from the ALT Parallel Corpus (NICT, CC BY 4.0; Riza et al. 2016).
- **Data checks:** these used the CMU Pronouncing Dictionary and Google's Khmer lexicon from `language-resources`
  (CC BY 4.0).

Rights in the original web pages remain with their authors. If you represent a source and have concerns, please
open a discussion on this repository.

## Citation
Base model: Xue et al. 2022, "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models",
*Transactions of the ACL* 10:291–306.
