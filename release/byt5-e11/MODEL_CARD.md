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

# en-km-phonetic-byt5-lora (E11): English word → Khmer phonetic spelling

> **Not published yet.** Adapter weights: CC BY-NC-SA 4.0 (`LICENSE`); package code: PolyForm Noncommercial 1.0.0.
> It's a research-quality model: one training run,
> judged by one project-affiliated person, not independently validated.

**This is a LoRA adapter, not a standalone model.** It needs the base model `google/byt5-small` at revision
`68377bdc18a2ffec8a0533fef03b1c513a4dd49d` and the `khmer-translit` package with its `[byt5]` extra. This is unlike
the existing standalone NumPy model
[`ric912/en-km-phonetic-transliterator`](https://huggingface.co/ric912/en-km-phonetic-transliterator), which is a
separate, older model and stays unchanged.

## Model details
- **Task:** one English word or name → Khmer script that approximates its English (General American) pronunciation.
  It writes sound, not meaning. For example, E11's greedy output for `coffee` is ខូហ្វី and for `london` ឡុនដុន
  (from the verified smoke check). This is the English sound, not the established Khmer loanword កាហ្វេ.
- **Architecture:** `google/byt5-small`, frozen (300M parameters; base weights `pytorch_model.bin`, sha256
  `5c5aaf56299d6f2d4eaadad550a40765198828ead4d74f0a15f91cbe0961931a`). LoRA adapters (r 16, α 32) sit on
  attention q/k/v/o and on the gated feed-forward wi_0/wi_1/wo. That's 256 tensors and 6,258,688 parameters.
- **Files:**
  - `adapter.safetensors`, sha256 `698a23ae1526d3da8f3b2bd00f0177799a939d2005c165569fcba7d58e76cc3f`;
  - `adapter_config.json`: the base pin, hashes, decoding settings and provenance.
- **Checkpoint:** experiment E11, run `r_s0`, update 3,000 of 3,000, seed 0. Training checkpoint sha256
  `286292a4f21131d0dd238d48919485192b0b0eb268d005bbc804d3ff78229713`. The export to safetensors is bit-identical,
  tensor by tensor.
- **Initialisation:** a fresh LoRA adapter on the pinned base. It isn't initialised from any other project
  adapter (E08, E10, E12 or E13).
- **Developed by:** the khmer-transliterator project (one owner), on a laptop CPU.

## Intended use
- **For:** drafting Khmer phonetic spellings of English loanwords and names for a person to check, for example in
  glossaries or text-to-speech front-ends.
- **Not for:**
  - translation, sentences, romanised Khmer chat or other source languages;
  - names read in their original language;
  - unchecked publication of spellings.

  See `LIMITATIONS.md`.

## How to use
These commands were tested in a fresh virtual environment, with networking disabled at inference
(`evidence/install_check.json`).

```bash
# 1. Package (CPU-only PyTorch). The wheel ships in the release bundle.
uv pip install --torch-backend cpu "khmer-translit[byt5] @ file:///path/to/khmer_translit-0.2.0-py3-none-any.whl"
# or: pip install --extra-index-url https://download.pytorch.org/whl/cpu "khmer-translit[byt5] @ file:///…whl"

# 2. Base model at the pinned revision (about 1.2 GB, once)
hf download google/byt5-small --revision 68377bdc18a2ffec8a0533fef03b1c513a4dd49d --local-dir byt5-small

# 3. This adapter (adapter.safetensors + adapter_config.json in one folder)
hf download ric912/en-km-phonetic-byt5-lora --local-dir e11-adapter     # after publication only

# 4. Predict, offline
khmer-translit predict --backend byt5 --model e11-adapter --base byt5-small --offline --word coffee
khmer-translit predict --backend byt5 --model e11-adapter --base byt5-small --offline --word london --top-k 4 --json
khmer-translit predict --backend byt5 --model e11-adapter --base byt5-small --offline --input words.txt --jsonl
khmer-translit inspect --backend byt5 --model e11-adapter --base byt5-small --offline
```

```python
from khmer_translit.byt5 import ByT5Transliterator
tl = ByT5Transliterator.from_pretrained("e11-adapter", base="byt5-small", offline=True, threads=6)
tl.predict("coffee").text                                   # one greedy answer: ខូហ្វី
[c.text for c in tl.predict("london", top_k=4).candidates]  # up to 4 distinct canonical-beam candidates
tl.predict_many(["taxi", "new york", ""])                   # input order kept; refused inputs carry a status
```

- **Loading:** this is **not a PEFT or `AutoModel` checkpoint**, and the hosted inference widget isn't
  supported. The adapter loads only through `khmer_translit.byt5`, which wraps the base's linear layers in plain
  PyTorch.
- **Base pin:** loading refuses an adapter whose recorded base revision or base-weights hash differs from the
  pinned one.
- **Input:** one word, normalised by lower-casing and removing accents, apostrophes and hyphens; letters a–z only;
  at most 32 bytes. Empty, multi-word, non-Latin, digit-containing and over-length inputs are refused with a
  status. They are never truncated or split.
- **Decoding:**
  - single answer: greedy, at most 121 new bytes;
  - candidates: canonical beam search, width max(k, 4), `early_stopping="never"`, `length_penalty=0`.
    `early_stopping=True` truncates byte-level beams (project experiment E09).
- **Scores** are summed log-probabilities. They rank candidates and are not calibrated confidence.
- **No fallback:** the `byt5` backend never silently uses another model. The CLI default backend is still the
  E02 NumPy model, so `--backend byt5` is required.

## Training data
The model was trained on 4,764 English → Khmer pairs over 3,585 words: frozen arm R, `arm_R.tsv`, sha256
`d00706cf…8ec88`, with optimisation-row hash `5dd38022…c1d61`. There are **no Google AI Mode or other
LLM-generated labels**, which was verified from the file (`DATA_PROVENANCE.md`).

| Source | Rows |
|---|---|
| Brackets mined from FineWeb-2 `khm_Khmr` and Khmer Wikipedia, automatic consonant screen | 2,255 |
| Mined pairs recovered from that screen by documented rules | 1,292 |
| Spellings attested verbatim in the ALT Parallel Corpus (NICT, CC BY 4.0) | 1,217 |

No human reviewed the training labels. Every evaluation and test word is filtered out of training by word family.

## Training procedure
- **Optimiser:** AdamW, learning rate 1e-3, betas (0.9, 0.999), no weight decay, gradient clip 1.0.
- **Schedule:** warm-up 100 updates, then linear decay; batch 32; 3,000 updates.
- **Regularisation:** LoRA dropout 0.05, base dropout 0.1.
- **Run:** 4.6 h on 6 CPU threads, one run (seed 0).
- **Selection:** dev greedy exact agreement with mined references, then CER, then the earlier update.

## Evaluation
`EVALUATION.md` has the full tables and how each number was made. In short:

| Evidence | Sample | E11 result | What it is |
|---|---|---|---|
| Mined-reference agreement (exploratory) | historical eval, 228 items | 52.3% exact, CER 0.163 | agreement with unreviewed web spellings; the split was examined many times |
| Owner dev review | 60 dev words (40 representative + 20 disagreement) | acceptable 29/40 and 10/20 | development evidence |
| Blinded test 1 (E12) | 120 fresh families, 40 ordinary / 40 irregular / 40 names | **77/120 confirmed acceptable (64.2%)**, 3 uncertain | one judge, blinded |
| Blinded test 2 (E13) | another 120 fresh families, same 40/40/40 design | **90/120 confirmed acceptable (75.0%)**, 1 uncertain | one judge, blinded |

- **Two test samples, not two versions.** The checkpoint is identical in both tests. The two figures come from two
  different word samples, judged on different days. They aren't successive improvements and aren't pooled.
- **Judging:** one project-affiliated judge (the owner) judged each deduplicated, blinded spelling under the
  project's pronunciation policy (General American reading, shown as CMUdict ARPAbet). This isn't an independent
  or population benchmark.
- **Weakest group:** irregular spellings, at 14/40 and 20/40 across the two tests.
- **Candidate coverage** (development diagnostic, `EVALUATION.md` §5): whether an acceptable spelling exists among
  4 beam candidates. It is not system accuracy.

## Compute and runtime
See `EVALUATION.md` §6 for load time, latency, throughput and memory, measured with 6 threads on the laptop CPU.

## Limitations, risks and licence
- **Errors:** vowel choices dominate. Irregular English spellings are weak, and silent letters and letter-by-letter
  readings occur. See `LIMITATIONS.md`.
- **Licences:**
  - base model: Apache 2.0;
  - package code: PolyForm Noncommercial 1.0.0 (`LICENSE-code`);
  - adapter weights: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (`LICENSE`).
    Non-commercial use only, with attribution; share adaptations under the same licence;
  - copyright: 2026 Richard Cheam (khmer-transliterator project).
- **Data terms:** FineWeb-2 is ODC-By 1.0 plus the Common Crawl terms, and the rights in the underlying pages are
  not granted. Khmer Wikipedia is CC BY-SA 4.0, and whether its share-alike reaches the weights is open. ALT is
  CC BY 4.0. See `NOTICES.md`.
- **Attribution:**
  > Training pairs include material mined from FineWeb-2 (HuggingFaceFW/fineweb-2, ODC-By 1.0) and Khmer Wikipedia
  > (CC BY-SA 4.0), and Khmer spellings attested in the ALT Parallel Corpus (NICT, CC BY 4.0; Riza et al. 2016,
  > "Introduction of the Asian Language Treebank", Oriental COCOSDA). The base model is google/byt5-small
  > (Apache 2.0; Xue et al. 2022, TACL 10:291–306).

## Citation
Xue et al. 2022, "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models", TACL 10:291–306
(2022.tacl-1.17).
