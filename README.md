# khmer-transliterator

**English word → Khmer phonetic spelling.** Give it one English word or name and it writes how the word *sounds*
in Khmer script. It does not translate.

```
coffee → ខូហ្វ៊ី    Johnson → ចនសុន
```

Two models, both research previews (their quality has not been independently validated):

| Model | What it is | Hugging Face |
|---|---|---|
| **Small model** | a character Transformer trained from scratch (2.6M parameters, 10.5 MB); CPU, NumPy only, about 17 ms a word | [ric912/en-km-phonetic-transliterator](https://huggingface.co/ric912/en-km-phonetic-transliterator) |
| **ByT5 adapter** | LoRA layers on `google/byt5-small`; better spellings, about 35 times slower, needs PyTorch | [ric912/en-km-phonetic-byt5-lora](https://huggingface.co/ric912/en-km-phonetic-byt5-lora) |

Technical report: [`report/khmer-transliterator-technical-report.pdf`](report/khmer-transliterator-technical-report.pdf).

## Use
```bash
pip install git+https://github.com/richardcheam/khmer-transliterator                 # NumPy only
hf download ric912/en-km-phonetic-transliterator model.npz --local-dir small-model

khmer-translit predict --model small-model/model.npz --word coffee --top-k 3 --json
printf 'computer\nJohnson\nLondon\n' | khmer-translit predict --model small-model/model.npz --input - --jsonl
```
```python
from khmer_translit.english import Transliterator

tl = Transliterator.from_pretrained("small-model/model.npz")   # load once, reuse
print(tl.predict("coffee").text)                                # ខូហ្វ៊ី
```
The ByT5 adapter:
```bash
pip install "khmer-translit[byt5] @ git+https://github.com/richardcheam/khmer-transliterator"
hf download ric912/en-km-phonetic-byt5-lora --local-dir byt5-adapter
khmer-translit predict --backend byt5 --model byt5-adapter --word coffee --top-k 4 --json
```
The base model (`google/byt5-small`, pinned revision) is downloaded on first use, or passed with `--base FOLDER`.

**Inputs:** one English word or name, at most 32 letters (normalised: NFKC, lower case, accents removed,
apostrophes and hyphens dropped). Empty input, several words, digits or other scripts, and longer words are refused
with a status rather than guessed. **Outputs:** up to 8 candidate spellings, best first, with model log-scores for
ranking (not calibrated confidence).

## How well it works
| | Small model | ByT5 adapter |
|---|---|---|
| Top answer equals a reference spelling checked by the author (165 loanwords and names) | 16.7% | |
| Spellings judged acceptable by one native speaker on two blinded samples of 120 unseen words | | 64% and 75% |

Exact agreement with one reference spelling undercounts spellings that sound right but are written differently.
The judge was affiliated with the project and saw the spellings without knowing which system wrote them. Words with
irregular English spelling are the weakest group for both models; spellings follow a US English reading; the
models write the English sound, not established Khmer forms (`coffee` gives ខូហ្វ៊ី, not the usual កាហ្វេ).
Details and caveats: the model cards and the report.

## Train them yourself
[`TRAINING.md`](TRAINING.md) retrains both models from scratch with the code in [`training/`](training/) and the
exact training data in [`training-data/`](training-data/README.md).

## Repository
| Path | What it is |
|---|---|
| `src/khmer_translit/` | the package: `english.py` (small model), `nnet.py` (its NumPy network), `byt5.py` (ByT5 adapter backend), `phonetic_source.py` (input normalisation and encoding), `cli.py` (`khmer-translit`) |
| `tests/` | package tests |
| `training/` | training code and settings (PyTorch), with its own tests |
| `training-data/` | the training and development data of both models, with sources and notices |
| `report/` | the technical report (PDF) |

## Licence
- **Code:** PolyForm Noncommercial 1.0.0 (`LICENSE`).
- **Model weights:** CC BY-NC-SA 4.0 (see the model cards).
- **Training data:** CC BY-SA 4.0; the sources keep their own terms (`training-data/NOTICE.md`).
