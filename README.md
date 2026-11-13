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
pip install "khmer-translit[hub]"          # NumPy + huggingface_hub; the model (10 MB) downloads on first use

khmer-translit predict --word coffee --top-k 3 --json
printf 'computer\nJohnson\nLondon\n' | khmer-translit predict --input - --jsonl
```
```python
from khmer_translit.english import Transliterator

tl = Transliterator.from_pretrained()     # the published model (pinned, sha256-checked); load once, reuse
print(tl.predict("coffee").text)                                # ខូហ្វ៊ី
```
The ByT5 adapter:
```bash
pip install "khmer-translit[byt5]"         # adds PyTorch and Transformers
khmer-translit predict --backend byt5 --word coffee --top-k 4 --json
```
The adapter and the base model (`google/byt5-small`, pinned revision) are downloaded on first use; `--model FOLDER`
and `--base FOLDER` use local copies, and `--offline` never downloads. Without `[hub]`, download `model.npz`
yourself and pass `--model path/to/model.npz`.

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

## Reproduce them
[`TRAINING.md`](TRAINING.md) runs the whole pipeline: prepare the training data from raw inputs
([`training-data/raw/`](training-data/raw/README.md); the result is checked byte for byte against the published
training files), train the small model and the ByT5 adapter, and evaluate on every published test set
([`test-sets/`](test-sets/README.md)) next to the published models' numbers. `pipeline/run_all.sh` does it in one
command. In our check, retraining the small model on a CPU gave exactly the published model.

## Repository
| Path | What it is |
|---|---|
| `src/khmer_translit/` | the package: `english.py` (small model), `nnet.py` (its NumPy network), `byt5.py` (ByT5 adapter backend), `phonetic_source.py` (input normalisation and encoding), `cli.py` (`khmer-translit`) |
| `tests/` | package tests |
| `training/` | training code and settings (PyTorch), with its own tests |
| `training-data/` | raw inputs (`raw/`) and the prepared training and development data of both models, with sources and notices |
| `test-sets/` | the test sets behind the published numbers, and what each was used for |
| `pipeline/` | data preparation, evaluation and the run-everything script |
| `report/` | the technical report (PDF) |

## Licence
- **Code:** MIT (`LICENSE`), from version 0.3.0 (0.2.x was PolyForm Noncommercial).
- **Model weights:** CC BY-SA 4.0 (see the model cards), since 2026-11-13 (before: CC BY-NC-SA 4.0).
- **Training data:** CC BY-SA 4.0; the sources keep their own terms (`training-data/NOTICE.md`).
