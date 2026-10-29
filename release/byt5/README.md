# khmer-translit-byt5 0.2.0 (local research preview)

English word or name → Khmer-script spelling of its English pronunciation. Example: `coffee` → a Khmer spelling
that sounds like "coffee". It does not translate.

**Status:** research preview with **unverified phonetic quality**. Not published, not served. Read
`LIMITATIONS.md` and `NOTICES.md` before using or sharing it.

## What is in this bundle
| Path | What it is |
|---|---|
| `wheel/khmer_translit-0.2.0-py3-none-any.whl`, `wheel/*.tar.gz` | The package (`khmer_translit`), including the `byt5` backend and the `khmer-translit` CLI |
| `adapter/adapter.safetensors`, `adapter/adapter_config.json` | The LoRA adapter (≈ 25 MB) and its config: base revision, base-weights hash, adapter hash, decoding |
| `MODEL_CARD.md`, `BENCHMARK_CARD.md`, `DATA_PROVENANCE.md`, `NOTICES.md`, `LIMITATIONS.md`, `CHANGELOG.md`, `REPRODUCE.md` | Documents |
| `evidence/` | Frozen candidates, gate, results, release verification, install check, confirmation and review manifests |
| `huggingface/` | A prepared model directory (`README.md` = model card, adapter files). **Not uploaded** |
| `SHA256SUMS` | Checksums of every file above |

The base model (`google/byt5-small`, about 1.2 GB) is **not** included. You fetch it once yourself.

## Install
Python 3.10+. CPU only.

```bash
python -m venv .venv && . .venv/bin/activate
pip install "khmer-translit[byt5] @ file://$PWD/wheel/khmer_translit-0.2.0-py3-none-any.whl"
```

The `byt5` extra pulls in `torch`, `transformers` and `safetensors`. Without it, only the historical E02 backend
(NumPy) works.

## Get the base model once (the only download)
```bash
hf download google/byt5-small --revision 68377bdc18a2ffec8a0533fef03b1c513a4dd49d --local-dir byt5-small
```

- The adapter checks the base weights' SHA-256 against `adapter_config.json` and refuses a different base.
- After this you can work fully offline: pass `--base byt5-small --offline`.
- Without `--base`, the package loads `google/byt5-small` at the pinned revision from the Hugging Face cache. With
  `--offline` (or `HF_HUB_OFFLINE=1`) and an empty cache it stops with an error. It never downloads silently in
  offline mode.

## Use
```bash
# one word: one greedy answer
khmer-translit predict --backend byt5 --model adapter --base byt5-small --offline --word coffee

# up to 8 candidates (canonical beam search), as JSON
khmer-translit predict --backend byt5 --model adapter --base byt5-small --offline --word london --top-k 4 --json

# a file of words, one per line → one JSON line per input line, same order, `id` = line number
khmer-translit predict --backend byt5 --model adapter --base byt5-small --offline --input words.txt --jsonl

# what is loaded (revision, hashes, decoding)
khmer-translit inspect --backend byt5 --model adapter --base byt5-small --offline
```

- **`--backend byt5` is required.** The default backend is still the historical E02 model.
- **Refused inputs:** empty lines, several words, unsupported characters and words over 32 letters are refused with
  a `status`, never truncated or guessed. The exit code is 1 when any input was refused, and 2 for usage or file
  errors.
- **`log_score`** is a summed log-probability. It ranks candidates and is **not** a confidence.
- **CPU:** `--threads N` sets the PyTorch threads.

Python:

```python
from khmer_translit.byt5 import ByT5Transliterator

tl = ByT5Transliterator.from_pretrained("adapter", base="byt5-small", offline=True)   # load once
print(tl.predict("coffee").text)
for r in tl.predict_many(["london", "", "doctor"], top_k=3):
    print(r.input, r.status, [c.text for c in r.candidates])
```

## Check the bundle
```bash
sha256sum -c SHA256SUMS
```
