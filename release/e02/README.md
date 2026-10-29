# khmer-translit E02 letters model 0.2.0: local release candidate

English word or name → Khmer-script spelling of its English pronunciation (`coffee` → ខូហ្វ៊ី). Small NumPy model,
CPU only, offline.

> **Research preview. Phonetic quality has not been independently validated. Reported exact-match results measure
> agreement with the specified reference spellings.**

**Status:**
- This is E02 (`letters_s0`), **not** the ByT5 model.
- Not published and not served.
- Publication is **unresolved**: there is no licence, and the open data-permission questions are listed in
  `PUBLICATION_CHECKLIST.md`.

## Contents
| Path | What |
|---|---|
| `checkpoint/model.npz` | The model, sha256 `8901050b1093cf90f60523c7941d6c12dbd8571d71698e37abcf874ba437904b` |
| `wheel/khmer_translit-0.2.0-py3-none-any.whl`, `wheel/khmer_translit-0.2.0.tar.gz` | The package (code only) |
| `requirements.txt` | The tested dependency pin (`numpy==2.2.6`); the wheel declares `numpy>=2` |
| `MODEL_CARD.md`, `BENCHMARK_CARD.md`, `PROVENANCE_AUDIT.md`, `NOTICES.md`, `LICENSING_DECISION.md`, `PUBLICATION_CHECKLIST.md` | Documents |
| `evidence/` | Provenance facts, the build record, the isolated install check, historical results and runtime |
| `huggingface/` | An upload-ready model directory (`README.md` = model card, `model.npz`, notices). **Not uploaded** |
| `SHA256SUMS` | Checksums of every file |

## Install (tested)
The check ran in a fresh `uv` venv outside the repository, with no `PYTHONPATH`, no system site-packages and no
`.pth` bridge. NumPy came from the declared requirement (offline from the uv cache).

```bash
sha256sum -c SHA256SUMS
uv venv --python 3.10 .venv            # or: python -m venv .venv
uv pip install --python .venv/bin/python wheel/khmer_translit-0.2.0-py3-none-any.whl
#   plain pip: .venv/bin/pip install -r requirements.txt wheel/khmer_translit-0.2.0-py3-none-any.whl
```

Only `khmer-translit` and `numpy` get installed. The optional `[byt5]` extra (torch, transformers) belongs to the
other model and is **not** needed here.

## Use (tested commands and outputs)
```bash
.venv/bin/khmer-translit predict --model checkpoint/model.npz --word coffee --top-k 3 --json
# {"id": 1, "backend": "e02", "input": "coffee", "normalized": "coffee", "status": "ok",
#  "candidates": [{"text": "ខូហ្វ៊ី", "log_score": -1.6166}, {"text": "កូហ្វ៊ី", "log_score": -1.6696},
#                 {"text": "កូហ្វី", "log_score": -2.6095}], "pronunciation": null, "message": ""}

printf 'computer\n\nhello world\nJohnson\n' | .venv/bin/khmer-translit predict --model checkpoint/model.npz --input - --jsonl
# one JSON line per input line, same order, "id" = line number; statuses ok, empty, unsupported_multiword, ok;
# exit status 1 because some inputs were refused

.venv/bin/khmer-translit inspect --model checkpoint/model.npz        # format, hashes, input policy
```

```python
from khmer_translit.english import Transliterator
tl = Transliterator.from_pretrained("checkpoint/model.npz")
tl.predict("coffee").text                     # 'ខូហ្វ៊ី'
[r.status for r in tl.predict_many(["computer", "", "a" * 33])]   # ['ok', 'empty', 'too_long']
```

- **Refused, never truncated:** empty input, several words, other characters (digits, Khmer) and more than 32
  letters.
- **Exit codes:** 0 when all inputs succeed, 1 when some were refused, 2 for usage errors and missing or corrupt
  models.
- **Integrity:** `--sha256 HASH` refuses a model with a different hash.
- **Network:** no network access is used after installation.

## Verified (`evidence/install_verify.json`)
All 17 checks pass. Highlights:
- **Parity:** identical top-4 candidates and scores, to the bit, versus the 400 outputs recorded during the E02
  evaluation, and versus the package code at the training commit on 103 dev inputs.
- **Robustness:** deterministic repeats; corrupt, truncated and missing models give exit 2.
- **Isolation:** no torch, transformers, PEFT or speech modules are imported.
- **Speed (i7-8750H laptop):** load 0.15 s; 17 ms per word p50 (30 ms p95); peak RSS 65 MiB.
