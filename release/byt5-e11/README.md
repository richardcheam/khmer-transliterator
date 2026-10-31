# khmer-translit-byt5-e11 0.2.0: local release bundle (E11 ByT5-small adapter)

**What this is:** English word → Khmer phonetic spelling, using the E11 LoRA adapter on `google/byt5-small`
(revision `68377bdc…`). It's a new bundle built from E11's own checkpoint and evidence. It isn't the E10/E08
bundle (`khmer-translit-byt5-0.2.0`) renamed, and it isn't related to the public NumPy model
`ric912/en-km-phonetic-transliterator`.

**Status:** not published. The adapter licence is undecided (`LICENSING_DECISION.md`).

## Contents
| Path | What |
|---|---|
| `wheel/khmer_translit-0.2.0-py3-none-any.whl`, `wheel/khmer_translit-0.2.0.tar.gz` | The package (code licence PolyForm Noncommercial 1.0.0, `LICENSE-code`) |
| `adapter/adapter.safetensors`, `adapter/adapter_config.json` | The E11 adapter (sha256 `698a23ae…`) and its configuration: base pin, decoding, provenance |
| `requirements.txt` | Dependency specification (CPU) |
| `MODEL_CARD.md`, `EVALUATION.md`, `LIMITATIONS.md` | What it does, how it was measured, what it can't do |
| `DATA_PROVENANCE.md`, `NOTICES.md`, `LICENSING_DECISION.md` | Training sources, terms, open decisions |
| `PUBLICATION_CHECKLIST.md`, `CHANGELOG.md` | The prepared (not executed) upload steps, and the change history |
| `huggingface/` | The prepared Hugging Face repository contents, with `MANIFEST.json` |
| `evidence/` | Identity, export, verification, install check, coverage and score summaries (aggregates only) |
| `SHA256SUMS` | Per-file checksums. The archive's checksum is in `../khmer-translit-byt5-e11-0.2.0.tar.gz.sha256` |

**Not included:**
- the base model, which you download at the pinned revision;
- the training checkpoint (`.pt`);
- training pairs, mined references, review batches, origin keys, judgments, caches and secrets.

## Install and run (CPU)
```bash
uv venv --python 3.10 kt && source kt/bin/activate
uv pip install --torch-backend cpu "khmer-translit[byt5] @ file://$PWD/wheel/khmer_translit-0.2.0-py3-none-any.whl"
hf download google/byt5-small --revision 68377bdc18a2ffec8a0533fef03b1c513a4dd49d --local-dir byt5-small
khmer-translit predict --backend byt5 --model adapter --base byt5-small --offline --word coffee
khmer-translit predict --backend byt5 --model adapter --base byt5-small --offline --word london --top-k 4 --json
khmer-translit predict --backend byt5 --model adapter --base byt5-small --offline --input words.txt --jsonl
```

- **`--sha256`:** add `--sha256 698a23ae1526d3da8f3b2bd00f0177799a939d2005c165569fcba7d58e76cc3f` to refuse any
  other adapter.
- **Exit status:** 0 when every input was answered, 1 if any input was refused, 2 for file or usage errors (a
  missing or corrupt adapter, or a missing base).
- **Checking downloads:** `sha256sum -c SHA256SUMS` checks every file in the bundle.
