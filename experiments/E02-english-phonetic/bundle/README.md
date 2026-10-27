# English → Khmer phonetic transliterator (E02) — local research preview

Give it one English word or name; it writes how the word **sounds in English**, in Khmer script (*coffee* →
ខូហ្វ៊ី). It does not translate. Everything runs offline on CPU with NumPy.

**Evidence status: inconclusive.** The benchmark is unqualified (no independent review of the reference spellings),
so the scores in `run/RESULTS.md` are agreement with the owner's list, not proven phonetic quality. The default model
is the plain letters baseline; see `FINAL_REPORT.md`. **Local only**: read `NOTICES.md` before sharing anything.

## Install (from this folder)
```bash
uv venv -p 3.10 .venv-e02 && VIRTUAL_ENV=.venv-e02 uv pip install wheel/khmer_translit-*.whl   # needs numpy>=2
.venv-e02/bin/khmer-translit prepare --manifest prepare_manifest.json      # checks every file's sha256, no downloads
```
(`python -m venv .venv-e02 && .venv-e02/bin/pip install wheel/khmer_translit-*.whl` works as well.)

## Use
```bash
M=checkpoints/$(cat checkpoints/DEFAULT)/model.npz
.venv-e02/bin/khmer-translit predict --model $M --word coffee --top-k 3 --json   # one word, candidates + log-scores
printf 'computer\nbanana\nJohnson\n' | .venv-e02/bin/khmer-translit predict --model $M --input - --jsonl
.venv-e02/bin/khmer-translit inspect --model $M                                 # metadata, checksums, evidence status
# phone-input research model: needs CMUdict
.venv-e02/bin/khmer-translit predict --model checkpoints/phones_s0/model.npz \
    --cmudict resources/cmudict/cmudict.dict --word record --pronunciation "R IH0 K AO1 R D"
```
Python:
```python
from khmer_translit.english import Transliterator
tl = Transliterator.from_pretrained("checkpoints/letters_s0/model.npz")
print(tl.predict("coffee", top_k=3).to_dict())
```
- Input: one English word, `a–z` after normalisation (lower case, accents removed, apostrophes/hyphens dropped),
  at most 32 letters. Empty, multi-word, non-Latin or too long inputs are refused with a status, never guessed.
- `log_score` is the model's beam log-probability, **not** a probability of being right.
- Exit status: 0 all ok, 1 some input refused or a check failed, 2 usage/file error (missing or corrupt model,
  missing CMUdict, checksum mismatch).

## Measured demo
Run on 2026-10-27 from a clean venv (wheel installed from the uv cache, Python 3.10), in a folder laid out like this
bundle, with no CMUdict environment variable. Output copied as printed:
```text
$ khmer-translit predict --model $M --word coffee --top-k 3 --json
{"id": 1, "input": "coffee", "normalized": "coffee", "status": "ok",
 "candidates": [{"text": "ខូហ្វ៊ី", "log_score": -1.6165986741551936},
                {"text": "កូហ្វ៊ី", "log_score": -1.6696176475814446},
                {"text": "កូហ្វី", "log_score": -2.6094601861724205}],
 "pronunciation": null, "message": ""}                         (indentation compacted)
$ printf 'computer\nbanana\nJohnson\n' | khmer-translit predict --model $M --input - --jsonl
{"id": 1, "input": "computer", "normalized": "computer", "status": "ok", "candidates": [{"text": "កំពូទែរ", "log_score": -3.373198978254069}], "pronunciation": null, "message": ""}
{"id": 2, "input": "banana", "normalized": "banana", "status": "ok", "candidates": [{"text": "បាណាណា", "log_score": -0.8378093634280463}], "pronunciation": null, "message": ""}
{"id": 3, "input": "Johnson", "normalized": "johnson", "status": "ok", "candidates": [{"text": "ចនសុន", "log_score": -0.8865165169312785}], "pronunciation": null, "message": ""}
$ printf 'television\nhello world\nfacebook\nchocolate\n' | khmer-translit predict --model $M --input -; echo "exit $?"
television	តេឡេវីស្យូន
hello world	[unsupported_multiword]
facebook	ហ្វេសប៊ុក
chocolate	ឆូកូឡេត
exit 1
$ khmer-translit predict --model checkpoints/phones_s0/model.npz --cmudict resources/cmudict/cmudict.dict --word record --pronunciation "R IH0 K AO1 R D"
record	រ៉េខដ
```
Exit 1 in the plain-text run is the refused multi-word line. The phone model mostly follows the letters: for
*record*, the verb reading (R IH0 K AO1 R D) changes only the second candidate (រ៉េក័រ), not the first.

## Contents
`wheel/` package · `checkpoints/` all completed runs (`DEFAULT` = the default) · `resources/` CMUdict + notices ·
`cards/` model and benchmark cards · `config/` registered plan, config, data manifests · `run/` run ledger,
results, runtime and wheel checks · `outcomes/` per-item predictions and outcomes · `SHA256SUMS`.

Rebuild everything from the repository (branch `english-phonetic`): see "Rerun" in `FINAL_REPORT.md`.
