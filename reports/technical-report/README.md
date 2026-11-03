# Technical report: English-to-Khmer Phonetic Transliteration

*Learning from Noisy Web Pairs and Measuring Acceptability.* Richard Cheam (richardcheam912@gmail.com),
November 2026. Written for ML engineers and practitioners; earlier version 31 October 2026.

This is a technical report. It has not been peer-reviewed, and it has not been submitted to or
published on arXiv or any other venue.

| File | What it is |
|---|---|
| `main.tex` | The report (two columns, 10 pt; one-column appendix with methods and worked examples). Empirical values are `\val{key}` macros; architecture and recipe settings, sample-design constants, dates and bibliographic data are literal. All three figures (model architectures, data and review pipeline, candidate coverage) are TikZ drawings in the source |
| `references.bib` | Bibliography; each entry checked against its primary record |
| `tools/extract.py` | Builds every empirical value at a pinned revision, by kind: `json` (read from a committed result file), `derived` (computed from those) or `transcribed` (copied from a committed report's text or table, with file and section). Writes `data/results_extract.json` and `generated/values.tex`. Asserts the stated relationships and fails if `main.tex` uses an undefined `\val` key |
| `data/results_extract.json` | The checked results extract: value, source file, JSON field, revision, meaning and denominator for each key |
| `generated/` | `values.tex` (macros), generated |
| `EVIDENCE_LEDGER.md` | Claim → file, field, revision, definition, limitation; example and figure provenance; reconciled discrepancies; operational details kept out of the paper |
| `SUMMARY.md` | A practitioner's summary of about 700 words |
| `main.pdf` | The compiled report |
| `tools/build.sh` | Extracts, compiles and writes `technical-report-src.tar.gz` (portable sources) |

## Build
From this folder:

```bash
tools/build.sh                 # python3 tools/extract.py, then latexmk -xelatex main.tex, then the source archive
```

The equivalent explicit commands:

```bash
python3 tools/extract.py       # needs the git repository (reads git show d7e7d80:<file>)
xelatex main && bibtex main && xelatex main && xelatex main
```

From the portable archive (no repository needed), run `tools/build.sh --no-extract`. The generated files are
included in the archive.

**Requirements:**
- XeLaTeX or LuaLaTeX, with `fontspec`, TikZ, `natbib`, `booktabs`, `tabularx`, `caption`, `enumitem`,
  `microtype`, `amssymb`, `xurl` and `hyperref`, plus BibTeX;
- the font **Noto Serif Khmer**, with Khmer shaping by HarfBuzz. XeTeX shapes natively; LuaLaTeX uses
  `Renderer=HarfBuzz`;
- Python 3.10+ and git, for `extract.py` only.

Tectonic (`tectonic main.tex`) also works.

## Data policy
**Included:** only aggregate numbers that are already committed in the repository.

**Not included:** training pairs, judgments, review keys, model weights, and protected reference or reserve
spellings. The Khmer examples are recorded model outputs from committed prediction files, judgments quoted in
committed reports, and a handful of individual training pairs and one mining match, each listed with its source in
`EVIDENCE_LEDGER.md` §6. Full training files and judgment files are not distributed.

## Using the released models
The models are in two Hugging Face repositories:
- the small NumPy model: https://huggingface.co/ric912/en-km-phonetic-transliterator (tag `v0.2.0`);
- the ByT5-small LoRA adapter: https://huggingface.co/ric912/en-km-phonetic-byt5-lora.

The commands below are the tested quick starts from the two model cards.

```bash
# small model (NumPy only)
hf download ric912/en-km-phonetic-transliterator --revision v0.2.0 --local-dir small
pip install small/khmer_translit-0.2.0-py3-none-any.whl
khmer-translit predict --model small/model.npz --word coffee --top-k 3 --json

# ByT5 adapter (CPU PyTorch; the base model is downloaded separately and pinned)
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  "khmer-translit[byt5] @ https://huggingface.co/ric912/en-km-phonetic-transliterator/resolve/v0.2.0/khmer_translit-0.2.0-py3-none-any.whl"
hf download google/byt5-small --revision 68377bdc18a2ffec8a0533fef03b1c513a4dd49d --local-dir byt5-small
hf download ric912/en-km-phonetic-byt5-lora --local-dir byt5-adapter
khmer-translit predict --backend byt5 --model byt5-adapter --base byt5-small --offline --word coffee
```
