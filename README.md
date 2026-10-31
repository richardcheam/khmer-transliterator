# khmer-transliterator

Latin → Khmer transliteration for the Khmer TTS (`../pocket-tts-khmer`). The speech model reads Khmer script only.
Any Latin word the front-end cannot look up (a romanised Khmer word like *knhom*, a name like *sophea*, an English
word like *safety*) is spelled in Khmer here: by a small learned character model (2.6M parameters, NumPy inference,
about 16 ms a word on CPU), by English sound-out rules, and by re-ranking towards known Khmer words.

**Served today:** model B (names, romanised Khmer) and model E (English dictionary words).
**Current goal:** one clean, releasable model without the Chhunneng data. The gap is real romanised-typing data.
Everything about the data, models, results and next steps: `docs/TRANSLITERATOR.md`. How experiments are run
(plan before training, configs, provenance, W&B, frozen benchmark): `docs/RESEARCH_PROTOCOL.md`.

## Quick start
```bash
uv sync                                                        # numpy + torch, pyarrow, wandb, pyyaml (training group)
.venv/bin/python scripts/data_manifest.py check                # is the local data the pinned version?
.venv/bin/python scripts/evaluate.py models/model_b            # score a model on the frozen benchmark v1
.venv/bin/python scripts/translit_try.py knhom nham bay safety ronaldo    # served vs clean, side by side
.venv/bin/python -c "from khmer_translit import translit; print(translit.guess_many(['sophea', 'safety']))"
.venv/bin/python -m unittest discover -s tests
```
The data and models (`data/`, `models/`) are not in git; they live on this machine.

## Layout
```
src/khmer_translit/        the package (what the speech repo imports)
  translit.py                model inference (NumPy), routing, re-ranking, configure()
  latin_g2p.py               English sound-out rules
  en_g2p.py                  CMUdict pronunciations (phone-aware model input)
  romanise.py                rule romanisation of Khmer words (clean training pairs)
  lexicon.py                 Google Khmer lexicon: spellings, word splitting
  loanwords.py               where the speech repo's loanword list is
  experiment.py              strict configs, provenance, run records + W&B mirror
  benchmark.py               frozen test panels, scoring, paired bootstrap
  english.py                 E02: English word → Khmer phonetic spelling (Transliterator, local model files)
  byt5.py                    the ByT5 LoRA adapter backend (optional `[byt5]` extra: torch, transformers)
  phonetic_source.py         E02: input normalisation, CMUdict pronunciation, source encoding
  reference_data.py          E02: benchmark items, lexical families, splits
  evaluation.py              E02: multi-reference scoring, family bootstrap, evidence fields
  cli.py                     the `khmer-translit` command (prepare / predict / inspect)
scripts/
  translit_data.py           source pairs, splits, Wikidata names
  translit_mine.py           bracket pairs mined from Khmer Wikipedia and FineWeb-2
  translit_extra.py          filters the mined pairs
  translit_roman.py          rule-romanised clean pairs
  translit_train.py          train / export a model (torch → .npz)
  translit_unified.py        experiment E01: data, train, export, score (config-driven, recorded)
  evaluate.py                score any exported model on a benchmark version
  data_manifest.py           pin / check the data (datasets/manifest.json)
  translit_try.py            served vs clean on any words
  english_phonetic_*.py      experiment E02: data, train, run (durable queue), eval, bench, wheelcheck, bundle
  review.py, review_ui.html  local blinded review page (click acceptable / not acceptable / uncertain)
experiments/               one folder per experiment: plan, report, results.json (README.md is the index)
release/                   the public card and licence of each model published on Hugging Face
benchmarks/v1.json         the frozen benchmark: panel hashes and descriptions (panel files in data/benchmark/)
datasets/                  data card, manifest.json (hash of every data file), build/ summaries
model_cards/               the served models: recipe, results, limits
tests/
docs/  TRANSLITERATOR.md (start here), RESEARCH_PROTOCOL.md, DECISIONS.md
data/   (not in git)       pairs, mined pairs, word lists, lexicon, raw sources (2.2 GB in data/src)
models/ (not in git)       model (A), model_b (B, served), model_e (E, served), unified/ (clean candidates)
```

## English word → Khmer phonetic spelling (published, research preview)
Two clean models for one English word at a time, spelled as it sounds in English. Neither is served.
- [ric912/en-km-phonetic-transliterator](https://huggingface.co/ric912/en-km-phonetic-transliterator): the small
  NumPy model (E02 `letters_s0`).
- [ric912/en-km-phonetic-byt5-lora](https://huggingface.co/ric912/en-km-phonetic-byt5-lora): a LoRA adapter for
  `google/byt5-small` (E11), about 35× slower. On two blinded 120-word samples, 64% and 75% of its spellings were
  judged acceptable.

Their cards are in `release/`. Running the small model locally (offline, NumPy only):
```bash
.venv/bin/python -m khmer_translit.cli predict --model models/e02/letters_s0/model.npz --word coffee --top-k 3
.venv/bin/python -m khmer_translit.cli inspect --model models/e02/letters_s0/model.npz
.venv/bin/python -c "from khmer_translit.english import Transliterator as T; print(T.from_pretrained('models/e02/letters_s0').predict('banana').text)"
```
From an installed wheel (or after `uv sync`) the command is simply `khmer-translit`; add `--backend byt5 --model
<adapter> --base <byt5-small>` for the adapter. Example: `coffee` → ខូហ្វ៊ី (then កូហ្វ៊ី, កូហ្វី); `hello world` is
refused (`unsupported_multiword`).

## How the speech repo uses it
`pocket-tts-khmer` installs this package in editable mode (`pyproject.toml`: path `../khmer-transliterator`) and
calls `translit.configure(...)` with the models named in its `configs/serving.json`. To serve a new model: train
and export it into `models/<name>/`, then set `translit_model` (or `translit_en_model`) there and restart the
server. The speech repo's loanword list (`configs/latin_khmer.tsv`) is read here through `khmer_translit.loanwords`
(override the path with `KHMER_LOANWORD_LIST`).

## Licences
Google Khmer lexicon CC BY 4.0 (`data/lexicon/ATTRIBUTION.md`); Khmer Wikipedia CC BY-SA 4.0; FineWeb-2 ODC-By 1.0;
CMUdict BSD-style; SCOWL. The Chhunneng pairs (`data/src/chhunneng/`) are of unverified origin: models trained on
them (A, B, E) are not releasable. This code: PolyForm Noncommercial 1.0.0 (`LICENSE`). Published weights:
CC BY-NC-SA 4.0.
