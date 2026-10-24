# khmer-transliterator

Latin → Khmer transliteration for the Khmer TTS (`../pocket-tts-khmer`). The speech model reads Khmer script only.
Any Latin word the front-end cannot look up (a romanised Khmer word like *knhom*, a name like *sophea*, an English
word like *safety*) is spelled in Khmer here: by a small learned character model (2.6M parameters, NumPy inference,
about 16 ms a word on CPU), by English sound-out rules, and by re-ranking towards known Khmer words.

**Served today:** model B (names, romanised Khmer) and model E (English dictionary words).
**Current goal:** one clean, releasable model without the Chhunneng data. The gap is real romanised-typing data.
Everything about the data, models, results and next steps: `docs/TRANSLITERATOR.md`.

## Quick start
```bash
uv sync                                                        # numpy + torch, pyarrow, wandb (training group)
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
scripts/
  translit_data.py           source pairs, splits, Wikidata names
  translit_mine.py           bracket pairs mined from Khmer Wikipedia and FineWeb-2
  translit_extra.py          filters the mined pairs
  translit_roman.py          rule-romanised clean pairs
  translit_train.py          train / export a model (torch → .npz)
  translit_unified.py        the one-model experiment: data, train, export, score (D173)
  translit_try.py            served vs clean on any words
  track_translit.py          mirror a training run to W&B
tests/
docs/  TRANSLITERATOR.md (start here), DECISIONS.md, FUNCTION_DOCUMENTATION.md
reports/                   results (JSON), e.g. translit_unified.json
data/   (not in git)       pairs, mined pairs, word lists, lexicon, raw sources (2.2 GB in data/src)
models/ (not in git)       model (A), model_b (B, served), model_e (E, served), unified/ (clean candidates)
```

## How the speech repo uses it
`pocket-tts-khmer` installs this package in editable mode (`pyproject.toml`: path `../khmer-transliterator`) and
calls `translit.configure(...)` with the models named in its `configs/serving.json`. To serve a new model: train
and export it into `models/<name>/`, then set `translit_model` (or `translit_en_model`) there and restart the
server. The speech repo's loanword list (`configs/latin_khmer.tsv`) is read here through `khmer_translit.loanwords`
(override the path with `KHMER_LOANWORD_LIST`).

## Licences
Google Khmer lexicon CC BY 4.0 (`data/lexicon/ATTRIBUTION.md`); Khmer Wikipedia CC BY-SA 4.0; FineWeb-2 ODC-By 1.0;
CMUdict BSD-style; SCOWL. The Chhunneng pairs (`data/src/chhunneng/`) are of unverified origin: models trained on
them (A, B, E) are not releasable. Licence for this code: not chosen yet.
