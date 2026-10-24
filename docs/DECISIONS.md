# Decisions

Consequential choices and results, newest last. Numbers continue the speech repo's log: D001–D174 are archived
there (`git -C ../pocket-tts-khmer show archive/checkpoints-p0-p6:state/DECISIONS.md`), and D175 is its cleanup.

## D176 — The transliterator becomes its own repository (2026-10-24)
- **Owner's request:** the transliterator is a model of its own, while the speech fine-tune will change little.
  Give each its own repository, so each has one job and a thinner context.
- **Design (owner's choice, "package"):**
  - This repository owns all transliterator code: inference, rules, lexicon, training, data and experiments.
  - The package `khmer_translit` is installed in editable mode by the speech repo (path dependency
    `../khmer-transliterator`, plus an editable install in its upstream torch env).
  - The speech front-end calls `translit.configure(...)` with the models named in its `configs/serving.json`.
    Without the call, the package uses the same defaults (`SERVED`: B, router "all", E).
  - The loanword list stays in the speech repo; it is read here through `khmer_translit.loanwords`.
- **Moved here:**
  - **Code:** `translit.py`, `latin_g2p.py`, `en_g2p.py`, `romanise.py`, `lexicon.py`, 8 scripts, 2 test files, the
    transliterator reports, `docs/TRANSLITERATOR.md`.
  - **Files:** `data/translit/*` → `data/`; models A, B, E and `unified/` → `models/`. The Google lexicon →
    `data/lexicon/`.
- **Archived** to `/mnt/hdd/data/khmer-tts-archive/data/translit/`: unused old A/B folders, the failed ranker, raw
  LLM draft batches, the cancelled B′/E′ folders, the training logs of the archived arms, and two pre-change backups.
- **Own environment:** Python 3.10 with numpy, torch 2.14.0 (the same build as the speech repo's upstream env, from
  the shared uv cache), pyarrow and wandb.
- **Verified:**
  - 22 tests pass here. With the speech repo, 22 tests pass with each of its interpreters.
  - A test sentence gives byte-identical audio before and after the split.
  - `translit_try.py` gives the same spellings as before.
  - Re-scoring the six unified models reproduces `reports/translit_unified.json`.
