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

## D177 — Research protocol and tooling (2026-10-24)
- **Owner's request:** make the repository follow research-grade procedure, as the base for continued work.
- **Added:**
  - **Protocol:** `docs/RESEARCH_PROTOCOL.md` (plan → run → evaluate → results → decide), one folder per
    experiment, `experiments/TEMPLATE/`.
  - **`khmer_translit.experiment`:**
    - strict configs: every key required, unknown keys refused;
    - provenance: commit, uncommitted-change flag and diff hash, versions, input hashes;
    - per-run records (`config.json`, `provenance.json`, `metrics.jsonl`, `summary.json`), mirrored to W&B (team
      `richardcheam912-ensiie`, project `khmer-transliterator`, group = experiment). The team is fixed in code,
      because the shell's `WANDB_ENTITY` names an account W&B does not know. Override runs go to a separate
      `<experiment>-smoke` group.
  - **`khmer_translit.benchmark`:** frozen, hash-checked panels. Version v1 = the D173/D174 panels; the reserve
    stays sealed. Also `outcomes` and `paired_bootstrap`.
  - **Scripts:** `scripts/evaluate.py` (any model on a benchmark version); `scripts/data_manifest.py` and
    `datasets/manifest.json` (212 data files pinned); the data card `datasets/README.md`; data-build summaries in
    `datasets/build/`.
  - **`experiments/E00-history/`:** the pre-protocol arms and their reports. **`experiments/E01-unified/`:** the
    D173 plan, config, results and the owner's listening.
  - **`model_cards/`:** B and E, with benchmark-v1 results.
- **`translit_unified.py` is now config-driven:**
  - it reads `experiments/E01-unified/config.yaml` and refuses data, model or path settings it does not implement;
  - `--set` overrides require `--out`, so a test never writes into a registered run;
  - it records each run through `experiment.Run`, and scoring uses benchmark v1.

  `track_translit.py` (the old W&B follower) is removed.
- **Verified:**
  - Old vs new trainer, aux arm, seed 0, one epoch: identical training log (losses, validation hits, batch-order
    hash) and byte-identical checkpoint.
  - Re-scoring the six E01 models through benchmark v1 wrote a fresh `results.json`, byte-identical to D174's.
  - `evaluate.py` reproduces the E01 reference numbers for B and E exactly.
  - An end-to-end smoke run (CLI, override, records, W&B) completed.
  - 29 tests pass.
- **Mistake during verification:** four memory-heavy jobs at once ran the 7.6 GB laptop out of memory. The kernel
  killed one evaluation and a re-score, plus the desktop SSH agent and two dbus processes. A re-score check then
  passed falsely, because it compared an unwritten file with itself. It was redone alone, with the old file moved
  aside first. **Rule kept:** run one heavy job at a time.
