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

## D178 — E02: English words to Khmer phonetic spelling (2026-10-27)
- **Task (owner's handoff, `handoff/english-phonetic/`):** one English word → its English pronunciation written in
  Khmer script; translation, romanised Khmer chat and audio out of scope. Executed autonomously; every ordinary
  choice is logged in `docs/decisions/english-phonetic.md` (1–27).
- **Audit:** E01 was letters-only input (its exported models say `phones: false`), with a Latin → Google-phones side
  decoder; `docs/TRANSLITERATOR.md` corrected. Only historical model E (Chhunneng-trained) has phone input.
- **Benchmark `en-phonetic-v1`** (`data/e02/references.jsonl`): dev 103, final 191 (primary 165 items / 162
  families), proxy 400, from the owner's loanword list; split by lexical family; no training overlap. **Unqualified**
  (no independent review under the phonetic policy), so the registered evidence status is `inconclusive`.
- **Experiment** (`experiments/E02-english-phonetic/`, registered at `b9c7ea4` before training): letters vs
  letters + CMUdict phones vs phones + Khmer P2G pretraining/replay; seed-0 pilots, the dev winner (phones) and
  letters confirmed at seeds 1–2. 7/7 supervised runs, 1/3 P2G jobs, 2.81 h of CPU time, no GPU, no spending.
- **Result:** phones vs letters on the primary track +1.65 points (seeds +0.62/+1.54/+2.78), 95% paired family
  bootstrap [−1.65, +4.94]: a meaningful (≥ 5) gain ruled out on this benchmark, a positive effect not shown. P2G:
  no gain in its one-seed pilot. Proxy track +1.42 [−0.62, +3.56].
- **Decision:** `letters_baseline_retained`; default model `models/e02/letters_s0` (raw 16.7% primary agreement).
  Nothing changes in serving (still B/E; the owner decides).
- **Product:** `khmer_translit.english.Transliterator` and the `khmer-translit` CLI (prepare / predict / inspect),
  offline, NumPy only; installed-wheel checks pass outside the checkout (`reports/wheel_check.json`); runtime guards
  pass (letters: cold 0.09 s, p95 32 ms/word, 65 MiB). Local bundle `dist/english-phonetic-0.1.0/` + ZIP; weights
  `restricted_artifacts_local_only`.
- **Open:** independent review of the references (the only way to qualify the benchmark); the share-alike status
  of weights trained on Wikipedia pairs; a code licence.

## D179 — E02 diagnostics: pronunciation compatibility and phone sensitivity (2026-10-27)
- **Scope:** diagnostics only (`experiments/E02-diagnostics-2026-10-27/REPORT.md`); E02's registered result and
  verdict (`inconclusive`, letters retained) are unchanged.
- **Compatibility:**
  - No training target has an exact Google-lexicon reading: the miner's `khmer_like` class, which E02 drops, is
    assigned exactly when the Khmer side is a known word.
  - On consonants read from the spelling (rules validated at 88.6% on held-out lexicon headwords), 69.9% of rows
    show compatible evidence and 30.1% a suspected conflict.
  - 54% of rows belong to words with several targets; translations and compounds are present among the worst rows.
- **Sensitivity** (saved checkpoints, frozen dev sample):
  - donor phones change the top-1 output for 33–45% of words, yet rarely move its consonants toward the donor
    (2–7% of cases);
  - only 7 words change in all three seeds;
  - the letters controls never change.
- **Recommended next (not launched):** re-admit mined pairs whose Khmer side is a known word (8,725 before
  filters), keep the exact-reading-consistent ones, and test letters vs phones with phone steering as the primary
  outcome.

## D180 — E03: English phones under dictionary-assessable supervision (2026-10-27)
- **Repair:** the miner's `khmer_like` merged "romanised-Khmer Latin input" with "known Khmer target", and E02
  dropped both. E03 separates them (the stored class is reproduced for all 65,422 pairs):
  - romanised Latin stays excluded, with 549 detector collisions with English words recorded;
  - a known Khmer target alone is no reason to reject: 11,377 rows recovered.
- **Data (DACS):**
  - after E02's unchanged filters, 4,239 pairs have an exact Google reading;
  - the frozen consonant screen keeps 2,967 (2,485 words), split by family 2,348 / 323 / 296;
  - quarantine is kept on file. Post hoc, it over-flags doubled letters and glides and misses the coda s → h.
- **Two arms, three paired seeds** (6/6 attempts, 0.56 h).
- **Quality proxy (eval):** phones − letters +0.60 points [−2.40, +3.75], no detectable effect. On E02's dev words,
  −3.88 [−7.75, −0.39] (secondary).
- **Mechanism:** donor phones change 34% of outputs, but net consonant steering is +1.4% [−0.8, +3.5]; changes are
  mostly vowel or diacritic edits and seed-specific. The letters control is exactly 0.
- **Decision:** the product default and serving are unchanged. Evidence is still missing for a quality claim:
  independently reviewed references and a validated Khmer reader. Report:
  `experiments/E03-assessable-phones/REPORT.md`.
