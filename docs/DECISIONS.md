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

## D181 — E04: English pronunciation as a standalone and joint input (2026-10-27)
- **Arms:** G (letters), P (CMUdict phones only, no letters) and GP (both), on E03's DACS data. Corrected
  schedule: 3,000 updates, 150 warm-up, then cosine. Sanity: all three arms reach 100% on 64 words. 7/8 full
  attempts, 2.0 h.
- **Pilots (dev):** G 43.8, P 38.8, GP 46.2; G ∪ P top-4 +11.8 points over G → useful signal; GP strongest →
  branch B (no three-view run); GP passed the +2-point screen.
- **Eval (historical E03 set, unqualified references):**
  - GP − G +2.20 points [−1.48, +6.01] over 3 paired seeds: no detectable effect;
  - P − G −8.03 [−14.64, −1.35], one seed, exploratory;
  - P collapses on phone-seen homophones (1/14).
- **Post hoc:** G ∪ P top-4 72.7 vs GP 65.8. GP keeps 8/20 of P's unique wins, the same rate as for a second G
  seed's (8/17), so no specific joint-training loss is shown.
- **Clarified 2026-10-27 (E05):**
  - the G ∪ P vs G top-4 gain compares unequal candidate budgets;
  - the retention counts cannot separate pronunciation-specific complementarity from seed variation;
  - the sanity check shows learnability, not the absence of bugs.
- **Decision:** the product default and serving are unchanged. Phones alone are a weaker but complementary
  pathway. The next useful step still needs reviewed references (about 50 ranking failures per model are the
  cheapest lead). Report: `experiments/E04-pronunciation-pathways/REPORT.md`.

## D182 — E05: is G+P candidate pooling more than "more candidates"? (2026-10-27)
- **Setup:** inference only, exploratory on historical data. E04's G_s0/G_s1/G_s2 and P_s0, beam-8 for every
  model. Compared: G4, G8, G+G (3 seed pairs) and G+P (3 combinations sharing P_s0). Macro coverage over families,
  paired family bootstrap.
- **Eval (registered primary):**
  - G+P − G8 +1.63 points [−1.08, +4.51];
  - G+P − G+G +0.75 [−2.25, +3.90];
  - G8 − G4 +5.88 and G+G − G4 +6.76, i.e. most of E04's +8 came from extra slots;
  - rule outcome: **no clear advantage**, uncertainty flag set (one negative per-combination sign).
- **Dev (descriptive):** G+P beats both controls (+4.9, +5.1) but eval does not reproduce it. The checkpoints were
  selected on dev.
- **Cost:** G8 21.9 ms/word with one model resident; G+P 30.2 ms/word (beam-4 each) with two resident, and it
  needs CMUdict.
- **Decision:**
  - keep the simpler letters-only generator;
  - no pronunciation-specific reranker and no further phone/fusion training;
  - a blinded review queue (56 words, 726 candidates) is prepared locally in `data/e05/review/`, not collected.
- **E04 clarification:** the interpretation was corrected (unequal budgets; retention ≠ evidence of seed churn;
  the sanity check is not a bug proof).
- Report: `experiments/E05-candidate-diversity/REPORT.md`.

## D183 — E06: a pretrained reference model (ByT5-small + LoRA) as a possible teacher (2026-10-27)
- **Routes:**
  - Gemini unavailable: no authorised free interface, no request sent;
  - ByT5-small (Apache-2.0, pinned) feasible on the laptop: 1.18 GiB downloads, 3.9 GB RSS, 0.61 h for the
    registered 1,000-update LoRA run.
- **Result:**
  - the adapted ByT5 ignores its input (loss 0.79 vs 0.85 per byte with shuffled inputs) and outputs a near-constant
    Khmer word;
  - historical eval top-1 0.0% vs G 40–43% (−41.3 points [−47.1, −35.7]);
  - this is an inadequate adaptation, not evidence against pretrained transfer (Kirov et al. used full fine-tuning
    of ByT5-base for 50k steps).
- **Gate:** `teacher_advantage_unresolved` (no independent judgments). Distillation skipped, 0 student runs.
- **Fresh pool:** 56 words with no historical family overlap; blinded queue (643 candidates) prepared locally in
  `data/e06/review/`, not collected.
- **Discovery audit:** 200 words, 1.6M local documents; 3 plausible existing targets (multi-word bracketed names),
  20 spurious matches. Not a credible supervision source at scale.
- **Decision:**
  - product default and serving unchanged;
  - reliable reviewed supervision/evaluation remains the blocker.
- Report: `experiments/E06-reference-model/REPORT.md`.

## D184 — E07: why E06's ByT5 collapsed, and one corrected adaptation (2026-10-28)
- **Diagnosis:** no pipeline bug (ten boundary checks pass); an adaptation limitation.
  - E06's per-byte loss was dominated by near-deterministic UTF-8 lead/block bytes.
  - On the informative final byte, the input helped by 0.04 nats, so decoding followed a Khmer letter model
    (ហ្វេន for 89% of eval words).
  - One latent defect (no room for EOS at the 120-byte limit) never triggered; it is fixed in E07.
- **Gate:** a 32-pair memorisation from the base model passed on attempt 1 (32/32 after 100 updates,
  input-dependent, reload identical).
- **Full run** (registered before training): LoRA on attention + FFN, 3,000 updates × batch 32, 4.67 h CPU;
  selected update 1,200 (dev 52.2%).
- **Historical E03 eval:**
  - top-1 47.2% vs G mean 41.3% (+5.8 [+0.05, +11.6]);
  - vs product default 46.2% (+1.0 [−6.1, +8.0]);
  - top-4 oracle at equal beam −1.4 [−6.8, +4.1];
  - one run, mined unreviewed references.
- **Outcome:** a marginal "promising ByT5 research candidate"; phonetic quality unverified.
- **Cost:** about 0.5 s/word and 2 GB RSS vs 15 ms and 0.13 GB for the letters model.
- **Decision:**
  - product default and serving unchanged;
  - no distillation;
  - the wall-time cap was exceeded only by post-training analysis after an idle session (reported).
- Report: `experiments/E07-byt5-diagnosis/REPORT.md`.

## D185 — E08: additional training data for ByT5-small, prepared (2026-10-28)
- **Policy:** frozen before any example was accepted (English reading specified; no translations; no silent
  switch to a name's original-language reading; alternatives allowed; uncertain → quarantine).
- **Recovered:** 1,292 E03-excluded mined pairs, by rules for documented screen weaknesses:
  - coda ស = h, y glides, doubled letters, final ិច, unwritten y;
  - screen-compatible noise-filtered forms.
- **Attested:** 1,217 pairs from the ALT Parallel Corpus (CC BY 4.0): translator spellings found verbatim and then
  screened.
- **Quarantined:** 1,981 uncertain and 114 error rows, kept apart.
- **Agent inspection** (not review): about 80–85% plausible; vowel errors pass the consonant-only screen.
- **Sources not used:** ECCC (signed form needed), Khmerologie (no reuse terms), Google lexicon (auxiliary only).
- **Synthetic:** 200 fresh, unprotected inputs frozen, with 10 Google AI Mode prompts and a strict importer.
  State `awaiting_ai_mode_responses`.
- **Phase-B plan registered, not run:** C vs R (4,764 rows) vs R+S on the E07 recipe, with a word-level sampler and
  bounded 3-seed replication.
- **Unchanged:** no training, serving or product-default change.
- Report: `experiments/E08-data-expansion/REPORT.md`.

## D186 — E08: one exploratory ByT5 run on the expanded data (R+S) (2026-10-28)
- **Run:** fresh LoRA on ByT5-small, E07 recipe, seed 0, on R+S (5,148 rows: control, recovered, ALT, 193 synthetic
  words); 3,000 updates in 5.2 h on CPU; checkpoint chosen on dev (update 3,000, dev greedy 54.2% vs E07 52.2%).
- **Historical eval** (repeatedly examined, mined, unreviewed): top-1 47.5% vs E07 47.2%, **+0.3 [−5.7, +6.5]**;
  CER 0.191 vs 0.189; vs G mean +6.1 [+0.1, +12.2]; vs default +1.3 [−6.1, +8.8].
- **Classification: small or uncertain change** (pre-registered rule); no replication proposed.
- **Limitation:** gains and losses cancel (24 vs 25 items; 56 shared wrong answers); agent inspection of blinded
  disagreements preferred E07 17–7. More data of this quality does not show a measurable gain; label and
  reference quality is the likelier bottleneck. Greedy beats beam-4 by about 5 points for both models (noted only).
- **Unchanged:** serving, product default; no distillation, merge, release or publication; model local only.
- Report: `experiments/E08-data-expansion/REPORT.md` (update 3).

## D187 — E09: ByT5 decoding audit; greedy's advantage was a beam-search configuration defect (2026-10-28)
- **Reproduced exactly:** registered greedy and beam-4 match the saved E07/E08 dev predictions.
- **Cause:** the registered beam call used `early_stopping=True`. With byte targets, junk hypotheses (EOS at once,
  or one letter) finish first and stop the search. Beam top-1 was then often less probable than greedy and
  byte-shorter, often without ៉ / ៊.
  - No implementation defect.
  - Regression test, then the fix `e07.gen(early_stopping=…)`, with the registered defaults kept.
- **Dev selection:** E07 beam-2 canonical, E08 beam-8 canonical; ≈ greedy (+0.4 / +1.2 points).
  - No length-penalty problem, so the follow-up was not run.
- **Historical eval** (repeated, mined, exploratory):
  - vs registered beam-4: E07 +2.3 [−0.9, +5.4] and E08 **+5.9 [+1.4, +10.4]**;
  - vs greedy: −0.5 and +0.9 (noise).
  - E08 − E07 matched: greedy +2.6 [−2.4, +7.5], selected +3.9 [−1.1, +9.0]. The earlier +0.3 was depressed by the
    defect.
- **Practical rule:** use greedy, or canonical beam search when several candidates are needed. Never use
  `early_stopping=True` for byte-level beams.
- **Unchanged:** phonetic quality is not measured; no seed variance; D186's pre-registered verdict stands as
  recorded; no serving, training or release.
- Report: `experiments/E09-byt5-decoding/REPORT.md`.


## D188 — E10: bounded improvement attempt did not pass its gate; E08 packaged as a local ByT5 research preview (2026-10-29)
- **Diagnosis:** E08 was under-trained (dev still rising at 3,000; train loss 0.147), and its exposure was skewed
  towards multi-spelling words (46.6% of draws).
- **Candidate:** one recipe, A (word-level sampling + LR 2e-3), frozen with its gate before training. A_s0: dev 55.4%
  vs E08 54.2% (+1.2 [−3.2, +5.6]), CER 0.141 vs 0.151.
- **Gate:** +3.0 points required, so it **failed**. No replication; E08 `rs_s0` kept as the release candidate.
- **Historical eval** (frozen first, exploratory): A_s0 − E08 +0.6 [−4.5, +5.9], CER reduction −0.5 [−2.5, +1.6].
  0 invalid, empty, repeated or truncated outputs.
- **Confirmation set:** operational diagnostics only. The two models agree on only 28% of 120 fresh words.
- **Independent review:** a blinded 60-word batch is prepared, not collected. One authoritative phonetic policy
  (`docs/PHONETIC_POLICY.md`) with three evidence tracks, never pooled.
- **Product:** `khmer_translit.byt5` and CLI `--backend byt5` (the E02 default is unchanged), extra `[byt5]`,
  version 0.2.0.
  - Greedy single answers and canonical-beam candidates; refusals, never truncation; no silent fallback.
  - Base and adapter hashes are checked; offline use is supported.
  - Public functions unchanged.
- **Bundle:** `dist/khmer-translit-byt5-0.2.0/` (local), with checksums. Fresh-venv install verified; the torch
  stack was borrowed, not installed offline.
- **Publication blocked:** the Google AI Mode terms on the synthetic rows, CC BY-SA, and no code licence (owner).
- **Compute:** training 5.39 h (one attempt), inference about 0.3 h, CPU only. Memory guard relaxed and threads set
  to 12 for later runs at the owner's request.
- Report: `experiments/E10-release-cycle/REPORT.md`.

## D189 — E02 letters model prepared as a local Hugging Face release candidate (2026-10-29)
- **Checkpoint:** `models/e02/letters_s0/model.npz`, sha256 `8901050b…904b`, verified. Lineage: `best.pt`
  `199df48c…` → config `941a8453…` → data manifest `6cdc7c8a…`, every recorded input and output re-hashed.
- **Training data** (verified from `train.tsv`, the data builder and the training code, not from the report):
  - 9,735 optimisation rows, all web-mined: FineWeb-2 9,338 only, Khmer Wikipedia 224 only, both 469;
  - random initialisation, no P2G or auxiliary data;
  - no Chhunneng, LLM or owner labels. The owner's 150 picks were all dropped as held-out-related.
- **Indirect influences:**
  - the mining aligner and filter used a Chhunneng-trained model (exact file not recorded);
  - checkpoint selection used 86 owner-verified dev items, 17 of them LLM-suggested;
  - the training commit `b9c7ea4` was dirty, with no diff recorded.
- **Upstream sources:** the Wikipedia dump matches Wikimedia's sha1 for `kmwiki-20261001`; the FineWeb-2 shard
  matches the Hub LFS hash at revision `af9c1333`.
- **Permissions:** recorded source by source (`release/e02/NOTICES.md`). Unresolved:
  - CC BY-SA and the weights;
  - rights in the underlying web pages (ODC-By covers the database only);
  - no code or model licence chosen. The Hugging Face card has no `license` field.
- **Package:** wheel built from a clean `git archive`. Isolated install (only khmer-translit + numpy 2.2.6 from the
  uv cache) passes 17/17 checks:
  - bit-identical to the recorded E02 outputs (400 items) and to the training-commit code (103 dev inputs);
  - refusals instead of truncation; no torch, transformers or PEFT imports;
  - 0.15 s load, 17 ms per word p50, 65 MiB.
- **Bundle:** `dist/khmer-translit-e02-letters-0.2.0/` (local); `dist/english-phonetic-0.1.0` untouched.
- **Not done:** no upload, merge, tag or serving change.
- Documents: `release/e02/`; evidence: `experiments/E02-release/reports/`.

## D190 — Code licence: PolyForm Noncommercial 1.0.0 (owner, 2026-10-29)
- **Decision:** the owner wants research/study-only terms and chose PolyForm Noncommercial 1.0.0 for the
  `khmer_translit` code.
- **Applied:**
  - `LICENSE`: the official text, plus `Required Notice: Copyright 2026 Richard (khmer-transliterator project)`;
  - `pyproject.toml`: `license = "PolyForm-Noncommercial-1.0.0"`, `license-files`.
- **Scope:** code only. Model weights stay unlicensed until the owner decides how research-only terms fit the
  Wikipedia CC BY-SA training rows. The options are a non-commercial licence on the current weights, accepting the
  share-alike risk, or a retrain without those rows.
- **Pending:** the local bundles predate this. They need a rebuild before any use.

## D191 — E02 weights licence: CC BY-NC-SA 4.0 on the current letters_s0 (owner, 2026-10-29)
- **Decision:** the owner wants research/study-only terms and chose **CC BY-NC-SA 4.0** for the existing weights
  (`8901050b…`), with no retrain.
- **Options presented:** retrain without the 224 Wikipedia-only rows or all 693 (est. effect within seed noise; it
  could only be checked on 86 dev words, since the final test is locked).
- **Accepted risk:** if Wikipedia's ShareAlike reaches the weights, CC BY-SA would be required, and NC-SA is not
  BY-SA-compatible. Also, FineWeb-2 may itself contain share-alike pages, so a retrain would not have removed all
  such content.
- **Applied:**
  - `license: cc-by-nc-sa-4.0` in the E02 Hugging Face card;
  - `release/e02/LICENSE-weights.md`, shipped as `LICENSE` in the Hugging Face folder;
  - the code stays PolyForm Noncommercial 1.0.0 (D190).
- **Still needed:** publication approval. The open questions on page rights, Chhunneng-influenced mining and
  LLM-suggested selection items are recorded in `release/e02/NOTICES.md`.

## D192 — en-km-phonetic-transliterator uploaded to Hugging Face, private (owner, 2026-10-29)
- **Upload:** the owner approved it ("let do on hf first", private first). Repository
  `ric912/en-km-phonetic-transliterator` (private), commit `b7f31150`: public card, `model.npz` `8901050b…`, weights
  `LICENSE` (CC BY-NC-SA 4.0) and the `khmer-translit` 0.2.0 wheel (PolyForm Noncommercial).
- **Verified on the Hub:** metadata parsed; a fresh download installs offline and reproduces the quick start.
- **Not done:** the repository is not public yet (the owner switches it after review); nothing on PyPI (later); no
  serving change.

## D193 — en-km-phonetic-transliterator made public (owner, 2026-10-29)
- `ric912/en-km-phonetic-transliterator` switched to public at the owner's instruction, after the owner reviewed the
  private page. Tag `v0.2.0` marks this release.
- Anonymous access was checked (model page, card and `model.npz` reachable without login).
- Weights CC BY-NC-SA 4.0, code PolyForm Noncommercial 1.0.0. Serving is unchanged and nothing is on PyPI.

## D194 — E11: ByT5-small on real-data pool R (no AI Mode labels) is an uncertain alternative to E07 (2026-10-29)
- **Run:** E07's recipe unchanged on E08's arm R (4,764 rows: E03 control 2,255 + recovered 1,292 + ALT 1,217;
  0 AI Mode rows, verified from the files), fresh adapter on the pinned base, seed 0, 3,000 updates, 4.6 h CPU.
  Selected update 3,000 (dev 53.0%, CER 0.157); still rising at the end.
- **Result (greedy, historical references, frozen before scoring):** eval E11 − E07 +2.5 [−2.7, +7.7], CER
  reduction +1.7 [−0.7, +4.1] → registered class **uncertain alternative**. E11 − E08 −0.1 [−4.7, +4.6]: the 384
  AI Mode rows added nothing measurable. Fewer missing endings (eval 4 vs 8); vowel-only mismatches unchanged
  (≈ 43% of errors).
- **Phonetic quality:** unresolved; blinded 60-word batch in `data/e11/review/` (local), no judgments.
- **Permissions:** AI Mode blocker gone; CC BY-SA and page-rights questions on mined pairs remain; adapter unlicensed.
- **Next:** review judgments on vowel choices before more training; second seed recommended, not launched.
  Nothing published, packaged or served. Report: `experiments/E11-real-data-byt5/REPORT.md`.
- **Update (owner review, same day):** the owner judged 40 of the 60 blinded dev words (single judge, not
  independent; transcribed from chat under owner-confirmed rules). Greedy output acceptable: representative 28 words
  E07 75% / E11 82% / E08 89%; disagreement 12 words 42% / 50% / 25%; no detectable model difference. Against the
  mined references, 20 of 38 "wrong" spellings were acceptable and 7 of 29 "right" ones were not, so exact
  agreement with these references is a weak quality measure. Next: a reviewed reference set before further training.
- **Update (owner review complete):** all 60 words judged. Acceptable greedy output: E07 29/60 (48%), E11 39/60
  (65%), E08 38/60 (63%); representative 60 / 72.5 / 75%, disagreement 25 / 50 / 40%. Paired by word, E11 vs E07
  14 vs 4 (exact sign test p = 0.031, post hoc), E11 vs E08 8 vs 7. 28 of 72 reference "misses" were acceptable and
  8 of 37 reference matches were not. The registered class (reference agreement) stays **uncertain alternative**;
  the owner's judgment favours E11 over E07 (one non-independent judge, dev words). Next: reviewed references
  (acceptable/not on model outputs) as the scoring basis before further training.

## D195 — E12: longer schedule (6,000 updates) for ByT5-small gives no detectable gain over E11 (2026-10-30)
- **Run:** E11's recipe with linear decay over 6,000 updates (fresh adapter on the pinned base, seed 0, arm R only,
  192,000 draws = 40.3 epochs), one attempt, 9.08 h CPU, no interruption. Registered before training (1f058f5).
- **Selection (registered rule, dev CER first):** update 6,000 (dev CER 0.154, exact 55.0%; sha256 `566a673e…`),
  frozen with all predictions before scoring (11dc318).
- **Historical diagnostics (mined references, exploratory):** E12 − E11 eval +0.7 [−4.1, +5.4], dev +2.0
  [−2.4, +6.4]; CER differences ≈ 0.3 points. No detectable change; item flips are mostly churn.
- **Primary test (owner, blinded, 120 fresh families frozen before training, all 185 spellings judged explicitly):**
  confirmed acceptable E12 77/120 (64.2%) = E11 77/120 (64.2%), difference 0.0 [−8.3, +8.3]; 14 vs 14 discordant
  words, exact p = 1.0; sensitivity +0.8 / 0.0. **No detectable difference** (not shown equivalent; ±8 points
  remain possible). Exploratory: irregular-spelling words weakest for both (E12 19/40, E11 14/40); names E11 36 vs
  E12 31.
- **Decision:** E12 is not adopted over E11; longer optimisation of this recipe on R is not the lever. No
  replication. Next quality work should target the data (irregular spellings, reviewed references), not schedule
  length.
- Nothing published, packaged, served, merged or tagged; no second seed. Report:
  `experiments/E12-longer-schedule/REPORT.md`.

## D196 — E13: pronunciation-guided ByT5-small (word + CMUdict reading, owner-accepted difficult words, R replay) gives no detectable gain over E11 (2026-10-31)
- **Change (one combination, no ablations):** input `word|ARPAbet` (first CMUdict reading; `<no_pron>` if none);
  init from E11; 90% R replay / 10% owner-accepted spellings (42 train families, 23 irregular); 10% reading masked;
  lr 3e-4, 2,000 updates. Amendment before training: compact source + 2×16 micro-batches (memory).
- **Run:** stopped on the 6 h cap at update 1,969; the registered rule (collection-dev CER) picked update 800
  (CER 0.085 vs E11 0.125; checkpoint `e32cd817…`), frozen before test generation.
- **Historical (exploratory):** dev +1.6 [−2.4, +5.6], eval −1.3 [−4.8, +2.3].
- **Mechanism:** removing the reading changes 7–10% of outputs, a wrong reading 8–15%. The reading is used only weakly,
  in line with E02–E05.
- **Primary test (owner, blinded, 120 fresh families):** E13 90/120 = E11 90/120 (75.0%), 0.0 [−7.5, +7.5],
  11 vs 11 discordant, p = 1.0. Groups: ordinary 32 vs 34, irregular 21 vs 20, names 37 vs 36 (exploratory).
- **Decision:** uncertain change; not adopted; no second seed. The cause cannot be separated from the combination.
  Nothing published, packaged, served, merged or tagged. Report: `experiments/E13-pronunciation-guided/REPORT.md`.

## D197 — E11 ByT5-small adapter prepared for a possible Hugging Face release; local bundle verified, publication pending owner decisions (2026-10-31)
- **Candidate:** E11 `r_s0`, update 3,000 (checkpoint `286292a4…`), arm R only (4,764 rows, no Google AI Mode
  labels), fresh LoRA r16/α32 on attention + FFN of `google/byt5-small` @ `68377bdc…`. Identity checks 12/12; gap:
  training ran from a dirty tree (pinned by the registered source hash).
- **Export:** `adapter.safetensors` (256 tensors, 6,258,688 parameters, bit-identical to the checkpoint; sha256
  `698a23ae…`). Project format, not PEFT; loads only through `khmer_translit.byt5`; the base stays separate.
- **Verification:** greedy parity 0 mismatches on 276 dev + 245 eval words, top-4 parity 0/276; refusal statuses,
  batch alignment, error exits, deterministic reload, offline. Isolated install (fresh uv venv, CPU torch 2.14.1,
  no `.pth` bridge, no CUDA packages): 19/19 checks, dev parity 0/276. Network off via a Python socket guard
  because AppArmor refuses unprivileged network namespaces here. Laptop CPU, 6 threads: load 5.5–7 s, greedy
  0.55 s/word or 14 words/s batched, peak 2.0 GB.
- **Bundle:** `dist/khmer-translit-byt5-e11-0.2.0/` (archive sha256 `9c70dace…`, local; the E10 and E02 bundles are
  untouched), with an HF folder for `ric912/en-km-phonetic-byt5-lora` (upload command prepared, not run).
- **Coverage diagnostic (development only):** 23 irregular dev words frozen before generation; 43 judgments reused
  exactly, 50 new spellings judged by the owner (blinded; all 93 candidates judged). Greedy acceptable 8/23, beam
  top-1 10/23, at least one acceptable among 4 beam candidates 15/23 (oracle ceiling, not accuracy); 7 words
  recoverable by a perfect reranker, 8 with no acceptable candidate. No ranker built.
- **Evaluation write-up:** mined-reference results (exploratory), the owner's dev review and the two 120-family
  tests (64.2% and 75.0%, two samples of one design, not an improvement) kept separate; no pooled headline.
- **Licences:** base Apache-2.0; code PolyForm Noncommercial (repository LICENSE); data notices for FineWeb-2
  (ODC-By + Common Crawl terms), Wikipedia (CC BY-SA 4.0) and ALT (CC BY 4.0); the E02 weights licence is not
  copied. **Owner decisions pending:** the adapter licence, the data questions, the repo name/private start, and
  whether the wheel is uploaded. Nothing published, merged, tagged or served.

## D198 — E11 adapter weights licence: CC BY-NC-SA 4.0, same as E02 (owner, 2026-10-31)
- **Decision:** the owner chose the E02 terms (D191) for the E11 ByT5 adapter (`adapter.safetensors`, `698a23ae…`):
  **CC BY-NC-SA 4.0**. The code stays PolyForm Noncommercial 1.0.0; the base `google/byt5-small` stays Apache 2.0
  and is not redistributed.
- **Accepted risk (as for E02):** if Wikipedia's share-alike reaches the weights, CC BY-SA would be required, and
  NC-SA is not BY-SA-compatible; the Common Crawl page-rights and indirect Chhunneng questions are recorded in
  `release/byt5-e11/NOTICES.md`.
- **Applied:** `license: cc-by-nc-sa-4.0` in the card, `huggingface/LICENSE` (`release/byt5-e11/LICENSE-weights.md`),
  bundle rebuilt. Still open: the repository name and private start, and whether the wheel is uploaded. Nothing
  uploaded.
