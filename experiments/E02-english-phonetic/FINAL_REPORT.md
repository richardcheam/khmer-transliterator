# English phonetic transliterator — final report

## Outcome

- **Product status:** `runnable_research_preview`. Installable wheel, `khmer-translit` CLI and Python API, a trained
  default checkpoint, all offline on CPU. The research label comes from the unqualified benchmark.
- **Benchmark qualification:** `unqualified`. No primary item has independent acceptance under the phonetic policy
  (`BENCHMARK_CARD.md`).
- **Experiment status:** `complete_paired`. Seed-0 pilots of all three arms, then the dev winner (phones) and
  letters at seeds 1 and 2: 7/7 supervised runs and 1/3 P2G jobs, none failed or superseded.
- **Evidence status:** `inconclusive`, as registered for an unqualified benchmark. Phones vs letters: +1.65 points,
  95% paired family-bootstrap interval [−1.65, +4.94]. Effect fields:
  - `positive_effect_supported` = false;
  - `meaningful_gain_ruled_out_conditional` = true;
  - `practically_equivalent_conditional` = true.
- **Adoption/default model:** `letters_baseline_retained`; default `letters_s0`
  (`models/e02/letters_s0/model.npz`, sha256 `8901050b1093cf90f60523c7941d6c12dbd8571d71698e37abcf874ba437904b`).
- **Redistribution status:** `restricted_artifacts_local_only`. The training pairs are CC BY-SA / ODC-By, and the
  references are the owner's material. No code licence has been chosen yet.
- **Branch and exact commit:** branch `english-phonetic`. The bundle records the commit it was built from in
  `dist/english-phonetic-0.1.0.verify.json` (`git_commit`, `git_dirty`). The registration commit was `b9c7ea4`.
- **Artifact directory/ZIP and SHA256:** `dist/english-phonetic-0.1.0/` and `dist/english-phonetic-0.1.0.zip`. The
  ZIP's sha256 and its verification results are in `dist/english-phonetic-0.1.0.verify.json`; per-file sums are in
  the bundle's `SHA256SUMS`.

## Reproduce and use

```bash
# from the bundle folder: clean install, resource check, predictions (offline after install)
uv venv -p 3.10 .venv-e02 && VIRTUAL_ENV=.venv-e02 uv pip install wheel/khmer_translit-0.1.0-py3-none-any.whl
.venv-e02/bin/khmer-translit prepare --manifest prepare_manifest.json
.venv-e02/bin/khmer-translit predict --model checkpoints/letters_s0/model.npz --word coffee --top-k 3 --json
printf 'computer\nbanana\nJohnson\n' | .venv-e02/bin/khmer-translit predict --model checkpoints/letters_s0/model.npz --input - --jsonl
```
- **Offline requirements:** NumPy ≥ 2. Phone-input research models also need CMUdict (`--cmudict
  resources/cmudict/cmudict.dict` or `KHMER_TRANSLIT_CMUDICT`); the default letters model needs nothing else.
- **Executed example** (2026-10-27, clean venv outside the checkout; full transcript in the bundle README):
  - `coffee` → ខូហ្វ៊ី (log score −1.62), then កូហ្វ៊ី (−1.67) and កូហ្វី (−2.61);
  - `computer` → កំពូទែរ, `banana` → បាណាណា, `Johnson` → ចនសុន;
  - `hello world` → `[unsupported_multiword]`, exit status 1.

**Rerun** (repository, branch `english-phonetic`, `CUDA_VISIBLE_DEVICES=`):
```bash
.venv/bin/python scripts/english_phonetic_data.py check                                  # data hashes
.venv/bin/python scripts/english_phonetic_run.py --config experiments/E02-english-phonetic/config.yaml --resume
.venv/bin/python scripts/english_phonetic_eval.py report                                 # results.json from saved predictions
.venv/bin/python scripts/english_phonetic_bench.py models/e02/letters_s0/model.npz models/e02/phones_s0/model.npz models/e02/phones_p2g_s0/model.npz
.venv/bin/python scripts/english_phonetic_wheelcheck.py --letters models/e02/letters_s0 --phones models/e02/phones_s0
.venv/bin/python scripts/english_phonetic_bundle.py --force
```
- The queue is finished, so `--resume` only reports that it is done.
- The final evaluation (`english_phonetic_eval.py final`) is locked by `data/e02/results/final.lock` and refuses to
  run again. That is deliberate: the final references are opened once.
- A full retrain from scratch needs a new experiment folder.

## Data and reference quality

- **Supervised training data:** 67,337 web-mined bracket pairs (Khmer Wikipedia 3,532; FineWeb-2 63,805), of which
  10,031 were kept (9,735 train, 296 validation). Rows dropped:

  | Reason | Rows |
  |---|---|
  | noise filter (D135) | 23,326 |
  | no CMUdict reading | 17,641 |
  | romanised Khmer, not English | 13,433 |
  | related to a held-out word | 940 |
  | target equal to a held-out spelling | 22 |
  | bad Khmer or length | 29 |

  - All 150 owner pick examples are benchmark words, so none was used for training.
  - No Chhunneng data, LLM output or owner-list entry was used for training.
  - Permission evidence: `resources.json`, `SOURCE_VERIFICATION.md`.
- **P2G data:** 61,809 Google-lexicon pairs (CC BY 4.0; 58,716 train / 3,093 dev, split by connected component).
  The 284 pairs whose spelling is a held-out accepted form were removed. There are 3,872 homophone groups, all of
  whose spellings are kept as targets.
- **References (`en-phonetic-v1`):**
  - Source: owner-verified single words from the speech front-end list.
  - Split: dev 103 / final 191 by lexical-family hash, plus a proxy track of 400 LLM spellings.
  - Excluded by policy: `attested`/`attested_web` rows and drafts, which mix conventional and translated forms.
- **Conflict policy:** the 26 items flagged `pronunciation_ambiguous` (24) or `possible_translation` (2) are scored
  separately, not in the primary track.
- **Accepted-alternative coverage:** exactly one accepted spelling per final item. This is the main reason exact
  match under-counts.
- **Pronunciation default:** the first CMUdict reading (US). A word missing from CMUdict gets `<no_pron>`; there is
  no G2P fallback. The final track has 23 dictionary misses (22 in primary).
- **Leakage:** no family is in two splits; no training word is related to a held-out word; no training or P2G
  target equals a held-out accepted spelling. Every final target is unseen.
- **Independent review coverage:** 0%. 94 of the 191 final items were picked by the owner among historical model
  E's candidates (`candidate_anchored`).
- **What the metric can and cannot show:** exact agreement with the owner's single spelling. It can show which
  model agrees more often with that list. It cannot show phonetic correctness: a correct-sounding variant counts
  as wrong, and an agreeing spelling may follow the owner's front-end convention rather than the English sound.

## Experimental evidence

| Arm | Seed | Best epoch | Raw primary | Fixed-reranker diagnostic | Time | Peak RSS (inference) |
| --- | --- | --- | --- | --- | --- | --- |
| letters | 0 | 27 | 16.7 | 18.5 | 0.25 h | 65 MiB |
| letters | 1 | 18 | 15.1 | 18.2 | 0.26 h | — |
| letters | 2 | 23 | 16.0 | 19.1 | 0.25 h | — |
| phones | 0 | 27 | 17.3 | 22.8 | 0.35 h | 137 MiB |
| phones | 1 | 19 | 16.7 | 21.9 | 0.36 h | — |
| phones | 2 | 26 | 18.8 | 23.8 | 0.36 h | — |
| phones_p2g | 0 | 15 | 16.7 | 21.9 | 0.40 h + 0.54 h P2G | 138 MiB |

Notes on the table:
- Raw primary = macro over 162 families, in percent.
- Peak RSS was measured only for the three seed-0 models, in fresh processes.
- Training memory was not instrumented; one job ran at a time on the 7.6 GB laptop without issue.

**Selection** (dev only, 86 items, greedy): letters 20.9, phones 23.3, phones_p2g 22.1, so phones went to
confirmation. Selection used only the dev references; the final references were opened once, after training.

**Confirmatory comparison, phones vs letters:**
- Paired gains: +0.62, +1.54 and +2.78 points (positive in 3/3 seeds); mean **+1.65**.
- 95% interval: **[−1.65, +4.94]**, from 10,000 family-bootstrap resamples (seed 20261027) of per-family differences
  averaged over seeds. The interval is conditional on these six models.
- Seed spread of the raw primary: letters 15.95 ± 0.78, phones 17.59 ± 1.11 (mean ± sd).

**Exploratory comparisons:**
- Proxy track: +1.42 [−0.62, +3.56].
- One-seed pilots:
  - phones vs letters: +0.62 [−3.70, +5.56];
  - phones_p2g vs letters: 0.00 [−5.56, +5.56];
  - phones_p2g vs phones: −0.62 [−6.17, +4.94].
- The B-vs-C (phones vs phones_p2g) comparison has one seed. Arm C mixes pretraining and replay. Even a confirmed C
  gain would not have shown C better than B.

**Subgroups** (primary, mean over seeds, letters → phones):
- dictionary hit (143): 15.6 → 17.3;
- dictionary miss (22): 18.2 → 19.7;
- candidate-anchored: 16.9 → 18.5;
- not anchored: 14.9 → 16.5.

The difference is similar in both anchoring strata, so anchoring on model E does not explain the phone gain.

**Diagnostics** (not decision measures):
- The fixed re-ranker raises phones more than letters: 22.84 vs 18.62.
- Top-4 coverage: 42.7 vs 41.1.

**Power** (registered before training): 0.29 for a true +5 points at a discordance of 0.20 with 162 families; about
628 families would be needed. The sensitivity was inadequate, as declared.

**Cost and performance:**
- P2G pretraining (0.54 h) plus replay (+0.05 h of supervised time) bought no measurable gain in its pilot.
- Total CPU training time: 2.81 h.

**Adoption** is a separate question from the evidence fields:
- Conditions that failed:
  - the benchmark is not qualified;
  - the mean gain is below 5;
  - the interval's lower bound is not above 0.
- Conditions that passed:
  - positive in 2 of 3 seeds;
  - controls;
  - runtime guards.
- None of this shows the phones have zero effect. The data allow anything from about −1.6 to +4.9 points.

## Product and verification

- **Installed wheel** (`reports/wheel_check.json`, all 11 checks pass):
  - set-up: a fresh venv in a temporary folder outside the checkout, with no `PYTHONPATH` and no CMUdict variable;
  - the letters model works by word (JSON) and in batches (JSONL, ids in input order);
  - refusals: empty, multi-word and Khmer-script inputs;
  - a phone model without CMUdict is refused with an actionable message (exit 2); with CMUdict it reports the
    pronunciation and its ambiguity;
  - a supplied pronunciation is used, and unparsed phones are refused;
  - `inspect`;
  - a corrupt or missing checkpoint gives exit 2;
  - repeated runs are identical;
  - the API runs with sockets disabled and loads no torch, wandb or speech-repo modules.
- **Export parity:** all 7 runs pass. On 12 fixtures each, NumPy and torch agree on source ids and greedy outputs,
  with a largest log-probability gap ≤ 5e-6 (`models/e02/*/parity.json`).
- **Tests:**
  - `CUDA_VISIBLE_DEVICES= WANDB_MODE=disabled .venv/bin/python -m unittest discover -s tests`: exit 0, 67 tests,
    0 failures, 0 skips. The torch- and CMUdict-dependent tests ran, because the data is present.
  - Speech repo (`../pocket-tts-khmer`, `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests`): 42
    tests OK.
  - Unit tests check the code, not the scientific claim.
- **Runtime** (`reports/runtime.json`): 103 dev words, 3 passes, 6 threads, 12-CPU laptop, a fresh process per
  model.

  | Model | Cold load | Batch-1 p50 / p95 | p95 dict hit / miss | Batch-16 per word | Peak RSS |
  |---|---|---|---|---|---|
  | letters_s0 | 0.09 s | 18 / 32 ms | 32 / 35 ms | 12 ms | 65 MiB |
  | phones_s0 | 1.47 s (CMUdict parse) | 21 / 36 ms | 35 / 37 ms | 15 ms | 137 MiB |

  Pronunciation lookup costs 0.1 ms per word. All guards pass; the phone p95 is 1.14× the letters p95.
- **Malformed and ambiguous input:** explicit statuses (`empty`, `unsupported_multiword`, `unsupported_characters`,
  `too_long`, `unknown_phone`). The default reading of an ambiguous word is reported along with its alternatives
  (`ambiguous: true`).
- **Measured limitation:** the phone model mostly follows the letters.
  - For *record*, a different supplied reading changes only the second candidate.
  - *live* comes out as ឡៃវ៍ although CMUdict has only L IH1 V.
- **Hashes:**
  - checkpoints: `prepare_manifest.json` and `run_ledger.json`;
  - data: `data/e02/manifest.json` (references `11d89a0a…`, dev `58541efd…`) and `datasets/manifest.json`
    (re-pinned, 221 files);
  - CMUdict: `81917843…3d22`.

## Preservation and boundaries

- **E01, benchmark v1 and old models:** unchanged.
  - No tracked change under `experiments/E01-unified`, `benchmarks` or `data/benchmark`.
  - No file in `models/model*`, `models/unified` changed after registration; model E's sha256 still matches its
    card.
  - The previous data manifest matches every old file, except `lexicon/ATTRIBUTION.md`, whose usage sentence was
    corrected.
- **Reserve:** `data/e2/reserve.json` is unchanged (same pinned hash).
  - The E02 code reads only its Latin words, to exclude them from training and evaluation (`read_reserve_latin` in
    `english_phonetic_data.py`).
  - No reserve prediction was generated and the reserve was not scored.
  - This states what this execution's code does; it is not a claim about unobservable history.
- **Spending and compute:** no paid request, GPU or cloud.
  - Run metrics were mirrored to the owner's existing W&B project, per the research protocol.
  - No new package or model download: NumPy came from the local uv cache.
- **Budget used:**
  - 7/7 supervised runs and 1/3 P2G jobs;
  - 2.81 of 96 hours;
  - 8/10 smoke jobs (before registration).
- **Not done, by design:** no PyPI upload, merge into `main`, release tag or serving change; the speech repo is
  untouched.
- **Restricted local artifacts:** `models/e02/` (weights, torch checkpoints, W&B folders), `data/e02/` (training
  pairs, references, predictions) and `dist/` are git-ignored and stay on this machine.

## Sources and unresolved blockers

- **Documents:**
  - `SOURCE_VERIFICATION.md`: each source, what it does and does not support, and whether it was used. The papers
    were not re-fetched; their entries rely on the handoff's reading.
  - Model card: `model_cards/e02_english_phonetic.md`.
  - Benchmark card: `BENCHMARK_CARD.md`.
- **Hypotheses, not sourced findings:**
  - that phone input or Khmer P2G transfer helps this model;
  - the three-arm design, the budgets and the runtime guards.
- **No incomplete work** remains in this project.
- **Prerequisites for a stronger claim (not for this project):**
  - an independent reviewer who accepts or rejects each primary item under the written phonetic policy and adds
    the alternatives they accept, which is the only way to qualify the benchmark;
  - a decision on the share-alike status of weights trained on Wikipedia pairs;
  - a code licence.
