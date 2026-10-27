# E02 — English words to Khmer phonetic spelling: letters vs + English phones vs + Khmer P2G transfer

**Status:** registered 2026-10-27, before any full training. The registration commit is the first commit that
contains this file; its hash is in `state/english-phonetic/status.json`. Executed autonomously from the owner's
handoff (`handoff/english-phonetic/`).

**Files:**
- config: `config.yaml`;
- code: `scripts/english_phonetic_{data,train,run,eval}.py` and `src/khmer_translit/{phonetic_source,
  reference_data,evaluation,english}.py`;
- audit: `reports/audit.md`, `resources.json`, `data_policy.json`;
- results: `RESULTS.md`, `results.json`.

This file is the registered protocol (the handoff's `templates/PROTOCOL.md`, with every value filled in).

## Question and scope
1. Does English phone input help a small clean English → Khmer phonetic model?
2. Does Khmer phones → spelling (P2G) pretraining plus replay help the phone-input recipe?

Out of scope: translation, romanised Khmer chat, audio and TTS.

## Fixed inputs
- **Code:** branch `english-phonetic`, from `main` @ `b3f2096`. Each run's provenance records the commit and the
  dirty flag.
- **Data:** `data/e02/` from `scripts/english_phonetic_data.py build`. Hashes are in `data/e02/manifest.json`
  (copy: `reports/data_build.json`). The runner freezes the hashes of the config and of all data files when it
  starts, and stops if any changes.
- **Supervised:** 9,735 train rows and 296 training-validation rows of web-mined pairs (tier proxy). Filters:
  CMUdict hit, not Khmer-like, D135 noise filter, leak filter (`data_policy.json`).
- **P2G:** 61,809 Google-lexicon Khmer pairs (58,716 train / 3,093 dev, split by connected component). Accepted
  dev/final/proxy spellings are removed, so every final item is unseen as a P2G target.
- **Vocabularies:**
  - source: 157 tokens (`phonetic_source.EN_TOKENS` plus 55 Khmer phones), encoder `en-phonetic-src-v1`;
  - target: 78 Khmer characters, taken from the training and P2G targets only.
- **Pronunciation:** CMUdict sha256 `81917843…3d22`. Default reading: the first listed. A miss is encoded
  `<no_pron>`; there is no G2P. Accent: US (CMUdict).
- **Normalisation:**
  - English: NFKC, lower case, accents removed, apostrophes and hyphens dropped, `[a-z]{1,32}`;
  - Khmer: NFC.
- **Lexical families:** shared simple suffix stems, joined transitively (`reference_data.group_families`).
- **E01 encoding conflict:** settled. E01 had letters-only input (`reports/audit.md` §2).

## Benchmark (`en-phonetic-v1`, frozen in `data/e02/references.jsonl`; hash in the manifest)

| Track | Items / families | What | Use |
|---|---|---|---|
| dev | 103 / 101 (86 primary-eligible) | owner-verified list words | checkpoint and arm selection only |
| **final → primary** | 165 / 162 | owner-verified list words without ambiguity / translation flags | the primary measure, opened once |
| final → flagged | 26 | `pronunciation_ambiguous` or `possible_translation` | diagnostic |
| final_proxy | 400 / 375 | LLM spellings, owner spot check only | exploratory, never pooled |
| v1 English panels | 150 + 41 + 125 | historical | after selection, old rule, never pooled |

- **Qualification: unqualified.** No item has documented independent acceptance under this phonetic policy. The
  final items are owner-verified for the speech front-end (tier `attested`), and 94 of them are
  candidate-anchored picks among model E's outputs. The primary metric is therefore called
  **reference agreement**. The evidence status will be `inconclusive` whatever the numbers are; this is fixed
  now.
- **Measurement controls:** `evaluation.measurement_controls`. Every accepted form and its NFD form must pass, a
  corrupted form must fail, and missing or unknown ids must be refused. These are checked in the tests and again
  in the final report.

## Arms and training (`config.yaml`)
- **The arms:**
  - `letters`: `<task:en>` + letters;
  - `phones`: + `<sep>` + CMUdict phones; 10% of training examples get `<no_pron>` (private RNG
    `"{seed}:phone_dropout:{epoch}"`, drawn in every arm);
  - `phones_p2g`: `phones`, initialised from P2G pretraining. Pretraining runs at most 10 epochs or 2 hours,
    with its own warm-up + cosine schedule, best by P2G dev loss, one job per used seed. After every 8
    supervised updates comes one P2G replay batch (128 rows, RNG `"{seed}:replay:{epoch}"`, the same loss), at
    the current learning rate, without advancing the supervised schedule.
- **Shared recipe:**
  - CPU FP32; AdamW, lr 1e-3, weight decay 0.01;
  - 500 warm-up steps then cosine over the supervised steps;
  - label smoothing 0.1, batch 128, gradient clip 1.0, 40 epochs, 6 threads.
- **Pairing:** `torch.manual_seed(seed)` before building the model. The runner checks that the initial-state
  hashes are equal across arms, and the batch-order hashes too (order RNG `"{seed}:order:{epoch}"`).
- **No E01 auxiliary decoder.** Model capacity is the same in every arm.
- **Checkpoint per run:** greedy top-1 macro reference agreement on the 86 primary-eligible dev items. Ties go to
  the lower training-validation loss, then the earlier epoch. Validation is greedy and the final scoring is
  beam-4: this difference is declared.
- **Budget:**
  - at most 7 full supervised attempts and 3 P2G jobs;
  - at most 96 hours of summed job time;
  - smoke runs: 8 of the 10 allowed were used before registration.

## Selection and confirmation
- **Seed-0 pilots of all three arms.** The arm with the best dev score wins; ties prefer `letters`, then
  `phones`, then `phones_p2g`. Final and v1 predictions are not opened for this.
- **If a treatment wins,** that treatment and `letters` run at seeds 1 and 2 (for `phones_p2g`, with its own
  P2G job per seed).
- **If letters wins:** no confirmation, and no confirmatory claim.
- **Limits of the comparison:**
  - B vs C is a one-seed exploratory comparison;
  - a three-seed C-vs-letters gain would not show that C beats B;
  - C mixes pretraining and replay.

## Measurement
- **Primary:**
  - raw beam-4 top-1, multi-reference exact match (NFC);
  - **macro over lexical families**: each family's mean over its items, then the unweighted mean over families;
  - plain item accuracy alongside.
- **Inputs:** the deployed default provider only. Reference phones are never fed to a model. A wrong provider
  reading counts as an error.
- **Supplied-pronunciation track:** not run. The intended reading of every item *is* the registered default, so
  the track would repeat the primary one exactly.
- **Diagnostics:**
  - top-4 coverage and minimum CER;
  - a fixed re-ranker (`translit.pick` towards Google-lexicon words, the same for every arm);
  - strata: dictionary hit/miss, candidate-anchored or not, ≤ 4 or ≥ 9 letters;
  - the flagged and proxy tracks;
  - v1 English panels.

  No automatic phonetic-distance score is used as a gate.
- **Power** (`reports/power.json`, computed before training on a declared discordance grid): with 162 primary
  families, the power to show a +5-point gain is 0.29 at a discordance of 0.20 (E01's pick-test rate was 0.19),
  and 0.50 at 0.10. About 628 families would be needed at 0.20. On the proxy track (375 families) the power is
  0.57 at 0.20. **Sensitivity is inadequate; an inconclusive interval is expected.**
- **Paired comparison:** 10,000 paired family bootstrap resamples, seed 20261027. Each family's difference is
  averaged over the three seeds. The interval is conditional on these trained seeds, and every seed is reported.

## Decision (fixed now)
- **Practical threshold:** +5 percentage points.
- **Evidence fields:**
  - `positive_effect_supported`: lower bound > 0;
  - `meaningful_gain_ruled_out_conditional`: upper bound < 5;
  - `practically_equivalent_conditional`: the whole interval inside (−5, +5).
- **Summary:** for an unqualified benchmark, `inconclusive` always.
- **Adoption, all of:**
  - a qualified benchmark;
  - a mean gain ≥ 5;
  - positive in at least 2 of 3 seeds;
  - lower bound > 0;
  - controls pass;
  - runtime guards pass (Task 5: cold load ≤ 30 s, warm batch-1 p95 ≤ 1 s/word, RSS ≤ 1.5 GiB, and p95 ≤ 2×
    the letters baseline).

  The first condition fails by registration, so **no treatment can be promoted**.
- **Product default:** the letters model (seed 0, the dev-selected checkpoint). The other weights stay as
  research artifacts.
- No threshold changes, reference edits or extra variants after the test.

## Product fallback
- Treatment passes → promote it. Otherwise → the letters baseline.
- An unqualified benchmark → a research preview with an inconclusive evidence status.
- No eligible data → tooling only, without weights.
- A Chhunneng-derived model is never the fallback.
