# E11: ByT5-small on the real-data pool R (no AI Mode labels): plan (written before training)

**Question.** Does E07's ByT5-small recipe, trained on E08's expanded real-data arm R instead of the E03 control
data alone, agree more with the historical references than E07, without any Google AI Mode labels?

This is a forward step: one data change against E07, one run. It is not an ablation.

## Data (frozen; verified by `scripts/e11.py verify` → `reports/data_checks.json`)
- **Arm:** E08's arm R, `data/e08/training/arm_R.tsv`, sha256 `d00706cf…ec88`, unchanged:
  - 4,764 optimisation rows: E03 control 2,255, recovered mined 1,292, ALT-attested 1,217;
  - plus E03's 93 train-val rows.
- **Checks, from the files themselves:**
  - exact and NFC deduplication;
  - control rows equal to E07's training rows;
  - no `google_ai_mode_synthetic` rows, and R ⊂ R+S with R+S − R consisting only of AI Mode rows;
  - pairs that also appear as AI Mode spellings or in E08 quarantine files, with their accepting sources;
  - dev/eval, protected-family and train-val-family exclusions;
  - byte limits.
- **Provenance across duplicate sources:** the arm keeps one provenance per pair. All sources per pair are
  rebuilt into `data/e11/pair_sources.json` (local).
- **Not done:** no relabelling or new filtering. Uncertain vowel quality in the mined and ALT rows is a known
  limitation.

## Run (one authorised attempt)
- **Recipe:** E07's `full_recipe.json` unchanged (hash-checked):
  - LoRA r16 / α32 / dropout 0.05 on attention + FFN, base dropout 0.1;
  - AdamW lr 1e-3, 100 warm-up, linear decay to 0;
  - batch 32 (no accumulation), 3,000 updates or 6 h;
  - dev checks at 600 / 1,200 / 1,800 / 2,400 / 3,000.
- **Initialisation:** the pinned `google/byt5-small` base plus a fresh adapter, seed 0. Not from E08 or E10, whose
  adapters saw AI Mode labels.
- **Sampling:** row-level, as in E07: each epoch is a seeded permutation of the rows, with no source weighting.
  Exposure and source proportions are recorded in `registration.json`.
- **Selection:** dev greedy top-1 macro, then lower dev CER, then the earlier update.
- **Resources:**
  - CPU only, 6 threads, one job.
  - Memory guard: a resumable stop after 3 consecutive readings with RSS > 6.5 GiB or MemAvailable < 0.4 GiB. These
    are the current thresholds (E10 decision 8).
  - Resumes check config, data and base hashes.
  - Active training time is logged per segment, separately from idle time.
- **Not done:** no pronunciation fusion, larger model, synthetic data, distillation or hyperparameter search.

## Evaluation (frozen before scoring)
- **Predictions:** greedy dev/eval predictions of E07, E11 and E08, written and hashed in `reports/frozen.json`
  before any score is computed.
- **Primary contrast:** E11 − E07, greedy, on historical eval (paired family bootstrap, 10,000, seed 20261027).
- **Reference only:** E08 greedy is shown as historical context, never as E11's source.
- **Also reported:**
  - dev and eval exact agreement and CER;
  - validity, empty and repeated outputs, missing endings;
  - error kinds and English-length bins (≤ 4, 5–8, ≥ 9 letters);
  - item outcome flips;
  - exposure, runtime and memory.
- **Caveat:** eval has been examined repeatedly and its references are unreviewed. No number measures phonetic
  accuracy.

## Decision rule (fixed now; `scripts/e11.py:DECISION_RULE`)
| Class | Condition |
|---|---|
| Useful real-data-only candidate | Eval E11 − E07 CI lower bound > 0, eval CER not higher, and dev top-1 not below E07 |
| Regression relative to E07 | Eval CI upper bound < 0, or point ≤ −3.0 with eval CER higher |
| Uncertain alternative | Otherwise |

No tuning after evaluation. A second seed may be *recommended* if promising, but is not launched.

## Review
A blinded ≤ 60-word dev comparison (40 representative + 20 disagreements, reported separately):
- the distinct greedy outputs of E07, E11 and E08, with origins hidden;
- the intended English reading;
- whether candidates differ in vowels or marks only, or in consonants or syllables;
- judgments: acceptable, not acceptable or uncertain per candidate, plus "none acceptable";
- no transcription is typed.

No reviewer is contacted and no judgment is invented.
