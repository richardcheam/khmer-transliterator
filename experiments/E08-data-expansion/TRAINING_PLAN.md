# E08 phase B — does the additional data help ByT5-small? (registered 2026-10-28; NOT executed)

**Status:** written and committed before any phase-B training.
- Execution needs:
  1. the synthetic import and `scripts/e08.py freeze` (state `data_frozen_ready_for_registration`);
  2. the owner's go-ahead for the compute below.
- Nothing in this file has been run.

## Arms (data only differs)
| Arm | Optimisation data | Rows | Words |
|---|---|---|---|
| **C** | E03 optimisation split (control) | 2,255 | 1,883 |
| **R** | C + recovered (1,292) + ALT-attested (1,217) | 4,764 | 3,585 |
| **R+S** | R + accepted `google_ai_mode_synthetic` rows (≤ 200 words, preferred + alternatives) | after import | after import |

- **Train-val:** the 93 E03 train-val rows stay identical in every arm, and new rows from train-val families are
  excluded.
- **Frozen hashes:** arm files are `data/e08/training/arm_{C,R,RS}.tsv`. The C and R hashes are in
  `reports/training_arms.json`; R+S is frozen by `freeze`.
- **Fallback:** a real-data expansion survived, so the comparison is C vs R vs R+S. The C-vs-C+S fallback is not
  needed.

## Fixed recipe (identical for every arm)
- **Recipe:** `experiments/E07-byt5-diagnosis/full_recipe.json` (sha256 `4df01024…c1d`), unchanged.
  - Base: pinned ByT5-small; LoRA r16, α32, dropout 0.05 on attention + FFN; base dropout 0.1.
  - Optimiser: AdamW, lr 1e-3, 100 warm-up, then linear decay to 0.
  - Batch 32, **3,000 updates**; dev checks at 600 / 1,200 / 1,800 / 2,400 / 3,000.
  - Each run starts from the base model.
- **Later work:** any recipe change is a separate experiment, so data and optimisation effects stay apart.

## Sampling and alternatives (fixed now)
- **Word-level sampler, the same in all arms:**
  - each epoch visits every optimisation **word** once, in a seeded order;
  - for each visit, one of that word's spellings in the arm is drawn uniformly (seeded);
  - so alternatives share one word's exposure instead of multiplying it, and repeated spellings cannot dominate.
- **Provenance weights:** equal. No up-weighting of synthetic or attested rows.
- **Arm C is a new run:** C uses this sampler too, so it is not E07's checkpoint (E07 sampled rows). E07 remains
  the reference for the recipe only.
- **Exposure:** at 3,000 × 32 = 96,000 draws, C makes about 51 passes over its words, R about 27, and R+S slightly
  fewer.
  - Equal updates mean unequal repetition. That is part of what "adding data" changes, and it is reported, not
    corrected.

## Selection and evaluation (frozen now)
- **Checkpoint selection:** per run, highest dev greedy top-1 macro (E03 dev primary, NFC exact), then lower dev
  CER, then the earlier checkpoint. These are the same 5 opportunities in every arm.
- **Historical evaluation** (E03 eval primary, 228 items; mined, unreviewed; a historical comparison):
  - predictions are frozen before scoring;
  - top-1 = beam-4 top-1 (sum log-prob); also greedy top-1, mean CER, top-4 oracle at equal beam 4, validity and
    diversity;
  - paired family bootstrap (10,000, seed 20261027);
  - never used for tuning or selection.
- **Primary contrasts:** R − C (real data), then (R+S) − R (synthetic on top of real data).
- **Kept distinct:** model outputs, agreement with mined references, and independently reviewed phonetic
  acceptability. None of the review queues has judgments, so the last is "not measured".

## Bounded replication
1. Seed 0 for C, R and R+S: 3 runs.
2. If a contrast's dev top-1 difference (seed 0) is **≥ +3.0 points**, run seeds 1 and 2 for both arms of that
   contrast: at most 4 more runs, **7 in total**.
3. A data effect is claimed only from the mean over 3 paired seeds, with the eval-set family-bootstrap interval of
   the seed-mean difference above 0.
   - One-seed results are reported as exploratory.
   - Seed-to-seed variation is reported.
4. No further runs, recipe changes or larger synthetic sets follow automatically. The 200-word set is a pilot.

## Resources (same limits as E07)
- CPU only, one training job at a time, 6 threads, RSS ≤ 5 GiB with ≥ 1.5 GiB available.
- ≤ 6 h per run (each E07 run took 4.7 h). At most 7 runs: about 33–42 CPU hours in total, run sequentially.
- No downloads, paid services or GPU.

## Not authorised by this plan
Distillation, promoting a product default, serving changes, publication, and treating synthetic spellings as
references.
