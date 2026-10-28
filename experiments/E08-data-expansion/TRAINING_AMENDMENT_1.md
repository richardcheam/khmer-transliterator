# E08 training-plan amendment 1 (2026-10-28, before any E08 training)

This amends `TRAINING_PLAN.md`; that file and `training_registration.json` are kept as registered. It replaces the
three-arm comparison with a single exploratory candidate, on the owner's instruction.

## What changes
- **Arms:** C and R stay frozen and available (hashes unchanged), but they are **not trained in this phase**.
- **One candidate,** `rs_s0`:
  - data: the frozen arm R+S (`data/e08/training/arm_RS.tsv`, sha256 `03a4988f…`);
  - recipe: the E07 recipe (`experiments/E07-byt5-diagnosis/full_recipe.json`, sha256 `4df01024…`);
  - fresh LoRA initialisation on the same pinned base checkpoint, **not** continued from E07's adapter;
  - seed 0; ≤ 3,000 updates or 6 training hours;
  - E07's checkpoint selection: dev greedy top-1 macro, then lower CER, then the earlier checkpoint, at updates
    600/1,200/1,800/2,400/3,000.
- **Baseline:** the saved E07 checkpoint and its frozen predictions (`experiments/E07-byt5-diagnosis/predictions/`).
- **Attribution:** none. The comparison cannot say which source caused a difference: recovered pairs, ALT pairs,
  synthetic pairs, or simply more data and fewer passes per row.
- **Sampling = E07's convention:** row-level.
  - Each epoch is a seeded permutation of the optimisation rows, and every alternative spelling is its own row.
  - This replaces the plan's word-level sampler for this run, to avoid an undocumented objective change at the same
    time as the data change.
  - There is no oversampling of any source.
  - The excluded synthetic words (5 look-alike-script, 2 uncertain) are not regenerated, and accepted spellings are
    not edited.

## Exposure checks (`reports/rs_exposure.json`, before training)
- **Deduplication:** exact pairs are deduplicated across sources. 0 pairs occur in more than one source, so no
  provenance was dropped.
- **Exclusions:** protected-family and dev/eval exclusions are re-checked on every training row.
- **Row shares:** 2,255 control, 1,292 recovered, 1,217 ALT and 384 synthetic.
  - The synthetic rows cover **193 words**: one preferred spelling plus up to two alternatives each. They are not
    384 independent words.
- **Alternatives:** with row-level sampling, a word with k spellings is seen k times per epoch, so alternatives
  raise that word's exposure. This is the same as in E07.
- **Actual draws:** the actual draws per provenance over the run are counted from the deterministic order and
  reported.

## Pre-registered decision rule (eval = historical E03 eval primary, beam-4 top-1)
Let Δ be the top-1 difference rs_s0 − E07, with its paired family-bootstrap 95% interval.

| Classification | Condition |
|---|---|
| **Promising combined candidate** | Δ ≥ +3.0 points **and** CI lower bound > 0 **and** dev greedy top-1 not lower than E07's **and** eval mean CER not higher than E07's |
| **Regression** | CI upper bound < 0, **or** Δ ≤ −3.0 points |
| **Small or uncertain change** | Otherwise |

- **If promising:** the next action is one additional-seed replication (seed 1), proposed and not launched.
- **Otherwise:** the result is kept, the strongest evidence about the limitation is identified, and no speculative
  fixes are made.

## Interpretation limits
- **The eval set:** the historical E03 eval set has been examined repeatedly and is unreviewed (mined,
  consonant-screened). Results are exploratory reference agreement, not phonetic accuracy.
- **Synthetic labels:** never used as references.
- **Agent inspection:** labelled as such.
- **One seed:** one seed measures no training variance.
