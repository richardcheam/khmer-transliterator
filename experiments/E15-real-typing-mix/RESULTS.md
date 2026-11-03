# E15 results (D208, 2026-11-03)

**Outcome: the rule fails. E15 is not adopted, there is no A/B, and serving is unchanged.** Dropping the
rule-romanised dictionary rows did not close the gap to served model B: on the owner test E15 is +2.6 points over
E14, an uncertain change, and still 19 points behind B. Full numbers: `results.json` (`scripts/e15_train.py score`).

## Technical checks
- Three seeds, each 23,200 spelling steps and 2,900 side-task steps on 74,168 pairs per epoch; 182 / 185 / 180 min on
  12 threads. No NaN; all on the same manifest (`e7096b1c…`).
- Best epochs: 31 / 26 / 30.
- **Seed 1 was restarted.** The first attempt was killed at epoch 10 when the agent session that launched it closed
  (not a crash). It was rerun from scratch with the same seed and recipe; the partial run is kept aside in
  `models/e15/aux_s1_killed/` and not used. No test score had been looked at before the restart.

## Primary: owner-typed test (160 words, 209 typed spellings; second use, declared)
| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| E15 | 45.7% | 45.2% | 41.1% | 44.0% |
| served B | | | | **63.1%** |
| E15 − B (points) | −17.4 | −17.9 | −22.0 | **−19.1** (rule: ≥ −3) |

Word bootstrap 95% CI of the mean difference: [−25.8, −12.3]. Words E15 better / worse / tied: 13 / 57 / 90.

## Guards (rule: romanised ≥ −5, names ≥ −3)
- `romanised_test`: 66.5 / 69.0 / 68.0% vs B 70.0%, giving a mean of **−2.1** (passes; E14 was −5.7).
- `names_test2`: 38.1 / 38.6 / 37.7% vs B 48.0%, giving a mean of **−9.9** (fails; E14 was −7.9).

## Hypothesis check (registered, not part of adoption)
E15 − E14 on the owner test: **+2.6 points**, word bootstrap 95% CI [−2.4, +7.6]. The registered bar was ≥ +5 with
a CI excluding 0, so "dropping the rule rows helps" is **not supported**. It is not a clear negative either: the
interval allows a small gain or a small loss.

## Diagnostics (not part of the rule)
- **Clean E01 aux models** on the owner test: 24.0%. E15 is +20.0 points above them.
- **The served router:** 61.3%.
- **English panels:** `pick_test` 28.0 / 28.7 / 26.7% (B 28.7%); `list_test` 33.6 / 32.8 / 34.4% (B 49.6%).

## Interpretation
Removing the 108k formal-rule rows helped where the data is Chhunneng-style (`romanised_test` moved from −5.7 to
−2.1 against B) and cost a little on names, but it did not move the owner's own chat typing much. The rule rows were
therefore not the main reason E14 trails B on the owner test. Part of the gap is that B saw 130 of the 160
owner-test words in training while E14/E15 saw none (filtered out by design), but in E14 B also led on the 30 words
it never saw, so that is not the whole story. Within the existing sources, changing the mix has now been tried; the
remaining lever is more real chat-typing data. The owner test has now been used twice.
