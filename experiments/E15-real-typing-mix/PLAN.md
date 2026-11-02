# E15: E14 without the rule-romanised dictionary rows (registered D207, 2026-11-02, before training)

**Question:** E14 spelled chat words in the formal rule style (*ber* → ប៊ែរ, not បើ). Does it get closer to served
model B once the 107,617 rule-romanised dictionary rows are removed?

This tests the post-hoc hypothesis of D206, in which the rule rows drown out real typing (they were ~59% of E14's
pairs, and B was trained without them).

## The only change from E14
Source `roman` is dropped (`scripts/e15_train.py`, `DROP = {"roman"}`). Everything else is identical to E14
(`scripts/e14_train.py`, `REGISTRATION.md` there):
- the other sources, the owner rows and the repeat factor (×4);
- the leak filter, validation sets and checkpoint selection;
- model, side task, optimiser, 40 epochs, seeds 0, 1, 2 and 12 threads.

This gives 74,168 spelling pairs per epoch (E14: 181,785). A one-epoch smoke run took 6.7 min, so about 4.5 h per seed.

## Test-set use (declared)
The owner-typed test (160 words) was scored once, for E14 (D206). This is its second use. No word, threshold or
scoring rule changes. The only things learned from it were E14's overall result and the kind of error, which is
what motivates this experiment.

## Evaluation
As E14, with the same owner test, guards, re-ranking and served-B reference. E14's three seeds are also scored on the
owner test for a paired comparison (the same words, the same scoring).

## Adoption rule (from exact counts, mean over seeds 0, 1, 2)
E15 goes to the owner's blinded A/B against served B only if all of these hold. The thresholds are E14's, unchanged:
- owner test: E15 − B ≥ **−3 points**;
- `romanised_test`: E15 − B ≥ **−5 points**;
- `names_test2`: E15 − B ≥ **−3 points**.

**Hypothesis check** (reported, not part of adoption): E15 − E14 on the owner test, with a word bootstrap 95% CI.
"Dropping the rule rows helps" is supported if the mean is ≥ +5 points and the CI excludes 0. A clear negative
result counts against the hypothesis.

If the rule fails, the result is recorded and serving is unchanged. There is no recipe change after seeing results.

## Run (owner starts it)
```
scripts/e15_run.sh        # seeds 0, 1, 2 one after another, export each, then score (about 13-14 h)
```
