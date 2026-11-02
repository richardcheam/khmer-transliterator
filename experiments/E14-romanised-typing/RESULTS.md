# E14 results (D206, 2026-11-02)

**Outcome: the rule fails. E14 is not adopted, there is no A/B, and serving is unchanged.** The new data helps a lot over the
clean E01 model (+17.4 points on the owner test), but E14 stays well behind served model B. Full numbers:
`results.json` (`scripts/e14_train.py score`).

## Technical checks
- Three seeds, each 56,840 spelling steps and 7,105 side-task steps; 434 / 442 / 443 min on 12 threads. No NaN;
  all on manifest `5cf574d4…`.
- Best epochs: 17 / 29 / 14.

## Primary: owner-typed test (160 words, 209 typed spellings)
| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| E14 | 40.0% | 41.7% | 42.7% | 41.4% |
| served B | | | | **63.1%** |
| E14 − B (points) | −23.1 | −21.5 | −20.5 | **−21.7** (rule: ≥ −3) |

Word bootstrap 95% CI of the mean difference: [−28.8, −15.0]. Words E14 better / worse / tied: 7 / 53 / 100.

## Guards (rule: romanised ≥ −5, names ≥ −3)
- `romanised_test`: 64.5 / 64.5 / 63.8% vs B 70.0%, giving a mean of **−5.7** (fails).
- `names_test2`: 41.3 / 39.5 / 39.5% vs B 48.0%, giving a mean of **−7.9** (fails).

## Diagnostics (not part of the rule)
- **Clean E01 aux models** on the owner test: 24.1 / 24.1 / 24.0%. The new data adds **+17.4 points**.
- **The served router** (the full system with routing to E for English-dictionary words): 61.3%.
- **B saw 130 of the 160 test words** in its training (Chhunneng train split); E14 saw none. B leads on both groups:
  - the 130 seen words: B 63.8%, E14 39.2–41.3%;
  - the 30 unseen words: B 60.0%, E14 43.3–51.7%.

  So the gap is not only memorisation, but the unseen group is small.
- **The re-ranking list is not the cause.** E14 seed 1 with B's served list gets 84/209 spellings right, the same as
  with the Google list (B: 132/209).
- **English panels:** `pick_test` 21.3 / 28.7 / 21.3% (B 28.7%; English stays with model E in serving anyway).
- **Typical misses:** E14 spells chat forms literally rather than as the Khmer word people mean. Examples, with
  the wanted word and then E14's output:
  - *ber* → បើ (E14 ប៊ែរ);
  - *nis* → នេះ (នីស);
  - *tha* → ថា (តា);
  - *yok* → យក (យក់);
  - *tmey* → ថ្មី (ត្មី).

  B gets these right.

## Interpretation (a post-hoc hypothesis, not tested)
Rule-romanised dictionary rows are 107,617 of E14's ~182k pairs per epoch, against 22,814 Chhunneng rows. B was
trained on Chhunneng and mined pairs only (59k pairs). In E14, the formal rule spellings may drown out the chat
conventions. Testing this needs a new registered experiment, for example a different mix or weight. The owner
test has now been used once, so a second use must be declared.
