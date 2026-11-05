# E16 results (D210, 2026-11-06)

**Outcome: the adoption rule fails. E16 is not adopted, there is no A/B, and serving is unchanged.** The data check
passes: bulk Gemini spellings give a clear gain over E15 (+6.5 points, CI [+3.1, +10.1]), the first clearly
supported gain in this line of experiments, but E16 is still 12.6 points behind served model B on the owner test.
Full numbers: `results.json` (`scripts/e16_train.py score`).

## Data
- **Gemini:** all 40 replies received (the free key's daily quota stopped the fetch at 17, then 31; it resumed with
  `scripts/e16_resume.sh`, no setting changed). 9,998 words answered, 16,816 spellings.
- **Automatic filter:** kept 14,467 spellings for 9,016 words; dropped 8 malformed, 260 trusted clashes, 2,081
  given for several words.
- **Leak filter:** 12,608 Gemini rows entered training (805 touched an owner-test or benchmark spelling, 1,054 a
  held-out Khmer target). 86,776 spelling pairs per epoch (E15: 74,168). Manifest `160edb5a…`.

## Technical checks
Three seeds, same manifest, no NaN. Best epochs 20 / 26 / 39; 240 / 243 / 213 min on 12 threads.

## Primary: owner-typed test (160 words, 209 spellings; third use, declared)
| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| E16 | 51.9% | 50.6% | 49.2% | 50.6% |
| served B | | | | **63.1%** |
| E16 − B (points) | −11.3 | −12.5 | −14.0 | **−12.6** (rule: ≥ −3) |

Word bootstrap 95% CI: [−19.5, −5.9]. Words E16 better / worse / tied: 16 / 45 / 99.

## Guards (rule: romanised ≥ −5, names ≥ −3)
- `romanised_test`: 66.6 / 66.5 / 67.2% vs B 70.0%: mean **−3.2** (passes).
- `names_test2`: 41.3 / 43.5 / 41.3% vs B 48.0%: mean **−6.0** (fails; E15 was −9.9).

## Data check (registered, not part of adoption)
E16 − E15 on the owner test: **+6.5 points**, CI [+3.1, +10.1]. The bar was ≥ +5 with the CI excluding 0, so
"bulk Gemini data helps" is **supported**.

## Diagnostics (not part of the rule)
- Owner-test trend: clean E01 24.0% → E14 41.4% → E15 44.0% → E16 50.6% (E16 − E14 +9.1, CI [+4.1, +14.8]).
- The served router: 61.3%.
- **Words B saw in training vs not.** B was trained on 130 of the 160 owner-test words; E16 saw none (filtered out).
  On those 130: B 63.8%, E16 48.1%. On the 30 words B never saw: B 60.0%, E16 61.1% (E15 52.8%). On unseen words
  E16 is level with B, but 30 words is too few for a firm conclusion, and this split was not registered.
- English panels: `pick_test` 21.3 / 22.0 / 28.0% (B 28.7%); `list_test` 28.0 / 28.8 / 29.6% (B 49.6%).

## Interpretation
Bulk LLM spellings, filtered only by fixed rules, are the first data source that clearly moves the owner's chat
typing. Most of the remaining gap to B is on words B memorised from its training data; on words neither model saw,
E16 is about level. Names stay weaker than B. More of the same data (more words, or a second LLM for words Gemini
gets wrong) is the obvious next lever; a fair comparison with B needs fresh test words B never saw (the 150 frozen,
untyped words in `test_frozen.json`).
