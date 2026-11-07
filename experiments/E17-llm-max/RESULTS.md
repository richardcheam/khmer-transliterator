# E17 results (D212, 2026-11-07)

**Outcome: the adoption rule fails on the owner test. E17 is not adopted, there is no A/B, and serving is
unchanged.** Both guards pass (names is now level with B). E17 is the best model of this line on the owner test
(54.8%), but still 8.3 points behind served model B. The data check (E17 − E16 ≥ +3 with a CI excluding 0) is
**not supported**: the gain is +4.2 points, but the CI includes 0. Full numbers: `results.json`
(`scripts/e17_train.py score`).

## Data
- **Replies:** Claude 106 of 106, GPT 106 of 106, Gemini 26 of 33 new-word prompts complete plus the first 200
  lines of one cut-off reply (the owner ended collection early, before training). All 40 E16 Gemini replies.
- **Automatic filter:** 26,301 words, 80,169 spellings; kept 65,600 spellings for 23,504 words; dropped 197
  malformed, 687 trusted clashes, 13,685 given for several words. Kept by source: Claude only 16,689, Gemini only
  15,069, GPT only 11,019, all three 9,041, Claude+GPT 7,113, Claude+Gemini 4,113, Gemini+GPT 2,556.
- **Leak filter:** 60,859 LLM rows entered training; 135,027 spelling pairs per epoch (E16: 86,776). Manifest
  `eeaa905c…`.

## Technical checks
One model (seed 0, owner's choice), no NaN. Best epoch 30 of 40 (validation mean 0.403; E16 seed 0 at the same
epoch 0.379); 314 min on 12 threads.

## Primary: owner-typed test (160 words, 209 spellings; fourth use, declared)
| | owner test |
|---|---|
| E17 (seed 0) | **54.8%** |
| served B | 63.1% |
| E17 − B (points) | **−8.3** (rule: ≥ −3) |

Word bootstrap 95% CI: [−15.8, −1.4]. Words E17 better / worse / tied: 14 / 29 / 117.

## Guards (rule: romanised ≥ −5, names ≥ −3)
- `romanised_test`: 65.4% vs B 70.0%: **−4.5** (passes; E16 −3.2).
- `names_test2`: 48.4% vs B 48.0%: **+0.4** (passes; E16 −6.0).

## Data check (registered, not part of adoption)
- E17 − E16 (mean of three seeds): **+4.2 points**, CI [−1.3, +9.9]. The bar was ≥ +3 with the CI excluding 0, so
  "more LLM data helps further" is **not supported** (likely, not shown).
- E17 − E16 seed 0 alone: +2.9 points, CI [−3.8, +9.3]; words better / worse 20 / 16.

## Registered diagnostic: words B saw in training vs not
B was trained on 130 of the 160 owner-test words; E17 saw none. On the 30 words B never saw: **E17 65.0%**, B 60.0%,
E16 60.0 / 60.0 / 63.3%, E15 52.8% (mean). On the 130 words B saw: E17 52.4%, B 63.8%. Thirty words is too few for a
firm conclusion.

## Other diagnostics
- Owner-test trend: clean E01 24.0% → E14 41.4% → E15 44.0% → E16 50.6% → E17 54.8% (E17 − E15 +10.8, CI
  [+5.0, +16.9]).
- English panels: `pick_test` 22.7% (B 28.7%), `list_test` 25.6% (B 49.6%), `dev41` 34.1% (B 48.8%).

## Interpretation
Three LLMs' spellings for 2.6 times more words keep moving the owner's chat typing upward and closed the names gap,
but one model cannot separate a 4-point gain from chance. The remaining gap to B is on the words B memorised; on
words neither model saw, E17 is ahead of B. The owner test (fourth use, mostly B's training words) can no longer
settle E17 against B: that needs the 150 frozen fresh words (`../E16-gemini-bulk/test_frozen.json`), typed by the
owner, scored once for B, E16 and E17 with the same thresholds.
