# E01 — Results

**Outcome: the side task helps, but misses the adoption rule** (mean +4.67 points < 5). By the plan there was no
listening stage; the reserve stays sealed. (D174, 2026-10-16.) Full numbers: `results.json` (reproduced
byte-identically from the exported models on 2026-10-24 with `scripts/translit_unified.py score`).

## Technical checks
- Every pair: identical spelling batches (batch-order hashes equal), 48,800 spelling steps; the aux arm added
  6,100 aux steps. All trained on the current data manifest; no NaN; exports verified.
- Best epochs: base 28 / 20 / 16, aux 31 / 20 / 32. Compute: base 571 / 591 / 570 min, aux 613 / 631 / 607 min
  (+7%). Laptop CPU, 6 threads.

## Primary: pick test (re-ranked exact, 150 English words)
| Seed | base | aux | Δ (points) |
|---|---|---|---|
| 0 | 24.7% | 28.0% | +3.33 |
| 1 | 24.7% | 26.7% | +2.00 |
| 2 | 20.7% | 29.3% | +8.67 |
| **Mean** | | | **+4.67** (rule: ≥ 5) |

Positive in 3 of 3 seeds. Word bootstrap 95% CI of the mean: [+2.0, +7.6]. Words better / worse / tied (mean over
seeds): 23 / 6 / 121.

## Guards (rule: mean regression ≤ 1 point)
- Romanised test: +0.40 / +0.67 / −0.58, mean **+0.16** (passes).
- Test2 names: +8.52 / 0.00 / +2.24, mean **+3.59** (passes).

## Diagnostics (not part of the rule)
- dev-41: +4.9 / +2.4 / +2.4 points.
- List test: −4.8 / −3.2 / +4.8 (mean −1.07): mixed.
- Raw (no re-ranking) pick test: +6.0 / +5.3 / +9.3.

## Against the served system (B + E, served routing as if unlisted)
| Panel | clean models (6 runs) | served |
|---|---|---|
| Romanised test | 33–35% | **70.0%** |
| Test2 names | 35–44% | 48.0% |
| Pick test | 21–29% | 30.7% |
| dev-41 | 44–51% | 61.0% |
| List test | 27–34% | 55.2% |

## Owner's informal listening (outside the plan; no adoption rule)
Clean `aux_s1` (best validation) vs served, 10 sentences through the full speech pipeline
(`reports/translit_listen_clean.json`): served only right 3, clean only right 0, both right 5, neither 2. The clean
model matched served on names and English words and lost only on everyday romanised chat words (mok, ot, te,
rean, nis, laor).

## Conclusions
1. The side task is a consistent, cheap gain on English (every seed positive, CI above 0) with no harm elsewhere:
   keep it in future recipes.
2. The 5-point bar was not reached: not adopted, no further tuning of this recipe (as planned).
3. The blocker for a releasable single model is **data**: no clean source of real romanised typing. Rule-romanised
   dictionary words do not transfer to how people type chat words.

## Next (candidate experiments)
- Clean real-typing data: ask the Chhunneng author about provenance; collect owner/tester typing of common words;
  mine romanised Khmer from web text.
- Then E02: the E01 aux recipe plus that data, against the same benchmark v1.
