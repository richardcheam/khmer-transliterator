# E02 results — English words to Khmer phonetic spelling

**In one paragraph.** Feeding the model the English pronunciation (CMUdict phones) next to the letters gave a small
gain in agreement with the owner's list: +1.65 points on average over three paired seeds, positive in all three, 95%
interval [−1.65, +4.94]. The interval excludes the registered +5-point threshold and includes zero. Adding Khmer
phones → spelling pretraining (P2G) did not help in the one-seed pilot. The benchmark is unqualified, so by
registration the **evidence status is `inconclusive`** and the **letters baseline stays the default**
(`letters_s0`). All numbers are *reference agreement* (match with the owner's spellings), not proven phonetic
accuracy. Source of every number: `results.json` (written by `scripts/english_phonetic_eval.py report`).

## Status
| Dimension | State |
|---|---|
| Benchmark | `unqualified` (no independent review under the phonetic policy; `BENCHMARK_CARD.md`) |
| Experiment | `complete_paired`: 3 pilots, then phones and letters at seeds 1–2; 7/7 supervised runs, 1/3 P2G jobs |
| Evidence | `inconclusive` (registered for an unqualified benchmark). Fields: `positive_effect_supported` false, `meaningful_gain_ruled_out_conditional` **true**, `practically_equivalent_conditional` true |
| Adoption | `letters_baseline_retained`; default model `letters_s0` |

## Selection (dev only, before the final references were opened)
Greedy top-1 macro on the 86 primary-eligible dev items, seed 0: letters 20.9%, **phones 23.3%**, phones_p2g 22.1%
→ phones went to confirmation. Dev at seeds 1/2: letters 18.6% / 22.1%, phones 23.3% / 23.3%.

## Final results (raw beam-4 top-1; primary = 165 items in 162 families)
| Run | Best epoch | Dev | **Primary macro** | Items | Re-ranked¹ | Top-4² | Mean CER | Flagged (26) | Proxy (400) | Hours |
|---|---|---|---|---|---|---|---|---|---|---|
| letters_s0 | 27 | 20.9 | **16.7** | 28 | 18.5 | 40.4 | 0.353 | 19.2 | 30.1 | 0.25 |
| letters_s1 | 18 | 18.6 | **15.1** | 25 | 18.2 | 39.2 | 0.375 | 23.1 | 25.6 | 0.26 |
| letters_s2 | 23 | 22.1 | **16.0** | 27 | 19.1 | 43.5 | 0.347 | 26.9 | 27.1 | 0.25 |
| phones_s0 | 27 | 23.3 | **17.3** | 29 | 22.8 | 42.9 | 0.337 | 23.1 | 28.9 | 0.35 |
| phones_s1 | 19 | 23.3 | **16.7** | 28 | 21.9 | 42.9 | 0.334 | 23.1 | 30.0 | 0.36 |
| phones_s2 | 26 | 23.3 | **18.8** | 32 | 23.8 | 42.3 | 0.333 | 26.9 | 28.1 | 0.36 |
| phones_p2g_s0 | 15 | 22.1 | **16.7** | 28 | 21.9 | 46.0 | 0.350 | 19.2 | 30.7 | 0.40 + 0.54 P2G |

Percentages except CER. ¹ Fixed diagnostic re-ranker (`translit.pick` towards Google-lexicon words, the same for
every arm), not registered as a decision measure. ² Any of the 4 beam candidates accepted.

Arm means over seeds 0–2 (standard deviation across seeds): primary macro letters 15.95 (0.78), phones 17.59 (1.11);
re-ranked letters 18.62 (0.47), phones 22.84 (0.93); top-4 letters 41.05 (2.23), phones 42.70 (0.36).

## Confirmatory paired comparison: phones vs letters (primary track)
- Per-seed gains: +0.62, +1.54, +2.78 points (one family is 0.62 points). Mean **+1.65**.
- 95% paired family bootstrap (10,000 resamples, seed 20261027, per-family differences averaged over seeds):
  **[−1.65, +4.94]**. Conditional on these six trained models; it is not a seed-to-seed interval.
- Adoption conditions: qualified benchmark ✗, mean ≥ 5 ✗, positive in ≥ 2 of 3 seeds ✓, lower bound > 0 ✗,
  controls ✓, runtime guards ✓ → not adopted.
- What it means: the data are compatible with no gain and with a gain of up to about 5 points; a gain of 5 or more
  is unlikely *on this benchmark*. It does not show the phones make no difference.

## Exploratory comparisons (never pooled with the primary)
- **Proxy track** (400 LLM spellings, 375 families, the same paired seeds): +1.42, interval [−0.62, +3.56]; per seed
  −1.20, +4.40, +1.07.
- **One-seed pilots (seed 0):** phones vs letters +0.62 [−3.70, +5.56]; phones_p2g vs letters 0.00 [−5.56, +5.56];
  phones_p2g vs phones −0.62 [−6.17, +4.94]. P2G pretraining + replay cost 0.54 h + 0.05 h more training, and its
  dev best came early (epoch 15). One seed cannot separate it from phones; it mixes pretraining and replay.
- **Strata** (primary, mean over seeds): dictionary hit (143 items) letters 15.6, phones 17.3; dictionary miss (22,
  the phone arms get `<no_pron>`) letters 18.2, phones 19.7. Miss strata are small and noisy (22 items: one item is
  4.5 points).
- **Benchmark v1 panels** (historical, raw top-1, after selection): pick_test letters 33/24/24, phones 31/29/28 (of
  150); dev41 letters 5/7/9, phones 9/5/9 (of 41); list_test letters 18/18/17, phones 21/17/18 (of 125). Not
  comparable to E01's numbers (different models' inputs, re-ranking and rule).

## Runtime (Task 5 guards, `reports/runtime.json`; laptop, 12 CPUs, 6 threads, fresh process per model)
| Model | Cold load | Batch-1 p50 / p95 | p95 dict hit / miss | Batch-16 per word | Lookup per word | Peak RSS |
|---|---|---|---|---|---|---|
| letters_s0 | 0.09 s | 18 / 32 ms | 32 / 35 ms | 12 ms | — | 65 MiB |
| phones_s0 | 1.47 s | 21 / 36 ms | 35 / 37 ms | 15 ms | 0.1 ms | 137 MiB |
| phones_p2g_s0 | 1.50 s | 23 / 41 ms | 41 / 40 ms | 16 ms | 0.1 ms | 138 MiB |

Every guard passes (cold ≤ 30 s, p95 ≤ 1 s, RSS ≤ 1.5 GiB, phones p95 1.14× letters ≤ 2×). The phone models' extra
load time and memory are the CMUdict parse.

## Checks
- Measurement controls pass (accepted and NFD forms correct, corrupted form wrong, missing/unknown ids refused).
- Pairing: the arms of each seed share the initial-state hash (s0 `40dba049…`, s1 `2fdb7e13…`, s2 `6de41899…`) and
  the batch-order hash; the P2G arm differs only by its initialisation file (sha `f32300be…`).
- Export parity (NumPy vs torch) passes for all seven runs: 12 fixtures, equal source ids and greedy outputs, largest
  log-probability gap ≤ 5e-6.
- The final references were opened once (`data/e02/results/final.lock`, 2026-10-27 05:46); predictions for all
  seven models are in `data/e02/results/predictions.json`, per-item outcomes in `per_item.json`.
- Provenance: letters_s0 ran at `b9c7ea4`, the others at `ac5a157` (the difference is the CLI and an unset
  CMUdict path override, nothing in training). All are flagged `git_dirty` because uncommitted tests/docs were in
  the tree; their diff hash is in each `provenance.json`. Run ledger: `run_ledger.json`.

## Limits
- Reference agreement with one accepted spelling per item; real Khmer allows several, so all absolute numbers are
  low and the same correct-sounding output can count as wrong.
- 94 final items were picked from historical model E's candidates (E is phone-input); this could favour the phone
  arm. Primary strata: anchored letters 16.9 avg vs phones 18.5; not anchored 14.9 vs 16.5 (both differences
  similar, from `results.json`).
- Power was 0.29 for a true +5 points at this size (registered); a null-looking result here is weak evidence.
- Three seeds measure some training variability, not all of it.
