# E07 — ByT5 diagnosis: report (2026-10-28)

**Outcome class (registered, PLAN §6): promising ByT5 research candidate**, but only marginally:
- It learned an input-dependent transliteration.
- On the historical E03 eval set, its top-1 agreement with mined references beats the mean of the three letters
  models (E04 G): +5.8 points, 95% interval [+0.05, +11.6].
- It is not distinguishable from the product default: +1.0 [−6.1, +8.0].
- Independent phonetic quality is unverified. This authorises no distillation, publication or deployment.

**Commits:** plan `0e3250b` (before any training), registration `2020fef` (before full adaptation), results in
the commit that adds this file. Product default and serving unchanged.

## 1. What caused E06's behaviour
Not a software bug: an **adaptation limitation**.
- **All ten pipeline checks pass** (`reports/pipeline_audit.md`): ids, alignment, masking, label padding, shift and
  EOS, adapter placement, gradients and updates, reload, limits, scores.
- **How a falling loss coexisted with constant outputs:**
  - Two of the three UTF-8 bytes of a Khmer character are nearly free to predict.
  - E06's per-byte loss (0.79) is therefore mostly Khmer byte structure plus a Khmer letter model.
  - On the informative final byte, the correct English word helped by only 0.04 nats over someone else's word
    (1.99 vs 2.03).
  - Free-running decoding then follows the letter model's favourite path: ហ្វេន for 204 of 228 eval words.
- **Tiny memorisation from the base model passed at once:** 32/32 after 100 updates (3 minutes). So the pipeline can
  learn an input-dependent mapping.
- **The corrected full run learned the task** (below).
- **Unresolved:** which of E06's differences mattered most is not isolated, because several changed at once:
  - attention-only adapters vs attention + FFN;
  - 1,000 updates × batch 16 (7 epochs) vs 3,000 × 32 (43 epochs; the best checkpoint at 1,200 updates is 17
    epochs).
- **Defects found:** one latent one, never triggered. E06 left no room for EOS on a 120-byte target, and the longest
  real target is 51 bytes. It is fixed in E07 with a regression test.

## 2. Checks and what they establish
| Check | Result | Establishes |
|---|---|---|
| Pipeline audit 1–10 | pass | The data, loss, adapter, reload and decoding boundaries behave as intended on the tested batches. It does not prove the code bug-free |
| E06 reproduction (16 fixed rows) | 0/16 exact, 8 distinct outputs | E06's collapse is reproducible from its saved adapter |
| Tiny gate (32 pairs) | 32/32 exact; 32 distinct; shuffled input → the other word's target 32/32; final-byte loss 0.0001 correct vs 5.67 shuffled; reload identical | The setup can memorise an input-dependent mapping. This is not generalisation |
| Full run, input dependence (train-val, 93 rows unseen in optimisation) | final-byte loss 0.41 correct vs 6.99 shuffled vs 3.88 empty; 75 distinct greedy outputs; 35/93 exact | Outputs now depend on the input (E06: 1.99 vs 2.03; 13 distinct; 0 exact). A diagnostic, not quality |

## 3. Tiny-subset results (`reports/tiny_overfit.json`)
- **Setup:** attempt 1 only. LoRA r16/α32 on attention + FFN, no dropout, lr 1e-3 constant, batch 16.
- **Result:** stopped at 32/32 after 100 updates, 0.05 h, peak RSS 2.6 GB.
- **Saved:** the full 32-row table (target, output, loss, output when given another word's input), in the JSON.
- **Sample caveat:** the sample includes noisy mined targets (*bracket* → ប្រកួត, "competition"). It was kept
  because selection was fixed before any output.

## 4. Full adaptation (`full_recipe.json`, registered before the run)
- **Recipe:**
  - LoRA r16, α32, dropout 0.05 on q/k/v/o and wi_0/wi_1/wo of every block: 6.26M trainable parameters;
  - base dropout 0.1;
  - AdamW, lr 1e-3, 100-update warm-up then linear decay to 0;
  - batch 32, 3,000 updates;
  - E03 optimisation split, 2,255 pairs (verified).
- **Ran:** 3,000 updates in **4.67 h** (cap 6 h; 5.6 s per update), ending with `max_updates`.
- **Memory:** peak RSS 4.8 GB during dev generation (cap 5 GiB); about 3.6 GB while training.

| Update | Dev greedy top-1 (macro) | Dev CER | Distinct dev outputs |
|---|---|---|---|
| 600 | 37.8% | 0.287 | 244 |
| **1,200 (selected)** | **52.2%** | 0.186 | 241 |
| 1,800 | 51.4% | 0.166 | 243 |
| 2,400 | 51.0% | 0.175 | 244 |
| 3,000 | 50.6% | 0.168 | 244 |

Dev is the selection set, so these are optimistic.

## 5. Historical E03 eval (228 primary items, 222 families)
- **Labels:** mined, consonant-screened, unreviewed spellings, already used by E03–E06. A historical research
  comparison, not a fresh benchmark.
- **Frozen first:** predictions were frozen and hashed before scoring (`reports/predictions_frozen.json`).
- **Decoding:** top-1 = beam-4 top-1 (sum log-prob) for ByT5 and G; the product default uses its own decoder.

| System | Top-1 | Top-4 (oracle, beam 4) | Mean CER | Distinct top-1 | Most frequent top-1 |
|---|---|---|---|---|---|
| **ByT5 E07** (beam-4 top-1) | **47.2%** | 64.7% | 0.189 | 220 | 2 (0.9%) |
| ByT5 E07 (greedy) | 49.8% | — | 0.178 | 220 | 2 |
| ByT5 E06 | 0.0% | 0.0% | 0.981 | 13 | 204 (89%) |
| Product default (E02 `letters_s0`) | 46.2% | 75.6% | 0.207 | 225 | 2 |
| E04 G_s0 / G_s1 / G_s2 | 42.7 / 41.2 / 40.0% | 64.6 / 66.8 / 66.8% | 0.204 / 0.219 / 0.227 | 222–225 | 2 |
| G mean | 41.3% | 66.1% | 0.217 | | |

**Paired family bootstrap** (10,000, seed 20261027), in points. The intervals are conditional on these
checkpoints: one ByT5 run measures no seed variability.

| Comparison | Difference | 95% interval |
|---|---|---|
| ByT5 E07 − G mean, top-1 | +5.8 | [+0.05, +11.6] |
| ByT5 E07 − G_s0 / G_s1 / G_s2, top-1 | +4.4 / +5.9 / +7.1 | [−2.1, +11.0] / [−0.5, +12.4] / [+0.5, +13.9] |
| ByT5 E07 − product default, top-1 | +1.0 | [−6.1, +8.0] |
| ByT5 E07 − ByT5 E06, top-1 | +47.1 | [+40.5, +53.9] |
| ByT5 E07 − G mean, top-4 oracle (equal beam-4 budget) | −1.4 | [−6.8, +4.1] |
| CER reduction vs G mean | 2.7 | [0.3, 5.0] |

**Item flips vs G_s0:** ByT5 right and G_s0 wrong on 33 items; the reverse on 23.
- ByT5 right, G_s0 wrong: *ali* អាលី (G អេលី), *arena* អារីណា (G អារ៉ាណា), *barcelona* បាសេឡូណា (G បាស្គេលូណា).
- G_s0 right, ByT5 wrong: *boak* (ByT5 បុក vs reference បូក), *cobalt* (ByT5 កូបាល vs reference កូបាល់).

**Strata (top-1 macro; small cells are noisy):**

| Stratum | Items | ByT5 E07 | Default | G_s0 / s1 / s2 |
|---|---|---|---|---|
| Length ≤ 4 | 87 | 50.6% | 49.4% | 50.6 / 47.1 / 47.1% |
| Length 5–6 | 102 | 47.5% | 46.0% | 42.6 / 42.6 / 41.6% |
| Length 7–8 | 30 | 37.9% | 34.5% | 32.8 / 24.1 / 22.4% |
| Length ≥ 9 | 9 | 55.6% | 44.4% | 11.1 / 22.2 / 22.2% |
| Eval word in the default's E02 training data | 110 | 42.1% | 33.3% | 35.2 / 34.3 / 38.9% |
| Not in it | 118 | 53.0% | 57.7% | 50.4 / 47.9 / 41.9% |
| Ambiguous track (excluded from primary) | 17 | 35.3% | 29.4% | 17.6 / 17.6 / 29.4% |

- **Default's exposure:** 110 of the 228 eval words occur in the product default's E02 training file
  (decision 8, added before scoring). On those words the default is *lower*, not higher. The likely reason is
  that E02 and E03 mined different spellings for the same words. The default's total is therefore not simply
  inflated, but the two settings are not like for like.
- **Validity:** every greedy output is valid UTF-8 ending with EOS. 3 of 980 beam-4 candidates are malformed
  (counted as failures). Orthography rules accept every top-1.

## 6. Fresh E06 pool (56 words; no references; heuristics only)
- **Distinct top-1:** ByT5 E07 56 (E06: 8).
- **Orthography:** every system passes on every word.
- **Khmer-lexicon word as top-1:** ByT5 E07 5, against 53 for E06. E06 output common Khmer words.
- **Frozen consonant screen at level 0** (48 primary words; a heuristic, **not accuracy**): ByT5 E07 28, default 30,
  G 18–21, E06 0.
  - This counts the primary track only, so it differs from E06's table, which counted all 56.
- Predictions: `predictions/byt5_e07_pool.jsonl`. The blinded E06 queue could be extended with them; no judgments
  exist.

## 7. Cost (`reports/cost.json`; i7-8750H laptop, 6 threads, CPU)
| | ByT5 E07 | Letters model G (E05) |
|---|---|---|
| Load | 5.1 s (warm page cache) | 0.04 s |
| Batch-1 latency | 501 ms greedy, 650 ms beam 4 (median) | 14.7 ms beam 4, 21.9 ms beam 8 |
| Throughput | 13.8 words/s (greedy, batch 32) | — |
| Peak RSS | 2.0 GB | 0.13 GB |
| Files | 1.2 GB base + 25 MB adapter | a few MB, NumPy |

ByT5 is about 30–40× slower and 15× larger in memory. It could serve only as an offline research model, not in
the serving path.

## 8. What can and cannot be claimed
- **Can be claimed:**
  - the E06 collapse was an adaptation limitation, not a pipeline bug;
  - the corrected recipe learns input-dependent English → Khmer spellings;
  - on mined historical references, one ByT5-small run agrees slightly more often than E04's letters models
    (barely excluding 0) and about as often as the product default.
- **Cannot be claimed:**
  - better phonetic quality: references are mined and unreviewed, and vowels are effectively unassessed;
  - robustness to training seeds: one run;
  - a qualified teacher;
  - fitness for serving: the cost above.
- **Selection effects:** the checkpoint was selected on dev, and eval had informed earlier experiments.

## 9. Budget and deviations
- **Training:** 0.05 h (tiny) + 4.67 h (full) = 4.72 h of the 8 h. 1 of 2 tiny attempts; 1 of 1 full run.
- **Other:** no downloads, requests or GPU. 6 threads. Peak RSS 4.8 GB (cap 5 GiB). Available memory stayed above
  2.3 GB at the checks taken.
- **Deviation, wall time:** the registered 12-hour wall cap (from 20:17 UTC) was exceeded.
  - Training ended at 01:11 UTC; the session was then idle with no process running until 08:18 UTC.
  - Inference and analysis (prediction, scoring, cost: about 10 minutes of computation) ran after the cap.
  - This follows the instruction to finish analysis when a cap is reached. No training ran after it.

## 10. Reproduce
```bash
uv sync && uv pip install --python .venv/bin/python -r research/requirements-byt5.txt
export CUDA_VISIBLE_DEVICES=
.venv/bin/python scripts/e07.py env
.venv/bin/python scripts/e07.py reproduce
.venv/bin/python scripts/e07.py audit
.venv/bin/python scripts/e07.py tiny --attempt 1
.venv/bin/python scripts/e07.py register        # needs a passed tiny gate
.venv/bin/python scripts/e07.py train           # resumable; refuses changed config/data
.venv/bin/python scripts/e07.py predict
.venv/bin/python scripts/e07.py analyse
.venv/bin/python scripts/e07.py cost
.venv/bin/python -m unittest tests.test_e07
```
`scripts/e07.py` is now `register`'s frozen version plus the decision-8 stratum. The ledger (`runs/ledger.jsonl`)
blocks a second tiny pair or full run. Delete it and `models/e07/` to rerun from scratch.

## Artifacts
- **Plan and recipe:** `PLAN.md`, `config.yaml`, `full_recipe.json`, `registration.json`,
  `SOURCE_VERIFICATION.md`.
- **Environment:** `research/requirements-byt5.txt`, `.lock.txt`.
- **Reports:**
  - `reports/environment.json`, `reports/e06_reproduction.json`;
  - `reports/pipeline_audit.{json,md}`;
  - `reports/tiny_overfit.json`;
  - `reports/predictions_frozen.json`, `reports/cost.json`.
- **Ledger:** `runs/ledger.jsonl`.
- **Predictions:** `predictions/byt5_e07_{dev,eval,pool}.jsonl` and `predictions/default_e02_eval.jsonl`.
- **Results:** `results.json`, `results/per_item.jsonl`.
- **Code and decisions:** `scripts/e07.py`, `tests/test_e07.py`, `docs/decisions/e07-byt5-diagnosis.md`.
- **Local only (git-ignored):** `models/e07/{tiny_a1,full}/`, holding `best.pt` adapters, `last.pt`, metrics and
  W&B runs `E07-byt5-diagnosis-*`.
