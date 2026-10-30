# E12 report: longer schedule for ByT5-small (6,000 updates) (2026-10-30)

**Outcome: no detectable difference from E11.** On the blinded 120-word test the owner accepted exactly the same share
of E12 and E11 spellings: 77/120 (64.2%) each, difference 0.0 points [−8.3, +8.3]; 14 words favour each model
(exact p = 1.0). A non-significant difference is not equivalence: the interval allows about ±8 points either way.

- **Run:** one run, seed 0, all 6,000 updates in 9.08 h of active CPU compute, one segment, no interruption.
- **Selected:** update 6,000, by the registered rule (lowest dev CER 0.1542; also the highest dev exact, 55.0%).
- **Historical dev (mined references, exploratory):** E12 55.0% vs E11 53.0% (+2.0 [−2.4, +6.4]), CER 0.154 vs 0.157.
- **Historical eval (exploratory):** E12 53.0% vs E11 52.3% (+0.7 [−4.1, +5.4]), CER 0.160 vs 0.163. No
  detectable difference.
- **Blinded test (primary):** 120 fresh words, 185 spellings; owner-confirmed acceptable E12 64.2% vs E11 64.2%.
- **Nothing was published, packaged, served, merged or tagged. No second run or seed was launched.**

## 1. Starting state and registration
- **Start checks** (`reports/start_checks.json`, all pass): E11 checkpoint `286292a4…` at update 3,000, pinned
  base, arm R (4,764 rows), E11 recipe hash `f5d7cf92…`, the 60 owner dev judgments with provenance, E09 greedy
  decoding.
- **Registration** (`registration.json`, cfg hash `a6159bed…`, commit 1f058f5, before training): E11's recipe with
  `max_updates` 6,000, linear decay over 6,000, checkpoints at 1,200 / 2,400 / 3,000 / 3,600 / 4,800 / 6,000 and the
  CER-first selection rule. Fresh adapter on the base; E11's exhausted scheduler was not resumed.
- **Exposure:** 192,000 example draws = 40.3 epochs over R (E11: 20.15). Draw shares: control 47.3%, recovered
  27.1%, ALT 25.6%; draws per word min 37, median 41, max 326. (`exposure.e07_reference.epochs` = 85.14 is E07's
  2,255 rows at this 6,000-update horizon, not E07's actual 42.6.)
- **Test inputs** (`test/input_manifest.jsonl`, sha256 `e6d75165…`): 120 families frozen before training; overlap
  audit in `test/frozen.json` and `tests/test_e12.py`. Amendments before freezing are in PLAN.md.

## 2. Run (`runs/ledger.jsonl`, `models/e12/long_s0/`, local)
- **Resources:** CPU, 6 threads, one process, one attempt. Peak RSS 4,865 MB; the memory guard never triggered.
- **Time:** 9.08 h active training (about 5.4 s/update including six dev checks). Freezing took 45 s, scoring 2 s,
  test generation 33 s.
- **Loss** (mean over 100 updates): 3.28 at the start, 0.199 at 1,200, 0.116 at 3,000, 0.072 at 6,000.

| Update | 1,200 | 2,400 | 3,000 | 3,600 | 4,800 | **6,000** |
|---|---|---|---|---|---|---|
| LR applied | 8.1e-4 | 6.1e-4 | 5.1e-4 | 4.1e-4 | 2.0e-4 | 1.7e-7 |
| Dev exact | 45.0% | 50.6% | 53.4% | 51.4% | 53.0% | **55.0%** |
| Dev CER | 0.204 | 0.166 | 0.170 | 0.166 | 0.156 | **0.154** |

- **LR trace:** `results.json` → `training.lr_applied_at_update` (1e-5 at update 1, 1e-3 at 100–101, then linear to
  1.7e-7 at 6,000), matching the registered plan.
- **Curve:** dev CER stalled between 2,400 and 3,600 while the LR was high, then improved as it decayed. The last
  checkpoint was best, so the curve may not have plateaued; this is one seed.

## 3. Selection and freeze (`reports/frozen.json`, commit 11dc318)
- **Rule (registered):** lowest dev greedy CER, then highest dev exact, then the earlier update. Update 6,000 wins on
  CER (0.1542 vs 0.1559 at 4,800). Checkpoint sha256 `566a673e…982b`.
- **Frozen before scoring and before any test output:** greedy dev/eval predictions of E12, E11, E07 and E08 with
  file hashes; decoding hash `9115a497…` (greedy, `num_beams=1`, `max_new_tokens=121`).

## 4. Historical diagnostics (mined references, exploratory only)
> Eval has been examined repeatedly and its references are unreviewed web-mined spellings. E11's owner review found
> 28 of 72 reference "misses" acceptable. These numbers do not measure phonetic quality and triggered no tuning or
> further run.

| Split | E07 | E11 | **E12** | E08 (historical) |
|---|---|---|---|---|
| Dev exact (253 items) | 52.2% | 53.0% | **55.0%** | 54.2% |
| Dev CER | 0.186 | 0.157 | **0.154** | 0.151 |
| Eval exact (228 items) | 49.9% | 52.3% | **53.0%** | 52.4% |
| Eval CER | 0.178 | 0.163 | **0.160** | 0.161 |

Paired family bootstrap (10,000, seed 20261027; points, 95% CI):

| Contrast | Eval exact | Eval CER reduction | Dev exact | Dev CER reduction |
|---|---|---|---|---|
| E12 − E11 | +0.7 [−4.1, +5.4] | +0.3 [−1.3, +2.0] | +2.0 [−2.4, +6.4] | +0.3 [−1.6, +2.2] |
| E12 − E07 (historical) | +3.2 [−2.3, +8.6] | +2.0 [−0.4, +4.5] | +2.8 [−3.2, +8.8] | +3.2 [+0.2, +6.3] |
| E12 − E08 (historical) | +0.6 [−4.8, +5.9] | +0.2 [−1.9, +2.2] | +0.8 [−4.4, +6.0] | −0.3 [−2.6, +1.9] |

- **Item flips E11 → E12:** dev 18 gained / 13 lost; eval 15 / 13. Most change is churn, not a net gain.
- **Error kinds (eval, wrong items):** vowel-only differences stay the largest kind (E12 52, E11 47).
  Consonant additions fall (6 vs 9).
- **Missing endings:** eval 5 (E11 4, E07 8); dev 4 (E11 3).
- **Length bins (eval exact):** ≤ 4 letters 55.2% (E11 54.0%), 5–8 50.4% (50.8%), ≥ 9 77.8% (66.7%, 9 items).
- **Operational:** every output is non-empty and ends normally; no repetition. One dev output fails the
  orthography check (E07 also has one).

**Owner dev judgments as coverage (E11 review, 60 words, diagnosis only).** E12's greedy spelling matches an
already-judged spelling on 46 of 60 words: 36 acceptable, 9 not acceptable, 1 uncertain. The other 14 are
**unknown**: new spellings that were never judged. Absence from the accepted set is not rejection. For comparison,
E11 39 / 19 / 2 (fully covered). This cannot rank the models and was not used for selection.

## 5. Blinded test
- **Outputs** (`test/predictions.json`, hashes; files local in `data/e12/test/`): greedy E12 (`03baa2a7…`) and E11
  (`d17e26e8…`) on the 120 frozen inputs, identical processing. No empty, truncated or repeated output; 120
  distinct spellings each.
- **Batches** (`test/batches.json`, hashes): `data/e12/test/batch_1.md` … `batch_4.md` (readable) and `.tsv`
  (fill-in templates); 30 words each, 185 spellings, seeded A/B labels, one spelling when both agree (55 words).
  Shown: word, CMUdict reading, spellings. Hidden: model identity, scores, references. The origin key
  (`origin_key.json`) is read only by the importer.
- **How to answer:** `test/README.md`.
- **Score:** `CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e12.py import`. It rejects unknown ids, edited spellings, invalid values, conflicting duplicates
  and blanks; it scores only when all 185 spellings have an explicit answer.
- **Analysis (predeclared, `TEST_ANALYSIS`):** E12 − E11 confirmed-acceptable share on 120 items; counts,
  uncertainty rates, paired table, family bootstrap, exact McNemar on clear discordant pairs, two sensitivity
  analyses. A non-significant difference is not equivalence.

### 5.1 Result (`reports/test_scores.json`, judgments local in `data/e12/test/`)
- **Judging:** the owner, on the local review page (`scripts/e12_review.py`), 2026-10-30 09:20–09:39 UTC;
  all 185 spellings answered explicitly (197 clicks including changes); single judge, blinded, not independent.
  Judgment file hashes are in `test_scores.json`.

| | E12 | E11 |
|---|---|---|
| Acceptable | 77 | 77 |
| Not acceptable | 39 | 40 |
| Uncertain | 4 | 3 |
| **Confirmed-acceptable share (primary)** | **64.2%** | **64.2%** |

- **Primary:** E12 − E11 = 0.0 points, family bootstrap 95% CI [−8.3, +8.3] (10,000, seed 20261027).
- **Paired:** both acceptable 63, E12 only 14, E11 only 14, neither 29. Exact McNemar on 14 vs 14 clear
  discordant pairs: p = 1.0.
- **Sensitivity:** uncertain counted as acceptable +0.8 points (n = 120); items with any uncertain excluded 0.0
  (n = 115). Uncertainty rate 4.2% of items.
- **Agreement:** on the 55 words where both models wrote the same spelling, 50 were acceptable. On the 65 where they
  differ, each model was acceptable on 27.

**Exploratory breakdown (not predeclared; small cells, no test).** Acceptable E12 / E11:

| Group | n | E12 | E11 |
|---|---|---|---|
| Ordinary | 40 | 27 | 27 |
| Irregular spelling | 40 | 19 | 14 |
| Names | 40 | 31 | 36 |
| ≤ 4 letters | 32 | 24 | 28 |
| 5–8 letters | 42 | 30 | 27 |
| ≥ 9 letters | 46 | 23 | 22 |

The longer schedule may trade some names and short words for irregular words, but with 40 items per group these
differences are within noise and were not tested.

### 5.2 Reading
- The longer schedule changed about half the spellings (65 of 120 differ) without changing how many are acceptable.
  Together with the historical diagnostics (§4), there is no evidence that more optimisation of this recipe on R
  improves quality. One seed; this does not rule out a small effect.
- Irregular-spelling words are the weakest group for both models (35–48% acceptable), against 68% for ordinary words
  and 78–90% for names.
- No replication is recommended on this evidence: neither the predeclared result nor the diagnostics warrant it.

## 6. Limitations
- One seed; dev-41-scale noise (about ±5 words per retrain) is comparable to every historical difference above.
- Selection used the mined-reference CER, a weak surrogate for phonetic quality.
- The test judge is the owner: legitimate but single and not independent. The test sample is declared, not
  universal.

## 7. Reproduce
```bash
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
.venv/bin/python scripts/e12.py verify      # start checks
.venv/bin/python scripts/e12.py testset     # frozen test inputs (idempotent)
.venv/bin/python scripts/e12.py register
.venv/bin/python scripts/e12.py train       # resumes safely from last.pt; refuses a mismatched config
.venv/bin/python scripts/e12.py freeze
.venv/bin/python scripts/e12.py evaluate
.venv/bin/python scripts/e12.py testgen
.venv/bin/python scripts/e12.py batches
.venv/bin/python scripts/e12.py import      # scores once judgments_<n>.tsv are complete
```
