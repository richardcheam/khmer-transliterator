# E11 report: ByT5-small on the real-data pool R, no AI Mode labels (2026-10-29)

**Outcome: uncertain alternative** on the registered reference-agreement rule (`results.json` → `decision`).
The owner's blinded review (§5) prefers E11 to E07 (14 words vs 4).
- **Eval:** E11 agrees with the historical references 2.5 points more often than E07, but the 95% interval
  includes 0 (+2.5 [−2.7, +7.7]).
- **E08 without AI Mode:** E11 matches E08 (eval −0.1 [−4.7, +4.6]) without any Google AI Mode label.
- **Phonetic quality:** the owner judged all 60 blinded words. E11 was acceptable on 65%, E07 on 48% and E08 on 63%.
- **Nothing was published, packaged, served, merged or tagged.**

## 1. Data (`reports/data_checks.json`)
E08's frozen arm R was reused unchanged (`data/e08/training/arm_R.tsv`, sha256 `d00706cf…ec88`). It was not
relabelled or filtered.

| Check | Result |
|---|---|
| Optimisation rows / words | 4,764 / 3,585, plus 93 train-val rows |
| Sources | E03 control 2,255 · recovered mined 1,292 · ALT-attested (CC BY 4.0) 1,217 |
| Control rows = E07's training rows | yes |
| Exact / NFC duplicates | 0 / 0 |
| Pairs with several sources | 0, so keeping one provenance per pair lost nothing (all sources in `data/e11/pair_sources.json`, local) |
| `google_ai_mode_synthetic` rows | 0; R ⊂ R+S, and R+S − R = the 384 AI Mode rows |
| R pairs identical to an AI Mode spelling | 0 |
| Pairs in E08 quarantine files | 0 (of 2,054 quarantined) |
| Dev/eval, protected-family, train-val-family overlap | 0 |
| Byte-limit rejections | 0 |
| Words with several spellings | 840 (max 8) |

**Limitation.** The mined and ALT rows carry uncertain vowel quality, and nothing here checks it. Vowel-only
differences are the largest error kind for every model (§4.4), and some of them are likely reference noise rather
than model errors.

## 2. Run (`registration.json`, frozen in c234de2 before training)
- **Recipe:** E07's recipe unchanged (cfg hash `f5d7cf92…`): LoRA r16/α32 on attention + FFN, AdamW 1e-3 with
  100 warm-up and linear decay, batch 32, 3,000 updates.
- **Initialisation:** the pinned `google/byt5-small` base, a fresh adapter and seed 0. Nothing came from E08 or E10.
- **Sampling:** row-level, as in E07, giving 96,000 example draws, or 20.15 epochs. E07 made 42.6 epochs over its
  smaller set.
  - **Draw shares:** control 47.4%, recovered 27.1%, ALT 25.5%.
  - **Draws per word:** min 18, median 20, max 161.
- **Resources:**
  - CPU, 6 threads, one job.
  - Memory guard: 3 consecutive readings with RSS > 6.5 GiB or MemAvailable < 0.4 GiB trigger a resumable stop. It
    never triggered.
  - Peak RSS was 4,603 MB. The lowest available memory was 2,671 MB.
- **Time:** 4.60 h active training in one segment, with no idle gap, at about 5.5 s/update. Freezing took 66 s and
  scoring 1 s.

| Update | 600 | 1,200 | 1,800 | 2,400 | **3,000** |
|---|---|---|---|---|---|
| Dev exact | 31.8% | 49.0% | 50.2% | 52.6% | **53.0%** |
| Dev CER | 0.283 | 0.190 | 0.173 | 0.161 | **0.157** |

- **Selected:** update 3,000, by dev exact, then CER (sha256 `286292a4…`).
- **Under-training:** the curve was still rising when the learning rate reached 0, the same pattern as E08 (E10
  §1).

## 3. Freeze (`reports/frozen.json`, commit 14858ee)
- **Committed before any score was computed:** greedy dev and eval predictions for E07 (`24c9d200…`), E11 and E08
  (`d5da8c29…`), with file hashes.
- **Decoding:** greedy (`num_beams=1`, `max_new_tokens=121`, decoding hash `9115a497…`).
- **Hash check:** `evaluate` verifies the hashes before scoring.

## 4. Results (historical mined-reference agreement only)
> **Caveat.** Eval has been examined repeatedly since E03, and its references are unreviewed web-mined
> spellings. These numbers measure agreement with those references, not phonetic accuracy. Nothing was tuned
> after evaluation.

### 4.1 Exact agreement and CER (greedy, family-macro)
| Split | E07 | **E11** | E08 (historical) |
|---|---|---|---|
| Dev (253 items, 250 families) exact | 52.2% | **53.0%** | 54.2% |
| Dev CER | 0.186 | **0.157** | 0.151 |
| Eval (228 items, 222 families) exact | 49.9% | **52.3%** | 52.4% |
| Eval CER | 0.178 | **0.163** | 0.161 |

Dev was used for checkpoint selection, so the eval rows are the primary result.

### 4.2 Paired family bootstrap (10,000 resamples, seed 20261027; points, 95% CI)
| Contrast | Eval exact | Eval CER reduction | Dev exact | Dev CER reduction |
|---|---|---|---|---|
| **E11 − E07 (primary)** | **+2.5 [−2.7, +7.7]** | +1.7 [−0.7, +4.1] | +0.8 [−4.0, +6.0] | +2.9 [+0.2, +5.9] |
| E08 − E07 (reference) | +2.6 [−2.4, +7.5] | +1.8 [−0.6, +4.4] | +2.0 [−3.2, +7.2] | +3.5 [+1.0, +6.0] |
| E11 − E08 (reference) | −0.1 [−4.7, +4.6] | −0.2 [−1.9, +1.5] | −1.2 [−5.2, +3.2] | −0.6 [−2.6, +1.5] |

**Item outcomes, E07 vs E11:**
- Eval: both right 100 · only E07 15 · only E11 20 · both wrong 93.
- Dev: 114 · 19 · 21 · 99.

The two models disagree on 35 eval items and 40 dev items in total. The net difference is 5 eval items, which is
within noise.

### 4.3 Validity and endings (all prediction ids: dev 276, eval 245)
| | E07 | E11 | E08 |
|---|---|---|---|
| Empty or no output | 0 | 0 | 0 |
| Invalid orthography (dev / eval) | 1 / 0 | 0 / 0 | 0 / 0 |
| Repeated output (dev / eval) | 1 / 0 | 0 / 0 | 0 / 0 |
| Truncated (no end token) | 0 | 0 | 0 |
| Missing ending (dev / eval) | 5 / 8 | **3 / 4** | 4 / 6 |

A missing ending means the consonant skeleton is a proper prefix of the nearest reference's.

### 4.4 Error kinds of wrong top-1 items (eval)
| Kind | E07 | E11 | E08 |
|---|---|---|---|
| Vowels only | 46 | 47 | 53 |
| Consonants changed | 11 | 16 | 12 |
| Consonant dropped | 15 | 8 | 10 |
| Consonant added | 12 | 9 | 8 |
| Marks only | 8 | 8 | 7 |
| Both | 21 | 20 | 18 |

- **E11 vs E07:** E11 drops and adds fewer consonants (17 vs 27). It changes more consonants (16 vs 11).
- **Vowels:** vowel-only differences are unchanged and remain the largest class.

### 4.5 English word length (eval exact)
| Bin | Items | E07 | E11 | E08 |
|---|---|---|---|---|
| ≤ 4 letters | 87 | 52.9% | 54.0% | 57.5% |
| 5–8 letters | 132 | 48.0% | 50.8% | 50.8% |
| ≥ 9 letters | 9 | 55.6% | 66.7% | 33.3% |

The ≥ 9 bin has 9 items, which is too few to interpret.

## 5. Review (`review/review_manifest.json`; files local in `data/e11/review/`)
- **Batch:** 60 dev words, 40 representative and 20 disagreement, reported separately. The candidates are the
  109 distinct greedy outputs of E07, E11 and E08, at most 3 per word, with origins hidden. Each word shows its
  intended English reading (ARPAbet) and a `vowels_or_marks` or `consonants_or_syllables` tag.
- **Representative sample:** 21 words with one shared output, 10 differing in vowels or marks, 9 in consonants or
  syllables.
- **Disagreement sample:** 8 vowel or mark differences, 12 consonant or syllable differences.
- **Status (update, 2026-10-29):** the project owner judged all 60 words, blinded (`reports/review_scores.json`;
  the file and its provenance are local in `data/e11/review/`).
  - The owner replied in chat. I transcribed the replies using three rules the owner confirmed: candidates not
    named are not acceptable; "ok" without a label means all candidates; skipped single-spelling words are
    acceptable. The ambiguous replies were clarified with the owner: the five skipped two-spelling words, r49, r58
    and r59. "Not sure" was recorded as uncertain for every candidate (r47, r57). No spelling was typed.
  - The owner is **not independent** of the project, and the words are dev items (the selection set).
  - The owner noted for *visa* that ហ្ស would be better for the "z" sound. This is kept as a comment only.

**Share of words where the model's greedy spelling was acceptable** (uncertain counts as not acceptable):
| Sample | E07 | E11 | E08 |
|---|---|---|---|
| Representative (40) | 24 (60%) | **29 (72.5%)** | 30 (75%) |
| Disagreement (20) | 5 (25%) | **10 (50%)** | 8 (40%) |
| All 60 | 29 (48%) | **39 (65%)** | 38 (63%) |

**Paired by word** (exact two-sided sign test over the words where exactly one of the two models is acceptable):
| Contrast | Only the first acceptable | Only the second acceptable | p |
|---|---|---|---|
| E11 vs E07 | 14 | 4 | 0.031 |
| E08 vs E07 | 12 | 3 | 0.035 |
| E11 vs E08 | 8 | 7 | 1.0 |

E11 is preferred over E07 by this judge, and it equals E08. Caveats: one non-independent judge, dev words, and a
test chosen after the data were seen. This is evidence, not a registered result.

**Owner judgment vs the old reference spellings** (the 109 judged candidate spellings):
| | Owner: acceptable | Owner: not acceptable | Uncertain |
|---|---|---|---|
| Matches the reference | 28 | 8 | 1 |
| Differs from the reference | **28** | 41 | 3 |

- **Differs but acceptable:** 28 of 72 spellings that "missed" the reference (39%) were acceptable, for example
  *cameroon* ខេមឺរ៉ុន, *bolt* បូល, *band* បេន, *phi* ហ្វី and *acer* អេស៊ើ. Many were vowel differences.
- **Matches but not acceptable:** 8 of 37 spellings that matched the reference (22%) were not, for example
  *band* បាន, *bolt* បុល, *phi* ភី and *acer* អេសឺ.
- **So exact agreement with the mined references is a weak measure of quality here.** A large part of the
  "vowel errors" in §4.4 are reference noise, not model errors. That is why reference agreement showed
  E11 − E07 as uncertain, while the judge separates the two models.

## 6. Decision
| Axis | Finding |
|---|---|
| **Reference agreement** | Uncertain alternative to E07: eval +2.5 [−2.7, +7.7], CER +1.7 [−0.7, +4.1], dev not below E07. Equal to E08 within noise. Fewer missing endings, no invalid or repeated outputs. |
| **Phonetic quality** | Owner judgments on all 60 dev words (single judge, not independent): E11 acceptable on 65% (E07 48%, E08 63%). Paired, E11 is preferred over E07 on 14 words vs 4 (sign test p = 0.031), and equals E08 (8 vs 7). This is supporting evidence, not a registered result. |
| **Permissions** | Improved over E08: no Google AI Mode labels, so that blocker is gone. Still open: CC BY-SA (Wikipedia) and page rights in the mined E03 and recovered pairs; ALT is CC BY 4.0 (attribution); ByT5 base Apache 2.0. The adapter has no licence and is not published. |

**What improved:**
- E11 reaches E08's agreement without synthetic labels. The 384 AI Mode rows contributed nothing measurable, so
  E11 is the cleaner ByT5 candidate.
- Dev CER improved over E07, the only interval that excludes 0.
- Fewer dropped or added consonants and missing endings.

**What did not:**
- No contrast against E07 is detectable on eval.
- Vowel-only mismatches did not decrease.
- The model is still under-trained at 3,000 updates.
- Phonetic quality is not measured.

**Second seed:** recommended only after review, not launched. A second seed tests seed stability, but with
222 eval families the item-level interval would stay about ±5 points. It cannot by itself turn +2.5 into a
detectable result.

**Strongest evidence for the next quality step:**
1. **Better references, not more training.** Vowel-only differences are 40–50% of all mismatches for every model,
   and they did not respond to doubling the real data. The owner's review (§5) shows that 39% of spellings that
   missed the reference were acceptable, and 22% that matched it were not. The mined references cannot separate
   these models. The next step is a small reviewed reference set: acceptable/not judgments on model outputs,
   which needs no typing. Score future candidates on that instead of exact agreement.
2. **Under-training.** Dev was still rising at the final update for E11 and E08 alike. A longer schedule is the
   obvious training-side hypothesis, but it needs a new registered experiment. E10's higher learning rate did not
   help.

## Files
| File | Content |
|---|---|
| `PLAN.md`, `registration.json` | Plan, frozen recipe, exposure, decision rule |
| `reports/data_checks.json` | Data verification |
| `reports/frozen.json`, `predictions/` | Frozen greedy predictions and hashes |
| `results.json`, `results_per_item_{dev,eval}.jsonl` | Scores, contrasts, validity, lengths, error kinds |
| `runs/ledger.jsonl`, `state.json` | Run ledger and step state |
| `review/README.md`, `review/review_manifest.json` | Review instructions and manifest |
| Local only | `models/e11/r_s0/` (weights), `data/e11/` (pair sources, review batch, origin key) |

## Release preparation (D197, 2026-10-31)
- **Script:** `scripts/e11_release.py` (steps identity → cov-select → cov-generate → cov-batches → export → verify →
  bundle → install; `cov-import` after the owner's coverage review).
- **Results:** identity 12/12; adapter export bit-identical; parity 0 mismatches (dev, eval, top-4); isolated
  install 19/19.
- **Bundle:** `dist/khmer-translit-byt5-e11-0.2.0/` (local), documents in `release/byt5-e11/`.
- **Coverage review:** queued (50 new spellings), not yet judged.
- Evidence: `release/{identity,export,verify,bundle,install_check}.json`, `coverage/{frozen,generation,batches}.json`.
