# E03 report — English phones under dictionary-assessable supervision

**In one paragraph.**
- **The repair:** the miner's `khmer_like` rule threw away every English–Khmer pair whose Khmer side was a known
  word. Repairing it recovered 11,377 pairs. After E02's unchanged filters and the frozen consonant screen, 2,967
  dictionary-assessable, consonant-screened (DACS) pairs remained.
- **Training:** letters vs letters + English phones on those pairs, at three paired seeds.
- **Quality proxy:** no detectable effect. On the new eval partition, phones − letters = +0.60 points, 95% paired
  family-bootstrap interval [−2.40, +3.75].
- **Mechanism:** the phone model stays sensitive to the phones (34% of top-1 outputs change under another word's
  phones). But the changes rarely move its consonants toward the given phones: net +1.4% [−0.8, +3.5]. Most changes
  are vowel or diacritic edits, and few repeat across seeds.
- **Verdict:** cleaner, dictionary-checkable supervision did not make this small model use pronunciation input in
  a detectable way. The references are unqualified, so nothing here is a product-quality claim, and the product
  default is unchanged.

| | |
|---|---|
| Registration | `c1b7eca` (PLAN, config, frozen splits and evaluation, data audit), before any E03 training; base `0e5daf3` |
| Data | `data/e03/` (local): `manifest.json`, `candidates.jsonl` (the full per-candidate audit), splits; audit counts `reports/data_audit.json`, overlaps `reports/overlap.json` |
| Runs | `models/e03/{letters,phones}_s{0,1,2}` (local); ledger `run_ledger.json` |
| Results | `results.json`; per-item `results/quality_per_item.jsonl`, `results/mechanism_per_item.jsonl` |
| Compute | 6/6 attempts (0 failed), 0.56 of 12 h of training; evaluation 5.6 min (run 3 times, see §6); CPU, 6 threads, one job at a time; no GPU, downloads or paid calls |

## Q1. What selection error was repaired?
**The error.** `scripts/translit_mine.py` labels a pair `khmer_like` if **either** of two conditions holds, and
E02 dropped every `khmer_like` pair:
- its Latin side is in the romanised-Khmer list;
- its Khmer side is a known word (the Google lexicon, or `known_targets.txt`).

The second condition removed exactly the pairs whose Khmer spelling has a dictionary reading, which included
established loans. That is why the E02 diagnostics found 0% exact-lookup coverage.

**The repair** (`e03_data.classify`, `recovery_reason`; E02's build untouched):
- The two conditions are recomputed separately. Recomputed, they reproduce the stored class for all 65,422 mined
  pairs.
- A known Khmer target is no longer a reason to reject: 11,377 rows recovered.
- Latin-side romanised Khmer stays excluded, as declared: 1,507 rows.
- The romanised-Khmer detector is a heuristic word list, and it hits ordinary English words. 549 rows (196 Latin
  forms, e.g. *internet*, *post*, *email*, *real*, *son*) are recorded as `latin_romanised_collision`. They are
  excluded under the declared policy, not declared Khmer.
- Reconciliation: counting both conditions together gives exactly the diagnostics' 8,725 pairs with a CMUdict word
  and an exact reading; excluding the romanised-Latin rows leaves 7,714.

**Regression tests** (`tests/test_e03.py`, all pass):
- an English pair with a lexicon-headword target is recovered;
- a romanised-Khmer input is still excluded, and a collision is recorded;
- the extra "account" (គណនី), "gold" (មាស) and "South" (ខាងត្បូង) components are flagged;
- held-out family and target exclusions still work;
- the filter and decision counts equal the rows actually written;
- the frozen screen functions are byte-identical to `e4c0857`.

## Q2. What assessable data survived, and what uncertainty remains?
**Filters**, in E02's order, applied to the 11,377 recovered rows:

| Filter | Rows |
|---|---|
| noise filter (D135) | 3,870 |
| no CMUdict reading | 2,368 |
| related to a held-out family | 755 |
| target equal to a held-out spelling | 131 |
| passed | **4,253** |

**What happened to the 4,253:**
- 4,239 have an exact Google reading (the **assessable track**); 14 do not and stay outside it.
- **Screen** (the frozen E02-diagnostics consonant comparison): 3,218 compatible evidence, 1,017 suspected
  conflict, 4 unknown.
- **Quarantine** (kept in `candidates.jsonl`, never trained):
  - 98 `established_misalignment` (compound tail);
  - 920 `unresolved_conflict`;
  - 4 `unknown_screen`.
- A further 250 training pairs were removed because their target is accepted in dev or eval.
- **Retained: 2,967 pairs, 2,485 words.**
  - Several targets of one word are kept together as alternatives: 356 words have 2 targets, 63 have 3 (e.g.
    *andy* អេនឌី / អែនឌី / អ៊ែនឌី).
  - 137 words have both retained and quarantined targets.

**Sources:** FineWeb-2 2,643; both sources 296; Wikipedia 28. CMUdict-ambiguous: 278. Possible-translation flag
(rule similarity): 7.

**Splits** by English lexical family:

| Partition | Pairs | Words | Families |
|---|---|---|---|
| train | 2,348 (93 train_val) | 1,964 | 1,917 |
| dev | 323 | 276 | 270 |
| eval | 296 | 245 | 239 |

- Primary eval items, excluding ambiguous readings: 228 (222 families).
- No family crosses splits, and no training target is a dev/eval answer (both tested).

**Coverage and vowels:**
- Every retained pair has a CMUdict reading and an exact Khmer reading.
- Vowels are reported, never screened. Of the retained pairs, 3,008 have as many Khmer syllables as English
  vowels, 169 one fewer, 35 more.

**Held-out sets:**
- An isolated step mechanically read E02's references (all tracks), the v1 English panels, the reserve's Latin
  words and E01's list-validation words. Only sets and counts left it: 1,312 words, 1,908 stems, 674 targets.
- No reference was inspected, scored or used for selection. The reserve's answers were not read.

**Overlap with history** (`reports/overlap.json`, eval partition):
- every one of its 283 targets is a Google-lexicon word, and so appeared in E02's P2G data;
- 256 targets and 206 of the 245 words are in E01's mined `extra_train.tsv`, which historical models B/E also
  trained on;
- 122 of its words have *other* targets in E02's training data;
- 37 targets are in the A/B/E training targets;
- 89 words are in the speech repo's loanword list.

E03's own models saw none of this, but historical models' scores on this partition would not be clean.

**Uncertainty that remains** (post-hoc findings, `reports/posthoc_quarantine.json`; data unchanged):
- **The compound detector over-flags.** 87 of the 98 "established misalignments" have a one-syllable lexicon tail.
  Most of those are doubled letters or glides (អានណា *anna*, អាប់ឌុលឡា *abdulla*, អាលីយ៉ា *aaliyah*), not
  compounds. The clear compounds are a handful: *america* + ឡាទីន "Latin", *arabia* + សាអូឌីត "Saudi".
- **The exact-reading track penalises Khmer's coda h.** 207 of the 920 unresolved conflicts have no other problem
  than English s/z read as Khmer h. The frozen adaptation list lacks that rule (e.g. *ace* → អេស "ʔ ee h").
- **So the quarantine removed several hundred probably acceptable pairs.** Both arms train on the same rows, so
  this costs data, not fairness.
- **The screen is a proxy.** It shares rules with the mechanism reader, and it does not judge vowels.
- **References:** mined spellings with no independent review.

## Q3. Does phone input improve reference agreement on the same repaired dataset?
**No detectable difference.**

| Set (role) | Letters s0/s1/s2 | Phones s0/s1/s2 | Paired gain per seed | Mean [95% CI] |
|---|---|---|---|---|
| **eval primary** (registered) | 41.0 / 42.5 / 41.1 | 43.2 / 42.1 / 41.1 | +2.18 / −0.38 / 0.00 | **+0.60 [−2.40, +3.75]** |
| dev primary (used for checkpoint selection) | 47.0 / 45.4 / 46.2 | 44.6 / 47.8 / 45.8 | −2.4 / +2.4 / −0.4 | −0.13 [−3.07, +2.93] |
| E02 dev (owner list, 86 families; secondary) | 16.3 / 14.0 / 16.3 | 11.6 / 9.3 / 14.0 | −4.65 / −4.65 / −2.33 | −3.88 [−7.75, −0.39] |

Raw beam-4 top-1 macro over families, %. No re-ranker: every E03 reference is a lexicon word, so one would leak.

- **Eval details:** top-4 letters 68.3 / 68.5 / 69.4, phones 64.3 / 69.5 / 64.3. CER difference (phones −
  letters) −0.37 points [−1.9, +1.1].
- **Evidence fields** (E02's +5 threshold, for continuity): `positive_effect_supported` false;
  `meaningful_gain_ruled_out_conditional` true; `practically_equivalent_conditional` true. These are conditional on
  these six models and on unqualified references.
- **On E02's dev words, phones are worse in every seed.** These are harder, owner-list spellings, outside the
  training distribution. It is a secondary set, and these words were already examined in the diagnostics.
- **Checks:** measurement controls pass; export parity passes for all six (gap ≤ 8e-6); pairing holds (identical
  initial-state and batch-order hashes per seed).
- Training lasted 720 steps with a 500-step warm-up, as declared; best epochs were 22–34.

## Q4. Does phone input change pronunciation sensitivity?
**It does change outputs, but not in a way that follows the given sounds.** The letters models are an exact
negative control: 0 changes, identical candidates, scores equal within 1e-14.

Phones arm, eval partition, 244 donors, mean over seeds, 95% family-bootstrap intervals:
- **Donor phones** (another word's reading):
  - top-1 changes for 34.3% [29.3, 37.9];
  - only 20 words change in all three seeds;
  - top-4 overlap 0.50–0.56;
  - the default top-1 is still in the top 4 for 87–91%;
  - mean change −0.09 to −0.12 nats per character.
- **Registered mechanism measure, net consonant steering toward the donor: +1.37% [−0.77, +3.50].**
  - Closer / farther by seed: 10/1, 10/9, 8/8 of 244.
  - Of the changed outputs, 49/73, 59/88 and 58/90 keep their consonants (vowel or diacritic edits).
  - Syllable movement can be assessed for only 16–30 changes per seed, and is mostly "same".
- **`<no_pron>`:** changes 30.8% [26.2, 34.7]; 24 words change in all three seeds.
- **Legitimate alternative readings** (21 cases): change 28.6% [13.7, 44.1]; net steering +6.3% [−3.9, +12.8].
  The sample is too small to conclude anything.
- **E02 dev** (secondary): donor changes 46.5%, net steering +2.5% [−1.1, +6.2].
- **Reader coverage:** 99–100% of outputs readable; 54–59% of eval outputs are lexicon words.
- **Context only** (different data): the E02 diagnostics found 33–45% top-1 change and 2–7% of donor cases moving
  closer. E03's 3–4% moving closer is no stronger, so the repaired supervision did not create steering. No
  cross-experiment difference is attributed to the repair alone.

Sensitivity is not correctness, and resisting a deliberately wrong pronunciation is not a defect.

## Q5. What evidence is still missing for a product-quality claim?
- **Independently reviewed references** under the phonetic policy, with the accepted alternatives. Both the E02
  benchmark and the E03 eval partition lack this; the E03 references are mined and screened by rules related to
  the analysis.
- **A validated Khmer reader** (consonants and vowels) or human judgements. Steering and screening currently share
  one rule family, and vowels are unassessed.
- **Better coverage** of what the screen excluded: the s → h coda, doubled letters and glides. Re-admitting those
  would be a new registered data version, not a change to this one.
- **More training signal than 2,255 pairs.** The model learns spelling patterns from letters; whether phones would
  matter at a larger scale is untested here.
- **Product default:** unchanged (E02 `letters_s0`). Serving is untouched; no promotion follows from a mechanism
  study.

## 6. Deviations and corrections (after registration)
1. **The letters-control check was too strict.** It compared beam scores exactly, and scores differ by ≤ 1.2e-14
   between batch rows from floating-point accumulation. Texts and order were identical in every case.
   - The check now requires identical ids and texts and scores within 1e-9, and records the measured maximum.
   - Rerunning the deterministic evaluation gave identical quality results. Mechanism counts were identical, and
     10 log-prob means differed by ≤ 3e-17.
2. **Bootstrap intervals for alternative readings** keyed items by id alone, which dropped extra readings (17
   instead of 21). They are now keyed by (id, reading) and the evaluation was rerun. Nothing else changed.
3. **Post-hoc quarantine analysis** (§2): descriptive only; no data, rule or result changed.

No arm, seed, threshold, reference or schedule was changed.

## Verification and commands
- **Tests:**
  - `CUDA_VISIBLE_DEVICES= WANDB_MODE=disabled .venv/bin/python -m unittest discover -s tests`: 93 tests OK;
  - the E03 tests alone (`tests.test_e03`): 11 OK;
  - speech repo tests: OK.
  - Tests check behaviour, not linguistic quality.
- **Export checks:** `models/e03/*/parity.json`, all `ok`.
- **Rerun**, from the repository root:
```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e03_data.py exclusions   # isolated held-out sets
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e03_data.py build
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e03_data.py overlap
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e03_data.py check
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e03_eval.py freeze      # only for a new registration
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e03_run.py --config experiments/E03-assessable-phones/config.yaml --resume
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 .venv/bin/python scripts/e03_eval.py evaluate
```
