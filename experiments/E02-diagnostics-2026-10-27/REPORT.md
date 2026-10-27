# E02 diagnostics (2026-10-27): do the training targets follow the phones, and do the models use them?

A diagnostic phase after E02. No training, no new model, no download, no TTS, no serving change. **E02's
registered outcome is unchanged:**
- evidence `inconclusive`: phones vs letters +1.65 points [−1.65, +4.94] on an unqualified benchmark;
- letters baseline retained.

Nothing here re-scores the final track, and the upper bound below +5 is not read as evidence against phonetic
modelling in general.

| | |
|---|---|
| Source commit | `ac6bf6f` (clean tree). Script and config frozen at `e4c0857`; sensitivity sample frozen at `37df40a`, both before any outcome was seen |
| Config, hashes | `config.json`: every input and checkpoint sha256, the settings and the forbidden paths |
| Code | `scripts/e02_diagnostics.py` (subcommands `compat`, `freeze`, `sensitivity`, `posthoc`); tests in `tests/test_e02_diagnostics.py` |
| Compute | under 5 minutes of CPU (audit 7 s, sensitivity 104 s, post-hoc and tests under 1 min); one job at a time, GPU hidden |

Reading guide: **Observation** means a measured count from the files. **Hypothesis** means an interpretation that
these diagnostics do not establish.

## 1. What could be assessed (Q1)

**Pronunciation supplied to the model.**
- All 10,031 training rows have the first CMUdict reading. It is identical to what inference uses for the same
  word: 10,031/10,031 rows, checked by the audit and by a test.
- 1,030 rows (10.3%) belong to words with several readings that differ in sounds, not just stress.

**Khmer pronunciation of the target (preferred track: exact Google-lexicon lookup):** 0 of 10,031 hits.
- This is by construction, not chance. The miner labels a pair `khmer_like` when its Khmer side is a known Khmer
  word, a Google-lexicon or training headword (`scripts/translit_mine.py`).
- E02 dropped every `khmer_like` row, a filter aimed at romanised Cambodian names.
- So the filter also removed every training pair whose target the lexicon can read. Established loans such as
  ប៉ារីស (Paris) and កាហ្វេ (coffee) are lexicon headwords, so pairs like these cannot be in E02's training data.
- No Khmer G2P is installed in this repo or the speech repo. `khmer_translit.romanise` writes popular Latin typing,
  not pronunciation, so it was not used.

**Fallback track: consonants read from the Khmer spelling** (`km_skeleton`, fixed rules).
- It read 10,031/10,031 targets.
- **Validation against the Google lexicon's own transcriptions:** the consonant classes agree exactly for 88.6% of
  30,980 held-out Khmer headwords (90.1% of the 8,831 with loan markers).
- The rules were adjusted only on the other, calibration half (88.9%) before any training row was compared.
- **So roughly 1 reading in 10 is the reader's own error.** Most of those errors are silent letters in learned
  Pali/Sanskrit words.
- This track compares **consonants only**. Vowels, where French or original-language conventions mostly show, are
  not assessed.

**Second evidence track: Google's Khmer reading of the English word itself** (a Latin headword in the lexicon).
- Available for 1,196 rows.
- It describes the word's Khmer convention, not the specific target.

**Shared representation.**
- Both alphabets are mapped to broad consonant classes from each symbol's IPA value: Google's ɓ ɗ c ɲ j w,
  aspirates th/ph/kh/ch, loan-only f/g/z; ARPAbet's standard values (`EN_CONS`, `GOOGLE_CONS`, `KM_LETTER`).
- No ARPAbet token is treated as equal to a Google symbol except through these classes.
- A fixed list of known loan adaptations counts as compatible (`ADAPT`): f→ph, v→w, θ→t/s, ʃ→s/c, dʒ→s, ŋ→n,
  coda r and final-cluster reduction.
- Anything else counts as one "unexplained operation". Categories come from these explicit rules, not from a tuned
  distance threshold.

**Audit categories** (flags, not labels; not an error rate).

| Category | Rows | Share |
|---|---|---|
| `compatible_evidence` (consonants align with the supplied reading, with listed adaptations) | 7,013 | 69.9% |
| `suspected_conflict` (≥ 1 unexplained consonant operation) | 3,018 | 30.1% |
| `unknown` (no reading at all) | 0 (all on the fallback track; **100% unknown on the preferred exact-lookup track**) | — |

Discrepancy levels (unexplained operations): 0: 7,013 · 1: 2,607 · 2: 372 · 3+: 39.

**Breakdowns:**

| Group | Rows | % compatible evidence |
|---|---|---|
| FineWeb-2 only | 9,338 | 70.3 |
| Wikipedia only | 224 | 68.8 |
| Both sources | 469 | 63.5 |
| ≤ 4 letters | 1,077 | 77.4 |
| 5–8 letters | 7,608 | 70.9 |
| ≥ 9 letters | 1,346 | 58.4 |
| Common English word (SCOWL list) | 1,525 | 69.2 |
| Other words / names | 8,506 | 70.0 |
| CMUdict reading ambiguous | 1,030 | 66.3 |

**Files** (folder `compat/`):
- `summary.json`;
- `sample.tsv`: 200 rows, seed 20261027, stratified over the categories and levels 0/1/2/3+, the ambiguous
  readings, the several-target words and the Google-reading rows;
- the full per-row table `training_pairs.jsonl` (sha256 `97cd36cb…ae2d0`). It stays local and git-ignored,
  because it copies about 10k web-mined pairs.

## 2. Evidence of conflicting supervision (Q2)

These are observations; the share of real errors among them is **not estimated**, because no flag was adjudicated.

1. **One input, several targets.**
   - 2,340 words (5,455 rows, 54% of training) have 2–3 different Khmer targets for the same letters and the same
     phones.
   - For 961 of those words, the targets disagree on their consonants, not just on vowels or diacritics.
   - Within those words, at least some targets cannot all follow the one supplied reading. Each disagreement is
     not necessarily an error.
2. **Contamination at 3+ operations** (39 rows): translations or compounds glued onto the loan.
   - *class* → កណ្តាល ("middle");
   - *block* → ប្លុកគណនី ("block account");
   - *button* → ប៊ូតុងមាស ("gold button");
   - *carolina* → ការ៉ូលីណាខាងត្បូង ("South Carolina");
   - *contract* → កុងត្រាឆ្លាតវៃ ("smart contract");
   - *acid* → អាស្កូប៊ិក (ascorbic).
   - Also stray prefixes such as *aaron* → ងអារ៉ុន. The heuristic flag "extra consonant at the start" (311 rows)
     mixes such artefacts with doubled letters, so it is not a count of artefacts.
3. **Non-English readings, conflicting with the supplied English phones** (operation type "consonant class
   change", 933 operations, post-hoc):
   - *becerra/cerna* (CMUdict CH) → s;
   - *gerardo/gelb/geng* (CMUdict JH) → hard g, ហ្គ;
   - French nasals: *alain* → អាឡាំង, n written ង (155 operations).
4. **Reading mismatch:**
   - 45 rows fit another CMUdict reading better than the supplied first one;
   - 93 rows fit Google's Khmer reading of the word where that reading departs from the English consonants.
   - On the Google track, 355 of the 1,196 rows (30%) are words whose Khmer convention departs from the English
     consonants. In 147 of those rows the target follows the Khmer convention.
5. **Not conflicts, or not shown to be** (post-hoc types). These explain 1,075 of the 3,018 flagged rows entirely:
   - doubled English letters written twice (អាប់បាស for *abbas*): the target follows the spelling;
   - a y glide between vowels (អាណាតូលីយ៉ា);
   - English y not written;
   - ហ្ក = g, a gap in the frozen reader;
   - final k written ច after ិ. The lexicon confirms this one is real Khmer pronunciation: អាឡិច is "ʔ aa . l ə c".

   These types are post hoc, written after seeing the flags, and the categories were **not** changed. The
   remaining 1,943 flagged rows contain some other unexplained operation.

## 3. Do the phone models respond to phones? (Q3)

**Frozen design** (`sensitivity/frozen.json`, sha256 in `frozen.sha256`, committed before running):
- the 95 dev words with a dictionary reading, letters fixed;
- conditions:
  - **default** phones;
  - **`<no_pron>`**;
  - **donor**: another dev word's phones, phone count within ±1. 94 donors; 1 word skipped for lack of one;
  - **alternative**: the word's other CMUdict readings; 21 cases, all differing in sounds, not just stress.
- Beam 4, top 4, the actual `encode_source` and the NumPy beam search.

**Checks:**
- every encoded source holds exactly the intended phones with the letters unchanged (all 7 checkpoints);
- teacher-forced rescoring reproduces the beam score (largest gap 7e-15);
- the three letters-only controls give identical ids and outputs under every condition (0 changes).

| Checkpoint | Donor: top-1 changed | `<no_pron>`: changed | Alternative: changed | Donor: default top-1 still in top 4 | Donor: mean Δ log-prob per token |
|---|---|---|---|---|---|
| phones_s0 | 34/94 | 31/95 | 3/21 | 86/94 | −0.054 |
| phones_s1 | 31/94 | 25/95 | 7/21 | 89/94 | −0.029 |
| phones_s2 | 42/94 | 34/95 | 7/21 | 86/94 | −0.074 |
| phones_p2g_s0 | 21/94 | 20/95 | 2/21 | 93/94 | −0.014 |
| letters_s0/s1/s2 | 0 | 0 | 0 | all | 0 |

**What this shows (observations):**
- **The phone models are sensitive:** contradictory phones change the top-1 output for a third to almost half of
  the words, and removing the phones for a quarter to a third.
- **But the response is shallow:**
  - The original top-1 usually stays among the 4 candidates.
  - The score change is about −0.03 to −0.07 nats per token, against a typical −0.27 to −0.32 per token for the
    default output.
  - Only 7 words change under donor phones in all three seeds; 36 change in just one.
- **Post hoc: does a changed output follow the donor's sounds?**
  - Comparing consonants, outputs moved closer to the donor's phones in 7, 3 and 2 of the 34, 31 and 42 changed
    cases, and farther in 3, 4 and 4.
  - So most changes are in vowels or diacritics, not the consonants the donor dictates.
- **Legitimate alternative readings:**
  - Mostly reduced-vowel variants (AH0/IH0), and the outputs barely move.
  - *job* with the name reading JH OW1 B moves ចប → ចូប in 2 of 3 seeds, a vowel change in the expected direction.
  - *record* with R EH1 K ER0 D gives រ៉េក័រ in one seed.
  - *manager* switches to a hard-g ហ្គ although both readings have JH, so that change is not phonetically
    motivated.
  - No independent evidence says which adaptation is correct; none is claimed.
- **The P2G pilot is the least sensitive** on every measure.
- **Dev reference agreement** (secondary, unqualified, out of 95), default → donor / `<no_pron>`:
  - phones_s0: 18 → 12 / 14;
  - phones_s1: 16 → 17 / 15;
  - phones_s2: 19 → 12 / 17;
  - phones_p2g_s0: 18 → 16 / 17.

**Hypothesis** (not established here): the phone models use the phones as a weak, seed-dependent cue for vowel and
diacritic choices, while the letters drive the consonants. That would fit training targets that follow the phones
only partly (§2).

## 4. What remains unresolved (Q4)

- **No Khmer reading for any training target on the preferred track.** The fallback reads consonants from the
  spelling, with about 11% disagreement against Google on held-out headwords, and does not read vowels at all.
- **The adaptation list is a hypothesis.** Unlisted legitimate adaptations count as "unexplained" (see the post-hoc
  types), and a listed one may sometimes hide a real error.
- **CMUdict is US English, and names are often read in their own language.** Whether a name "should" follow the
  English reading is a policy question, not something these tools can settle.
- **The dev references are unqualified.** Agreement changes in §3 say nothing about phonetic correctness.
- **Unread outputs:** the post-hoc "toward the donor" check would skip outputs the reader cannot parse (none were
  skipped here), and it inherits the reader's errors.

## 5. Recommended next experiment (Q5), not launched

**Phone-consistent supervision, measured by phone steering.** The observed bottleneck is supervision whose Khmer
pronunciation cannot be checked and partly conflicts with the phones, together with models that barely steer by
the phones. The recommended test:

1. **Data.** Rebuild the supervised set with the reason for the 0% coverage removed: re-admit mined pairs excluded
   only because their **Khmer** side is a known word.
   - Romanised-Khmer **Latin** inputs stay excluded.
   - All E02 leak and noise filters still apply.
   - There are 8,725 such pairs (5,536 words) before those filters (`compat/excluded_assessable_pool.json`).
   - With an exact Google reading of each target, the frozen comparison flags 70% `compatible_evidence`.
   - Pre-register a consistency filter on that exact track (keep `compatible_evidence` only).
2. **Design.** letters vs phones, three paired seeds, trained on this phone-consistent set.
3. **Primary outcome** (does not need qualified references): this phase's frozen steering protocol. Under donor or
   alternative phones, the rate at which the output's consonants move toward the given phones.
   - Pre-register a minimum effect over the E02 phone models' baseline: 2–7% of donor cases (5–21% of the changed
     outputs).
4. **Secondary outcome:** dev agreement only.

What it would show: whether consistent supervision makes the model *use* the phones. That is the mechanism E02
could not test. A product-quality claim would still need independent review of the references.

## Tests and commands

- **New diagnostic tests:** `CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest tests.test_e02_diagnostics -v`:
  15 tests, OK, exit 0. They cover:
  - phone replacement with fixed letters;
  - training/inference encoding equality;
  - deterministic, valid donors;
  - alternative readings;
  - missing and ambiguous Khmer lookup;
  - an unreadable target giving `unknown`;
  - per-item/checkpoint alignment on a small real run;
  - tampering with the frozen sample being refused;
  - no forbidden path opened, by a file-open spy during a real audit and a real run.
- **Full suite:** `CUDA_VISIBLE_DEVICES= WANDB_MODE=disabled .venv/bin/python -m unittest discover -s tests`: 82
  tests OK.
- **Speech repo:** `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests`: OK.
- Unit tests establish implementation behaviour, not linguistic validity.

Rerun, from the repository root, in order:
```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 .venv/bin/python scripts/e02_diagnostics.py compat
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 .venv/bin/python scripts/e02_diagnostics.py freeze        # rewrites the frozen sample (same content for the same inputs)
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 .venv/bin/python scripts/e02_diagnostics.py sensitivity
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e02_diagnostics.py posthoc
```
- The excluded-pool count was a one-off read-only computation in this session. Its output and examples are in
  `compat/excluded_assessable_pool.json`.

## Preservation

- **What this phase read:**
  - only the inputs in `config.json`;
  - the dev references;
  - the 7 saved checkpoints, loaded with their recorded sha256.
- **Not read:** the final references, E02 result files, benchmark panels and the reserve. Reads go through
  `guarded`, and the spy tests confirm it.
- **Nothing in E02 changed:** results, checkpoints, benchmark files and manifests are untouched.
- **What is in git:**
  - `sensitivity/frozen.json` contains the accepted spellings of the 95 sampled dev items, as E02's committed
    audit sample already did for 40 dev items.
  - The full per-row training table is local only.
