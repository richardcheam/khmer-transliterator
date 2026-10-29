# E11 blinded comparison: reviewer instructions

**Status: prepared, not collected.** No reviewer has been contacted, and no judgment exists or has been invented.

## What it is
- **Words:** up to 60 English dev words.
  - **Representative:** 40, one per sampled word family.
  - **Disagreement:** 20 where the three compared ByT5 models give different greedy spellings.
  - The two samples are reported separately.
- **Candidates:** for each word, the distinct Khmer spellings the models produced, at most 3. Labels (A, B, C) are
  shuffled per word, and nothing tells you which model wrote which.
- **Intended reading:** each word shows its English reading (US, CMUdict ARPAbet with stress), so the target sound
  is explicit.
- **Differences:** each word is tagged `vowels_or_marks` when the candidates share their consonant letters, or
  `consonants_or_syllables` otherwise, so vowel and syllable choices are visible.
- **No reference spelling is shown:** only model outputs.

## Files (local only, `data/e11/review/`)
| File | Content |
|---|---|
| `review_batch.json` | The items |
| `judgments_template.tsv` | One row per candidate, plus one `NONE` row per word |
| `origin_key.json` | Which model wrote which label. Opened only by the scoring step, after judgments are complete |

## How to judge
Follow `docs/PHONETIC_POLICY.md`: would a Khmer reader, reading this spelling aloud, say the given English reading?
- **Per candidate:** `acceptable`, `not_acceptable` or `uncertain`. Several candidates can be acceptable.
- **`NONE` row:** `none_acceptable` if no candidate is acceptable. Leave it blank otherwise.
- **Nothing to type:** never type a spelling.
- **Provenance file:** save the filled file as `data/e11/review/judgments.tsv` with
  `judgments.tsv.provenance.json` (who judged, when, under which policy, and whether they are independent of the
  project).

Then run `scripts/e11.py import`. It rejects:
- missing or unknown ids;
- edited candidates;
- invalid values;
- duplicates;
- a `NONE` row that contradicts an `acceptable` candidate.

The manifest with counts and hashes is `review_manifest.json`.
