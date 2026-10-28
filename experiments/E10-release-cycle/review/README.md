# E10 dev-review batch (prepared; no judgments collected)

- **What:** 60 E03 **dev** words, each with 2–4 candidate Khmer spellings, plus a `NONE` row.
  - 40 are representative; 20 are cases where E07 and E08 greedy disagree. The two groups are reported
    separately.
- **Hidden:** which system or reference produced each candidate.
- **Not a fresh test:** dev is the checkpoint-selection set and was examined before; 8 of these words appeared in
  E09's inspected sample.
- **Files (local, `data/e10/review/`, because they contain mined references):**
  - `review_batch.json`;
  - `judgments_template.tsv` (fill in the `judgment` column);
  - `origin_key.json` (do **not** open while reviewing).
  - The hashes are in `review_manifest.json`.

## How to review (policy: `docs/PHONETIC_POLICY.md`)
- **For each candidate row,** write one of `acceptable`, `not_acceptable` or `uncertain`, for the English reading
  shown in `pronunciation`.
- **If no candidate is acceptable,** write `none_acceptable` on the word's `NONE` row. Otherwise leave that row
  blank.
- **Spellings:** never type a new spelling or edit a candidate.

Save the file as `data/e10/review/judgments.tsv`, and next to it `judgments.provenance.json`:
```json
{"track": "independent_human", "reviewer": "<role, not a name>", "khmer_literate": true,
 "blind_to_origins": true, "date": "YYYY-MM-DD", "policy": "docs/PHONETIC_POLICY.md"}
```
Use `"track": "ai_assisted"` for AI-drafted judgments; those are never reported as independent.

## Import, validation and scoring
```bash
.venv/bin/python scripts/e10.py import --judgments data/e10/review/judgments.tsv
```
- **Refused:** missing items, unknown ids or labels, invalid values, conflicting duplicates, edited candidates,
  `none_acceptable` together with an `acceptable` candidate, and a missing provenance file.
- **Scores:** per hidden origin and per sample, in `reports/review_scores.json`.
