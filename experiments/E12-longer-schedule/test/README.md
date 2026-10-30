# E12 blinded test: instructions for the owner

**What it is.**
- **Words:** 120 English words and names, sampled and frozen before training (`input_manifest.jsonl`). None of
  them is in any training, dev or review set.
- **Spellings:** each word shows its intended US reading (CMUdict ARPAbet) and one or two Khmer spellings, labelled
  A and B. Labels are shuffled per word, and nothing shows which model wrote which.
- **Agreement:** when both models wrote the same spelling, only one is shown. Your answer then counts for both.

**Files** (local only, in `data/e12/test/`):

| File | Content |
|---|---|
| `batch_1.md` … `batch_4.md` | Readable batches of 30 words |
| `batch_1.tsv` … `batch_4.tsv` | The same as fill-in templates (leave these unchanged) |
| `origin_key.json` | Which model wrote which label. Read only by the scoring step |

**How to judge** (`docs/PHONETIC_POLICY.md`).
- **The question:** would a Khmer reader, reading the spelling aloud, say the shown English reading?
- **Per spelling:** answer `acceptable`, `not_acceptable` or `uncertain`. Both, one or neither may be acceptable.
- **Approximations:** an approximation is fine (policy rules 4–6). A missing or added syllable, or a sound that
  changes the word, is not acceptable (rule 7).
- **Answer every spelling explicitly.** A skipped spelling is missing, not accepted. Scoring waits until every
  spelling has an answer.
- **No typing:** never type a Khmer spelling.

**Three ways to answer.**
0. **Review page (quickest):** `.venv/bin/python scripts/e12_review.py --reviewer "Richard"`, then open
   http://127.0.0.1:8766. One word per screen. Keys: spelling A `1` ok / `2` no / `3` unsure, spelling B `4` / `5` /
   `6`, `Y` all ok, `N` none ok. It moves to the next word when every spelling has an answer. Each click is
   saved to `judgments_<n>.tsv` (plus `review_log.jsonl` and `judgments.provenance.json`). The page never reads the
   origin key.
1. **In chat:** reply per item, for example `t001 A ok B no`, `t002 ok` (one spelling shown), or `t003 A ? B ok`.
   The answers are copied verbatim into `judgments_<n>.tsv`. Anything unclear is asked back, never guessed.
2. **Directly:** copy `batch_<n>.tsv` to `judgments_<n>.tsv` and fill the `judgment` column. Then add
   `judgments.provenance.json` (who, when, policy, blinded, not independent).

**Score it:**
```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e12.py import
```
- **Incomplete:** while answers are missing, `import` reports `incomplete` with the missing ids.
- **Complete and valid:** it writes `reports/test_scores.json` with the predeclared analysis (PLAN.md, "Test
  analysis").
