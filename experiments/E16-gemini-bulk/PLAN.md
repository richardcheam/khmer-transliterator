# E16: E15 plus Gemini spellings for 10,000 common words, filtered automatically (registered D209, 2026-11-03)

Written before the Gemini replies were complete, before the data build and before any training. The owner approved
running the whole pipeline automatically once the data is ready ("go ahead and train when data is ready
automatically").

**Question:** E14 and E15 trail served model B on the owner's chat typing because they lack chat-style examples
(D206, D208). Gemini's spellings were good in the owner's E14 check (at least one kept for 73% of 400 words). Does a
large, unchecked set of them, filtered by fixed rules only, close the gap?

## Data (`scripts/e16.py`, then `scripts/e16_train.py data` → `models/e16/manifest.json`)
- **Word list:** E14's word pool (the Chhunneng chat list, the extra chat list and the Chhunneng 1000-common list),
  then Khmer words in FineWeb-2 frequency order (first 200k documents of the local shard; Google-lexicon words that
  also occur at least 5 times as a stand-alone run). Excluded: all 320 E14 typing words (both halves) and the 400
  owner-checked words. First 10,000 words, in 40 prompts of 250 words.
- **Gemini:** `gemini-3.6-flash`, medium thinking, through the API with the owner's free key; the E14 prompt text
  unchanged (1 to 3 chat spellings per word). Each call is logged in `data/e16/gemini/fetch_log.jsonl`.
- **Reading replies (`e16.read_replies`):** aligned by position when a reply has one line per word. Otherwise
  lines are matched by their Khmer key, unmatched words are skipped, and a reply under 90% matched is skipped.
  Spaced spellings (*rieb chom*) are joined (*riebchom*), as with the owner's spaced answers.
- **Automatic filter (`e16.filter_rows`), no hand checking.** A spelling is dropped if:
  - it is not a–z only, or is longer than 3 × the Khmer length + 2;
  - the owner's typing (train half), the 400 checked words or the Chhunneng chat list map it to another word;
  - Gemini gave it for several words, in which case it is kept only for the most common word.
- **Training mix:** E15's sources unchanged (no rule-romanised rows), plus `gemini_bulk` counted once. Owner rows
  stay repeated ×4. The E14 leak filter applies to the new source too, so rows touching an owner test word or
  spelling, or any benchmark set, are dropped.
- **Completeness:** the pipeline trains only if at least 36 of the 40 replies were received and read. Otherwise it
  stops and reports.

## Model and training
Identical to E14 and E15 (E01 `aux` arm, `config.yaml`): seeds 0, 1, 2, 40 epochs, 12 threads, trained one after
another, with checkpoint selection on the validation sets only.

## Evaluation
- **Primary:** the owner-typed test (160 words, 209 spellings) with E14's scoring. This is its **third use**,
  declared. Nothing was tuned on it, and no E16 word or setting was chosen from it: the prompts exclude all 320
  typing words by construction.
- **Guards:** `romanised_test` and `names_test2` (benchmark v1).
- **References:** served B (the adoption reference); E15 and E14 (three seeds each) and clean E01 on the owner test.
- The 150 fresh test words (`test_frozen.json`) are not used in this experiment; they were never typed.

## Adoption rule (from exact counts, mean over seeds 0, 1, 2; thresholds as E14 and E15)
E16 goes to the owner's blinded A/B against served B only if all of these hold:
- owner test: E16 − B ≥ **−3 points**;
- `romanised_test`: E16 − B ≥ **−5 points**;
- `names_test2`: E16 − B ≥ **−3 points**.

**Data check** (reported, not part of adoption): E16 − E15 on the owner test, with a word bootstrap 95% CI. "Bulk
Gemini data helps" is supported if the mean is ≥ +5 points and the CI excludes 0.

If the rule fails, the result is recorded and serving is unchanged. There is no recipe change after seeing results.

## Run
`scripts/e16_run.sh`: waits for the Gemini fetch, then filter → data → three seeds → score (about 11–12 h).
