# E17: as much LLM chat-spelling data as the free tools allow, Gemini, Claude and GPT (registered D211, 2026-11-06)

Written before any E17 prompt was sent, before the data build and before any training. The owner asked for it
after E16 ("add maximum data as we can then if it looks like we getting performance") and had already approved
automatic training once the data is ready (E16).

**Question:** E16 showed that bulk Gemini spellings, filtered by fixed rules, clearly help (+6.5 points over E15 on
the owner test). Does more of it help further: more words, and a second model (Claude) whose spellings often cover
what Gemini misses (in a check on the owner's 400 reviewed words, Claude had the owner's own spelling for 40% of the
108 words where the owner rejected every Gemini suggestion), plus a third model (GPT, which the owner offered)?

## Data (`scripts/e17.py`, then `scripts/e17_train.py data` → `models/e17/manifest.json`)
- **Words:** E16's 10,000 prompted words, then up to 20,000 new words: E14's chat pool first, then Khmer words from
  the whole local FineWeb-2 shard (1.59M documents) that occur at least 3 times as a stand-alone run and stand
  alone in at least 0.3% of their occurrences (this drops segmentation fragments such as តល). Excluded:
  all 320 owner typing words (both halves), the 400 owner-checked words and the 150 frozen fresh test words
  (checked by the script; it refuses to write a prompt that contains one).
- **Prompt (all E17 calls, `e17.PROMPT`):** E16's instructions made explicit (informal chat spelling, not a formal
  romanisation; a-z only, no spaces; spell as spoken) plus 30 examples of the owner's own typing, taken from the
  training half only (never a test word; those 30 words are excluded from the prompts anyway). E16's Gemini replies
  keep E16's plain prompt.
- **Gemini:** the new words only (E16 already covers the first 10,000), `gemini-3.6-flash`, medium thinking, owner's
  free key, 500 words per call (the free key allows only about 13 to 20 calls a day).
- **Claude:** all words (E16's and the new ones), `claude-opus-5-5`, effort medium, through `claude -p` on the owner's
  claude.ai subscription (no tools, no project files, an empty working folder), 250 words per call.
- **GPT:** all words, `gpt-6.1-sol` (pinned), reasoning effort medium, through `codex exec` on the owner's ChatGPT login
  (read-only sandbox, empty working folder), 250 words per call.
- **Logged per call** (`fetch_log.jsonl` in each folder): CLI version, requested and reported model, thinking or
  effort level, token counts, time, retries.
- **Reading and filtering:** E16's reader (`e16.read_replies`) and automatic filter (`e16.filter_rows`), unchanged.
  The three models' spellings are merged per word (Gemini, then Claude, then GPT, duplicates removed) before
  filtering, so a spelling given for several words is kept only for the most common word across all sources. No hand checking. Which model
  gave each kept spelling is recorded (`data/e17/llm_sources.tsv`).
- **Training mix:** E15's sources (no rule-romanised rows) plus `llm_bulk` (the merged, filtered rows, which
  includes all of E16's Gemini spellings), counted once. Owner rows stay repeated ×4. The E14 leak filter applies.
- **Collection deadline:** collection stops when every prompt is answered or at 2026-11-09 11:00 UTC, whichever is
  first. Gemini prompts still unanswered at the deadline are left out. Training goes ahead only if at least 95% of
  the Claude prompts and 95% of the GPT prompts were answered; otherwise the pipeline stops and reports.

- **Prompts written (before any call):** 16,301 new words (all that pass the rules); Gemini 33 prompts (new words),
  Claude 106 and GPT 106 prompts (26,301 words: E16's 10,000 plus the new ones).

## Model and training
Identical to E14 to E16 (E01 `aux` arm, `config.yaml`), but **one model only (seed 0)**, at the owner's request
("let train once only", to get the result about 9 hours sooner): 40 epochs, 12 threads, checkpoint selection on the
validation sets only. A single model's score moves by a few points from chance alone (E16's three seeds spanned
2.7 points on the owner test), so a borderline result is less certain than in E14 to E16.

## Evaluation
- **Primary:** the owner-typed test (160 words, 209 spellings) with E14's scoring. This is its **fourth use**,
  declared. Nothing was tuned on it, and no E17 word or setting was chosen from it.
- **Guards:** `romanised_test` and `names_test2` (benchmark v1).
- **References:** served B (adoption), E16, E15, E14, clean E01.
- **Registered diagnostic:** the owner test split into the 130 words served B was trained on and the 30 it never
  saw (`b_unseen_owner_test_pct`, all models). Reported, not part of adoption.

## Adoption rule (from exact counts of the seed-0 model; thresholds as E14 to E16)
E17 goes to the owner's blinded A/B against served B only if all of these hold:
- owner test: E17 − B ≥ **−3 points**;
- `romanised_test`: E17 − B ≥ **−5 points**;
- `names_test2`: E17 − B ≥ **−3 points**.

**Data check** (reported, not part of adoption): E17 (seed 0) − E16 (mean of its three seeds) on the owner test,
word bootstrap 95% CI; E17 − E16 seed 0 alone is also reported. "More LLM data
helps further" is supported if the mean is ≥ **+3 points** and the CI excludes 0.

If the rule fails, the result is recorded and serving is unchanged. No recipe change after seeing results. Because
the owner test is now on its fourth use and B was trained on most of its words, a decisive comparison with B needs
the 150 fresh words typed by the owner; that is a separate step, not part of this rule.

## Run
`scripts/e17_run.sh` (detached): Gemini, Claude and GPT collection in parallel, then filter → data → one model →
score (one model). Logs and a DONE marker in `models/e17/`.

## Provenance note
The training rows are LLM outputs (Google Gemini, Anthropic Claude, OpenAI GPT) checked only by fixed rules. Any
release of a model trained on them must say so and respect the providers' terms.
