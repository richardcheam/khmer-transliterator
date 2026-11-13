# Test sets

What each set is and what it was used for. A set that guided a choice gives an optimistic number for the model it
chose. Scored by `pipeline/evaluate.py`.

| File | Size | What it is | Used for during development |
|---|---|---|---|
| `small_final.jsonl` | 191 words (165 + 26 flagged) | English loanwords from the author's spelling list whose Khmer spelling the author added, confirmed, chose or corrected; split by word family (65% of families) | the small model's one final evaluation, read once after all choices; the closest to an untouched test |
| `small_proxy.jsonl` | 400 words | loanwords with LLM-suggested spellings, of which the author spot-checked a sample | scored once with `final`, as a separate exploratory track |
| `small_dev.jsonl` | 103 words (86 unflagged) | the other 35% of the author-checked families | choosing the small model's epoch (also in `../training-data/small/dev.jsonl`) |
| `byt5_eval.jsonl` | 245 words (228 unflagged) | English words with web-mined reference spellings | scoring several ByT5 experiments; the released adapter was chosen by human review, not by this set |
| `../training-data/byt5/dev.jsonl` | 276 words (253 unflagged) | the same kind of words | choosing the adapter's update |

**Flags.** `pronunciation_ambiguous` (the word has several common English readings) and `possible_translation`
(the reference looks like a translation, not a sound spelling) are left out of the main score and reported apart.

**What a score means.** Exact agreement with the reference spellings, averaged within each word family and then
over families. A spelling that sounds right but is written differently counts as wrong, so these numbers understate
usefulness. The human acceptability judgments in the report (one judge, the author) belong to the outputs that were
judged and are not included; a retrained model would need new judgments.

**Licence.** CC BY-SA 4.0 (`../training-data/NOTICE.md`); web-mined references keep their sources' terms.
