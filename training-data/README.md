# Training data of the two published models

The exact files the published models were trained and selected on. `TRAINING.md` (repository root) uses them.

## `small/`: the small model
| File | Rows | Columns | Used for |
|---|---|---|---|
| `train.tsv` | 9,735 train + 296 train_val | `word`, `khmer`, `part` | training pairs; `train_val` pairs are held back and only measure loss |
| `dev.jsonl` | 103 words | `id`, `word`, `family`, `accepted`, `preferred`, `flags` | choosing the epoch to keep |
| `vocab.json` | | source tokens, Khmer characters | the model's input and output alphabets |

The training pairs were mined automatically from Khmer web text where a Khmer spelling is followed by the English
word in brackets, e.g. "ចនសុន (Johnson)": from FineWeb-2 (Khmer subset, one shard) and Khmer Wikipedia (dump of
2026-10-01). A pair was kept only if the English word is in the CMU Pronouncing Dictionary, does not look like
romanised Khmer, passes a noise filter and is unrelated to every word used for evaluation. No LLM-generated
spellings were used for training. An older in-house transliteration model helped choose and align the mined
pairs; its outputs were never used as training labels. The development words are English loanwords in the author's checked Khmer
spelling list; words whose reading is ambiguous (`pronunciation_ambiguous`) are left out of the selection score.

## `byt5/`: the ByT5 adapter
| File | Rows | Columns | Used for |
|---|---|---|---|
| `train.tsv` | 4,764 | `word`, `khmer`, `source` (`web` 3,547, `alt` 1,217) | training pairs |
| `dev.jsonl` | 276 words | as above | choosing the update to keep |

`web` pairs were mined from FineWeb-2 and Khmer Wikipedia as above; `alt` pairs are names and loanwords spelled in
the human Khmer translations of the ALT Parallel Corpus. All pairs were checked automatically against the English
pronunciation; no LLM-generated or hand-written spellings were used. The development words were mined the same
way.

**Not here:** the evaluation sets behind the reported results stay unpublished, so that they can still serve as
tests.

## Licence and sources
See `NOTICE.md`.
