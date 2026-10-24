# Data card

The data lives in `data/` (not in git). **`manifest.json`** pins the sha256 and size of every file (212 files on
2026-10-24); `scripts/data_manifest.py check` verifies it. **`build/`** holds the summaries written by the data
scripts. Raw-source licences and pins: `data/SOURCES.md`.

**Rule:** data changes only through a script, then `data_manifest.py write`, then a commit with an entry in the
change log below. An experiment's provenance records the hashes of the files it read.

## Training sources
| File | Rows | What | Origin / licence | Made by | Known problems |
|---|---|---|---|---|---|
| `pairs.tsv` | 28,067 (train 24,184 / val 1,643 / test 2,240, grouped by Khmer word) | Romanised Khmer → Khmer: mostly dictionary words, 301 informal chat spellings. No names, no English | Chhunneng repo @ fe8028f1, MIT declared, **origin of the pairs undocumented** | `translit_data.py` | **Not releasable.** One consistent typing style, so its test split flatters models trained on it |
| `roman/pairs.tsv` | 118,308 | Clean substitute: 61,032 Google-lexicon headwords romanised by rules (`romanise.py`) + one variant each; Chhunneng test targets excluded; split by word hash (val 5%) | Google lexicon CC BY 4.0 + our rules | `translit_roman.py build` | Rule typing matches real typing only roughly (top-3 exact 677 / 2,240 on the Chhunneng test; character similarity 0.86) |
| `mined_kmwiki.tsv` | 3,532 | Latin ↔ Khmer word pairs from brackets in Khmer Wikipedia ("ឈ្មោះ (Name)") | kmwiki 2026-10-01, CC BY-SA 4.0 / GFDL | `translit_mine.py --source kmwiki` | Automatic alignment; some mismatches |
| `mined_fineweb.tsv` | 63,805 | The same from FineWeb-2 `khm_Khmr` shard 0 | ODC-By 1.0 + Common Crawl terms | `translit_mine.py --source fineweb` | Noisy; several Khmer forms per Latin word |
| `extra_train.tsv` | 35,291 | Filtered union of the mined pairs, minus loanword-list words and Chhunneng test targets | as above | `translit_extra.py` | Contained 99 of the 150 pick-test words (D163): filter test sets before training (E01 does) |
| `list_split.json` | train 1,556 / val 182 / test 125 | Frozen split (D146) of the speech repo's loanword list (`configs/latin_khmer.tsv`, read via `khmer_translit.loanwords`) | owner-verified where marked (`build/latin_audit.json`) | `translit_train.py` (once) | Covers only the words listed at D146 |
| `llm_drafts.tsv`, `llm_drafts_r2.tsv` | 2,399 / 1,856 | English words spelled by LLM agents in the owner's style (owner spot check 39/40) | LLM output (`llm_suggested`) | archived drafting scripts | Copies LLM conventions; r2 did not help (D151) |
| `synth/examples.tsv` / `synth/test.tsv` | 150 / 150 | The owner's 300 spelling picks, split into training examples and a test-only half | owner-verified | archived picker | Answers come mostly from E's top 5 |
| `synth/synthetic_clean.tsv` | 3,976 | Gemini spellings for web English words, few-shot from the examples | LLM output | archived | Did not help (D164) |

## Evaluation data
| File | What | Rule |
|---|---|---|
| `benchmark/v1/*.tsv` | The frozen test panels (see `../benchmarks/v1.json`) | Never trained on, never tuned on; scored through `khmer_translit.benchmark` only |
| `names_dev.tsv` / `names_test2.tsv` | Wikidata Cambodian person names (CC0): dev 184 names / test2 131 | dev for development, test2 only through the benchmark |
| `e2/reserve.json` | 120 English words, answers never opened | **Sealed** for a final owner A/B; never scored |
| `e2/dev.json` | dev-41 | Development only |

## Inference resources
| File | What | Licence |
|---|---|---|
| `lexicon/km_lexicon.tsv` | Google Khmer pronunciation lexicon @ bc72dd65: 62,175 headwords | CC BY 4.0 (`lexicon/ATTRIBUTION.md`) |
| `en_common_words.txt` | SCOWL lower-case words (63,910): the English router | SCOWL notice |
| `src/cmudict/cmudict.dict` | CMUdict pronunciations (phone-aware model E) | BSD-style |
| `../models/model/known_targets.txt` | Chhunneng train/val Khmer targets, part of the served re-ranking list | Chhunneng: **must go for a release** |

## Change log
- 2026-10-24: manifest created (212 files), data moved from the speech repo's `data/translit/` (D176); benchmark
  v1 frozen.
