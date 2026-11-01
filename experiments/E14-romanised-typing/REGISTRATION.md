# E14 registration (D205, 2026-11-02, written before any training or any owner-test score)

No model, including served B and the E01 models, has been scored on the owner-typed test before this commit.

## Data (`scripts/e14_train.py data` → `models/e14/manifest.json`)
| Source | Rows in | Kept | Notes |
|---|---|---|---|
| Rule-romanised Google lexicon (train) | 112,581 | 107,617 | as E01 |
| Web-mined name pairs | 35,291 | 33,522 | as E01 |
| Chhunneng pairs, train split (dictionary + informal) | 24,184 | 22,814 | new (D204, MIT, origin unverified) |
| Loanword list train / owner examples / LLM drafts | 1,555 / 150 / 2,399 | 1,430 / 141 / 2,245 | as E01, ×4 |
| Owner typing (train half, 160 words) | 240 | 140 | new, ×4 |
| Owner-checked Gemini spellings (400 words; Gemini 3.6 Flash) | 636 | 502 | new, "LLM-suggested, owner-verified", ×4 |

181,785 spelling pairs per epoch. Side task: Google lexicon Latin pronunciations, as E01.

**Owner rows:** every spelling variant gives its joined row (*puk ke* → `pukke` → ពួកគេ). When the Khmer word splits
into as many known parts as there are typed words, each typed word also gives a row (`puk` → ពួក, `ke` → គេ).

**Leak filter:** E01's rule (`translit_unified.row_hits`) over every held-out set, plus `e14_test`. A training row is
dropped if its Khmer target is one of the 160 owner test words, or its Latin side equals a typed test spelling
(a–z only, spaces removed). Rows that meet a validation set are also dropped, as in E01.

**Validation (checkpoint selection only):** the E01 names and English sets, and for romanised the Chhunneng val
split (1,577 after pruning, real typing) in place of E01's rule-romanised val. Each set is pruned of every evaluation set.

## Model and training
E01's `aux` arm unchanged: d 192, 3 + 3 layers, 4 heads, FF 512; side-task decoder discarded at export. AdamW
lr 1e-3, 40 epochs, batch 128, etc. (`config.yaml`). Threads 12 (E01: 6); a one-epoch smoke run took 10.7 min (about 7 h per seed). Seeds 0, 1, 2, trained one after another.
Seed 0 is followed by a technical check (loss, no NaN, export, data counts) only; no test score is looked at until
all three seeds are trained.

## Evaluation
- **Primary panel:** the owner-typed test, i.e. 160 words with 209 typed spellings. Each spelling is split on spaces, every word
  is transliterated (beam 4, re-ranked) and the outputs are joined, as the speech front end does. A spelling is
  right if the result equals the Khmer word exactly. A word scores the share of its spellings that are right, and
  the panel score is the mean over the 160 words.
- **Re-ranking:** E14 uses the Google lexicon headwords only (the clean, releasable resource). Served B uses its
  served resources (lexicon + `known_targets.txt`), as it runs today.
- **Reference:** served model B. B was trained on the Chhunneng train split, which contains some test words; E14
  has none. The count is reported.
- **Guards (benchmark v1):** `romanised_test` (Chhunneng style) and `names_test2`, E14 vs B.
- **Diagnostics (not in the rule):** `pick_test`, `dev41`, `list_test`; the served router on the owner test; the
  clean E01 aux models on the owner test (the effect of the new data).

## Adoption rule (from exact counts, mean over seeds 0, 1, 2)
E14 goes to the owner's blinded A/B against served B only if all of these hold:
- owner test: E14 − B ≥ **−3 points** (E14 is meant to *replace* B as one model, so matching B is enough);
- `romanised_test`: E14 − B ≥ **−5 points**. E14 loses the Chhunneng rows that meet any test set, and this panel
  favours models trained on the full Chhunneng train split;
- `names_test2`: E14 − B ≥ **−3 points**.

Also reported: every seed, a word bootstrap 95% CI of the primary difference, and words better/worse/tied. If the
rule fails, the result is recorded and nothing changes in serving. There is no recipe change after seeing results.
English words stay with served model E either way.
