# Model card — E (`models/model_e`), served for English dictionary words

| | |
|---|---|
| **Served for** | Unknown Latin words that are English dictionary words (`translit.word_tag` = "en": lower-case SCOWL entries), read the Khmer way (*safety* → សេហ្វធី, *coffee* → កាហ្វេ) |
| **Since** | D150 (owner blinded A/B, E routed vs served: 14 / 8 / 8), 2026-10-14 |
| **Weights** | `models/model_e/translit.npz`, sha256 `8ac1d5d5b01f4daae8ef7e12f578705ea0008768a990426ae07cfcf15d97d831`; 2,587,475 parameters |
| **Experiment** | `experiments/E00-history/` (arm E, D148–D150) |
| **Releasable** | **No**: trained on the Chhunneng pairs (undocumented origin) and LLM drafts; re-ranks with `known_targets.txt` |

## Model
As B, but **phone-aware input** (`--phones`, max input 48): letters + "|" + the CMUdict pronunciation, one
private-use character per phone and stress digit (`translit.source`). Beam 4 + `translit.pick` with the served
known list.

## Training
`translit_train.py train --extra data/extra_train.tsv --loans data/llm_drafts.tsv --loan-repeat 4 --use-list
--phones --max-in 48 --out models/model_e` (seed 0, 40 epochs, as B); then `export`. Data: Chhunneng train +
mined pairs + the loanword list's train split + 2,399 LLM drafts, the loans repeated ×4: 75,136 pairs. Validation:
Chhunneng val + list val (1,825 pairs); best exact 0.471. Single seed, laptop CPU, ~192 min. Log:
`models/model_e/train_stdout.log`, `train_log.json`.

## Results on benchmark v1 (`model_e.eval.json`; re-ranked exact, raw beam top-1 in brackets)
| romanised_test | names_test2 | pick_test | dev41 | list_test |
|---|---|---|---|---|
| 68.4% (53.3%) | 46.6% (39.5%) | **30.7%** (30.7%) | **61.0%** (56.1%) | 36.0% (33.6%) |

## Limits
- English spelling is convention-heavy (vowel length, series marks, ័រ vs ឺរ); exact match counts only one accepted
  spelling, and the owner's ear is the final judge.
- dev-41's targets partly come from E's earlier outputs, which favours E (development set, not a test).
- The LLM drafts copy LLM conventions, which are not always the owner's.
- In serving, the router decides which words reach E: the full served system (B + E + rules, "as if unlisted")
  scores 55.2% on the list test, against E alone 36.0% (E01 `results.json`, `served_reference`).
