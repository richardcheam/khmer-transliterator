# Model card — B (`models/model_b`), served for names and romanised Khmer

| | |
|---|---|
| **Served for** | Unknown Latin words that are not English dictionary words (`translit.word_tag` ≠ "en") and have no f/q/w/x/z: romanised Khmer (*knhom* → ខ្ញុំ) and names (*sophea* → សុភា) |
| **Since** | D138 (owner blinded A/B vs arm A: 9 / 2 / 1), 2026-10-12 |
| **Weights** | `models/model_b/translit.npz`, sha256 `e2ece918b82d06c4efc9e51333527e6eac48742428f2ad3caae83dffadd5cb58`; 2,576,147 parameters |
| **Experiment** | `experiments/E00-history/` (arm B, D137–D138) |
| **Releasable** | **No**: trained on the Chhunneng pairs, whose origin is undocumented; re-ranks with `known_targets.txt` (also Chhunneng) |

## Model
Character-level Transformer encoder–decoder (d 192, 3 + 3 layers, 4 heads, FF 512, dropout 0.1), letters-only input
(≤ 32), NumPy inference with beam 4, then `translit.pick`: the best candidate that is a known Khmer word within 3.0
log-prob of the top (known = Google lexicon + `known_targets.txt`).

## Training
`translit_train.py train --extra data/extra_train.tsv --out models/model_b` (seed 0, 40 epochs, AdamW 1e-3,
label smoothing 0.1, batch 128); then `export`. Data: the Chhunneng train split (24,184 pairs) + the filtered
web-mined pairs (35,291): 59,475 pairs. Best validation exact (Chhunneng val, greedy): 0.482. Single seed, laptop
CPU, ~114 min. Log: `models/model_b/train_stdout.log`, `train_log.json`.

## Results on benchmark v1 (`model_b.eval.json`; re-ranked exact, raw beam top-1 in brackets)
| romanised_test | names_test2 | pick_test | dev41 | list_test |
|---|---|---|---|---|
| **70.0%** (52.9%) | **48.0%** (38.6%) | 28.7% (27.3%) | 48.8% (39.0%) | 49.6% (42.4%) |

## Limits
- The romanised test is Chhunneng's own test split: the same typing style B was trained on, so 70% overstates
  real-world romanised reading.
- Weaker than E on English words; the router sends English dictionary words to E.
- Single seed: differences of a few points between models trained like this are within run noise.
