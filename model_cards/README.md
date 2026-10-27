# Model cards

One card per model that is or was served: what it is for, the exact weights (sha256), how it was trained, its
results on the frozen benchmark, and its limits. The `*.eval.json` files are produced by
`scripts/evaluate.py models/<name> --known served --out model_cards/<name>.eval.json`.

| Model | Served for | Since | Releasable |
|---|---|---|---|
| [B](model_b.md) | names, romanised Khmer | D138 | no (Chhunneng) |
| [E](model_e.md) | English dictionary words | D150 | no (Chhunneng) |
| [E02 letters](e02_english_phonetic.md) | not served (research preview, English → Khmer phonetic) | D178 | local only (CC BY-SA training pairs) |

A new model gets a card when it is adopted (after the owner's blinded A/B), before `serving.json` changes.
