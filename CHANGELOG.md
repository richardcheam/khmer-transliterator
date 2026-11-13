# Changelog

## 0.3.0
- **Open licences:** the code is now MIT (0.2.x: PolyForm Noncommercial 1.0.0); the published model weights are
  CC BY-SA 4.0 (before: CC BY-NC-SA 4.0); the training data is CC BY-SA 4.0.
- `Transliterator.from_pretrained()` and `ByT5Transliterator.from_pretrained()` without a path download the
  published model on first use (pinned commit, sha256-checked), with the optional extra `khmer-translit[hub]`;
  `khmer-translit predict` no longer needs `--model`.
- The package holds only the English phonetic models (`english`, `nnet`, `byt5`, `phonetic_source`, `cli`);
  `--backend small` is the default name of the small model (`e02` still works).
- Training code and the exact training data of both models are in the repository (`TRAINING.md`).
- The models are unchanged; predictions are identical to 0.2.0.

## 0.2.0
- The small model and the ByT5 adapter backend, released with the models on Hugging Face (PolyForm Noncommercial).
