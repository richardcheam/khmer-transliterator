# Changelog

## khmer-translit-byt5-e11 0.2.0 (local release candidate, 2026-10-31)
- **New adapter:** E11 `r_s0`, update 3,000 (checkpoint `286292a4…`), trained on arm R only.
  - **Training data:** 4,764 rows from mined FineWeb-2/Wikipedia pairs and ALT-attested spellings, with no AI Mode
    labels.
  - **Initialisation:** a fresh adapter on the pinned base.
  - **Replaces:** the E08 `rs_s0` adapter of `khmer-translit-byt5-0.2.0`, which was trained with 384 AI Mode rows.
    That bundle stays as it was.
- **Rebuilt from E11:** the adapter weights, checksums, training-data counts, evaluation tables, examples, parity
  fixtures and provenance records were all rebuilt from E11's own files. None was copied from the E10 bundle.
- **Evaluation:** two blinded 120-family owner tests (64.2% and 75.0% confirmed acceptable, two different samples)
  plus a candidate-coverage diagnostic on irregular development words.
- **Package code:** the code is unchanged (`khmer-translit` 0.2.0, the `khmer_translit.byt5` backend). The wheel
  is rebuilt from the current tree.
- **Isolated install:** verified with CPU-only PyTorch from the official index, with no borrowed research
  environment. The E10 check used a `.pth` bridge.
