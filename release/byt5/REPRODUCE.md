# Reproducing khmer-translit-byt5 0.2.0

All steps run on a laptop CPU (`CUDA_VISIBLE_DEVICES=` hides any GPU). The repository is the `english-phonetic`
branch of `khmer-transliterator`. The data files below are local, so a reproduction needs the owner's copy of
`data/` (they are not redistributed; see `DATA_PROVENANCE.md`).

## Environment
```bash
uv sync                                                         # package + test dependencies
uv pip install -r research/requirements-byt5.txt                # torch, transformers 5.19, safetensors (pinned
                                                                # versions in research/requirements-byt5.lock.txt)
hf download google/byt5-small --revision 68377bdc18a2ffec8a0533fef03b1c513a4dd49d --local-dir models/e06/byt5-small
```

## Inputs (frozen, verified by hash)
| File | sha256 |
|---|---|
| `data/e08/training/arm_RS.tsv` (training pairs, arm R+S) | `03a4988f84cdc97aff6b8356e019d1076c8d09a1ecec4004a2a909ed3e229b0f` |
| Base weights `models/e06/byt5-small/pytorch_model.bin` | recorded in `adapter/adapter_config.json` → `base.weights_sha256` |
| E10 confirmation manifest `experiments/E10-release-cycle/confirmation/input_manifest.jsonl` | `3daec5e6…` (full hash in `confirmation/frozen.json`) |

## The released checkpoint (E08 `rs_s0`)
The release candidate is the E08 R+S run: the E07 recipe (LoRA r16/α32 on attention and FFN, AdamW lr 1e-3,
warm-up 100, linear decay, batch 32, 3,000 updates, seed 0), selected by dev greedy exact match.

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e08_train.py train           # rs_s0; about 5.2 h on 6 threads
```

Selection: update 3,000, dev greedy exact match 0.542, CER 0.1513; checkpoint sha256 `d5da8c29…` (full hash in
`evidence/candidates_frozen.json`). The E10 exploratory candidate A_s0 is reproduced by
`scripts/e10.py train --run A_s0` (recipe `experiments/E10-release-cycle/recipes/A.json`).

## Evaluation (historical track, exploratory)
```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e10.py freeze      # candidates + decoding frozen first
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e10.py evaluate    # dev/eval tables, paired family bootstrap,
                                                                  # confirmation-set diagnostics
```

- Decoding: greedy for the single answer; beam 4 with `early_stopping="never"`, `length_penalty=0` for candidates.
- Bootstrap: paired by word family, 10,000 resamples, seed 20261027.
- Generations are cached by (checkpoint hash, decoding hash, data hash) in `state/e09/cache/`, so a rerun reuses
  them. Delete the cache to regenerate.

## Release artefacts
```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e10_release.py export    # adapter.safetensors + config
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e10_release.py verify    # parity with cached predictions, offline,
                                                                        # refusals, latency, memory
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e10_release.py bundle    # dist/khmer-translit-byt5-0.2.0/
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e10_release.py install   # fresh venv outside the repo, offline
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e10_release.py bundle    # again, to include the install check
.venv/bin/python -m unittest discover -s tests
```

The exported adapter is a deterministic function of the checkpoint, so its sha256 in `SHA256SUMS` should match.
Greedy and beam outputs are deterministic on one machine and thread count. Other CPUs or thread counts can change
rare near-ties.
