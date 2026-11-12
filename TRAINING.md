# Training the two models from scratch

| Model | Script | Time |
|---|---|---|
| Small model (2.6M parameters, trained from scratch) | `training/train_small_model.py` | about 25 minutes on a laptop CPU |
| ByT5 adapter (LoRA on a frozen `google/byt5-small`) | `training/train_byt5_adapter.py` | about 45 minutes on a 4 GB laptop GPU (GTX 1050 Ti); several hours on a CPU |

All settings are in `training/config.yaml`; they are the ones the published models were trained with. The data is
in `training-data/` (what each file holds and where it comes from: `training-data/README.md`). No paid service is
needed; the ByT5 adapter downloads the public `google/byt5-small` base (about 1.2 GB) once.

## 1. Set up
Python 3.10–3.12.
```bash
git clone https://github.com/richardcheam/khmer-transliterator.git
cd khmer-transliterator
python -m venv .venv && source .venv/bin/activate
pip install -e '.[byt5]' -r training/requirements.txt \
  --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
# CUDA 12.6 build of PyTorch: it also runs on the CPU and still supports older GPUs (e.g. GTX 10xx).
# Without a GPU you may use https://download.pytorch.org/whl/cpu instead.
python -m unittest discover -s training/tests                    # quick checks of the training code
```

## 2. The small model
```bash
python training/train_small_model.py --seed 0 --out runs/small_s0
```
Each epoch prints the training loss, the loss on 296 held-back pairs and the development score: the share of the
86 development words (loanwords in the author's checked spelling list, ambiguous readings left out) spelled as one
of their accepted spellings, averaged within each word family and then over families. The best of 40 epochs is
kept and exported as `runs/small_s0/model.npz`. The published model (seed 0) kept epoch 27; rerunning this script
on a CPU with seed 0 reproduces it exactly (every weight identical), in about 25 minutes on a laptop.

How it trains: the input is a task token and the word's letters, the output its Khmer spelling, both as characters;
batches of 128 pairs in a seeded order, cross-entropy with label smoothing 0.1, AdamW with a learning rate warming
up over 500 steps and then decaying to 0 along a cosine. On the CPU a run with the same seed and thread count is
exactly repeatable.

Use the result like the published model:
```bash
khmer-translit predict --model runs/small_s0/model.npz --word coffee --top-k 3
```

## 3. The ByT5 adapter
```bash
python training/train_byt5_adapter.py --seed 0 --device cuda --out runs/byt5_s0
# or with a local copy of the base model: --base /path/to/byt5-small
```
ByT5 reads and writes UTF-8 bytes, so the input is the word's bytes and the output the Khmer spelling's bytes. The
base model is frozen; small low-rank layers (LoRA, rank 16) are added beside every attention and feed-forward
projection, and only these are trained: 3,000 updates of 32 pairs, AdamW, learning rate 0.001 with a 100-update
warm-up and a linear decay. At updates 600, 1,200, 1,800, 2,400 and 3,000 the model spells 253 development
words greedily (276 in the file; ambiguous readings left out), and the best point is kept (development top-1, then lower character error). The published adapter
kept update 3,000 (development top-1 0.530, trained on a CPU). A check run of this script on a GTX 1050 Ti took 42
minutes and reached 0.558 at update 3,000; GPU runs are not bit-identical, so some spellings differ. It uses
nearly all of a 4 GB card. Output: `runs/byt5_s0/adapter/`, used like the published one:
```bash
khmer-translit predict --backend byt5 --model runs/byt5_s0/adapter --word coffee --top-k 4
```
If a run stops, add `--resume`.

## Changing things
- **Your own word pairs:** append rows to a copy of `training-data/small/train.tsv` (or `byt5/train.tsv`), point
  `data` in `training/config.yaml` to the copy and retrain. Keep the words you want to test on out of training.
- **Model size, epochs, learning rate, adapter rank:** `training/config.yaml`.

How the models were evaluated and why they are built this way: the model cards and
`report/khmer-transliterator-technical-report.pdf`.
