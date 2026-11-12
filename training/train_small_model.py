#!/usr/bin/env python3
"""Train the small model: one English word in, its Khmer phonetic spelling out (NumPy at run time).

    python training/train_small_model.py --seed 0 --out runs/small_s0              # CPU, about an hour
    python training/train_small_model.py --seed 0 --out runs/small_s0 --device cuda
    python training/train_small_model.py --seed 0 --out runs/small_s0 --resume      # after an interruption

Data (`training-data/small/`): `train.tsv` (word, Khmer spelling, part: `train` or `train_val`), `vocab.json` (the
source tokens and Khmer characters) and `dev.jsonl` (development words with their accepted spellings).

The input is `<task:en>` followed by the word's letters, the output its Khmer spelling, both as character ids. Each
epoch visits the training pairs in a seeded order, in batches of 128, with cross-entropy and label smoothing 0.1,
AdamW and a learning rate that warms up over 500 steps and then decays to 0 along a cosine. After each epoch the
model spells every development word greedily; the kept epoch has the highest share of development words spelled
as one of their accepted spellings (averaged within each word family, then over families), ties going to the
lower loss on the held-back `train_val` pairs, then to the earlier epoch. Output: `best.pt`, `last.pt`,
`train_log.json` and `model.npz`, which `khmer_translit.english.Transliterator` loads.

On the CPU, runs with the same seed are exactly repeatable.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import unicodedata
from pathlib import Path
from statistics import mean

import torch
import yaml
from torch import nn

import nets
from khmer_translit import phonetic_source as ps

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / "config.yaml"
EXCLUDED_FLAGS = {"pronunciation_ambiguous", "possible_translation"}   # left out of the selection score


def load_data(folder: Path):
    """`(train pairs, train_val pairs, development items, source vocabulary, Khmer characters)`."""
    rows = [l.split("\t") for l in (folder / "train.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    train = [(r[0], r[1]) for r in rows if r[2] == "train"]
    train_val = [(r[0], r[1]) for r in rows if r[2] == "train_val"]
    dev = [json.loads(l) for l in (folder / "dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    dev = [d for d in dev if not EXCLUDED_FLAGS & set(d["flags"])]
    v = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
    vocab = ps.SourceVocab(v["km_phones"])
    if vocab.tokens != v["src_tokens"]:
        raise ValueError("vocab.json does not match this package's source tokens")
    return train, train_val, dev, vocab, v["tgt_chars"]


def source(word: str, vocab) -> list[int]:
    return ps.encode_source(word, None, "letters", vocab)


def target(khmer: str, chars: str) -> list[int]:
    if len(khmer) > ps.MAX_TGT:
        raise ValueError(f"target longer than {ps.MAX_TGT}: {khmer}")
    return [nets.BOS] + [chars.index(c) + 3 for c in khmer] + [nets.EOS]


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def dev_score(model, dev: list[dict], dev_src: list[list[int]], chars: str) -> tuple[float, int]:
    """(family-averaged share of development words spelled as an accepted spelling, number of words right)."""
    model.eval()
    texts = []
    with torch.no_grad():
        for i in range(0, len(dev_src), 256):
            for ids in nets.greedy(model, nets.pad_rows(dev_src[i:i + 256])):
                texts.append("".join(chars[t - 3] for t in ids if t >= 3))
    model.train()
    hit = {d["id"]: int(nfc(t) in {nfc(a) for a in d["accepted"]}) for d, t in zip(dev, texts)}
    fam: dict[str, list[int]] = {}
    for d in dev:
        fam.setdefault(d["family"], []).append(hit[d["id"]])
    return (mean(mean(v) for v in fam.values()) if fam else 0.0), sum(hit.values())


def mean_loss(model, pairs, loss_fn) -> float:
    model.eval()
    tot = n = 0.0
    with torch.no_grad():
        for i in range(0, len(pairs), 256):
            ch = pairs[i:i + 256]
            S, T = nets.pad_rows([a for a, _ in ch]), nets.pad_rows([b for _, b in ch])
            lg = model(S, T[:, :-1])
            tot += loss_fn(lg.reshape(-1, lg.shape[-1]), T[:, 1:].reshape(-1)).item() * len(ch)
            n += len(ch)
    model.train()
    return tot / max(1, n)


def train(seed: int, out: Path, device: str = "cpu", resume: bool = False, epochs: int | None = None,
          stop_after: int | None = None, cfg: dict | None = None, data=None) -> dict:
    """Train the small model; see the module docstring. `epochs`, `stop_after`, `data` are for tests and pilots."""
    cfg = cfg or yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    c, m = cfg["small_model"], cfg["small_model"]["model"]
    epochs = epochs or c["epochs"]
    torch.set_num_threads(c["threads"])
    tr, tv, dev, vocab, chars = data or load_data(ROOT / cfg["data"]["small"])
    dev_ = torch.device(device)
    tr_tgt = [target(k, chars) for _w, k in tr]
    tv_pairs = [(source(w, vocab), target(k, chars)) for w, k in tv]
    dev_src = [source(d["word"], vocab) for d in dev]
    torch.manual_seed(seed)
    model = nets.Seq2Seq(len(vocab.tokens), len(chars) + 3, ps.MAX_SRC, **m)
    model.to(dev_)
    if dev_.type == "cuda":
        torch.cuda.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = nets.cosine_with_warmup(opt, c["warmup"], epochs * math.ceil(len(tr) / c["batch"]))
    loss_fn = nn.CrossEntropyLoss(ignore_index=nets.PAD, label_smoothing=c["label_smoothing"])
    out.mkdir(parents=True, exist_ok=True)
    st = {"epoch": 0, "step": 0, "best": None, "log": [], "elapsed": 0.0}
    if resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", weights_only=False, map_location="cpu")
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        nets.set_rng_state(ck["torch_rng"], dev_)
        st = ck["state"]
        print(json.dumps({"resumed_from_epoch": st["epoch"]}), flush=True)
    t0 = time.time() - st["elapsed"]
    meta = {"format": "khmer-translit-english-v1", "encoder_version": ps.ENCODER_VERSION, "mode": "letters",
            "seed": seed, "src_tokens": vocab.tokens, "tgt_vocab": chars, "heads": m["heads"], "layers": m["layers"],
            "max_src": ps.MAX_SRC, "max_tgt": ps.MAX_TGT, "beam": 4, "pronunciation": None,
            "input_policy": "one English word; NFKC, lower case, accents removed, apostrophes/hyphens dropped; a-z, "
                            f"<= {ps.MAX_LETTERS} letters"}
    for ep in range(st["epoch"], epochs):
        order = list(range(len(tr)))
        random.Random(f"{seed}:order:{ep}").shuffle(order)
        tot = 0.0
        for i in range(0, len(order), c["batch"]):
            idx = order[i:i + c["batch"]]
            S = nets.pad_rows([source(tr[j][0], vocab) for j in idx]).to(dev_)
            T = nets.pad_rows([tr_tgt[j] for j in idx]).to(dev_)
            lg = model(S, T[:, :-1])
            loss = loss_fn(lg.reshape(-1, lg.shape[-1]), T[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
            opt.step()
            sched.step()
            st["step"] += 1
            tot += loss.item() * len(idx)
        score, right = dev_score(model, dev, dev_src, chars)
        tv_loss = mean_loss(model, tv_pairs, loss_fn)
        st["epoch"], st["elapsed"] = ep + 1, time.time() - t0
        rec = {"epoch": ep + 1, "train_loss": round(tot / len(tr), 4), "train_val_loss": round(tv_loss, 4),
               "dev_score": round(score, 4), "dev_right": right, "dev_n": len(dev),
               "minutes": round(st["elapsed"] / 60, 1)}
        st["log"].append(rec)
        print(json.dumps(rec), flush=True)
        key = [score, -tv_loss, -(ep + 1)]          # higher dev score, then lower loss, then earlier epoch
        if st["best"] is None or key > st["best"]["key"]:
            st["best"] = {"key": key, "epoch": ep + 1, "dev_score": score, "train_val_loss": tv_loss}
            torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "epoch": ep + 1},
                       out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "torch_rng": nets.rng_state(dev_), "state": st}, out / "last.pt.tmp")
        (out / "last.pt.tmp").replace(out / "last.pt")
        if stop_after is not None and ep + 1 >= stop_after and ep + 1 < epochs:
            return {"stopped_after_epoch": ep + 1, "log": st["log"]}
    ck = torch.load(out / "best.pt", weights_only=False, map_location="cpu")
    nets.export_npz(ck["state"], {**meta, "epoch": ck["epoch"]}, out / "model.npz")
    result = {"seed": seed, "epochs": epochs, "steps": st["step"], "train_pairs": len(tr), "dev_words": len(dev),
              "best_epoch": st["best"]["epoch"], "best_dev_score": round(st["best"]["dev_score"], 6),
              "device": str(dev_), "log": st["log"]}
    (out / "train_log.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="the released model is seed 0")
    ap.add_argument("--out", type=Path, required=True, help="run folder")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    ap.add_argument("--stop-after", type=int, default=None, help="stop after this many epochs (quick checks)")
    a = ap.parse_args()
    r = train(a.seed, a.out, a.device, a.resume, stop_after=a.stop_after)
    print(json.dumps({k: v for k, v in r.items() if k != "log"}))


if __name__ == "__main__":
    main()
