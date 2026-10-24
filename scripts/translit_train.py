#!/usr/bin/env python3
"""Train and export the learned Latin → Khmer transliterator (V02-N1), on CPU.

Run from the repository root with the interpreter that has torch:

    .venv/bin/python scripts/translit_train.py train [--epochs 40] [--seed 0]
    .venv/bin/python scripts/translit_train.py export   # → models/model/translit.npz

The two served models (see docs/TRANSLITERATOR.md):

    # B: names and romanised Khmer
    ... train --extra data/extra_train.tsv --out models/model_b
    # E: English dictionary words (loanword list + LLM drafts repeated 4x, phone-aware source)
    ... train --extra data/extra_train.tsv --loans data/llm_drafts.tsv --loan-repeat 4 --use-list \
        --phones --max-in 48 --out models/model_e

then `export --out <the same folder>`.

**Model:** a small pre-LayerNorm transformer encoder-decoder over characters (default d = 192, 3 + 3 layers,
4 heads, FF 512, learned positions; ≈ 2.5M parameters). Its parameter names match the NumPy inference in
`khmer_translit.translit`, which is what serving uses.

**Training:** cross-entropy with label smoothing 0.1, AdamW (lr 1e-3, weight decay 0.01), linear warm-up 500 steps
then cosine decay, batch 128, dropout 0.1. The checkpoint with the best **validation exact-match** (greedy) is kept;
the test split is never read here. Data: `data/pairs.tsv` (`scripts/translit_data.py`).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from khmer_translit.translit import BOS, EOS, MAX_IN, MAX_OUT, PAD, source  # noqa: E402

MODEL_DIR = ROOT / "models" / "model"
LIST_SPLIT = ROOT / "data" / "list_split.json"
VERIFIED = {"owner_confirmed", "owner_corrected", "owner_chosen", "owner_added", "llm_suggested_owner_verified"}
SRC_MAX = MAX_IN   # source length cap; set from --max-in in train()


class Attn(nn.Module):
    """Multi-head attention with separate q/k/v/o Linear layers (names mirrored in NumPy).

    Args:
        d: Model width.
        heads: Number of heads (`d % heads == 0`).
        dropout: Attention-probability dropout.
    """

    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.heads, self.drop = heads, nn.Dropout(dropout)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Attend `q_in` over `kv_in`.

        Args:
            q_in: `(B, Tq, d)`.
            kv_in: `(B, Tk, d)`.
            mask: Bool `(B or 1, Tq or 1, Tk)`, True = blocked; or None.

        Returns:
            `(B, Tq, d)`.
        """
        B, Tq, d = q_in.shape
        h, dh = self.heads, d // self.heads
        q = self.q(q_in).view(B, Tq, h, dh).transpose(1, 2)
        k = self.k(kv_in).view(B, -1, h, dh).transpose(1, 2)
        v = self.v(kv_in).view(B, -1, h, dh).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / math.sqrt(dh)
        if mask is not None:
            s = s.masked_fill(mask[:, None], -1e9)
        out = (self.drop(s.softmax(-1)) @ v).transpose(1, 2).reshape(B, Tq, d)
        return self.o(out)


class Block(nn.Module):
    """Pre-LN transformer block; `cross=True` adds encoder-decoder attention (decoder block).

    Args:
        d: Width.
        heads: Heads.
        ff: Feed-forward hidden size.
        dropout: Residual dropout.
        cross: Whether this is a decoder block.
    """

    def __init__(self, d: int, heads: int, ff: int, dropout: float, cross: bool) -> None:
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.self = Attn(d, heads, dropout)
        if cross:
            self.ln3, self.cross = nn.LayerNorm(d), Attn(d, heads, dropout)
        self.ff1, self.ff2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.is_cross, self.drop = cross, nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, self_mask: torch.Tensor | None, memory: torch.Tensor | None = None,
                mem_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the block.

        Args:
            h: `(B, T, d)`.
            self_mask: Self-attention mask.
            memory: Encoder states (decoder only).
            mem_mask: Encoder pad mask `(B, 1, Tk)` (decoder only).

        Returns:
            `(B, T, d)`.
        """
        a = self.ln1(h)
        h = h + self.drop(self.self(a, a, self_mask))
        if self.is_cross:
            h = h + self.drop(self.cross(self.ln2(h), memory, mem_mask))
            ln_ff = self.ln3
        else:
            ln_ff = self.ln2
        return h + self.drop(self.ff2(nn.functional.gelu(self.ff1(ln_ff(h)))))


class Model(nn.Module):
    """Character encoder-decoder.

    Args:
        n_src: Source vocabulary size including the 3 specials.
        n_tgt: Target vocabulary size including the 3 specials.
        d: Width. layers: Blocks per side. heads: Heads. ff: FF size. dropout: Dropout.
    """

    def __init__(self, n_src: int, n_tgt: int, d: int = 192, layers: int = 3, heads: int = 4, ff: int = 512,
                 dropout: float = 0.1, max_in: int = MAX_IN) -> None:
        super().__init__()
        self.src_emb, self.tgt_emb = nn.Embedding(n_src, d), nn.Embedding(n_tgt, d)
        self.src_pos, self.tgt_pos = nn.Embedding(max_in, d), nn.Embedding(MAX_OUT + 2, d)
        self.enc = nn.ModuleList(Block(d, heads, ff, dropout, False) for _ in range(layers))
        self.dec = nn.ModuleList(Block(d, heads, ff, dropout, True) for _ in range(layers))
        self.enc_ln, self.dec_ln = nn.LayerNorm(d), nn.LayerNorm(d)
        self.out = nn.Linear(d, n_tgt)
        self.drop = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits.

        Args:
            src: `(B, Ts)` source ids (PAD = 0).
            tgt_in: `(B, Tt)` target ids starting with BOS.

        Returns:
            `(B, Tt, n_tgt)` logits.
        """
        pad = (src == PAD)[:, None, :]
        h = self.drop(self.src_emb(src) + self.src_pos.weight[: src.shape[1]])
        for blk in self.enc:
            h = blk(h, pad)
        memory = self.enc_ln(h)
        L = tgt_in.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)[None]
        g = self.drop(self.tgt_emb(tgt_in) + self.tgt_pos.weight[:L])
        for blk in self.dec:
            g = blk(g, causal, memory, pad)
        return self.out(self.dec_ln(g))


PAIRS = ROOT / "data" / "pairs.tsv"


def load_pairs(split: str) -> list[tuple[str, str]]:
    """Read `(latin, khmer)` pairs of one split from `PAIRS` (default `data/pairs.tsv`; `--pairs` changes it).

    Args:
        split: "train", "val" or "test".

    Returns:
        The pairs, file order.
    """
    rows = PAIRS.read_text(encoding="utf-8").splitlines()[1:]
    return [(r.split("\t")[0], r.split("\t")[1]) for r in rows if r.split("\t")[3] == split]


def list_split() -> dict[str, list[str]]:
    """Frozen split of the loanword list's single words for the list-trained model (D146).

    - **test:** half of the owner-verified words (`VERIFIED` provenance in `datasets/build/latin_audit.json`),
      sha1(word) even: held out from all training, the comparison set;
    - **val:** non-verified words with sha1(word) mod 10 == 1 (checkpoint selection, with the romanised val);
    - **train:** every other single-word entry.
    Written once to `data/list_split.json`; later calls return the frozen file.

    Returns:
        `{"test": [...], "val": [...], "train": [...]}` (Latin words).
    """
    import hashlib
    import re
    if LIST_SPLIT.exists():
        return json.loads(LIST_SPLIT.read_text(encoding="utf-8"))
    prov = json.loads((ROOT / "datasets/build/latin_audit.json").read_text(encoding="utf-8"))["provenance_by_word"]
    h = lambda w: int(hashlib.sha1(w.encode()).hexdigest(), 16)
    words = sorted(w for w in prov if re.fullmatch("[a-z]+", w))
    split = {"test": [w for w in words if prov[w] in VERIFIED and h(w) % 2 == 0],
             "val": [w for w in words if prov[w] not in VERIFIED and h(w) % 10 == 1]}
    held = set(split["test"]) | set(split["val"])
    split["train"] = [w for w in words if w not in held]
    LIST_SPLIT.write_text(json.dumps(split, indent=1) + "\n", encoding="utf-8")
    return split


def batches(pairs: list[tuple[str, str]], src_v: str, tgt_v: str, size: int, shuffle: bool):
    """Yield padded `(src, tgt)` id tensors (tgt = BOS + chars + EOS).

    Args:
        pairs: `(latin, khmer)` pairs.
        src_v: Source vocabulary string. tgt_v: Target vocabulary string.
        size: Batch size. shuffle: Shuffle order (uses the global `random` state).

    Yields:
        `(src, tgt)` LongTensors.
    """
    order = list(range(len(pairs)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), size):
        chunk = [pairs[j] for j in order[i:i + size]]
        s = [[src_v.index(c) + 3 for c in lat][:SRC_MAX] for lat, _ in chunk]
        t = [[BOS] + [tgt_v.index(c) + 3 for c in km][:MAX_OUT] + [EOS] for _, km in chunk]
        S = torch.zeros(len(chunk), max(map(len, s)), dtype=torch.long)
        T = torch.zeros(len(chunk), max(map(len, t)), dtype=torch.long)
        for r, (a, b) in enumerate(zip(s, t)):
            S[r, :len(a)], T[r, :len(b)] = torch.tensor(a), torch.tensor(b)
        yield S, T


@torch.no_grad()
def greedy(model: Model, src: torch.Tensor) -> list[list[int]]:
    """Greedy decode a batch (validation only; serving uses NumPy beam search).

    Args:
        model: The model in eval mode.
        src: `(B, Ts)` source ids.

    Returns:
        Per item, the generated target ids without BOS/EOS.
    """
    out = torch.full((src.shape[0], 1), BOS, dtype=torch.long)
    finished = torch.zeros(src.shape[0], dtype=torch.bool)
    for _ in range(MAX_OUT + 1):
        nxt = model(src, out)[:, -1].argmax(-1)
        out = torch.cat([out, nxt[:, None]], 1)
        finished |= nxt == EOS
        if finished.all():
            break
    res = []
    for row in out[:, 1:].tolist():
        res.append(row[: row.index(EOS)] if EOS in row else row)
    return res


def evaluate(model: Model, pairs: list[tuple[str, str]], src_v: str, tgt_v: str) -> float:
    """Greedy exact-match rate on `pairs` (a Latin form counts as correct if it produces its Khmer target).

    Args:
        model: Model. pairs: Evaluation pairs. src_v, tgt_v: Vocabularies.

    Returns:
        Exact-match rate in [0, 1].
    """
    model.eval()
    hit = 0
    for i, (S, _) in enumerate(batches(pairs, src_v, tgt_v, 512, False)):
        for j, ids in enumerate(greedy(model, S)):
            hit += "".join(tgt_v[t - 3] for t in ids if t >= 3) == pairs[i * 512 + j][1]
    model.train()
    return hit / len(pairs)


def train(args: argparse.Namespace) -> None:
    """Train, keep the best-validation checkpoint in `models/model/best.pt`, log to `train_log.json`.

    Args:
        args: Parsed CLI arguments (`epochs`, `seed`, `d`, `layers`).
    """
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    global SRC_MAX
    SRC_MAX = args.max_in
    tr, va = load_pairs("train"), load_pairs("val")
    for path in args.extra:
        rows = [r.split("\t") for r in Path(path).read_text(encoding="utf-8").splitlines()[1:]]
        extra = [(r[0], r[1]) for r in rows]
        tr = tr + [p for p in extra if len(p[0]) <= MAX_IN and len(p[1]) <= MAX_OUT]
    if args.use_list:
        sys.path.insert(0, str(ROOT / "src"))
        from khmer_translit.loanwords import read as load_latin
        live, split = load_latin(), list_split()
        test = set(split["test"])
        loans = [(w, live[w]) for w in split["train"] if len(live[w]) <= MAX_OUT]
        for path in args.loans:   # D148: English-loan pairs, repeated with the list train split
            rows = [r.split("\t") for r in Path(path).read_text(encoding="utf-8").splitlines()[1:]]
            loans += [(r[0], r[1]) for r in rows if r[0] not in test and len(r[0]) <= MAX_IN and len(r[1]) <= MAX_OUT]
        tr = [p for p in tr if p[0] not in test]
        if args.loans_only:   # D151: English specialist; no romanised / mined pairs
            tr, va = [], []
        tr_tags = ["other"] * len(tr) + ["en"] * (len(loans) * args.loan_repeat)
        tr = tr + loans * args.loan_repeat
        va_tags = ["other"] * len(va)
        va = va + [(w, live[w]) for w in split["val"] if len(live[w]) <= MAX_OUT]
        va_tags += ["en"] * (len(va) - len(va_tags))
    else:
        tr_tags, va_tags = ["other"] * len(tr), ["other"] * len(va)
    if not args.tags:   # D149: training tags come from the data source (loan pairs "en", the rest "other")
        tr_tags, va_tags = [None] * len(tr), [None] * len(va)
    tr = [(source(lat, args.phones, g), km) for (lat, km), g in zip(tr, tr_tags)]
    va = [(source(lat, args.phones, g), km) for (lat, km), g in zip(va, va_tags)]
    if args.init:   # D151: continue from a trained model; keep its vocabularies, drop pairs it cannot encode
        ck = torch.load(args.init)
        src_v, tgt_v = ck["src_vocab"], ck["tgt_vocab"]
        fits = lambda p: set(p[0]) <= set(src_v) and set(p[1]) <= set(tgt_v)
        print(json.dumps({"init": str(args.init), "dropped_train": sum(not fits(p) for p in tr),
                          "dropped_val": sum(not fits(p) for p in va)}), flush=True)
        tr, va = [p for p in tr if fits(p)], [p for p in va if fits(p)]
    else:
        src_v = "".join(sorted({c for lat, _ in tr + va for c in lat}))
        tgt_v = "".join(sorted({c for _, km in tr + va for c in km}))
    model = Model(len(src_v) + 3, len(tgt_v) + 3, d=args.d, layers=args.layers, max_in=args.max_in)
    if args.init:
        model.load_state_dict(ck["state"])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(tr) / 128)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    model_dir = Path(args.out)
    model_dir.mkdir(parents=True, exist_ok=True)
    log, best, t0 = [], -1.0, time.time()
    for ep in range(args.epochs):
        total = 0.0
        for S, T in batches(tr, src_v, tgt_v, 128, True):
            logits = model(S, T[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), T[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(S)
        acc = evaluate(model, va, src_v, tgt_v)
        log.append({"epoch": ep + 1, "train_loss": round(total / len(tr), 4), "val_exact": round(acc, 4),
                    "minutes": round((time.time() - t0) / 60, 1)})
        print(json.dumps(log[-1]), flush=True)
        if acc > best:
            best = acc
            torch.save({"state": model.state_dict(), "src_vocab": src_v, "tgt_vocab": tgt_v,
                        "d": args.d, "layers": args.layers, "heads": 4, "epoch": ep + 1, "phones": args.phones,
                        "max_in": args.max_in, "tags": args.tags}, model_dir / "best.pt")
    (model_dir / "train_log.json").write_text(json.dumps(
        {"params": n_params, "seed": args.seed, "epochs": args.epochs, "d": args.d, "layers": args.layers,
         "extra": args.extra, "loans": args.loans, "loan_repeat": args.loan_repeat, "use_list": args.use_list, "phones": args.phones, "max_in": args.max_in,
         "tags": args.tags, "init": args.init, "lr": args.lr, "warmup": args.warmup,
         "loans_only": args.loans_only, "train_pairs": len(tr), "val_pairs": len(va),
         "pairs": str(PAIRS), "best_val_exact": best, "log": log}, indent=1) + "\n")


def export(model_dir: Path = MODEL_DIR) -> None:
    """Write `best.pt` as `translit.npz` (float32 arrays + `__meta__` JSON) for `khmer_translit.translit.load`.

    Args:
        model_dir: Folder holding `best.pt` (default the served model folder).
    """
    MODEL_DIR = model_dir  # noqa: N806 (local alias keeps the paths below unchanged)
    ck = torch.load(MODEL_DIR / "best.pt")
    arrays = {k: v.numpy().astype(np.float32) for k, v in ck["state"].items()}
    meta = {"src_vocab": ck["src_vocab"], "tgt_vocab": ck["tgt_vocab"], "heads": ck["heads"],
            "layers": ck["layers"], "epoch": ck["epoch"], "phones": ck.get("phones", False),
            "max_in": ck.get("max_in", MAX_IN), "tags": ck.get("tags", False)}
    np.savez(MODEL_DIR / "translit.npz", __meta__=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    print("wrote", MODEL_DIR / "translit.npz", meta["epoch"])


def main() -> int:
    """CLI entry point (`train` or `export`).

    Returns:
        0 on success.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["train", "export"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--extra", action="append", default=[],
                    help="extra training pairs TSV (latin, khmer, ... with header); repeatable")
    ap.add_argument("--out", default=str(MODEL_DIR), help="model folder (default the served one)")
    ap.add_argument("--use-list", action="store_true", help="add the loanword list's train split (D146)")
    ap.add_argument("--loans", action="append", default=[],
                    help="English-loan pairs TSV added with the list train split (needs --use-list); repeatable")
    ap.add_argument("--loan-repeat", type=int, default=1,
                    help="times the list train split + --loans pairs appear per epoch (D148)")
    ap.add_argument("--init", default=None, help="start from this best.pt (fine-tuning, D151)")
    ap.add_argument("--lr", type=float, default=1e-3, help="peak learning rate (1e-3 from scratch)")
    ap.add_argument("--warmup", type=int, default=500, help="linear warm-up steps")
    ap.add_argument("--loans-only", action="store_true",
                    help="with --use-list: train and validate on loan pairs only (English specialist, D151)")
    ap.add_argument("--tags", action="store_true",
                    help="prepend a group tag: loan pairs 'en', the rest 'other'; inference uses word_tag (D149)")
    ap.add_argument("--phones", action="store_true", help="append the CMUdict pronunciation to the source (D146)")
    ap.add_argument("--max-in", type=int, default=MAX_IN, help="source length cap (64 with --phones)")
    ap.add_argument("--pairs", default=None, help="romanised pair file instead of data/pairs.tsv (D170)")
    args = ap.parse_args()
    if args.pairs:
        global PAIRS
        PAIRS = Path(args.pairs)
    train(args) if args.mode == "train" else export(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
