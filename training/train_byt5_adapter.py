#!/usr/bin/env python3
"""Train the ByT5 adapter: LoRA layers on a frozen `google/byt5-small`, English word in, Khmer spelling out.

    python training/train_byt5_adapter.py --seed 0 --out runs/byt5_s0 --device cuda
    python training/train_byt5_adapter.py --seed 0 --out runs/byt5_s0 --device cuda --resume
    python training/train_byt5_adapter.py --base /path/to/byt5-small ...     # a local copy of the base model

ByT5 reads and writes UTF-8 bytes, so no vocabulary is needed: the input is the lower-cased word's bytes, the output
the Khmer spelling's bytes (NFC). The base model stays frozen. Small low-rank layers (LoRA: rank 16, scale 32/16)
are added beside every attention projection (q, k, v, o) and every feed-forward projection (wi_0, wi_1, wo); only
these are trained, about 3% of the base model's size.

Training: 3,000 updates of 32 pairs from `training-data/byt5/train.tsv`, visited in a seeded order, AdamW with
learning rate 0.001, a 100-update warm-up and a linear decay to 0. At updates 600, 1,200, 1,800, 2,400 and 3,000
the model spells every development word greedily; the kept point has the highest share of development words spelled
as one of their accepted spellings (averaged within each word family, then over families), ties going to the lower
mean character error, then to the earlier update. Output: `best.pt`, `last.pt`, `train_log.json` and `adapter/`,
which `khmer_translit.byt5.ByT5Transliterator.from_pretrained` loads with the base model.
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

from khmer_translit import byt5

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / "config.yaml"
EXCLUDED_FLAGS = {"pronunciation_ambiguous", "possible_translation"}
PAD, EOS, OFFSET = 0, 1, 3            # ByT5 ids: 0 padding, 1 end, 2 unknown; byte b is id b + 3
MODULES = {"attn": ("q", "k", "v", "o"), "ffn": ("wi_0", "wi_1", "wo")}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def load_data(folder: Path) -> tuple[list[tuple[str, str]], list[dict]]:
    """`(training pairs in file order, development items used for selection)`."""
    rows = [l.split("\t") for l in (folder / "train.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    dev = [json.loads(l) for l in (folder / "dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [(r[0], r[1]) for r in rows], [d for d in dev if not EXCLUDED_FLAGS & set(d["flags"])]


class LoRALinear(nn.Module):
    """`base(x) + (alpha / r) · B(A(dropout(x)))` beside a frozen linear layer; B starts at zero, so training starts
    from the unchanged base model."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    @property
    def weight(self):                      # T5's gated feed-forward reads `wo.weight.dtype`
        return self.base.weight

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale


def build(base: str, revision: str | None, adapter: dict, base_dropout: float, seed: int):
    """The frozen base model with LoRA layers on the configured modules."""
    from transformers import T5ForConditionalGeneration
    from transformers.models.t5.modeling_t5 import T5Attention, T5DenseGatedActDense
    model = T5ForConditionalGeneration.from_pretrained(base, revision=revision, dtype=torch.float32,
                                                       dropout_rate=base_dropout)
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    for m in list(model.modules()):
        names = (MODULES["attn"] if isinstance(m, T5Attention) and "attn" in adapter["targets"] else
                 MODULES["ffn"] if isinstance(m, T5DenseGatedActDense) and "ffn" in adapter["targets"] else ())
        for n in names:
            setattr(m, n, LoRALinear(getattr(m, n), adapter["r"], adapter["alpha"], adapter["dropout"]))
    return model


def adapter_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}


def batch(pairs: list[tuple[str, str]], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Padded `(input ids, attention mask, labels)`; labels padded with -100 (ignored by the loss)."""
    src = [[b + OFFSET for b in w.lower().encode("utf-8")] + [EOS] for w, _ in pairs]
    tgt = [[b + OFFSET for b in nfc(k).encode("utf-8")] + [EOS] for _, k in pairs]
    S, T = max(map(len, src)), max(map(len, tgt))
    ids = torch.tensor([s + [PAD] * (S - len(s)) for s in src])
    labels = torch.tensor([t + [-100] * (T - len(t)) for t in tgt])
    return ids.to(device), (ids != PAD).long().to(device), labels.to(device)


def greedy_words(model, words: list[str], max_new: int, device, bs: int = 64) -> list[str | None]:
    """Greedy spelling of each word (None when the bytes are not valid UTF-8 or no end mark was produced)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(words), bs):
            chunk = [[b + OFFSET for b in w.lower().encode("utf-8")] + [EOS] for w in words[i:i + bs]]
            S = max(map(len, chunk))
            ids = torch.tensor([s + [PAD] * (S - len(s)) for s in chunk], device=device)
            g = model.generate(input_ids=ids, attention_mask=(ids != PAD).long(), max_new_tokens=max_new,
                               do_sample=False, num_beams=1)
            for row in g.tolist():
                raw = row[1:]
                if EOS not in raw and any(t >= OFFSET for t in raw):
                    out.append(None)                         # ran out of length without an end mark
                    continue
                raw = raw[:raw.index(EOS)] if EOS in raw else raw
                bs_ = bytes(t - OFFSET for t in raw if t >= OFFSET)
                try:
                    out.append(nfc(bs_.decode("utf-8")) if bs_ else None)
                except UnicodeDecodeError:
                    out.append(None)
    return out


def edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def dev_score(model, dev: list[dict], max_new: int, device) -> dict:
    """Family-averaged top-1 agreement with the accepted spellings, and mean character error rate."""
    texts = greedy_words(model, [d["word"] for d in dev], max_new, device)
    hit = {d["id"]: int(t is not None and (t == nfc(d["preferred"]) or any(t == nfc(a) for a in d["accepted"])))
           for d, t in zip(dev, texts)}
    cer = mean(min(edit(t or "", nfc(a)) / max(1, len(a)) for a in d["accepted"]) for d, t in zip(dev, texts))
    fam: dict[str, list[int]] = {}
    for d in dev:
        fam.setdefault(d["family"], []).append(hit[d["id"]])
    top1 = mean(mean(v) for v in fam.values())
    return {"key": [top1, -cer], "dev_top1": round(top1, 4), "dev_cer": round(cer, 4), "dev_right": sum(hit.values())}


def lr_factor(k: int, warmup: int, total: int) -> float:
    return (k + 1) / warmup if k < warmup else max(0.0, (total - k) / (total - warmup))


def train(seed: int, out: Path, device: str = "cpu", resume: bool = False, base: str | None = None,
          cfg: dict | None = None, data=None, stop_after: int | None = None) -> dict:
    """Train the adapter; see the module docstring. `cfg`, `data`, `stop_after` are for tests and checks."""
    cfg = cfg or yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    c = cfg["byt5_adapter"]
    torch.set_num_threads(c["threads"])
    rows, dev = data or load_data(ROOT / cfg["data"]["byt5"])
    dev_ = torch.device(device)
    model = build(base or c["base"], None if base else c["base_revision"], c["adapter"], c["base_dropout"], seed)
    model.to(dev_)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=c["lr"], betas=tuple(c["betas"]), weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: lr_factor(k, c["warmup"], c["max_updates"]))
    out.mkdir(parents=True, exist_ok=True)
    st = {"update": 0, "epoch": 0, "pos": 0, "best": None, "log": [], "elapsed": 0.0}
    if resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", weights_only=False, map_location="cpu")
        model.load_state_dict(ck["adapter"], strict=False)
        opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["torch_rng"])
        st = ck["state"]
        print(json.dumps({"resumed_at_update": st["update"]}), flush=True)
    order = lambda ep: random.Random(f"{seed}:order:{ep}").sample(range(len(rows)), len(rows))  # noqa: E731
    perm = order(st["epoch"])
    t0 = time.time() - st["elapsed"]

    def save_last():
        st["elapsed"] = time.time() - t0
        torch.save({"adapter": adapter_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "torch_rng": torch.get_rng_state(), "state": st}, out / "last.pt.tmp")
        (out / "last.pt.tmp").replace(out / "last.pt")

    while st["update"] < c["max_updates"]:
        model.train()
        if st["pos"] + c["batch"] > len(perm):
            st["epoch"], st["pos"] = st["epoch"] + 1, 0
            perm = order(st["epoch"])
        b = [rows[i] for i in perm[st["pos"]:st["pos"] + c["batch"]]]
        st["pos"] += c["batch"]
        ids, mask, labels = batch(b, dev_)
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, c["grad_clip"])
        opt.step()
        sched.step()
        opt.zero_grad()
        st["update"] += 1
        if st["update"] in c["eval_at"] or st["update"] == c["max_updates"]:
            r = dev_score(model, dev, c["max_new_tokens"], dev_)
            rec = {"update": st["update"], "loss": round(float(loss), 4), **{k: v for k, v in r.items() if k != "key"},
                   "minutes": round((time.time() - t0) / 60, 1)}
            st["log"].append(rec)
            print(json.dumps(rec), flush=True)
            if st["best"] is None or r["key"] > st["best"]["key"]:
                st["best"] = {"key": r["key"], "update": st["update"], **rec}
                torch.save(adapter_state(model), out / "best.pt")
            save_last()
        elif st["update"] % 100 == 0:
            save_last()
        if stop_after is not None and st["update"] >= stop_after:
            save_last()
            return {"stopped_at_update": st["update"], "log": st["log"]}
    best = torch.load(out / "best.pt", weights_only=True, map_location="cpu")
    weights_file = "pytorch_model.bin"
    config = {"task": "one English word or name → Khmer script approximating its English pronunciation",
              "base": {"repo": c["base"], "revision": c["base_revision"], "weights_file": weights_file,
                       "weights_sha256": c["base_weights_sha256"]},
              "adapter": {"type": "LoRA (khmer_translit.byt5.LoRALinear)", "r": c["adapter"]["r"],
                          "alpha": c["adapter"]["alpha"], "targets": c["adapter"]["targets"],
                          "modules": {t: list(MODULES[t]) for t in c["adapter"]["targets"]}},
              "decode": {"single": "greedy", "candidate_beam": 4,
                         "candidate_search": "early_stopping='never', length_penalty 0",
                         "max_new_tokens": c["max_new_tokens"], "max_input_bytes": 32, "batch_size": 32},
              "training": {"seed": seed, "update": st["best"]["update"], "training_rows": len(rows),
                           "lr": c["lr"], "batch": c["batch"], "max_updates": c["max_updates"],
                           "warmup": c["warmup"], "schedule": "linear"}}
    byt5.export_adapter(best, out / "adapter", config)
    result = {"seed": seed, "updates": st["update"], "epochs_started": st["epoch"] + 1, "rows": len(rows),
              "best_update": st["best"]["update"], "best_dev_top1": st["best"]["dev_top1"],
              "best_dev_cer": st["best"]["dev_cer"], "device": str(dev_), "log": st["log"]}
    (out / "train_log.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="the released adapter is seed 0")
    ap.add_argument("--out", type=Path, required=True, help="run folder")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--base", default=None, help="local folder of google/byt5-small (default: download the pinned revision)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    ap.add_argument("--stop-after", type=int, default=None, help="stop after this many updates (quick checks)")
    a = ap.parse_args()
    r = train(a.seed, a.out, a.device, a.resume, a.base, stop_after=a.stop_after)
    print(json.dumps({k: v for k, v in r.items() if k != "log"}))


if __name__ == "__main__":
    main()
