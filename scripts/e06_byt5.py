"""E06 reference model: ByT5-small (google/byt5-small, pinned revision) with one registered LoRA recipe.

Byte-level, letters-only interface (no pronunciation assistance): the input is the lower-cased English word as UTF-8
bytes, the target the NFC Khmer spelling as UTF-8 bytes. Oversized examples are rejected explicitly, never truncated.

Steps:
    smoke     load, generate, backpropagate on a few batches; time and peak RSS → reports/feasibility.json
    train     LoRA adaptation on the E03 training partition; ≤ 1,000 updates or 4 elapsed hours; resumable
    predict   beam-4 candidates for dev, eval, the E06 pool and the discovery sample → predictions/byt5_*.jsonl

Example:
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e06_byt5.py smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e06  # noqa: E402

ROOT = e06.ROOT
BASE = ROOT / "models" / "e06" / "byt5-small"
OUT = ROOT / "models" / "e06" / "byt5_lora"
REVISION = "68377bdc18a2ffec8a0533fef03b1c513a4dd49d"
BASE_SHA256 = {"pytorch_model.bin": "5c5aaf56299d6f2d4eaadad550a40765198828ead4d74f0a15f91cbe0961931a"}
OFFSET = 3                   # ByT5: ids 0 pad, 1 eos, 2 unk; byte b → b + 3
PAD, EOS = 0, 1
RECIPE = {                   # registered before training (PLAN.md); not tuned
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "lora_targets": "q, k, v, o of every T5 attention block (encoder self, decoder self, decoder cross)",
    "base_frozen": True, "base_dropout": 0.1,
    "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999], "grad_clip": 1.0,
    "batch": 16, "grad_accumulation": 1, "warmup": 50, "schedule": "linear warm-up, then linear decay to 0 at max_updates",
    "max_updates": 1000, "max_hours": 4.0, "eval_every": 100,
    "select": "greedy exact-match macro on E03 dev primary items; ties: lower train-val loss, then earlier update",
    "max_src_bytes": 32, "max_tgt_bytes": 120, "threads": 6, "seed": 0,
    "decode": {"beam": 4, "return": 4, "length_penalty": 0.0, "max_new_tokens": 120, "early_stopping": True},
}


def peak_rss_mb() -> float:
    """Peak resident memory of this process (MB)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def check_base() -> dict:
    """The local base checkpoint must match the pinned hash."""
    got = {f: e06.sha(BASE / f) for f in BASE_SHA256}
    if got != BASE_SHA256:
        raise RuntimeError(f"ByT5 base checkpoint hash mismatch: {got}")
    return got


# ------------------------------------------------------------------ byte encoding (pure)
def encode_src(word: str) -> list[int]:
    """Lower-cased word → byte ids + EOS.

    Examples:
        >>> encode_src("Ab")
        [100, 101, 1]
    """
    return [b + OFFSET for b in word.lower().encode("utf-8")] + [EOS]


def encode_tgt(text: str) -> list[int]:
    """NFC Khmer spelling → byte ids + EOS."""
    return [b + OFFSET for b in e06.nfc(text).encode("utf-8")] + [EOS]


def decode_ids(ids: list[int]) -> tuple[str | None, str]:
    """Byte ids → `(text, status)`; status `ok`, `empty` or `malformed_utf8` (text None).

    Examples:
        >>> decode_ids([228, 155, 131, 1])[1], decode_ids([228, 1])[1], decode_ids([1])[1]
        ('ok', 'malformed_utf8', 'empty')
    """
    out = []
    for i in ids:
        if i == EOS:
            break
        if i >= OFFSET:
            out.append(i - OFFSET)
    if not out:
        return "", "empty"
    try:
        return e06.nfc(bytes(out).decode("utf-8")), "ok"
    except UnicodeDecodeError:
        return None, "malformed_utf8"


def fits(word: str, target: str | None = None) -> str | None:
    """None if the example fits the registered byte limits, else the rejection reason."""
    if len(encode_src(word)) - 1 > RECIPE["max_src_bytes"]:
        return "source_too_long"
    if target is not None and len(encode_tgt(target)) - 1 > RECIPE["max_tgt_bytes"]:
        return "target_too_long"
    return None


# ------------------------------------------------------------------ model and LoRA
def lora_linear_cls():
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        """`base(x) + scale · B(A(dropout(x)))`; base frozen, B starts at zero (the adapted model starts equal)."""

        def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.A = nn.Parameter(torch.empty(r, base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.scale = alpha / r
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale

    return LoRALinear


def load_model(lora: bool = True):
    """The pinned ByT5-small with the registered LoRA adapters (all base weights frozen)."""
    import torch
    from transformers import T5ForConditionalGeneration
    from transformers.models.t5.modeling_t5 import T5Attention
    torch.set_num_threads(RECIPE["threads"])
    check_base()
    model = T5ForConditionalGeneration.from_pretrained(str(BASE), dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    if lora:
        L = lora_linear_cls()
        torch.manual_seed(RECIPE["seed"])
        for m in model.modules():
            if isinstance(m, T5Attention):
                for name in ("q", "k", "v", "o"):
                    setattr(m, name, L(getattr(m, name), RECIPE["lora_r"], RECIPE["lora_alpha"], RECIPE["lora_dropout"]))
    return model


def lora_state(model) -> dict:
    """Only the adapter tensors."""
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}


def counts(model) -> dict:
    """Trainable and total parameter counts."""
    return {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in model.parameters())}


def batch_tensors(pairs: list[tuple[str, str]]):
    """Padded `(input_ids, attention_mask, labels)` with labels padded by −100."""
    import torch
    src = [encode_src(w) for w, _ in pairs]
    tgt = [encode_tgt(k) for _, k in pairs]
    S, T = max(map(len, src)), max(map(len, tgt))
    ids = torch.tensor([s + [PAD] * (S - len(s)) for s in src])
    mask = (ids != PAD).long()
    labels = torch.tensor([t + [-100] * (T - len(t)) for t in tgt])
    return ids, mask, labels


def generate(model, words: list[str], beam: int, ret: int, bs: int = 32) -> list[list[tuple[str | None, str, float]]]:
    """Per word: up to `ret` `(text, status, score)` results, best first (sum log-prob, no length normalisation)."""
    import torch
    model.eval()
    out = []
    d = RECIPE["decode"]
    with torch.no_grad():
        for i in range(0, len(words), bs):
            chunk = words[i:i + bs]
            src = [encode_src(w) for w in chunk]
            S = max(map(len, src))
            ids = torch.tensor([s + [PAD] * (S - len(s)) for s in src])
            g = model.generate(input_ids=ids, attention_mask=(ids != PAD).long(), num_beams=beam,
                               num_return_sequences=ret, max_new_tokens=d["max_new_tokens"],
                               length_penalty=d["length_penalty"], early_stopping=d["early_stopping"],
                               do_sample=False, output_scores=True, return_dict_in_generate=True)
            seqs = g.sequences.tolist()
            scores = g.sequences_scores.tolist() if beam > 1 else [0.0] * len(seqs)
            for n in range(len(chunk)):
                res = []
                for j in range(ret):
                    t, st = decode_ids(seqs[n * ret + j][1:])
                    res.append((t, st, round(float(scores[n * ret + j]), 6)))
                out.append(res)
    return out


# ------------------------------------------------------------------ data
def train_rows() -> tuple[list, list, dict]:
    """E03 training partition: (optimisation rows, train-val rows, rejection counts); oversized rows rejected."""
    opt, val, rej = [], [], {}
    for line in e06.guard(e06.E03 / "train.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        w, k, _s, _t, _ph, part = line.split("\t")
        why = fits(w, k)
        if why:
            rej[why] = rej.get(why, 0) + 1
            continue
        (opt if part == "train" else val).append((w, e06.nfc(k)))
    return opt, val, rej


# ------------------------------------------------------------------ smoke (feasibility)
def smoke() -> dict:
    """Load, zero-shot generate (recorded, not an assessment), and time a few LoRA updates."""
    import torch
    t0 = time.perf_counter()
    rss0 = peak_rss_mb()
    model = load_model()
    load_s = time.perf_counter() - t0
    rss_load = peak_rss_mb()
    opt_rows, _val, rej = train_rows()
    zs_words = ["coffee", "london", "computer", "michael"]
    t = time.perf_counter()
    zs = generate(model, zs_words, beam=1, ret=1)
    gen_s = time.perf_counter() - t
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=RECIPE["lr"])
    model.train()
    rng = random.Random("e06-smoke")
    times, losses = [], []
    for _ in range(5):
        b = rng.sample(opt_rows, RECIPE["batch"])
        ids, mask, labels = batch_tensors(b)
        t = time.perf_counter()
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], RECIPE["grad_clip"])
        opt.step()
        opt.zero_grad()
        times.append(time.perf_counter() - t)
        losses.append(round(float(loss), 4))
    dev_words = [i.word for i in e06.primary(e06.items("dev"))][:64]
    t = time.perf_counter()
    generate(model, dev_words, beam=1, ret=1)
    greedy64 = time.perf_counter() - t
    t = time.perf_counter()
    generate(model, dev_words[:16], beam=4, ret=4)
    beam16 = time.perf_counter() - t
    per_update = sorted(times[1:])[len(times[1:]) // 2]
    est_train_h = (RECIPE["max_updates"] * per_update + RECIPE["max_updates"] / RECIPE["eval_every"]
                   * (greedy64 / 64 * 253 + 2)) / 3600
    out = {"base_sha256": check_base(), "revision": REVISION, "params": counts(model),
           "load_s": round(load_s, 2), "rss_after_imports_mb": rss0, "rss_after_load_mb": rss_load,
           "zero_shot_greedy (not an assessment: the base model is not adapted)":
               {w: r[0][:2] for w, r in zip(zs_words, zs)}, "zero_shot_gen_s": round(gen_s, 2),
           "update_s": [round(x, 2) for x in times], "update_s_median_after_first": round(per_update, 2),
           "losses": losses, "greedy_64_dev_s": round(greedy64, 2), "beam4_16_dev_s": round(beam16, 2),
           "estimated_train_hours_1000_updates": round(est_train_h, 2), "peak_rss_mb": peak_rss_mb(),
           "train_rows": len(opt_rows), "rejected_rows": rej, "threads": RECIPE["threads"], "batch": RECIPE["batch"]}
    e06.REPORTS.mkdir(parents=True, exist_ok=True)
    (e06.REPORTS / "feasibility.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


# ------------------------------------------------------------------ training (resumable)
def lr_factor(k: int) -> float:
    """Applied learning-rate factor at update index k (0-based).

    Examples:
        >>> lr_factor(0), lr_factor(49), round(lr_factor(525), 3), lr_factor(1000)
        (0.02, 1.0, 0.5, 0.0)
    """
    w, n = RECIPE["warmup"], RECIPE["max_updates"]
    return (k + 1) / w if k < w else max(0.0, (n - k) / (n - w))


def dev_eval(model, dev, val) -> tuple[float, float, int]:
    """Greedy dev macro (exact match), train-val loss and the number of correct dev items."""
    import torch
    res = generate(model, [i.word for i in dev], beam=1, ret=1, bs=64)
    hit = {i.id: int(r[0][0] is not None and r[0][0] in {e06.nfc(a) for a in i.accepted}) for i, r in zip(dev, res)}
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for j in range(0, len(val), 32):
            ids, mask, labels = batch_tensors(val[j:j + 32])
            tot += float(model(input_ids=ids, attention_mask=mask, labels=labels).loss) * len(val[j:j + 32])
            n += len(val[j:j + 32])
    return e06.macro(dev, hit), tot / max(1, n), sum(hit.values())


def train() -> dict:
    """The registered LoRA adaptation; resumes from `last.pt` (validated) without repeating updates."""
    import torch
    from khmer_translit import experiment
    reg = e06.registration()
    if reg["byt5"]["recipe"] != RECIPE:
        raise RuntimeError("recipe differs from the registered recipe")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "done.json").exists():
        return json.loads((OUT / "done.json").read_text())
    model = load_model()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=RECIPE["lr"], betas=tuple(RECIPE["betas"]), weight_decay=RECIPE["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    opt_rows, val, rej = train_rows()
    dev = e06.primary(e06.items("dev"))
    st = {"update": 0, "epoch": 0, "pos": 0, "elapsed_s": 0.0, "best": None, "history": []}
    if (OUT / "last.pt").exists():
        ck = torch.load(OUT / "last.pt", weights_only=False)
        if ck["recipe"] != RECIPE or ck["base_sha256"] != BASE_SHA256:
            raise RuntimeError("last.pt does not belong to this recipe/base")
        model.load_state_dict(ck["lora"], strict=False)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["torch_rng"])
        st = ck["state"]
    run = experiment.Run(OUT, e06.EXPERIMENT, "byt5_lora", {"recipe": RECIPE, "base_revision": REVISION},
                         inputs=[e06.E03 / "train.tsv", e06.E03 / "dev_references.jsonl", BASE / "config.json"])
    order = lambda ep: random.Random(f"{RECIPE['seed']}:order:{ep}").sample(range(len(opt_rows)), len(opt_rows))  # noqa: E731
    perm = order(st["epoch"])
    t_start = time.perf_counter() - st["elapsed_s"]
    stop = None
    while st["update"] < RECIPE["max_updates"]:
        if time.perf_counter() - t_start > RECIPE["max_hours"] * 3600:
            stop = "time_limit"
            break
        model.train()
        if st["pos"] + RECIPE["batch"] > len(perm):
            st["epoch"] += 1
            st["pos"] = 0
            perm = order(st["epoch"])
        b = [opt_rows[i] for i in perm[st["pos"]:st["pos"] + RECIPE["batch"]]]
        st["pos"] += RECIPE["batch"]
        ids, mask, labels = batch_tensors(b)
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, RECIPE["grad_clip"]))
        lr_applied = opt.param_groups[0]["lr"]
        opt.step()
        sched.step()
        opt.zero_grad()
        st["update"] += 1
        rec = {"update": st["update"], "loss": round(float(loss), 4), "grad_norm": round(gn, 4), "lr": lr_applied,
               "epoch": st["epoch"], "elapsed_s": round(time.perf_counter() - t_start, 1), "rss_mb": peak_rss_mb()}
        if st["update"] % RECIPE["eval_every"] == 0 or st["update"] == RECIPE["max_updates"]:
            dm, vl, dc = dev_eval(model, dev, val)
            rec.update(dev_macro=round(dm, 4), dev_correct=dc, val_loss=round(vl, 4))
            key = (dm, -vl, -st["update"])
            if st["best"] is None or key > tuple(st["best"]["key"]):
                st["best"] = {"key": list(key), "update": st["update"], "dev_macro": dm, "val_loss": vl}
                torch.save(lora_state(model), OUT / "best.pt")
            st["history"].append(rec)
            st["elapsed_s"] = time.perf_counter() - t_start
            tmp = OUT / "last.tmp"
            torch.save({"lora": lora_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "torch_rng": torch.get_rng_state(), "state": st, "recipe": RECIPE,
                        "base_sha256": BASE_SHA256}, tmp)
            os.replace(tmp, OUT / "last.pt")
            e06.mark_progress("train", f"update {st['update']}, dev {dm:.3f}, elapsed {st['elapsed_s'] / 3600:.2f} h")
        run.log(rec, step=st["update"])
    stop = stop or "max_updates"
    if not st["history"] or st["history"][-1]["update"] != st["update"]:   # stopped between evaluations
        dm, vl, dc = dev_eval(model, dev, val)
        st["history"].append({"update": st["update"], "dev_macro": round(dm, 4), "dev_correct": dc,
                              "val_loss": round(vl, 4), "final_eval": True})
        key = (dm, -vl, -st["update"])
        if st["best"] is None or key > tuple(st["best"]["key"]):
            st["best"] = {"key": list(key), "update": st["update"], "dev_macro": dm, "val_loss": vl}
            torch.save(lora_state(model), OUT / "best.pt")
    done = {"stop": stop, "updates": st["update"], "epochs_started": st["epoch"] + 1,
            "elapsed_hours": round((time.perf_counter() - t_start) / 3600, 3), "best": st["best"],
            "effective_batch": RECIPE["batch"] * RECIPE["grad_accumulation"], "params": counts(model),
            "train_rows": len(opt_rows), "train_val_rows": len(val), "rejected": rej, "peak_rss_mb": peak_rss_mb(),
            "best_sha256": e06.sha(OUT / "best.pt"), "base_sha256": BASE_SHA256, "recipe": RECIPE}
    (OUT / "done.json").write_text(json.dumps(done, indent=1) + "\n")
    run.finish(done)
    return done


# ------------------------------------------------------------------ predictions
def score_check(model, words: list[str]) -> float:
    """Max |beam score − teacher-forced sum log-prob (EOS included)| over the top beam of a few words.

    Confirms that beam scores are unnormalised sums (length_penalty 0), as for the letters models.
    """
    import torch
    gaps = []
    for w, r in zip(words, generate(model, words, 4, 1)):
        t, st, sc = r[0]
        if st != "ok":
            continue
        ids, mask, labels = batch_tensors([(w, t)])
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask, labels=labels).logits
        lp = torch.log_softmax(logits, -1)[0, torch.arange(labels.shape[1]), labels[0]].sum()
        gaps.append(abs(float(lp) - sc))
    return max(gaps) if gaps else float("nan")


def predict() -> dict:
    """Beam-4 top-4 for dev, eval, the pool and the discovery sample, with the selected adapter."""
    import torch
    done = json.loads((OUT / "done.json").read_text())
    if e06.sha(OUT / "best.pt") != done["best_sha256"]:
        raise RuntimeError("best.pt hash differs from done.json")
    model = load_model()
    model.load_state_dict(torch.load(OUT / "best.pt"), strict=False)
    sets = e06.prediction_sets()
    e06.PRED.mkdir(parents=True, exist_ok=True)
    timing = {"score_check_max_gap": score_check(model, [w for _, w in sets["dev"][:8]])}
    for name, rows in sets.items():
        p = e06.PRED / f"byt5_{name}.jsonl"
        if p.exists():
            continue
        t = time.perf_counter()
        ok = [(i, w) for i, w in rows if fits(w) is None]
        res = dict(zip([i for i, _ in ok], generate(model, [w for _, w in ok], 4, 4)))
        timing[name] = {"words": len(rows), "seconds": round(time.perf_counter() - t, 1)}
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for i, w in rows:
                r = res.get(i)
                f.write(json.dumps({"id": i, "word": w, "system": "byt5_lora", "width": 4,
                                    "status": "refused_source_too_long" if r is None else "ok",
                                    "candidates": [t for t, s, _ in (r or []) if s == "ok" and t],
                                    "raw": [{"text": t, "status": s, "score": sc} for t, s, sc in (r or [])]},
                                   ensure_ascii=False) + "\n")
        os.replace(tmp, p)
    return {"timing": timing, "peak_rss_mb": peak_rss_mb()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["smoke", "train", "predict"])
    a = ap.parse_args()
    r = {"smoke": smoke, "train": train, "predict": predict}[a.step]()
    print(json.dumps(r, ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
