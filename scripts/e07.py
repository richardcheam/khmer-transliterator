"""E07 ByT5 diagnosis: why did E06's ByT5-small adaptation write nearly the same Khmer word for every input?

Reuses E06's byte encoding (`e06_byt5.encode_src/encode_tgt/decode_ids/batch_tensors`) and the pinned base model;
adds configurable adapters, a resource guard, a run ledger, a pipeline audit, a 32-pair memorisation gate and, only
if that gate passes, one registered full adaptation. E06 code and outputs are left unchanged.

Steps (state in `state/e07/`, outputs in `experiments/E07-byt5-diagnosis/`):
    env         versions, hashes, data sizes → reports/environment.json
    reproduce   E06's best adapter on a fixed 16-row train/train-val sample → reports/e06_reproduction.json
    audit       pipeline checks (alignment, masking, EOS, adapters, gradients, reload, limits) and the E06 loss
                decomposition → reports/pipeline_audit.json
    tiny        one memorisation attempt (`--attempt 1|2`) on the frozen 32 pairs → reports/tiny_overfit*.json
    register    freeze the full recipe (needs a passed tiny gate) → registration.json
    train       the registered full adaptation (≤ 3,000 updates or 6 h; resumable)
    predict     greedy and beam-4 predictions for dev, eval, the E06 pool and the diagnostic samples
    analyse     historical agreement, input dependence, pool heuristics → results.json, results/per_item.jsonl
    cost        load time, batch-1 latency, throughput, memory, checkpoint size → reports/cost.json

Example:
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e07.py audit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e05  # noqa: E402
import e06  # noqa: E402
import e06_byt5 as B  # noqa: E402

ROOT = e06.ROOT
EXPERIMENT = "E07-byt5-diagnosis"
EXP = ROOT / "experiments" / EXPERIMENT
REPORTS = EXP / "reports"
PRED = EXP / "predictions"
LEDGER = EXP / "runs" / "ledger.jsonl"
STATE = ROOT / "state" / "e07"
MODELS = ROOT / "models" / "e07"
E06_ADAPTER = ROOT / "models" / "e06" / "byt5_lora" / "best.pt"
E06_ADAPTER_SHA = "609dd83545cba689b9558214bf82bbdaf6ace8e8823a7a617abe7b7fd8ba6f8c"
DIAG_SEED, TINY_SEED = "e07-diag", "e07-tiny"
GiB = 1 << 30

LIMITS = {                   # registered (PLAN.md); enforced by `Guard` and `budget_left`
    "threads": 6, "rss_max_bytes": 5 * GiB, "min_available_bytes": int(1.5 * GiB), "pressure_checks": 3,
    "tiny_attempts": 2, "tiny_max_updates": 1000, "tiny_max_hours": 0.75,
    "full_runs": 1, "full_max_updates": 3000, "full_max_hours": 6.0,
    "training_hours_total": 8.0, "wall_hours_total": 12.0,
}
MAX_NEW = B.RECIPE["max_tgt_bytes"] + 1          # every allowed target plus its EOS (E06 allowed 120: one short)

# The frozen tiny-subset diagnostic (PLAN.md §3); attempt 2 may change exactly one key, recorded with a diagnosis.
TINY = {
    "n": 32, "bins": [[1, 4], [5, 6], [7, 8], [9, 99]], "per_bin": 8,
    "adapter": {"r": 16, "alpha": 32, "dropout": 0.0, "targets": ["attn", "ffn"]},
    "base_dropout": 0.0, "optimizer": "AdamW", "lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.0,
    "grad_clip": 1.0, "batch": 16, "grad_accumulation": 1, "warmup": 20, "schedule": "constant",
    "max_updates": 1000, "max_hours": 0.75, "eval_every": 100, "stop_when_all_correct": True, "seed": 0,
}
TINY_GATE = {"exact_min": 30, "n": 32}


# ------------------------------------------------------------------ small helpers
def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cfg_hash(cfg: dict) -> str:
    return sha_text(json.dumps(cfg, sort_keys=True))


def write_json(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def rows_hash(rows: list[tuple[str, str]]) -> str:
    return sha_text("\n".join(f"{w}\t{k}" for w, k in rows))


def git_rev() -> dict:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    return {"commit": rev, "dirty_tracked": dirty}


# ------------------------------------------------------------------ resources
def mem() -> dict:
    """Current RSS of this process and system MemAvailable (bytes), from /proc."""
    rss = avail = 0
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) * 1024
    return {"rss": rss, "available": avail}


class Guard:
    """Sustained memory pressure detector: `pressure_checks` consecutive bad readings → stop.

    Examples:
        >>> g = Guard(read=lambda: {"rss": 6 * GiB, "available": 4 * GiB})
        >>> [g.check() for _ in range(3)]
        [None, None, 'memory_pressure']
    """

    def __init__(self, read=mem, limits: dict = LIMITS):
        self.read, self.lim, self.bad, self.peak_rss, self.min_avail = read, limits, 0, 0, None

    def check(self) -> str | None:
        m = self.read()
        self.peak_rss = max(self.peak_rss, m["rss"])
        self.min_avail = m["available"] if self.min_avail is None else min(self.min_avail, m["available"])
        over = m["rss"] > self.lim["rss_max_bytes"] or m["available"] < self.lim["min_available_bytes"]
        self.bad = self.bad + 1 if over else 0
        return "memory_pressure" if self.bad >= self.lim["pressure_checks"] else None


def ledger() -> list[dict]:
    return [json.loads(x) for x in LEDGER.read_text().splitlines()] if LEDGER.exists() else []


def ledger_add(rec: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": e06.now(), **rec}, ensure_ascii=False) + "\n")


def budget_left(kind: str, entries: list[dict] | None = None, now: float | None = None) -> dict:
    """Remaining attempts and hours for a run kind ("tiny" or "full"), from the ledger.

    A run counts as an attempt once started. Training hours sum the `hours` of finished segments (and the open
    segment of the current run is the caller's). Wall time counts from the first ledger entry.

    Examples:
        >>> budget_left("tiny", [{"event": "start", "run": "tiny_a1", "kind": "tiny", "t": 0.0},
        ...                      {"event": "segment", "run": "tiny_a1", "kind": "tiny", "hours": 0.5, "t": 1800.0}],
        ...             now=3600.0)["attempts_left"]
        1
    """
    entries = ledger() if entries is None else entries
    now = time.time() if now is None else now
    started = {e["run"] for e in entries if e.get("event") == "start" and e.get("kind") == kind}
    hours = sum(e.get("hours", 0.0) for e in entries if e.get("event") == "segment")
    t0 = min((e["t"] for e in entries if "t" in e), default=now)
    cap = LIMITS["tiny_attempts"] if kind == "tiny" else LIMITS["full_runs"]
    return {"attempts_used": len(started), "attempts_left": cap - len(started),
            "training_hours_left": LIMITS["training_hours_total"] - hours,
            "wall_hours_left": LIMITS["wall_hours_total"] - (now - t0) / 3600, "started": sorted(started)}


# ------------------------------------------------------------------ data
def load_train() -> tuple[list, list, dict]:
    """E03 training partition through E06's validated reader: (optimisation rows, train-val rows, rejections)."""
    return B.train_rows()


def diag_sample() -> list[dict]:
    """Fixed 16-row diagnostic sample: 8 optimisation + 8 train-val rows in sha256 order (never dev/eval)."""
    opt, val, _ = load_train()
    pick = lambda rows, part: [{"word": w, "target": k, "part": part}  # noqa: E731
                               for w, k in sorted(rows, key=lambda r: sha_text(f"{DIAG_SEED}:{r[0]}:{r[1]}"))[:8]]
    return pick(opt, "train") + pick(val, "train_val")


def select_tiny(rows: list[tuple[str, str]], cfg: dict = TINY) -> list[tuple[str, str]]:
    """The 32 memorisation pairs: unique inputs with one target, distinct targets, `per_bin` per length bin.

    Ordered by sha256(seed:word); depends only on the training partition, never on model outputs.

    Examples:
        >>> rows = [(w, f"ក{n}") for n, w in enumerate(["cat", "cat", "tiger", "lion", "horse", "zebra"])]
        >>> select_tiny(rows, {"bins": [[1, 4], [5, 9]], "per_bin": 1, "n": 2})
        [('lion', 'ក3'), ('horse', 'ក4')]
    """
    by_word, by_tgt = Counter(w for w, _ in rows), Counter(k for _, k in rows)
    ok = sorted({(w, k) for w, k in rows if by_word[w] == 1 and by_tgt[k] == 1},
                key=lambda r: sha_text(f"{TINY_SEED}:{r[0]}"))
    out = []
    for lo, hi in cfg["bins"]:
        out += [r for r in ok if lo <= len(r[0]) <= hi][:cfg["per_bin"]]
    if len(out) != cfg["n"] or len({w for w, _ in out}) != cfg["n"] or len({k for _, k in out}) != cfg["n"]:
        raise ValueError(f"tiny selection failed: {len(out)} rows")
    return out


def check_disjoint(rows: list[tuple[str, str]]) -> dict:
    """Training rows must not contain any dev or eval input word (split protection)."""
    held = {i.word for i in e06.items("dev")} | {i.word for i in e06.items("eval")}
    bad = sorted({w for w, _ in rows} & held)
    if bad:
        raise ValueError(f"training rows overlap dev/eval: {bad[:5]}")
    return {"rows": len(rows), "overlap_with_dev_eval": 0}


# ------------------------------------------------------------------ model
ATTN, FFN = ("q", "k", "v", "o"), ("wi_0", "wi_1", "wo")


def lora_cls():
    """E06's `LoRALinear` plus a read-only `weight` (the frozen base weight): T5's gated FFN reads `wo.weight.dtype`.

    The forward pass is E06's unchanged: `base(x) + (alpha / r) · B(A(dropout(x)))`, B initialised to zero.
    """
    base = B.lora_linear_cls()

    class LoRALinear(base):
        @property
        def weight(self):
            return self.base.weight

    return LoRALinear


def build(adapter: dict | None, base_dropout: float = 0.1, seed: int = 0):
    """The pinned ByT5-small (frozen) with LoRA on the configured targets ("attn": q/k/v/o of every attention
    block; "ffn": wi_0/wi_1/wo of every gated feed-forward). `adapter=None` → the base model."""
    import torch
    from transformers import T5ForConditionalGeneration
    torch.set_num_threads(LIMITS["threads"])
    B.check_base()
    model = T5ForConditionalGeneration.from_pretrained(str(B.BASE), dtype=torch.float32, dropout_rate=base_dropout)
    return insert_adapters(model, adapter, seed)


def insert_adapters(model, adapter: dict | None, seed: int = 0):
    """Freeze every base weight and wrap the configured linear layers with LoRA (in place; returns the model)."""
    import torch
    from transformers.models.t5.modeling_t5 import T5Attention, T5DenseGatedActDense
    for p in model.parameters():
        p.requires_grad_(False)
    if adapter:
        L = lora_cls()
        torch.manual_seed(seed)
        for m in list(model.modules()):
            names = (ATTN if isinstance(m, T5Attention) and "attn" in adapter["targets"] else
                     FFN if isinstance(m, T5DenseGatedActDense) and "ffn" in adapter["targets"] else ())
            for n in names:
                setattr(m, n, L(getattr(m, n), adapter["r"], adapter["alpha"], adapter["dropout"]))
    return model


def adapter_groups(model) -> dict:
    """Trainable parameter counts by location (encoder self, decoder self, decoder cross, encoder/decoder FFN)."""
    g = Counter()
    for n, p in model.named_parameters():
        if p.requires_grad:
            side = "encoder" if n.startswith("encoder.") else "decoder"
            kind = "cross" if "EncDecAttention" in n else "self" if "SelfAttention" in n else "ffn"
            g[f"{side}_{kind}"] += p.numel()
    return dict(g)


def load_adapter(model, path: Path) -> dict:
    """Load adapter tensors strictly: every adapter key present, no unknown keys."""
    import torch
    sd = torch.load(path, weights_only=True)
    r = model.load_state_dict(sd, strict=False)
    missing = [k for k in r.missing_keys if k.endswith((".A", ".B"))]
    if missing or r.unexpected_keys:
        raise RuntimeError(f"adapter mismatch: missing {missing[:3]}, unexpected {r.unexpected_keys[:3]}")
    return {"tensors": len(sd)}


def e06_model():
    """E06's selected adapter on the base model, exactly as E06 built it."""
    if e06.sha(E06_ADAPTER) != E06_ADAPTER_SHA:
        raise RuntimeError("E06 best.pt hash changed")
    r = B.RECIPE
    m = build({"r": r["lora_r"], "alpha": r["lora_alpha"], "dropout": r["lora_dropout"], "targets": ["attn"]},
              base_dropout=r["base_dropout"], seed=r["seed"])
    load_adapter(m, E06_ADAPTER)
    return m


def decode(ids: list[int]) -> tuple[str | None, str]:
    """E06 decoding plus a `no_eos` status when generation hit the length limit (truncation is never silent).

    Examples:
        >>> decode([228, 161, 131, 1]), decode([228, 161, 131])
        (('ក', 'ok'), (None, 'no_eos'))
    """
    if B.EOS not in ids and len(ids) >= 1 and any(i >= B.OFFSET for i in ids):
        return None, "no_eos"
    return B.decode_ids(ids)


def gen(model, srcs: list[list[int]], beam: int = 1, ret: int = 1, bs: int = 32, max_new: int = MAX_NEW,
        early_stopping: bool | str = True, length_penalty: float = 0.0):
    """Deterministic generation in eval mode from encoded sources.

    Returns per source: `[(text, status, score, raw_ids)]`, best first. Greedy score = sum of chosen log-probs;
    beam score = sum log-prob / length**length_penalty.

    The defaults are the registered E07/E08 beam call. E09 found that `early_stopping=True` ends the search once
    `beam` hypotheses have finished, so short byte sequences that end early can fill the beam before the most
    probable one finishes; `early_stopping="never"` is canonical beam search (Transformers' note).
    """
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(srcs), bs):
            chunk = srcs[i:i + bs]
            S = max(map(len, chunk))
            ids = torch.tensor([s + [B.PAD] * (S - len(s)) for s in chunk])
            kw = dict(num_beams=beam, num_return_sequences=ret, length_penalty=length_penalty,
                      early_stopping=early_stopping) if beam > 1 \
                else dict(num_beams=1)
            g = model.generate(input_ids=ids, attention_mask=(ids != B.PAD).long(), max_new_tokens=max_new,
                               do_sample=False, output_scores=True, return_dict_in_generate=True, **kw)
            seqs = g.sequences.tolist()
            if beam > 1:
                scores = g.sequences_scores.tolist()
            else:
                lp = torch.stack([torch.log_softmax(s, -1) for s in g.scores], 1)       # (n, steps, vocab)
                tok = g.sequences[:, 1:]
                chosen = lp.gather(2, tok.unsqueeze(-1)).squeeze(-1)
                alive = torch.ones_like(chosen, dtype=torch.bool)
                alive[:, 1:] = (tok[:, :-1] != B.EOS).cumprod(1).bool()                 # up to and incl. EOS
                scores = (chosen * alive).sum(1).tolist()
            for n in range(len(chunk)):
                res = []
                for j in range(ret):
                    raw = seqs[n * ret + j][1:]
                    if B.EOS in raw:                         # drop the padding after a finished sequence
                        raw = raw[:raw.index(B.EOS) + 1]
                    t, st = decode(raw)
                    res.append((t, st, round(float(scores[n * ret + j]), 6), raw))
                out.append(res)
    return out


def words_gen(model, words: list[str], beam: int = 1, ret: int = 1, bs: int = 32):
    return gen(model, [B.encode_src(w) for w in words], beam, ret, bs)


# ------------------------------------------------------------------ losses
LEAD, MIDS = 0xE1, (0x9E, 0x9F)


def byte_class(label: int) -> str:
    """Class of a target position: Khmer UTF-8 lead byte (0xE1), block byte (0x9E/0x9F), final byte, EOS, other.

    Examples:
        >>> [byte_class(i) for i in (0xE1 + 3, 0x9E + 3, 0x80 + 3, 1, 32 + 3)]
        ['lead', 'block', 'final', 'eos', 'other']
    """
    if label == B.EOS:
        return "eos"
    b = label - B.OFFSET
    return "lead" if b == LEAD else "block" if b in MIDS else "final" if 0x80 <= b <= 0xBF else "other"


def token_losses(model, srcs: list[list[int]], tgts: list[list[int]], bs: int = 32) -> list[list[tuple[int, float]]]:
    """Per example: `[(label, −log p)]` for every non-ignored target position (teacher forcing, eval mode)."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(srcs), bs):
            s, t = srcs[i:i + bs], tgts[i:i + bs]
            S, T = max(map(len, s)), max(map(len, t))
            ids = torch.tensor([x + [B.PAD] * (S - len(x)) for x in s])
            labels = torch.tensor([x + [-100] * (T - len(x)) for x in t])
            logits = model(input_ids=ids, attention_mask=(ids != B.PAD).long(), labels=labels).logits
            nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100,
                                                    reduction="none")
            for n in range(len(s)):
                out.append([(int(labels[n, j]), float(nll[n, j])) for j in range(len(t[n]))])
    return out


def loss_summary(tl: list[list[tuple[int, float]]]) -> dict:
    """Mean loss per target byte, per byte class, per example (sum), and the share of each class."""
    flat = [x for ex in tl for x in ex]
    cls = Counter(byte_class(l) for l, _ in flat)
    by = {c: round(mean(v for l, v in flat if byte_class(l) == c), 4) for c in cls}
    return {"per_byte": round(mean(v for _, v in flat), 4), "by_class": by,
            "class_share": {c: round(n / len(flat), 3) for c, n in cls.items()},
            "per_example_sum": round(mean(sum(v for _, v in ex) for ex in tl), 3), "positions": len(flat)}


def derangement(n: int, seed: str) -> list[int]:
    """A fixed permutation without fixed points (shuffled-input control).

    Examples:
        >>> p = derangement(10, "x"); sorted(p) == list(range(10)) and all(p[i] != i for i in range(10))
        True
    """
    rng = random.Random(seed)
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return p


def input_dependence(model, rows: list[tuple[str, str]], seed: str) -> dict:
    """Correct vs shuffled vs empty inputs on the same targets: loss and greedy-output changes."""
    words, tgts = [w for w, _ in rows], [B.encode_tgt(k) for _, k in rows]
    perm = derangement(len(rows), seed)
    srcs = {"correct": [B.encode_src(w) for w in words], "shuffled": [B.encode_src(words[p]) for p in perm],
            "empty": [[B.EOS] for _ in words]}
    loss = {k: loss_summary(token_losses(model, s, tgts)) for k, s in srcs.items()}
    g = [r[0][0] for r in gen(model, srcs["correct"])]
    e = gen(model, [[B.EOS]])[0][0][0]
    top = Counter(g).most_common(1)[0]
    return {"rows": len(rows), "loss": loss,
            "loss_gap_shuffled_minus_correct": round(loss["shuffled"]["per_byte"] - loss["correct"]["per_byte"], 4),
            "loss_gap_empty_minus_correct": round(loss["empty"]["per_byte"] - loss["correct"]["per_byte"], 4),
            "greedy_distinct_outputs": len(set(g)), "most_common_output": [top[0], top[1]],
            "empty_input_output": e, "outputs_equal_to_empty_input_output": sum(x == e for x in g),
            "exact_correct_input": sum(x == k for x, (_, k) in zip(g, rows))}


# ------------------------------------------------------------------ steps: environment and reproduction
def env() -> dict:
    import torch
    import transformers
    try:
        import peft
        peft_v = peft.__version__
    except ImportError:
        peft_v = None
    opt, val, rej = load_train()
    files = ["scripts/e07.py", "scripts/e06_byt5.py", "scripts/e06.py", "scripts/e05.py",
             "research/requirements-byt5.txt", "research/requirements-byt5.lock.txt"]
    out = {"git": git_rev(), "python": sys.version.split()[0], "platform": platform.platform(),
           "torch": torch.__version__, "transformers": transformers.__version__,
           "peft": peft_v or "not installed (LoRA is the repository's own 20-line `LoRALinear`, checked against "
                             "PEFT's documented defaults in reports/pipeline_audit.md)",
           "cpu_threads": LIMITS["threads"], "cpu_count": os.cpu_count(), "limits": LIMITS,
           "base": {"revision": B.REVISION, "sha256": B.check_base(),
                    "config": json.loads((B.BASE / "config.json").read_text()),
                    "generation_config": json.loads((B.BASE / "generation_config.json").read_text())},
           "e06_adapter": {"path": str(E06_ADAPTER.relative_to(ROOT)), "sha256": e06.sha(E06_ADAPTER)},
           "source_sha256": {f: e06.sha(ROOT / f) for f in files if (ROOT / f).exists()},
           "data": {"train.tsv": e06.sha(e06.E03 / "train.tsv"), "optimisation_rows": len(opt),
                    "train_val_rows": len(val), "rejected": rej, "optimisation_hash": rows_hash(opt),
                    "train_val_hash": rows_hash(val),
                    "dev_references.jsonl": e06.sha(e06.E03 / "dev_references.jsonl"),
                    "eval_references.jsonl": e06.sha(e06.E03 / "eval_references.jsonl"),
                    "dev_items": len(e06.items("dev")), "dev_primary": len(e06.primary(e06.items("dev"))),
                    "eval_items": len(e06.items("eval")), "eval_primary": len(e06.primary(e06.items("eval"))),
                    "pool": e06.sha(e06.POOL), "split_ids": "E03 parts train / train_val; dev/eval references "
                                                           "split 'final' with review.partition 'E03 dev'/'E03 eval'"},
           "split_protection": check_disjoint(opt + val)}
    return write_json(REPORTS / "environment.json", out) and out


def reproduce() -> dict:
    """E06's near-constant generation on the fixed diagnostic sample, with every raw id saved."""
    m = e06_model()
    rows = diag_sample()
    ids, mask, labels = B.batch_tensors([(r["word"], r["target"]) for r in rows])
    res = gen(m, [B.encode_src(r["word"]) for r in rows])
    beam = gen(m, [B.encode_src(r["word"]) for r in rows], beam=4, ret=4)
    ex = []
    for n, r in enumerate(rows):
        inp = [i for i in ids[n].tolist()]
        lab = labels[n].tolist()
        ex.append({**r, "input_ids": inp, "attention_mask": mask[n].tolist(),
                   "decoded_input": bytes(i - 3 for i in inp if i >= 3).decode("utf-8"),
                   "labels": lab, "decoded_target": B.decode_ids([x for x in lab if x != -100])[0],
                   "generated_ids": res[n][0][3], "output": res[n][0][0], "status": res[n][0][1],
                   "beam4": [t for t, *_ in beam[n]]})
    outs = [e["output"] for e in ex]
    out = {"adapter": str(E06_ADAPTER.relative_to(ROOT)), "adapter_sha256": E06_ADAPTER_SHA, "decode": "greedy, eval mode",
           "distinct_outputs": len(set(outs)), "rows": len(rows), "most_common": Counter(outs).most_common(3),
           "exact": sum(e["output"] == e["target"] for e in ex), "examples": ex}
    return write_json(REPORTS / "e06_reproduction.json", out) and {k: v for k, v in out.items() if k != "examples"}


# ------------------------------------------------------------------ step: audit
def audit() -> dict:
    """Checks 1–10 of PLAN.md §2 plus the E06 loss decomposition. Every check records evidence."""
    import torch
    opt, val, _ = load_train()
    chk = {}
    # 1–2: encoding and alignment
    sample = [(w, k) for w, k in opt[:6]]
    ids, mask, labels = B.batch_tensors(sample)
    rows_in = [bytes(i - 3 for i, m_ in zip(ids[n].tolist(), mask[n].tolist()) if m_ and i >= 3).decode() for n in range(len(sample))]
    rows_tg = [B.decode_ids([x for x in labels[n].tolist() if x != -100])[0] for n in range(len(sample))]
    chk["1_inputs_distinct_and_aligned"] = {
        "pass": rows_in == [w.lower() for w, _ in sample] and
        len({tuple(ids[n][mask[n].bool()].tolist()) for n in range(len(sample))}) == len(sample),
        "decoded_unmasked_inputs": rows_in, "words": [w for w, _ in sample],
        "note": "decoded input bytes drop EOS (id 1 < offset 3); unmasked id rows are pairwise distinct"}
    chk["2_targets_aligned"] = {"pass": rows_tg == [k for _, k in sample], "decoded": rows_tg}
    # 3–5: padding, masks, EOS, shift
    lens = [len(B.encode_tgt(k)) for _, k in sample]
    eos_ok = all(labels[n, lens[n] - 1].item() == B.EOS for n in range(len(sample)))
    pad_in_loss = int(((labels == B.PAD) & (labels != -100)).sum())
    m = build(None)
    out = m(input_ids=ids, attention_mask=mask, labels=labels)
    nll = torch.nn.functional.cross_entropy(out.logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
    manual = float(nll.sum() / (labels != -100).sum())
    dec_in = m._shift_right(labels)
    exp_dec = torch.cat([torch.zeros(len(sample), 1, dtype=torch.long), labels[:, :-1].clamp(min=0)], 1)
    chk["3_padding_and_masks"] = {"pass": bool((mask.sum(1) == torch.tensor([len(B.encode_src(w)) for w, _ in sample])).all())
                                  and pad_in_loss == 0,
                                  "attention_mask_counts_equal_source_lengths": True, "pad_ids_in_loss_positions": pad_in_loss}
    chk["4_label_padding_ignored"] = {"pass": abs(manual - float(out.loss)) < 1e-5, "model_loss": float(out.loss),
                                      "manual_mean_over_non_ignored": manual,
                                      "loss_positions": int((labels != -100).sum()), "target_bytes_plus_eos": sum(lens)}
    chk["5_shift_and_eos"] = {"pass": eos_ok and bool((dec_in == exp_dec).all()) and m.config.decoder_start_token_id == 0,
                              "eos_last_label": eos_ok, "decoder_input_equals_[start]+labels[:-1] (−100→pad)":
                                  bool((dec_in == exp_dec).all()),
                              "config_ids": {k: getattr(m.config, k) for k in ("pad_token_id", "eos_token_id",
                                                                               "decoder_start_token_id", "vocab_size")}}
    base_logits = out.logits.detach()
    del m
    # 6–7: adapter insertion, optimiser, gradients, updates (E06 attention-only recipe and E07 tiny recipe)
    for name, ad, bd in (("e06_attn_r16", {"r": 16, "alpha": 32, "dropout": 0.05, "targets": ["attn"]}, 0.1),
                         ("e07_tiny", TINY["adapter"], TINY["base_dropout"])):
        m = build(ad, bd)
        groups = adapter_groups(m)
        params = [p for p in m.parameters() if p.requires_grad]
        o = torch.optim.AdamW(params, lr=1e-3)
        names = [n for n, p in m.named_parameters() if p.requires_grad]
        base_before = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad and "block.0." in n}
        before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
        m.train()
        grads = {}
        for step in range(2):
            loss = m(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            g = {n: p.grad for n, p in m.named_parameters() if p.requires_grad}
            grads[step] = {"finite": all(torch.isfinite(x).all().item() for x in g.values() if x is not None),
                           "none": sum(x is None for x in g.values()),
                           "nonzero_A": sum(bool(x.abs().sum() > 0) for n, x in g.items() if n.endswith(".A") and x is not None),
                           "nonzero_B": sum(bool(x.abs().sum() > 0) for n, x in g.items() if n.endswith(".B") and x is not None)}
            o.step()
            o.zero_grad()
        after = {n: p.detach() for n, p in m.named_parameters() if p.requires_grad}
        changed = sum(not torch.equal(before[n], after[n]) for n in names)
        base_same = all(torch.equal(v, dict(m.named_parameters())[n]) for n, v in base_before.items())
        opt_ids = {id(p) for gr in o.param_groups for p in gr["params"]}
        chk[f"6_7_adapters_{name}"] = {
            "pass": changed == len(names) and base_same and grads[1]["finite"] and grads[1]["none"] == 0
                    and opt_ids == {id(p) for p in params},
            "trainable_by_location": groups, "trainable_tensors": len(names), "optimizer_params_equal_trainable": opt_ids == {id(p) for p in params},
            "grads_step0": grads[0], "grads_step1": grads[1], "changed_after_2_updates": changed,
            "base_weights_unchanged": base_same, "names_sample": names[:4] + names[-2:],
            "note": "at step 0 B = 0, so A gets zero gradient by construction; after one update A grads are nonzero"}
        if name == "e07_tiny":
            # 8: save → reload → identical generation; and the zero-init adapter reproduces the base logits
            fresh = build(ad, bd)
            fresh.eval()
            with torch.no_grad():
                same0 = torch.allclose(fresh(input_ids=ids, attention_mask=mask, labels=labels).logits, base_logits, atol=1e-4)
            p = STATE / "audit_adapter.pt"
            STATE.mkdir(parents=True, exist_ok=True)
            torch.save(B.lora_state(m), p)
            a = [r[0][3] for r in words_gen(m, [w for w, _ in sample])]
            re_ = build(ad, bd)
            info = load_adapter(re_, p)
            b = [r[0][3] for r in words_gen(re_, [w for w, _ in sample])]
            chk["8_save_reload"] = {"pass": a == b and same0, "identical_generated_ids": a == b,
                                    "zero_init_adapter_equals_base_logits": same0, **info}
            del fresh, re_
            p.unlink()
        del m, o
    # 9–10: limits, decoding, scores (E06 adapter)
    tl = max(len(k.encode()) for _, k in opt + val)
    m = e06_model()
    words = [w for w, _ in opt[:8]]
    g1 = words_gen(m, words)
    tf = token_losses(m, [B.encode_src(w) for w in words], [B.encode_tgt(t) for t, *_ in (r[0] for r in g1)])
    gaps = [abs(-sum(v for _, v in ex) - r[0][2]) for ex, r in zip(tf, g1)]
    b4 = words_gen(m, words, beam=4, ret=1)
    tfb = token_losses(m, [B.encode_src(w) for w in words], [B.encode_tgt(r[0][0]) for r in b4])
    gaps_b = [abs(-sum(v for _, v in ex) - r[0][2]) for ex, r in zip(tfb, b4)]
    chk["9_byte_limits_and_decoding"] = {
        "pass": tl + 1 <= MAX_NEW and all(r[0][1] == "ok" for r in g1),
        "longest_training_target_bytes": tl, "registered_target_limit_bytes": B.RECIPE["max_tgt_bytes"],
        "e06_max_new_tokens": B.RECIPE["decode"]["max_new_tokens"], "e07_max_new_tokens": MAX_NEW,
        "latent_defect": "E06 allowed 120-byte targets but generated at most 120 new tokens, leaving no room for EOS "
                         "on a 120-byte target, and its decoder did not flag a missing EOS. No E06 target exceeds "
                         f"{tl} bytes, so it never triggered; E07 uses {MAX_NEW} and a `no_eos` status.",
        "statuses": Counter(r[0][1] for r in g1)}
    chk["10_scores"] = {"pass": max(gaps) < 1e-3 and max(gaps_b) < 1e-3, "greedy_max_gap": max(gaps),
                        "beam4_max_gap": max(gaps_b), "definition": "sum of log-probs incl. EOS; length_penalty 0"}
    # E06 decomposition: why can the loss fall while outputs stay constant?
    dv = [(i.word, i.preferred) for i in e06.primary(e06.items("dev"))][:64]
    tgt_counts = Counter(k for _, k in opt)
    first_char = Counter(k[:2] for _, k in opt)
    dep_val = input_dependence(m, val, "e07-audit-val")
    dep_tr = input_dependence(m, [(r["word"], r["target"]) for r in diag_sample()], "e07-audit-diag")
    base = build(None)
    chk["e06_decomposition"] = {
        "train_val": dep_val, "diag_sample": dep_tr,
        "base_model_train_val": loss_summary(token_losses(base, [B.encode_src(w) for w, _ in val],
                                                          [B.encode_tgt(k) for _, k in val])),
        "most_common_training_targets": tgt_counts.most_common(5),
        "most_common_first_two_chars": first_char.most_common(5),
        "dev_words_used_for_generation_only": len(dv),
    }
    out = {"checks": chk, "all_pass": all(v.get("pass", True) for v in chk.values())}
    write_json(REPORTS / "pipeline_audit.json", out)
    return {k: v.get("pass") for k, v in chk.items()} | {"all_pass": out["all_pass"]}


# ------------------------------------------------------------------ training loop (shared by tiny and full)
def lr_factor(k: int, cfg: dict) -> float:
    """LR factor at update k (0-based): linear warm-up, then constant or linear decay to 0 at max_updates.

    Examples:
        >>> c = {"warmup": 10, "schedule": "constant", "max_updates": 100}
        >>> lr_factor(0, c), lr_factor(9, c), lr_factor(99, c)
        (0.1, 1.0, 1.0)
        >>> lr_factor(55, {**c, "schedule": "linear"})
        0.5
    """
    w, n = cfg["warmup"], cfg["max_updates"]
    if k < w:
        return (k + 1) / w
    return 1.0 if cfg["schedule"] == "constant" else max(0.0, (n - k) / (n - w))


def fit(run: str, kind: str, cfg: dict, rows: list[tuple[str, str]], evaluate, eval_at, out: Path,
        guard: Guard | None = None, max_hours: float | None = None, make=None, order=None) -> dict:
    """Train adapters from the base model; resumable from `out/last.pt` (config, data and base hashes validated).

    `order(epoch) -> [row index]` is the epoch's visiting order; default (E07/E08): a seeded permutation of all
    rows. A custom order must be deterministic in the epoch (resume rebuilds it) and be named in `cfg`.

    `evaluate(model) -> dict` with a `key` tuple (higher is better) runs at the updates in `eval_at` and at the
    stop; the best adapter is saved to `out/best.pt`. Stops at max_updates, the time cap (run cap ∩ remaining
    training and wall budget), sustained memory pressure, or `evaluate` returning `stop: True`.
    """
    import torch
    from khmer_translit import experiment
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        return json.loads((out / "done.json").read_text())
    ident = {"cfg": cfg_hash(cfg), "data": rows_hash(rows), "base": B.BASE_SHA256}
    left = budget_left(kind)
    resuming = (out / "last.pt").exists()
    if not resuming and run not in left["started"]:
        if left["attempts_left"] <= 0:
            raise RuntimeError(f"no {kind} attempts left: {left}")
        ledger_add({"event": "start", "run": run, "kind": kind, "cfg_hash": ident["cfg"], "data_hash": ident["data"],
                    "t": time.time()})
    cap_h = min(max_hours or cfg["max_hours"], left["training_hours_left"], left["wall_hours_left"])
    if cap_h <= 0:
        raise RuntimeError(f"budget exhausted: {left}")
    model = (make or build)(cfg["adapter"], cfg["base_dropout"], cfg["seed"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: lr_factor(k, cfg))
    st = {"update": 0, "epoch": 0, "pos": 0, "elapsed_s": 0.0, "best": None, "history": []}
    if resuming:
        ck = torch.load(out / "last.pt", weights_only=False)
        if ck["ident"] != ident:
            raise RuntimeError(f"last.pt does not match this run: {ck['ident']} vs {ident}")
        model.load_state_dict(ck["adapter"], strict=False)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["torch_rng"])
        st = ck["state"]
    trk = experiment.Run(out, EXPERIMENT, run, {"cfg": cfg, "ident": ident, "limits": LIMITS},
                         inputs=[e06.E03 / "train.tsv", B.BASE / "config.json"])
    guard = guard or Guard()
    order = order or (lambda ep: random.Random(f"{cfg['seed']}:order:{ep}").sample(range(len(rows)), len(rows)))
    perm = order(st["epoch"])
    seg_start, seg_from = time.perf_counter(), st["elapsed_s"]
    elapsed = lambda: seg_from + time.perf_counter() - seg_start  # noqa: E731
    seg_cap = cap_h * 3600          # this segment may use at most the remaining budget

    def save_last():
        st["elapsed_s"] = elapsed()
        tmp = out / "last.tmp"
        torch.save({"adapter": B.lora_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "torch_rng": torch.get_rng_state(), "state": st, "ident": ident}, tmp)
        os.replace(tmp, out / "last.pt")

    def do_eval(final=False):
        r = evaluate(model)
        rec = {"update": st["update"], **{k: v for k, v in r.items() if k not in ("key", "stop", "detail")},
               "elapsed_s": round(elapsed(), 1), **({"final_eval": True} if final else {})}
        st["history"].append(rec)
        if st["best"] is None or tuple(r["key"]) > tuple(st["best"]["key"]):
            st["best"] = {"key": list(r["key"]), "update": st["update"], **{k: v for k, v in rec.items() if k != "update"}}
            torch.save(B.lora_state(model), out / "best.pt")
        trk.log(rec, step=st["update"])
        e06_progress(run, rec)
        return r

    stop = None
    tokens_seen = 0
    while st["update"] < cfg["max_updates"]:
        if time.perf_counter() - seg_start > seg_cap:
            stop = "time_limit"
            break
        stop = guard.check()
        if stop:
            break
        model.train()
        loss_sum, acc = 0.0, cfg["grad_accumulation"]
        for _ in range(acc):                         # equal-size micro-batches → mean of micro-batch losses
            if st["pos"] + cfg["batch"] > len(perm):
                st["epoch"] += 1
                st["pos"] = 0
                perm = order(st["epoch"])
            b = [rows[i] for i in perm[st["pos"]:st["pos"] + cfg["batch"]]]
            st["pos"] += cfg["batch"]
            ids, mask, labels = B.batch_tensors(b)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss / acc
            loss.backward()
            loss_sum += float(loss)
            tokens_seen += int((labels != -100).sum())
        loss = loss_sum
        gn = float(torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"]))
        lr = opt.param_groups[0]["lr"]
        opt.step()
        sched.step()
        opt.zero_grad()
        st["update"] += 1
        rec = {"update": st["update"], "loss": round(float(loss), 4), "grad_norm": round(gn, 4), "lr": lr,
               "epoch": st["epoch"], "elapsed_s": round(elapsed(), 1), "rss_mb": round(guard.peak_rss / 2**20)}
        if not all(torch.isfinite(p).all() for p in params[:2]):
            stop = "non_finite"
            break
        trk.log(rec, step=st["update"])
        if st["update"] in eval_at or st["update"] == cfg["max_updates"]:
            r = do_eval()
            save_last()
            if r.get("stop"):
                stop = "all_correct"
                break
        elif st["update"] % 100 == 0:
            save_last()
    stop = stop or "max_updates"
    if stop != "memory_pressure" and (not st["history"] or st["history"][-1]["update"] != st["update"]):
        do_eval(final=True)
    save_last()
    hours = (elapsed() - seg_from) / 3600
    ledger_add({"event": "segment", "run": run, "kind": kind, "hours": round(hours, 4), "updates_to": st["update"],
                "stop": stop, "t": time.time()})
    if stop == "memory_pressure":                  # resumable later; not a finished run
        return {"stop": stop, "updates": st["update"], "resumable": True}
    done = {"run": run, "stop": stop, "updates": st["update"], "epochs_started": st["epoch"] + 1,
            "elapsed_hours": round(elapsed() / 3600, 3), "best": st["best"], "history": st["history"],
            "params": B.counts(model), "trainable_by_location": adapter_groups(model), "rows": len(rows),
            "ident": ident, "peak_rss_mb": round(guard.peak_rss / 2**20), "min_available_mb":
                round((guard.min_avail or 0) / 2**20), "best_sha256": e06.sha(out / "best.pt"),
            "time_cap_hours": round(cap_h, 3), "cfg": cfg}
    write_json(out / "done.json", done)
    ledger_add({"event": "done", "run": run, "kind": kind, "stop": stop, "updates": st["update"], "t": time.time()})
    trk.finish({k: v for k, v in done.items() if k != "history"})
    return done


def e06_progress(run: str, rec: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "progress.log", "a") as f:
        f.write(f"{e06.now()} {run} {json.dumps(rec, ensure_ascii=False)}\n")


# ------------------------------------------------------------------ step: tiny memorisation
def tiny_cfg(attempt: int) -> dict:
    """Attempt 1 is TINY; attempt 2 is TINY with the single change recorded in `reports/tiny_attempt2_change.json`
    (written, with its diagnosis, before attempt 2 starts)."""
    if attempt == 1:
        return TINY
    ch = json.loads((REPORTS / "tiny_attempt2_change.json").read_text())
    if len(ch["change"]) != 1:
        raise ValueError("attempt 2 must change exactly one setting")
    cfg = json.loads(json.dumps(TINY))
    for k, v in ch["change"].items():
        if "." in k:
            a, b = k.split(".")
            cfg[a][b] = v
        else:
            cfg[k] = v
    return cfg


def nfc_eq(a: str | None, b: str) -> bool:
    return a is not None and e06.nfc(a) == e06.nfc(b)


def tiny(attempt: int) -> dict:
    """One 32-pair memorisation attempt from the base model, then the diagnostic gate."""
    opt, _val, _ = load_train()
    pairs = select_tiny(opt)
    check_disjoint(pairs)
    cfg = tiny_cfg(attempt)
    run = f"tiny_a{attempt}"
    out = MODELS / run
    words = [w for w, _ in pairs]

    def evaluate(model):
        g = words_gen(model, words)
        n = sum(nfc_eq(r[0][0], k) for r, (_, k) in zip(g, pairs))
        return {"key": (n,), "exact": n, "distinct": len({r[0][0] for r in g}),
                "stop": cfg["stop_when_all_correct"] and n == len(pairs)}

    done = fit(run, "tiny", cfg, pairs, evaluate, set(range(cfg["eval_every"], cfg["max_updates"] + 1, cfg["eval_every"])), out)
    if done.get("resumable"):
        return done
    # final diagnostics on the best adapter, reloaded from disk
    import torch
    m = build(cfg["adapter"], cfg["base_dropout"], cfg["seed"])
    load_adapter(m, out / "best.pt")
    g = words_gen(m, words)
    tl = token_losses(m, [B.encode_src(w) for w in words], [B.encode_tgt(k) for _, k in pairs])
    dep = input_dependence(m, pairs, f"e07-tiny-{attempt}")
    perm = derangement(len(pairs), f"e07-tiny-{attempt}")
    shuffled_out = [r[0][0] for r in words_gen(m, [words[p] for p in perm])]
    table = [{"word": w, "target": k, "output": r[0][0], "status": r[0][1], "exact_nfc": nfc_eq(r[0][0], k),
              "loss_sum": round(sum(v for _, v in t), 4), "loss_per_byte": round(mean(v for _, v in t), 4),
              "output_when_given_input_of": words[p], "that_output": so}
             for (w, k), r, t, p, so in zip(pairs, g, tl, perm, shuffled_out)]
    # outputs depend on inputs: a shuffled input reproduces the *other* word's target, not this one's
    follows = sum(nfc_eq(so, pairs[p][1]) for p, so in zip(perm, shuffled_out))
    exact = sum(t["exact_nfc"] for t in table)
    reload_same = [r[0][3] for r in g] == [r[0][3] for r in words_gen(m, words)]
    fresh = build(cfg["adapter"], cfg["base_dropout"], cfg["seed"])
    load_adapter(fresh, out / "best.pt")
    reload_same = reload_same and [r[0][3] for r in words_gen(fresh, words)] == [r[0][3] for r in g]
    aud = json.loads((REPORTS / "pipeline_audit.json").read_text())
    gate = {"exact_at_least_30": exact >= TINY_GATE["exact_min"],
            "outputs_depend_on_inputs": dep["greedy_distinct_outputs"] >= 30 and follows >= 30
                                        and dep["loss_gap_shuffled_minus_correct"] > 0,
            "save_reload_preserves_predictions": reload_same, "pipeline_checks_pass": aud["all_pass"]}
    res = {"attempt": attempt, "run": run, "cfg": cfg, "cfg_hash": cfg_hash(cfg), "pairs_hash": rows_hash(pairs),
           "training": {k: v for k, v in done.items() if k not in ("cfg",)},
           "exact_nfc": exact, "n": len(pairs), "distinct_outputs": len({t["output"] for t in table}),
           "byte_loss": loss_summary(tl), "input_dependence": dep,
           "shuffled_inputs_reproduce_the_other_words_target": follows, "gate": gate, "passed": all(gate.values()),
           "table": table, "decode": "greedy, eval mode, NFC exact match", "torch_threads": torch.get_num_threads()}
    write_json(REPORTS / f"tiny_overfit_a{attempt}.json", res)
    write_json(REPORTS / "tiny_overfit.json", {"attempts": sorted(
        {int(p.stem[-1]) for p in REPORTS.glob("tiny_overfit_a*.json")}), "latest": attempt, **res})
    return {k: v for k, v in res.items() if k not in ("table", "cfg")}


# ------------------------------------------------------------------ full adaptation (registered after the gate)
FULL_PATH = EXP / "full_recipe.json"


def full_cfg() -> dict:
    reg = json.loads((EXP / "registration.json").read_text())
    cfg = json.loads(FULL_PATH.read_text())
    if reg["full_recipe_sha256"] != e06.sha(FULL_PATH):
        raise RuntimeError("full_recipe.json differs from the registration")
    return cfg


def gate_passed() -> bool:
    p = REPORTS / "tiny_overfit.json"
    return p.exists() and any(json.loads((REPORTS / f"tiny_overfit_a{a}.json").read_text())["passed"]
                              for a in json.loads(p.read_text())["attempts"])


def register() -> dict:
    """Freeze the full recipe, code and data hashes (after a passed tiny gate, before full adaptation)."""
    if not gate_passed():
        raise RuntimeError("tiny gate not passed: full adaptation is skipped")
    files = ["scripts/e07.py", "scripts/e06_byt5.py", "scripts/e06.py", "scripts/e05.py", "tests/test_e07.py",
             "experiments/E07-byt5-diagnosis/PLAN.md", "experiments/E07-byt5-diagnosis/config.yaml",
             "research/requirements-byt5.txt", "research/requirements-byt5.lock.txt"]
    opt, val, _ = load_train()
    reg = {"registered": e06.now(), "git": git_rev(), "full_recipe_sha256": e06.sha(FULL_PATH),
           "full_recipe": json.loads(FULL_PATH.read_text()), "limits": LIMITS, "tiny": TINY, "tiny_gate": TINY_GATE,
           "source_sha256": {f: e06.sha(ROOT / f) for f in files},
           "data": {"train.tsv": e06.sha(e06.E03 / "train.tsv"), "optimisation_hash": rows_hash(opt),
                    "train_val_hash": rows_hash(val), "dev_references.jsonl": e06.sha(e06.E03 / "dev_references.jsonl"),
                    "pool.jsonl": e06.sha(e06.POOL)},
           "base_sha256": B.BASE_SHA256}
    return write_json(EXP / "registration.json", reg) and reg


def cer_of(top1: str | None, accepted: list[str]) -> float:
    from khmer_translit.evaluation import edit_distance
    t = e06.nfc(top1 or "")
    return min(edit_distance(t, e06.nfc(a)) / max(1, len(a)) for a in accepted)


def dev_selection(model, dev) -> dict:
    """Greedy dev top-1 macro and mean CER (registered selection: top-1, then lower CER, then earlier)."""
    g = words_gen(model, [i.word for i in dev], bs=64)
    hit = {i.id: int(nfc_eq(r[0][0], i.preferred) or any(nfc_eq(r[0][0], a) for a in i.accepted)) for i, r in zip(dev, g)}
    cer = mean(cer_of(r[0][0], i.accepted) for i, r in zip(dev, g))
    top1 = e06.macro(dev, hit)
    return {"key": (top1, -cer), "dev_top1": round(top1, 4), "dev_cer": round(cer, 4), "dev_correct": sum(hit.values()),
            "dev_distinct": len({r[0][0] for r in g})}


def train() -> dict:
    cfg = full_cfg()
    opt, val, _ = load_train()
    check_disjoint(opt)
    dev = e06.primary(e06.items("dev"))
    return fit("full", "full", cfg, opt, lambda m: dev_selection(m, dev), set(cfg["eval_at"]), MODELS / "full")


# ------------------------------------------------------------------ predictions
def selected_model():
    cfg = full_cfg()
    done = json.loads((MODELS / "full" / "done.json").read_text())
    if e06.sha(MODELS / "full" / "best.pt") != done["best_sha256"]:
        raise RuntimeError("best.pt changed")
    m = build(cfg["adapter"], cfg["base_dropout"], cfg["seed"])
    load_adapter(m, MODELS / "full" / "best.pt")
    return m, done


def predict() -> dict:
    """Greedy top-1 and beam-4 top-4 (sum log-prob) for dev, eval and the E06 pool; frozen before scoring."""
    m, done = selected_model()
    sets = {k: v for k, v in e06.prediction_sets().items() if k in ("dev", "eval", "pool")}
    PRED.mkdir(parents=True, exist_ok=True)
    timing = {}
    for name, rows in sets.items():
        p = PRED / f"byt5_e07_{name}.jsonl"
        if p.exists():
            continue
        t = time.perf_counter()
        ok = [(i, w) for i, w in rows if B.fits(w) is None]
        gr = dict(zip([i for i, _ in ok], words_gen(m, [w for _, w in ok])))
        bm = dict(zip([i for i, _ in ok], words_gen(m, [w for _, w in ok], beam=4, ret=4, bs=16)))
        timing[name] = {"words": len(rows), "seconds": round(time.perf_counter() - t, 1)}
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for i, w in rows:
                g, b = gr.get(i), bm.get(i)
                f.write(json.dumps({"id": i, "word": w, "system": "byt5_e07", "checkpoint_update": done["best"]["update"],
                                    "status": "refused_source_too_long" if g is None else "ok",
                                    "greedy": g[0][0] if g and g[0][1] == "ok" else None,
                                    "greedy_status": g[0][1] if g else None,
                                    "beam4": e05.dedup([t for t, s, *_ in (b or []) if s == "ok" and t]),
                                    "beam4_raw": [{"text": t, "status": s, "score": sc} for t, s, sc, _ in (b or [])]},
                                   ensure_ascii=False) + "\n")
        os.replace(tmp, p)
    hashes = {p.name: e06.sha(p) for p in sorted(PRED.glob("byt5_e07_*.jsonl"))}
    write_json(REPORTS / "predictions_frozen.json", {"frozen": e06.now(), "sha256": hashes,
                                                    "checkpoint": done["best"], "timing": timing})
    return {"timing": timing, "hashes": hashes}


def default_eval_preds() -> dict[str, list[str]]:
    """Product default (E02 letters_s0, its own decoder, top-4) on all eval items; cached once, frozen before scoring."""
    p = PRED / "default_e02_eval.jsonl"
    if not p.exists():
        its = e06.items("eval")
        res = e06.default_predict([i.word for i in its])
        with open(p, "w", encoding="utf-8") as f:
            for i, r in zip(its, res):
                f.write(json.dumps({"id": i.id, "word": i.word, "system": "default_e02", "status": r["status"],
                                    "candidates": e05.dedup(r["candidates"]) if r["status"] == "ok" else []},
                                   ensure_ascii=False) + "\n")
    return {r["id"]: r["candidates"] for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]


def len_bin(w: str) -> str:
    n = len(w)
    return "≤4" if n <= 4 else "5–6" if n <= 6 else "7–8" if n <= 8 else "≥9"


def system_table(its, preds: dict[str, list[str]]) -> dict:
    from khmer_translit import evaluation as ev
    sc = ev.score_predictions(its, preds, 4)
    tops = [preds[i.id][0] if preds[i.id] else None for i in its]
    c = Counter(t for t in tops if t)
    return {"top1_macro": round(sc["primary_macro"], 4), "top4_macro": round(sc["topk_macro"], 4),
            "mean_cer": round(sc["mean_cer"], 4), "correct_items": sc["correct_items"], "no_output": sc["abstained"],
            "distinct_top1": len(c), "most_common_top1": c.most_common(1)[0] if c else None,
            "share_most_common_top1": round(c.most_common(1)[0][1] / len(its), 4) if c else None,
            "invalid_orthography_top1": sum(e06.orthography(t) != "ok" for t in tops if t)}, sc["items"]


def analyse() -> dict:
    """Historical E03 eval agreement (top-1 from beam-4 for every system; greedy also for ByT5-E07), uncertainty,
    strata, input dependence, and pool heuristics."""
    import numpy as np
    m, done = selected_model()
    ev_all = e06.items("eval")
    prim = e06.primary(ev_all)
    e07 = {r["id"]: r for r in load_jsonl(PRED / "byt5_e07_eval.jsonl")}
    systems = {"byt5_e07_beam4": {i: r["beam4"] for i, r in e07.items()},
               "byt5_e07_greedy": {i: [r["greedy"]] if r["greedy"] else [] for i, r in e07.items()},
               "byt5_e06": e06.load_pred("byt5", "eval"),
               "default_e02": default_eval_preds(),
               **{g: e06.e04_eval(g) for g in ("G_s0", "G_s1", "G_s2")}}
    for s, p in systems.items():
        e05.check_ids({i.id for i in ev_all}, p)
    tab, per = {}, {}
    for s, p in systems.items():
        tab[s], per[s] = system_table(prim, {i.id: p[i.id] for i in prim})
    gs = ["G_s0", "G_s1", "G_s2"]
    tab["G_mean"] = {k: round(mean(tab[g][k] for g in gs), 4) for k in ("top1_macro", "top4_macro", "mean_cer")}
    idx = e05.boot_indices(len({i.family for i in prim}))
    fm = {s: e05.family_means(prim, {i: v["top1"] for i, v in per[s].items()}) for s in per}
    fm |= {f"{s}@4": e05.family_means(prim, {i: v["topk"] for i, v in per[s].items()}) for s in per}
    fc = {s: e05.family_means(prim, {i: -v["cer"] for i, v in per[s].items()}) for s in per}   # −CER: higher better
    con = {"byt5_e07_minus_G_mean_top1": e05.contrast(fm, ["byt5_e07_beam4"], gs, idx),
           "byt5_e07_minus_default_top1": e05.contrast(fm, ["byt5_e07_beam4"], ["default_e02"], idx),
           "byt5_e07_minus_byt5_e06_top1": e05.contrast(fm, ["byt5_e07_beam4"], ["byt5_e06"], idx),
           "byt5_e07_minus_G_mean_top4_oracle": e05.contrast(fm, ["byt5_e07_beam4@4"], [f"{g}@4" for g in gs], idx),
           "byt5_e07_greedy_minus_beam4_top1": e05.contrast(fm, ["byt5_e07_greedy"], ["byt5_e07_beam4"], idx),
           **{f"byt5_e07_minus_{g}_top1": e05.contrast(fm, ["byt5_e07_beam4"], [g], idx) for g in gs},
           "cer_reduction_byt5_e07_vs_G_mean_pp": e05.contrast(fc, ["byt5_e07_beam4"], gs, idx)}
    strata = {}
    e02_train = {line.split("\t")[0].lower() for line in
                 e06.guard(ROOT / "data/e02/train.tsv").read_text(encoding="utf-8").splitlines()[1:]}
    for name, key in (("length", lambda i: len_bin(i.word)), ("source", lambda i: "+".join(sorted(i.sources))),
                      ("in_default_e02_training", lambda i: str(i.word.lower() in e02_train))):
        for v in sorted({key(i) for i in prim}):
            sub = [i for i in prim if key(i) == v]
            strata[f"{name}={v}"] = {"items": len(sub), **{s: round(e06.macro(sub, {i.id: per[s][i.id]["top1"] for i in sub}), 4)
                                                          for s in ("byt5_e07_beam4", "byt5_e06", "default_e02", *gs)}}
    amb = [i for i in ev_all if "pronunciation_ambiguous" in i.flags]
    strata["ambiguous_track (excluded from primary)"] = {"items": len(amb), **{
        s: round(system_table(amb, {i.id: systems[s][i.id] for i in amb})[0]["top1_macro"], 4)
        for s in ("byt5_e07_beam4", "default_e02", *gs)}}
    status = Counter(r["greedy_status"] for r in e07.values())
    beam_status = Counter(x["status"] for r in e07.values() for x in r["beam4_raw"])
    # input dependence on fixed samples (train-val: unseen in optimisation; diagnostic sample)
    _opt, val, _ = load_train()
    dep = {"train_val": input_dependence(m, val, "e07-dep-val"),
           "diag_sample": input_dependence(m, [(r["word"], r["target"]) for r in diag_sample()], "e07-dep-diag"),
           "e06_train_val": json.loads((REPORTS / "pipeline_audit.json").read_text())["checks"]["e06_decomposition"]["train_val"]}
    # pool: predictions and heuristics only
    import e02_diagnostics as DG
    km_lex, lat_lex = DG.load_lexicon()
    from khmer_translit import phonetic_source as ps
    cmu, _ = ps.cmudict()
    pool = e06.load_pool()
    pr = {r["id"]: r for r in load_jsonl(PRED / "byt5_e07_pool.jsonl")}
    others = {"byt5_e06": e06.load_pred("byt5", "pool"), "default_e02": e06.load_pred("default_e02", "pool"),
              **{g: e06.load_pred(g, "pool") for g in gs}}
    ph = lambda w: " ".join(cmu[w][0]) if cmu.get(w) else None  # noqa: E731
    pool_sys = {"byt5_e07": {i: r["beam4"] for i, r in pr.items()}, **others}
    pool_diag = {}
    for s, p in pool_sys.items():
        tops = {r["id"]: (p.get(r["id"]) or [None])[0] for r in pool}
        pool_diag[s] = {"no_output": sum(t is None for t in tops.values()),
                        "orthography_ok_top1": sum(e06.orthography(t) == "ok" for t in tops.values() if t),
                        "consonant_screen_level0_top1 (heuristic, not accuracy)": sum(
                            bool(t) and e06.consonant_screen(pw, t, ph(pw), km_lex, lat_lex) == "0"
                            for pw, t in ((r["word"], tops[r["id"]]) for r in pool if r["track"] == "primary")),
                        "distinct_top1": len({t for t in tops.values() if t}),
                        "top1_is_lexicon_word": sum(t in km_lex for t in tops.values() if t)}
    res = {"label": "historical research comparison on E03 eval (mined, unreviewed, consonant-screened; already used "
                    "by E03–E06), not a fresh confirmatory phonetic benchmark",
           "checkpoint": done["best"], "updates_completed": done["updates"], "stop": done["stop"],
           "decode": {"primary_top1": "beam 4, sum log-prob (length_penalty 0), top-1 — the same as the G baselines",
                      "greedy": "reported separately", "oracle": "top-4 of the same beam, equal budget for every system",
                      "max_new_tokens": MAX_NEW, "default_e02": "its own Transliterator.predict_many(top_k=4)"},
           "historical_eval": {"items": len(prim), "families": len({i.family for i in prim}), "systems": tab,
                               "contrasts": con, "ci_note": "paired family bootstrap (10,000, seed 20261027); "
                               "conditional on these checkpoints; one ByT5 run measures no seed variability"},
           "strata": strata, "byt5_e07_output_validity": {"greedy": dict(status), "beam4_all_candidates": dict(beam_status)},
           "input_dependence": dep, "pool": {"label": "fresh E06 pool, no references: operational checks and "
                                                      "heuristics only", "diagnostics": pool_diag,
                                             "examples": [{"word": r["word"], "category": r["category"],
                                                           **{s: (p.get(r["id"]) or [None])[0] for s, p in pool_sys.items()}}
                                                          for r in pool]}}
    _ = np
    write_json(EXP / "results.json", res)
    (EXP / "results").mkdir(exist_ok=True)
    with open(EXP / "results" / "per_item.jsonl", "w", encoding="utf-8") as f:
        for i in prim:
            f.write(json.dumps({"id": i.id, "word": i.word, "family": i.family, "accepted": i.accepted,
                                **{s: {"top1": per[s][i.id]["top1"], "top4": per[s][i.id]["topk"],
                                       "cer": round(per[s][i.id]["cer"], 4), "prediction": per[s][i.id]["prediction"]}
                                   for s in per}}, ensure_ascii=False) + "\n")
    return {"systems": tab, "contrasts": con}


# ------------------------------------------------------------------ step: cost
def cost() -> dict:
    import torch
    t = time.perf_counter()
    m, done = selected_model()
    load_s = time.perf_counter() - t
    words = [i.word for i in e06.primary(e06.items("dev"))][:40]
    lat = {}
    for beam in (1, 4):
        ts = []
        for w in words:
            t = time.perf_counter()
            words_gen(m, [w], beam=beam, ret=1)
            ts.append(time.perf_counter() - t)
        lat[f"batch1_beam{beam}_ms"] = {"median": round(median(ts) * 1000, 1), "max": round(max(ts) * 1000, 1)}
    t = time.perf_counter()
    words_gen(m, [i.word for i in e06.primary(e06.items("dev"))], bs=32)
    n = len(e06.primary(e06.items("dev")))
    thr = n / (time.perf_counter() - t)
    import resource
    out = {"hardware": f"{platform.processor() or platform.machine()}, {LIMITS['threads']} threads, CPU only",
           "cold_load_s (base + adapter, warm page cache)": round(load_s, 2), **lat,
           "greedy_batch32_words_per_s": round(thr, 1), "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
           "adapter_bytes": (MODELS / "full" / "best.pt").stat().st_size,
           "base_bytes": (B.BASE / "pytorch_model.bin").stat().st_size, "torch_threads": torch.get_num_threads(),
           "letters_model_reference": "E05 reports/cost.json: G (NumPy) beam-8 21.9 ms/word"}
    return write_json(REPORTS / "cost.json", out) and out


STEPS = {"env": env, "reproduce": reproduce, "audit": audit, "register": register, "train": train,
         "predict": predict, "analyse": analyse, "cost": cost}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=[*STEPS, "tiny"])
    ap.add_argument("--attempt", type=int, default=1)
    a = ap.parse_args()
    r = tiny(a.attempt) if a.step == "tiny" else STEPS[a.step]()
    print(json.dumps(r, ensure_ascii=False, default=str)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
