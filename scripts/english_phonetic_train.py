#!/usr/bin/env python3
"""Experiment E02: train, pretrain, export and verify the English → Khmer phonetic models (CPU).

Run from the repository root (the runner `scripts/english_phonetic_run.py` calls these in its fixed queue):

    .venv/bin/python scripts/english_phonetic_train.py pretrain --seed 0 --out models/e02/p2g_s0
    .venv/bin/python scripts/english_phonetic_train.py train --arm phones_p2g --seed 0 --init models/e02/p2g_s0/best.pt \
        --out models/e02/phones_p2g_s0
    .venv/bin/python scripts/english_phonetic_train.py train --arm letters --seed 0 --out models/e02/letters_s0
    .venv/bin/python scripts/english_phonetic_train.py export --out models/e02/letters_s0
    .venv/bin/python scripts/english_phonetic_train.py verify --out models/e02/letters_s0

**Arms** (one shared model, vocabulary and recipe; `experiments/E02-english-phonetic/config.yaml`):
- `letters`: source = `<task:en>` + letters.
- `phones`: source = `<task:en>` + letters + `<sep>` + the first CMUdict reading (or `<no_pron>`); during training
  10% of examples get `<no_pron>` instead of their phones (private RNG), so a dictionary miss at run time is a
  case the model has seen.
- `phones_p2g`: `phones`, initialised from Khmer phones → Khmer spelling pretraining (`pretrain`), plus one P2G
  replay batch after every 8 supervised updates (private RNG; the replay update uses the current learning rate
  and does not advance the supervised schedule).

**Pairing:** for a seed, every arm starts from the same initial tensors (`torch.manual_seed(seed)` before building
the model; the initial-state hash is recorded and compared by the runner) and sees the same supervised batches
(private order RNG per epoch; batch-id hash recorded). Dropout masks differ once inputs differ.

**Selection:** after each epoch, greedy top-1 multi-reference exact match on the primary-eligible dev items,
macro over families (`khmer_translit.evaluation`); ties → lower training-validation loss → earlier epoch. Only
`data/e02/dev_references.jsonl` is read; final references are never opened here.

**Resumable:** `last.pt` (model, optimizer, scheduler, step, epoch, best, torch RNG) is written after every epoch;
`train`/`pretrain` resume from it when present. `done.json` marks a completed job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from khmer_translit import evaluation as ev  # noqa: E402
from khmer_translit import experiment  # noqa: E402
from khmer_translit import phonetic_source as ps  # noqa: E402
from khmer_translit import reference_data as rd  # noqa: E402

DATA = ROOT / "data" / "e02"
CONFIG = ROOT / "experiments" / "E02-english-phonetic" / "config.yaml"
ARMS = ("letters", "phones", "phones_p2g")
PRIMARY_EXCLUDE = {"pronunciation_ambiguous", "possible_translation"}
EXPECTED = {
    "paths": {"runs_dir": "models/e02"},
    "model": {"d": 192, "layers": 3, "heads": 4, "ff": 512, "dropout": 0.1},
    "train": {"epochs": 40, "batch": 128, "lr": 0.001, "weight_decay": 0.01, "warmup": 500, "label_smoothing": 0.1,
              "grad_clip": 1.0, "threads": 6, "phone_dropout": 0.1},
    "p2g": {"epochs": 10, "max_hours": 2.0, "lr": 0.001, "warmup": 500, "batch": 128, "replay_every": 8,
            "replay_batch": 128},
    "eval": {"beam": 4, "top_k": 4},
}


def load_cfg(overrides: list[str]) -> dict:
    """The resolved E02 config (strict: every key required)."""
    return experiment.load_config(CONFIG, EXPECTED, overrides)


def load_vocab() -> tuple[ps.SourceVocab, str, dict]:
    """Source vocabulary, target characters and the raw vocab file of the data build.

    Raises:
        ValueError: If the data build used another encoder version.
    """
    v = json.loads((DATA / "vocab.json").read_text(encoding="utf-8"))
    if v["encoder_version"] != ps.ENCODER_VERSION:
        raise ValueError(f"data built with {v['encoder_version']}, code is {ps.ENCODER_VERSION}")
    vocab = ps.SourceVocab(v["km_phones"])
    if vocab.tokens != v["src_tokens"]:
        raise ValueError("source vocabulary in vocab.json does not match the code")
    return vocab, v["tgt_chars"], v


def load_supervised() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """`(train, train_val)` rows `(word, khmer)` from `data/e02/train.tsv`."""
    rows = [l.split("\t") for l in (DATA / "train.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    return [(r[0], r[1]) for r in rows if r[5] == "train"], [(r[0], r[1]) for r in rows if r[5] == "train_val"]


def load_p2g() -> tuple[list[tuple[list[str], str]], list[tuple[list[str], str]]]:
    """`(p2g_train, p2g_dev)` rows `(khmer phones, khmer)`."""
    rows = [l.split("\t") for l in (DATA / "p2g.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    return ([(r[0].split(), r[1]) for r in rows if r[2] == "p2g_train"],
            [(r[0].split(), r[1]) for r in rows if r[2] == "p2g_dev"])


def load_dev() -> list[rd.ReferenceItem]:
    """Primary-eligible dev items (the only references the trainer reads)."""
    items = rd.load_reference_items(DATA / "dev_references.jsonl")
    assert all(i.split == "dev" for i in items), "dev_references.jsonl must hold dev items only"
    return [i for i in items if not PRIMARY_EXCLUDE & set(i.flags)]


def source_mode(arm: str) -> str:
    """Encoder mode of an arm: "letters" or "phones"."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return "letters" if arm == "letters" else "phones"


def encode_word(word: str, mode: str, vocab: ps.SourceVocab, drop: bool = False) -> list[int]:
    """Source ids exactly as the product computes them (`phonetic_source`), with optional phone dropout."""
    pron = ps.get_pronunciation(word) if mode == "phones" else None
    return ps.encode_source(word, pron, mode, vocab, drop_phones=drop)


def target_ids(khmer: str, tgt: str) -> list[int]:
    """BOS + character ids + EOS (ids = index in `tgt` + 3).

    Raises:
        ValueError: A character outside the target vocabulary, or longer than `MAX_TGT`.
    """
    if len(khmer) > ps.MAX_TGT:
        raise ValueError(f"target longer than {ps.MAX_TGT}: {khmer}")
    return [ps.BOS] + [tgt.index(c) + 3 for c in khmer] + [ps.EOS]


def pad(seqs: list[list[int]]):
    """Right-pad id lists into a LongTensor (PAD = 0)."""
    import torch
    t = torch.zeros(len(seqs), max(map(len, seqs)), dtype=torch.long)
    for r, s in enumerate(seqs):
        t[r, :len(s)] = torch.tensor(s)
    return t


def build_model(n_src: int, n_tgt: int, cfg: dict, seed: int):
    """The shared model, initialised from `torch.manual_seed(seed)`; also returns the initial-state hash."""
    import torch
    from translit_train import Model
    torch.manual_seed(seed)
    m = cfg["model"]
    model = Model(n_src, n_tgt, d=m["d"], layers=m["layers"], heads=m["heads"], ff=m["ff"], dropout=m["dropout"],
                  max_in=ps.MAX_SRC)
    return model, state_hash(model.state_dict())


def state_hash(state: dict) -> str:
    """sha256 over the names, shapes and bytes of a state dict (sorted by name)."""
    h = hashlib.sha256()
    for k in sorted(state):
        v = state[k].detach().cpu().contiguous().numpy()
        h.update(k.encode() + str(v.shape).encode() + v.tobytes())
    return h.hexdigest()


def greedy_texts(model, srcs: list[list[int]], tgt: str, max_len: int = ps.MAX_TGT) -> list[str]:
    """Greedy decode (eval mode, no grad) → Khmer strings."""
    import torch
    from translit_train import greedy
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(srcs), 256):
            for ids in greedy(model, pad(srcs[i:i + 256])):
                out.append("".join(tgt[t - 3] for t in ids if t >= 3))
    model.train()
    return out


def mean_loss(model, pairs: list[tuple[list[int], list[int]]], loss_fn) -> float:
    """Mean token loss over pairs (eval mode, no grad)."""
    import torch
    model.eval()
    tot = n = 0.0
    with torch.no_grad():
        for i in range(0, len(pairs), 256):
            ch = pairs[i:i + 256]
            S, Tt = pad([a for a, _ in ch]), pad([b for _, b in ch])
            lg = model(S, Tt[:, :-1])
            tot += loss_fn(lg.reshape(-1, lg.shape[-1]), Tt[:, 1:].reshape(-1)).item() * len(ch)
            n += len(ch)
    model.train()
    return tot / max(1, n)


def lr_lambda(warmup: int, total: int):
    """Linear warm-up then cosine to 0 (the repository's schedule)."""
    return lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))


def _save_atomic(obj, path: Path) -> None:
    """torch.save to a temporary file, then rename (no half-written checkpoints)."""
    import torch
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_json_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def data_inputs() -> list[Path]:
    """Files every job reads (hashed into provenance)."""
    return [CONFIG, DATA / "manifest.json", DATA / "train.tsv", DATA / "p2g.tsv", DATA / "dev_references.jsonl",
            DATA / "vocab.json", ps.CMUDICT]


def pretrain(seed: int, out: Path, cfg: dict, limit_rows: int | None = None) -> dict:
    """Khmer phones → Khmer spelling pretraining for one seed; best checkpoint by P2G dev loss.

    Stops after `p2g.epochs` epochs or `p2g.max_hours` elapsed (checked after every update), whichever is first.

    Args:
        seed: Paired seed. out: Job folder. cfg: Resolved config. limit_rows: Smoke runs only.

    Returns:
        The summary (also `summary.json` and `done.json`).
    """
    import torch
    from torch import nn
    out.mkdir(parents=True, exist_ok=True)
    pc, tc = cfg["p2g"], cfg["train"]
    torch.set_num_threads(tc["threads"])
    vocab, tgt, _v = load_vocab()
    tr, dv = load_p2g()
    if limit_rows:
        tr, dv = tr[:limit_rows], dv[: max(16, limit_rows // 10)]
    trp = [(ps.encode_p2g(p, vocab), target_ids(k, tgt)) for p, k in tr]
    dvp = [(ps.encode_p2g(p, vocab), target_ids(k, tgt)) for p, k in dv]
    model, init_hash = build_model(len(vocab.tokens), len(tgt) + 3, cfg, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=pc["lr"], weight_decay=tc["weight_decay"])
    total = pc["epochs"] * math.ceil(len(trp) / pc["batch"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(pc["warmup"], total))
    loss_fn = nn.CrossEntropyLoss(ignore_index=ps.PAD, label_smoothing=tc["label_smoothing"])
    run = experiment.Run(out, "E02-english-phonetic", out.name, cfg, data_inputs())
    state = {"epoch": 0, "step": 0, "best_loss": float("inf"), "best_epoch": 0, "elapsed": 0.0}
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", weights_only=False)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["torch_rng"])
        state = ck["state"]
    t_start = time.time() - state["elapsed"]
    stopped = "epochs"
    for ep in range(state["epoch"], pc["epochs"]):
        order = list(range(len(trp)))
        random.Random(f"{seed}:p2g_order:{ep}").shuffle(order)
        tot = 0.0
        for i in range(0, len(order), pc["batch"]):
            ch = [trp[j] for j in order[i:i + pc["batch"]]]
            S, Tt = pad([a for a, _ in ch]), pad([b for _, b in ch])
            lg = model(S, Tt[:, :-1])
            loss = loss_fn(lg.reshape(-1, lg.shape[-1]), Tt[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            sched.step()
            state["step"] += 1
            tot += loss.item() * len(ch)
            if time.time() - t_start > pc["max_hours"] * 3600:
                stopped = "max_hours"
                break
        dev_loss = mean_loss(model, dvp, loss_fn)
        state["epoch"] = ep + 1
        state["elapsed"] = time.time() - t_start
        rec = {"epoch": ep + 1, "train_loss": tot / len(trp), "p2g_dev_loss": dev_loss, "step": state["step"],
               "minutes": state["elapsed"] / 60, "finite": math.isfinite(dev_loss)}
        run.log(rec, step=ep + 1)
        print(json.dumps(rec), flush=True)
        if not math.isfinite(dev_loss):
            raise FloatingPointError("non-finite P2G dev loss")
        if dev_loss < state["best_loss"]:
            state["best_loss"], state["best_epoch"] = dev_loss, ep + 1
            _save_atomic({"state": model.state_dict(), "epoch": ep + 1, "kind": "p2g", "seed": seed}, out / "best.pt")
        _save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "torch_rng": torch.get_rng_state(), "state": state}, out / "last.pt")
        if stopped == "max_hours":
            break
    summary = {"kind": "p2g", "seed": seed, "init_sha256": init_hash, "best_epoch": state["best_epoch"],
               "best_p2g_dev_loss": state["best_loss"], "steps": state["step"], "epochs_run": state["epoch"],
               "stopped_by": stopped, "elapsed_hours": state["elapsed"] / 3600, "train_rows": len(trp),
               "dev_rows": len(dvp), "best_sha256": ps.sha256_file(out / "best.pt")}
    run.finish(summary)
    _write_json_atomic(summary, out / "done.json")
    return summary


class Interrupted(RuntimeError):
    """Raised by `train(stop_after=...)` (tests): the job stops as if killed after an epoch's checkpoint."""


def train(arm: str, seed: int, out: Path, cfg: dict, init: Path | None = None, limit_rows: int | None = None,
          stop_after: int | None = None) -> dict:
    """Supervised English → Khmer training of one arm and seed (see the module docstring).

    Args:
        arm: One of `ARMS`. seed: Paired seed. out: Job folder. cfg: Resolved config.
        init: `best.pt` of the P2G pretraining (required for `phones_p2g`, refused otherwise).
        limit_rows: Smoke runs only (first n training rows).
        stop_after: Tests only: raise `Interrupted` after this epoch's checkpoint is written.

    Returns:
        The summary (also `summary.json` and `done.json`).
    """
    import torch
    from torch import nn
    if (arm == "phones_p2g") != (init is not None):
        raise ValueError("--init is required for phones_p2g and not allowed for other arms")
    out.mkdir(parents=True, exist_ok=True)
    tc, pc = cfg["train"], cfg["p2g"]
    torch.set_num_threads(tc["threads"])
    vocab, tgt, _v = load_vocab()
    mode = source_mode(arm)
    tr, tv = load_supervised()
    if limit_rows:
        tr, tv = tr[:limit_rows], tv[: max(16, limit_rows // 10)]
    dev = load_dev()
    tr_tgt = [target_ids(k, tgt) for _w, k in tr]
    tv_pairs = [(encode_word(w, mode, vocab), target_ids(k, tgt)) for w, k in tv]
    dev_src = [encode_word(i.word, mode, vocab) for i in dev]
    replay = []
    if arm == "phones_p2g":
        p2g_tr, _ = load_p2g()
        replay = [(ps.encode_p2g(p, vocab), target_ids(k, tgt)) for p, k in p2g_tr]
    model, init_hash = build_model(len(vocab.tokens), len(tgt) + 3, cfg, seed)
    init_sha = None
    if init is not None:
        init_sha = ps.sha256_file(init)
        model.load_state_dict(torch.load(init, weights_only=False)["state"])
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total = tc["epochs"] * math.ceil(len(tr) / tc["batch"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tc["warmup"], total))
    loss_fn = nn.CrossEntropyLoss(ignore_index=ps.PAD, label_smoothing=tc["label_smoothing"])
    run = experiment.Run(out, "E02-english-phonetic", out.name, {**cfg, "arm": arm, "seed": seed,
                                                                  "init": str(init) if init else None},
                         data_inputs() + ([init] if init else []))
    state = {"epoch": 0, "step": 0, "replay_steps": 0, "best": None, "elapsed": 0.0, "batch_digest": "",
             "drop_count": 0}
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", weights_only=False)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["torch_rng"])
        state = ck["state"]
    t_start = time.time() - state["elapsed"]
    for ep in range(state["epoch"], tc["epochs"]):
        order = list(range(len(tr)))
        random.Random(f"{seed}:order:{ep}").shuffle(order)
        drop_rng = random.Random(f"{seed}:phone_dropout:{ep}")
        replay_rng = random.Random(f"{seed}:replay:{ep}")
        tot = rep_tot = 0.0
        grad_norms, sup_in_epoch, src_lens = [], 0, []
        digest = hashlib.sha256()   # this epoch's batch ids; chained into state["batch_digest"] (resume-safe)
        for i in range(0, len(order), tc["batch"]):
            idx = order[i:i + tc["batch"]]
            digest.update(",".join(map(str, idx)).encode() + b";")
            drops = [drop_rng.random() < tc["phone_dropout"] for _ in idx]   # drawn in every arm: same stream
            srcs = [encode_word(tr[j][0], mode, vocab, drop=(mode == "phones" and d)) for j, d in zip(idx, drops)]
            state["drop_count"] += sum(drops) if mode == "phones" else 0
            src_lens += [len(s) for s in srcs]
            S, Tt = pad(srcs), pad([tr_tgt[j] for j in idx])
            lg = model(S, Tt[:, :-1])
            loss = loss_fn(lg.reshape(-1, lg.shape[-1]), Tt[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            grad_norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])))
            opt.step()
            sched.step()
            state["step"] += 1
            sup_in_epoch += 1
            tot += loss.item() * len(idx)
            if replay and state["step"] % pc["replay_every"] == 0:
                ch = [replay[replay_rng.randrange(len(replay))] for _ in range(pc["replay_batch"])]
                S2, T2 = pad([a for a, _ in ch]), pad([b for _, b in ch])
                lg2 = model(S2, T2[:, :-1])
                l2 = loss_fn(lg2.reshape(-1, lg2.shape[-1]), T2[:, 1:].reshape(-1))
                opt.zero_grad()
                l2.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                opt.step()   # current LR; the supervised schedule is not stepped
                state["replay_steps"] += 1
                rep_tot += l2.item()
        texts = greedy_texts(model, dev_src, tgt)
        sc = ev.score_predictions(dev, {it.id: [t] for it, t in zip(dev, texts)}, top_k=1)
        tv_loss = mean_loss(model, tv_pairs, loss_fn)
        state["epoch"] = ep + 1
        state["elapsed"] = time.time() - t_start
        state["batch_digest"] = hashlib.sha256((state["batch_digest"] + digest.hexdigest()).encode()).hexdigest()
        rec = {"epoch": ep + 1, "train_loss": tot / len(tr), "train_val_loss": tv_loss,
               "dev_primary_macro": sc["primary_macro"], "dev_item_accuracy": sc["item_accuracy"],
               "dev_correct": sc["correct_items"], "dev_n": len(dev), "grad_norm_mean": sum(grad_norms) / len(grad_norms),
               "grad_norm_max": max(grad_norms), "supervised_steps": state["step"],
               "replay_steps": state["replay_steps"], "replay_loss": rep_tot / max(1, state["replay_steps"]) if replay else None,
               "mean_src_len": sum(src_lens) / len(src_lens), "lr": sched.get_last_lr()[0],
               "minutes": state["elapsed"] / 60}
        if not (math.isfinite(rec["train_loss"]) and math.isfinite(tv_loss)):
            raise FloatingPointError("non-finite loss")
        run.log(rec, step=ep + 1)
        print(json.dumps(rec), flush=True)
        key = (sc["primary_macro"], -tv_loss, -(ep + 1))   # higher dev, then lower loss, then earlier epoch
        if state["best"] is None or key > tuple(state["best"]["key"]):
            state["best"] = {"key": list(key), "epoch": ep + 1, "dev_primary_macro": sc["primary_macro"],
                             "train_val_loss": tv_loss}
            _save_atomic({"state": model.state_dict(), "epoch": ep + 1, "arm": arm, "seed": seed, "mode": mode,
                          "src_tokens": vocab.tokens, "tgt_vocab": tgt, "model": cfg["model"]}, out / "best.pt")
        _save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "torch_rng": torch.get_rng_state(), "state": state}, out / "last.pt")
        if stop_after == ep + 1:
            raise Interrupted(f"stopped after epoch {ep + 1}")
    summary = {"arm": arm, "seed": seed, "mode": mode, "init_sha256": init_hash, "p2g_init_file_sha256": init_sha,
               "best_epoch": state["best"]["epoch"], "best_dev_primary_macro": state["best"]["dev_primary_macro"],
               "best_train_val_loss": state["best"]["train_val_loss"], "supervised_steps": state["step"],
               "replay_steps": state["replay_steps"], "phone_dropout_examples": state["drop_count"],
               "batch_order_sha256": state["batch_digest"], "train_rows": len(tr), "dev_items": len(dev),
               "elapsed_hours": state["elapsed"] / 3600, "best_sha256": ps.sha256_file(out / "best.pt"),
               "params": sum(p.numel() for p in model.parameters())}
    run.finish(summary)
    _write_json_atomic(summary, out / "done.json")
    return summary


def export(out: Path, cfg: dict) -> Path:
    """Write `best.pt` as `model.npz` (float32 weights + `__meta__` JSON) for `khmer_translit.english`.

    Returns:
        The written path.
    """
    import torch
    ck = torch.load(out / "best.pt", weights_only=False)
    data_manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    meta = {"format": "khmer-translit-english-v1", "encoder_version": ps.ENCODER_VERSION, "mode": ck["mode"],
            "arm": ck["arm"], "seed": ck["seed"], "epoch": ck["epoch"], "src_tokens": ck["src_tokens"],
            "tgt_vocab": ck["tgt_vocab"], "heads": ck["model"]["heads"], "layers": ck["model"]["layers"],
            "max_src": ps.MAX_SRC, "max_tgt": ps.MAX_TGT, "beam": cfg["eval"]["beam"],
            "pronunciation": {"provider": "cmudict", "alphabet": ps.ALPHABET, "accent": "en-US-cmudict",
                              "default": "first listed reading", "miss": "<no_pron>",
                              "cmudict_sha256": data_manifest["cmudict_sha256"]} if ck["mode"] == "phones" else None,
            "input_policy": "one English word; NFKC, lower case, accents removed, apostrophes/hyphens dropped; a-z, "
                            f"<= {ps.MAX_LETTERS} letters",
            "data_manifest_sha256": ps.sha256_file(DATA / "manifest.json"), "config_sha256": ps.sha256_file(CONFIG),
            "checkpoint_sha256": ps.sha256_file(out / "best.pt"),
            "training_sources": data_manifest["counts"]["supervised_kept"],
            "evidence_status": "inconclusive (benchmark unqualified; see experiments/E02-english-phonetic)"}
    arrays = {k: v.numpy().astype(np.float32) for k, v in ck["state"].items()}
    path = out / "model.npz"
    tmp = out / "model.tmp.npz"
    np.savez(tmp, __meta__=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    os.replace(tmp, path)
    print("wrote", path, "epoch", meta["epoch"])
    return path


def verify(out: Path, n: int = 12) -> dict:
    """Export parity on fixed dev fixtures: source ids, greedy tokens, teacher-forced logits (torch vs NumPy).

    Returns:
        The report (also `parity.json`); `ok` is False on any token mismatch or a logit gap > 1e-4.
    """
    import torch
    from translit_train import greedy
    from khmer_translit import english
    ck = torch.load(out / "best.pt", weights_only=False)
    vocab, tgt, _v = load_vocab()
    model, _ = build_model(len(vocab.tokens), len(tgt) + 3, {"model": ck["model"]}, 0)
    model.load_state_dict(ck["state"])
    model.eval()
    tl = english.Transliterator.from_pretrained(out / "model.npz")
    dev = load_dev()[:n]
    src_ok = greedy_ok = 0
    max_gap = 0.0
    for it in dev:
        t_src = encode_word(it.word, ck["mode"], vocab)
        n_src = tl.encode_ids(it.word)[0]
        src_ok += t_src == n_src
        with torch.no_grad():
            t_ids = greedy(model, pad([t_src]))[0]
        prefix = [ps.BOS]
        for _ in range(ps.MAX_TGT + 1):   # NumPy greedy, same stopping rule as the torch one
            nxt = int(np.argmax(tl.next_logprobs(n_src, prefix)))
            prefix.append(nxt)
            if nxt == ps.EOS:
                break
        n_ids = prefix[1:prefix.index(ps.EOS)] if ps.EOS in prefix else prefix[1:]
        greedy_ok += t_ids == n_ids
        tf = ([ps.BOS] + t_ids)[: ps.MAX_TGT + 1]
        with torch.no_grad():
            tlog = torch.log_softmax(model(pad([t_src]), pad([tf])), -1)[0].numpy()
        for k in range(len(tf)):
            max_gap = max(max_gap, float(np.abs(tlog[k] - tl.next_logprobs(n_src, tf[: k + 1])).max()))
    rep = {"fixtures": len(dev), "source_ids_equal": src_ok, "greedy_equal": greedy_ok, "max_abs_logprob_gap": max_gap,
           "ok": src_ok == len(dev) and greedy_ok == len(dev) and max_gap <= 1e-4,
           "npz_sha256": ps.sha256_file(out / "model.npz")}
    _write_json_atomic(rep, out / "parity.json")
    print(json.dumps(rep))
    return rep


def main() -> int:
    """CLI entry point.

    Returns:
        0 on success, 1 if verification fails.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["pretrain", "train", "export", "verify"])
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--set", action="append", default=[], help="section.key=value (smoke runs only)")
    ap.add_argument("--limit-rows", type=int, default=None, help="smoke runs only")
    args = ap.parse_args()
    cfg = load_cfg(args.set)
    out = Path(args.out)
    if args.mode == "pretrain":
        pretrain(args.seed, out, cfg, args.limit_rows)
    elif args.mode == "train":
        if not args.arm:
            ap.error("--arm is required")
        train(args.arm, args.seed, out, cfg, Path(args.init) if args.init else None, args.limit_rows)
    elif args.mode == "export":
        export(out, cfg)
    else:
        return 0 if verify(out)["ok"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
