#!/usr/bin/env python3
"""Experiment E04: pronunciation pathways. G (letters), P (phones only), GP (both), conditional three-view model.

    .venv/bin/python scripts/e04.py audit                                   # phone-ambiguity audit (train/dev)
    .venv/bin/python scripts/e04.py sanity                                  # tiny overfit checks for G, P, GP
    .venv/bin/python scripts/e04.py train --arm P --seed 0 --out models/e04/P_s0
    .venv/bin/python scripts/e04.py export --out models/e04/P_s0
    .venv/bin/python scripts/e04.py verify --out models/e04/P_s0            # torch vs NumPy parity per view
    .venv/bin/python scripts/e04.py devcheck                                # pilot dev comparison + registered branch
    .venv/bin/python scripts/e04.py combine                                 # Branch C only: G/P score combination
    .venv/bin/python scripts/e04.py evaluate                                # after selection: historical E03 eval set

The queue (pilots → branch → conditional → confirmation → evaluation) is `scripts/e04_run.py`.

**Data:** E03's DACS partitions, unchanged (`data/e03/`). Phones are the first CMUdict reading with stress, as
in E03.

**Source views:** the E02/E03 157-token vocabulary, unchanged, so all arms have identical parameter counts:
- G = `<task:en>` + letters;
- P = `<task:en>` `<sep>` + `en:` phones (no letters, no word id, no spelling length);
- GP = `<task:en>` + letters + `<sep>` + phones (exactly E03's phones encoding).

The structure identifies the view; nothing identifies the word.

**Schedule (E04 only):** 3,000 supervised updates; linear warm-up to the base rate over 150 updates, then cosine
over the remaining 2,850. No phone or view dropout, except that the conditional three-view arm (V3) samples one view
per example uniformly. Model, optimizer, batch, dropout and label smoothing as in E03.

**Checkpoint:** greedy top-1 macro on the primary E03 dev items in the arm's main view (V3: GP), every 100 updates;
ties go to the lower train-val loss, then the earlier update.
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
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import e02_diagnostics as DG  # noqa: E402
import english_phonetic_train as T  # noqa: E402
from khmer_translit import evaluation as ev  # noqa: E402
from khmer_translit import experiment  # noqa: E402
from khmer_translit import phonetic_source as ps  # noqa: E402
from khmer_translit import reference_data as rd  # noqa: E402

EXPERIMENT = "E04-pronunciation-pathways"
EXP = ROOT / "experiments" / EXPERIMENT
REPORTS = EXP / "reports"
CONFIG = EXP / "config.yaml"
DATA = ROOT / "data" / "e03"
RUNS = ROOT / "models" / "e04"
STATE = ROOT / "state" / "e04"
FORMAT = "e04-research-v1"
ARMS = {"G": ("g",), "P": ("p",), "GP": ("gp",), "V3": ("g", "p", "gp")}
MAIN_VIEW = {"G": "g", "P": "p", "GP": "gp", "V3": "gp"}
SIMPLICITY = ("P", "GP", "V3")                      # tie order for candidate selection (simpler first)
PRIMARY_EXCLUDE = {"pronunciation_ambiguous", "possible_translation"}
EXPECTED = {
    "paths": {"runs_dir": "models/e04"},
    "model": {"d": 192, "layers": 3, "heads": 4, "ff": 512, "dropout": 0.1},
    "train": {"updates": 3000, "warmup": 150, "batch": 128, "lr": 0.001, "weight_decay": 0.01,
              "label_smoothing": 0.1, "grad_clip": 1.0, "threads": 6, "eval_every": 100},
    "sanity": {"max_examples": 64, "updates": 500, "warmup": 50, "batch": 64, "lr": 0.001, "threshold": 0.95,
               "max_attempts": 6},
    "eval": {"beam": 4, "top_k": 4},
    "rules": {"signal_union_gain_pp": 5.0, "screen_margin_pp": 2.0, "n_boot": 10000, "boot_seed": 20261027},
}


def load_cfg() -> dict:
    """The registered E04 config (strict schema)."""
    return experiment.load_config(CONFIG, EXPECTED)


# ------------------------------------------------------------------ encoding
def vocab() -> tuple[ps.SourceVocab, str]:
    """E03's source vocabulary (checked against the code) and target characters."""
    v = json.loads((DATA / "vocab.json").read_text(encoding="utf-8"))
    voc = ps.SourceVocab(v["km_phones"])
    if voc.tokens != v["src_tokens"]:
        raise ValueError("vocab.json does not match the code's source vocabulary")
    return voc, v["tgt_chars"]


def phones_of(word: str) -> tuple[str, ...]:
    """The registered default reading (first CMUdict reading, with stress).

    Raises:
        ValueError: If the word has no dictionary reading (every E03 word has one).
    """
    p = ps.get_pronunciation(word)
    if p.status != "dictionary":
        raise ValueError(f"{word!r} has no CMUdict reading")
    return tuple(p.phones)


def encode(word: str, phones: tuple[str, ...], view: str, voc: ps.SourceVocab) -> list[int]:
    """Source ids of one view.

    Examples:
        >>> voc = ps.ENGLISH_VOCAB
        >>> [voc.tokens[i] for i in encode("one", ("W", "AH1", "N"), "p", voc)]
        ['<task:en>', '<sep>', 'en:W', 'en:AH1', 'en:N']
        >>> [voc.tokens[i] for i in encode("one", ("W", "AH1", "N"), "g", voc)]
        ['<task:en>', 'o', 'n', 'e']
    """
    task, sep = voc.id("<task:en>"), voc.id("<sep>")
    letters = [voc.id(c) for c in word]
    ph = [voc.id("en:" + p) for p in phones]
    ids = {"g": [task] + letters, "p": [task, sep] + ph, "gp": [task] + letters + [sep] + ph}[view]
    if len(ids) > ps.MAX_SRC:
        raise ps.InputError("too_long", f"{word}: {len(ids)} source tokens")
    return ids


def seq_lambda(warmup: int, total: int):
    """Sequential schedule: factor (k+1)/warmup for update k < warmup, then cosine from 1 to 0 over the rest.

    `LambdaLR` evaluates the factor at index k before the (k+1)-th optimizer step, so update k (0-based) runs at
    `base * f(k)`: the base rate is reached at update `warmup - 1` and decay starts at update `warmup`.

    Examples:
        >>> f = seq_lambda(150, 3000)
        >>> round(f(0), 4), f(149), f(150), f(151) < 1.0
        (0.0067, 1.0, 1.0, True)
    """
    def f(k: int) -> float:
        if k < warmup:
            return (k + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (k - warmup) / max(1, total - warmup))))
    return f


# ------------------------------------------------------------------ data
def rows(part: str) -> list[tuple[str, str, tuple[str, ...]]]:
    """E03 training rows `(word, khmer, phones)` of one part ("train" or "train_val")."""
    out = []
    for line in (DATA / "train.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        w, k, _s, _t, ph, p = line.split("\t")
        if p == part:
            if tuple(ph.split()) != phones_of(w):
                raise ValueError(f"{w}: stored phones differ from the registered default reading")
            out.append((w, k, tuple(ph.split())))
    return out


def items(name: str) -> list[rd.ReferenceItem]:
    """E03 dev or eval references (through the diagnostics guard)."""
    return rd.load_reference_items(DG.guarded(DATA / f"{name}_references.jsonl"))


def primary(its: list[rd.ReferenceItem]) -> list[rd.ReferenceItem]:
    """Items without an ambiguity or translation flag."""
    return [i for i in its if not PRIMARY_EXCLUDE & set(i.flags)]


# ------------------------------------------------------------------ audit
def audit() -> dict:
    """Phone-sequence ambiguity on training/dev material, and phone-seen flags for dev and eval words.

    Writes `reports/ambiguity_audit.json` (counts and examples) and `data/e04/phone_seen.json` (per word).
    """
    cmu_sha = ps.cmudict()[1]
    tr, tv = rows("train"), rows("train_val")
    dev = items("dev")
    mat = [("train", w, k, ph) for w, k, ph in tr] + [("train_val", w, k, ph) for w, k, ph in tv] + \
          [("dev", i.word, a, phones_of(i.word)) for i in dev for a in i.accepted]
    groups: dict[tuple, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for part, w, k, ph in mat:
        groups[ph][w].add(k)
    collide = {ph: g for ph, g in groups.items() if len(g) > 1}
    disjoint = {ph: g for ph, g in collide.items()
                if all(not (g[a] & g[b]) for a in g for b in g if a < b)}
    target_words = defaultdict(set)
    for _p, w, k, ph in mat:
        target_words[k].add(w)
    spelling_collisions = {k: ws for k, ws in target_words.items() if len(ws) > 1}
    multi = Counter(len(v) for g in groups.values() for v in g.values())
    tokens = set(vocab()[0].tokens)
    unknown = [p for ph in groups for p in ph if "en:" + p not in tokens]
    # dev: distinctions P cannot make (same phones as another word with disjoint targets)
    train_groups = defaultdict(lambda: defaultdict(set))
    for _p, w, k, ph in mat:
        if _p != "dev":
            train_groups[ph][w].add(k)
    dev_items = primary(dev)
    lost_dev = []
    for it in dev_items:
        ph = phones_of(it.word)
        others = {w: ks for w, ks in groups[ph].items() if w != it.word}
        if others and all(not (set(it.accepted) & ks) for ks in others.values()):
            lost_dev.append({"word": it.word, "phones": " ".join(ph), "other_words": sorted(others),
                             "with_train_word": any(w in train_groups[ph] for w in others)})
    # empirical P ceiling on dev: one output per phone sequence
    by_ph = defaultdict(list)
    for it in dev_items:
        by_ph[phones_of(it.word)].append(it)
    best = 0
    for ph, its in by_ph.items():
        cands = Counter(a for it in its for a in set(it.accepted))
        best += max(cands.values()) if cands else 0
    train_ph = {ph for _p, _w, _k, ph in mat if _p != "dev"}
    seen = {}
    for name in ("dev", "eval"):
        for it in items(name):
            ph = phones_of(it.word)
            seen[f"{name}:{it.id}"] = {"phones": " ".join(ph), "phone_seen_in_training": ph in train_ph,
                                       "phone_group_size_train": len(train_groups[ph])}
    out = {"provider": {"name": "cmudict", "sha256": cmu_sha, "reading": "first listed, stress kept"},
           "material": {"train_rows": len(tr), "train_val_rows": len(tv), "dev_items": len(dev),
                        "phone_sequences": len(groups), "english_words": len({w for _p, w, _k, _ph in mat})},
           "phone_collisions": {"sequences_with_several_words": len(collide),
                                "words_in_them": sum(len(g) for g in collide.values()),
                                "of_which_targets_disjoint": len(disjoint),
                                "examples_disjoint": [{"phones": " ".join(ph), "words": {w: sorted(k) for w, k in g.items()}}
                                                      for ph, g in list(disjoint.items())[:15]],
                                "examples_shared_target": [{"phones": " ".join(ph), "words": {w: sorted(k) for w, k in g.items()}}
                                                           for ph, g in list((p, g) for p, g in collide.items()
                                                                             if p not in disjoint)[:10]]},
           "spelling_collisions": {"targets_with_several_english_words": len(spelling_collisions),
                                   "examples": [{"khmer": k, "words": sorted(ws)} for k, ws in
                                                list(spelling_collisions.items())[:10]]},
           "targets_per_word": dict(sorted((str(k), v) for k, v in multi.items())),
           "coverage": {"words_with_dictionary_reading": "all (E03 requires one)", "unknown_phone_tokens": len(unknown)},
           "dev_primary": {"items": len(dev_items),
                           "p_cannot_distinguish": len(lost_dev), "examples": lost_dev[:20],
                           "empirical_exact_match_ceiling_for_any_phones_only_model": best / max(1, len(dev_items)),
                           "note": "property of these observed references (one output per phone sequence), not a "
                                   "limit on phonetic correctness"},
           "phone_seen": {name: {"items": sum(1 for k in seen if k.startswith(name + ":")),
                                 "phone_seen_in_training": sum(v["phone_seen_in_training"] for k, v in seen.items()
                                                               if k.startswith(name + ":"))}
                          for name in ("dev", "eval")}}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "ambiguity_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (ROOT / "data" / "e04").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "e04" / "phone_seen.json").write_text(json.dumps(seen, ensure_ascii=False) + "\n",
                                                          encoding="utf-8")
    return {k: v for k, v in out.items() if k not in ("phone_collisions", "spelling_collisions")} | {
        "phone_collisions": {k: v for k, v in out["phone_collisions"].items() if not k.startswith("examples")}}


# ------------------------------------------------------------------ training
def train(arm: str, seed: int, out: Path, cfg: dict, data: list | None = None, val: list | None = None,
          dev: list | None = None, tc: dict | None = None, name: str | None = None) -> dict:
    """Train one arm (see the module docstring); resumable from `last.pt`.

    Args:
        arm: G, P, GP or V3. seed: paired seed. out: job folder. cfg: resolved config.
        data, val, dev: override the training rows, train-val rows and dev items (sanity checks only).
        tc: override the training section (sanity checks only). name: run-record name.

    Returns:
        The summary (also `done.json`).
    """
    import torch
    from torch import nn
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    tc = tc or cfg["train"]
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(tc["threads"])
    voc, tgt = vocab()
    tr = data if data is not None else rows("train")
    tv = val if val is not None else rows("train_val")
    dv = dev if dev is not None else primary(items("dev"))
    views, main = ARMS[arm], MAIN_VIEW[arm]
    src = {v: [encode(w, ph, v, voc) for w, _k, ph in tr] for v in ("g", "p", "gp")}
    tr_tgt = [T.target_ids(k, tgt) for _w, k, _ph in tr]
    tv_pairs = [(encode(w, ph, main, voc), T.target_ids(k, tgt)) for w, k, ph in tv]
    dev_src = [encode(i.word, phones_of(i.word), main, voc) for i in dv]
    accepted = defaultdict(set)
    for w, k, _ph in tr:
        accepted[w].add(k)
    sample = sorted(range(len(tr)), key=lambda j: hashlib.sha256(f"{seed}:trainsample:{j}".encode()).hexdigest())[:256]
    model, init_hash = T.build_model(len(voc.tokens), len(tgt) + 3, cfg, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, seq_lambda(tc["warmup"], tc["updates"]))
    loss_fn = nn.CrossEntropyLoss(ignore_index=ps.PAD, label_smoothing=tc["label_smoothing"])
    run = experiment.Run(out, EXPERIMENT, name or out.name, {**cfg, "train": tc, "arm": arm, "seed": seed},
                         [CONFIG, DATA / "manifest.json", DATA / "train.tsv", DATA / "dev_references.jsonl",
                          DATA / "vocab.json", ps.CMUDICT])
    st = {"update": 0, "epoch": 0, "pos": 0, "best": None, "elapsed": 0.0, "digest": "", "lr": [],
          "views": Counter(), "window": {"grad": [], "srclen": [], "loss": []}}
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", weights_only=False)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["torch_rng"])
        st = ck["state"]
    t0 = time.time() - st["elapsed"]
    U, B = tc["updates"], tc["batch"]
    while st["update"] < U:
        order = list(range(len(tr)))
        random.Random(f"{seed}:order:{st['epoch']}").shuffle(order)
        vrng = random.Random(f"{seed}:view:{st['epoch']}")
        draws = [vrng.randrange(3) for _ in order]               # drawn in every arm: identical streams
        for b in range(st["pos"], len(order), B):
            if st["update"] >= U:
                break
            idx = order[b:b + B]
            vs = [("g", "p", "gp")[draws[b + n]] if len(views) == 3 else views[0] for n in range(len(idx))]
            S = T.pad([src[v][j] for v, j in zip(vs, idx)])
            Tt = T.pad([tr_tgt[j] for j in idx])
            lg = model(S, Tt[:, :-1])
            loss = loss_fn(lg.reshape(-1, lg.shape[-1]), Tt[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            st["window"]["grad"].append(float(nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])))
            st["lr"].append(opt.param_groups[0]["lr"])            # the rate this optimizer step applies
            opt.step()
            sched.step()
            st["update"] += 1
            st["pos"] = b + B
            st["views"].update(vs)
            st["window"]["srclen"].append(float(mean(len(x) for x in S.tolist())))
            st["window"]["loss"].append(loss.item())
            st["digest"] = hashlib.sha256((st["digest"] + ",".join(map(str, idx))
                                           + ("|" + "".join(v[0] for v in vs) if len(views) == 3 else "")).encode()).hexdigest()
            if st["update"] % tc["eval_every"] == 0 or st["update"] == U:
                rec = checkpoint_eval(model, st, dv, dev_src, tv_pairs, loss_fn, tgt, tr, src[main], sample, accepted)
                rec["minutes"] = (time.time() - t0) / 60
                run.log(rec, step=st["update"])
                print(json.dumps(rec), flush=True)
                key = (rec["dev_macro"], -rec["val_loss"], -st["update"])
                if dv and (st["best"] is None or key > tuple(st["best"]["key"])):
                    st["best"] = {"key": list(key), "update": st["update"], "dev_macro": rec["dev_macro"],
                                  "val_loss": rec["val_loss"]}
                    T._save_atomic({"state": model.state_dict(), "update": st["update"], "arm": arm, "seed": seed,
                                    "views": list(views), "main_view": main, "src_tokens": voc.tokens,
                                    "tgt_vocab": tgt, "model": cfg["model"]}, out / "best.pt")
                st["elapsed"] = time.time() - t0
                T._save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                                "torch_rng": torch.get_rng_state(), "state": st}, out / "last.pt")
        else:
            st["epoch"] += 1
            st["pos"] = 0
    final_agree = None
    if not dv:                                                    # sanity checks: final greedy training agreement
        texts = T.greedy_texts(model, [src[main][j] for j in range(len(tr))], tgt)
        final_agree = mean(t in accepted[tr[j][0]] for j, t in enumerate(texts))
    lr = st["lr"]
    summary = {"arm": arm, "seed": seed, "views": list(views), "main_view": main, "init_sha256": init_hash,
               "batch_order_sha256": st["digest"], "updates": st["update"], "epochs_started": st["epoch"] + 1,
               "train_rows": len(tr), "dev_items": len(dv), "view_counts": dict(st["views"]),
               "lr_applied": {"first": lr[0], "at_warmup_end": lr[tc["warmup"] - 1], "first_decay": lr[tc["warmup"]],
                              "max": max(lr), "argmax": lr.index(max(lr)), "last": lr[-1]},
               "elapsed_hours": (time.time() - t0) / 3600, "params": sum(p.numel() for p in model.parameters())}
    if dv:
        summary.update(best_update=st["best"]["update"], best_dev_macro=st["best"]["dev_macro"],
                       best_val_loss=st["best"]["val_loss"], best_sha256=ps.sha256_file(out / "best.pt"))
    else:
        summary["final_train_agreement"] = final_agree
    (out / "lr_trace.json").write_text(json.dumps(lr) + "\n")
    run.finish(summary)
    T._write_json_atomic(summary, out / "done.json")
    return summary


def checkpoint_eval(model, st, dv, dev_src, tv_pairs, loss_fn, tgt, tr, tr_src_main, sample, accepted,
                    keep_window: bool = False) -> dict:
    """Dev macro (greedy), train-val loss, training agreement on a fixed sample, window statistics."""
    w = st["window"]
    rec = {"update": st["update"], "lr": st["lr"][-1], "train_loss_window": mean(w["loss"]),
           "grad_norm_mean": mean(w["grad"]), "grad_norm_max": max(w["grad"]), "mean_src_len": mean(w["srclen"]),
           "val_loss": T.mean_loss(model, tv_pairs, loss_fn) if tv_pairs else 0.0}
    texts = T.greedy_texts(model, [tr_src_main[j] for j in sample], tgt)
    rec["train_agreement"] = mean(t in accepted[tr[j][0]] for t, j in zip(texts, sample))
    if dv:
        dt = T.greedy_texts(model, dev_src, tgt)
        sc = ev.score_predictions(dv, {it.id: [t] for it, t in zip(dv, dt)}, top_k=1)
        rec["dev_macro"], rec["dev_correct"] = sc["primary_macro"], sc["correct_items"]
    else:
        rec["dev_macro"] = 0.0
    if not keep_window:
        st["window"] = {"grad": [], "srclen": [], "loss": []}
    return rec


# ------------------------------------------------------------------ export, NumPy inference, parity
def export(out: Path) -> Path:
    """`best.pt` → `model.npz` (float32 + `__meta__`); a research format the product API does not load."""
    import torch
    ck = torch.load(out / "best.pt", weights_only=False)
    meta = {"format": FORMAT, "experiment": EXPERIMENT, "arm": ck["arm"], "seed": ck["seed"], "update": ck["update"],
            "views": ck["views"], "main_view": ck["main_view"], "src_tokens": ck["src_tokens"],
            "tgt_vocab": ck["tgt_vocab"], "heads": ck["model"]["heads"], "layers": ck["model"]["layers"],
            "max_src": ps.MAX_SRC, "beam": 4, "cmudict_sha256": ps.cmudict()[1],
            "encoding": {"g": "<task:en> + letters", "p": "<task:en> <sep> + en:phones",
                         "gp": "<task:en> + letters + <sep> + en:phones"},
            "checkpoint_sha256": ps.sha256_file(out / "best.pt"),
            "note": "E04 research checkpoint (unqualified references); not a product model"}
    arrays = {k: v.numpy().astype(np.float32) for k, v in ck["state"].items()}
    tmp = out / "model.tmp.npz"
    np.savez(tmp, __meta__=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    os.replace(tmp, out / "model.npz")
    return out / "model.npz"


def load_np(path: Path):
    """`(numpy_model, meta)` for an E04 `model.npz`."""
    from khmer_translit.translit import Transliterator as NumpyModel
    z = np.load(path)
    meta = json.loads(str(z["__meta__"]))
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: not an E04 model")
    w = {k: z[k].astype(np.float64) for k in z.files if k != "__meta__"}
    return NumpyModel(w, "", meta["tgt_vocab"], meta["heads"], meta["layers"], False, meta["max_src"], False), meta


def beam(npm, srcs: list[list[int]], width: int = 4) -> list[list[tuple[str, float]]]:
    """NumPy beam search; distinct NFC candidates, best first."""
    out = []
    for nb in npm.beam_search(*npm.encode_ids(srcs), width):
        seen, c = set(), []
        for t, s in nb:
            t = ps.nfc(t)
            if t and t not in seen:
                seen.add(t)
                c.append((t, float(s)))
        out.append(c)
    return out


def tf_score(npm, src: list[int], text: str) -> tuple[float, int]:
    """Teacher-forced total log-probability of a spelling (+ EOS) and its token count."""
    memory, pad = npm.encode_ids([src])
    toks = [ps.BOS] + [npm.tgt_vocab.index(c) + 3 for c in text] + [ps.EOS]
    return sum(float(npm.decode_step(memory, pad, np.array([toks[:t]]))[0][toks[t]])
               for t in range(1, len(toks))), len(toks) - 1


def verify(out: Path, n: int = 12) -> dict:
    """Parity of the export on dev fixtures, for every view the arm was trained with."""
    import torch
    from translit_train import greedy
    ck = torch.load(out / "best.pt", weights_only=False)
    voc, tgt = vocab()
    model, _ = T.build_model(len(voc.tokens), len(tgt) + 3, {"model": ck["model"]}, 0)
    model.load_state_dict(ck["state"])
    model.eval()
    npm, meta = load_np(out / "model.npz")
    dv = primary(items("dev"))[:n]
    ok_g, gap, total = 0, 0.0, 0
    for view in meta["views"]:
        for it in dv:
            s = encode(it.word, phones_of(it.word), view, voc)
            with torch.no_grad():
                t_ids = greedy(model, T.pad([s]))[0]
            prefix = [ps.BOS]
            memory, pad = npm.encode_ids([s])
            for _ in range(ps.MAX_TGT + 1):
                nxt = int(np.argmax(npm.decode_step(memory, pad, np.array([prefix]))[0]))
                prefix.append(nxt)
                if nxt == ps.EOS:
                    break
            n_ids = prefix[1:prefix.index(ps.EOS)] if ps.EOS in prefix else prefix[1:]
            ok_g += t_ids == n_ids
            total += 1
            tf = ([ps.BOS] + t_ids)[: ps.MAX_TGT + 1]
            with torch.no_grad():
                tlog = torch.log_softmax(model(T.pad([s]), T.pad([tf])), -1)[0].numpy()
            for k in range(len(tf)):
                gap = max(gap, float(np.abs(tlog[k] - npm.decode_step(memory, pad, np.array([tf[:k + 1]]))[0]).max()))
    rep = {"views": meta["views"], "fixtures": total, "greedy_equal": ok_g, "max_abs_logprob_gap": gap,
           "ok": ok_g == total and gap <= 1e-4, "npz_sha256": ps.sha256_file(out / "model.npz")}
    T._write_json_atomic(rep, out / "parity.json")
    print(json.dumps(rep))
    return rep


# ------------------------------------------------------------------ sanity checks
def sanity_subset(k: int = 64) -> tuple[list, dict]:
    """At most `k` training rows with a unique phone sequence and one target; the rule and exclusions."""
    tr = rows("train") + rows("train_val")
    by_ph, by_w = defaultdict(set), defaultdict(set)
    for w, kh, ph in tr:
        by_ph[ph].add(w)
        by_w[w].add(kh)
    pool = sorted({(w, ph) for w, _k, ph in rows("train") if len(by_ph[ph]) == 1 and len(by_w[w]) == 1},
                  key=lambda x: hashlib.sha256(f"e04-sanity:{x[0]}".encode()).hexdigest())
    chosen = [(w, next(iter(by_w[w])), ph) for w, ph in pool[:k]]
    info = {"rule": "training-part words whose phone sequence belongs to one word only (train + train_val) and "
                    "which have one target; ordered by sha256('e04-sanity:'+word); first 64",
            "excluded_phone_collision_words": sum(1 for w, _k, ph in rows("train") if len(by_ph[ph]) > 1),
            "excluded_multi_target_words": len({w for w, _k, _ph in rows("train") if len(by_w[w]) > 1}),
            "selected": len(chosen), "sha256": hashlib.sha256(json.dumps(chosen, ensure_ascii=False).encode()).hexdigest()}
    return chosen, info


def sanity() -> dict:
    """Overfit G, P and GP on the subset (no dropout, no label smoothing); ≥ threshold greedy training agreement."""
    cfg = load_cfg()
    sc = cfg["sanity"]
    path = STATE / "sanity.json"
    st = json.loads(path.read_text()) if path.exists() else {"attempts": 0, "results": {}}
    data, info = sanity_subset(sc["max_examples"])
    tc = {**cfg["train"], "updates": sc["updates"], "warmup": sc["warmup"], "batch": sc["batch"], "lr": sc["lr"],
          "label_smoothing": 0.0, "eval_every": 50}
    scfg = {**cfg, "model": {**cfg["model"], "dropout": 0.0}}
    for arm in ("G", "P", "GP"):
        if st["results"].get(arm, {}).get("passed"):
            continue
        if st["attempts"] >= sc["max_attempts"]:
            st["blocker"] = "sanity attempt budget exhausted"
            break
        st["attempts"] += 1
        out = STATE / "sanity" / f"{arm}_attempt{st['attempts']}"
        s = train(arm, 0, out, scfg, data=data, val=[], dev=[], tc=tc, name=f"sanity_{arm}")
        curve = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]
        st["results"][arm] = {"attempt": st["attempts"], "folder": str(out.relative_to(ROOT)),
                              "final_train_agreement": s["final_train_agreement"],
                              "passed": s["final_train_agreement"] >= sc["threshold"],
                              "curve": [(r.get("update"), round(r.get("train_agreement", 0), 3),
                                         round(r.get("train_loss_window", 0), 3)) for r in curve],
                              "lr_applied": s["lr_applied"], "updates": s["updates"]}
        STATE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(st, indent=1) + "\n")
    st["subset"] = info
    st["passed"] = all(st["results"].get(a, {}).get("passed") for a in ("G", "P", "GP"))
    path.write_text(json.dumps(st, indent=1) + "\n")
    (REPORTS / "sanity.json").write_text(json.dumps(st, indent=1) + "\n")
    return st


# ------------------------------------------------------------------ development comparison and branch
def predict_set(model_dir: Path, its: list, view: str | None = None, width: int = 4) -> dict:
    """`{item id: [(text, score), ...]}` for one model in one view (default: its main view)."""
    npm, meta = load_np(model_dir / "model.npz")
    voc, _ = vocab()
    v = view or meta["main_view"]
    res = beam(npm, [encode(i.word, phones_of(i.word), v, voc) for i in its], width)
    return {i.id: r for i, r in zip(its, res)}


def set_scores(its, preds, k=4) -> dict:
    """Top-1 macro, top-k macro, CER, plus per-item outcomes."""
    sc = ev.score_predictions(its, {i: [t for t, _ in p] for i, p in preds.items()}, k)
    return {"macro": sc["primary_macro"], "item_accuracy": sc["item_accuracy"], "correct": sc["correct_items"],
            "topk_macro": sc["topk_macro"], "mean_cer": sc["mean_cer"], "n": sc["n_items"],
            "families": sc["n_families"], "items": sc["items"]}


def phone_seen() -> dict:
    """Per-item phone-seen flags from the audit."""
    return json.loads((ROOT / "data" / "e04" / "phone_seen.json").read_text(encoding="utf-8"))


def devcheck() -> dict:
    """Pilot dev comparison (beam-4, primary dev items) and the registered branch decision."""
    cfg = load_cfg()
    dv = primary(items("dev"))
    seen = phone_seen()
    preds = {a: predict_set(RUNS / f"{a}_s0", dv) for a in ("G", "P", "GP")}
    sc = {a: set_scores(dv, p) for a, p in preds.items()}
    union = {i.id: [(t, 0.0) for t in dict.fromkeys([t for t, _ in preds["G"][i.id]] + [t for t, _ in preds["P"][i.id]])]
             for i in dv}
    union_cov = ev.macro(dv, {i.id: int(any(ps.nfc(t) in {ps.nfc(a) for a in i.accepted} for t, _ in union[i.id]))
                              for i in dv})
    g_cov = sc["G"]["topk_macro"]
    gain = (union_cov - g_cov) * 100
    useful = sc["P"]["macro"] > sc["G"]["macro"] or gain >= cfg["rules"]["signal_union_gain_pp"]
    gp_strongest = sc["GP"]["macro"] >= sc["P"]["macro"] and sc["GP"]["macro"] >= sc["G"]["macro"]
    branch = "A" if not useful else "B" if gp_strongest else "C"
    io = {a: sc[a]["items"] for a in sc}
    p_not_g = [i.id for i in dv if io["P"][i.id]["topk"] and not io["G"][i.id]["topk"]]
    g_not_p = [i.id for i in dv if io["G"][i.id]["topk"] and not io["P"][i.id]["topk"]]
    p1_not_g1 = [i.id for i in dv if io["P"][i.id]["top1"] and not io["G"][i.id]["top1"]]
    g1_not_p1 = [i.id for i in dv if io["G"][i.id]["top1"] and not io["P"][i.id]["top1"]]
    strata = {}
    for flag in (True, False):
        sub = [i for i in dv if seen[f"dev:{i.id}"]["phone_seen_in_training"] is flag]
        if sub:
            strata[f"phone_seen={flag}"] = {"items": len(sub), **{a: set_scores(sub, {i.id: preds[a][i.id] for i in sub})["macro"]
                                                                 for a in preds}}
    out = {"set": "E03 dev, primary items (beam-4)", "items": len(dv),
           "scores": {a: {k: v for k, v in s.items() if k != "items"} for a, s in sc.items()},
           "g_top4_macro": g_cov, "g_union_p_top4_macro": union_cov, "union_gain_pp": gain,
           "p_macro_exceeds_g": sc["P"]["macro"] > sc["G"]["macro"], "useful_signal": useful,
           "gp_strongest": gp_strongest, "branch": branch,
           "complementarity": {"p_top4_not_g_top4": len(p_not_g), "g_top4_not_p_top4": len(g_not_p),
                               "p_top1_not_g_top1": len(p1_not_g1), "g_top1_not_p_top1": len(g1_not_p1),
                               "examples_p_not_g": [(i, [t for t, _ in preds["P"][i]][:2], [t for t, _ in preds["G"][i]][:2])
                                                    for i in p1_not_g1[:12]],
                               "examples_g_not_p": [(i, [t for t, _ in preds["G"][i]][:2], [t for t, _ in preds["P"][i]][:2])
                                                    for i in g1_not_p1[:12]]},
           "strata": strata,
           "rule": "useful = P macro > G macro, or G∪P top-4 coverage − G top-4 ≥ 5 pp; B if useful and GP ≥ G and "
                   "GP ≥ P; C if useful otherwise; A if not useful"}
    (REPORTS / "pilot_dev.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with open(REPORTS / "pilot_dev_per_item.jsonl", "w", encoding="utf-8") as f:
        for i in dv:
            f.write(json.dumps({"id": i.id, **{a: {"top4": [t for t, _ in preds[a][i.id]], "top1": io[a][i.id]["top1"],
                                                   "topk": io[a][i.id]["topk"], "cer": io[a][i.id]["cer"]} for a in preds},
                                "phone_seen": seen[f"dev:{i.id}"]["phone_seen_in_training"]},
                               ensure_ascii=False) + "\n")
    return out


def select(st_jobs: list[dict]) -> dict:
    """Candidate selection on dev (beam-4 macro, primary dev): eligible P, GP and (if run) V3; ≥ G + margin."""
    cfg = load_cfg()
    dv = primary(items("dev"))
    done = {j["arm"] for j in st_jobs if j["seed"] == 0 and j["state"] == "completed"}
    sc = {a: set_scores(dv, predict_set(RUNS / f"{a}_s0", dv))["macro"] for a in ["G", *SIMPLICITY] if a in done}
    return {"dev_macro_seed0": sc, "margin_pp": cfg["rules"]["screen_margin_pp"],
            **choose(sc, cfg["rules"]["screen_margin_pp"])}


def choose(sc: dict[str, float], margin_pp: float) -> dict:
    """The registered screen: eligible arms with dev macro ≥ G + margin; highest wins, ties to the simpler recipe.

    Examples:
        >>> choose({"G": 0.40, "P": 0.42, "GP": 0.42}, 2.0)["candidate"]
        'P'
        >>> choose({"G": 0.40, "P": 0.41, "GP": 0.419}, 2.0)["candidate"] is None
        True
    """
    passing = [a for a in SIMPLICITY if a in sc and (sc[a] - sc["G"]) * 100 >= margin_pp - 1e-9]
    best = max(passing, key=lambda a: (sc[a], -SIMPLICITY.index(a))) if passing else None
    return {"passing": passing, "candidate": best}


def combine() -> dict:
    """Exploratory: union G/P top-4 per dev item, rescore under both (mean log-prob per token), weights on dev."""
    dv = primary(items("dev"))
    voc, _ = vocab()
    g, _m = load_np(RUNS / "G_s0" / "model.npz")
    p, _m = load_np(RUNS / "P_s0" / "model.npz")
    pg, pp_ = predict_set(RUNS / "G_s0", dv), predict_set(RUNS / "P_s0", dv)
    t0 = time.time()
    scored = {}
    for it in dv:
        ph = phones_of(it.word)
        sg, sp = encode(it.word, ph, "g", voc), encode(it.word, ph, "p", voc)
        cands = list(dict.fromkeys([t for t, _ in pg[it.id]] + [t for t, _ in pp_[it.id]]))
        scored[it.id] = [(c, *(lambda a, b: (a[0] / a[1], b[0] / b[1]))(tf_score(g, sg, c), tf_score(p, sp, c)))
                         for c in cands]
    rescore_s = (time.time() - t0) / len(dv)
    res = {}
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        top = {i: [max(c, key=lambda x: w * x[1] + (1 - w) * x[2])[0]] for i, c in scored.items()}
        res[str(w)] = ev.score_predictions(dv, top, 1)["primary_macro"]
    best_w = max(res, key=lambda w: (res[w], float(w)))
    out = {"note": "exploratory reference only: two models, not a deployable candidate",
           "weights_dev_macro (w = G weight)": res, "selected_w": best_w,
           "normalisation": "mean log-probability per target token (characters + EOS)",
           "extra_runtime_s_per_word": rescore_s, "requires": "both G and P models and the CMUdict reading"}
    (REPORTS / "combination_dev.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


# ------------------------------------------------------------------ final evaluation (historical E03 eval set)
def evaluate() -> dict:
    """All completed E04 models on the E03 eval set, the confirmed pair, strata and the alternative-reading
    diagnostic. Reads `state/e04/status.json` for the frozen selection."""
    cfg = load_cfg()
    st = json.loads((STATE / "status.json").read_text())
    ev_items = items("eval")
    prim = primary(ev_items)
    seen = phone_seen()
    voc, _ = vocab()
    models = sorted(j["id"] for j in st["jobs"] if j["state"] == "completed")
    preds = {m: predict_set(RUNS / m, ev_items) for m in models}
    out = {"set": "E03 eval partition (historical; informed earlier work; unqualified references)",
           "selection": st.get("selection"), "branch": st.get("branch"), "models": {}, "comparisons": {}}
    for m in models:
        p = {i.id: preds[m][i.id] for i in prim}
        s = set_scores(prim, p)
        fail = Counter("ok" if s["items"][i]["top1"] else "ranking" if s["items"][i]["topk"] else "generation"
                       for i in s["items"])
        strata = {}
        for flag in (True, False):
            sub = [i for i in prim if seen[f"eval:{i.id}"]["phone_seen_in_training"] is flag]
            strata[f"phone_seen={flag}"] = {"items": len(sub), "macro": set_scores(sub, {i.id: p[i.id] for i in sub})["macro"]
                                            if sub else None}
        flagged = [i for i in ev_items if PRIMARY_EXCLUDE & set(i.flags)]
        out["models"][m] = {**{k: v for k, v in s.items() if k != "items"}, "failures": dict(fail), "strata": strata,
                            "flagged_track": set_scores(flagged, {i.id: preds[m][i.id] for i in flagged})["macro"]
                            if flagged else None,
                            "model_sha256": ps.sha256_file(RUNS / m / "model.npz")}
    cand = (st.get("selection") or {}).get("candidate")
    pairs = [("G", a) for a in ("P", "GP", "V3") if f"{a}_s0" in models]
    for ctl, trt in pairs:
        seeds = [s for s in (0, 1, 2) if f"{ctl}_s{s}" in models and f"{trt}_s{s}" in models]
        oc = {m: set_scores(prim, {i.id: preds[m][i.id] for i in prim})["items"] for m in models}
        L = [{i: v["top1"] for i, v in oc[f"{ctl}_s{s}"].items()} for s in seeds]
        P_ = [{i: v["top1"] for i, v in oc[f"{trt}_s{s}"].items()} for s in seeds]
        c = ev.compare_paired_runs(prim, L, P_, cfg["rules"]["n_boot"], cfg["rules"]["boot_seed"])
        c["seeds"] = seeds
        c["label"] = "confirmatory (3 paired seeds)" if trt == cand and len(seeds) == 3 else "pilot (exploratory, one seed)"
        out["comparisons"][f"{trt}_vs_{ctl}"] = c
    out["alternative_readings"] = alternatives(models, voc)
    (EXP / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (EXP / "results").mkdir(exist_ok=True)
    with open(EXP / "results" / "eval_per_item.jsonl", "w", encoding="utf-8") as f:
        for m in models:
            for i in ev_items:
                f.write(json.dumps({"model": m, "id": i.id, "top4": [t for t, _ in preds[m][i.id]],
                                    "phone_seen": seen[f"eval:{i.id}"]["phone_seen_in_training"]}, ensure_ascii=False) + "\n")
    return {k: v for k, v in out.items() if k != "alternative_readings"}


def alternatives(models: list[str], voc) -> dict:
    """Documented alternative CMUdict readings (E03's frozen eval sample): output changes per phone-using model."""
    fz = json.loads((ROOT / "experiments" / "E03-assessable-phones" / "frozen_eval.json").read_text(encoding="utf-8"))
    km_lex, _ = DG.load_lexicon()
    res = {}
    for m in models:
        npm, meta = load_np(RUNS / m / "model.npz")
        if meta["main_view"] == "g":
            continue
        rows_ = []
        for it in fz["samples"]["eval"]:
            alts = [c for c in it["conditions"] if c["condition"] == "alternative"]
            if not alts:
                continue
            base = beam(npm, [encode(it["word"], tuple(it["conditions"][0]["phones"].split()), meta["main_view"], voc)])[0]
            for c in alts:
                alt = beam(npm, [encode(it["word"], tuple(c["phones"].split()), meta["main_view"], voc)])[0]
                acc = {ps.nfc(a) for a in it["accepted"]}
                rows_.append({"word": it["word"], "default": it["conditions"][0]["phones"], "alternative": c["phones"],
                              "default_top1": base[0][0], "alt_top1": alt[0][0], "changed": base[0][0] != alt[0][0],
                              "default_agrees": base[0][0] in acc, "alt_agrees": alt[0][0] in acc,
                              "both_in_lexicon": base[0][0] in km_lex and alt[0][0] in km_lex})
        res[m] = {"cases": len(rows_), "changed": sum(r["changed"] for r in rows_),
                  "agreement_default": sum(r["default_agrees"] for r in rows_),
                  "agreement_alternative": sum(r["alt_agrees"] for r in rows_),
                  "vowel_assessable_both_lexicon": sum(r["both_in_lexicon"] for r in rows_), "rows": rows_}
    return res


def retention() -> dict:
    """Post-hoc (not registered): from the saved eval predictions only, does GP keep what only P gets right?

    Uses `results/eval_per_item.jsonl`; makes no new predictions. Seed 0 is the only seed with all three arms.
    Writes `reports/retention_posthoc.json`.

    Returns:
        Counts of P-only / G-only top-1 and top-4 hits on the primary eval items and how many of each GP keeps,
        plus G ∪ P top-4 coverage (macro).
    """
    prim = primary(items("eval"))
    acc = {i.id: {ps.nfc(a) for a in i.accepted} for i in prim}
    top4 = defaultdict(dict)
    for line in (EXP / "results" / "eval_per_item.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        top4[r["model"]][r["id"]] = [ps.nfc(t) for t in r["top4"]]
    hit = {m: {i.id: (top4[m][i.id][:1][0] in acc[i.id] if top4[m][i.id] else False,
                      any(t in acc[i.id] for t in top4[m][i.id])) for i in prim} for m in ("G_s0", "P_s0", "GP_s0", "G_s1")}
    out = {"label": "post-hoc, exploratory, seed 0, E03 eval primary items", "items": len(prim)}
    for k, name in ((0, "top1"), (1, "top4")):
        p_only = [i for i in hit["P_s0"] if hit["P_s0"][i][k] and not hit["G_s0"][i][k]]
        g_only = [i for i in hit["G_s0"] if hit["G_s0"][i][k] and not hit["P_s0"][i][k]]
        out[name] = {"p_only": len(p_only), "gp_keeps_p_only": sum(hit["GP_s0"][i][k] for i in p_only),
                     "g_only": len(g_only), "gp_keeps_g_only": sum(hit["GP_s0"][i][k] for i in g_only),
                     "both": sum(hit["G_s0"][i][k] and hit["P_s0"][i][k] for i in hit["G_s0"]),
                     "gp_keeps_both": sum(hit["GP_s0"][i][k] for i in hit["G_s0"]
                                          if hit["G_s0"][i][k] and hit["P_s0"][i][k]),
                     "p_only_examples": p_only[:10]}
    # Seed-churn baseline: a second G seed also "wins" items G_s0 misses; how many of those does GP_s0 keep?
    churn = [i for i in hit["G_s1"] if hit["G_s1"][i][0] and not hit["G_s0"][i][0]]
    out["churn_baseline_top1"] = {"g_s1_only": len(churn), "gp_keeps_g_s1_only": sum(hit["GP_s0"][i][0] for i in churn)}
    out["g_top4_macro"] = ev.macro(prim, {i.id: int(hit["G_s0"][i.id][1]) for i in prim})
    out["g_union_p_top4_macro"] = ev.macro(prim, {i.id: int(hit["G_s0"][i.id][1] or hit["P_s0"][i.id][1]) for i in prim})
    out["gp_top4_macro"] = ev.macro(prim, {i.id: int(hit["GP_s0"][i.id][1]) for i in prim})
    (REPORTS / "retention_posthoc.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


def main() -> int:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["audit", "sanity", "train", "export", "verify", "devcheck", "combine", "evaluate", "retention"])
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.step == "train":
        train(a.arm, a.seed, Path(a.out), load_cfg())
    elif a.step == "export":
        export(Path(a.out))
    elif a.step == "verify":
        return 0 if verify(Path(a.out))["ok"] else 1
    else:
        r = {"audit": audit, "sanity": sanity, "devcheck": devcheck, "combine": combine, "evaluate": evaluate, "retention": retention}[a.step]()
        print(json.dumps(r, ensure_ascii=False, indent=1, default=str)[:5000])
        if a.step == "sanity" and not r["passed"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
