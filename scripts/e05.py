"""E05 — candidate diversity: does G+P add coverage beyond more candidates (G8) or a second letters model (G+G)?

Inference only, exploratory on historical data (E03/E04 dev and eval partitions). It uses the saved E04 models
`G_s0`, `G_s1`, `G_s2` and `P_s0`; there is no training, no reranker and no product change.

Steps (each resumable; state in `state/e05/status.json`):
    register  hashes of commit, checkpoints, data, saved predictions and decoder settings → registration.json
    predict   beam-8 candidates for the four models on dev and eval; beam-4 for dev G_s1/G_s2 (not saved by E04)
    analyse   systems G4, G8, G+G, G+P; coverage, budgets, overlap, strata, paired family bootstrap → results.json
    cost      warm/cold timing and peak RSS in fresh processes → reports/cost.json
    queue     blinded review queue (local, git-ignored) and its separate origin key

Example:
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e05.py run      # all steps, skipping finished ones
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e02_diagnostics as DG  # noqa: E402
import e04  # noqa: E402
from khmer_translit import phonetic_source as ps  # noqa: E402
from khmer_translit import translit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "E05-candidate-diversity"
EXP = ROOT / "experiments" / EXPERIMENT
PRED = EXP / "predictions"
REPORTS = EXP / "reports"
STATE = ROOT / "state" / "e05"
REVIEW = ROOT / "data" / "e05" / "review"           # git-ignored: local only
RUNS = ROOT / "models" / "e04"
G_MODELS = ("G_s0", "G_s1", "G_s2")
P_MODEL = "P_s0"
MODELS = G_MODELS + (P_MODEL,)
WIDTH = 8                                            # primary matched search width for every constituent
BUDGETS = (1, 2, 4, 8)
N_BOOT = 10_000
BOOT_SEED = 20261027
REVIEW_SEED = "e05-review"
SPLITS = ("dev", "eval")
# E02's final references, benchmark panels and the reserve (E02 diagnostics list) plus E02's final predictions
FORBIDDEN = DG.FORBIDDEN + ("data/e02/final", "final_predictions")
SAVED_W4 = {"eval": EXP.parent / "E04-pronunciation-pathways" / "results" / "eval_per_item.jsonl",
            "dev": EXP.parent / "E04-pronunciation-pathways" / "reports" / "pilot_dev_per_item.jsonl"}
DECODER = {"search": "NumPy beam search (khmer_translit.translit.Transliterator.beam_search)",
           "score": "sum of token log-probabilities, EOS included; no length normalisation",
           "max_output_tokens": translit.MAX_OUT, "source": "scripts/e04.py encode (main view of each model)",
           "pronunciation": "CMUdict first listed reading with stress (scripts/e04.py phones_of)",
           "candidates": "NFC; empty strings dropped; identical strings deduplicated keeping the best rank"}


def guard(path: Path | str) -> Path:
    """Refuse forbidden evaluation/reserve paths.

    Raises:
        PermissionError: For E02's final references or results, benchmark panels, the reserve.

    Examples:
        >>> guard(ROOT / "data/e03/eval_references.jsonl").name
        'eval_references.jsonl'
    """
    p = str(Path(path).resolve())
    if any(f in p for f in FORBIDDEN):
        raise PermissionError(f"E05 may not read {p}")
    return Path(path)


def sha(path: Path) -> str:
    """sha256 of a file (through the guard)."""
    return hashlib.sha256(guard(path).read_bytes()).hexdigest()


def now() -> str:
    """UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ candidate rules (pure)
def dedup(cands: list[str]) -> list[str]:
    """NFC-normalise, drop empty strings, keep the first occurrence of each string.

    Examples:
        >>> dedup(["e\u0301", "\u00e9", "", "x"]) == ["\u00e9", "x"]
        True
    """
    seen, out = set(), []
    for c in cands:
        c = ps.nfc(c)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def top(cands: list[str], k: int) -> list[str]:
    """The first `k` distinct candidates."""
    return dedup(cands)[:k]


def interleave(a: list[str], b: list[str], k: int) -> list[str]:
    """Alternate the ranked lists `a` and `b` (a first), skipping duplicates, up to `k` distinct candidates.

    No filling beyond the two lists: the result may be shorter than `k`.

    Examples:
        >>> interleave(["x", "y", "z"], ["y", "w"], 4)
        ['x', 'y', 'w', 'z']
    """
    a, b = dedup(a), dedup(b)
    out, seen = [], set()
    for i in range(max(len(a), len(b))):
        for lst in (a, b):
            if i < len(lst) and lst[i] not in seen:
                seen.add(lst[i])
                out.append(lst[i])
    return out[:k]


def check_ids(item_ids: set[str], preds: dict) -> None:
    """Prediction ids must equal the item ids.

    Raises:
        ValueError: Naming missing or unknown ids.
    """
    missing, unknown = item_ids - set(preds), set(preds) - item_ids
    if missing or unknown:
        raise ValueError(f"missing ids {sorted(missing)[:5]} ({len(missing)}), unknown ids {sorted(unknown)[:5]} "
                         f"({len(unknown)})")


def covered(accepted: list[str], cands: list[str]) -> int:
    """1 if any candidate equals any accepted reference after NFC; an empty candidate list is a failure (0).

    Examples:
        >>> covered(["ក", "ខ"], ["គ", "ខ"]), covered(["ក"], [])
        (1, 0)
    """
    acc = {ps.nfc(a) for a in accepted}
    return int(any(c in acc for c in dedup(cands)))


# ------------------------------------------------------------------ systems
def build_systems(cand: dict[str, dict[str, list[str]]], ids: list[str]) -> dict[str, dict]:
    """All comparison systems from per-model ranked candidates.

    Args:
        cand: `{model: {item id: ranked candidates}}` (already deduplicated per model).
        ids: Item ids.

    Returns:
        `{system name: {"family": G4/G8/G+G/G+P or an ordered budget system, "constituents": [...], "slots": n,
        "k": budget or None, "cands": {id: [...]}}}`. Constituents record shared checkpoints (P_s0 in every G+P).
    """
    S = {}
    for g in G_MODELS:
        S[f"G4[{g}]"] = {"family": "G4", "constituents": [g], "slots": 4, "k": None,
                         "cands": {i: top(cand[g][i], 4) for i in ids}}
        S[f"G8[{g}]"] = {"family": "G8", "constituents": [g], "slots": 8, "k": None,
                         "cands": {i: top(cand[g][i], 8) for i in ids}}
        S[f"G+P[{g}]"] = {"family": "G+P", "constituents": [g, P_MODEL], "slots": 8, "k": None,
                          "cands": {i: dedup(top(cand[g][i], 4) + top(cand[P_MODEL][i], 4)) for i in ids}}
        for k in BUDGETS:
            S[f"G@{k}[{g}]"] = {"family": "G@k", "constituents": [g], "slots": k, "k": k,
                                "cands": {i: top(cand[g][i], k) for i in ids}}
            for a, b in ((g, P_MODEL), (P_MODEL, g)):
                S[f"G+P@{k}[{a}>{b}]"] = {"family": "G+P@k", "constituents": [a, b], "slots": k, "k": k,
                                          "cands": {i: interleave(top(cand[a][i], 4), top(cand[b][i], 4), k)
                                                    for i in ids}}
    for g, h in combinations(G_MODELS, 2):
        S[f"G+G[{g},{h}]"] = {"family": "G+G", "constituents": [g, h], "slots": 8, "k": None,
                              "cands": {i: dedup(top(cand[g][i], 4) + top(cand[h][i], 4)) for i in ids}}
        for k in BUDGETS:
            for a, b in ((g, h), (h, g)):
                S[f"G+G@{k}[{a}>{b}]"] = {"family": "G+G@k", "constituents": [a, b], "slots": k, "k": k,
                                          "cands": {i: interleave(top(cand[a][i], 4), top(cand[b][i], 4), k)
                                                    for i in ids}}
    return S


def shared_checkpoints(S: dict[str, dict]) -> dict[str, list[str]]:
    """`{checkpoint: [systems using it]}` for the unordered full-set systems (bookkeeping for the report)."""
    out: dict[str, list[str]] = {}
    for name, s in S.items():
        if s["k"] is None:
            for c in s["constituents"]:
                out.setdefault(c, []).append(name)
    return out


# ------------------------------------------------------------------ metrics and bootstrap
def family_index(items) -> tuple[list[str], dict[str, int]]:
    """Ordered family names and `{item id: family position}`."""
    fams = sorted({i.family for i in items})
    pos = {f: n for n, f in enumerate(fams)}
    return fams, {i.id: pos[i.family] for i in items}


def family_means(items, per_item: dict[str, float]) -> np.ndarray:
    """Per-family mean of an item value (the macro is the mean of this vector)."""
    fams, where = family_index(items)
    tot, cnt = np.zeros(len(fams)), np.zeros(len(fams))
    for i in items:
        tot[where[i.id]] += per_item[i.id]
        cnt[where[i.id]] += 1
    return tot / cnt


def boot_indices(n_fam: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """One shared `(n_boot, n_fam)` matrix of resampled family positions, so every contrast sees the same draws."""
    return np.random.default_rng(seed).integers(0, n_fam, size=(n_boot, n_fam))


def contrast(fm: dict[str, np.ndarray], plus: list[str], minus: list[str], idx: np.ndarray) -> dict:
    """mean(plus systems) − mean(minus systems) in macro points, with a paired family-bootstrap 95% interval.

    Every system's family vector is resampled with the same rows of `idx` (families, not items).
    """
    d = np.mean([fm[s] for s in plus], 0) - np.mean([fm[s] for s in minus], 0)
    b = np.sort(d[idx].mean(1)) * 100
    n = len(b)
    return {"plus": plus, "minus": minus, "diff_pp": round(float(d.mean() * 100), 3),
            "ci95_pp": [round(float(b[int(0.025 * n)]), 3), round(float(b[int(0.975 * n) - 1]), 3)]}


def system_metrics(items, s: dict) -> dict:
    """Coverage (item and family macro), distinct counts, duplicate rate, empty and under-filled counts."""
    cov = {i.id: covered(i.accepted, s["cands"][i.id]) for i in items}
    n = [len(s["cands"][i.id]) for i in items]
    return {"item_coverage": round(mean(cov.values()), 4), "macro_coverage": round(float(family_means(items, cov).mean()), 4),
            "items": len(items), "families": len({i.family for i in items}),
            "distinct_mean": round(mean(n), 3), "distinct_min": min(n), "distinct_max": max(n),
            "duplicate_rate": round(1 - sum(n) / (s["slots"] * len(items)), 4),
            "underfilled": sum(x < s["slots"] for x in n), "empty": sum(x == 0 for x in n)}


# ------------------------------------------------------------------ status (durable progress)
def status() -> dict:
    """The resume state."""
    p = STATE / "status.json"
    return json.loads(p.read_text()) if p.exists() else {"steps": {}}


def mark(step: str, outputs: list[Path], **extra) -> None:
    """Record a finished step with output hashes."""
    st = status()
    st["steps"][step] = {"done": True, "at": now(), "outputs": {str(p.relative_to(ROOT)): sha(p) for p in outputs}, **extra}
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "status.json").write_text(json.dumps(st, indent=1) + "\n")
    lines = [f"# E05 progress ({now()})", ""] + [f"- {k}: done {v['at']}" for k, v in st["steps"].items()]
    (STATE / "PROGRESS.md").write_text("\n".join(lines) + "\n")


def is_done(step: str) -> bool:
    """A step is done if recorded and all its outputs still have the recorded hashes."""
    s = status()["steps"].get(step)
    return bool(s) and all((ROOT / p).exists() and sha(ROOT / p) == h for p, h in s["outputs"].items())


# ------------------------------------------------------------------ data and predictions
def item_sets() -> dict[str, list]:
    """`{split: items}` (all E03 dev/eval items, through the guard)."""
    return {s: e04.items(s) for s in SPLITS}


def analysis_items(sets: dict, split: str, width: int) -> list:
    """Items of one configuration: E04 saved dev beam-4 outputs for primary items only, so dev beam-4 is primary-only."""
    return e04.primary(sets[split]) if (split, width) == ("dev", 4) else sets[split]


def pred_path(split: str, width: int, model: str) -> Path:
    return PRED / f"{split}_w{width}_{model}.jsonl"


def register() -> dict:
    """Hashes and settings, written before any new prediction."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    data = {str(p.relative_to(ROOT)): sha(p) for p in
            [e04.DATA / "dev_references.jsonl", e04.DATA / "eval_references.jsonl", e04.DATA / "train.tsv",
             ROOT / "data" / "e04" / "phone_seen.json", *SAVED_W4.values(),
             EXP.parent / "E04-pronunciation-pathways" / "results.json"]}
    out = {"experiment": EXPERIMENT, "registered_at": now(), "commit": commit,
           "label": "exploratory on historical data (E03/E04 partitions already examined); not fresh validation",
           "checkpoints": {m: sha(RUNS / m / "model.npz") for m in MODELS}, "data": data,
           "cmudict_sha256": ps.cmudict()[1], "source_vocab": e04.vocab()[1], "decoder": DECODER,
           "width": WIDTH, "budgets": list(BUDGETS), "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
           "code": {p: sha(ROOT / p) for p in ("scripts/e05.py", "scripts/e04.py", "src/khmer_translit/translit.py")}}
    (EXP / "registration.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def predict() -> dict:
    """Beam-8 for all four models on dev and eval; beam-4 for dev G_s1/G_s2. One process, sequential."""
    reg = json.loads((EXP / "registration.json").read_text())
    PRED.mkdir(parents=True, exist_ok=True)
    sets = item_sets()
    jobs = [(s, WIDTH, m) for s in SPLITS for m in MODELS] + [("dev", 4, m) for m in ("G_s1", "G_s2")]
    done, t0 = [], time.time()
    for split, w, m in jobs:
        p = pred_path(split, w, m)
        if sha(RUNS / m / "model.npz") != reg["checkpoints"][m]:
            raise RuntimeError(f"{m}: checkpoint hash differs from registration")
        if not p.exists():
            its = analysis_items(sets, split, w)
            preds = e04.predict_set(RUNS / m, its, width=w)
            tmp = p.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for i in its:
                    f.write(json.dumps({"id": i.id, "model": m, "width": w, "candidates": [t for t, _ in preds[i.id]],
                                        "scores": [round(s_, 6) for _, s_ in preds[i.id]]}, ensure_ascii=False) + "\n")
            os.replace(tmp, p)
        done.append(p)
    mark("predict", done, seconds=round(time.time() - t0, 1))
    return {"files": len(done), "seconds": round(time.time() - t0, 1)}


def load_preds(split: str, width: int, model: str, ids: set[str]) -> dict[str, list[str]]:
    """`{id: ranked candidates}` for one model, from E05 files (and E04's saved beam-4 where E05 made none)."""
    p = pred_path(split, width, model)
    out = {}
    if p.exists():
        for line in guard(p).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["width"] != width or r["model"] != model:
                raise ValueError(f"{p}: inconsistent beam configuration")
            out[r["id"]] = dedup(r["candidates"])
    elif width == 4:
        arm = model.split("_")[0]
        for line in guard(SAVED_W4[split]).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if split == "eval" and r["model"] == model:
                out[r["id"]] = dedup(r["top4"])
            elif split == "dev" and model.endswith("_s0"):
                out[r["id"]] = dedup(r[arm]["top4"])
    else:
        raise FileNotFoundError(p)
    check_ids(ids, out)
    return out


# ------------------------------------------------------------------ analysis
def analyse() -> dict:
    """All metrics and contrasts; writes results.json and results/per_item_{split}_w{width}.jsonl."""
    sets = item_sets()
    seen = e04.phone_seen()
    km_lex, _ = DG.load_lexicon()
    res = {"label": "exploratory on historical data; references mined, screened and unreviewed; coverage is an "
                    "oracle (reference-dependent) measure of candidate-generation opportunity only",
           "decoder": DECODER, "configs": {}}
    (EXP / "results").mkdir(exist_ok=True)
    for width in (WIDTH, 4):
        for split in SPLITS:
            its = analysis_items(sets, split, width)
            ids = {i.id for i in its}
            cand = {m: load_preds(split, width, m, ids) for m in MODELS}
            S = build_systems(cand, [i.id for i in its])
            prim = e04.primary(its)
            flagged = [i for i in its if e04.PRIMARY_EXCLUDE & set(i.flags)]
            key = f"{split}_w{width}"
            cfg = {"items_primary": len(prim), "families_primary": len({i.family for i in prim}),
                   "items_flagged": len(flagged),
                   "systems": {n: system_metrics(prim, s) for n, s in S.items()},
                   "flagged_track": {n: system_metrics(flagged, s) for n, s in S.items() if s["k"] is None}
                   if flagged else None,
                   "shared_checkpoints": shared_checkpoints(S)}
            cfg["strata"] = strata(prim, S, seen, split, km_lex, cand)
            cfg["overlap"] = overlap(prim, cand)
            cfg["wins"] = wins(prim, S)
            cfg["pool"] = pool(prim, cand, width)
            cfg["contrasts"] = contrasts(prim, S)
            res["configs"][key] = cfg
            with open(EXP / "results" / f"per_item_{key}.jsonl", "w", encoding="utf-8") as f:
                for i in its:
                    f.write(json.dumps({"id": i.id, "family": i.family, "primary": i in prim,
                                        "covered": {n: covered(i.accepted, s["cands"][i.id])
                                                    for n, s in S.items() if s["k"] is None},
                                        "distinct": {n: len(s["cands"][i.id]) for n, s in S.items() if s["k"] is None}},
                                       ensure_ascii=False) + "\n")
    res["search_width_effect"] = {
        split: {g: {"G4_at_w4": res["configs"][f"{split}_w4"]["systems"][f"G4[{g}]"]["macro_coverage"],
                    "G4_at_w8": res["configs"][f"{split}_w8"]["systems"][f"G4[{g}]"]["macro_coverage"]}
                for g in MODELS[:3]} for split in SPLITS}
    res["e04_reproduction"] = {"eval_G_s0_top4": res["configs"]["eval_w4"]["systems"]["G4[G_s0]"]["macro_coverage"],
                               "eval_G_s0_union_P_top4": res["configs"]["eval_w4"]["systems"]["G+P[G_s0]"]["macro_coverage"],
                               "e04_reported": {"eval_G_s0_top4": 0.6456, "eval_union": 0.7267, "dev_G_top4": 0.642,
                                                "dev_union": 0.76},
                               "dev_G_s0_top4": res["configs"]["dev_w4"]["systems"]["G4[G_s0]"]["macro_coverage"],
                               "dev_union": res["configs"]["dev_w4"]["systems"]["G+P[G_s0]"]["macro_coverage"]}
    res["verdict"] = verdict(res["configs"]["eval_w8"]["contrasts"])
    (EXP / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    mark("analyse", [EXP / "results.json"] + sorted((EXP / "results").glob("per_item_*.jsonl")))
    return res["verdict"]


def contrasts(prim, S: dict) -> dict:
    """Registered paired contrasts (macro coverage points), all sharing one family-resampling matrix."""
    cov = {n: {i.id: covered(i.accepted, s["cands"][i.id]) for i in prim} for n, s in S.items()}
    fm = {n: family_means(prim, v) for n, v in cov.items()}
    idx = boot_indices(len(next(iter(fm.values()))))
    out = {"per_combination": {}, "summary": {}, "budget4": {}}
    gp = [f"G+P[{g}]" for g in G_MODELS]
    gg = [f"G+G[{g},{h}]" for g, h in combinations(G_MODELS, 2)]
    for g in G_MODELS:
        c = {"vs_G4": contrast(fm, [f"G+P[{g}]"], [f"G4[{g}]"], idx),
             "vs_G8": contrast(fm, [f"G+P[{g}]"], [f"G8[{g}]"], idx)}
        for pair in gg:
            if g in pair:
                c[f"vs_{pair}"] = contrast(fm, [f"G+P[{g}]"], [pair], idx)
        out["per_combination"][g] = c
    out["summary"] = {"G+P_vs_G8": contrast(fm, gp, [f"G8[{g}]" for g in G_MODELS], idx),
                      "G+P_vs_G+G": contrast(fm, gp, gg, idx),
                      "G+P_vs_G4": contrast(fm, gp, [f"G4[{g}]" for g in G_MODELS], idx),
                      "G8_vs_G4": contrast(fm, [f"G8[{g}]" for g in G_MODELS], [f"G4[{g}]" for g in G_MODELS], idx),
                      "G+G_vs_G4": contrast(fm, gg, [f"G4[{g}]" for g in G_MODELS], idx)}
    # equal four-slot budget, both model-priority orders reported (no order chosen)
    for first in ("G", "P"):
        gp4 = [f"G+P@4[{g}>{P_MODEL}]" if first == "G" else f"G+P@4[{P_MODEL}>{g}]" for g in G_MODELS]
        out["budget4"][f"G+P@4({first}-first)_vs_G4"] = contrast(fm, gp4, [f"G@4[{g}]" for g in G_MODELS], idx)
        out["budget4"][f"G+P@4({first}-first)_vs_G+G@4"] = contrast(
            fm, gp4, [n for n in S if n.startswith("G+G@4[")], idx)
    out["budget4"]["G+G@4_vs_G4"] = contrast(fm, [n for n in S if n.startswith("G+G@4[")],
                                             [f"G@4[{g}]" for g in G_MODELS], idx)
    out["procedure"] = {"resampled_unit": "lexical family", "n_boot": N_BOOT, "seed": BOOT_SEED,
                        "interval": "95% percentile, every system resampled with the same family draws",
                        "conditional_on": "these trained checkpoints (G_s0..s2, P_s0); every G+P shares P_s0; "
                                          "comparisons share checkpoints; no training-population uncertainty"}
    return out


def verdict(c: dict) -> dict:
    """Registered reading of the eval beam-8 contrasts (PLAN.md, 'Decision rule')."""
    s = c["summary"]
    lo8, logg = s["G+P_vs_G8"]["ci95_pp"][0], s["G+P_vs_G+G"]["ci95_pp"][0]
    per = [v["diff_pp"] for g in c["per_combination"].values() for k, v in g.items() if k != "vs_G4"]
    widths = [v["ci95_pp"][1] - v["ci95_pp"][0] for v in s.values()]
    consistent = lo8 > 0 and logg > 0 and all(p > 0 for p in per)
    mixed = any(p > 0 for p in per) and any(p <= 0 for p in per)
    return {"outcome": "consistent advantage over both controls" if consistent else
            "no clear advantage over G8 and G+G",
            "uncertain_or_varied": mixed or max(widths) > 10, "per_combination_signs_mixed": mixed,
            "summary_lower_bounds_pp": {"vs_G8": lo8, "vs_G+G": logg}, "per_combination_diffs_pp": per}


def strata(prim, S, seen, split, km_lex, cand) -> dict:
    """Full-set coverage by phone-seen flag and by a reference-free dictionary-hit flag (G_s0 top-1 in lexicon)."""
    out = {"reference_in_lexicon": sum(any(ps.nfc(a) in km_lex for a in i.accepted) for i in prim)}
    groups = {}
    for i in prim:
        groups.setdefault(f"phone_seen={seen[f'{split}:{i.id}']['phone_seen_in_training']}", []).append(i)
        hit = bool(cand["G_s0"][i.id]) and cand["G_s0"][i.id][0] in km_lex
        groups.setdefault(f"G_s0_top1_in_lexicon={hit}", []).append(i)
    for gname, its in sorted(groups.items()):
        out[gname] = {"items": len(its), **{n: round(mean(covered(i.accepted, s["cands"][i.id]) for i in its), 4)
                                            for n, s in S.items() if s["k"] is None}}
    return out


def overlap(prim, cand) -> dict:
    """Shared distinct candidates between the two constituents' top-4 (mean count and Jaccard)."""
    def o(a, b):
        x = [len(set(top(cand[a][i.id], 4)) & set(top(cand[b][i.id], 4))) for i in prim]
        j = [len(set(top(cand[a][i.id], 4)) & set(top(cand[b][i.id], 4))) /
             max(1, len(set(top(cand[a][i.id], 4)) | set(top(cand[b][i.id], 4)))) for i in prim]
        return {"shared_mean": round(mean(x), 3), "jaccard_mean": round(mean(j), 3),
                "top1_equal": sum(bool(cand[a][i.id]) and bool(cand[b][i.id]) and cand[a][i.id][0] == cand[b][i.id][0]
                                  for i in prim)}
    pairs = {f"{g}&{P_MODEL}": o(g, P_MODEL) for g in G_MODELS}
    pairs.update({f"{g}&{h}": o(g, h) for g, h in combinations(G_MODELS, 2)})
    return pairs


def wins(prim, S) -> dict:
    """Per G checkpoint: items G+P covers that neither control covers, and the reverse."""
    out = {}
    for g in G_MODELS:
        ctl = [f"G8[{g}]"] + [n for n in S if n.startswith("G+G[") and g in n]
        cov = {n: {i.id: covered(i.accepted, S[n]["cands"][i.id]) for i in prim} for n in ctl + [f"G+P[{g}]"]}
        gp_only = [i.id for i in prim if cov[f"G+P[{g}]"][i.id] and not any(cov[c][i.id] for c in ctl)]
        ctl_any = [i.id for i in prim if not cov[f"G+P[{g}]"][i.id] and any(cov[c][i.id] for c in ctl)]
        ctl_all = [i.id for i in prim if not cov[f"G+P[{g}]"][i.id] and all(cov[c][i.id] for c in ctl)]
        out[g] = {"controls": ctl, "gp_not_any_control": len(gp_only), "any_control_not_gp": len(ctl_any),
                  "all_controls_not_gp": len(ctl_all), "gp_not_any_control_ids": gp_only,
                  "all_controls_not_gp_ids": ctl_all}
    return out


def pool(prim, cand, width) -> dict:
    """Reference present anywhere in the pooled candidates of all four models (top-`width`) vs absent."""
    pres = [i.id for i in prim if covered(i.accepted, [c for m in MODELS for c in cand[m][i.id][:width]])]
    return {"present": len(pres), "absent": len(prim) - len(pres),
            "distinct_mean": round(mean(len(dedup([c for m in MODELS for c in cand[m][i.id][:width]])) for i in prim), 2)}


# ------------------------------------------------------------------ practical cost (fresh processes)
COST_CONFIGS = [("G4", ["G_s0"], 4), ("G8", ["G_s0"], 8), ("G+P (beam-4 each)", ["G_s0", "P_s0"], 4),
                ("G+P (beam-8 each, matched protocol)", ["G_s0", "P_s0"], 8), ("G+G (beam-4 each)", ["G_s0", "G_s1"], 4)]
COST_SAMPLE = 100
COST_REPS = 3


def cost_sample() -> list:
    """Fixed dev sample: the first 100 primary dev items by sha256('e05-cost:'+id)."""
    its = e04.primary(e04.items("dev"))
    return sorted(its, key=lambda i: hashlib.sha256(f"e05-cost:{i.id}".encode()).hexdigest())[:COST_SAMPLE]


def probe(models: list[str], width: int) -> dict:
    """Run inside a fresh process: cold load, warm per-word latency (batch 1), peak RSS."""
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t = time.perf_counter()
    loaded = {m: e04.load_np(RUNS / m / "model.npz") for m in models}
    cold = time.perf_counter() - t
    rss_loaded = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    voc, _ = e04.vocab()
    its = cost_sample()
    srcs = {m: [e04.encode(i.word, e04.phones_of(i.word), loaded[m][1]["main_view"], voc) for i in its] for m in models}
    t = time.perf_counter()
    for m in models:                                    # first call per model (includes first-use effects)
        e04.beam(loaded[m][0], [srcs[m][0]], width)
    first = time.perf_counter() - t
    reps, counts = [], []
    for _ in range(COST_REPS):
        t = time.perf_counter()
        for n in range(len(its)):
            c = []
            for m in models:
                c += e04.beam(loaded[m][0], [srcs[m][n]], width)[0][:4 if len(models) > 1 else width]
            counts.append(len(dedup([x for x, _ in c])))
        reps.append((time.perf_counter() - t) / len(its) * 1000)
    return {"models": models, "width": width, "cold_load_s": round(cold, 3), "first_call_s": round(first, 3),
            "warm_ms_per_word": [round(r, 2) for r in reps], "warm_ms_per_word_median": round(sorted(reps)[1], 2),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "rss_after_imports_mb": round(rss0 / 1024, 1), "rss_after_load_mb": round(rss_loaded / 1024, 1),
            "distinct_candidates_mean": round(mean(counts), 2), "resident_models": len(models)}


def cost() -> dict:
    """Each configuration in its own fresh process, one at a time, 6 threads."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "6", "OPENBLAS_NUM_THREADS": "6",
           "MKL_NUM_THREADS": "6"}
    runs = []
    for name, models, w in COST_CONFIGS:
        t = time.perf_counter()
        r = subprocess.run([sys.executable, __file__, "probe", "--models", ",".join(models), "--width", str(w)],
                           env=env, capture_output=True, text=True, check=True)
        runs.append({"system": name, **json.loads(r.stdout.strip().splitlines()[-1]),
                     "process_wall_s": round(time.perf_counter() - t, 2)})
    cpu = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                if ln.startswith("model name")), platform.processor())
    out = {"sample": f"{COST_SAMPLE} primary dev items, sha256('e05-cost:'+id) order", "repetitions": COST_REPS,
           "batch": 1, "threads": 6, "machine": {"cpu": cpu, "logical_cpus": os.cpu_count(),
                                                 "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
                                                 "python": platform.python_version(), "numpy": np.__version__,
                                                 "kernel": platform.release()},
           "note": "peak RSS includes the Python/NumPy/package imports (rss_after_imports_mb); two-model systems keep "
                   "both models resident; timings are wall-clock on a laptop and vary between runs",
           "runs": runs}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "cost.json").write_text(json.dumps(out, indent=1) + "\n")
    mark("cost", [REPORTS / "cost.json"])
    return out


# ------------------------------------------------------------------ blinded review queue (local only)
REVIEW_PER_STRATUM = 10


def review_strata(prim, S) -> dict[str, list[str]]:
    """Eval primary items by agreement pattern of the G_s0 systems (beam-8); disjoint, reference-dependent."""
    ctl = ["G8[G_s0]"] + [n for n in S if n.startswith("G+G[") and "G_s0" in n]
    out = {"gp_only": [], "controls_only": [], "all_cover": [], "none_cover": [], "mixed": []}
    for i in prim:
        gp = covered(i.accepted, S["G+P[G_s0]"]["cands"][i.id])
        cs = [covered(i.accepted, S[c]["cands"][i.id]) for c in ctl]
        k = ("gp_only" if gp and not any(cs) else "controls_only" if not gp and any(cs) else
             "all_cover" if gp and all(cs) else "none_cover" if not gp and not any(cs) else "mixed")
        out[k].append(i.id)
    return out


def queue() -> dict:
    """Blinded queue (TSV + JSON) and a separate origin key in data/e05/review/ (git-ignored)."""
    its = e04.items("eval")
    prim = e04.primary(its)
    ids = {i.id for i in its}
    cand = {m: load_preds("eval", WIDTH, m, ids) for m in MODELS}
    S = build_systems(cand, [i.id for i in its])
    strata_ = review_strata(prim, S)
    order = lambda i: hashlib.sha256(f"{REVIEW_SEED}:{i}".encode()).hexdigest()  # noqa: E731
    chosen = []
    for k in ("gp_only", "controls_only", "all_cover", "none_cover", "mixed"):
        chosen += [(k, i) for i in sorted(strata_[k], key=order)[:REVIEW_PER_STRATUM]]
    agree = [i.id for i in prim if cand["G_s0"][i.id][:1] == cand[P_MODEL][i.id][:1] and cand["G_s0"][i.id]]
    taken = {i for _, i in chosen}
    chosen += [("g_p_top1_agree", i) for i in sorted(agree, key=order) if i not in taken][:REVIEW_PER_STRATUM]
    by_id = {i.id: i for i in its}
    rng = random.Random(REVIEW_SEED)
    rng.shuffle(chosen)
    rows, key = [], {"note": "origin key: keep separate from the queue; do not show to reviewers", "items": {}}
    for n, (stratum, iid) in enumerate(chosen, 1):
        it = by_id[iid]
        origins: dict[str, list[str]] = {}
        sources = [("G_s0", 8), ("G_s1", 4), ("G_s2", 4), (P_MODEL, 4)]
        for m, k in sources:
            for r, c in enumerate(top(cand[m][iid], k), 1):
                origins.setdefault(c, []).append(f"{m}@{r}")
        for a in it.accepted:
            origins.setdefault(ps.nfc(a), []).append("mined_reference")
        texts = list(origins)
        random.Random(f"{REVIEW_SEED}:{iid}").shuffle(texts)
        item_key = f"w{n:03d}"
        key["items"][item_key] = {"item_id": iid, "stratum": stratum, "candidates": {}}
        for j, t in enumerate(texts, 1):
            cid = f"{item_key}-c{j:02d}"
            key["items"][item_key]["candidates"][cid] = {"text": t, "origins": origins[t]}
            rows.append({"item": item_key, "english": it.word, "pronunciation": it.intended_phones,
                         "candidate_id": cid, "candidate": t, "judgment": "", "comment": ""})
    REVIEW.mkdir(parents=True, exist_ok=True)
    cols = ["item", "english", "pronunciation", "candidate_id", "candidate", "judgment", "comment"]
    with open(REVIEW / "review_queue.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(r[c] for c in cols) + "\n")
    (REVIEW / "review_queue.json").write_text(json.dumps(
        {"instructions": "For each English word, pronounced as English, mark every candidate Khmer spelling as "
                         "acceptable, unacceptable or uncertain (several may be acceptable). Comments optional. "
                         "Candidate order is random; their origin is hidden.",
         "judgment_values": ["acceptable", "unacceptable", "uncertain"], "rows": rows}, ensure_ascii=False, indent=1)
        + "\n", encoding="utf-8")
    (REVIEW / "origin_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    summary = {"words": len(chosen), "candidates": len(rows),
               "per_stratum": {k: sum(s == k for s, _ in chosen) for k in
                               ("gp_only", "controls_only", "all_cover", "none_cover", "mixed", "g_p_top1_agree")},
               "stratum_sizes": {k: len(v) for k, v in strata_.items()},
               "files": {p.name: sha(p) for p in sorted(REVIEW.glob("*.json*")) + [REVIEW / "review_queue.tsv"]},
               "location": str(REVIEW.relative_to(ROOT)) + " (git-ignored; local only; not published)"}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "review_queue_manifest.json").write_text(json.dumps(summary, indent=1) + "\n")
    mark("queue", [REPORTS / "review_queue_manifest.json", REVIEW / "review_queue.tsv", REVIEW / "origin_key.json"])
    return summary


# ------------------------------------------------------------------ command line
STEPS = {"register": register, "predict": predict, "analyse": analyse, "cost": cost, "queue": queue}


def main() -> int:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=[*STEPS, "run", "probe"])
    ap.add_argument("--models")
    ap.add_argument("--width", type=int)
    ap.add_argument("--force", action="store_true", help="rerun a step even if its outputs are recorded")
    a = ap.parse_args()
    if a.step == "probe":
        print(json.dumps(probe(a.models.split(","), a.width)))
        return 0
    todo = ["predict", "analyse", "cost", "queue"] if a.step == "run" else [a.step]
    for s in todo:
        if s != "register" and not (EXP / "registration.json").exists():
            raise SystemExit("register first (and commit the registration)")
        if a.step == "run" and is_done(s) and not a.force:
            print(f"{s}: done, skipped")
            continue
        r = STEPS[s]()
        print(s, json.dumps(r, ensure_ascii=False, default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
