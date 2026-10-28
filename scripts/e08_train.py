"""E08 single exploratory candidate (TRAINING_AMENDMENT_1.md): ByT5-small + the E07 recipe on frozen arm R+S.

Reuses `scripts/e07.py` for model building, the resumable training loop, budget guard, generation and scoring;
only the data, the run folder, the ledger and the experiment name differ.

Steps:
    exposure   re-check the frozen arm and count the draws per provenance → reports/rs_exposure.json
    train      one run (seed 0, ≤ 3,000 updates / 6 h; resumable) → models/e08/rs_s0/
    predict    greedy + beam-4 for dev, eval and the E06 pool, frozen before scoring
    analyse    rs_s0 vs E07 (and context systems), blinded examples, decision → results_rs.json
    cost       load time, batch-1 latency, memory, adapter size

Example:
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e08_train.py train
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e05  # noqa: E402
import e06  # noqa: E402
import e06_byt5 as B  # noqa: E402
import e07  # noqa: E402
import e08  # noqa: E402

ROOT = e06.ROOT
EXP = e08.EXP
REPORTS = e08.REPORTS
PRED = EXP / "predictions"
OUT = ROOT / "models" / "e08" / "rs_s0"
ARM = e08.DATA / "training" / "arm_RS.tsv"
ARM_SHA = "03a4988f84cd"                      # prefix registered in TRAINING_AMENDMENT_1.md
E07_PRED = e07.PRED
RUN = "rs_s0"

# E08 bookkeeping, not E07's: own ledger (1 full run), own experiment name and state folder
e07.LEDGER = EXP / "runs" / "ledger.jsonl"
e07.EXPERIMENT = e08.EXPERIMENT
e07.STATE = ROOT / "state" / "e08"


def recipe() -> dict:
    """The E07 recipe, unchanged (hash-checked against its registration)."""
    return e07.full_cfg()


def arm_rows() -> tuple[list[tuple[str, str]], list[dict]]:
    """(optimisation pairs in file order, all rows) of the frozen arm; refuses a changed file."""
    if not e06.sha(ARM).startswith(ARM_SHA):
        raise RuntimeError("arm_RS.tsv differs from the frozen hash")
    lines = ARM.read_text(encoding="utf-8").splitlines()
    rows = [dict(zip(lines[0].split("\t"), ln.split("\t"))) for ln in lines[1:]]
    opt = [(r["word"], r["khmer"]) for r in rows if r["part"] == "train"]
    return opt, rows


def draw_counts(n_rows: int, cfg: dict) -> Counter:
    """Row draws of `e07.fit`'s deterministic order (row-level; epochs = seeded permutations) over the run."""
    order = lambda ep: random.Random(f"{cfg['seed']}:order:{ep}").sample(range(n_rows), n_rows)  # noqa: E731
    c, ep, pos, perm = Counter(), 0, 0, order(0)
    for _ in range(cfg["max_updates"] * cfg["grad_accumulation"]):
        if pos + cfg["batch"] > n_rows:
            ep, pos, perm = ep + 1, 0, order(ep + 1)
        c.update(perm[pos:pos + cfg["batch"]])
        pos += cfg["batch"]
    return c


def exposure() -> dict:
    cfg = recipe()
    opt, rows = arm_rows()
    e07.check_disjoint(opt)
    prot = e08.protected()
    prov = [r["provenance"] for r in rows if r["part"] == "train"]
    bad = [w for (w, _), q in zip(opt, prov) if q != "control" and e08.is_protected(w, prot)]
    ctrl = sorted({w for (w, _), q in zip(opt, prov) if q == "control" and e08.is_protected(w, prot)})
    rej = Counter(B.fits(w, k) for w, k in opt if B.fits(w, k))
    pairs = Counter(opt)
    draws = draw_counts(len(opt), cfg)
    by_prov = Counter()
    for i, n in draws.items():
        by_prov[prov[i]] += n
    tot = sum(by_prov.values())
    words = Counter(w for w, _ in opt)
    out = {"arm_sha256": e06.sha(ARM), "optimisation_rows": len(opt), "rows_by_provenance": dict(Counter(prov)),
           "words": len(words), "words_by_provenance": {p: len({w for (w, _), q in zip(opt, prov) if q == p})
                                                         for p in sorted(set(prov))},
           "synthetic_note": "384 synthetic rows = 193 words (preferred + up to two alternatives), not 384 words",
           "exact_duplicate_pairs": sum(n > 1 for n in pairs.values()), "protected_family_rows_added": len(bad),
           "control_words_related_to_e08_protected_set": {"words": len(ctrl), "examples": ctrl[:10], "note":
               "E03's immutable baseline; E08's protected set is wider than E03's (stem relatedness to E06 pool, "
               "review queues, panels); control rows are kept unchanged, as registered"},
           "rejected_by_byte_limits": dict(rej),
           "alternatives": {"words_with_several_spellings": sum(n > 1 for n in words.values()),
                            "rows_in_those_words": sum(n for n in words.values() if n > 1),
                            "max_spellings_per_word": max(words.values()),
                            "effect": "row-level sampling (E07 convention): a word with k spellings is drawn k times "
                                      "per epoch"},
           "sampling": "row-level, seeded permutation per epoch (e07.fit); no source oversampling",
           "draws_total": tot, "epochs": round(tot / len(opt), 2),
           "draws_by_provenance": {p: {"draws": n, "share": round(n / tot, 4)} for p, n in sorted(by_prov.items())},
           "e07_reference": {"rows": 2255, "epochs": round(cfg["max_updates"] * cfg["batch"] / 2255, 2)}}
    if bad or out["exact_duplicate_pairs"] or rej:
        raise RuntimeError(f"exposure check failed: {bad[:3]}, {out['exact_duplicate_pairs']}, {dict(rej)}")
    return e07.write_json(REPORTS / "rs_exposure.json", out) and out


def train() -> dict:
    if not (REPORTS / "rs_exposure.json").exists():
        exposure()
    cfg = recipe()
    opt, _ = arm_rows()
    dev = e06.primary(e06.items("dev"))
    return e07.fit(RUN, "full", cfg, opt, lambda m: e07.dev_selection(m, dev), set(cfg["eval_at"]), OUT)


def model():
    cfg = recipe()
    done = json.loads((OUT / "done.json").read_text())
    if e06.sha(OUT / "best.pt") != done["best_sha256"]:
        raise RuntimeError("best.pt changed")
    m = e07.build(cfg["adapter"], cfg["base_dropout"], cfg["seed"])
    e07.load_adapter(m, OUT / "best.pt")
    return m, done


def predict() -> dict:
    """Same inputs and decoding as E07's `predict` (greedy top-1; beam 4 top-4, sum log-prob; 121 new tokens)."""
    m, done = model()
    PRED.mkdir(parents=True, exist_ok=True)
    timing = {}
    for name, rows in {k: v for k, v in e06.prediction_sets().items() if k in ("dev", "eval", "pool")}.items():
        p = PRED / f"byt5_{RUN}_{name}.jsonl"
        if p.exists():
            continue
        t = time.perf_counter()
        ok = [(i, w) for i, w in rows if B.fits(w) is None]
        gr = dict(zip([i for i, _ in ok], e07.words_gen(m, [w for _, w in ok])))
        bm = dict(zip([i for i, _ in ok], e07.words_gen(m, [w for _, w in ok], beam=4, ret=4, bs=16)))
        timing[name] = {"words": len(rows), "seconds": round(time.perf_counter() - t, 1)}
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for i, w in rows:
                g, b = gr.get(i), bm.get(i)
                f.write(json.dumps({"id": i, "word": w, "system": f"byt5_{RUN}", "checkpoint_update": done["best"]["update"],
                                    "greedy": g[0][0] if g and g[0][1] == "ok" else None,
                                    "greedy_status": g[0][1] if g else None,
                                    "beam4": e05.dedup([t for t, s, *_ in (b or []) if s == "ok" and t]),
                                    "beam4_raw": [{"text": t, "status": s, "score": sc} for t, s, sc, _ in (b or [])]},
                                   ensure_ascii=False) + "\n")
        os.replace(tmp, p)
    hashes = {p.name: e06.sha(p) for p in sorted(PRED.glob(f"byt5_{RUN}_*.jsonl"))}
    e07.write_json(REPORTS / "rs_predictions_frozen.json", {"frozen": e06.now(), "sha256": hashes,
                                                            "checkpoint": done["best"], "timing": timing})
    return {"timing": timing, "hashes": hashes}


KH_VOWEL = set(chr(c) for c in range(0x17B6, 0x17C6)) | {"ំ", "ះ", "ៈ"}


def diff_kind(a: str, b: str) -> str:
    """How two spellings differ: `vowels_only`, `consonants_only` or `both` (code points outside/inside the
    Khmer vowel-sign range; coeng and other signs count as consonant-side).

    Examples:
        >>> diff_kind("កាត", "កេត"), diff_kind("កាត", "ខាត"), diff_kind("កាត", "ខេត")
        ('vowels_only', 'consonants_only', 'both')
    """
    va, vb = [c for c in a if c in KH_VOWEL], [c for c in b if c in KH_VOWEL]
    ca, cb = [c for c in a if c not in KH_VOWEL], [c for c in b if c not in KH_VOWEL]
    v, c = va != vb, ca != cb
    return "both" if v and c else "vowels_only" if v else "consonants_only" if c else "same_letters_reordered"


def analyse() -> dict:
    from khmer_translit import evaluation as ev
    m_done = json.loads((OUT / "done.json").read_text())
    load = lambda p: {r["id"]: r for r in e07.load_jsonl(p)}  # noqa: E731
    new_e, old_e = load(PRED / f"byt5_{RUN}_eval.jsonl"), load(E07_PRED / "byt5_e07_eval.jsonl")
    new_d, old_d = load(PRED / f"byt5_{RUN}_dev.jsonl"), load(E07_PRED / "byt5_e07_dev.jsonl")
    ev_all, prim = e06.items("eval"), e06.primary(e06.items("eval"))
    dev = e06.primary(e06.items("dev"))
    systems = {"rs_s0_beam4": {i: r["beam4"] for i, r in new_e.items()},
               "e07_beam4": {i: r["beam4"] for i, r in old_e.items()},
               "rs_s0_greedy": {i: [r["greedy"]] if r["greedy"] else [] for i, r in new_e.items()},
               "e07_greedy": {i: [r["greedy"]] if r["greedy"] else [] for i, r in old_e.items()},
               "default_e02": e07.default_eval_preds() if (E07_PRED / "default_e02_eval.jsonl").exists() else None,
               **{g: e06.e04_eval(g) for g in ("G_s0", "G_s1", "G_s2")}}
    tab, per = {}, {}
    for s, p in systems.items():
        e05.check_ids({i.id for i in ev_all}, p)
        tab[s], per[s] = e07.system_table(prim, {i.id: p[i.id] for i in prim})
    idx = e05.boot_indices(len({i.family for i in prim}))
    fm = {s: e05.family_means(prim, {i: v["top1"] for i, v in per[s].items()}) for s in per}
    fm |= {f"{s}@4": e05.family_means(prim, {i: v["topk"] for i, v in per[s].items()}) for s in per}
    fc = {s: e05.family_means(prim, {i: -v["cer"] for i, v in per[s].items()}) for s in per}
    gs = ["G_s0", "G_s1", "G_s2"]
    con = {"rs_minus_e07_top1": e05.contrast(fm, ["rs_s0_beam4"], ["e07_beam4"], idx),
           "rs_minus_e07_greedy_top1": e05.contrast(fm, ["rs_s0_greedy"], ["e07_greedy"], idx),
           "rs_minus_e07_top4_oracle": e05.contrast(fm, ["rs_s0_beam4@4"], ["e07_beam4@4"], idx),
           "rs_minus_e07_cer_reduction_pp": e05.contrast(fc, ["rs_s0_beam4"], ["e07_beam4"], idx),
           "rs_minus_G_mean_top1": e05.contrast(fm, ["rs_s0_beam4"], gs, idx),
           "rs_minus_default_top1": e05.contrast(fm, ["rs_s0_beam4"], ["default_e02"], idx)}
    # dev: greedy, the selection metric (descriptive: both checkpoints were selected on it)
    def dev_score(pred):
        hit = {i.id: int(e07.nfc_eq(pred[i.id]["greedy"], i.preferred) or
                         any(e07.nfc_eq(pred[i.id]["greedy"], a) for a in i.accepted)) for i in dev}
        return {"top1": round(e06.macro(dev, hit), 4), "cer": round(mean(e07.cer_of(pred[i.id]["greedy"], i.accepted)
                                                                       for i in dev), 4)}
    dev_tab = {"rs_s0": dev_score(new_d), "e07": dev_score(old_d)}
    # strata
    e02_train = {ln.split("\t")[0].lower() for ln in
                 e06.guard(ROOT / "data/e02/train.tsv").read_text(encoding="utf-8").splitlines()[1:]}
    arm_words = {w for w, _ in arm_rows()[0]}
    strata = {}
    for name, key in (("length", lambda i: e07.len_bin(i.word)), ("source", lambda i: "+".join(sorted(i.sources))),
                      ("in_default_e02_training", lambda i: str(i.word.lower() in e02_train))):
        for v in sorted({key(i) for i in prim}):
            sub = [i for i in prim if key(i) == v]
            strata[f"{name}={v}"] = {"items": len(sub), **{s: round(e06.macro(sub, {i.id: per[s][i.id]["top1"] for i in sub}), 4)
                                                          for s in ("rs_s0_beam4", "e07_beam4")}}
    amb = [i for i in ev_all if "pronunciation_ambiguous" in i.flags]
    strata["ambiguous_track (excluded from primary)"] = {"items": len(amb), **{
        s: round(e07.system_table(amb, {i.id: systems[s][i.id] for i in amb})[0]["top1_macro"], 4)
        for s in ("rs_s0_beam4", "e07_beam4")}}
    # flips, difference kinds and a balanced blinded sample
    flips = Counter()
    kinds = Counter()
    groups = {"rs_only_correct": [], "e07_only_correct": [], "both_wrong_different": []}
    for i in prim:
        a, b = per["rs_s0_beam4"][i.id], per["e07_beam4"][i.id]
        pa, pb = a["prediction"], b["prediction"]
        if pa != pb:
            kinds[diff_kind(pa, pb)] += 1
        key = ("rs_only_correct" if a["top1"] and not b["top1"] else "e07_only_correct" if b["top1"] and not a["top1"]
               else "both_wrong_different" if not a["top1"] and not b["top1"] and pa != pb else None)
        flips[key or ("both_correct" if a["top1"] else "both_wrong_same")] += 1
        if key:
            groups[key].append({"word": i.word, "reference": i.accepted, "rs": pa, "e07": pb, "kind": diff_kind(pa, pb)})
    rng = random.Random("e08-blind")
    blind, keymap = [], {}
    for g, rows in groups.items():
        for r in sorted(rows, key=lambda r: e06.order_key("e08-blind-" + g, r["word"]))[:10]:
            swap = rng.random() < 0.5
            item = f"b{len(blind) + 1:02d}"
            blind.append({"item": item, "english": r["word"], "reference_mined": r["reference"][0],
                          "A": r["e07"] if swap else r["rs"], "B": r["rs"] if swap else r["e07"], "difference": r["kind"]})
            keymap[item] = {"A": "e07" if swap else "rs_s0", "B": "rs_s0" if swap else "e07", "group": g}
    e07.write_json(REPORTS / "rs_blinded_examples.json", {"note": "balanced: 10 per group (rs-only correct, "
                   "e07-only correct, both wrong and different); A/B order random; key in rs_blinded_key.json; "
                   "references are mined and unreviewed", "items": blind})
    e07.write_json(REPORTS / "rs_blinded_key.json", keymap)
    # validity, repetition
    status = Counter(r["greedy_status"] for r in new_e.values())
    beam_status = Counter(x["status"] for r in new_e.values() for x in r["beam4_raw"])
    # decision (TRAINING_AMENDMENT_1.md)
    d = con["rs_minus_e07_top1"]
    promising = (d["diff_pp"] >= 3.0 and d["ci95_pp"][0] > 0 and dev_tab["rs_s0"]["top1"] >= dev_tab["e07"]["top1"]
                 and tab["rs_s0_beam4"]["mean_cer"] <= tab["e07_beam4"]["mean_cer"])
    regression = d["ci95_pp"][1] < 0 or d["diff_pp"] <= -3.0
    decision = "promising_combined_candidate" if promising else "regression" if regression else "small_or_uncertain_change"
    res = {"label": "exploratory reference agreement on the historical E03 eval set (examined repeatedly, mined, "
                    "unreviewed); not phonetic accuracy. Synthetic labels are never references.",
           "checkpoint": m_done["best"], "updates": m_done["updates"], "stop": m_done["stop"],
           "training_hours": m_done["elapsed_hours"], "peak_rss_mb": m_done["peak_rss_mb"],
           "dev_greedy": dev_tab, "dev_history": m_done["history"],
           "historical_eval": {"items": len(prim), "families": len({i.family for i in prim}), "systems": tab,
                               "contrasts": con},
           "strata": strata, "flips_vs_e07": dict(flips), "difference_kinds_vs_e07": dict(kinds),
           "validity": {"greedy": dict(status), "beam4_all_candidates": dict(beam_status)},
           "eval_words_in_rs_training": sum(i.word in arm_words for i in prim),
           "decision": decision, "rule": "TRAINING_AMENDMENT_1.md"}
    e07.write_json(EXP / "results_rs.json", res)
    with open(EXP / "results_rs_per_item.jsonl", "w", encoding="utf-8") as f:
        for i in prim:
            f.write(json.dumps({"id": i.id, "word": i.word, "accepted": i.accepted,
                                **{s: per[s][i.id] for s in ("rs_s0_beam4", "e07_beam4")}}, ensure_ascii=False) + "\n")
    return {k: res[k] for k in ("dev_greedy", "decision", "flips_vs_e07", "difference_kinds_vs_e07")} | {
        "contrasts": con, "top1": {s: tab[s]["top1_macro"] for s in tab}, "cer": {s: tab[s].get("mean_cer") for s in tab}}


def cost() -> dict:
    import resource
    t = time.perf_counter()
    m, _ = model()
    load_s = time.perf_counter() - t
    words = [i.word for i in e06.primary(e06.items("dev"))][:40]
    lat = []
    for w in words:
        t = time.perf_counter()
        e07.words_gen(m, [w])
        lat.append(time.perf_counter() - t)
    out = {"load_s": round(load_s, 2), "batch1_greedy_ms_median": round(median(lat) * 1000, 1),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
           "adapter_bytes": (OUT / "best.pt").stat().st_size, "threads": e07.LIMITS["threads"]}
    return e07.write_json(REPORTS / "rs_cost.json", out) and out


STEPS = {"exposure": exposure, "train": train, "predict": predict, "analyse": analyse, "cost": cost}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=list(STEPS))
    r = STEPS[ap.parse_args().step]()
    print(json.dumps(r, ensure_ascii=False, default=str)[:5000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
