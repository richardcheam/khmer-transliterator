#!/usr/bin/env python3
"""Experiment E03: the frozen evaluation (quality proxies and pronunciation mechanism, kept apart).

    .venv/bin/python scripts/e03_eval.py freeze     # before training: samples, settings, metrics -> frozen_eval.json
    .venv/bin/python scripts/e03_eval.py evaluate   # after the 6 runs: results.json + per-item files

Item sets (all read through `guarded`; E02's final references, results and the reserve are refused):
- `eval`: `data/e03/eval_references.jsonl`, the new mechanism-evaluation partition (DACS pairs; unqualified);
- `dev`: `data/e03/dev_references.jsonl` (checkpoint selection: secondary only);
- `e02_dev`: `data/e02/dev_references.jsonl`, the words examined in the E02 diagnostics (development material).

**Quality proxies** (registered primary: raw beam-4 top-1 macro over lexical families on the primary `eval` items,
phones − letters, 3 paired seeds, 10,000 family-bootstrap resamples, seed 20261027): reference agreement, top-4,
character error rate. No re-ranker: every E03 reference is a Google-lexicon word, so re-ranking towards the
lexicon would leak the answer.

**Mechanism** (the E02-diagnostics protocol, `e02_diagnostics.build_sample`, frozen here): letters fixed; default
phones, `<no_pron>`, a length-matched donor's phones, the word's other CMUdict readings. Per row: top-1 change,
top-4 overlap, rank of the default top-1, teacher-forced length-normalised log-probability change, consonant
movement towards the given phones (frozen reader), change type (consonant-preserving = vowel/diacritic change) and,
where both outputs are lexicon words, syllable-count movement towards the given phones' vowel count. Letters
models are the negative control. Sensitivity is not correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from khmer_translit import evaluation as ev  # noqa: E402
from khmer_translit import phonetic_source as ps  # noqa: E402
from khmer_translit import reference_data as rd  # noqa: E402

EXP = ROOT / "experiments" / "E03-assessable-phones"
RES = EXP / "results"
RUNS = ROOT / "models" / "e03"
SETS = {"eval": ROOT / "data" / "e03" / "eval_references.jsonl",
        "dev": ROOT / "data" / "e03" / "dev_references.jsonl",
        "e02_dev": ROOT / "data" / "e02" / "dev_references.jsonl"}
MODELS = [f"{arm}_s{s}" for s in (0, 1, 2) for arm in ("letters", "phones")]
PRIMARY_EXCLUDE = {"pronunciation_ambiguous", "possible_translation"}
SETTINGS = {"beam": 4, "top_k": 4, "seed": 20261027, "n_boot": 10000, "threshold_pp": 5.0,
            "conditions": ["default", "no_pron", "donor", "alternative"], "mechanism_sets": ["eval", "e02_dev"],
            "primary_quality": "raw beam-4 top-1 macro over families, primary eval items, phones - letters, paired seeds",
            "primary_mechanism": "donor condition: share of items whose top-1 consonants move closer to the donor's "
                                 "phones minus share moving farther (frozen reader), phones arm, mean over seeds",
            "selection_rule": "per run: best epoch by greedy dev macro (E03 dev, primary-eligible), then lower "
                              "train-val loss, then earlier epoch (E02 trainer, unchanged)"}


def load(name: str) -> list[rd.ReferenceItem]:
    """One item set, through the diagnostics guard."""
    return rd.load_reference_items(DG.guarded(SETS[name]))


def primary(items: list[rd.ReferenceItem]) -> list[rd.ReferenceItem]:
    """Items without an ambiguity or translation flag (E02's primary rule)."""
    return [i for i in items if not PRIMARY_EXCLUDE & set(i.flags)]


def freeze() -> dict:
    """Write `frozen_eval.json` (+ sha256): settings, item-set hashes and mechanism samples, before training."""
    cmu, cmu_sha = ps.cmudict()
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "settings": SETTINGS, "cmudict_sha256": cmu_sha,
           "sets": {k: {"path": str(p.relative_to(ROOT)), "sha256": DG.sha(p), "items": len(load(k)),
                        "primary_items": len(primary(load(k)))} for k, p in SETS.items()},
           "models": MODELS, "samples": {s: DG.build_sample(load(s), cmu) for s in SETTINGS["mechanism_sets"]}}
    body = json.dumps(out, ensure_ascii=False, indent=1) + "\n"
    (EXP / "frozen_eval.json").write_text(body, encoding="utf-8")
    (EXP / "frozen_eval.sha256").write_text(hashlib.sha256(body.encode()).hexdigest() + "\n", encoding="utf-8")
    return {k: v for k, v in out.items() if k != "samples"} | {
        "sample_items": {s: len(v) for s, v in out["samples"].items()}}


def frozen() -> dict:
    """The frozen evaluation, refusing a changed file."""
    body = (EXP / "frozen_eval.json").read_text(encoding="utf-8")
    if hashlib.sha256(body.encode()).hexdigest() != (EXP / "frozen_eval.sha256").read_text().strip():
        raise RuntimeError("frozen_eval.json changed after freezing")
    return json.loads(body)


def family_ci(items, per_item: list[dict[str, float]], seed: int, n_boot: int) -> tuple[float, float]:
    """95% family-bootstrap interval (×100) of a per-item quantity averaged over seeds, then over families."""
    fam = defaultdict(list)
    for it in items:
        fam[it.family].append(mean(d[it.id] for d in per_item))
    vals = [mean(v) for v in fam.values()]
    return ev.bootstrap_ci(vals, n_boot, seed)


def quality(fz: dict, preds: dict) -> dict:
    """Per-run scores on every set, paired comparisons, CER differences."""
    st = fz["settings"]
    out = {"runs": {}, "comparisons": {}}
    for s in SETS:
        items = load(s)
        tracks = {"primary": primary(items), "all": items,
                  "flagged": [i for i in items if PRIMARY_EXCLUDE & set(i.flags)]}
        for m in MODELS:
            for t, its in tracks.items():
                if its:
                    sc = ev.score_predictions(its, {i.id: preds[m][s][i.id] for i in its}, st["top_k"])
                    out["runs"].setdefault(m, {}).setdefault(s, {})[t] = {k: v for k, v in sc.items() if k != "items"}
        its = tracks["primary"]
        oc = {m: ev.item_outcomes(its, {i.id: preds[m][s][i.id] for i in its}, st["top_k"]) for m in MODELS}
        L = [{i: v["top1"] for i, v in oc[f"letters_s{k}"].items()} for k in range(3)]
        P = [{i: v["top1"] for i, v in oc[f"phones_s{k}"].items()} for k in range(3)]
        c = ev.compare_paired_runs(its, L, P, st["n_boot"], st["seed"])
        c["evidence"] = ev.evidence_fields(c["ci95_pp"], st["threshold_pp"])
        dc = [{i: oc[f"phones_s{k}"][i]["cer"] - oc[f"letters_s{k}"][i]["cer"] for i in oc["letters_s0"]}
              for k in range(3)]
        c["cer_diff_pp_mean"] = mean(mean(d.values()) for d in dc) * 100
        c["cer_diff_pp_ci95"] = list(family_ci(its, dc, st["seed"], st["n_boot"]))
        c["role"] = {"eval": "registered primary (quality proxy)", "dev": "secondary (used for checkpoint selection)",
                     "e02_dev": "secondary (E02 development material, owner-list references)"}[s]
        out["comparisons"][s] = c
    return out


def mechanism(fz: dict, tls: dict) -> tuple[dict, list[dict]]:
    """Run the frozen sensitivity samples through every model; aggregates and per-item rows."""
    km_lex, _ = DG.load_lexicon()
    st = fz["settings"]
    rows, checks = [], {}
    for m, tl in tls.items():
        enc_ok, gap, control_same = True, 0.0, True
        for sname, sample in fz["samples"].items():
            for it in sample:
                conds = [c for c in it["conditions"] if "skipped" not in c]
                ids = [DG.condition_ids(tl, it["word"], c) for c in conds]
                enc_ok &= all(DG.check_encoding(tl, it["word"], c, x) for c, x in zip(conds, ids))
                beams = tl.model.beam_search(*tl.model.encode_ids(ids), st["beam"])
                if tl.mode == "letters":
                    control_same &= all(x == ids[0] for x in ids) and all(b == beams[0] for b in beams)
                ref, ref_score = beams[0][0]
                base_tf, ntok = DG.score_text(tl, ids[0], ref)
                gap = max(gap, abs(base_tf - ref_score))
                base_top = [t for t, _ in beams[0]][:st["top_k"]]
                for c, x, nb in zip(conds, ids, beams):
                    top = [t for t, _ in nb][:st["top_k"]]
                    tf, _ = DG.score_text(tl, x, ref)
                    r = {"model": m, "arm": m.split("_")[0], "seed": int(m[-1]), "set": sname, "id": it["id"],
                         "word": it["word"], "condition": c["condition"], "phones": c.get("phones"),
                         "donor_id": c.get("donor_id"), "stress_only": c.get("stress_only"), "source_ids": x,
                         "top1": top[0], "topk": top, "top1_changed": top[0] != base_top[0],
                         "topk_overlap": len(set(top) & set(base_top)) / max(1, len(set(top) | set(base_top))),
                         "default_top1_rank": top.index(ref) + 1 if ref in top else None,
                         "delta_norm_logprob": (tf - base_tf) / ntok}
                    r.update(movement(r, base_top[0], it, c, km_lex))
                    rows.append(r)
        checks[m] = {"encoding_as_intended": enc_ok, "teacher_forced_vs_beam_max_gap": gap,
                     **({"letters_control_identical": control_same} if tl.mode == "letters" else {})}
    return {"checks": checks, "aggregate": aggregate(rows, fz)}, rows


def movement(r: dict, old: str, it: dict, c: dict, km_lex: dict) -> dict:
    """Consonant and syllable movement of a changed top-1 towards the given phones (frozen reader)."""
    out = {"reader_ok": DG.km_skeleton(r["top1"]) is not None, "output_in_lexicon": r["top1"] in km_lex}
    if not r["top1_changed"] or c["condition"] not in ("donor", "alternative"):
        return out
    new_k, old_k = DG.km_skeleton(r["top1"]), DG.km_skeleton(old)
    if new_k is None or old_k is None:
        out["consonant_move"] = "unreadable"
        return out
    given = DG.en_skeleton(c["phones"].split())
    dn, do = DG.align(given, new_k)[0], DG.align(given, old_k)[0]
    out["consonant_move"] = "closer" if dn < do else "farther" if dn > do else "same"
    out["change_type"] = ("consonant_preserving" if DG.align(old_k, new_k, adapt_labels=False)[0] == 0
                          else "consonant_changing")
    if r["top1"] in km_lex and old in km_lex:
        nv = sum(p.rstrip("012") in {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY",
                                     "UH", "UW"} for p in c["phones"].split())
        syl = lambda w: min(abs(len([s for s in g.split(".") if s.strip()]) - nv) for g in km_lex[w])
        out["syllable_move"] = "closer" if syl(r["top1"]) < syl(old) else "farther" if syl(r["top1"]) > syl(old) \
            else "same"
    return out


def aggregate(rows: list[dict], fz: dict) -> dict:
    """Per model × set × condition counts, per-arm means, and family-bootstrap intervals for the phones arm."""
    st = fz["settings"]
    g = defaultdict(list)
    for r in rows:
        cond = r["condition"] if r["condition"] != "alternative" else (
            "alternative_stress_only" if r["stress_only"] else "alternative_segmental")
        g[(r["model"], r["set"], cond)].append(r)
    per = {}
    for (m, s, c), rs in sorted(g.items()):
        ch = [r for r in rs if r["top1_changed"]]
        mv = Counter(r.get("consonant_move") for r in ch)
        per.setdefault(m, {}).setdefault(s, {})[c] = {
            "n": len(rs), "top1_changed": len(ch), "top1_change_rate": len(ch) / len(rs),
            "mean_topk_overlap": mean(r["topk_overlap"] for r in rs),
            "default_top1_still_in_topk": sum(r["default_top1_rank"] is not None for r in rs),
            "mean_delta_norm_logprob": mean(r["delta_norm_logprob"] for r in rs),
            "consonant_closer": mv["closer"], "consonant_farther": mv["farther"], "consonant_same": mv["same"],
            "consonant_unreadable": mv["unreadable"],
            "net_consonant_steering_rate": (mv["closer"] - mv["farther"]) / len(rs),
            "consonant_preserving_changes": sum(r.get("change_type") == "consonant_preserving" for r in ch),
            "syllable_closer": sum(r.get("syllable_move") == "closer" for r in ch),
            "syllable_farther": sum(r.get("syllable_move") == "farther" for r in ch),
            "syllable_assessable": sum("syllable_move" in r for r in ch),
            "reader_coverage": sum(r["reader_ok"] for r in rs) / len(rs),
            "output_in_lexicon": sum(r["output_in_lexicon"] for r in rs) / len(rs)}
    intervals = {}
    for s in st["mechanism_sets"]:
        items = {it.id: it for it in load(s)}
        for c in ("donor", "no_pron", "alternative"):
            for arm in ("phones", "letters"):
                vals_change, vals_net = [], []
                for k in range(3):
                    rs = [r for r in rows if r["model"] == f"{arm}_s{k}" and r["set"] == s and r["condition"] == c]
                    vals_change.append({r["id"]: float(r["top1_changed"]) for r in rs})
                    vals_net.append({r["id"]: {"closer": 1.0, "farther": -1.0}.get(r.get("consonant_move"), 0.0)
                                     for r in rs})
                ids = [i for i in vals_change[0] if all(i in v for v in vals_change)]
                if not ids:
                    continue
                its = [items[i] for i in ids]
                intervals[f"{s}|{c}|{arm}"] = {
                    "items": len(ids), "families": len({it.family for it in its}),
                    "change_rate_pct": mean(mean(v[i] for v in vals_change) for i in ids) * 100,
                    "change_rate_ci95_pct": list(family_ci(its, vals_change, st["seed"], st["n_boot"])),
                    "net_steering_pct": mean(mean(v[i] for v in vals_net) for i in ids) * 100,
                    "net_steering_ci95_pct": list(family_ci(its, vals_net, st["seed"], st["n_boot"])),
                    "changed_in_all_3_seeds": sum(all(v[i] for v in vals_change) for i in ids)}
    return {"per_model": per, "phones_vs_letters_intervals": intervals}


def evaluate() -> dict:
    """Predict, score and run the mechanism; writes results.json and per-item files."""
    from khmer_translit.english import Transliterator
    fz = frozen()
    t0 = time.time()
    for k, v in fz["sets"].items():
        if DG.sha(SETS[k]) != v["sha256"]:
            raise RuntimeError(f"{k} changed since freezing")
    tls, shas = {}, {}
    for m in MODELS:
        path = DG.guarded(RUNS / m / "model.npz")
        shas[m] = DG.sha(path)
        tls[m] = Transliterator.from_pretrained(path)
    preds: dict = {}
    for m, tl in tls.items():
        for s in SETS:
            items = load(s)
            res = tl.predict_many([i.word for i in items], top_k=fz["settings"]["top_k"], beam=fz["settings"]["beam"])
            preds.setdefault(m, {})[s] = {i.id: [c.text for c in r.candidates] for i, r in zip(items, res)}
    q = quality(fz, preds)
    mech, rows = mechanism(fz, tls)
    controls = ev.measurement_controls(load("eval"))
    out = {"experiment": "E03-assessable-phones", "frozen_eval_sha256": (EXP / "frozen_eval.sha256").read_text().strip(),
           "models": shas, "controls": controls, "quality": q, "mechanism": mech,
           "elapsed_s": round(time.time() - t0, 1),
           "status": {"benchmark": "unqualified (screened mined pairs; no independent review)",
                      "product_default": "unchanged (E02 letters_s0); no promotion from a mechanism study"}}
    RES.mkdir(parents=True, exist_ok=True)
    (EXP / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with open(RES / "quality_per_item.jsonl", "w", encoding="utf-8") as f:
        for m in MODELS:
            for s in SETS:
                for i, cands in preds[m][s].items():
                    f.write(json.dumps({"model": m, "set": s, "id": i, "candidates": cands}, ensure_ascii=False) + "\n")
    with open(RES / "mechanism_per_item.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {k: out[k] for k in ("frozen_eval_sha256", "elapsed_s", "controls")}


def main() -> int:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["freeze", "evaluate"])
    a = ap.parse_args()
    print(json.dumps({"freeze": freeze, "evaluate": evaluate}[a.step](), ensure_ascii=False, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
