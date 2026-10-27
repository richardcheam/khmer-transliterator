#!/usr/bin/env python3
"""Experiment E02: the one final evaluation event, and its report.

    .venv/bin/python scripts/english_phonetic_eval.py final    # once, after selection (the runner calls it)
    .venv/bin/python scripts/english_phonetic_eval.py report   # recompute statistics from the saved predictions

`final` refuses to run twice (`data/e02/results/final.lock`). It predicts, with every completed model of the
queue and the frozen pronunciation provider (first CMUdict reading, `<no_pron>` on a miss; never the reference
phones), on:

- **primary:** `final` items without the `pronunciation_ambiguous` / `possible_translation` flags; raw beam-4
  top-1 multi-reference exact match, macro over lexical families. Tier `attested` only (no item is independently
  reviewed under the E02 policy), so the metric is **reference agreement** and the benchmark is unqualified;
- **final_flagged:** the flagged `final` items (diagnostic);
- **final_proxy:** the LLM-spelling exploratory track (diagnostic, never pooled);
- a **fixed re-ranker** diagnostic (`translit.pick` towards Google-lexicon words, same for every arm);
- **benchmark v1** English panels with the v1 rule (`benchmark.outcomes`), after selection, never pooled.

The supplied-pronunciation track is not run: every item's intended reading is the registered default reading,
so it would repeat the primary track exactly.

Per-item predictions stay in `data/e02/results/` (they contain reference spellings); aggregate statistics go to
`experiments/E02-english-phonetic/results.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from khmer_translit import benchmark, evaluation as ev, lexicon, translit  # noqa: E402
from khmer_translit import phonetic_source as ps  # noqa: E402
from khmer_translit import reference_data as rd  # noqa: E402
from khmer_translit.english import Transliterator  # noqa: E402

DATA = ROOT / "data" / "e02"
RES = DATA / "results"
EXP = ROOT / "experiments" / "E02-english-phonetic"
STATE = ROOT / "state" / "english-phonetic" / "status.json"
RUNS = ROOT / "models" / "e02"
FLAGGED = {"pronunciation_ambiguous", "possible_translation"}
V1_PANELS = ("pick_test", "dev41", "list_test")


def tracks(items: list[rd.ReferenceItem]) -> dict[str, list[rd.ReferenceItem]]:
    """The evaluation tracks (see the module docstring)."""
    fin = [i for i in items if i.split == "final"]
    return {"primary": [i for i in fin if not FLAGGED & set(i.flags)],
            "final_flagged": [i for i in fin if FLAGGED & set(i.flags)],
            "final_proxy": [i for i in items if i.split == "final_proxy"]}


def completed_models(st: dict) -> dict[str, Path]:
    """Completed training jobs of the queue → their `model.npz` (hash-checked against `parity.json`)."""
    out = {}
    for j in st["jobs"]:
        if j["kind"] == "train" and j["state"] == "completed":
            npz = RUNS / j["id"] / "model.npz"
            want = json.loads((RUNS / j["id"] / "parity.json").read_text())["npz_sha256"]
            if ps.sha256_file(npz) != want:
                raise RuntimeError(f"{npz} changed after verification")
            out[j["id"]] = npz
    return out


def final() -> int:
    """Predict once with every completed model on every track; then `report`."""
    RES.mkdir(parents=True, exist_ok=True)
    lock = RES / "final.lock"
    if lock.exists():
        print(f"final evaluation already ran ({lock.read_text().strip()}); use `report`")
        return 1
    st = json.loads(STATE.read_text(encoding="utf-8"))
    if not st.get("selection"):
        raise RuntimeError("selection is not frozen yet")
    models = completed_models(st)
    items = rd.load_reference_items(DATA / "references.jsonl")
    tr = tracks(items)
    known = lexicon.keys()
    v1 = benchmark.load("v1")
    lock.write_text(json.dumps({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "models": {k: ps.sha256_file(v)
                                for k, v in models.items()}}) + "\n")
    preds = {}
    for name, npz in models.items():
        tl = Transliterator.from_pretrained(npz)
        rec = {}
        for tname, its in tr.items():
            res = tl.predict_many([i.word for i in its], top_k=4)
            rec[tname] = {i.id: {"input": i.word, "status": r.status, "candidates": [[c.text, c.log_score] for c in r.candidates],
                                 "pronunciation": (r.pronunciation or {}).get("phones"),
                                 "pron_status": (r.pronunciation or {}).get("status")} for i, r in zip(its, res)}
        rec["v1"] = {}
        for pname in V1_PANELS:
            res = tl.predict_many([w for w, _ in v1[pname]], top_k=4)
            rec["v1"][pname] = [[c.text for c in r.candidates] for r in res]
        preds[name] = rec
    (RES / "predictions.json").write_text(json.dumps(preds, ensure_ascii=False) + "\n", encoding="utf-8")
    return report()


def run_outcomes(rec: dict, its: list[rd.ReferenceItem]) -> dict[str, int]:
    """Top-1 outcomes `{id: 0/1}` of one model on one track."""
    o = ev.item_outcomes(its, {i.id: [c for c, _s in rec[i.id]["candidates"]] for i in its})
    return {k: v["top1"] for k, v in o.items()}


def report() -> int:
    """Statistics from `predictions.json` into `results.json` (and per-item outcomes into `data/e02/results`)."""
    st = json.loads(STATE.read_text(encoding="utf-8"))
    preds = json.loads((RES / "predictions.json").read_text(encoding="utf-8"))
    items = rd.load_reference_items(DATA / "references.jsonl")
    tr = tracks(items)
    known = lexicon.keys()
    v1 = benchmark.load("v1")
    out = {"benchmark": {"qualified": False, "status": "unqualified",
                         "reason": "no item has documented independent acceptance under the E02 phonetic policy; "
                                   "the final items are owner-verified loanword-list spellings (tier attested)",
                         "metric_name": "reference_agreement",
                         "sizes": {t: {"items": len(v), "families": len({i.family for i in v})} for t, v in tr.items()}},
           "selection": st["selection"], "controls": ev.measurement_controls(tr["primary"]), "runs": {}}
    per_item = {}
    for name, rec in preds.items():
        r = {}
        for tname, its in tr.items():
            sc = ev.score_predictions(its, {i.id: [c for c, _s in rec[tname][i.id]["candidates"]] for i in its})
            rr = {i.id: [translit.pick([(c, s) for c, s in rec[tname][i.id]["candidates"]], known)]
                  if rec[tname][i.id]["candidates"] else [] for i in its}
            sr = ev.score_predictions(its, rr)
            strata = {}
            for sname, pred in (("dictionary_hit", lambda i: "dictionary_miss" not in i.flags),
                                ("dictionary_miss", lambda i: "dictionary_miss" in i.flags),
                                ("candidate_anchored", lambda i: "candidate_anchored" in i.flags),
                                ("not_anchored", lambda i: "candidate_anchored" not in i.flags),
                                ("short_le4", lambda i: len(i.word) <= 4), ("long_ge9", lambda i: len(i.word) >= 9)):
                sub = [i for i in its if pred(i)]
                if sub:
                    strata[sname] = {"n": len(sub), "macro": ev.macro(sub, {i.id: sc["items"][i.id]["top1"] for i in sub})}
            r[tname] = {"primary_macro": sc["primary_macro"], "item_accuracy": sc["item_accuracy"],
                        "correct_items": sc["correct_items"], "n_items": sc["n_items"], "topk_macro": sc["topk_macro"],
                        "mean_cer": sc["mean_cer"], "abstained": sc["abstained"],
                        "reranked_macro": sr["primary_macro"], "strata": strata}
            per_item.setdefault(name, {})[tname] = sc["items"]
        r["v1"] = {}
        for pname in V1_PANELS:
            raw = [c[0] if c else "" for c in rec["v1"][pname]]
            o = benchmark.outcomes(raw, v1[pname])
            r["v1"][pname] = {"raw_correct": sum(o), "n": len(o), "raw": sum(o) / len(o)}
        out["runs"][name] = r
    def runtime_ok(run: str) -> bool | None:
        """The measured guards of one run (`reports/runtime.json`, Task 5); None if it was not measured."""
        f = EXP / "reports" / "runtime.json"
        rows = json.loads(f.read_text(encoding="utf-8"))["models"] if f.exists() else []
        return next((r["guards_ok"] for r in rows if Path(r["model"]).parent.name == run), None)

    # paired comparisons on the primary track
    winner = st["selection"]["winner"]
    seeds = lambda arm: [n for n in (f"{arm}_s0", f"{arm}_s1", f"{arm}_s2") if n in preds]
    prim = tr["primary"]
    comps = {}
    if winner != "letters" and len(seeds(winner)) == 3 and len(seeds("letters")) == 3:
        c = ev.compare_paired_runs(prim, [run_outcomes(preds[n]["primary"], prim) for n in seeds("letters")],
                                   [run_outcomes(preds[n]["primary"], prim) for n in seeds(winner)])
        c["evidence"] = ev.evidence_fields(c["ci95_pp"])
        c["adoption"] = ev.adoption(False, c, out["controls"]["ok"], runtime_ok=runtime_ok(f"{winner}_s0"))
        c["kind"] = f"confirmatory: {winner} vs letters, seeds 0-2"
        comps["confirmatory"] = c
        proxy = tr["final_proxy"]
        cp = ev.compare_paired_runs(proxy, [run_outcomes(preds[n]["final_proxy"], proxy) for n in seeds("letters")],
                                    [run_outcomes(preds[n]["final_proxy"], proxy) for n in seeds(winner)])
        cp["evidence"] = ev.evidence_fields(cp["ci95_pp"])
        cp["kind"] = "exploratory proxy track (LLM spellings), same paired seeds"
        comps["proxy_track"] = cp
    for a, b in (("letters", "phones"), ("letters", "phones_p2g"), ("phones", "phones_p2g")):
        if f"{a}_s0" in preds and f"{b}_s0" in preds:
            c = ev.compare_paired_runs(prim, [run_outcomes(preds[f"{a}_s0"]["primary"], prim)],
                                       [run_outcomes(preds[f"{b}_s0"]["primary"], prim)])
            c["kind"] = f"exploratory one-seed pilot: {b} vs {a}"
            comps[f"pilot_{b}_vs_{a}"] = c
    out["comparisons"] = comps
    conf = comps.get("confirmatory")
    out["evidence_status"] = ev.summary_status(False, conf["evidence"]) if conf else "inconclusive"
    out["experiment_status"] = ("complete_paired" if conf else
                                "pilot_only" if winner == "letters" else "incomplete")
    out["adoption"] = "letters_baseline_retained"
    out["default_model"] = f"letters_s0"
    out["power"] = json.loads((EXP / "reports" / "power.json").read_text()) if (EXP / "reports" / "power.json").exists() else None
    (EXP / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (RES / "per_item.json").write_text(json.dumps(per_item, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("evidence_status", "experiment_status", "adoption", "default_model")}))
    return 0


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["final", "report"])
    return final() if ap.parse_args().mode == "final" else report()


if __name__ == "__main__":
    sys.exit(main())
