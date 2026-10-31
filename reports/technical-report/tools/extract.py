#!/usr/bin/env python3
"""Build the technical report's results extract from the project's committed result files.

Every number the report uses is read from a JSON file committed in this repository, at a pinned git revision,
with `git show <rev>:<path>`. The script writes:

- `data/results_extract.json`: each value with its source file, JSON field, revision and definition;
- `generated/values.tex`: one LaTeX macro per value (`\\val{key}`), so the text and tables use the extract;
- `generated/*.csv`: the data behind the figures.

It also re-derives a few quantities (for example the candidate-coverage differences) and asserts the
relationships the report states. No model is run and no judgment, reference or training file is read.

Run from the repository root (needs only Python 3.10+ and git):

    python3 reports/technical-report/tools/extract.py
"""
from __future__ import annotations

import csv
import json
import subprocess
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
REV = "d7e7d80c4187b363219343d9e088a01ad6ae56fa"      # = git tag archive/research-e02-e13 (before the cleanup)
# the small model's install check was run three times; each run is its own commit of the same file
E02_TIMING_REVS = ("3ea7019", "03452be", "1422153")

E02 = "experiments/E02-english-phonetic/results.json"
E02D = "experiments/E02-diagnostics-2026-10-27/sensitivity/summary.json"
E02R = "experiments/E02-release/reports/install_verify.json"
E03 = "experiments/E03-assessable-phones/results.json"
E04 = "experiments/E04-pronunciation-pathways/results.json"
E05 = "experiments/E05-candidate-diversity/results.json"
E07 = "experiments/E07-byt5-diagnosis/results.json"
E08 = "experiments/E08-data-expansion/results_rs.json"
E09 = "experiments/E09-byt5-decoding/results.json"
E11 = "experiments/E11-real-data-byt5/results.json"
E11R = "experiments/E11-real-data-byt5/reports/review_scores.json"
E11V = "experiments/E11-real-data-byt5/release/verify.json"
E11I = "experiments/E11-real-data-byt5/release/install_check.json"
E11C = "experiments/E11-real-data-byt5/coverage/results.json"
E12 = "experiments/E12-longer-schedule/results.json"
E12T = "experiments/E12-longer-schedule/reports/test_scores.json"
E13 = "experiments/E13-pronunciation-guided/results.json"
E13T = "experiments/E13-pronunciation-guided/reports/test_scores.json"
E13M = "experiments/E13-pronunciation-guided/reports/mechanism.json"

_cache: dict[tuple[str, str], object] = {}


def load(path: str, rev: str = REV):
    """The JSON file `path` as committed at `rev`."""
    if (rev, path) not in _cache:
        txt = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout
        _cache[(rev, path)] = json.loads(txt)
    return _cache[(rev, path)]


def field(obj, dotted: str):
    """Follow a dotted path; integer parts index lists. Keys may contain spaces but not dots.

    >>> field({"a": [{"b": 3}]}, "a.0.b")
    3
    """
    for part in dotted.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


VALUES: dict[str, dict] = {}


def put(key: str, value, fmt: str, *, src: str | None = None, path: str | None = None, rev: str = REV,
        meaning: str, n: str = "", derived: str = "") -> None:
    """Record one value; `fmt` renders it for LaTeX ('pct' share→%, 'pp' already points, 'int', 'f3', 'f2', 'raw')."""
    VALUES[key] = {"value": value, "fmt": fmt, "source": src, "field": path, "revision": rev if src else None,
                   "meaning": meaning, "denominator": n, "derived_from": derived}


def get(key: str, src: str, path: str, fmt: str, meaning: str, n: str = "", rev: str = REV) -> object:
    v = field(load(src, rev), path)
    put(key, v, fmt, src=src, path=path, rev=rev, meaning=meaning, n=n)
    return v


def ci(key: str, src: str, path: str, meaning: str, n: str = "") -> None:
    """A difference in points and its 95% interval (`<path>.diff_pp` / `mean_gain_pp`, `<path>.ci95_pp`)."""
    o = field(load(src), path)
    d = o.get("diff_pp", o.get("mean_gain_pp"))
    put(key, d, "pp", src=src, path=path + (".diff_pp" if "diff_pp" in o else ".mean_gain_pp"), meaning=meaning, n=n)
    put(key + ".ci", o["ci95_pp"], "ci", src=src, path=path + ".ci95_pp", meaning="95% interval of " + meaning, n=n)


def q(x: float, places: int = 1) -> Decimal:
    """Round half up from the decimal text of `x`, as the project's reports do.

    >>> str(q(51.05)), str(q(-1.276)), str(q(1.6460905349794235))
    ('51.1', '-1.3', '1.6')
    """
    return Decimal(repr(x)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def signed(y: float, places: int = 1) -> str:
    d = q(y, places)
    return ("+" if d > 0 else "−" if d < 0 else "") + str(abs(d))


def render(v: dict) -> str:
    x, f = v["value"], v["fmt"]
    if f == "pct":
        return str(q(100 * x if not isinstance(x, int) else x * 100))
    if f == "pp":
        return signed(x)
    if f == "ci":
        return f"[{signed(x[0])}, {signed(x[1])}]"
    if f == "ci2":
        return f"[{signed(x[0], 2)}, {signed(x[1], 1)}]"
    if f == "int":
        return f"{int(x):,}"
    if f in ("f3", "f2", "f1", "f0"):
        return str(q(x, int(f[1])))
    return str(x)


def build() -> None:
    # ---------------------------------------------------------------- small model (E02 letters_s0)
    r = "runs.letters_s0."
    get("s.primary.acc", E02, r + "primary.primary_macro", "pct", "family-macro exact agreement, beam-4 top-1, "
        "owner reference list (primary track)", "165 items / 162 families")
    get("s.primary.n", E02, r + "primary.n_items", "int", "primary items")
    get("s.primary.top4", E02, r + "primary.topk_macro", "pct", "any of 4 beam candidates matches", "165 items")
    get("s.primary.cer", E02, r + "primary.mean_cer", "f3", "mean character error rate vs reference", "165 items")
    get("s.proxy.acc", E02, r + "final_proxy.primary_macro", "pct", "exact agreement, LLM-suggested spot-checked "
        "spellings (proxy track)", "400 items")
    get("s.proxy.top4", E02, r + "final_proxy.topk_macro", "pct", "top-4 agreement, proxy track", "400 items")
    get("s.proxy.n", E02, r + "final_proxy.n_items", "int", "proxy items")
    ci("e02.phones", E02, "comparisons.confirmatory", "phones − letters, family-macro top-1, mean over 3 paired "
       "seeds; paired family bootstrap conditional on the 6 models", "162 families")
    get("e02.families", E02, "comparisons.confirmatory.families", "int", "families in the confirmatory comparison")
    for i, rev in enumerate(E02_TIMING_REVS, 1):
        get(f"s.t{i}.p50", E02R, "performance.single_answer_ms.p50", "f0", "single answer latency p50 (ms)", rev=rev)
        get(f"s.t{i}.p95", E02R, "performance.single_answer_ms.p95", "f0", "single answer latency p95 (ms)", rev=rev)
        get(f"s.t{i}.load", E02R, "performance.load_s", "f2", "cold load (s)", rev=rev)
        get(f"s.t{i}.peak", E02R, "performance.peak_rss_mib", "f0", "peak RSS (MiB)", rev=rev)
        get(f"s.t{i}.b16", E02R, "performance.batch16_ms_per_word", "f0", "batch-16 ms per word", rev=rev)
    # ---------------------------------------------------------------- pronunciation input
    for s in ("s0", "s1", "s2"):
        get(f"sens.{s}.donor", E02D, f"aggregate.phones_{s}.donor.top1_change_rate", "pct",
            "share of top-1 outputs that change when another word's phones are supplied", "94 words")
        get(f"sens.{s}.nopron", E02D, f"aggregate.phones_{s}.no_pron.top1_change_rate", "pct",
            "share of top-1 outputs that change when the phones are removed", "95 words")
    get("sens.letters.donor", E02D, "aggregate.letters_s0.donor.top1_change_rate", "pct",
        "letters-only control: change rate under donor phones (must be 0)", "94 words")
    ci("e03.phones", E03, "quality.comparisons.eval", "phones − letters on consonant-screened pairs, eval, 3 paired "
       "seeds", "E03 eval partition")
    ci("e04.gp", E04, "comparisons.GP_vs_G", "letters+phones − letters, 3 paired seeds")
    ci("e04.p", E04, "comparisons.P_vs_G", "phones-only − letters (one seed for P)")
    ci("e05.gp_g8", E05, "configs.eval_w8.contrasts.summary.G+P_vs_G8", "top-8 oracle: pooled letters+phones "
       "candidates vs letters top-8 (equal budget)")
    ci("e05.gp_gg", E05, "configs.eval_w8.contrasts.summary.G+P_vs_G+G", "top-8 oracle: pooled letters+phones vs "
       "two letters seeds")
    ci("e13.dev", E13, "dev.e13_minus_e11.top1", "pronunciation-guided − base adapter, historical dev top-1",
       "253 items")
    ci("e13.eval", E13, "eval.e13_minus_e11.top1", "pronunciation-guided − base adapter, historical eval top-1",
       "228 items")
    get("e13.mech.missing", E13M, "historical_dev.output_changed_vs_correct.missing", "pct",
        "share of historical-dev outputs that change when the reading is removed", "253 items")
    get("e13.mech.wrong", E13M, "historical_dev.output_changed_vs_correct.mismatched", "pct",
        "share of historical-dev outputs that change under another word's reading", "253 items")
    get("e13.mech.coll.missing", E13M, "collection_dev.output_changed_vs_correct.missing", "pct",
        "same, collection dev", "40 items")
    get("e13.mech.coll.wrong", E13M, "collection_dev.output_changed_vs_correct.mismatched", "pct",
        "same, collection dev", "40 items")
    # ---------------------------------------------------------------- ByT5 adaptation, data, schedule, decoding
    ci("e07.vs_g", E07, "historical_eval.contrasts.byt5_e07_minus_G_mean_top1", "corrected ByT5 adapter (registered "
       "beam-4) − mean of 3 small letters models, historical eval top-1", "228 items")
    VALUES["e07.vs_g.ci"]["fmt"] = "ci2"                       # the recorded lower bound is +0.05
    ci("e08.vs_e07", E08, "historical_eval.contrasts.rs_minus_e07_top1", "adapter on real+synthetic data − E07, "
       "registered beam-4 top-1", "228 items")
    ci("e08.vs_e07.greedy", E08, "historical_eval.contrasts.rs_minus_e07_greedy_top1", "same, greedy top-1", "228 items")
    ci("e09.e08.sel_vs_reg", E09, "contrasts.e08:selected_minus_beam4_reg.top1", "canonical beam (selected) − "
       "registered beam-4 (early_stopping=True), synthetic-data adapter, historical eval top-1", "228 items")
    ci("e09.e08.greedy_vs_reg", E09, "contrasts.e08:greedy_minus_beam4_reg.top1", "greedy − registered beam-4, "
       "synthetic-data adapter, historical eval top-1", "228 items")
    ci("e09.e08.sel_vs_greedy", E09, "contrasts.e08:selected_minus_greedy.top1", "canonical beam − greedy, "
       "synthetic-data adapter, historical eval top-1", "228 items")
    ev = "eval.table."
    for m in ("e07", "e08", "e11"):
        get(f"h.{m}.top1", E11, ev + m + ".top1_macro", "pct", f"{m} greedy family-macro exact agreement, "
            "historical eval", "228 items / 222 families")
        get(f"h.{m}.cer", E11, ev + m + ".mean_cer", "f3", f"{m} mean CER, historical eval", "228 items")
    get("h.e12.top1", E12, "eval.table.e12.top1_macro", "pct", "e12 greedy exact agreement, historical eval", "228 items")
    get("h.e13.top1", E13, "eval.table.e13.top1_macro", "pct", "e13 greedy exact agreement, historical eval", "228 items")
    get("h.items", E11, "eval.items", "int", "historical eval items (primary)")
    get("h.families", E11, "eval.families", "int", "historical eval families")
    get("d.items", E11, "dev.items", "int", "historical dev items (primary)")
    get("d.e11.top1", E11, "dev.table.e11.top1_macro", "pct", "e11 greedy dev exact agreement", "253 items")
    ci("e11.vs_e07", E11, "eval.contrasts.e11_minus_e07.top1", "real-data adapter − E07, greedy top-1, historical "
       "eval", "222 families")
    ci("e11.vs_e08", E11, "eval.contrasts.e11_minus_e08 (historical reference).top1", "real-data adapter − adapter "
       "with synthetic rows, greedy top-1", "222 families")
    ci("e12.vs_e11.hist", E12, "eval.contrasts.e12_minus_e11.top1", "6,000-update − 3,000-update schedule, "
       "historical eval top-1", "222 families")
    get("e11.update", E11, "selected.update", "int", "selected update of the released adapter")
    get("e11.sha", E11, "selected.checkpoint_sha256", "raw", "training checkpoint sha256")
    get("e12.update", E12, "selected.update", "int", "selected update of the long-schedule run")
    get("e13.update", E13, "selected.update", "int", "selected update of the pronunciation-guided run")
    # ---------------------------------------------------------------- owner development review (E11)
    for smp, n in (("representative", 40), ("disagreement", 20)):
        for m in ("e07", "e08", "e11"):
            get(f"rev.{smp}.{m}", E11R, f"acceptable_items.{smp}.{m}.acceptable", "int",
                f"{m} greedy spelling judged acceptable, {smp} dev sample", f"{n} words")
    # ---------------------------------------------------------------- blinded tests
    for t, src, other in (("t1", E12T, "e12"), ("t2", E13T, "e13")):
        for m in ("e11", other):
            for k in ("acceptable", "not_acceptable", "uncertain"):
                get(f"{t}.{m}.{k}", src, f"analysis.counts.{m}.{k}", "int", f"{m} spellings judged {k}",
                    "120 families, one word each")
            get(f"{t}.{m}.share", src, f"analysis.confirmed_acceptable_share.{m}", "pct",
                f"{m} confirmed-acceptable share (uncertain counts as not confirmed)", "120 words")
        ci(f"{t}.diff", src, "analysis.primary", f"{other} − e11 confirmed-acceptable share, paired family "
           "bootstrap (10,000)", "120 families")
        for k in ("both", "a_only", "b_only", "neither"):
            get(f"{t}.paired.{k}", src, f"analysis.paired.{k}", "int", f"paired outcome '{k}' (a = {other})", "120")
        get(f"{t}.p", src, "analysis.exact_test.p_two_sided", "f2", "exact two-sided sign test over discordant words")
    for g in ("ordinary", "irregular", "names"):
        for m in ("e11", "e13"):
            get(f"t2.{g}.{m}", E13T, f"analysis.secondary_groups.{g}.counts.{m}.acceptable", "int",
                f"{m} acceptable in group {g}", "40 words")
    # E12-test groups exist only as a report table (E12 REPORT §5.1); recorded as transcribed values
    for g, a, b in (("ordinary", 27, 27), ("irregular", 19, 14), ("names", 31, 36)):
        put(f"t1.{g}.e12", a, "int", src=None, meaning=f"e12 acceptable in group {g} (E12 REPORT.md §5.1 table)",
            n="40 words", derived="experiments/E12-longer-schedule/REPORT.md §5.1 (no JSON field)")
        put(f"t1.{g}.e11", b, "int", src=None, meaning=f"e11 acceptable in group {g} (E12 REPORT.md §5.1 table)",
            n="40 words", derived="experiments/E12-longer-schedule/REPORT.md §5.1 (no JSON field)")
    # ---------------------------------------------------------------- candidate coverage (E11)
    c = "overall."
    get("cov.words", E11C, c + "words", "int", "irregular development words in the coverage diagnostic")
    get("cov.greedy", E11C, c + "greedy_top1.acceptable", "int", "words whose greedy output is acceptable", "23")
    get("cov.beam1", E11C, c + "beam_top1.acceptable", "int", "words whose canonical beam top-1 is acceptable", "23")
    get("cov.any4", E11C, c + "any_acceptable_in_beam4", "int", "words with ≥1 acceptable among 4 beam candidates", "23")
    get("cov.any5", E11C, c + "any_acceptable_in_greedy_or_beam4", "int", "same, greedy output included", "23")
    get("cov.recov", E11C, c + "recoverable_words", "int", "greedy not acceptable but an acceptable beam candidate "
        "exists", "23")
    get("cov.none", E11C, c + "acceptable_per_word.0", "int", "words with no acceptable candidate", "23")
    get("cov.cands", E11C, c + "distinct_candidates", "int", "distinct candidate spellings judged")
    get("cov.unc", E11C, c + "candidate_judgments.uncertain", "int", "candidates judged uncertain", "93")
    get("cov.coverage", E11C, c + "judgment_coverage", "pct", "share of candidates with a judgment", "93")
    v = {k: VALUES[k]["value"] for k in ("cov.greedy", "cov.beam1", "cov.any4", "cov.any5", "cov.recov", "cov.none",
                                          "cov.words")}
    put("cov.gain_vs_greedy", v["cov.any4"] - v["cov.greedy"], "int", meaning="oracle words added over greedy",
        derived="cov.any4 − cov.greedy (valid because every greedy-acceptable word is in the any-of-4 set: "
                "cov.any5 == cov.any4)")
    put("cov.gain_vs_beam1", v["cov.any4"] - v["cov.beam1"], "int", meaning="oracle words added over beam top-1",
        derived="cov.any4 − cov.beam1 (beam top-1 is one of the 4)")
    assert v["cov.any5"] == v["cov.any4"] and v["cov.any4"] - v["cov.greedy"] == v["cov.recov"]
    assert v["cov.words"] - v["cov.any5"] == v["cov.none"]
    assert (v["cov.words"], v["cov.greedy"], v["cov.beam1"], v["cov.any4"]) == (23, 8, 10, 15)
    # ---------------------------------------------------------------- release verification (adapter)
    for k, p in (("par.dev", "parity.dev.greedy_mismatches_vs_frozen_predictions"),
                 ("par.dev.n", "parity.dev.items"),
                 ("par.eval", "parity.eval.greedy_mismatches_vs_frozen_predictions"),
                 ("par.eval.n", "parity.eval.items"),
                 ("par.top4", "parity.dev_top4.mismatches_vs_research_canonical_beam"),
                 ("b.lat1", "latency_ms.single_word_greedy.median"), ("b.lat4", "latency_ms.single_word_top4.median"),
                 ("b.load", "load_seconds"), ("b.peak", "peak_rss_mb"), ("b.rss", "rss_after_load_mb"),
                 ("b.wps", "parity.dev.greedy_words_per_second")):
        get(k, E11V, p, "int" if "par" in k else "f1" if k in ("b.wps", "b.load") else "f0",
            "adapter release verification: " + p)
    get("b.i.load", E11I, "api.load_seconds", "f1", "isolated install: load (s)")
    get("b.i.wps", E11I, "api.greedy_words_per_second", "f1", "isolated install: greedy words/s")
    get("b.i.par", E11I, "api.greedy_mismatches_vs_frozen_predictions", "int", "isolated install: dev parity mismatches")
    get("b.i.checks", E11I, "ok", "raw", "isolated install: all checks passed")
    put("b.i.nchecks", len(load(E11I)["checks"]), "int", src=E11I, path="checks (count)",
        meaning="isolated-install checks")
    # relationships the text states
    assert VALUES["t1.e11.share"]["value"] == VALUES["t1.e12.share"]["value"]
    assert VALUES["t2.e11.share"]["value"] == VALUES["t2.e13.share"]["value"]
    for t, o in (("t1", "e12"), ("t2", "e13")):
        assert sum(VALUES[f"{t}.paired.{k}"]["value"] for k in ("both", "a_only", "b_only", "neither")) == 120
        assert VALUES[f"{t}.e11.acceptable"]["value"] == VALUES[f"{t}.paired.both"]["value"] + \
            VALUES[f"{t}.paired.b_only"]["value"]
    assert sum(VALUES[f"t2.{g}.e11"]["value"] for g in ("ordinary", "irregular", "names")) == \
        VALUES["t2.e11.acceptable"]["value"]
    assert sum(VALUES[f"t1.{g}.e11"]["value"] for g in ("ordinary", "irregular", "names")) == \
        VALUES["t1.e11.acceptable"]["value"]
    assert sum(VALUES[f"t1.{g}.e12"]["value"] for g in ("ordinary", "irregular", "names")) == \
        VALUES["t1.e12.acceptable"]["value"]


def write() -> None:
    (OUT / "data").mkdir(exist_ok=True)
    (OUT / "generated").mkdir(exist_ok=True)
    ext = {"revision": REV, "note": "values read with `git show <revision>:<source>`; see EVIDENCE_LEDGER.md",
           "values": {k: {**v, "rendered": render(v)} for k, v in VALUES.items()}}
    (OUT / "data" / "results_extract.json").write_text(json.dumps(ext, ensure_ascii=False, indent=1) + "\n",
                                                        encoding="utf-8")
    lines = ["% generated by tools/extract.py from data/results_extract.json; do not edit",
             "\\makeatletter"]
    for k, v in VALUES.items():
        if v["fmt"] != "raw" or isinstance(v["value"], (str, bool)):
            txt = render(v).replace("%", "\\%")
            lines.append(f"\\@namedef{{val@{k}}}{{{txt}}}")
    lines += ["\\newcommand{\\val}[1]{\\@ifundefined{val@#1}{\\textbf{??#1}}{\\@nameuse{val@#1}}}", "\\makeatother"]
    (OUT / "generated" / "values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # figure 2: two kinds of evidence, drawn in separate panels (no shared axis, no trend lines)
    V = lambda k: VALUES[k]["value"]  # noqa: E731
    with open(OUT / "generated" / "fig_reference.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "y", "top1"])
        for i, (lab, k) in enumerate((("First corrected adapter (E07)", "h.e07.top1"),
                                      ("+ recovered/ALT/synthetic (E08)", "h.e08.top1"),
                                      ("Released adapter (E11)", "h.e11.top1"),
                                      ("Longer schedule (E12)", "h.e12.top1"),
                                      ("Pronunciation-guided (E13)", "h.e13.top1"))):
            w.writerow([lab, 5 - i, q(100 * V(k))])
    with open(OUT / "generated" / "fig_accept.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "y", "share", "sample"])
        rows = (("Released adapter (E11)", "t1.e11.share", 1), ("Longer schedule (E12)", "t1.e12.share", 1),
                ("Released adapter (E11)", "t2.e11.share", 2), ("Pronunciation-guided (E13)", "t2.e13.share", 2))
        for i, (lab, k, s) in enumerate(rows):
            w.writerow([lab, [5, 4, 2, 1][i], q(100 * V(k)), s])


if __name__ == "__main__":
    build()
    write()
    print(f"{len(VALUES)} values from revision {REV[:8]} → {OUT.relative_to(ROOT)}/data/results_extract.json")
