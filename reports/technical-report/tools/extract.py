#!/usr/bin/env python3
"""Build the technical report's results extract from the project's committed result files.

Empirical values in the report come from three kinds of source, recorded per value in `kind`:

- `json`: read from a JSON file committed in this repository, at a pinned git revision (`git show <rev>:<path>`);
- `derived`: computed here from `json` values (differences, ranges, shares), with the rule in `derived_from`;
- `transcribed`: a number that exists only in a committed report's prose or table. It is copied with its file,
  section and revision, and checked against `json` values where a relationship exists.

Architecture and recipe settings (layer sizes, learning rate, number of updates), sample-design constants (120
families, 40/40/40), dates, section numbers and bibliographic data are written literally in `main.tex`.

The script writes:

- `data/results_extract.json`: each value with its kind, source, field or section, revision and meaning;
- `generated/values.tex`: one LaTeX macro per value (`\\val{key}`);
- `generated/*.csv`: the data behind the figures.

It asserts the relationships the report states (sums, ranges, paired denominators, rounding of derived
percentages) and fails if `main.tex` uses a `\\val{key}` that the extract does not define. No model is run and no
judgment, reference or training file is read.

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
E02DC = "experiments/E02-diagnostics-2026-10-27/compat/summary.json"
E02DP = "experiments/E02-diagnostics-2026-10-27/sensitivity/posthoc.json"
E02B = "experiments/E02-english-phonetic/reports/data_build.json"
E02P = "experiments/E02-release/reports/provenance_facts.json"
E01 = "experiments/E01-unified/results.json"
E03A = "experiments/E03-assessable-phones/reports/data_audit.json"
E03O = "experiments/E03-assessable-phones/reports/overlap.json"
E08A = "experiments/E08-data-expansion/reports/training_arms.json"
E09D = "experiments/E09-byt5-decoding/reports/dev_comparison.json"
E11X = "experiments/E11-real-data-byt5/release/export.json"
E11B = "experiments/E11-real-data-byt5/coverage/batches.json"
E02RES = "experiments/E02-english-phonetic/RESULTS.md"
E08R = "experiments/E08-data-expansion/REPORT.md"
E11RP = "experiments/E11-real-data-byt5/REPORT.md"
E12R = "experiments/E12-longer-schedule/REPORT.md"
E13RP = "experiments/E13-pronunciation-guided/REPORT.md"

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
        meaning: str, n: str = "", derived: str = "", kind: str | None = None) -> None:
    """Record one value; `fmt` renders it for LaTeX ('pct' share→%, 'pp' already points, 'int', 'f3', 'f2', 'raw')."""
    kind = kind or ("json" if src and path else "derived")
    VALUES[key] = {"kind": kind, "value": value, "fmt": fmt, "source": src, "field": path,
                   "revision": rev if (src or kind == "transcribed") else None,
                   "meaning": meaning, "denominator": n, "derived_from": derived}


def tr(key: str, value, fmt: str, src: str, section: str, meaning: str, n: str = "") -> None:
    """A value that exists only in a committed report's text or table (file and section at REV)."""
    put(key, value, fmt, src=src, path=section, meaning=meaning, n=n, kind="transcribed")


def pct(num: int, den: int) -> float:
    return num / den


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
    if f == "pct0":
        return str(q(100 * x, 0))
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
        tr(f"t1.{g}.e12", a, "int", E12R, "§5.1 table", f"e12 acceptable in group {g}", "40 words")
        tr(f"t1.{g}.e11", b, "int", E12R, "§5.1 table", f"e11 acceptable in group {g}", "40 words")
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
    # ---------------------------------------------------------------- correction pass (D203): more sources
    # E02 benchmark composition: candidate-anchored items per final track (aggregate strata only)
    get("s.anch.primary", E02, r + "primary.strata.candidate_anchored.n", "int",
        "primary items whose reference was chosen from an earlier model's candidates", "165 items")
    get("s.anch.flagged", E02, r + "final_flagged.strata.candidate_anchored.n", "int",
        "flagged items whose reference was chosen from an earlier model's candidates", "26 items")
    get("s.flagged.n", E02, r + "final_flagged.n_items", "int", "flagged-track items")
    V = lambda k: VALUES[k]["value"]  # noqa: E731
    put("s.anch.final", V("s.anch.primary") + V("s.anch.flagged"), "int", meaning="candidate-anchored items in "
        "both final tracks", derived="s.anch.primary + s.anch.flagged", n="191 items")
    put("s.final.n", V("s.primary.n") + V("s.flagged.n"), "int", meaning="items in both final tracks",
        derived="s.primary.n + s.flagged.n")
    assert all(V(f"s.anch.{k}") == x for k, x in (("primary", 82), ("flagged", 12), ("final", 94)))
    assert (V("s.primary.n"), V("s.flagged.n"), V("s.final.n")) == (165, 26, 191)
    get("s.seed.0", E02, "comparisons.confirmatory.seed_gains_pp.0", "pp", "phones − letters, seed 0")
    get("s.seed.1", E02, "comparisons.confirmatory.seed_gains_pp.1", "pp", "phones − letters, seed 1")
    get("s.seed.2", E02, "comparisons.confirmatory.seed_gains_pp.2", "pp", "phones − letters, seed 2")
    grid = field(load(E02), "power.power_primary")
    put("s.power.min", min(grid.values()), "f2", src=E02, path="power.power_primary", kind="derived",
        meaning="simulated power for a true +5-point gain, minimum over the registered discordance grid",
        derived="min over power.power_primary")
    put("s.power.max", max(grid.values()), "f2", src=E02, path="power.power_primary", kind="derived",
        meaning="same, maximum over the grid", derived="max over power.power_primary")
    tr("s.dev.n", 86, "int", E02RES, "Selection", "primary-eligible development items used to select the epoch")
    for arm, sc in field(load(E02), "selection.dev_primary_macro").items():      # each dev score is k/86
        assert abs(sc * 86 - round(sc * 86)) < 1e-9, (arm, sc)
    get("s.train", E02B, "train_rows.train", "int", "small model training pairs")
    get("s.trainval", E02B, "train_rows.train_val", "int", "small model validation-loss pairs")
    get("s.epoch", E02P, "checkpoint.meta.epoch", "int", "selected epoch of the released small model")
    # sensitivity: range over all three seeds (not first and last)
    rates = [V(f"sens.s{i}.donor") for i in range(3)]
    put("sens.min", min(rates), "pct", meaning="minimum donor-phone change rate over seeds 0–2",
        derived="min(sens.s0.donor, sens.s1.donor, sens.s2.donor)", n="94 words")
    put("sens.max", max(rates), "pct", meaning="maximum donor-phone change rate over seeds 0–2",
        derived="max(sens.s0.donor, sens.s1.donor, sens.s2.donor)", n="94 words")
    closer = [field(load(E02DP), f"toward_given_phones.phones_s{i}|donor.closer_to_given_phones") for i in range(3)]
    put("sens.closer.min", min(closer), "int", src=E02DP, path="toward_given_phones.phones_s*|donor.closer_to_given_phones",
        kind="derived", meaning="changed outputs closer to the donor's consonants, minimum over seeds",
        derived="min over seeds 0–2")
    put("sens.closer.max", max(closer), "int", src=E02DP, path="toward_given_phones.phones_s*|donor.closer_to_given_phones",
        kind="derived", meaning="same, maximum over seeds", derived="max over seeds 0–2")
    # historical evaluation overlap with the small model's training words: two populations, two denominators
    get("ov.primary", E07, "strata.in_default_e02_training=True.items", "int",
        "primary historical-eval items whose word is in the small model's training file", "228 items")
    get("ov.primary.not", E07, "strata.in_default_e02_training=False.items", "int", "primary items not in it")
    get("ov.primary.n", E07, "historical_eval.items", "int", "primary historical-eval items")
    get("ov.full", E03O, "eval.e02_train_words", "int",
        "words of the full eval partition that occur in the small model's training data", "245 words")
    get("ov.full.n", E03O, "eval.words", "int", "words in the full eval partition (incl. non-primary)")
    assert V("ov.primary") + V("ov.primary.not") == V("ov.primary.n") == V("h.items")
    assert (V("ov.primary"), V("ov.primary.n"), V("ov.full"), V("ov.full.n")) == (110, 228, 122, 245)
    # data pipeline counts
    get("d.recovered", E03A, "recovery.recovered:khmer_target_known_only", "int", "rows recovered by the selection repair")
    get("d.romanised", E03A, "recovery.excluded:latin_romanised", "int", "rows still excluded (romanised Latin side)")
    get("d.dacs", E03A, "retained.pairs", "int", "dictionary-assessable, consonant-screened pairs")
    get("d.rows", E02DC, "rows", "int", "small model training rows audited")
    get("d.lexhits", E02DC, "coverage.google_exact_target_hits", "int", "training targets found by exact lexicon lookup")
    get("d.heldout", E02DC, "orthography_validation.heldout.all.headwords", "int", "held-out lexicon headwords")
    get("d.consrate", E02DC, "orthography_validation.heldout.all.rate", "pct", "consonant-class agreement", "headwords")
    get("d.r.rows", E08A, "R.optimisation_rows", "int", "released adapter: optimised training rows")
    get("d.r.words", E08A, "R.optimisation_words", "int", "released adapter: training words")
    get("d.r.recovered", E08A, "R.by_provenance.recovered", "int", "released adapter: recovered pairs")
    get("d.r.alt", E08A, "R.by_provenance.attested_alt", "int", "released adapter: ALT-attested pairs")
    put("d.r.control", V("d.r.rows") - V("d.r.recovered") - V("d.r.alt"), "int", meaning="released adapter: "
        "screened mined pairs optimised", derived="d.r.rows − d.r.recovered − d.r.alt (the 93 train-val rows of "
        "the control part are not optimised; identity.json records control:train = 2,255)")
    assert V("d.r.control") == 2255
    get("d.synth.rows", E08A, "RS.by_provenance.google_ai_mode_synthetic", "int", "synthetic rows (historical arm)")
    tr("d.synth.words", 193, "int", E08R, "Update 3", "words covered by the synthetic rows")
    get("x.tensors", E11X, "tensors", "int", "adapter tensors")
    get("x.params", E11X, "parameters", "int", "adapter parameters")
    get("x.hours", E11, "training.active_training_hours", "f1", "released adapter training time (active hours)")
    put("x.params.m", V("x.params") / 1e6, "f2", meaning="adapter parameters in millions", derived="x.params / 10^6")
    get("sens.n", E02D, "aggregate.phones_s0.donor.n", "int", "words in the donor-phone condition")
    assert all(field(load(E02D), f"aggregate.phones_s{i}.donor.n") == V("sens.n") for i in range(3))
    get("e06.correct", E07, "historical_eval.systems.byt5_e06.correct_items", "int",
        "collapsed attention-only adapter: correct eval items", "228 items")
    get("e06.top", E07, "historical_eval.systems.byt5_e06.most_common_top1.1", "int",
        "eval words given the single most common output by the collapsed attention-only adapter", "228 items")
    get("e09.cands", E09D, "table.e08:beam4_reg.distinct_candidates_per_item", "f2",
        "distinct candidates per item, registered beam-4, synthetic-data adapter (dev)")
    ek = field(load(E11), "eval.error_kinds")
    shares = {m: d["vowels_only"] / sum(d.values()) for m, d in ek.items()}
    put("vow.min", min(shares.values()), "pct0", src=E11, path="eval.error_kinds", kind="derived",
        meaning="vowel-only share of reference mismatches, minimum over adapters", derived="vowels_only / all kinds")
    put("vow.max", max(shares.values()), "pct0", src=E11, path="eval.error_kinds", kind="derived",
        meaning="same, maximum over adapters", derived="vowels_only / all kinds")
    get("e01.gain", E01, "gate.english_mean_pp", "f2", "E01 aux-task gain on the English panels (points)")
    # development review cross-table (E11 REPORT §5 table only)
    for k, v, m in (("x.judged", 109, "distinct judged candidate spellings"),
                    ("x.diff", 72, "judged spellings that differ from the mined reference"),
                    ("x.diff.acc", 28, "of those, judged acceptable"),
                    ("x.match", 37, "judged spellings that match the mined reference"),
                    ("x.match.not", 8, "of those, judged not acceptable"),
                    ("x.diff.unc", 3, "differing spellings judged uncertain"),
                    ("x.match.acc", 28, "matching spellings judged acceptable"),
                    ("x.match.unc", 1, "matching spellings judged uncertain")):
        tr(k, v, "int", E11RP, "§5 table 'Owner judgment vs the old reference spellings'", m)
    assert V("x.diff") + V("x.match") == V("x.judged")
    assert V("x.diff.acc") + 41 + V("x.diff.unc") == V("x.diff")
    assert V("x.match.acc") + V("x.match.not") + V("x.match.unc") == V("x.match")
    put("x.diff.share", pct(V("x.diff.acc"), V("x.diff")), "pct0", meaning="share of differing spellings judged "
        "acceptable", derived="x.diff.acc / x.diff")
    put("x.match.share", pct(V("x.match.not"), V("x.match")), "pct0", meaning="share of matching spellings judged "
        "not acceptable", derived="x.match.not / x.match")
    tr("t1.changed", 65, "int", E12R, "§5.2", "test-1 words whose spelling differs between E12 and E11", "120 words")
    assert V("t1.changed") <= 120 - 0 and 120 - V("t1.changed") == 55       # E12 REPORT: one spelling shown for 55
    tr("e13.families", 42, "int", E13RP, "summary 'Data'", "difficult training families with accepted spellings")
    # coverage review provenance: how its 93 judgments were obtained
    get("cov.reused", E11B, "reused_judgments", "int", "coverage judgments reused from earlier reviews")
    get("cov.reused.e13", E11B, "reused_by_source.e13_collection", "int", "reused from the E13 collection review")
    get("cov.reused.dev", E11B, "reused_by_source.e11_review", "int", "reused from the chat-transcribed dev review")
    get("cov.new", E11B, "new_judgments_needed", "int", "new coverage judgments clicked on the review page")
    assert V("cov.reused") + V("cov.new") == V("cov.cands")
    assert V("cov.reused.e13") + V("cov.reused.dev") == V("cov.reused")
    put("cov.clicked", V("cov.new") + V("cov.reused.e13"), "int", meaning="coverage judgments given by clicks on "
        "the review page (new plus reused from the clicked E13 collection review)", derived="cov.new + cov.reused.e13")
    # derived quantities used in the prose
    hw = lambda p, n: 1.96 * (p * (1 - p) / n) ** 0.5  # noqa: E731
    put("t1.hw", 100 * hw(V("t1.e11.share"), 120), "f1", meaning="normal-approximation 95% half-width of the "
        "released adapter's rate, test 1 (points)", derived="1.96·sqrt(p(1−p)/120)")
    put("t2.hw", 100 * hw(V("t2.e11.share"), 120), "f1", meaning="same, test 2 (points)",
        derived="1.96·sqrt(p(1−p)/120)")
    put("irr.lo", min(V("t1.irregular.e11"), V("t2.irregular.e11")) / 40, "pct0", meaning="released adapter, "
        "irregular group acceptable share, lower of the two tests", derived="min(t1, t2 irregular)/40")
    put("irr.hi", max(V("t1.irregular.e11"), V("t2.irregular.e11")) / 40, "pct0", meaning="same, higher",
        derived="max(t1, t2 irregular)/40")
    put("acc.lo", min(V("t1.e11.share"), V("t2.e11.share")), "pct0", meaning="released adapter acceptable share, "
        "lower of the two tests", derived="min(t1, t2)")
    put("acc.hi", max(V("t1.e11.share"), V("t2.e11.share")), "pct0", meaning="same, higher", derived="max(t1, t2)")
    put("cov.none.share", V("cov.none") / V("cov.words"), "pct0", meaning="hard words with no acceptable "
        "candidate", derived="cov.none / cov.words")
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
    V = lambda k: VALUES[k]["value"]  # noqa: E731
    slow = [V("b.lat1") / V(f"s.t{i}.p50") for i in (1, 2, 3)]
    put("slow.lo", min(slow), "f0", meaning="adapter greedy median latency / small-model median latency, lowest "
        "over the small model's three timing runs", derived="b.lat1 / s.t*.p50")
    put("slow.hi", max(slow), "f0", meaning="same, highest", derived="b.lat1 / s.t*.p50")
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


def check_tex() -> None:
    """Every `\\val{key}` in main.tex must exist; derived percentages must match their counts when rounded."""
    import re
    used = set(re.findall(r"\\val\{([^}]+)\}", (OUT / "main.tex").read_text(encoding="utf-8")))
    missing = sorted(used - set(VALUES))
    assert not missing, f"main.tex uses undefined values: {missing}"
    for share, num, den in (("t1.e11.share", "t1.e11.acceptable", 120), ("t2.e11.share", "t2.e11.acceptable", 120),
                            ("t1.e12.share", "t1.e12.acceptable", 120), ("t2.e13.share", "t2.e13.acceptable", 120)):
        assert render(VALUES[share]) == str(q(100 * VALUES[num]["value"] / den)), share
    assert render(VALUES["x.diff.share"]) == "39" and render(VALUES["x.match.share"]) == "22"
    assert render(VALUES["sens.min"]) == "33.0" and render(VALUES["sens.max"]) == "44.7"
    print(f"checked: {len(used)} \\val keys used in main.tex, all defined; derived percentages consistent")


if __name__ == "__main__":
    build()
    write()
    check_tex()
    print(f"{len(VALUES)} values from revision {REV[:8]} → {OUT.relative_to(ROOT)}/data/results_extract.json")
