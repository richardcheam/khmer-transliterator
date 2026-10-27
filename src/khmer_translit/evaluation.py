"""Scoring, paired comparison and decision rules for the English phonetic benchmark (experiment E02).

- `score_predictions`: multi-reference exact match (NFC) of each item's top-1 candidate, top-k coverage and
  minimum character error rate, aggregated **macro over lexical families** (each family's mean, then the mean
  over families) and as plain item accuracy. Every reference id must have a prediction entry: a missing entry is
  an error, an empty candidate list counts as a failure (abstention), an unknown id is an error.
- `compare_paired_runs`: treatment vs control over paired seeds: per-seed macro scores, the mean paired gain and
  a paired **family bootstrap** (each family's difference averaged over seeds, families resampled). The interval
  is conditional on these trained seeds.
- `evidence_fields` / `adoption`: the E02 decision rules, from unrounded values.
- `power_grid`: simulated power of the paired family test for a declared grid of discordance rates.
- `measurement_controls`: checks that the scorer accepts every known alternative and NFC equivalents and rejects
  wrong spellings and misaligned predictions.

Standard library only.
"""
from __future__ import annotations

import random
import unicodedata
from collections.abc import Mapping, Sequence
from statistics import mean

from khmer_translit.reference_data import ReferenceItem

THRESHOLD = 5.0          # practical gain, percentage points
N_BOOT = 10_000
BOOT_SEED = 20261027


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def candidates_of(pred: object) -> list[str]:
    """Ordered candidate spellings of one prediction: a `PredictionResult`-like object (`.candidates`, each with
    `.text`) or a plain sequence of strings.

    Examples:
        >>> candidates_of(["ក", "ខ"])
        ['ក', 'ខ']
    """
    if hasattr(pred, "candidates"):
        return [c.text for c in pred.candidates]
    if isinstance(pred, str):
        raise TypeError("a prediction must be a sequence of candidates, not one string")
    return list(pred)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings (code points).

    Examples:
        >>> edit_distance("កខ", "កគខ"), edit_distance("", "ក")
        (1, 1)
    """
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def item_outcomes(items: Sequence[ReferenceItem], predictions: Mapping[str, object], top_k: int = 4) -> dict:
    """Per-item outcomes after strict id alignment.

    Args:
        items: Reference items (ids unique).
        predictions: `{item id: candidates}` covering every item id, nothing else.
        top_k: Candidates considered for coverage.

    Returns:
        `{id: {"top1": 0/1, "topk": 0/1, "cer": float, "abstained": bool, "prediction": str}}`.

    Raises:
        ValueError: Missing or unknown prediction ids.
    """
    ids = [it.id for it in items]
    missing = [i for i in ids if i not in predictions]
    extra = sorted(set(predictions) - set(ids))
    if missing or extra:
        raise ValueError(f"prediction ids do not match references: {len(missing)} missing (e.g. {missing[:3]}), "
                         f"{len(extra)} unknown (e.g. {extra[:3]})")
    out = {}
    for it in items:
        cands = [_nfc(c) for c in candidates_of(predictions[it.id])]
        acc = {_nfc(a) for a in it.accepted}
        top1 = cands[0] if cands else ""
        cer = min(edit_distance(top1, a) / max(1, len(a)) for a in acc)
        out[it.id] = {"top1": int(bool(cands) and top1 in acc), "topk": int(any(c in acc for c in cands[:top_k])),
                      "cer": cer, "abstained": not cands, "prediction": top1}
    return out


def macro(items: Sequence[ReferenceItem], values: Mapping[str, float]) -> float:
    """Macro average over lexical families: mean of each family's item mean.

    Examples:
        >>> from khmer_translit.reference_data import ReferenceItem as R
        >>> its = [R(f"{w}#default", w, w, f, "default", "", "a", ["ក"], "ក", [], "attested", "dev")
        ...        for w, f in [("a", "f1"), ("b", "f1"), ("c", "f2")]]
        >>> macro(its, {"a#default": 1, "b#default": 0, "c#default": 1})
        0.75
    """
    fam: dict[str, list[float]] = {}
    for it in items:
        fam.setdefault(it.family, []).append(values[it.id])
    return mean(mean(v) for v in fam.values()) if fam else 0.0


def score_predictions(items: Sequence[ReferenceItem], predictions: Mapping[str, object], top_k: int = 4) -> dict:
    """Score one run on one item set.

    Args:
        items: Reference items.
        predictions: `{id: candidates}` (see `item_outcomes`).
        top_k: Coverage depth.

    Returns:
        `{"n_items", "n_families", "primary_macro", "item_accuracy", "topk_macro", "mean_cer", "abstained",
        "correct_items", "items": {id: outcome}}`; rates in [0, 1], unrounded.
    """
    o = item_outcomes(items, predictions, top_k)
    return {"n_items": len(items), "n_families": len({it.family for it in items}),
            "primary_macro": macro(items, {i: v["top1"] for i, v in o.items()}),
            "item_accuracy": sum(v["top1"] for v in o.values()) / max(1, len(o)),
            "correct_items": sum(v["top1"] for v in o.values()),
            "topk_macro": macro(items, {i: v["topk"] for i, v in o.items()}),
            "mean_cer": mean(v["cer"] for v in o.values()) if o else 0.0,
            "abstained": sum(v["abstained"] for v in o.values()), "items": o}


def family_diffs(items: Sequence[ReferenceItem], control: Sequence[Mapping[str, int]],
                 treatment: Sequence[Mapping[str, int]]) -> dict[str, float]:
    """Per family: treatment − control top-1, averaged over the family's items and over paired seeds.

    Args:
        items: Reference items.
        control: Per seed, `{id: 0/1}`.
        treatment: Per seed (same order of seeds), `{id: 0/1}`.

    Returns:
        `{family: difference in [-1, 1]}`.
    """
    if len(control) != len(treatment) or not control:
        raise ValueError("control and treatment need the same, non-zero number of seeds")
    fam: dict[str, list[ReferenceItem]] = {}
    for it in items:
        fam.setdefault(it.family, []).append(it)
    return {f: mean(mean(t[it.id] - c[it.id] for it in its) for c, t in zip(control, treatment))
            for f, its in fam.items()}


def bootstrap_ci(diffs: Sequence[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    """95% percentile interval (percentage points) of the mean family difference, resampling families.

    Examples:
        >>> bootstrap_ci([1.0] * 5, n_boot=100)
        (100.0, 100.0)
    """
    n = len(diffs)
    rng = random.Random(seed)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n * 100 for _ in range(n_boot))
    return boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def compare_paired_runs(items: Sequence[ReferenceItem], control: Sequence[Mapping[str, int]],
                        treatment: Sequence[Mapping[str, int]], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Paired comparison of two arms over the same seeds.

    Args:
        items: Reference items scored by both.
        control: Per seed `{id: top1 0/1}`.
        treatment: Per seed, same seeds in the same order.
        n_boot: Bootstrap resamples. seed: Bootstrap seed.

    Returns:
        `{"seed_gains_pp", "mean_gain_pp", "ci95_pp", "control_macro", "treatment_macro", "families"}`,
        unrounded percentage points.
    """
    cm = [macro(items, c) * 100 for c in control]
    tm = [macro(items, t) * 100 for t in treatment]
    gains = [t - c for c, t in zip(cm, tm)]
    fd = family_diffs(items, control, treatment)
    lo, hi = bootstrap_ci(list(fd.values()), n_boot, seed)
    return {"seed_gains_pp": gains, "mean_gain_pp": mean(gains), "ci95_pp": [lo, hi], "control_macro": cm,
            "treatment_macro": tm, "families": len(fd)}


def evidence_fields(ci: Sequence[float], threshold: float = THRESHOLD) -> dict:
    """The three E02 evidence fields from a conditional 95% interval (percentage points).

    Examples:
        >>> evidence_fields([1.0, 4.0])
        {'positive_effect_supported': True, 'meaningful_gain_ruled_out_conditional': True, 'practically_equivalent_conditional': True}
    """
    lo, hi = ci
    return {"positive_effect_supported": lo > 0, "meaningful_gain_ruled_out_conditional": hi < threshold,
            "practically_equivalent_conditional": -threshold < lo and hi < threshold}


def summary_status(qualified: bool, fields: Mapping[str, bool]) -> str:
    """`supported_improvement`, `meaningful_gain_not_supported` or `inconclusive` (only qualified data decide).

    Examples:
        >>> summary_status(False, {"positive_effect_supported": True, "meaningful_gain_ruled_out_conditional": False})
        'inconclusive'
    """
    if not qualified:
        return "inconclusive"
    if fields["positive_effect_supported"]:
        return "supported_improvement"
    if fields["meaningful_gain_ruled_out_conditional"]:
        return "meaningful_gain_not_supported"
    return "inconclusive"


def adoption(qualified: bool, comparison: Mapping, controls_ok: bool, runtime_ok: bool | None,
             threshold: float = THRESHOLD) -> dict:
    """The conservative adoption rule: every condition must hold (`runtime_ok=None`, not measured, fails).

    Returns:
        `{"adopt": bool, "conditions": {name: bool}}`.
    """
    g = comparison["seed_gains_pp"]
    cond = {"benchmark_qualified": qualified, "mean_gain_ge_threshold": comparison["mean_gain_pp"] >= threshold,
            "positive_in_2_of_3_seeds": sum(x > 0 for x in g) >= 2 and len(g) >= 3,
            "ci_lower_gt_0": comparison["ci95_pp"][0] > 0, "controls_ok": controls_ok, "runtime_ok": bool(runtime_ok)}
    return {"adopt": all(cond.values()), "conditions": cond}


def power_grid(n_families: int, discordance: Sequence[float], gain_pp: float = THRESHOLD, n_sim: int = 400,
               n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Simulated power of the paired family bootstrap to show a positive gain (lower bound > 0).

    Model: each family is discordant with probability `d`; a discordant family favours the treatment with the
    probability that makes the expected gain `gain_pp`. One seed, one item per family: a conservative stand-in
    (averaging three seeds lowers the variance of each family's difference).

    Args:
        n_families: Families in the evaluation set.
        discordance: Discordance rates to try.
        gain_pp: True gain, percentage points.
        n_sim: Simulated benchmarks per rate. n_boot: Resamples per benchmark. seed: RNG seed.

    Returns:
        `{str(d): power}`.
    """
    rng = random.Random(seed)
    out = {}
    for d in discordance:
        p_win = 0.5 + gain_pp / 100 / (2 * d)
        if p_win > 1:
            out[str(d)] = float("nan")
            continue
        hits = 0
        for _ in range(n_sim):
            diffs = [(1.0 if rng.random() < p_win else -1.0) if rng.random() < d else 0.0 for _ in range(n_families)]
            lo, _hi = bootstrap_ci(diffs, n_boot, rng.randrange(1 << 30))
            hits += lo > 0
        out[str(d)] = hits / n_sim
    return out


def measurement_controls(items: Sequence[ReferenceItem]) -> dict:
    """Scorer checks on real items: every accepted form and its NFD form must pass, a corrupted form must fail,
    and misaligned prediction maps must be refused.

    Args:
        items: Reference items (any split).

    Returns:
        `{"accepted_pass", "nfd_pass", "corrupt_fail", "missing_refused", "unknown_refused", "ok"}`.
    """
    alt = {it.id: [a] for it in items for a in it.accepted[-1:]}
    ok_alt = sum(v["top1"] for v in item_outcomes(items, alt).values()) == len(items)
    nfd = {it.id: [unicodedata.normalize("NFD", it.accepted[0])] for it in items}
    ok_nfd = sum(v["top1"] for v in item_outcomes(items, nfd).values()) == len(items)
    bad = {it.id: [it.accepted[0] + "ក"] for it in items}
    ok_bad = sum(v["top1"] for v in item_outcomes(items, bad).values()) == 0
    refused = {}
    for name, preds in (("missing_refused", dict(list(alt.items())[1:])), ("unknown_refused", {**alt, "zz#x": []})):
        try:
            item_outcomes(items, preds)
            refused[name] = False
        except ValueError:
            refused[name] = True
    res = {"accepted_pass": ok_alt, "nfd_pass": ok_nfd, "corrupt_fail": ok_bad, **refused}
    res["ok"] = all(res.values())
    return res
