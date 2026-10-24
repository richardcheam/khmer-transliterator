"""The frozen evaluation benchmark: versioned test panels, scoring and paired confidence intervals.

Every experiment is scored on the same panels, so numbers are comparable across experiments. A panel file is
`latin<TAB>form1|form2...` (one Latin word, every accepted Khmer spelling; a prediction is right if it equals any
of them, the rule since D133). Files live in `data/benchmark/<version>/` (not in git: test data); their hashes,
sizes and descriptions are pinned in the tracked manifest `benchmarks/<version>.json`. `load` refuses a panel
whose hash differs, so a benchmark can never change silently. A new benchmark is a new version.

**v1** (frozen 2026-10-24, = the D173/D174 panels): `romanised_test` (2,226 words), `names_test2` (223),
`pick_test` (150), `dev41` (41, diagnostic: development-contaminated), `list_test` (125). The 120-word reserve is
not a panel: it stays sealed for a final owner A/B.
"""
from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "benchmarks"
FILES = ROOT / "data" / "benchmark"

Panel = list[tuple[str, frozenset[str]]]


def _sha(path: Path) -> str:
    """sha256 of a file (`experiment.sha256`)."""
    from khmer_translit.experiment import sha256
    return sha256(path)


def load(version: str = "v1") -> dict[str, Panel]:
    """The panels of one benchmark version, checked against its manifest.

    Args:
        version: e.g. "v1".

    Returns:
        `{panel: [(latin, frozenset of accepted Khmer forms), ...]}` in manifest order, items sorted by Latin word.

    Raises:
        FileNotFoundError: If the manifest or a panel file is missing.
        ValueError: If a panel file's sha256 or item count differs from the manifest.

    Example:
        `len(load("v1")["pick_test"])` → 150.
    """
    manifest = json.loads((MANIFESTS / f"{version}.json").read_text(encoding="utf-8"))
    out = {}
    for name, info in manifest["panels"].items():
        path = FILES / version / f"{name}.tsv"
        if _sha(path) != info["sha256"]:
            raise ValueError(f"{path} does not match benchmarks/{version}.json (sha256 differs)")
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            latin, forms = line.split("\t")
            items.append((latin, frozenset(forms.split("|"))))
        if len(items) != info["n"]:
            raise ValueError(f"{path}: {len(items)} items, manifest says {info['n']}")
        out[name] = items
    return out


def freeze(panels: Mapping[str, Panel], version: str, descriptions: Mapping[str, str], note: str) -> Path:
    """Write a new benchmark version: one TSV per panel and the manifest. Refuses to overwrite.

    Args:
        panels: `{panel: [(latin, forms), ...]}`.
        version: The new version name (e.g. "v2").
        descriptions: One line per panel: what it is and its caveats.
        note: Where the panels came from.

    Returns:
        The manifest path (commit it).

    Raises:
        FileExistsError: If the version already exists.
    """
    manifest_path = MANIFESTS / f"{version}.json"
    folder = FILES / version
    if manifest_path.exists() or folder.exists():
        raise FileExistsError(f"benchmark {version} already exists; make a new version instead")
    folder.mkdir(parents=True)
    entries = {}
    for name, items in panels.items():
        path = folder / f"{name}.tsv"
        path.write_text("".join(f"{a}\t{'|'.join(sorted(b))}\n" for a, b in sorted(items)), encoding="utf-8")
        entries[name] = {"n": len(items), "sha256": _sha(path), "description": descriptions[name]}
    MANIFESTS.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps({"version": version, "note": note, "panels": entries},
                                        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest_path


def outcomes(predictions: Sequence[str], panel: Panel) -> list[int]:
    """1 for each prediction that equals an accepted form of its word, else 0.

    Args:
        predictions: One spelling per panel item, in panel order.
        panel: The panel.

    Returns:
        The 0/1 outcomes.

    Raises:
        ValueError: If the lengths differ.

    Examples:
        >>> outcomes(["ក", "ខ"], [("ka", frozenset({"ក"})), ("kha", frozenset({"គ"}))])
        [1, 0]
    """
    if len(predictions) != len(panel):
        raise ValueError(f"{len(predictions)} predictions for {len(panel)} items")
    return [int(p in forms) for p, (_latin, forms) in zip(predictions, panel)]


def evaluate_model(npz: Path, known: frozenset[str], panels: Mapping[str, Panel], beam: int = 4) -> dict:
    """Score one exported model on every panel: raw beam top-1 and re-ranked (`translit.pick` with `known`).

    Args:
        npz: The model's `translit.npz`.
        known: The re-ranking dictionary (e.g. `lexicon.keys()`).
        panels: `load(version)`.
        beam: Beam width (4, as served).

    Returns:
        `{panel: {"raw", "reranked", "raw_correct", "reranked_correct", "n", "items"}}` with accuracies rounded
        to 4 places for display, exact correct counts, and per-item re-ranked outcomes.
    """
    from khmer_translit import translit
    model = translit.load(npz)
    res = {}
    for name, panel in panels.items():
        raw, rer = [], []
        for nb in model.nbest_batch([a for a, _ in panel], beam):
            raw.append(nb[0][0] if nb else "")
            rer.append(translit.pick(nb, known))
        o_raw, o_rer = outcomes(raw, panel), outcomes(rer, panel)
        res[name] = {"raw": round(sum(o_raw) / len(panel), 4), "reranked": round(sum(o_rer) / len(panel), 4),
                     "raw_correct": sum(o_raw), "reranked_correct": sum(o_rer), "n": len(panel), "items": o_rer}
    return res


def paired_bootstrap(diffs: Sequence[Fraction], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI, in percentage points, of the mean of per-item paired differences.

    Items are resampled with replacement (`random.Random(seed)`), so the interval reflects which words happen to be
    in the panel. Seed-to-seed variation is separate: average each item's difference over seeds first.

    Args:
        diffs: One difference per item, e.g. (treatment − control) outcome averaged over seeds.
        n_boot: Resamples (2,000).
        seed: Resampling seed (0).

    Returns:
        `(low, high)` in percentage points, rounded to 2 places.

    Examples:
        >>> paired_bootstrap([Fraction(1)] * 10)
        (100.0, 100.0)
    """
    n = len(diffs)
    rng = random.Random(seed)
    boots = sorted(float(sum(diffs[rng.randrange(n)] for _ in range(n)) / n * 100) for _ in range(n_boot))
    return round(boots[int(0.025 * n_boot)], 2), round(boots[int(0.975 * n_boot) - 1], 2)
