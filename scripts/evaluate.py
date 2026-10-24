#!/usr/bin/env python3
"""Score exported models on the frozen benchmark: the one way to get comparable numbers.

    .venv/bin/python scripts/evaluate.py models/model_b models/model_e            # benchmark v1, lexicon re-ranking
    .venv/bin/python scripts/evaluate.py models/model_b --known served --out model_cards/model_b.eval.json

For each model folder (with `translit.npz`): raw beam-4 top-1 and re-ranked exact match on every panel of the
benchmark (`khmer_translit.benchmark`). Re-ranking dictionary: `lexicon` = Google lexicon headwords only (the
clean list, used by E01), or `served` = the served list (lexicon + Chhunneng `known_targets.txt`, not releasable).
Prints a table; `--out` writes JSON with the model's sha256, the benchmark manifest's sha256 and per-item outcomes.
Experiment-specific comparisons (paired seeds, adoption rules) live in each experiment's script.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from khmer_translit import benchmark, lexicon  # noqa: E402
from khmer_translit.experiment import provenance, sha256  # noqa: E402


def known_words(kind: str) -> frozenset[str]:
    """The re-ranking dictionary.

    Args:
        kind: "lexicon" (Google headwords only) or "served" (+ `models/model/known_targets.txt`).

    Returns:
        The Khmer words.
    """
    words = lexicon.keys()
    if kind == "served":
        words |= frozenset((ROOT / "models/model/known_targets.txt").read_text(encoding="utf-8").split())
    return frozenset(words)


def main() -> int:
    """CLI (see the module docstring).

    Returns:
        0.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", type=Path, help="model folders containing translit.npz")
    ap.add_argument("--benchmark", default="v1")
    ap.add_argument("--known", choices=["lexicon", "served"], default="lexicon")
    ap.add_argument("--out", type=Path, default=None, help="JSON report (one model) or folder (several)")
    args = ap.parse_args()
    panels = benchmark.load(args.benchmark)
    known = known_words(args.known)
    print(f"{'model':28}" + "".join(f"{p:>16}" for p in panels) + "   (re-ranked exact; raw)")
    for folder in args.models:
        npz = folder / "translit.npz"
        res = benchmark.evaluate_model(npz, known, panels)
        print(f"{str(folder):28}" + "".join(f"{r['reranked']:>9.1%} {r['raw']:>5.1%}" for r in res.values()))
        if args.out:
            path = args.out if len(args.models) == 1 else args.out / f"{folder.name}.eval.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            report = {"model": str(folder), "model_sha256": sha256(npz), "benchmark": args.benchmark,
                      "benchmark_manifest_sha256": sha256(benchmark.MANIFESTS / f"{args.benchmark}.json"),
                      "known": args.known, "provenance": provenance(), "panels": res}
            path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
