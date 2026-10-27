#!/usr/bin/env python3
"""Experiment E02: runtime guards of a packaged model (cold load, warm latency, peak memory), on dev words only.

    .venv/bin/python scripts/english_phonetic_bench.py models/e02/letters_s0/model.npz [more models ...]

Each model is measured in a **fresh process** (`--child`), so cold load and peak RSS are not shared:
- cold load: first `Transliterator.from_pretrained` (+ CMUdict parse and hash for phone models);
- pronunciation lookup (phone models only): `get_pronunciation` per word after the dictionary is loaded;
- warm batch-1 latency: 3 passes over the 103 dev words, p50/p95 per word, split by dictionary hit/miss (the items' `dictionary_miss` flag);
- warm batch-16 throughput: per-word time;
- peak RSS of the process (`ru_maxrss`).

Guards (PLAN.md): cold load ≤ 30 s, warm batch-1 p95 ≤ 1 s/word, peak RSS ≤ 1.5 GiB, and for a treatment
p95 ≤ 2× the letters baseline. Output: `experiments/E02-english-phonetic/reports/runtime.json`.
"""
from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "experiments" / "E02-english-phonetic" / "reports" / "runtime.json"


def pct(xs: list[float], q: float) -> float:
    """Nearest-rank percentile."""
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * len(xs) + 0.5)) - 1))]


def child(model: str) -> dict:
    """Measure one model in this process."""
    from khmer_translit import phonetic_source as ps
    from khmer_translit import reference_data as rd
    t0 = time.perf_counter()
    from khmer_translit.english import Transliterator
    tl = Transliterator.from_pretrained(model)
    if tl.mode == "phones":
        ps.get_pronunciation("warmup")
    cold = time.perf_counter() - t0
    items = rd.load_reference_items(ROOT / "data/e02/dev_references.jsonl")
    words = [i.word for i in items]
    hit = {i.word: "dictionary_miss" not in i.flags for i in items}
    lookup = None                                     # a letters model never looks a word up
    if tl.mode == "phones":
        t = time.perf_counter()
        for w in words:
            ps.get_pronunciation(w)
        lookup = (time.perf_counter() - t) / len(words)
    lat: dict[str, list[float]] = {"hit": [], "miss": []}
    for _ in range(3):
        for w in words:
            t = time.perf_counter()
            tl.predict(w, top_k=1)
            lat["hit" if hit[w] else "miss"].append(time.perf_counter() - t)
    allv = lat["hit"] + lat["miss"]
    t = time.perf_counter()
    for _ in range(3):
        for i in range(0, len(words), 16):
            tl.predict_many(words[i:i + 16])
    b16 = (time.perf_counter() - t) / (3 * len(words))
    return {"model": model, "mode": tl.mode, "cold_load_s": cold, "pron_lookup_s_per_word": lookup,
            "batch1_p50_s": pct(allv, 0.5), "batch1_p95_s": pct(allv, 0.95),
            "batch1_p95_hit_s": pct(lat["hit"], 0.95) if lat["hit"] else None,
            "batch1_p95_miss_s": pct(lat["miss"], 0.95) if lat["miss"] else None,
            "batch16_s_per_word": b16, "words": len(words), "passes": 3,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def main() -> int:
    """Measure each model in a fresh process and apply the guards."""
    if sys.argv[1:2] == ["--child"]:
        print(json.dumps(child(sys.argv[2])))
        return 0
    rows = []
    for m in sys.argv[1:]:
        res = subprocess.run([sys.executable, __file__, "--child", m], capture_output=True, text=True, check=True,
                             env={**os.environ, "OMP_NUM_THREADS": "6"})
        rows.append(json.loads(res.stdout.strip().splitlines()[-1]))
    base = next((r for r in rows if r["mode"] == "letters"), None)
    for r in rows:
        r["guards"] = {"cold_load_le_30s": r["cold_load_s"] <= 30, "batch1_p95_le_1s": r["batch1_p95_s"] <= 1.0,
                       "peak_rss_le_1_5gib": r["peak_rss_mib"] <= 1536,
                       "p95_le_2x_letters": (r["batch1_p95_s"] <= 2 * base["batch1_p95_s"]) if base else None}
        r["guards_ok"] = all(v is not False for v in r["guards"].values())
    out = {"machine": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "python": sys.version.split()[0],
                       "threads": "OMP_NUM_THREADS=6"}, "measured": time.strftime("%Y-%m-%d %H:%M"), "models": rows}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
