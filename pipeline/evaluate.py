#!/usr/bin/env python3
"""Evaluate the small model and the ByT5 adapter on the published test sets, next to the released models' numbers.

    python pipeline/evaluate.py                                     # the released models (downloaded, pinned)
    python pipeline/evaluate.py --small runs/small_s0/model.npz --byt5 runs/byt5_s0/adapter
    python pipeline/evaluate.py --skip-byt5                         # small model only (NumPy, a few seconds)

Measures (what each set is and what it was used for: `test-sets/README.md`):
- **Small model**, beam search with 4 candidates per word: *top-1* = the first candidate is an accepted spelling;
  *top-4* = one of the four is. Each is averaged within each lexical family, then over families. Sets: `final` (the
  165 words without an ambiguity flag, reference spellings checked by the author), the 26 flagged `final` words,
  `proxy` (400 words, LLM-suggested spellings spot-checked), and `dev` (86 words, used to choose the epoch).
- **ByT5 adapter**, greedy spelling: top-1 (the spelling is an accepted one, averaged over families) and the mean
  character error rate against the closest accepted spelling. Sets: `eval` (the 228 mined words without an ambiguity
  flag) and `dev` (253 words, used to choose the update).

All comparisons are exact agreement with reference spellings, which undercounts spellings that sound right but are
written differently. The released small model reproduces its numbers exactly; a retrained model lands near them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "test-sets"
FLAGGED = {"pronunciation_ambiguous", "possible_translation"}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def macro(items: list[dict], values: dict[str, float]) -> float:
    fam: dict[str, list[float]] = {}
    for it in items:
        fam.setdefault(it["family"], []).append(values[it["id"]])
    return mean(mean(v) for v in fam.values()) if fam else 0.0


def score_small(tl, items: list[dict]) -> dict:
    res = tl.predict_many([it["word"] for it in items], top_k=4)
    top1, top4 = {}, {}
    for it, r in zip(items, res):
        acc = {nfc(a) for a in it["accepted"]}
        cands = [nfc(c.text) for c in r.candidates]
        top1[it["id"]] = int(bool(cands) and cands[0] in acc)
        top4[it["id"]] = int(any(c in acc for c in cands[:4]))
    return {"top1_pct": 100 * macro(items, top1), "top4_pct": 100 * macro(items, top4), "words": len(items)}


def score_byt5(tl, items: list[dict]) -> dict:
    hit, cer = {}, []
    for it in items:
        t = tl.predict(it["word"]).text
        t = nfc(t) if t else None
        hit[it["id"]] = int(t is not None and (t == nfc(it["preferred"]) or any(t == nfc(a) for a in it["accepted"])))
        cer.append(min(edit(t or "", nfc(a)) / max(1, len(a)) for a in it["accepted"]))
    return {"top1_pct": 100 * macro(items, hit), "mean_cer": mean(cer), "words": len(items)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--small", help="model.npz (default: the released small model)")
    ap.add_argument("--byt5", help="adapter folder (default: the released adapter)")
    ap.add_argument("--base", help="local google/byt5-small folder (default: download the pinned revision)")
    ap.add_argument("--skip-byt5", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    expected = json.loads((ROOT / "pipeline" / "expected_results.json").read_text(encoding="utf-8"))
    res: dict = {}
    t0 = time.time()
    from khmer_translit.english import Transliterator
    tl = Transliterator.from_pretrained(a.small)
    fin = jsonl(TESTS / "small_final.jsonl")
    res["small"] = {"final": score_small(tl, [i for i in fin if not FLAGGED & set(i["flags"])]),
                    "final_flagged": score_small(tl, [i for i in fin if FLAGGED & set(i["flags"])]),
                    "proxy": score_small(tl, jsonl(TESTS / "small_proxy.jsonl")),
                    "dev": score_small(tl, [i for i in jsonl(TESTS / "small_dev.jsonl") if not FLAGGED & set(i["flags"])])}
    if not a.skip_byt5:
        from khmer_translit.byt5 import ByT5Transliterator
        bt = ByT5Transliterator.from_pretrained(a.byt5, base=a.base)
        prim = lambda f: [i for i in jsonl(f) if not FLAGGED & set(i["flags"])]  # noqa: E731
        res["byt5"] = {"eval": score_byt5(bt, prim(TESTS / "byt5_eval.jsonl")),
                       "dev": score_byt5(bt, prim(ROOT / "training-data" / "byt5" / "dev.jsonl"))}
    res["minutes"] = round((time.time() - t0) / 60, 1)

    def show(label, got, key, places=1):
        exp = expected.get(key)
        line = f"  {label:<46} {got:7.{places}f}"
        if exp is not None:
            line += f"   released {exp:7.{places}f}   difference {got - exp:+.{places}f}"
        print(line)
    s = res["small"]
    print(f"small model: {a.small or 'released'}")
    show(f"final ({s['final']['words']} words): top-1 %", s["final"]["top1_pct"], "small_final_top1_pct")
    show("final: top-4 %", s["final"]["top4_pct"], "small_final_top4_pct")
    show(f"final, flagged ({s['final_flagged']['words']}): top-1 %", s["final_flagged"]["top1_pct"], "small_flagged_top1_pct")
    show(f"proxy ({s['proxy']['words']}): top-1 %", s["proxy"]["top1_pct"], "small_proxy_top1_pct")
    show("proxy: top-4 %", s["proxy"]["top4_pct"], "small_proxy_top4_pct")
    show(f"dev ({s['dev']['words']}, used for selection): top-1 %", s["dev"]["top1_pct"], "small_dev_top1_pct")
    if "byt5" in res:
        b = res["byt5"]
        print(f"ByT5 adapter: {a.byt5 or 'released'}")
        show(f"eval ({b['eval']['words']} words): top-1 %", b["eval"]["top1_pct"], "byt5_eval_top1_pct")
        show("eval: mean character error", b["eval"]["mean_cer"], "byt5_eval_cer", 4)
        show(f"dev ({b['dev']['words']}, used for selection): top-1 %", b["dev"]["top1_pct"], "byt5_dev_top1_pct")
    print(f"  ({res['minutes']} minutes)")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
