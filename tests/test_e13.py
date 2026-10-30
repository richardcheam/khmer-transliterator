"""E13: pronunciation lookup and misses, input limits, split exclusions, mixed sampling, explicit review responses,
missing judgments, the selection rule and safe resume."""
from __future__ import annotations

import doctest
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

try:
    import e07
    import e09
    import e11
    # e12/e13 rebind e07/e09/e11 bookkeeping globals on import; restore them so later test modules see the originals
    saved = {m: {k: getattr(m, k) for k in ("LEDGER", "EXPERIMENT", "STATE", "LIMITS", "STATE_FILE") if hasattr(m, k)}
             for m in (e07, e09, e11)}
    saved_checkpoints = dict(e09.CHECKPOINTS)
    import e13
    for m, attrs in saved.items():
        for k, v in attrs.items():
            setattr(m, k, v)
    e09.CHECKPOINTS.clear()
    e09.CHECKPOINTS.update(saved_checkpoints)
except ImportError:                                   # research extras (torch) not installed
    e13 = None


@unittest.skipIf(e13 is None, "research extras not installed")
class TestE13(unittest.TestCase):
    def test_doctests(self):
        self.assertEqual(doctest.testmod(e13).failed, 0)

    def test_pronunciation_variants_and_misses(self):
        self.assertEqual(e13.pron("yacht")["phones"], "Y AA1 T")
        self.assertIsNone(e13.pron("qzxvbn")["phones"])               # missing → no invented phones
        self.assertTrue(e13.source("qzxvbn", None).endswith("<no_pron>"))
        rec = e13.pron("record")                                       # heteronym: default first reading, flagged
        self.assertTrue(rec["ambiguous"])
        self.assertEqual(rec["phones"], " ".join(e13.ps.cmudict()[0]["record"][0]))

    def test_input_limits(self):
        e13.source("supercalifragilistic", "S UW2 P ER0 K AE2 L AH0 F R AE1 JH AH0 L IH2 S T IH0 K")
        with self.assertRaises(ValueError):
            e13.source("a" * 200, "AH0")
        with self.assertRaises(ValueError):
            e13.check_target("ក" * 50)                                  # 150 bytes > 120
        with self.assertRaises(ValueError):
            e13.check_target("")

    def test_split_exclusions(self):
        coll, test = e13.frozen_rows(e13.COLL), e13.frozen_rows(e13.TEST)
        self.assertEqual(Counter((r["split"], r["category"]) for r in coll),
                         Counter({(s, c): n for s, d in e13.COLL_PLAN.items() for c, n in d.items()}))
        self.assertEqual(Counter(r["category"] for r in test), Counter({c: 40 for c in e13.CATEGORIES}))
        keys = lambda rows: {k for r in rows for k in e13.e12.family_keys(r["word"])}  # noqa: E731
        self.assertFalse(keys(coll) & keys(test))
        self.assertFalse(keys([r for r in coll if r["split"] == "train"]) & keys([r for r in coll if r["split"] == "dev"]))
        prot = e13.e08.protected()
        _, arm = e11.arm_rows()
        train = {r["word"] for r in arm}
        stems = e13.e06.stem_set(train)
        e11_review = {it["word"] for it in json.loads((e13.E11_REVIEW / "review_batch.json").read_text())["items"]}
        e12_test = {r["word"] for r in e11.jl(e13.E12_TEST / "input_manifest.jsonl")}
        for r in coll + test:
            w = r["word"]
            self.assertFalse(e13.e08.is_protected(w, prot), w)
            self.assertNotIn(w, train | e11_review | e12_test)
            self.assertFalse(e13.rd.related(w, stems), w)
            self.assertEqual(e13.pron(w)["phones"], r["pronunciation"])   # frozen reading = lookup
            self.assertFalse(e13.e06.is_ambiguous(e13.ps.cmudict()[0][w]))

    def test_mixed_sampling(self):
        r_rows = [(f"r{i}", "ក") for i in range(50)]
        reviewed = [{"word": "abc", "pronunciation": "AE1 B K", "accepted": ["ខ", "គ"]},
                    {"word": "def", "pronunciation": "D EH1 F", "accepted": ["ឃ"]}]
        rows, lay = e13.row_sources(r_rows, reviewed)
        self.assertEqual(len(rows), 2 * 50 + 2 * 3)
        order = e13.make_order(lay)
        blk = order(0)
        nR, nA = lay["nR"], lay["nA"]
        self.assertEqual(len(blk), 3200)
        self.assertEqual(sum(i >= 2 * nR for i in blk), 320)                 # exactly 10% reviewed
        self.assertEqual(sum((nR <= i < 2 * nR) or i >= 2 * nR + nA for i in blk), 320)   # exactly 10% masked
        self.assertEqual(order(0), blk)                                     # deterministic (safe resume)
        for i in range(nR):                                                 # masking never changes the target
            self.assertEqual(rows[i][1], rows[i + nR][1])
            self.assertTrue(rows[i + nR][0].endswith("<no_pron>"))
        fam = Counter("abc" if (i - 2 * nR) % nA < 2 else "def" for i in blk if i >= 2 * nR)
        self.assertLess(abs(fam["abc"] - fam["def"]), 80)                  # family-uniform, not alternative-uniform
        r_seen = Counter(i % nR for i in blk if i < 2 * nR)
        self.assertLessEqual(max(r_seen.values()) - min(r_seen.values()), 1)   # R's permutation convention

    def test_explicit_review_responses(self):
        items = [{"item_id": "c1", "candidates": [{"label": "A", "text": "ក"}, {"label": "B", "text": "ខ"}]}]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.assertEqual(e13.validate(d, items)["status"], "awaiting")
            head = "item_id\tword\tintended_reading\tlabel\tcandidate\tjudgment\n"
            (d / "judgments_1.tsv").write_text(head + "c1\tx\tX\tA\tក\tacceptable\nc1\tx\tX\tB\tខ\t\n", encoding="utf-8")
            v = e13.validate(d, items)
            self.assertEqual(v["status"], "incomplete")                     # blank = missing, never accepted
            (d / "judgments_1.tsv").write_text(head + "c1\tx\tX\tA\tក\tacceptable\nc1\tx\tX\tB\tខ\tok\n", encoding="utf-8")
            self.assertEqual(e13.validate(d, items)["status"], "invalid")
            (d / "judgments_1.tsv").write_text(head + "c1\tx\tX\tA\tក\tacceptable\nc1\tx\tX\tB\tខ\tuncertain\n",
                                               encoding="utf-8")
            self.assertEqual(e13.validate(d, items)["status"], "provenance_missing")
            (d / "judgments.provenance.json").write_text("{}")
            self.assertEqual(e13.validate(d, items)["status"], "complete")

    def test_missing_judgments_and_datasets(self):
        man = [{"id": "c1", "word": "w", "category": "irregular", "split": "train", "pronunciation": "W",
                "pron_id": "cmudict:w", "pattern": "other"},
               {"id": "c2", "word": "v", "category": "names", "split": "dev", "pronunciation": "V",
                "pron_id": "cmudict:v", "pattern": "other"}]
        its = [{"item_id": "c1", "candidates": [{"label": "A", "text": "ក"}, {"label": "B", "text": "ខ"}]},
               {"item_id": "c2", "candidates": [{"label": "A", "text": "គ"}]}]
        ds = e13.datasets_from(man, its, {("c1", "A"): "not_acceptable", ("c1", "B"): "acceptable",
                                          ("c2", "A"): "not_acceptable"})
        self.assertEqual(ds["train"][0]["accepted"], ["ខ"])                  # rejected never becomes a target
        self.assertEqual(ds["dev"][0]["accepted"], [])                       # none acceptable → unlabelled
        self.assertEqual(ds["dev"][0]["label_status"], "none_acceptable")
        self.assertFalse(e13.feasibility(ds)["pass"])
        cov = e13.e12.judgment_coverage({"c1": "ឃ"}, {"c1": {"ខ": "acceptable"}})
        self.assertEqual(cov, {"judged_items": 1, "unknown": 1})              # a novel spelling is unknown

    def test_selection_rule(self):
        hist = [{"update": 400, "cer": 0.30, "exact": 0.40}, {"update": 800, "cer": 0.25, "exact": 0.30},
                {"update": 1200, "cer": 0.25, "exact": 0.30}]
        best = max(hist, key=lambda r: (e13.selection_key(r), -r["update"]))
        self.assertEqual(best["update"], 800)                                # lower CER, then the earlier update

    def test_safe_resume(self):
        import torch
        ident = {"cfg": "a", "data": "b", "base": {"x": "c"}}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self.assertEqual(e13.e12.check_resume(out, ident, [], e13.RUN), "fresh")
            with self.assertRaises(RuntimeError):
                e13.e12.check_resume(out, ident, [e13.RUN], e13.RUN)
            torch.save({"ident": ident}, out / "last.pt")
            self.assertEqual(e13.e12.check_resume(out, ident, [e13.RUN], e13.RUN), "resume")
            with self.assertRaises(RuntimeError):
                e13.e12.check_resume(out, {**ident, "data": "z"}, [e13.RUN], e13.RUN)

    def test_decision_and_groups(self):
        man = [{"id": str(i), "category": e13.CATEGORIES[i % 3]} for i in range(30)]
        a = {str(i): "acceptable" for i in range(30)}
        b = {str(i): "acceptable" if i % 3 else "not_acceptable" for i in range(30)}
        r = e13.analyse({"e13": a, "e11": b}, man)
        self.assertEqual(r["decision"]["class"], "promising_improvement")
        self.assertEqual(r["secondary_groups"]["ordinary"]["paired"]["a_only"], 10)
        self.assertEqual(e13.analyse({"e13": b, "e11": a}, man)["decision"]["trade_off_groups"], ["ordinary"])


if __name__ == "__main__":
    unittest.main()
