"""E12: schedule boundaries, safe resume, protected-family exclusion of the frozen test inputs, prediction alignment,
unknown-judgment handling, review blinding and the predeclared test scoring."""
from __future__ import annotations

import doctest
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

try:
    import e07
    import e09
    import e11
    # e11/e12 rebind e07/e09/e11 bookkeeping globals on import; restore them so later test modules see the originals
    saved = {m: {k: getattr(m, k) for k in ("LEDGER", "EXPERIMENT", "STATE", "LIMITS", "STATE_FILE") if hasattr(m, k)}
             for m in (e07, e09, e11)}
    saved_checkpoints = dict(e09.CHECKPOINTS)
    import e12
    for m, attrs in saved.items():
        for k, v in attrs.items():
            setattr(m, k, v)
    e09.CHECKPOINTS.clear()
    e09.CHECKPOINTS.update(saved_checkpoints)
except ImportError:                                   # research extras (torch) not installed
    e12 = None


@unittest.skipIf(e12 is None, "research extras not installed")
class TestE12(unittest.TestCase):
    def test_doctests(self):
        self.assertEqual(doctest.testmod(e12).failed, 0)

    def test_schedule_boundaries(self):
        cfg = e12.recipe()
        self.assertEqual((cfg["max_updates"], cfg["warmup"], cfg["schedule"], cfg["lr"]), (6000, 100, "linear", 1e-3))
        f = lambda u: e07.lr_factor(u - 1, cfg)       # noqa: E731  (1-based update → factor applied at it)
        self.assertAlmostEqual(f(1), 0.01)
        self.assertEqual(f(100), 1.0)
        self.assertEqual(f(101), 1.0)
        self.assertGreater(f(3000), 0.5)                # E11's scheduler had reached 0 here; E12's has not
        self.assertAlmostEqual(f(6000), 1 / 5900)
        self.assertEqual(e07.lr_factor(6000, cfg), 0.0)
        self.assertEqual(e12.recipe_values(cfg), e12.recipe_values(e11.recipe()))
        self.assertEqual(cfg["eval_at"], [1200, 2400, 3000, 3600, 4800, 6000])

    def test_selection_rule(self):
        hist = [{"key": (0.53, -0.157), "update": 3000}, {"key": (0.55, -0.160), "update": 3600},
                {"key": (0.52, -0.157), "update": 4800}]
        best = max(hist, key=lambda r: (e12.selection_key(r), -r["update"]))
        self.assertEqual(best["update"], 3000)        # lower CER wins; equal CER → higher exact, then earlier

    def test_safe_resume(self):
        import torch
        ident = {"cfg": "a", "data": "b", "base": {"x": "c"}}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self.assertEqual(e12.check_resume(out, ident, []), "fresh")
            with self.assertRaises(RuntimeError):        # started, but the resumable state is gone
                e12.check_resume(out, ident, ["long_s0"])
            torch.save({"ident": ident}, out / "last.pt")
            self.assertEqual(e12.check_resume(out, ident, ["long_s0"]), "resume")
            with self.assertRaises(RuntimeError):        # another config
                e12.check_resume(out, {**ident, "cfg": "z"}, ["long_s0"])
            (out / "done.json").write_text("{}")
            self.assertEqual(e12.check_resume(out, ident, ["long_s0"]), "done")

    def test_frozen_test_inputs_are_fresh(self):
        rows = e12.manifest_rows()
        self.assertEqual(len(rows), 120)
        self.assertEqual(len({r["word"] for r in rows}), 120)
        prot = e12.e08.protected()
        opt, arm = e11.arm_rows()
        train = {r["word"] for r in arm}
        dev_eval = {i.word for s in ("dev", "eval") for i in e12.e06.items(s)}
        keys = [k for r in rows for k in e12.family_keys(r["word"])]
        self.assertEqual(len(keys), len(set(keys)))   # one item per family
        for r in rows:
            w = r["word"]
            self.assertFalse(e12.e08.is_protected(w, prot), w)
            self.assertNotIn(w, train)
            self.assertNotIn(w, dev_eval)
            self.assertFalse(e12.rd.related(w, e12.e06.stem_set(train)), w)

    def test_alignment(self):
        rows = [("t1", "cat"), ("t2", "dog")]
        with self.assertRaises(RuntimeError):
            e12.aligned(rows, [{"id": "t2", "word": "dog", "cands": []}, {"id": "t1", "word": "cat", "cands": []}])

    def test_unknown_judgments_are_not_wrong(self):
        cov = e12.judgment_coverage({"a": "ក", "b": "ខ", "c": None},
                                    {"a": {"ក": "not_acceptable"}, "b": {"គ": "acceptable"}, "c": {"ក": "acceptable"}})
        self.assertEqual(cov, {"judged_items": 3, "not_acceptable": 1, "unknown": 2})

    def test_blinding(self):
        man = [{"id": "t1", "word": "cat", "pronunciation": "K AE1 T"},
               {"id": "t2", "word": "dog", "pronunciation": "D AO1 G"}]
        items, key = e12.build_items(man, {"e12": {"t1": "ខាត", "t2": "ដក"}, "e11": {"t1": "ខាត", "t2": "ដុក"}})
        blob = json.dumps(items, ensure_ascii=False)
        self.assertNotIn("e12", blob)
        self.assertNotIn("e11", blob)
        self.assertEqual(len(items[0]["candidates"]), 1)          # identical outputs shown once
        self.assertEqual(len(items[1]["candidates"]), 2)
        self.assertEqual(sorted(o for v in key["t2"]["origins"].values() for o in v), ["e11", "e12"])

    def test_scoring(self):
        a = {str(i): v for i, v in enumerate(["acceptable"] * 6 + ["not_acceptable"] * 2 + ["uncertain"] * 2)}
        b = {str(i): v for i, v in enumerate(["acceptable"] * 3 + ["not_acceptable"] * 5 + ["acceptable"] * 2)}
        r = e12.analyse_test({"e12": a, "e11": b})
        self.assertEqual(r["n"], 10)
        self.assertEqual(r["confirmed_acceptable_share"], {"e12": 0.6, "e11": 0.5})
        self.assertEqual(r["primary"]["diff_pp"], 10.0)
        self.assertEqual(r["paired"], {"both": 3, "a_only": 3, "b_only": 2, "neither": 2})
        self.assertEqual(r["exact_test"]["discordant_clear"], [3, 0])   # the two uncertain items are not in the test
        self.assertEqual(r["sensitivity"]["uncertain_as_acceptable"]["diff_pp"], 30.0)
        self.assertEqual(r["sensitivity"]["uncertain_excluded"]["denominator"], 8)
        lo, hi = r["primary"]["ci95_pp"]
        self.assertLessEqual(lo, 10.0)
        self.assertGreaterEqual(hi, 10.0)

    def test_importer_requires_explicit_answers(self):
        items = [{"item_id": "t1", "candidates": [{"label": "A", "text": "ក"}, {"label": "B", "text": "ខ"}]}]
        with tempfile.TemporaryDirectory() as d:
            old = e12.TEST_DATA
            e12.TEST_DATA = Path(d)
            try:
                (Path(d) / "judgments_1.tsv").write_text(
                    "item_id\tword\tintended_reading\tlabel\tcandidate\tjudgment\n"
                    "t1\tx\tX\tA\tក\tacceptable\nt1\tx\tX\tB\tខ\t\n", encoding="utf-8")
                seen, prob = e12.read_test_judgments(items)
                self.assertEqual(prob["missing"], ["t1:B"])     # a blank is missing, never accepted
                (Path(d) / "judgments_1.tsv").write_text(
                    "item_id\tword\tintended_reading\tlabel\tcandidate\tjudgment\n"
                    "t1\tx\tX\tA\tគ\tacceptable\nt1\tx\tX\tB\tខ\tmaybe\n", encoding="utf-8")
                _, prob = e12.read_test_judgments(items)
                self.assertEqual(prob["edited_candidate"], ["t1:A"])
                self.assertEqual(prob["invalid_value"], ["t1:B:maybe"])
            finally:
                e12.TEST_DATA = old

    def test_review_page_round_trip(self):
        import e12_review
        self.assertEqual(e12_review.JUDGMENTS, e12.e10.JUDGMENTS)
        self.assertEqual(doctest.testmod(e12_review).failed, 0)
        items = [{"item_id": "t1", "candidates": [{"label": "A", "text": "ក"}, {"label": "B", "text": "ខ"}]}]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "batch_1.tsv").write_text("\t".join(e12_review.FIELDS) + "\n"
                                           "t1\tx\tX\tA\tក\t\nt1\tx\tX\tB\tខ\t\n", encoding="utf-8")
            (d / "origin_key.json").write_text('{"t1": {"origins": {"A": ["e12"], "B": ["e11"]}}}')
            store = e12_review.Store(d, "tester")
            self.assertNotIn("e12", json.dumps(store.state()))        # the page never sees the origin key
            store.save("t1", "A", "acceptable")
            with self.assertRaises(ValueError):
                store.save("t1", "B", "ok")
            old = e12.TEST_DATA
            e12.TEST_DATA = d
            try:
                seen, prob = e12.read_test_judgments(items)
                self.assertEqual(prob["missing"], ["t1:B"])           # unanswered stays missing
                e12_review.Store(d, "tester").save("t1", "B", "uncertain")   # resume keeps A
                seen, prob = e12.read_test_judgments(items)
                self.assertEqual(seen, {("t1", "A"): "acceptable", ("t1", "B"): "uncertain"})
                self.assertEqual(sum(map(len, prob.values())), 0)
            finally:
                e12.TEST_DATA = old
            self.assertFalse(json.loads((d / "judgments.provenance.json").read_text())["independent"])


if __name__ == "__main__":
    unittest.main()
