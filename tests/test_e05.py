"""Tests of E05 (`scripts/e05.py`): candidate rules, system construction, coverage, id checks, shared-checkpoint
bookkeeping, family resampling, beam-configuration consistency and the forbidden-path guard."""
from __future__ import annotations

import doctest
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import e05  # noqa: E402
from khmer_translit import reference_data as rd  # noqa: E402


def item(word: str, family: str, accepted: list[str], flags: list[str] | None = None) -> rd.ReferenceItem:
    return rd.ReferenceItem(f"{word}#default", word, word, family, "default", "", "a", accepted, accepted[0],
                            flags or [], "proxy", "final")


def models_cands(ids: list[str]) -> dict[str, dict[str, list[str]]]:
    """Toy ranked candidates for the four E05 models."""
    base = {"G_s0": ["a", "b", "c", "d", "e", "f", "g", "h"], "G_s1": ["a", "c", "x", "y", "z"],
            "G_s2": ["b", "a", "q"], "P_s0": ["p", "a", "r", "s", "t"]}
    return {m: {i: list(v) for i in ids} for m, v in base.items()}


class TestCandidateRules(unittest.TestCase):
    def test_doctests(self):
        self.assertEqual(doctest.testmod(e05, verbose=False).failed, 0)

    def test_nfc_dedup(self):
        decomposed, composed = "e\u0301", "\u00e9"             # NFD vs NFC spellings of one string
        self.assertEqual(e05.dedup([decomposed, composed, "", "ខ", composed]), [composed, "ខ"])

    def test_limits_and_interleaving_deterministic(self):
        a, b = ["1", "2", "3", "4"], ["2", "5", "6", "7"]
        self.assertEqual(e05.interleave(a, b, 8), ["1", "2", "5", "3", "6", "4", "7"])  # 7 distinct, not filled
        self.assertEqual(e05.interleave(b, a, 4), ["2", "1", "5", "6"])
        for k in (1, 2, 4, 8):
            self.assertEqual(e05.interleave(a, b, k), e05.interleave(a, b, k))
            self.assertLessEqual(len(e05.interleave(a, b, k)), k)
        self.assertEqual(e05.top(["x", "x", "y"], 4), ["x", "y"])

    def test_coverage_with_alternative_references(self):
        self.assertEqual(e05.covered(["ក", "ខ"], ["គ", "ខ"]), 1)
        self.assertEqual(e05.covered(["e\u0301"], ["\u00e9"]), 1)    # NFC on both sides
        self.assertEqual(e05.covered(["ក"], ["គ"]), 0)

    def test_empty_output_is_failure(self):
        self.assertEqual(e05.covered(["ក"], []), 0)
        self.assertEqual(e05.covered(["ក"], [""]), 0)
        its = [item("a", "f1", ["ក"])]
        m = e05.system_metrics(its, {"slots": 4, "cands": {"a#default": []}})
        self.assertEqual((m["empty"], m["item_coverage"], m["underfilled"]), (1, 0, 1))

    def test_missing_and_unknown_ids(self):
        with self.assertRaises(ValueError):
            e05.check_ids({"a", "b"}, {"a": ["x"]})
        with self.assertRaises(ValueError):
            e05.check_ids({"a"}, {"a": ["x"], "z": ["y"]})
        e05.check_ids({"a"}, {"a": []})


class TestSystems(unittest.TestCase):
    def setUp(self):
        self.S = e05.build_systems(models_cands(["w"]), ["w"])

    def test_construction(self):
        S = self.S
        self.assertEqual(S["G4[G_s0]"]["cands"]["w"], ["a", "b", "c", "d"])
        self.assertEqual(len(S["G8[G_s0]"]["cands"]["w"]), 8)
        self.assertEqual(S["G+P[G_s0]"]["cands"]["w"], ["a", "b", "c", "d", "p", "r", "s"])  # 'a' shared, not refilled
        self.assertEqual(S["G+G[G_s0,G_s2]"]["cands"]["w"], ["a", "b", "c", "d", "q"])
        self.assertEqual(S["G+P@2[P_s0>G_s0]"]["cands"]["w"], ["p", "a"])
        self.assertEqual(sum(n.startswith("G+G[") for n in S), 3)          # all unordered pairs
        self.assertEqual(sum(n.startswith("G+P[") for n in S), 3)

    def test_shared_checkpoint_bookkeeping(self):
        sh = e05.shared_checkpoints(self.S)
        self.assertEqual(sorted(sh["P_s0"]), ["G+P[G_s0]", "G+P[G_s1]", "G+P[G_s2]"])
        self.assertEqual(sorted(sh["G_s1"]), ["G+G[G_s0,G_s1]", "G+G[G_s1,G_s2]", "G+P[G_s1]", "G4[G_s1]", "G8[G_s1]"])


class TestBootstrap(unittest.TestCase):
    def test_resamples_families_not_items(self):
        # family f0 has 100 items that all differ by 1; 99 one-item families differ by 0
        its = [item(f"a{n}", "f0", ["x"]) for n in range(100)] + [item(f"b{n}", f"g{n:02d}", ["x"]) for n in range(99)]
        on = {i.id: 1.0 if i.family == "f0" else 0.0 for i in its}
        fm = {"T": e05.family_means(its, on), "C": e05.family_means(its, {i.id: 0.0 for i in its})}
        self.assertEqual(len(fm["T"]), 100)
        c = e05.contrast(fm, ["T"], ["C"], e05.boot_indices(100, 2000, 1))
        self.assertAlmostEqual(c["diff_pp"], 1.0)                          # family macro, not 100/199 items
        self.assertLess(c["ci95_pp"][1], 5.0)

    def test_shared_draws_and_seed(self):
        a, b = e05.boot_indices(10, 50, 7), e05.boot_indices(10, 50, 7)
        self.assertTrue(np.array_equal(a, b))
        fm = {"A": np.array([1.0, 0, 0, 1]), "B": np.array([1.0, 0, 0, 1])}
        self.assertEqual(e05.contrast(fm, ["A"], ["B"], e05.boot_indices(4, 100, 3))["ci95_pp"], [0.0, 0.0])


class TestFilesAndGuard(unittest.TestCase):
    def test_forbidden_paths(self):
        for p in ("data/e02/references.jsonl", "data/e2/reserve.json", "data/benchmark/panel.json"):
            with self.assertRaises(PermissionError):
                e05.guard(ROOT / p)
        e05.guard(ROOT / "data/e03/eval_references.jsonl")

    def test_consistent_beam_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(e05, "PRED", Path(d)):
                p = e05.pred_path("eval", 8, "G_s0")
                p.write_text(json.dumps({"id": "a", "model": "G_s0", "width": 4, "candidates": ["x"]}) + "\n")
                with self.assertRaises(ValueError):
                    e05.load_preds("eval", 8, "G_s0", {"a"})
                p.write_text(json.dumps({"id": "a", "model": "G_s0", "width": 8, "candidates": ["x", "x"]}) + "\n")
                self.assertEqual(e05.load_preds("eval", 8, "G_s0", {"a"}), {"a": ["x"]})
                with self.assertRaises(ValueError):
                    e05.load_preds("eval", 8, "G_s0", {"a", "b"})

    def test_verdict_rule(self):
        def c(d, lo):
            return {"diff_pp": d, "ci95_pp": [lo, lo + 4]}
        per = {g: {"vs_G4": c(9, 5), "vs_G8": c(1, -1), "vs_G+G[x]": c(2, -2)} for g in ("G_s0", "G_s1", "G_s2")}
        summ = {"G+P_vs_G8": c(1, -1), "G+P_vs_G+G": c(2, -2)}
        v = e05.verdict({"per_combination": per, "summary": summ})
        self.assertEqual(v["outcome"], "no clear advantage over G8 and G+G")
        summ = {"G+P_vs_G8": c(3, 0.5), "G+P_vs_G+G": c(3, 0.2)}
        per = {g: {"vs_G8": c(3, 1), "vs_G+G[x]": c(3, 1)} for g in ("G_s0",)}
        self.assertEqual(e05.verdict({"per_combination": per, "summary": summ})["outcome"],
                         "consistent advantage over both controls")


if __name__ == "__main__":
    unittest.main()
