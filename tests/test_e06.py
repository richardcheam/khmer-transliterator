"""Tests of E06 (`scripts/e06.py`, `scripts/e06_byt5.py`): byte alignment, limits, split exclusions, deterministic
sampling, prompt freezing, malformed outputs, the gate, blinding and resume safety. Behaviour only, not quality."""
from __future__ import annotations

import doctest
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import e06  # noqa: E402
import e06_byt5 as B  # noqa: E402

CMU = {"table": (("T", "EY1", "B", "AH0", "L"),), "knight": (("N", "AY1", "T"),), "cat": (("K", "AE1", "T"),),
       "read": (("R", "IY1", "D"), ("R", "EH1", "D")), "elephants": (("EH1", "L", "AH0", "F", "AH0", "N", "T", "S"),),
       "holst": (("HH", "OW1", "L", "S", "T"),), "tables": (("T", "EY1", "B", "AH0", "L", "Z"),)}


class TestBytes(unittest.TestCase):
    def test_doctests(self):
        self.assertEqual(doctest.testmod(e06, verbose=False).failed, 0)
        self.assertEqual(doctest.testmod(B, verbose=False).failed, 0)

    def test_alignment_roundtrip(self):
        for k in ("កាហ្វេ", "ឡុងដ៍", "ក"):
            ids = B.encode_tgt(k)
            self.assertEqual(ids[-1], B.EOS)
            self.assertEqual(len(ids) - 1, len(k.encode("utf-8")))     # 3 bytes per Khmer character, no truncation
            self.assertEqual(B.decode_ids(ids), (k, "ok"))
        self.assertEqual(B.encode_src("Coffee")[:-1], [b + 3 for b in b"coffee"])

    def test_malformed_and_empty(self):
        good = B.encode_tgt("ក")
        self.assertEqual(B.decode_ids(good[:2]), (None, "malformed_utf8"))  # a cut multi-byte character
        self.assertEqual(B.decode_ids([B.EOS]), ("", "empty"))
        self.assertEqual(B.decode_ids([0, 0]), ("", "empty"))
        self.assertEqual(e06.orthography(None), "malformed_utf8")
        self.assertEqual(e06.orthography("ក៊ាា"), "double_dependent_vowel")

    def test_resource_limits_reject_not_truncate(self):
        self.assertIsNone(B.fits("coffee", "កាហ្វេ"))
        self.assertEqual(B.fits("a" * 33), "source_too_long")
        self.assertEqual(B.fits("cat", "ក" * 41), "target_too_long")         # 123 bytes > 120
        self.assertEqual(len(B.encode_tgt("ក" * 41)), 124)                 # the encoder itself never truncates

    def test_lr_schedule(self):
        f = [B.lr_factor(k) for k in range(B.RECIPE["max_updates"])]
        self.assertEqual(max(f), 1.0)
        self.assertEqual(f.index(1.0), B.RECIPE["warmup"] - 1)
        self.assertGreater(f[-1], 0)


class TestPool(unittest.TestCase):
    def cands(self):
        return [(w, "lower") for w in ("table", "tables", "knight", "cat", "read", "elephants")] + [("holst", "name")]

    def test_split_exclusion(self):
        hist = e06.stem_set(["tabled", "share"])                  # "tabled" stems include "table"
        rows, excl = e06.select_pool(self.cands(), CMU, hist, per=3, seed="t")
        words = {r["word"] for r in rows}
        self.assertFalse(words & {"table", "tables"})
        self.assertGreaterEqual(excl["historical_family"], 2)

    def test_deterministic_and_order_independent(self):
        a = e06.select_pool(self.cands(), CMU, set(), per=3, seed="t")[0]
        c = self.cands()
        random.Random(5).shuffle(c)
        b = e06.select_pool(c, CMU, set(), per=3, seed="t")[0]
        self.assertEqual(a, b)
        self.assertNotEqual([r["word"] for r in a], [r["word"] for r in e06.select_pool(self.cands(), CMU, set(),
                                                                                         per=3, seed="u")[0]])

    def test_one_word_per_family_and_categories(self):
        rows, _ = e06.select_pool(self.cands(), CMU, set(), per=3, seed="t")
        self.assertFalse({"table", "tables"} <= {r["word"] for r in rows})
        cat = {r["word"]: r["category"] for r in rows}
        self.assertEqual(cat.get("knight"), "difficult")
        self.assertEqual(cat.get("read"), "ambiguous")             # two readings that differ beyond stress
        self.assertEqual(cat.get("holst"), "name")
        self.assertEqual(e06.categorise("speciously", "lower", CMU), "dictionary_miss")
        self.assertEqual(e06.categorise("cruel", "lower", {"cruel": (("K", "R", "UW1", "AH0", "L"),
                                                                     ("K", "R", "UW1", "L"))}), "ambiguous")

    def test_forbidden_paths(self):
        for p in ("data/e02/references.jsonl", "data/e2/reserve.json", "data/benchmark/v1/dev41.tsv"):
            with self.assertRaises(PermissionError):
                e06.guard(ROOT / p)
        self.assertTrue(e06.forbidden(ROOT / "experiments/E02-english-phonetic/results.json"))


class TestPromptAndQueue(unittest.TestCase):
    def test_prompt_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "prompt.txt"
            with mock.patch.object(e06, "PROMPT", p):
                p.write_text(e06.ROUTE_A_PROMPT, encoding="utf-8")
                txt = e06.render_prompt("coffee", None)
                self.assertIn("not available", txt)
                self.assertIn("not translation", txt)
                p.write_text(e06.ROUTE_A_PROMPT + "be creative", encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    e06.render_prompt("coffee", "K AA1 F IY0")

    def test_queue_blinded_and_deduplicated(self):
        pool = [{"id": "p01", "word": "cat", "category": "short", "track": "primary", "default_phones": "K AE1 T",
                 "all_readings": ["K AE1 T"]}]
        preds = {"byt5_lora": {"p01": ["ខេត", "ខាត"]}, "G_s0": {"p01": ["ខាត", "កាត"]}, "default_e02": {"p01": []}}
        rows, key = e06.build_queue(pool, preds)
        self.assertEqual(sorted(r["candidate"] for r in rows), sorted(["ខេត", "ខាត", "កាត"]))
        self.assertTrue(all(set(r) == {"item", "english", "pronunciation", "candidate_id", "candidate", "judgment",
                                       "comment"} for r in rows))
        self.assertNotIn("byt5", json.dumps(rows))
        k = key["items"]["w001"]["candidates"]
        self.assertEqual(sorted(sum((c["origins"] for c in k.values()), [])),
                         sorted(["byt5_lora@1", "byt5_lora@2", "G_s0@1", "G_s0@2"]))
        self.assertEqual(rows, e06.build_queue(pool, preds)[0])            # deterministic order


class TestGate(unittest.TestCase):
    def rows(self, n, ref, base, cats=("ordinary", "name", "short", "long")):
        return [{"category": cats[i % len(cats)], "ref": ref(i), "base": base(i)} for i in range(n)]

    def test_insufficient_sample(self):
        self.assertEqual(e06.gate_decision(self.rows(20, lambda i: 1, lambda i: 0))["status"],
                         "teacher_advantage_unresolved")

    def test_pass_and_fail(self):
        good = e06.gate_decision(self.rows(48, lambda i: 1, lambda i: int(i % 3 == 0)), n_boot=2000)
        self.assertEqual(good["status"], "teacher_passes")
        bad = e06.gate_decision(self.rows(48, lambda i: int(i % 3 == 0), lambda i: 1), n_boot=2000)
        self.assertEqual(bad["status"], "teacher_not_better")
        tie = e06.gate_decision(self.rows(48, lambda i: i % 2, lambda i: (i + 1) % 2), n_boot=2000)
        self.assertEqual(tie["status"], "teacher_advantage_unresolved")

    def test_concentrated_advantage_does_not_pass(self):
        rows = self.rows(48, lambda i: 0, lambda i: 0)
        for r in rows:
            if r["category"] == "ordinary":
                r["ref"] = 1
        self.assertEqual(e06.gate_decision(rows, n_boot=2000)["status"], "teacher_advantage_unresolved")

    def test_no_judgments_is_unresolved(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with mock.patch.multiple(e06, ROOT=d, REPORTS=d / "r", STATE=d / "s", JUDGMENTS=d / "none.tsv"):
                (d / "r").mkdir()
                out = e06.gate()
                self.assertEqual(out["status"], "teacher_advantage_unresolved")
                self.assertTrue(e06.is_done("gate"))
                (d / "r" / "gate.json").write_text("{}")                     # changed output → not done (resume safety)
                self.assertFalse(e06.is_done("gate"))


class TestDiscovery(unittest.TestCase):
    def test_scan_doc(self):
        text = "ការប្រកួត Holst Cup ហូលស្ទ ... " + "x" * 200 + " ហូល"
        r = e06.scan_doc(text, {"holst": ["ហូលស្ទ", "ហូល"], "cat": ["ខាត"]})
        self.assertEqual(set(r), {"holst"})
        self.assertTrue(r["holst"]["near"])
        self.assertIn("ហូលស្ទ", r["holst"]["candidates_in_doc"])
        self.assertEqual(e06.scan_doc("no latin here ខាត", {"cat": ["ខាត"]}), {})


if __name__ == "__main__":
    unittest.main()
