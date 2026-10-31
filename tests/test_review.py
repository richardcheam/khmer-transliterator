"""Review page: blinding (the page never sees the origin key), explicit answers only, resume, and saved files."""
from __future__ import annotations

import csv
import doctest
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import review  # noqa: E402


class TestReview(unittest.TestCase):
    def test_doctests(self):
        self.assertEqual(doctest.testmod(review).failed, 0)

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "batch_1.tsv").write_text("\t".join(review.FIELDS) + "\n"
                                           "t1\tx\tX\tA\tក\t\nt1\tx\tX\tB\tខ\t\n", encoding="utf-8")
            (d / "origin_key.json").write_text('{"t1": {"A": ["model_one"], "B": ["model_two"]}}')
            store = review.Store(d, "tester")
            self.assertNotIn("model_one", json.dumps(store.state()))          # the page never sees the origin key
            store.save("t1", "A", "acceptable")
            with self.assertRaises(ValueError):                                # only the three explicit answers
                store.save("t1", "B", "ok")
            review.Store(d, "tester").save("t1", "B", "uncertain")             # a restart keeps A
            with open(d / "judgments_1.tsv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual([r["judgment"] for r in rows], ["acceptable", "uncertain"])
            self.assertEqual(len((d / "review_log.jsonl").read_text().splitlines()), 2)
            self.assertFalse(json.loads((d / "judgments.provenance.json").read_text())["independent"])


if __name__ == "__main__":
    unittest.main()
