"""E11 helpers: missing-ending detection, candidate-difference labels, length bins and the registered decision rule."""
from __future__ import annotations

import doctest
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

try:
    import e11
except ImportError:                                   # research extras (torch) not installed
    e11 = None


def result(diff, lo, hi, cer_red, dev11=0.55, dev07=0.52):
    return {"eval": {"contrasts": {"e11_minus_e07": {"top1": {"diff_pp": diff, "ci95_pp": [lo, hi]},
                                                     "cer_reduction": {"diff_pp": cer_red}}}},
            "dev": {"table": {"e11": {"top1_macro": dev11}, "e07": {"top1_macro": dev07}}}}


@unittest.skipIf(e11 is None, "research extras not installed")
class TestE11(unittest.TestCase):
    def test_doctests(self):
        self.assertEqual(doctest.testmod(e11).failed, 0)

    def test_missing_ending(self):
        self.assertTrue(e11.missing_ending("កុំពូ", ["កុំពូទ័រ"]))
        self.assertFalse(e11.missing_ending("កំពូទ័រ", ["កុំពូទ័រ"]))      # vowel change, same skeleton
        self.assertFalse(e11.missing_ending(None, ["ក"]))

    def test_difference(self):
        self.assertEqual(e11.difference(["ក"]), "single")
        self.assertEqual(e11.difference(["កុំពូទ័រ", "កំពូទ័រ"]), "vowels_or_marks")
        self.assertEqual(e11.difference(["កុំពូ", "កុំពូទ័រ"]), "consonants_or_syllables")

    def test_length_bins(self):
        self.assertEqual([e11.length_bin(w) for w in ("cat", "coffee", "computers")], ["le4", "5to8", "ge9"])

    def test_decision_rule(self):
        self.assertEqual(e11.classify(result(4.0, 0.5, 8.0, 0.2))["class"], "useful_real_data_only_candidate")
        self.assertEqual(e11.classify(result(4.0, 0.5, 8.0, -0.2))["class"], "uncertain_alternative")
        self.assertEqual(e11.classify(result(4.0, 0.5, 8.0, 0.2, dev11=0.50))["class"], "uncertain_alternative")
        self.assertEqual(e11.classify(result(1.0, -3.0, 5.0, 0.1))["class"], "uncertain_alternative")
        self.assertEqual(e11.classify(result(-5.0, -9.0, -0.5, 0.0))["class"], "regression_relative_to_e07")
        self.assertEqual(e11.classify(result(-3.5, -7.0, 1.0, -0.4))["class"], "regression_relative_to_e07")


if __name__ == "__main__":
    unittest.main()
