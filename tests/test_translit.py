"""Unit tests for the learned transliterator's NumPy parts, routing and configuration. Run:
`.venv/bin/python -m unittest tests.test_translit`."""
import unittest

import numpy as np

from khmer_translit import translit


class TranslitTest(unittest.TestCase):
    def test_erf_matches_known_values(self):
        self.assertAlmostEqual(float(translit._erf(np.array([0.5]))[0]), 0.5204999, places=6)

    def test_route(self):
        self.assertEqual(translit.route("phone", {"phone"}, set()), "rules")
        self.assertEqual(translit.route("dara", {"dara"}, {"dara"}), "model")
        self.assertEqual(translit.route("zando", set(), set()), "rules")
        self.assertEqual(translit.route("sophea", set(), set()), "model")

    def test_pick_prefers_known_within_margin(self):
        nbest = [("ដរា", -0.8), ("តារា", -2.2)]
        self.assertEqual(translit.pick(nbest, {"តារា"}), "តារា")
        self.assertEqual(translit.pick(nbest, {"តារា"}, margin=1.0), "ដរា")
        self.assertEqual(translit.pick(nbest, set()), "ដរា")

    def test_word_tag(self):
        self.assertEqual(translit.word_tag("Shadow"), "en")
        self.assertEqual(translit.word_tag("joaquin"), "other")
        self.assertEqual(translit.source("one", False, "en"), translit.TAGS["en"] + "one")

    @unittest.skipUnless(translit.DEFAULT_WEIGHTS.exists(), "model weights not present")
    def test_english_words_use_en_model_when_configured(self):
        if translit._en_model() is None:
            self.skipTest("translit_en_model not configured")
        translit._GUESS_CACHE.clear()
        self.assertEqual([r[1] for r in translit.guess_many(["comfort", "sophea"])], ["model_en", "model"])


class CapsTest(unittest.TestCase):
    @unittest.skipUnless(translit.DEFAULT_WEIGHTS.exists(), "model files not present")
    def test_caps_names_vs_acronyms(self):
        for w in ("ABA", "USB", "UN", "TV", "HTML", "UNDP", "CPP", "NGO", "CEO"):
            self.assertFalse(translit.caps_reads_as_word(w), w)
        for w in ("DARA", "SOK", "SOPHEA", "HUN"):
            self.assertTrue(translit.caps_reads_as_word(w), w)


class ConfigureTest(unittest.TestCase):
    def tearDown(self):
        translit.configure(model="model_b", router="all", en_model="model_e")

    def test_configure_sets_and_validates(self):
        translit.configure(model="model", router="cmudict", en_model=None)
        self.assertEqual(translit.served_choice(), (translit.MODELS / "model" / "translit.npz", "cmudict"))
        self.assertIsNone(translit.SERVED["en_model"])
        translit.configure(router="all")   # other choices kept
        self.assertEqual(translit.SERVED["model"], "model")
        with self.assertRaises(ValueError):
            translit.configure(router="everything")


if __name__ == "__main__":
    unittest.main()
