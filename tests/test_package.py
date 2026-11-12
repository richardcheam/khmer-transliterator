"""Package tests. Model tests need a downloaded model: set KHMER_TRANSLIT_MODEL=path/to/model.npz (skipped otherwise).

    python -m unittest discover -s tests
"""
import contextlib
import doctest
import io
import json
import os
import unittest

from khmer_translit import nnet
from khmer_translit import phonetic_source as ps

MODEL = os.environ.get("KHMER_TRANSLIT_MODEL")


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(nnet))
    tests.addTests(doctest.DocTestSuite(ps))
    return tests


class Normalization(unittest.TestCase):
    def test_variants(self):
        for raw in ("Coffee", " coffee ", "COFFEE", "coffée", "cof-fee", "cof'fee", "ｃｏｆｆｅｅ"):
            self.assertEqual(ps.normalize_word(raw), ("coffee", "ok"), raw)

    def test_refusals(self):
        self.assertEqual(ps.normalize_word("")[1], "empty")
        self.assertEqual(ps.normalize_word("new york")[1], "unsupported_multiword")
        self.assertEqual(ps.normalize_word("b2b")[1], "unsupported_characters")
        self.assertEqual(ps.normalize_word("កាហ្វេ")[1], "unsupported_characters")
        self.assertEqual(ps.normalize_word("a" * (ps.MAX_LETTERS + 1))[1], "too_long")

    def test_letters_source(self):
        ids = ps.encode_source("ab", None, "letters")
        self.assertEqual([ps.ENGLISH_VOCAB.tokens[i] for i in ids], ["<task:en>", "a", "b"])


class ByteCodec(unittest.TestCase):
    def test_round_trip(self):
        from khmer_translit import byt5
        ids = byt5.encode("Coffee")
        self.assertEqual(ids[-1], 1)
        self.assertEqual(byt5.decode([b + 3 for b in "ក".encode("utf-8")] + [1]), ("ក", "ok"))


@unittest.skipUnless(MODEL, "set KHMER_TRANSLIT_MODEL to a model.npz")
class SmallModel(unittest.TestCase):
    def test_predict_and_refuse(self):
        from khmer_translit.english import Transliterator
        tl = Transliterator.from_pretrained(MODEL)
        r = tl.predict("coffee", top_k=3)
        self.assertEqual(r.status, "ok")
        self.assertTrue(1 <= len(r.candidates) <= 3)
        self.assertEqual(tl.predict("new york").status, "unsupported_multiword")

    def test_cli_json(self):
        from khmer_translit import cli
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["predict", "--model", MODEL, "--word", "johnson", "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["status"], "ok")


if __name__ == "__main__":
    unittest.main()
