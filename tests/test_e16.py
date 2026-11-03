"""Doctests of the E16 Gemini-bulk data helpers (segmentation, automatic filter) (no data files or torch needed)."""
import doctest
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def load_tests(loader, tests, ignore):
    import e16
    tests.addTests(doctest.DocTestSuite(e16))
    return tests


if __name__ == "__main__":
    unittest.main()
