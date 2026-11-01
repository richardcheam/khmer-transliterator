"""Doctests of the E14 data and gate helpers (no data files or torch needed)."""
import doctest
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def load_tests(loader, tests, ignore):
    import e14_train
    tests.addTests(doctest.DocTestSuite(e14_train))
    return tests


if __name__ == "__main__":
    unittest.main()
