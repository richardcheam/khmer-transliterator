"""Guards of the E02 release script: it refuses a substituted checkpoint and never bundles restricted files."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import e02_release as rel  # noqa: E402


class TestE02Release(unittest.TestCase):
    def test_missing_checkpoint_refused(self):
        with mock.patch.object(rel, "CHECKPOINT", Path("/nonexistent/model.npz")):
            with self.assertRaises(FileNotFoundError):
                rel.check_checkpoint()

    def test_substituted_checkpoint_refused(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "model.npz"
            fake.write_bytes(b"another model")
            with mock.patch.object(rel, "CHECKPOINT", fake), self.assertRaises(RuntimeError):
                rel.check_checkpoint()

    def test_restricted_names_are_forbidden(self):
        for name in ("train.tsv", "references.jsonl", "best.pt", "adapter.safetensors", "cmudict.dict",
                     "origin_key.json"):
            p = Path(name)
            self.assertTrue(p.suffix in rel.FORBIDDEN_SUFFIXES or any(n in p.name.lower()
                                                                      for n in rel.FORBIDDEN_NAMES), name)
        for name in ("model.npz", "MODEL_CARD.md", "SHA256SUMS", "requirements.txt"):
            p = Path(name)
            self.assertFalse(p.suffix in rel.FORBIDDEN_SUFFIXES or any(n in p.name.lower()
                                                                       for n in rel.FORBIDDEN_NAMES), name)


if __name__ == "__main__":
    unittest.main()
