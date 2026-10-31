"""E11 release preparation: frozen coverage sample, judgment reuse, coverage metrics, label blinding, bundle scan and
the adapter configuration's provenance (no model is loaded)."""
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
    # e11_release (via e12/e13) rebinds e07/e09/e11 bookkeeping globals on import; restore them afterwards
    saved = {m: {k: getattr(m, k) for k in ("LEDGER", "EXPERIMENT", "STATE", "LIMITS", "STATE_FILE") if hasattr(m, k)}
             for m in (e07, e09, e11)}
    saved_checkpoints = dict(e09.CHECKPOINTS)
    import e11_release as R
    for m, attrs in saved.items():
        for k, v in attrs.items():
            setattr(m, k, v)
    e09.CHECKPOINTS.clear()
    e09.CHECKPOINTS.update(saved_checkpoints)
except ImportError:                                   # research extras (torch) not installed
    R = None


@unittest.skipIf(R is None, "research extras not installed")
class TestE11Release(unittest.TestCase):
    def test_doctests(self):
        self.assertEqual(doctest.testmod(R).failed, 0)

    def test_frozen_sample(self):
        fz = json.loads((R.COV / "frozen.json").read_text())
        rows = R.cov_rows()                                              # raises if the manifest changed
        self.assertEqual(len(rows), fz["words"])
        self.assertTrue(20 <= len(rows) <= 30)
        self.assertLessEqual(fz["worst_case_new_judgments"], R.MAX_NEW_JUDGMENTS)
        self.assertEqual(len({r["word"] for r in rows}), len(rows))
        blocked = set()
        for w in [r["word"] for r in R.e13.frozen_rows(R.e13.TEST)] + \
                 [r["word"] for r in e11.jl(R.e13.E12_TEST / "input_manifest.jsonl")]:
            blocked |= R.e13.e12.family_keys(w)
        for r in rows:                                                   # never from a test set
            self.assertFalse(R.e13.e12.family_keys(r["word"]) & blocked, r["word"])
        self.assertTrue(all(r["pool"] != "B_e11_review_representative" or r["source_id"].endswith("#default")
                            for r in rows))
        with self.assertRaises(RuntimeError):                            # cannot be re-frozen
            R.cov_select()

    def test_judgment_reuse_is_exact(self):
        idx, meta = R.judged_index()
        self.assertTrue(idx)
        for (word, reading, text), v in list(idx.items())[:50]:
            self.assertEqual(text, R.e06.nfc(text))
            self.assertIn(v["judgment"], R.e10.JUDGMENTS)
            self.assertRegex(v["source"], r"^(e13_collection|e11_review):")
        self.assertEqual(meta["policy_sha256"], R.sha(R.e13.POLICY))

    def test_coverage_metrics(self):
        cands = [{"id": "a", "greedy": "ក", "beam": ["ក", "ខ"]},              # greedy wrong, beam 2 right
                 {"id": "b", "greedy": "គ", "beam": ["គ"]},                   # greedy right
                 {"id": "c", "greedy": "ឃ", "beam": ["ង"]}]                   # unjudged beam stays unknown
        j = {("a", "ក"): "not_acceptable", ("a", "ខ"): "acceptable", ("b", "គ"): "acceptable",
             ("c", "ឃ"): "uncertain"}
        m = R.coverage_metrics(cands, j)
        self.assertEqual(m["greedy_top1"], {"not_acceptable": 1, "acceptable": 1, "uncertain": 1})
        self.assertEqual(m["beam_top1"]["acceptable"], 1)
        self.assertEqual(m["any_acceptable_in_beam4"], 2)
        self.assertEqual(m["recoverable_words"], 1)
        self.assertEqual(m["candidate_judgments"]["missing"], 1)              # absence is not rejection
        self.assertLess(m["judgment_coverage"], 1)

    def test_labels_hide_rank(self):
        lab = R.label_texts(["ក", "ខ", "គ", "ឃ", "ង"], "s")
        self.assertEqual([lb for lb, _ in lab], list("ABCDE"))
        self.assertEqual(sorted(t for _, t in lab), sorted(["ក", "ខ", "គ", "ឃ", "ង"]))
        self.assertEqual(lab, R.label_texts(["ង", "ឃ", "គ", "ខ", "ក"], "s"))   # order-independent

    def test_bundle_scan(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name in ("ok.md", "best.pt", "origin_key.json", "judgments_1.tsv", "pytorch_model.bin"):
                (d / name).write_text("x")
            bad = R.scan(sorted(d.iterdir()), d)
            self.assertEqual(sorted(bad), ["best.pt", "judgments_1.tsv", "origin_key.json", "pytorch_model.bin"])

    def test_adapter_config_is_e11(self):
        done = json.loads((R.E11_DIR / "done.json").read_text())
        cfg = R.adapter_config(done)
        p = cfg["provenance"]
        self.assertEqual((p["candidate"], p["run"], p["update"], p["training_rows"]), ("e11", "r_s0", 3000, 4764))
        self.assertEqual(p["checkpoint_sha256"], R.E11_BEST_SHA)
        self.assertNotIn("google_ai_mode_synthetic", p["rows_by_provenance"])
        self.assertIn("no Google AI Mode", p["data_arm"])
        self.assertEqual(cfg["base"]["revision"], "68377bdc18a2ffec8a0533fef03b1c513a4dd49d")
        self.assertEqual(cfg["decode"]["candidate_beam"], 4)

    def test_public_folder_has_no_internal_labels(self):
        cfg = json.loads((R.ADAPTER_OUT / "adapter_config.json").read_text())
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "adapter_config.json").write_text(json.dumps(R.public_config(cfg), ensure_ascii=False))
            for src, name in ((R.DOCS / "public" / "README.md", "README.md"), (R.DOCS / "LICENSE-weights.md", "LICENSE")):
                (d / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            self.assertEqual(R.internal_terms(sorted(d.iterdir())), [])
        pub = R.public_config(cfg)
        for k in ("format", "base", "adapter", "adapter_sha256", "decode"):              # what the loader reads
            self.assertEqual(pub[k], cfg[k])


if __name__ == "__main__":
    unittest.main()
