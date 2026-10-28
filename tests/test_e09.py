"""Tests of E09 (`scripts/e09.py`): setting hashes, cache identity and resume, prediction alignment, comparison with
saved files, score reconstruction, the selection rule and the disagreement sample.

Model tests use a tiny random T5 (no download, seconds on CPU), never the pinned ByT5 checkpoints."""
from __future__ import annotations

import doctest
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import e06_byt5 as B  # noqa: E402
import e07  # noqa: E402
import e09  # noqa: E402

try:
    import torch
    from transformers import T5Config, T5ForConditionalGeneration
except ImportError:                                  # research extras not installed (research/requirements-byt5.txt)
    torch = None


def tiny_t5():
    torch.manual_seed(1)
    cfg = T5Config(vocab_size=384, d_model=32, d_kv=8, d_ff=48, num_layers=2, num_decoder_layers=1, num_heads=2,
                   feed_forward_proj="gated-gelu", dropout_rate=0.0, decoder_start_token_id=0, pad_token_id=0,
                   eos_token_id=1, tie_word_embeddings=False)
    return T5ForConditionalGeneration(cfg).eval()


class TestPure(unittest.TestCase):
    def test_doctests(self):
        self.assertEqual(doctest.testmod(e09).failed, 0)

    def test_setting_hash_changes_with_every_field(self):
        hashes = {e09.setting_hash(n) for n in [*e09.SETTINGS, *e09.FOLLOW_UP]}
        self.assertEqual(len(hashes), len(e09.SETTINGS) + len(e09.FOLLOW_UP))
        self.assertEqual(e09.setting_hash("beam4_reg"), e09.setting_hash("beam4_reg"))

    def test_registered_settings_match_e07_call(self):
        # the registered beam call must reach e07.gen with no extra flags (identical to the saved predictions)
        self.assertEqual(e09.gen_kwargs("beam4_reg"), {"beam": 4, "ret": 4, "bs": 16})
        self.assertEqual(e09.gen_kwargs("greedy"), {"beam": 1, "ret": 1, "bs": 32})
        self.assertEqual(e09.gen_kwargs("beam4_can")["early_stopping"], "never")

    def test_cache_path_depends_on_checkpoint_setting_and_data(self):
        rows = [("a", "cat")]
        p = e09.cache_path("e07", "greedy", "dev", "x" * 64, rows)
        self.assertNotEqual(p, e09.cache_path("e07", "greedy", "dev", "y" * 64, rows))
        self.assertNotEqual(p, e09.cache_path("e07", "beam4_reg", "dev", "x" * 64, rows))
        self.assertNotEqual(p, e09.cache_path("e07", "greedy", "dev", "x" * 64, [("a", "dog")]))

    def test_selection_rule(self):
        tab = {"greedy": {"top1_macro": .52, "mean_cer": .16, "generation_seconds": 10},
               "beam4_can": {"top1_macro": .52, "mean_cer": .15, "generation_seconds": 40},
               "beam8_can": {"top1_macro": .51, "mean_cer": .10, "generation_seconds": 80}}
        self.assertEqual(e09.select_setting(tab), "beam4_can")          # top-1 tie → lower CER
        tab["beam4_can"]["mean_cer"] = .16
        self.assertEqual(e09.select_setting(tab), "greedy")             # top-1 and CER tie → cheaper

    def test_alignment_refuses_missing_or_extra(self):
        class It:
            def __init__(self, i):
                self.id = i
        with self.assertRaises(ValueError):
            e09.check_alignment([It("a"), It("b")], {"a": ["x"]})
        with self.assertRaises(ValueError):
            e09.check_alignment([It("a")], {"a": ["x"], "z": ["y"]})

    def test_compare_saved_detects_differences(self):
        rec = [{"id": "a", "cands": [{"text": "ក", "status": "ok", "score": -1.0, "ids": []},
                                     {"text": "", "status": "empty", "score": -9.0, "ids": []}]}]
        sv = {"a": {"beam4": ["ក"], "beam4_raw": [{"text": "ក", "status": "ok", "score": -1.00001},
                                                  {"text": "", "status": "empty", "score": -9.0}]}}
        self.assertEqual(e09.compare_saved(rec, sv, "beam4_reg")["mismatches"], 0)
        sv["a"]["beam4_raw"][0]["score"] = -1.1
        self.assertEqual(e09.compare_saved(rec, sv, "beam4_reg")["mismatches"], 1)
        self.assertEqual(e09.compare_saved(rec, {"a": {"greedy": "ខ", "greedy_status": "ok"}}, "greedy")["mismatches"], 1)

    def test_balanced_sample_is_reproducible_and_order_free(self):
        g = {"agree": [f"i{n}" for n in range(20)], "both_wrong_different": ["i1", "i2"]}
        a = e09.balanced_sample(g, "e07")
        b = e09.balanced_sample({k: list(reversed(v)) for k, v in g.items()}, "e07")
        self.assertEqual(a, b)
        self.assertEqual((len(a["agree"]), len(a["both_wrong_different"])), (5, 2))

    def test_budget_refuses_when_spent(self):
        with mock.patch.object(e09, "ledger", return_value=[{"event": "generate", "seconds": 4 * 3600}]):
            with self.assertRaises(RuntimeError):
                e09.check_budget()


@unittest.skipIf(torch is None, "research extras not installed")
class TestModel(unittest.TestCase):
    def setUp(self):
        self.m = tiny_t5()
        self.rows = [("a", "cat"), ("b", "london"), ("c", "ali")]

    def test_score_reconstruction_matches_generation(self):
        # sequences_scores (lp 0) and the greedy chosen-token sum equal the teacher-forced log-prob incl. EOS
        for beam in (1, 4):
            res = e07.gen(self.m, [B.encode_src(w) for _, w in self.rows], beam=beam, ret=beam, max_new=12)
            pairs = [(w, ids) for (_, w), r in zip(self.rows, res) for _, _, _, ids in r]
            tf = e09.tf_logprobs(self.m, pairs)
            gen = [sc for r in res for _, _, sc, _ in r]
            for a, b in zip(gen, tf):
                self.assertAlmostEqual(a, b, places=3)

    def test_generate_caches_and_resumes(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(e09, "CACHE", Path(d)), \
                mock.patch.object(e09, "LEDGER", Path(d) / "ledger.jsonl"), \
                mock.patch.object(e07, "MAX_NEW", 12):
            a = e09.generate(self.m, "e07", "greedy", "dev", self.rows, "x" * 64)
            with mock.patch.object(e07, "gen", side_effect=AssertionError("regenerated")):
                b = e09.generate(self.m, "e07", "greedy", "dev", self.rows, "x" * 64)
            self.assertEqual(a, b)
            self.assertEqual([r["id"] for r in a], ["a", "b", "c"])
            # a partial (truncated) file is not trusted: it is regenerated in full
            p = e09.cache_path("e07", "greedy", "dev", "x" * 64, self.rows)
            p.write_text(p.read_text().splitlines()[0] + "\n")
            c = e09.generate(self.m, "e07", "greedy", "dev", self.rows, "x" * 64)
            self.assertEqual(len(c), 3)
            events = [json.loads(x)["event"] for x in (Path(d) / "ledger.jsonl").read_text().splitlines()]
            self.assertEqual(events, ["generate", "generate"])

    def test_memory_pressure_stops_before_writing(self):
        g = e07.Guard(read=lambda: {"rss": 6 * e09.GiB, "available": 4 * e09.GiB}, limits=e09.LIMITS)
        g.bad = 2
        with tempfile.TemporaryDirectory() as d, mock.patch.object(e09, "CACHE", Path(d)), \
                mock.patch.object(e09, "LEDGER", Path(d) / "ledger.jsonl"):
            with self.assertRaises(RuntimeError):
                e09.generate(self.m, "e07", "greedy", "dev", self.rows, "x" * 64, guard=g)
            self.assertFalse(any(Path(d).glob("*.jsonl")) and
                             any(p.name != "ledger.jsonl" for p in Path(d).glob("*.jsonl")))


if __name__ == "__main__":
    unittest.main()
