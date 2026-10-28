"""Tests of the ByT5 backend (`khmer_translit.byt5`) and its CLI profile: parity with the research generation code,
refusals instead of truncation, adapter/base checks, offline behaviour and stable JSONL alignment.

A tiny random T5 stands in for google/byt5-small (no download, seconds on CPU); the real-checkpoint parity check is
in `scripts/e10_release.py verify`."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch
    from transformers import T5Config, T5ForConditionalGeneration
    from khmer_translit import byt5
except ImportError:                                   # the optional `byt5` extra is not installed
    torch = None

ADAPTER = {"r": 4, "alpha": 8, "dropout": 0.0, "targets": ["attn", "ffn"]}


def tiny_base(folder: Path) -> str:
    torch.manual_seed(3)
    cfg = T5Config(vocab_size=384, d_model=32, d_kv=8, d_ff=48, num_layers=2, num_decoder_layers=1, num_heads=2,
                   feed_forward_proj="gated-gelu", dropout_rate=0.0, decoder_start_token_id=0, pad_token_id=0,
                   eos_token_id=1, tie_word_embeddings=False)
    T5ForConditionalGeneration(cfg).save_pretrained(folder)
    return byt5.sha256_file(next(folder / f for f in byt5.BASE_WEIGHTS if (folder / f).exists()))


def research_model(folder: Path):
    """The research path (`e07.insert_adapters`) on the same tiny base, with non-zero adapter weights."""
    import e07
    m = T5ForConditionalGeneration.from_pretrained(str(folder), dtype=torch.float32)
    m = e07.insert_adapters(m, ADAPTER, seed=0)
    g = torch.Generator().manual_seed(5)
    for k, p in m.named_parameters():
        if k.endswith((".A", ".B")):
            p.data = torch.randn(p.shape, generator=g) * 0.5
    return m.eval()


def make_adapter(base: Path, out: Path, sha: str):
    m = research_model(base)
    state = {k: v.detach().clone() for k, v in m.state_dict().items() if k.endswith((".A", ".B"))}
    byt5.export_adapter(state, out, {"base": {"repo": byt5.BASE_REPO, "revision": byt5.BASE_REVISION,
                                              "weights_sha256": sha},
                                     "adapter": {k: ADAPTER[k] for k in ("r", "alpha", "targets")},
                                     "decode": {"single": "greedy", "candidate_beam": 4, "batch_size": 32}})
    return m


@unittest.skipIf(torch is None, "the byt5 extra is not installed")
class TestByT5Backend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.base, cls.adapter = d / "base", d / "adapter"
        cls.sha = tiny_base(cls.base)
        cls.research = make_adapter(cls.base, cls.adapter, cls.sha)
        cls.tl = byt5.ByT5Transliterator.from_pretrained(cls.adapter, base=cls.base, offline=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_parity_with_research_generation(self):
        import e07
        words = ["cat", "london", "coffee", "ali", "x"]
        srcs = [byt5.encode(w) for w in words]
        greedy = e07.gen(self.research, srcs, max_new=byt5.MAX_NEW_TOKENS)
        beam = e07.gen(self.research, srcs, beam=4, ret=4, bs=16, max_new=byt5.MAX_NEW_TOKENS, early_stopping="never")
        for w, g, b in zip(words, greedy, beam):
            r1, r4 = self.tl.predict(w), self.tl.predict(w, top_k=4)
            want1 = [t for t, s, *_ in g if s == "ok" and t]
            self.assertEqual([c.text for c in r1.candidates], want1[:1])
            want4 = list(dict.fromkeys(t for t, s, *_ in b if s == "ok" and t))
            self.assertEqual([c.text for c in r4.candidates], want4[:4])
            if r1.candidates:
                self.assertAlmostEqual(r1.candidates[0].log_score, g[0][2], places=4)

    def test_batch_equals_single_and_keeps_order(self):
        words = ["cat", "", "new york", "coffee", "a" * 33, "ខ្មែរ", "london"]
        many = self.tl.predict_many(words)
        self.assertEqual([r.input for r in many], words)
        self.assertEqual([r.status for r in many][1:6:1][:5],
                         ["empty", "unsupported_multiword", many[3].status, "too_long", "unsupported_characters"])
        for w, r in zip(words, many):
            if r.status == "ok":
                self.assertEqual(r.text, self.tl.predict(w).text)

    def test_long_input_refused_not_truncated(self):
        r = self.tl.predict("b" * 33)
        self.assertEqual((r.status, r.candidates), ("too_long", []))
        self.assertIn(self.tl.predict("b" * 32).status, ("ok", "no_candidate"))

    def test_top_k_bounds(self):
        with self.assertRaises(ValueError):
            self.tl.predict("cat", top_k=0)
        with self.assertRaises(ValueError):
            self.tl.predict("cat", top_k=9)

    def test_missing_and_tampered_files(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                byt5.ByT5Transliterator.from_pretrained(d, base=self.base)
            bad = Path(d) / "adapter"
            bad.mkdir()
            for f in (byt5.CONFIG_FILE, byt5.ADAPTER_FILE):
                (bad / f).write_bytes((self.adapter / f).read_bytes())
            with open(bad / byt5.ADAPTER_FILE, "ab") as f:
                f.write(b"x")
            with self.assertRaises(byt5.ModelFormatError):
                byt5.ByT5Transliterator.from_pretrained(bad, base=self.base)
        with self.assertRaises(byt5.ModelFormatError):
            byt5.ByT5Transliterator.from_pretrained(self.adapter, base=self.base, expected_sha256="0" * 64)

    def test_wrong_base_refused(self):
        with tempfile.TemporaryDirectory() as d:
            other = Path(d)
            torch.manual_seed(9)
            cfg = T5Config.from_pretrained(self.base)
            T5ForConditionalGeneration(cfg).save_pretrained(other)
            with self.assertRaises(byt5.ModelFormatError):
                byt5.ByT5Transliterator.from_pretrained(self.adapter, base=other)
            for f in byt5.BASE_WEIGHTS:
                (other / f).unlink(missing_ok=True)
            with self.assertRaises(FileNotFoundError):
                byt5.ByT5Transliterator.from_pretrained(self.adapter, base=other)

    def test_offline_without_cache_fails_clearly_without_download(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"HF_HOME": d, "HF_HUB_OFFLINE": "1"}):
            with self.assertRaises(FileNotFoundError) as e:
                byt5.ByT5Transliterator.from_pretrained(self.adapter, base=None)
            self.assertIn("offline", str(e.exception))
            self.assertFalse(any(Path(d).rglob("*.bin")))

    def test_cli_jsonl_alignment_and_exit_status(self):
        from khmer_translit import cli
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "words.txt"
            inp.write_text("cat\n\nnew york\ncoffee\n", encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = cli.main(["predict", "--backend", "byt5", "--model", str(self.adapter), "--base",
                                 str(self.base), "--offline", "--input", str(inp), "--jsonl", "--top-k", "2"])
            rows = [json.loads(x) for x in buf.getvalue().splitlines()]
        self.assertEqual(code, 1)                                         # two inputs refused
        self.assertEqual([r["id"] for r in rows], [1, 2, 3, 4])
        self.assertEqual([r["input"] for r in rows], ["cat", "", "new york", "coffee"])
        self.assertEqual({r["backend"] for r in rows}, {"byt5"})
        self.assertEqual(set(rows[0]), {"id", "backend", "input", "normalized", "status", "candidates",
                                        "pronunciation", "message"})
        self.assertLessEqual(len(rows[0]["candidates"]), 2)

    def test_cli_missing_adapter_is_usage_error(self):
        from khmer_translit import cli
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = cli.main(["predict", "--backend", "byt5", "--model", "/nonexistent", "--word", "cat"])
        self.assertEqual(code, 2)
        self.assertIn("adapter file not found", err.getvalue())


if __name__ == "__main__":
    unittest.main()
