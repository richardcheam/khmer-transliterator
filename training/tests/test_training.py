"""Training code: tiny runs on made-up data, and the package reads what the trainers export.

    python -m unittest discover -s training/tests
The ByT5 test also needs a local google/byt5-small: set KHMER_TRANSLIT_BYT5_BASE=folder (skipped otherwise).
"""
import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_byt5_adapter as byt5_train  # noqa: E402
import train_small_model as small  # noqa: E402
from khmer_translit import phonetic_source as ps  # noqa: E402

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8"))
PAIRS = [("ka", "កា"), ("ko", "កូ"), ("ta", "តា"), ("na", "ណា"), ("ba", "បា"), ("mi", "មី"), ("la", "លា"),
         ("sa", "សា"), ("ha", "ហា"), ("kom", "កុំ")]
CHARS = "".join(sorted(set("".join(k for _w, k in PAIRS))))
DEV = [{"id": f"{w}#default", "word": w, "family": w, "accepted": [k], "preferred": k, "flags": []} for w, k in PAIRS]
BASE = os.environ.get("KHMER_TRANSLIT_BYT5_BASE")


class SmallModel(unittest.TestCase):
    def cfg(self):
        c = copy.deepcopy(CFG)
        c["small_model"].update({"model": {"d": 32, "layers": 1, "heads": 2, "ff": 64, "dropout": 0.0},
                                 "warmup": 10, "lr": 0.01, "batch": 10, "threads": 1})
        return c

    def test_learns_and_exports_for_the_package(self):
        from khmer_translit.english import Transliterator
        data = (PAIRS, PAIRS[:2], DEV, ps.ENGLISH_VOCAB, CHARS)
        with tempfile.TemporaryDirectory() as d:
            r = small.train(0, Path(d), epochs=120, cfg=self.cfg(), data=data)
            self.assertGreaterEqual(r["best_dev_score"], 0.9)
            tl = Transliterator.from_pretrained(Path(d) / "model.npz")
            self.assertEqual(tl.predict("kom").text, "កុំ")

    def test_same_seed_same_result(self):
        data = (PAIRS, PAIRS[:2], DEV, ps.ENGLISH_VOCAB, CHARS)
        with tempfile.TemporaryDirectory() as d:
            a = small.train(1, Path(d) / "a", epochs=3, cfg=self.cfg(), data=data)
            b = small.train(1, Path(d) / "b", epochs=3, cfg=self.cfg(), data=data)
        self.assertEqual([x["train_loss"] for x in a["log"]], [x["train_loss"] for x in b["log"]])


class ByT5Pieces(unittest.TestCase):
    def test_lora_starts_as_the_base_layer(self):
        base = torch.nn.Linear(4, 3)
        lo = byt5_train.LoRALinear(base, r=2, alpha=4, dropout=0.0)
        x = torch.randn(5, 4)
        self.assertTrue(torch.equal(lo(x), base(x)))
        self.assertEqual(lo.weight.shape, base.weight.shape)

    def test_schedule(self):
        f = [byt5_train.lr_factor(k, 10, 100) for k in (0, 9, 55, 99)]
        self.assertEqual([round(x, 3) for x in f], [0.1, 1.0, 0.5, 0.011])

    def test_batch_bytes(self):
        ids, mask, labels = byt5_train.batch([("Ab", "ក")], "cpu")
        self.assertEqual(ids.tolist(), [[100, 101, 1]])
        self.assertEqual(labels.tolist(), [[228, 161, 131, 1]])


@unittest.skipUnless(BASE, "set KHMER_TRANSLIT_BYT5_BASE to a local google/byt5-small")
class ByT5Adapter(unittest.TestCase):
    def test_short_run_exports_a_loadable_adapter(self):
        from khmer_translit.byt5 import ByT5Transliterator
        c = copy.deepcopy(CFG)
        c["byt5_adapter"].update({"max_updates": 4, "eval_at": [4], "batch": 4})
        with tempfile.TemporaryDirectory() as d:
            byt5_train.train(0, Path(d), "cpu", base=BASE, cfg=c, data=(PAIRS, DEV))
            tl = ByT5Transliterator.from_pretrained(Path(d) / "adapter", base=BASE, offline=True)
            r = tl.predict("ka")                 # 4 updates teach nothing; the point is that it loads and runs
            self.assertEqual(r.input, "ka")
            self.assertIsInstance(r.status, str)


if __name__ == "__main__":
    unittest.main()
