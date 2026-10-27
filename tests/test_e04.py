"""Tests of E04 (`scripts/e04.py`, `scripts/e04_run.py`): source views, schedule, masks, pairing, export parity,
sanity subset, selection rule and the branching queue. Implementation behaviour only, not linguistic quality."""
from __future__ import annotations

import doctest
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import e04  # noqa: E402
from khmer_translit import phonetic_source as ps  # noqa: E402
from khmer_translit import reference_data as rd  # noqa: E402

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_E03 = (e04.DATA / "train.tsv").exists() and ps.CMUDICT.exists()
VOC = ps.ENGLISH_VOCAB
PAIRS = [("one", "វ៉ាន់"), ("two", "ធូ"), ("time", "ថាម"), ("news", "ញូស៍"), ("ten", "ថែន"), ("tea", "ធី"),
         ("bob", "បប"), ("name", "ណេម")]


class TestViews(unittest.TestCase):
    def test_doctests(self):
        self.assertEqual(doctest.testmod(e04, verbose=False).failed, 0)

    def test_p_has_no_lexical_information(self):
        ph = ("K", "AE1", "T")
        a, b = e04.encode("cat", ph, "p", VOC), e04.encode("kat", ph, "p", VOC)
        self.assertEqual(a, b)                                           # same phones -> same input, any spelling
        letters = {VOC.id(c) for c in "abcdefghijklmnopqrstuvwxyz"}
        self.assertFalse(letters & set(a))
        self.assertEqual(len(e04.encode("catastrophe", ph, "p", VOC)), len(a))   # no spelling length

    def test_g_and_gp_match_the_e03_encodings(self):
        pron = ps.PronunciationResult("cat", ("K", "AE1", "T"), "provided", "test")
        self.assertEqual(e04.encode("cat", pron.phones, "g", VOC), ps.encode_source("cat", None, "letters", VOC))
        self.assertEqual(e04.encode("cat", pron.phones, "gp", VOC), ps.encode_source("cat", pron, "phones", VOC))


class TestSchedule(unittest.TestCase):
    @unittest.skipUnless(HAVE_TORCH, "needs torch")
    def test_optimizer_applied_rates(self):
        import torch
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=1e-3)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, e04.seq_lambda(150, 3000))
        lr = []
        for _ in range(3000):
            lr.append(opt.param_groups[0]["lr"])
            p.grad = torch.ones(1)
            opt.step()
            sched.step()
        self.assertAlmostEqual(lr[0], 1e-3 / 150)
        self.assertAlmostEqual(lr[149], 1e-3)                            # base rate reached at the last warm-up step
        self.assertAlmostEqual(lr[150], 1e-3)                            # decay starts after warm-up
        self.assertLess(lr[151], lr[150])
        self.assertTrue(all(a < b for a, b in zip(lr[:149], lr[1:150])))
        self.assertTrue(all(a >= b for a, b in zip(lr[150:], lr[151:])))
        self.assertLess(lr[-1], 1e-8)
        self.assertEqual(max(lr), 1e-3)

    def test_old_schedule_peak_is_0_379(self):
        import english_phonetic_train as T
        f = T.lr_lambda(500, 720)
        self.assertAlmostEqual(max(f(k) for k in range(720)), 0.379, places=3)


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class TestMasks(unittest.TestCase):
    def test_padding_does_not_change_outputs(self):
        import torch
        import english_phonetic_train as T
        cfg = {"model": {"d": 32, "layers": 1, "heads": 2, "ff": 64, "dropout": 0.0}}
        m, h = T.build_model(len(VOC.tokens), 20, cfg, 0)
        m2, h2 = T.build_model(len(VOC.tokens), 20, cfg, 0)
        self.assertEqual(h, h2)                                          # same seed, same tensors
        m.eval()
        short = e04.encode("cat", ("K", "AE1", "T"), "p", VOC)
        long_ = e04.encode("catastrophe", ("K", "AH0", "T", "AE1", "S", "T", "R", "AH0", "F", "IY0"), "gp", VOC)
        tgt = torch.tensor([[1, 5, 6]])
        with torch.no_grad():
            alone = m(T.pad([short]), tgt)
            batch = m(T.pad([short, long_]), torch.cat([tgt, tgt]))[:1]
        self.assertTrue(torch.allclose(alone, batch, atol=1e-6))


@unittest.skipUnless(HAVE_TORCH and HAVE_E03, "needs torch and the E03 data")
class TestTraining(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict("os.environ", {"WANDB_MODE": "disabled"})
        self.env.start()
        self.data = [(w, k, e04.phones_of(w)) for w, k in PAIRS]
        self.dev = [rd.ReferenceItem(f"{w}#default", w, w, w, "default", "", "en-US-cmudict", [k], k, ["t"],
                                     "proxy", "dev") for w, k in PAIRS[:4]]
        self.cfg = {"model": {"d": 32, "layers": 1, "heads": 2, "ff": 64, "dropout": 0.0}}
        self.tc = {"updates": 6, "warmup": 2, "batch": 4, "lr": 3e-3, "weight_decay": 0.0, "label_smoothing": 0.0,
                   "grad_clip": 1.0, "threads": 2, "eval_every": 3}

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def run_arm(self, arm, name=None):
        out = Path(self.tmp.name) / (name or arm)
        return e04.train(arm, 1, out, self.cfg, data=self.data, val=self.data[:2], dev=self.dev, tc=self.tc), out

    def test_pairing_views_export_and_parity(self):
        g, _ = self.run_arm("G")
        p, out_p = self.run_arm("P")
        v3, out_v = self.run_arm("V3")
        self.assertEqual((g["init_sha256"], g["batch_order_sha256"]), (p["init_sha256"], p["batch_order_sha256"]))
        self.assertEqual(g["params"], p["params"])
        self.assertEqual(g["updates"], 6)
        self.assertEqual(sum(v3["view_counts"].values()), 6 * 4)
        self.assertEqual(set(v3["view_counts"]) <= {"g", "p", "gp"}, True)
        self.assertNotEqual(v3["batch_order_sha256"], g["batch_order_sha256"])   # the digest records the views
        with mock.patch.object(e04, "items", lambda name: self.dev):
            for out in (out_p, out_v):
                e04.export(out)
                rep = e04.verify(out, n=4)
                self.assertTrue(rep["ok"], rep)
        self.assertEqual(json.loads((out_v / "parity.json").read_text())["views"], ["g", "p", "gp"])

    def test_resume_matches_uninterrupted(self):
        full, _ = self.run_arm("P", "full")
        out = Path(self.tmp.name) / "cut"
        tc = dict(self.tc)
        with mock.patch.dict(self.tc, {"updates": 6}):
            pass
        e04.train("P", 1, out, self.cfg, data=self.data, val=self.data[:2], dev=self.dev, tc={**tc, "updates": 3})
        ck_state = __import__("torch").load(out / "last.pt", weights_only=False)
        self.assertEqual(ck_state["state"]["update"], 3)
        (out / "done.json").unlink()
        resumed = e04.train("P", 1, out, self.cfg, data=self.data, val=self.data[:2], dev=self.dev, tc=tc)
        self.assertEqual(resumed["batch_order_sha256"], full["batch_order_sha256"])
        self.assertEqual(resumed["updates"], 6)


@unittest.skipUnless(HAVE_E03, "needs the E03 data")
class TestSanitySubset(unittest.TestCase):
    def test_subset_is_unique_unambiguous_and_deterministic(self):
        a, info = e04.sanity_subset(64)
        b, _ = e04.sanity_subset(64)
        self.assertEqual(a, b)
        self.assertLessEqual(len(a), 64)
        self.assertEqual(len({ph for _w, _k, ph in a}), len(a))
        self.assertEqual(len({w for w, _k, _ph in a}), len(a))
        dev_words = {i.word for i in e04.items("dev")}
        self.assertFalse(dev_words & {w for w, _k, _ph in a})


class TestRule(unittest.TestCase):
    def test_choose(self):
        self.assertEqual(e04.choose({"G": 0.40, "P": 0.43, "GP": 0.45}, 2.0)["candidate"], "GP")
        self.assertEqual(e04.choose({"G": 0.40, "P": 0.42}, 2.0)["candidate"], "P")
        self.assertIsNone(e04.choose({"G": 0.40, "P": 0.30, "GP": 0.41, "V3": 0.419}, 2.0)["candidate"])
        self.assertEqual(e04.choose({"G": 0.40, "GP": 0.45, "V3": 0.45}, 2.0)["candidate"], "GP")


class TestRunner(unittest.TestCase):
    def make(self, root: Path, branch: str, candidate: str | None):
        import e04_run as RR
        calls, hooks_called = [], []

        def commands(j, runs):
            calls.append(j["id"])
            body = (f"import pathlib; p=pathlib.Path({str(runs / j['id'])!r}); p.mkdir(parents=True, exist_ok=True);"
                    " (p/'done.json').write_text('{}')")
            return [[sys.executable, "-c", body, j["id"]]]

        def verify(j, runs):
            return (runs / j["id"] / "done.json").exists()

        hooks = {"sanity_passed": lambda: True,
                 "devcheck": lambda: (hooks_called.append("devcheck"), {
                     "branch": branch, "useful_signal": branch != "A", "gp_strongest": branch == "B",
                     "union_gain_pp": 0.0, "p_macro_exceeds_g": False,
                     "scores": {a: {"macro": 0.4} for a in ("G", "P", "GP")}})[1],
                 "combine": lambda: (hooks_called.append("combine"), {"selected_w": "0.5"})[1],
                 "select": lambda jobs: (hooks_called.append("select"), {"candidate": candidate, "passing": []})[1],
                 "evaluate": lambda: (hooks_called.append("evaluate"), 0)[1]}
        cfgf = root / "config.yaml"
        cfgf.write_text("x: 1\n")
        r = RR.E04Runner(cfgf, root / "state", root / "runs", commands, verify, [cfgf], poll=0.01, hooks=hooks)
        return r, calls, hooks_called

    def test_branch_a_stops_after_pilots(self):
        with tempfile.TemporaryDirectory() as t:
            r, calls, h = self.make(Path(t), "A", None)
            self.assertEqual(r.resume(), 0)
            self.assertEqual(calls, ["G_s0", "P_s0", "GP_s0"])
            self.assertEqual(h, ["devcheck", "evaluate"])

    def test_branch_b_confirms_gp(self):
        with tempfile.TemporaryDirectory() as t:
            r, calls, h = self.make(Path(t), "B", "GP")
            self.assertEqual(r.resume(), 0)
            self.assertEqual(calls, ["G_s0", "P_s0", "GP_s0", "G_s1", "GP_s1", "G_s2", "GP_s2"])
            self.assertEqual(r.load()["budget"]["supervised_attempts_used"], 7)
            self.assertEqual(r.resume(), 0)                               # resume: nothing duplicated
            self.assertEqual(len(calls), 7)

    def test_branch_c_three_view_and_no_candidate(self):
        with tempfile.TemporaryDirectory() as t:
            r, calls, h = self.make(Path(t), "C", None)
            self.assertEqual(r.resume(), 0)
            self.assertEqual(calls, ["G_s0", "P_s0", "GP_s0", "V3_s0"])
            self.assertEqual(h, ["devcheck", "combine", "select", "evaluate"])

    def test_branch_c_with_candidate_uses_all_eight_attempts(self):
        with tempfile.TemporaryDirectory() as t:
            r, calls, _h = self.make(Path(t), "C", "V3")
            self.assertEqual(r.resume(), 0)
            self.assertEqual(len(calls), 8)
            st = r.load()
            self.assertEqual(st["budget"]["supervised_attempts_used"], 8)
            j = st["jobs"][0]
            self.assertEqual(r.budget_ok(st, j, True), "supervised attempt cap reached")

    def test_sanity_gate(self):
        import e04_run as RR
        with tempfile.TemporaryDirectory() as t:
            cfgf = Path(t) / "c.yaml"
            cfgf.write_text("x: 1\n")
            r = RR.E04Runner(cfgf, Path(t) / "s", Path(t) / "r", registered=[cfgf], hooks={"sanity_passed": lambda: False})
            self.assertEqual(r.resume(), 1)
            self.assertIn("sanity checks have not passed", r.load()["blockers"])


if __name__ == "__main__":
    unittest.main()
