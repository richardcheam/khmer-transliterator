"""Tests for E02 training, export, the English API and the job runner (tiny artificial data; CPU).

The overfit test proves the implementation can learn and that NumPy inference matches PyTorch; it says nothing
about generalisation. Training tests skip without torch or the local CMUdict copy.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from khmer_translit import phonetic_source as ps  # noqa: E402
from khmer_translit import reference_data as rd  # noqa: E402

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_CMU = ps.CMUDICT.exists()
PAIRS = [("one", "វ៉ាន់"), ("two", "ធូ"), ("time", "ថាម"), ("news", "ញូស៍"), ("ten", "ថែន"), ("tea", "ធី"),
         ("bob", "បប"), ("name", "ណេម")]


def tiny_data(folder: Path) -> None:
    """A complete miniature `data/e02` folder."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "train.tsv").write_text("word\tkhmer\tsource\ttier\tphones\tpart\n" + "".join(
        f"{w}\t{k}\ttest\tproxy\t\t{'train_val' if w == 'bob' else 'train'}\n" for w, k in PAIRS) + "".join(
        f"{w}\t{k}\ttest\tproxy\t\ttrain\n" for w, k in PAIRS if w == "bob"), encoding="utf-8")
    (folder / "p2g.tsv").write_text("phones\tkhmer\tpart\nk a\tកា\tp2g_train\nk a\tការ\tp2g_train\nt i\tទី\tp2g_dev\n",
                                    encoding="utf-8")
    dev = [rd.ReferenceItem(f"{w}#default", w, w, w, "default", "", "en-US-cmudict", [k], k, ["t"], "attested", "dev")
           for w, k in PAIRS[:6]]
    rd.save_reference_items(dev, folder / "dev_references.jsonl")
    final = [rd.ReferenceItem("zzfinal#default", "zzfinal", "zzfinal", "zzfinal", "default", "", "en-US-cmudict",
                              ["ក"], "ក", ["t"], "attested", "final")]
    rd.save_reference_items(dev + final, folder / "references.jsonl")
    tgt = "".join(sorted({c for _w, k in PAIRS for c in k} | set("កាការទី")))
    km = ["a", "i", "k", "t"]
    (folder / "vocab.json").write_text(json.dumps({"encoder_version": ps.ENCODER_VERSION,
                                                   "src_tokens": ps.SourceVocab(km).tokens, "km_phones": km,
                                                   "tgt_chars": tgt}, ensure_ascii=False), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps({"counts": {"supervised_kept": {"test": 9}},
                                                      "cmudict_sha256": ps.sha256_file(ps.CMUDICT)}), encoding="utf-8")


def cfg(epochs: int = 2, lr: float = 1e-3) -> dict:
    return {"model": {"d": 32, "layers": 1, "heads": 2, "ff": 64, "dropout": 0.0},
            "train": {"epochs": epochs, "batch": 4, "lr": lr, "weight_decay": 0.0, "warmup": 1, "label_smoothing": 0.0,
                      "grad_clip": 1.0, "threads": 2, "phone_dropout": 0.1},
            "p2g": {"epochs": 2, "max_hours": 0.5, "lr": 1e-3, "warmup": 1, "batch": 2, "replay_every": 2,
                    "replay_batch": 2},
            "eval": {"beam": 4, "top_k": 4}, "overrides": []}


@unittest.skipUnless(HAVE_TORCH and HAVE_CMU, "needs torch and data/src/cmudict")
class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import english_phonetic_train as T
        cls.T = T
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.tmp.name) / "data"
        tiny_data(cls.data)
        cls.patch = mock.patch.object(T, "DATA", cls.data)
        cls.patch.start()
        cls.env = mock.patch.dict("os.environ", {"WANDB_MODE": "disabled"})
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls.env.stop()
        cls.tmp.cleanup()

    def out(self, name: str) -> Path:
        return Path(self.tmp.name) / name

    def test_paired_arms_share_init_and_batches(self):
        a = self.T.train("letters", 0, self.out("l0"), cfg())
        b = self.T.train("phones", 0, self.out("p0"), cfg())
        self.assertEqual(a["init_sha256"], b["init_sha256"])
        self.assertEqual(a["batch_order_sha256"], b["batch_order_sha256"])
        c = self.T.train("letters", 1, self.out("l1"), cfg())
        self.assertNotEqual(a["init_sha256"], c["init_sha256"])

    def test_p2g_transfer_and_replay(self):
        p = self.T.pretrain(0, self.out("p2g0"), cfg())
        self.assertEqual(p["epochs_run"], 2)
        r = self.T.train("phones_p2g", 0, self.out("c0"), cfg(), init=self.out("p2g0") / "best.pt")
        self.assertGreater(r["replay_steps"], 0)
        self.assertEqual(r["p2g_init_file_sha256"], ps.sha256_file(self.out("p2g0") / "best.pt"))
        with self.assertRaises(ValueError):
            self.T.train("phones", 0, self.out("bad"), cfg(), init=self.out("p2g0") / "best.pt")

    def test_trainer_never_opens_final_references(self):
        opened = []
        real = Path.read_text

        def spy(self_, *a, **k):
            opened.append(self_.name)
            return real(self_, *a, **k)

        with mock.patch.object(Path, "read_text", spy):
            self.T.train("letters", 0, self.out("spy"), cfg(epochs=1))
        self.assertIn("dev_references.jsonl", opened)
        self.assertNotIn("references.jsonl", opened)

    def test_resume_continues_and_matches(self):
        full = self.T.train("letters", 3, self.out("full"), cfg(epochs=3))
        with self.assertRaises(self.T.Interrupted):
            self.T.train("letters", 3, self.out("part"), cfg(epochs=3), stop_after=2)
        self.assertFalse((self.out("part") / "done.json").exists())
        resumed = self.T.train("letters", 3, self.out("part"), cfg(epochs=3))
        self.assertEqual(full["batch_order_sha256"], resumed["batch_order_sha256"])
        self.assertEqual(full["supervised_steps"], resumed["supervised_steps"])
        self.assertAlmostEqual(full["best_train_val_loss"], resumed["best_train_val_loss"], places=5)
        self.assertEqual(full["best_epoch"], resumed["best_epoch"])

    def test_overfit_export_parity_and_api(self):
        from khmer_translit.english import ModelFormatError, Transliterator
        import torch
        out = self.out("fit")
        s = self.T.train("phones", 0, out, cfg(epochs=150, lr=3e-3))
        self.assertGreaterEqual(s["best_dev_primary_macro"], 0.8)
        best = torch.load(out / "best.pt", weights_only=False)
        self.assertEqual(best["epoch"], s["best_epoch"])
        self.T.export(out, cfg())
        rep = self.T.verify(out, n=6)
        self.assertTrue(rep["ok"], rep)
        tl = Transliterator.from_pretrained(out)
        r1 = [r.to_dict() for r in tl.predict_many(["one", "Two", "", "x y", "zzqxv"], top_k=2)]
        r2 = [r.to_dict() for r in tl.predict_many(["one", "Two", "", "x y", "zzqxv"], top_k=2)]
        self.assertEqual(r1, r2)                                     # deterministic
        self.assertEqual([r["status"] for r in r1[:4]], ["ok", "ok", "empty", "unsupported_multiword"])
        self.assertEqual(r1[4]["pronunciation"]["status"], "unavailable")
        self.assertLessEqual(len(r1[0]["candidates"]), 2)
        texts = [c["text"] for c in r1[0]["candidates"]]
        self.assertEqual(len(texts), len(set(texts)))               # distinct candidates
        self.assertEqual(tl.predict("one", pronunciation="W AH1 N").pronunciation["status"], "provided")
        self.assertEqual(tl.predict("one", pronunciation="W XX N").status, "unknown_phone")
        with self.assertRaises(ValueError):
            tl.predict("one", top_k=0)
        with self.assertRaises(ModelFormatError):
            Transliterator.from_pretrained(out, expected_sha256="0" * 64)
        bad = out / "corrupt.npz"
        bad.write_bytes(b"not a zip")
        with self.assertRaises(ModelFormatError):
            Transliterator.from_pretrained(bad)
        with self.assertRaises(FileNotFoundError):
            Transliterator.from_pretrained(out / "absent.npz")


@unittest.skipUnless(HAVE_TORCH and HAVE_CMU, "needs torch and data/src/cmudict")
class TestCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import english_phonetic_train as T
        cls.tmp = tempfile.TemporaryDirectory()
        data = Path(cls.tmp.name) / "data"
        tiny_data(data)
        cls.out = Path(cls.tmp.name) / "m"
        with mock.patch.object(T, "DATA", data), mock.patch.dict("os.environ", {"WANDB_MODE": "disabled"}):
            T.train("letters", 0, cls.out, cfg(epochs=60, lr=3e-3))
            T.export(cls.out, cfg())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_cli(self, *argv: str) -> tuple[int, str]:
        import contextlib
        import io
        from khmer_translit import cli
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(list(argv))
        return code, buf.getvalue()

    def test_predict_formats_and_order(self):
        words = Path(self.tmp.name) / "words.txt"
        words.write_text("one\n\nTwo\nnew york\n", encoding="utf-8")
        code, out = self.run_cli("predict", "--model", str(self.out), "--input", str(words), "--jsonl")
        rows = [json.loads(l) for l in out.splitlines()]
        self.assertEqual(code, 1)                                  # two inputs refused
        self.assertEqual([r["id"] for r in rows], [1, 2, 3, 4])
        self.assertEqual([r["input"] for r in rows], ["one", "", "Two", "new york"])
        self.assertEqual([r["status"] for r in rows], ["ok", "empty", "ok", "unsupported_multiword"])
        code, out = self.run_cli("predict", "--model", str(self.out), "--word", "one", "--json", "--top-k", "2")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["status"], "ok")
        self.assertIsNone(json.loads(out)["pronunciation"])        # letters model: no pronunciation used

    def test_inspect_prepare_and_errors(self):
        code, out = self.run_cli("inspect", "--model", str(self.out))
        info = json.loads(out)
        self.assertEqual((code, info["mode"], info["requires"]), (0, "letters", ["nothing besides numpy"]))
        npz = self.out / "model.npz"
        man = Path(self.tmp.name) / "res.json"
        man.write_text(json.dumps({"files": [{"path": "m/model.npz", "sha256": ps.sha256_file(npz), "role": "model"}]}))
        self.assertEqual(self.run_cli("prepare", "--manifest", str(man))[0], 0)
        man.write_text(json.dumps({"files": [{"path": "m/model.npz", "sha256": "0" * 64, "role": "model"}]}))
        self.assertEqual(self.run_cli("prepare", "--manifest", str(man))[0], 1)
        self.assertEqual(self.run_cli("predict", "--model", str(self.out / "nope.npz"), "--word", "a")[0], 2)
        self.assertEqual(self.run_cli("predict", "--model", str(self.out), "--word", "a", "--sha256", "0" * 64)[0], 2)


class TestRunner(unittest.TestCase):
    def setUp(self):
        import english_phonetic_run as R
        self.R = R
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfgfile = self.root / "config.yaml"
        self.cfgfile.write_text("x: 1\n")
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, fail: set[str] = frozenset(), score: dict | None = None, verify=None):
        score = score or {"letters": 0.3, "phones": 0.4, "phones_p2g": 0.2}

        def commands(j, runs):
            self.calls.append(j["id"])
            folder = runs / j["id"]
            code = 1 if j["id"] in fail else 0
            arm = j["arm"] or "p2g"
            body = (f"import json,pathlib,sys; p=pathlib.Path({str(folder)!r}); p.mkdir(parents=True, exist_ok=True); "
                    f"(p/'done.json').write_text(json.dumps({{'best_dev_primary_macro': {score.get(arm, 0)}}})); "
                    f"sys.exit({code})")
            return [[sys.executable, "-c", body, j["id"]]]

        def ok(j, runs):
            return (runs / j["id"] / "done.json").exists() and j["id"] not in fail if verify is None else verify(j, runs)

        return self.R.Runner(self.cfgfile, self.root / "state", self.root / "runs", commands, ok, [self.cfgfile],
                             poll=0.05)

    def test_full_queue_phones_wins(self):
        r = self.make()
        with mock.patch.object(self.R.subprocess, "call", return_value=0):
            self.assertEqual(r.resume(), 0)
        st = r.load()
        self.assertEqual(st["selection"]["winner"], "phones")
        self.assertEqual(st["budget"]["supervised_attempts_used"], 7)
        self.assertEqual(st["budget"]["p2g_jobs_used"], 1)
        self.assertEqual(sorted(self.calls), sorted(["letters_s0", "phones_s0", "p2g_s0", "phones_p2g_s0",
                                                     "letters_s1", "phones_s1", "letters_s2", "phones_s2"]))
        self.calls.clear()
        with mock.patch.object(self.R.subprocess, "call", return_value=0):
            self.assertEqual(r.resume(), 0)                         # already done: nothing relaunched
        self.assertEqual(self.calls, [])

    def test_letters_wins_tie_no_confirmation(self):
        r = self.make(score={"letters": 0.4, "phones": 0.4, "phones_p2g": 0.4})
        with mock.patch.object(self.R.subprocess, "call", return_value=0):
            r.resume()
        st = r.load()
        self.assertEqual(st["selection"]["winner"], "letters")
        self.assertEqual(len(st["jobs"]), 4)

    def test_failure_stops_queue(self):
        r = self.make(fail={"phones_s0"})
        self.assertEqual(r.resume(), 1)
        st = r.load()
        self.assertEqual({j["id"]: j["state"] for j in st["jobs"]}["phones_s0"], "failed")
        self.assertNotIn("p2g_s0", self.calls)

    def test_changed_config_stops(self):
        r = self.make()
        r.load()
        self.cfgfile.write_text("x: 2\n")
        self.assertEqual(r.resume(), 2)
        self.assertEqual(self.calls, [])

    def test_budget_exhaustion(self):
        r = self.make()
        st = r.load()
        st["budget"]["supervised_attempts_used"] = 7
        r.save(st)
        r.resume()
        st = r.load()
        self.assertEqual(st["jobs"][0]["state"], "unavailable")
        self.assertIn("cap", st["jobs"][0]["reason"])

    def test_live_job_waited_not_relaunched_and_stale_pid(self):
        r = self.make()
        st = r.load()
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)", "letters_s0"])
        st["jobs"][0].update(state="running", pid=proc.pid, attempts=1)
        st["budget"]["supervised_attempts_used"] = 1
        r.save(st)
        j = r.load()
        t0 = time.time()
        with mock.patch.object(self.R.subprocess, "call", return_value=0):
            r.resume()
        self.assertGreaterEqual(time.time() - t0, 0.4)
        proc.wait()
        st = r.load()
        self.assertEqual(st["jobs"][0]["attempts"], 1)            # interrupted → resumed, not a new attempt
        self.assertEqual(st["budget"]["supervised_attempts_used"], 7)
        self.assertFalse(self.R.pid_alive(proc.pid, "letters_s0"))  # stale pid is not "alive"

    def test_partial_outputs_resume_as_same_attempt(self):
        calls = {"n": 0}

        def verify(j, runs):
            return (runs / j["id"] / "done.json").exists() and (runs / j["id"] / "export.ok").exists()

        r = self.make(verify=verify)
        st = r.load()
        (self.root / "runs" / "letters_s0").mkdir(parents=True)
        (self.root / "runs" / "letters_s0" / "done.json").write_text("{}")   # done.json but no export: partial
        st["jobs"][0].update(state="interrupted", attempts=1)
        st["budget"]["supervised_attempts_used"] = 1
        r.save(st)
        self.assertEqual(r.resume(), 1)                                      # never verifies → failed
        st = r.load()
        self.assertEqual((st["jobs"][0]["state"], st["jobs"][0]["attempts"]), ("failed", 1))
        self.assertEqual(st["budget"]["supervised_attempts_used"], 1)


if __name__ == "__main__":
    unittest.main()
