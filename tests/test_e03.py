"""Tests of E03 (`scripts/e03_data.py`, `e03_train.py`, `e03_run.py`, `e03_eval.py`): implementation behaviour only.

They check the repaired selection rule, the screens, the frozen-rule reuse, the trainer wrapper's pairing and
export, the queue and its budget, and the evaluation's access guard. They say nothing about linguistic quality.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import e02_diagnostics as DG  # noqa: E402
import e03_data as E  # noqa: E402
from khmer_translit import phonetic_source as ps  # noqa: E402

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_BUILD = (E.OUT / "candidates.jsonl").exists() and (E.OUT / "manifest.json").exists()
LEX = {"គណនី": ["k ɔ . n ɔ . n ii"], "មាស": ["m ie h"], "ខាង": ["kh aa ŋ"], "ត្បូង": ["t ɓ oo ŋ"]}


class TestRepairedRule(unittest.TestCase):
    def test_known_khmer_target_alone_is_not_a_rejection(self):
        c = E.classify("paris", "ប៉ារីស", frozenset(), frozenset({"ប៉ារីស"}), frozenset(), frozenset({"paris"}))
        self.assertEqual(c["old_class"], "khmer_like")                    # E02 dropped it
        self.assertEqual(E.recovery_reason(c), "recovered:khmer_target_known_only")
        c = E.classify("coffee", "កាហ្វេ", frozenset(), frozenset(), frozenset({"កាហ្វេ"}), frozenset())
        self.assertEqual(E.recovery_reason(c), "recovered:khmer_target_known_only")

    def test_romanised_khmer_input_still_excluded_and_collisions_recorded(self):
        c = E.classify("sophea", "សុភា", frozenset({"sophea"}), frozenset({"សុភា"}), frozenset(), frozenset())
        self.assertEqual(E.recovery_reason(c), "excluded:latin_romanised")
        c = E.classify("long", "ឡុង", frozenset({"long"}), frozenset({"ឡុង"}), frozenset(), frozenset({"long"}))
        self.assertTrue(c["romanised_english_collision"])
        self.assertEqual(E.recovery_reason(c), "excluded:latin_romanised_collision")
        self.assertIsNone(E.recovery_reason(E.classify("x", "ក", frozenset(), frozenset(), frozenset(), frozenset())))

    def test_compound_extra_components_flagged(self):
        self.assertEqual(E.compound_extra_component("block", "ប្លុកគណនី", LEX, ("B", "L", "AA1", "K")), "គណនី")
        self.assertEqual(E.compound_extra_component("button", "ប៊ូតុងមាស", LEX, ("B", "AH1", "T", "AH0", "N")), "មាស")
        self.assertEqual(E.compound_extra_component(
            "carolina", "ការ៉ូលីណាខាងត្បូង", LEX, tuple("K EH2 R AH0 L AY1 N AH0".split())), "ខាងត្បូង")
        self.assertIsNone(E.compound_extra_component("coffee", "កាហ្វេ", LEX, ("K", "AA1", "F", "IY0")))

    def test_heldout_family_and_target_exclusions(self):
        args = dict(kept=["ខុនណិច"], cmu={"connected": (), "zebra": ()}, held_words={"connect"},
                    held_stems={"connect"}, held_targets={"ហ្សេប្រា"}, tgt_chars=set("ខុនណិចហ្សេប្រា"))
        self.assertEqual(E.filter_reason("connected", "connected", "ok", "ខុនណិច", **args), "heldout_related")
        args["kept"] = ["ហ្សេប្រា"]
        self.assertEqual(E.filter_reason("zebra", "zebra", "ok", "ហ្សេប្រា", **args), "heldout_target")
        args["held_targets"] = set()
        self.assertIsNone(E.filter_reason("zebra", "zebra", "ok", "ហ្សេប្រា", **args))
        self.assertEqual(E.filter_reason("zebra", "zebra", "ok", "ហ្សេប្រា", **{**args, "kept": []}), "noise_filter")

    def test_frozen_screen_functions_unchanged_since_e4c0857(self):
        old = subprocess.run(["git", "show", "e4c0857:scripts/e02_diagnostics.py"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout
        for f in (DG.en_skeleton, DG.google_skeleton, DG.km_skeleton, DG.align, DG.compat_row, DG.is_coda,
                  DG.coda_seg):
            self.assertIn(inspect.getsource(f), old, f.__name__)
        self.assertIn(f"ADAPT = {{", old)


@unittest.skipUnless(HAVE_BUILD, "needs the E03 build in data/e03")
class TestBuildConsistency(unittest.TestCase):
    def test_counts_match_retained_rows(self):
        rows = [json.loads(l) for l in (E.OUT / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        audit = json.loads((E.REPORTS / "data_audit.json").read_text(encoding="utf-8"))
        dec = {}
        for r in rows:
            dec[r["decision"]] = dec.get(r["decision"], 0) + 1
        self.assertEqual(sum(audit["filters_in_order"].values()), audit["recovery"]["recovered:khmer_target_known_only"])
        self.assertEqual(len(rows), audit["recovery"]["recovered:khmer_target_known_only"])
        for k, v in audit["decisions"].items():
            self.assertEqual(dec.get(k, 0), v, k)
        for k, v in audit["filters_in_order"].items():
            if k != "passed":
                self.assertEqual(dec.get("filtered:" + k, 0), v, k)
        ret = [r for r in rows if r["decision"] == "retained"]
        self.assertEqual(len(ret), audit["retained"]["pairs"])
        train = (E.OUT / "train.tsv").read_text(encoding="utf-8").splitlines()[1:]
        self.assertEqual(len(train), sum(r["split"] == "train" for r in ret))
        src = {}
        for r in ret:
            src["+".join(r["sources"])] = src.get("+".join(r["sources"]), 0) + 1
        self.assertEqual(src, audit["retained"]["sources"])
        self.assertEqual(audit["selection_rule_reproduction"].get("differ", 0), 0)
        fam_split = {}
        for r in ret:                                                   # no family crosses splits
            self.assertEqual(fam_split.setdefault(r["family"], r["split"]), r["split"])
        held = {r["target"] for r in ret if r["split"] != "train"}
        self.assertFalse(any(r["target"] in held for r in ret if r["split"] == "train"))
        self.assertTrue(all(r["screen"]["category"] == "compatible_evidence" and r["khmer_readings"] for r in ret))


@unittest.skipUnless(HAVE_TORCH and ps.CMUDICT.exists(), "needs torch and CMUdict")
class TestTrainerWrapper(unittest.TestCase):
    def test_paired_arms_and_e03_export(self):
        import english_phonetic_train as T
        import e03_train as W
        from test_english_training import cfg, tiny_data
        with tempfile.TemporaryDirectory() as t, mock.patch.multiple(
                T, DATA=T.DATA, CONFIG=T.CONFIG, ARMS=T.ARMS, experiment=T.experiment, data_inputs=T.data_inputs), \
                mock.patch.dict("os.environ", {"WANDB_MODE": "disabled"}):
            W.configure()
            self.assertEqual(T.ARMS, ("letters", "phones"))
            data = Path(t) / "data"
            tiny_data(data)
            T.DATA = data
            T.data_inputs = lambda: [data / "train.tsv"]
            a = T.train("letters", 1, Path(t) / "l", cfg(epochs=2))
            b = T.train("phones", 1, Path(t) / "p", cfg(epochs=2))
            self.assertEqual((a["init_sha256"], a["batch_order_sha256"]), (b["init_sha256"], b["batch_order_sha256"]))
            with self.assertRaises(ValueError):
                T.train("phones_p2g", 1, Path(t) / "x", cfg(epochs=1))
            W.export(Path(t) / "p", cfg())
            self.assertTrue(T.verify(Path(t) / "p", n=3)["ok"])
            from khmer_translit.english import Transliterator
            meta = Transliterator.from_pretrained(Path(t) / "p" / "model.npz").info()
            self.assertEqual(meta["experiment"], "E03-assessable-phones")
            self.assertIn("E03", meta["evidence_status"])
            run_cfg = json.loads((Path(t) / "p" / "config.json").read_text())
            self.assertEqual(run_cfg["arm"], "phones")


class TestRunner(unittest.TestCase):
    def make(self, root: Path, fail=frozenset()):
        import e03_run as RR
        calls = []

        def commands(j, runs):
            calls.append(j["id"])
            body = (f"import pathlib,sys; p=pathlib.Path({str(runs / j['id'])!r}); p.mkdir(parents=True, exist_ok=True);"
                    f" (p/'done.json').write_text('{{}}'); sys.exit({1 if j['id'] in fail else 0})")
            return [[sys.executable, "-c", body, j["id"]]]

        def verify(j, runs):
            return (runs / j["id"] / "done.json").exists() and j["id"] not in fail

        cfgf = root / "config.yaml"
        cfgf.write_text("x: 1\n")
        r = RR.E03Runner(cfgf, root / "state", root / "runs", commands, verify, [cfgf], poll=0.01,
                         eval_cmd=[sys.executable, "-c", "print('eval')"])
        return r, calls

    def test_full_queue_six_jobs_then_evaluation(self):
        with tempfile.TemporaryDirectory() as t:
            r, calls = self.make(Path(t))
            self.assertEqual(r.resume(), 0)
            st = r.load()
            self.assertEqual(calls, ["letters_s0", "phones_s0", "letters_s1", "phones_s1", "letters_s2", "phones_s2"])
            self.assertEqual((st["phase"], st["budget"]["supervised_attempts_used"], st["final_eval"]["exit"]),
                             ("done", 6, 0))
            self.assertEqual(r.resume(), 0)                               # idempotent: nothing relaunched
            self.assertEqual(len(calls), 6)

    def test_failure_stops_and_budget_counts_failed_attempts(self):
        with tempfile.TemporaryDirectory() as t:
            r, calls = self.make(Path(t), fail={"phones_s1"})
            self.assertEqual(r.resume(), 1)
            st = r.load()
            self.assertEqual(calls[-1], "phones_s1")
            self.assertEqual(st["budget"]["supervised_attempts_used"], 4)
            self.assertIsNone(st["final_eval"])
            st["budget"]["supervised_attempts_used"] = 6                  # cap reached: no new attempt allowed
            j = next(x for x in st["jobs"] if x["id"] == "letters_s2")
            self.assertEqual(r.budget_ok(st, j, True), "supervised attempt cap reached")


class TestEvalGuard(unittest.TestCase):
    def test_sets_are_allowed_and_forbidden_paths_refused(self):
        import e03_eval as V
        for p in V.SETS.values():
            DG.guarded(p)                                                 # no PermissionError
        for bad in ("data/e02/references.jsonl", "data/e02/results/per_item.json", "data/e2/reserve.json"):
            with self.assertRaises(PermissionError):
                DG.guarded(ROOT / bad)

    def test_frozen_tamper_detected(self):
        import e03_eval as V
        with tempfile.TemporaryDirectory() as t, mock.patch.object(V, "EXP", Path(t)):
            (Path(t) / "frozen_eval.json").write_text("{}\n")
            (Path(t) / "frozen_eval.sha256").write_text("0" * 64 + "\n")
            with self.assertRaises(RuntimeError):
                V.frozen()


if __name__ == "__main__":
    unittest.main()
