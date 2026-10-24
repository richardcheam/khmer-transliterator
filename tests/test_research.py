"""The research tooling: strict configs, run records, the frozen benchmark and the data manifest. Run:
`.venv/bin/python -m unittest tests.test_research`."""
import json
import os
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from khmer_translit import benchmark, experiment  # noqa: E402

SCHEMA = {"train": {"epochs": 40, "lr": 1e-3, "seeds": [0, 1]}, "aux": {"weight": 0.5}}


def write(tmp: Path, text: str) -> Path:
    path = tmp / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class ConfigTest(unittest.TestCase):
    def test_exact_keys_types_and_overrides(self):
        with tempfile.TemporaryDirectory() as t:
            p = write(Path(t), "experiment: E9\ntrain: {epochs: 40, lr: 1, seeds: [0, 1]}\naux: {weight: 0.5}\n")
            cfg = experiment.load_config(p, SCHEMA, ["train.epochs=2"])
            self.assertEqual((cfg["experiment"], cfg["train"]["epochs"], cfg["overrides"]), ("E9", 2, ["train.epochs=2"]))
            for text, why in (("train: {epochs: 40, lr: 1, seeds: [0]}\n", "missing section"),
                              ("train: {epochs: 40, seeds: [0]}\naux: {weight: 0.5}\n", "missing key"),
                              ("train: {epochs: 40, lr: 1, seeds: [0], x: 1}\naux: {weight: 0.5}\n", "unknown key"),
                              ("train: {epochs: 4.5, lr: 1, seeds: [0]}\naux: {weight: 0.5}\n", "wrong type"),
                              ("train: {epochs: true, lr: 1, seeds: [0]}\naux: {weight: 0.5}\n", "bool for int")):
                with self.assertRaises(ValueError, msg=why):
                    experiment.load_config(write(Path(t), text), SCHEMA)
            with self.assertRaises(ValueError):
                experiment.load_config(p, SCHEMA, ["epochs=1"])

    def test_registered_e01_config_matches_the_code(self):
        import translit_unified as tu
        cfg = experiment.load_config(tu.CONFIG, tu.CONFIG_KEYS)
        tu.check_implemented(cfg)
        self.assertEqual({k: cfg[k] for k in tu.CONFIG_KEYS}, tu.CONFIG_KEYS)
        with self.assertRaises(ValueError):
            tu.check_implemented(experiment.load_config(tu.CONFIG, tu.CONFIG_KEYS, ["model.d=256"]))


class RunTest(unittest.TestCase):
    def test_records_without_wandb(self):
        old = os.environ.get("WANDB_MODE")
        os.environ["WANDB_MODE"] = "disabled"
        try:
            with tempfile.TemporaryDirectory() as t:
                inp = Path(t) / "in.txt"
                inp.write_text("x")
                run = experiment.Run(Path(t) / "r", "E9", "r", {"a": 1}, [inp])
                run.log({"val": {"names": 0.5}}, step=1)
                run.finish({"best": 1})
                folder = Path(t) / "r"
                prov = json.loads((folder / "provenance.json").read_text())
                self.assertEqual(len(prov["git_commit"]), 40)
                self.assertEqual(list(prov["inputs"].values()), [experiment.sha256(inp)])
                self.assertEqual(json.loads((folder / "metrics.jsonl").read_text()), {"step": 1, "val": {"names": 0.5}})
                self.assertEqual(json.loads((folder / "summary.json").read_text()), {"best": 1})
        finally:
            os.environ.pop("WANDB_MODE") if old is None else os.environ.__setitem__("WANDB_MODE", old)


class BenchmarkTest(unittest.TestCase):
    def test_outcomes_and_bootstrap(self):
        panel = [("ka", frozenset({"ក"})), ("kha", frozenset({"ខ", "គ"}))]
        self.assertEqual(benchmark.outcomes(["ក", "គ"], panel), [1, 1])
        with self.assertRaises(ValueError):
            benchmark.outcomes(["ក"], panel)
        self.assertEqual(benchmark.paired_bootstrap([Fraction(0)] * 20), (0.0, 0.0))
        lo, hi = benchmark.paired_bootstrap([Fraction(1), Fraction(0)] * 50)
        self.assertTrue(lo < 50 < hi)

    @unittest.skipUnless((benchmark.FILES / "v1").exists(), "benchmark v1 panel files not present")
    def test_v1_loads_and_is_frozen(self):
        panels = benchmark.load("v1")
        self.assertEqual({k: len(v) for k, v in panels.items()},
                         {"romanised_test": 2226, "names_test2": 223, "pick_test": 150, "dev41": 41, "list_test": 125})
        with self.assertRaises(FileExistsError):
            benchmark.freeze(panels, "v1", {}, "")

    def test_tampered_panel_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            old = benchmark.MANIFESTS, benchmark.FILES
            benchmark.MANIFESTS, benchmark.FILES = Path(t) / "m", Path(t) / "f"
            try:
                benchmark.freeze({"p": [("ka", frozenset({"ក"}))]}, "vT", {"p": "test"}, "note")
                self.assertEqual(len(benchmark.load("vT")["p"]), 1)
                (Path(t) / "f" / "vT" / "p.tsv").write_text("ka\tខ\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    benchmark.load("vT")
            finally:
                benchmark.MANIFESTS, benchmark.FILES = old


class DataManifestTest(unittest.TestCase):
    def test_compare(self):
        import data_manifest
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "a.txt").write_text("1")
            old = data_manifest.scan(Path(t))
            (Path(t) / "a.txt").write_text("2")
            (Path(t) / "b.txt").write_text("3")
            self.assertEqual(data_manifest.compare(old, data_manifest.scan(Path(t))),
                             {"changed": ["a.txt"], "missing": [], "added": ["b.txt"]})


if __name__ == "__main__":
    unittest.main()
