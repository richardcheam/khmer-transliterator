"""Focused checks for the unified transliterator experiment (D173): leakage, alternatives, loss masking, paired
spelling exposure, isolated aux randomness, truncation, export and inference.

Data checks run in any environment once `translit_unified.py data` has been run. The torch checks are skipped
without torch:

    .venv/bin/python -m unittest tests.test_translit_unified -v
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import translit_unified as tu  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None
BUILT = (tu.D / "manifest.json").exists()


def tiny_data(n_spell: int = 300, n_aux: int = 80):
    """A small slice of the real built data, with the real vocabularies (fast training in tests)."""
    spell, aux, vals, manifest = tu.load_data()
    return spell[:n_spell], aux[:n_aux], {k: v[:20] for k, v in vals.items()}, manifest


class LeakFilterTests(unittest.TestCase):
    """The filter keeps every held-out word, its variants and its Khmer target out of all training tasks."""

    def test_normalisation_and_set_policy(self):
        self.assertEqual(tu.norm("Wi-Fi"), "wifi")
        sets = tu.match_sets({"names_test2": {"pairs": [("sokha", "សុខា")]}, "list_test": {"pairs": [("share", "ស៊ែរ")]}})
        self.assertEqual(tu.latin_hits("sokhas", sets), [])            # no suffix stripping for names
        self.assertEqual(tu.latin_hits("Sokha", sets), ["names_test2"])  # exact normalised name match
        self.assertEqual(tu.latin_hits("sharing", sets), ["list_test"])  # English inflection still matches
        self.assertEqual(tu.row_hits("other", "សុខា", sets), ([], ["names_test2"]))   # Khmer-target rule kept

    @unittest.skipUnless(BUILT, "run translit_unified.py data first")
    def test_manifest_counts_agree_with_filter(self):
        m = json.loads((tu.D / "manifest.json").read_text(encoding="utf-8"))
        sets = tu.match_sets(tu.heldout())
        rows = [l.split("\t") for l in (tu.D / "spell_train.tsv").read_text(encoding="utf-8").splitlines()[1:]]
        for name, info in m["sources"].items():
            self.assertEqual(sum(r[2] == name for r in rows), info["kept"], name)
            self.assertEqual(info["rows_in"], info["kept"] + sum(info["dropped"].values()), name)
        mined = tu._tsv(tu.ROOT / "data/extra_train.tsv")
        lat = sum(1 for a, b in mined if tu.latin_hits(a.lower(), sets))
        km = sum(1 for a, b in mined if not tu.latin_hits(a.lower(), sets) and tu.row_hits(a, b, sets)[1])
        self.assertEqual((lat, km), (m["sources"]["mined"]["dropped"]["latin_related"],
                                     m["sources"]["mined"]["dropped"]["khmer_target"]))

    @unittest.skipUnless(BUILT, "run translit_unified.py data first")
    def test_validation_has_no_evaluation_material(self):
        held = tu.heldout()
        ev = tu.match_sets({n: held[n] for n in tu.EVAL_SETS})
        for v in tu.VAL_SETS:
            self.assertTrue(held[v]["pairs"], v)
            self.assertFalse([p for p in held[v]["pairs"] if any(tu.row_hits(p[0], p[1], ev))], v)
        for n in ("romanised", "names", "english"):
            val = set(tu._tsv(tu.D / f"val_{n}.tsv"))
            self.assertFalse([p for p in val if any(tu.row_hits(p[0], p[1], ev))], n)

    def test_variants_are_related(self):
        held = tu.stems("tablets")
        self.assertTrue(tu.related("tablet", held))
        self.assertTrue(tu.related("tablets", held))
        self.assertFalse(tu.related("table", held))

    @unittest.skipUnless(BUILT, "run translit_unified.py data first")
    def test_built_training_files_have_no_heldout_material(self):
        sets = tu.match_sets(tu.heldout())
        rows = [l.split("\t") for l in (tu.D / "spell_train.tsv").read_text(encoding="utf-8").splitlines()[1:]]
        self.assertTrue(rows)
        self.assertFalse([r for r in rows if any(tu.row_hits(r[0], r[1], sets))])
        aux = [l.split("\t") for l in (tu.D / "aux_train.tsv").read_text(encoding="utf-8").splitlines()[1:]]
        self.assertFalse([r for r in aux if tu.latin_hits(r[0], sets)])
        # the pronunciation diagnostic words are validation words, so never in the aux training rows
        diag = {l.split("\t")[0] for l in (tu.D / "pron_diag.tsv").read_text(encoding="utf-8").splitlines()[1:]}
        self.assertFalse(diag & {r[0] for r in aux})

    @unittest.skipUnless(BUILT, "run translit_unified.py data first")
    def test_no_chhunneng_training_source(self):
        manifest = json.loads((tu.D / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["sources"]), {"roman", "mined", "list_train", "owner_examples", "llm_drafts"})
        sources = {l.split("\t")[2] for l in (tu.D / "spell_train.tsv").read_text(encoding="utf-8").splitlines()[1:]}
        self.assertNotIn("chh_dict", sources)


class AuxDataTests(unittest.TestCase):
    """Alternative pronunciations stay separate; empty labels are excluded."""

    @unittest.skipUnless(BUILT, "run translit_unified.py data first")
    def test_alternatives_kept_and_no_empty(self):
        aux = [l.split("\t") for l in (tu.D / "aux_train.tsv").read_text(encoding="utf-8").splitlines()[1:]]
        self.assertTrue(all(r[1].strip() for r in aux))
        words = [r[0] for r in aux]
        self.assertGreater(len(words), len(set(words)))       # some words keep two pronunciations
        source = {(w, " ".join(p)) for w, p in tu.google_latin() if p}
        self.assertTrue({(r[0], r[1]) for r in aux} <= source)


class GateTests(unittest.TestCase):
    """Decisions use exact integer-derived values; only registered seeds and current-manifest runs count."""

    def test_regression_guard_is_exact(self):
        c = {s: {"pick_test": (60, 50, 150), "romanised_test": (1000 - loss, 1000, 2226), "names_test2": (100, 100, 228)}
             for s, loss in zip((0, 1, 2), (22, 22, 23))}
        g = tu.gate(c)
        self.assertLess(g["romanised_mean_pp"], -1.0)
        self.assertFalse(g["passes"])                 # rounded −1.00 would wrongly pass

    def test_gate_requires_exactly_three_paired_seeds(self):
        good = {"pick_test": (60, 50, 150), "romanised_test": (1000, 1000, 2226), "names_test2": (100, 100, 228)}
        self.assertTrue(tu.gate({0: good, 1: good, 2: good})["passes"])
        self.assertFalse(tu.gate({0: good, 1: good})["passes"])
        self.assertFalse(tu.gate({0: good, 1: good, 2: good, 3: good})["passes"])

    @unittest.skipUnless(BUILT, "run translit_unified.py data first")
    def test_valid_runs_ignores_stale_and_unrelated(self):
        import shutil
        made = []
        try:
            for name, manifest in (("base_s0", "stale"), ("aux_s9", None), ("old_aux_s0", None)):
                run = tu.D / name
                if run.exists():
                    continue
                run.mkdir()
                made.append(run)
                (run / "translit.npz").write_bytes(b"x")
                (run / "train_log.json").write_text(json.dumps({"manifest_sha256": manifest or tu.sha(tu.D / "manifest.json")}))
            self.assertFalse(set(tu.valid_runs()) & {r.name for r in made})
        finally:
            for run in made:
                shutil.rmtree(run)


class LengthTests(unittest.TestCase):
    """No silent truncation (the earlier `max_in` slip)."""

    def test_encode_refuses_long_source(self):
        if not HAS_TORCH:
            self.skipTest("torch")
        with self.assertRaises(ValueError):
            tu.encode_batch(["a" * (tu.MAX_IN + 1)], "abcdefghijklmnopqrstuvwxyz")

    @unittest.skipUnless(BUILT, "run translit_unified.py data first")
    def test_built_lengths_within_caps(self):
        m = json.loads((tu.D / "manifest.json").read_text(encoding="utf-8"))["lengths"]
        self.assertLessEqual(m["max_latin_seen"], m["max_in"])
        self.assertLessEqual(m["max_khmer_seen"], m["max_out"])
        self.assertLessEqual(m["max_aux_seen"], m["aux_max"])


@unittest.skipUnless(HAS_TORCH and BUILT, "needs torch and the built data")
class TrainingControlTests(unittest.TestCase):
    """Loss masking, identical initial parameters, identical spelling exposure and RNG stream, export parity."""

    def test_loss_averages_over_valid_tokens_only(self):
        import torch
        from torch import nn
        ce = nn.CrossEntropyLoss(ignore_index=tu.PAD)
        logits = torch.randn(2, 3, 5)
        tgt = torch.tensor([[3, 4, tu.PAD], [3, tu.PAD, tu.PAD]])
        manual = nn.functional.cross_entropy(logits.reshape(-1, 5)[[0, 1, 3]], tgt.reshape(-1)[[0, 1, 3]])
        self.assertAlmostEqual(ce(logits.reshape(-1, 5), tgt.reshape(-1)).item(), manual.item(), places=6)

    def test_shared_init_identical_across_arms(self):
        a, _ = tu.build_models(30, 80, 60, "base", 1)
        b, aux = tu.build_models(30, 80, 60, "aux", 1)
        self.assertIsNotNone(aux)
        for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
            self.assertEqual(ka, kb)
            self.assertTrue(bool((va == vb).all()), ka)

    def test_paired_exposure_and_isolated_aux_rng(self):
        data = tiny_data()
        with tempfile.TemporaryDirectory() as tmp:
            rb = tu.train("base", 0, epochs=1, out=Path(tmp) / "b", data=data, trace=True)
            ra = tu.train("aux", 0, epochs=1, out=Path(tmp) / "a", data=data, trace=True)
            self.assertEqual(rb["spell_steps"], ra["spell_steps"])
            self.assertEqual(rb["batch_order_sha256"], ra["batch_order_sha256"])
            self.assertEqual(rb["rng_trace"], ra["rng_trace"])     # aux steps leave the dropout stream untouched
            self.assertEqual(rb["aux_steps"], 0)
            self.assertEqual(ra["aux_steps"], -(-ra["spell_steps"] // tu.AUX_EVERY))
            npz = tu.export(Path(tmp) / "a")
            import numpy as np
            keys = set(np.load(npz).files)
            self.assertFalse([k for k in keys if k.startswith("aux")])
            from khmer_translit import translit
            import torch
            from translit_train import greedy
            ck = torch.load(Path(tmp) / "a" / "best.pt")
            model, _ = tu.build_models(len(ck["src_vocab"]) + 3, len(ck["tgt_vocab"]) + 3, 4, "base", 0)
            model.load_state_dict(ck["state"])
            model.eval()
            words = [w for w, _ in data[0][:12]]
            ids = greedy(model, tu.encode_batch(words, ck["src_vocab"]))
            torch_out = ["".join(ck["tgt_vocab"][t - 3] for t in r if t >= 3) for r in ids]
            np_out = [nb[0][0] if nb else "" for nb in translit.load(npz).nbest_batch(words, 1)]
            # compare words whose greedy decode finished (an unfinished beam is empty in NumPy, cut in torch)
            done = [i for i, r in enumerate(ids) if len(r) <= translit.MAX_OUT]
            self.assertGreaterEqual(len(done), 5)
            self.assertEqual([torch_out[i] for i in done], [np_out[i] for i in done])


if __name__ == "__main__":
    unittest.main()
