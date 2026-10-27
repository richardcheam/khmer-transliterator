"""Tests of the E02 diagnostics (`scripts/e02_diagnostics.py`): implementation behaviour, not linguistic validity."""
from __future__ import annotations

import builtins
import doctest
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import e02_diagnostics as D  # noqa: E402
from khmer_translit import phonetic_source as ps  # noqa: E402

HAVE_DATA = D.INPUTS["train"].exists() and D.INPUTS["lexicon"].exists() and D.INPUTS["cmudict"].exists()
HAVE_MODELS = all((ROOT / "models" / "e02" / m / "model.npz").exists() for m in ("phones_s0", "letters_s0"))
FROZEN = D.OUT / "sensitivity" / "frozen.json"


class FakeModel:
    """Just what the encoding helpers read from a `Transliterator`."""

    def __init__(self, mode: str, cmu: Path):
        self.mode, self.vocab, self.cmudict_path = mode, ps.ENGLISH_VOCAB, cmu


class Spy:
    """Record every path opened through `open` / `io.open` while active."""

    def __init__(self):
        self.paths: list[str] = []
        self._open = builtins.open

    def __enter__(self):
        def spy(file, *a, **k):
            if isinstance(file, (str, Path)):
                self.paths.append(str(Path(file).resolve()))
            return self._open(file, *a, **k)
        self._patches = [mock.patch("builtins.open", spy), mock.patch("io.open", spy)]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()

    def forbidden(self) -> list[str]:
        return [p for p in self.paths if any(f in Path(p).as_posix() for f in D.FORBIDDEN)]


class TestReadingRules(unittest.TestCase):
    def test_doctests(self):
        self.assertEqual(doctest.testmod(D, verbose=False).failed, 0)

    def test_adaptations_and_conflicts(self):
        self.assertEqual(D.align(D.en_skeleton("K AA1 F IY0".split()), D.km_skeleton("កាហ្វេ"))[0], 0)
        self.assertEqual(D.align(D.en_skeleton("V IH1 L AH0".split()), D.km_skeleton("វីឡា"))[0], 0)   # v→w
        self.assertGreater(D.align(D.en_skeleton("JH EH1 L B".split()), D.km_skeleton("ហ្គេល្ប"))[0], 0)  # j→g
        self.assertIsNone(D.km_skeleton("abc"))                                              # unreadable

    def test_silencing_and_coda(self):
        self.assertEqual([s.label for s in D.km_skeleton("ឡៃវ៍")], ["ឡ"])                    # ៍ silences វ
        self.assertTrue(D.km_skeleton("ការ")[-1].optional)                                  # coda r may be silent
        self.assertEqual(sorted(D.km_skeleton("ប៉ារីស")[-1].classes), ["GLOT", "SIB"])      # coda s may be h


class TestCompatRow(unittest.TestCase):
    def test_missing_khmer_pronunciation_uses_orthographic_track(self):
        r = D.compat_row("coffee", "កាហ្វេ", ("K", "AA1", "F", "IY0"), (("K", "AA1", "F", "IY0"),), {}, {})
        self.assertEqual((r["khmer_lookup"]["status"], r["evidence_track"], r["category"]),
                         ("miss", "orthographic", "compatible_evidence"))
        self.assertIn("vowels not assessed", r["reasons"])

    def test_ambiguous_khmer_lookup_takes_any_reading(self):
        lex = {"កាហ្វេ": ["k aa . s ee", "k aa . f ee"]}                  # two readings, the second fits
        r = D.compat_row("coffee", "កាហ្វេ", ("K", "AA1", "F", "IY0"), (("K", "AA1", "F", "IY0"),), lex, {})
        self.assertEqual((r["evidence_track"], r["khmer_lookup"]["readings"], r["category"]),
                         ("google_exact", lex["កាហ្វេ"], "compatible_evidence"))

    def test_unreadable_target_is_unknown(self):
        r = D.compat_row("coffee", "ka", ("K", "AA1", "F", "IY0"), (), {}, {})
        self.assertEqual(r["category"], "unknown")

    def test_alternative_reading_reason(self):
        sup, alt = ("JH", "AA1", "B"), ("G", "AA1", "B")
        r = D.compat_row("gab", "ហ្គាប", sup, (sup, alt), {}, {})
        self.assertEqual(r["category"], "suspected_conflict")
        self.assertTrue(any(x.startswith("fits another CMUdict reading") for x in r["reasons"]))


class TestSensitivityHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cmu = Path(self.tmp.name) / "cmu.dict"
        self.cmu.write_text("record R AH0 K AO1 R D\nrecord(2) R EH1 K ER0 D\nalpha AE1 L F AH0\n"
                            "omega OW0 M EY1 G AH0\nbeta B EY1 T AH0\n", encoding="utf-8")
        ps._cmudict.cache_clear()

    def tearDown(self):
        self.tmp.cleanup()
        ps._cmudict.cache_clear()

    def items(self):
        from khmer_translit.reference_data import ReferenceItem
        return [ReferenceItem(id=f"{w}#default", input=w, word=w, family=w, pron_id="default", intended_phones="",
                              accent="en-US-cmudict", accepted=["ក"], preferred="ក", sources=["t"], tier="attested",
                              split="dev") for w in ("record", "alpha", "omega", "beta")]

    def test_phone_replacement_keeps_letters(self):
        tl = FakeModel("phones", self.cmu)
        for cond in ({"condition": "default", "phones": "R AH0 K AO1 R D"}, {"condition": "no_pron", "phones": None},
                     {"condition": "donor", "phones": "AE1 L F AH0"}, {"condition": "alternative", "phones": "R EH1 K ER0 D"}):
            ids = D.condition_ids(tl, "record", cond)
            self.assertTrue(D.check_encoding(tl, "record", cond, ids), cond)
            self.assertEqual(ids[:7], [tl.vocab.id(t) for t in ["<task:en>", *"record"]])
        wrong = D.condition_ids(tl, "record", {"condition": "donor", "phones": "AE1 L F AH0"})
        self.assertFalse(D.check_encoding(tl, "record", {"condition": "donor", "phones": "B EY1 T AH0"}, wrong))

    def test_letters_control_ignores_phones(self):
        tl = FakeModel("letters", self.cmu)
        ids = {json.dumps(D.condition_ids(tl, "record", c)) for c in (
            {"condition": "default", "phones": "R AH0 K AO1 R D"}, {"condition": "no_pron", "phones": None},
            {"condition": "donor", "phones": "AE1 L F AH0"})}
        self.assertEqual(len(ids), 1)

    def test_training_and_inference_encoding_agree(self):
        import english_phonetic_train as T
        tl = FakeModel("phones", self.cmu)
        with mock.patch.object(ps, "CMUDICT", self.cmu), mock.patch.object(ps.get_pronunciation, "__kwdefaults__",
                                                                           {"supplied": None, "path": self.cmu}):
            train_ids = T.encode_word("record", "phones", ps.ENGLISH_VOCAB)
            train_drop = T.encode_word("record", "phones", ps.ENGLISH_VOCAB, drop=True)
        self.assertEqual(train_ids, D.condition_ids(tl, "record", {"condition": "default", "phones": "R AH0 K AO1 R D"}))
        self.assertEqual(train_drop, D.condition_ids(tl, "record", {"condition": "no_pron", "phones": None}))

    def test_donors_deterministic_and_valid(self):
        cmu, _ = ps.cmudict(self.cmu)
        a, b = D.build_sample(self.items(), cmu), D.build_sample(self.items(), cmu)
        self.assertEqual(a, b)
        strip = lambda s: tuple(p.rstrip("012") for p in s.split())
        for it in a:
            own = cmu[it["word"]][0]
            for c in it["conditions"]:
                if c["condition"] == "donor" and "skipped" not in c:
                    self.assertNotEqual(c["donor_id"], it["id"])
                    self.assertLessEqual(abs(len(c["phones"].split()) - len(own)), 1)
                    self.assertNotEqual(strip(c["phones"]), strip(" ".join(own)))

    def test_alternative_readings_listed(self):
        cmu, _ = ps.cmudict(self.cmu)
        rec = next(it for it in D.build_sample(self.items(), cmu) if it["word"] == "record")
        alts = [c for c in rec["conditions"] if c["condition"] == "alternative"]
        self.assertEqual([(c["phones"], c["stress_only"]) for c in alts], [("R EH1 K ER0 D", False)])

    def test_guard(self):
        for bad in ("data/e02/references.jsonl", "data/e02/results/predictions.json", "data/e2/reserve.json",
                    "data/benchmark/v1/dev41.tsv"):
            with self.assertRaises(PermissionError):
                D.guarded(ROOT / bad)
        self.assertEqual(D.guarded(ROOT / "data/e02/dev_references.jsonl").name, "dev_references.jsonl")


@unittest.skipUnless(HAVE_DATA, "needs data/e02, the Google lexicon and CMUdict")
class TestRealCompat(unittest.TestCase):
    def test_compat_reads_no_forbidden_path_and_matches_default_readings(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(D, "OUT", Path(t)), Spy() as spy:
            s = D.compat()
        self.assertEqual(spy.forbidden(), [])
        self.assertTrue(any(p.endswith("train.tsv") for p in spy.paths))                 # the spy saw the reads
        self.assertEqual(s["coverage"]["supplied_equals_dictionary_default"], s["rows"])   # train = inference reading
        self.assertEqual(sum(s["categories"].values()), s["rows"])


@unittest.skipUnless(HAVE_DATA and HAVE_MODELS and FROZEN.exists(), "needs checkpoints and the frozen sample")
class TestRealSensitivity(unittest.TestCase):
    def test_small_run_alignment_and_no_forbidden_path(self):
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        frozen["sample"] = frozen["sample"][:3]
        frozen["checkpoints"] = {m: frozen["checkpoints"][m] for m in ("phones_s0", "letters_s0")}
        with tempfile.TemporaryDirectory() as t, mock.patch.object(D, "OUT", Path(t)):
            d = Path(t) / "sensitivity"
            d.mkdir()
            body = json.dumps(frozen, ensure_ascii=False, indent=1) + "\n"
            (d / "frozen.json").write_text(body, encoding="utf-8")
            (d / "frozen.sha256").write_text(hashlib.sha256(body.encode()).hexdigest() + "\n")
            with Spy() as spy:
                res = D.sensitivity()
            rows = [json.loads(l) for l in (d / "per_item.jsonl").read_text(encoding="utf-8").splitlines()]
            (d / "frozen.json").write_text(body.replace('"beam": 4', '"beam": 5'), encoding="utf-8")
            with self.assertRaises(RuntimeError):                                          # tampering detected
                D.sensitivity()
        self.assertEqual(spy.forbidden(), [])
        self.assertTrue(any(p.endswith("model.npz") for p in spy.paths))
        self.assertTrue(all(c["encoding_as_intended"] for c in res["checks"].values()))
        self.assertTrue(res["checks"]["letters_s0"]["letters_control_identical"])
        n_conds = sum(1 for it in frozen["sample"] for c in it["conditions"] if "skipped" not in c)
        for m in frozen["checkpoints"]:
            mine = [r for r in rows if r["checkpoint"] == m]
            self.assertEqual(len(mine), n_conds)
            self.assertEqual(len({(r["id"], r["condition"], r["phones"]) for r in mine}), n_conds)
        self.assertTrue(all(not r["top1_changed"] for r in rows if r["mode"] == "letters"))
        self.assertLess(max(c["teacher_forced_vs_beam_max_gap"] for c in res["checks"].values()), 1e-6)


if __name__ == "__main__":
    unittest.main()
