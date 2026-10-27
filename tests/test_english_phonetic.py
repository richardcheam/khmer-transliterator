"""Tests for experiment E02's English phonetic pipeline: source processing, references, scoring and the API.

These check plumbing (normalisation, encoding, alignment, decision arithmetic), not linguistic truth. Tests that
need the local CMUdict copy skip when it is absent.
"""
from __future__ import annotations

import json
import os
import tempfile
import unicodedata
import unittest
from pathlib import Path

from khmer_translit import evaluation as ev
from khmer_translit import phonetic_source as ps
from khmer_translit import reference_data as rd

HAVE_CMU = ps.CMUDICT.exists()


def item(word: str, accepted: list[str], family: str | None = None, split: str = "dev", pron_id: str = "default",
         flags: list[str] | None = None) -> rd.ReferenceItem:
    return rd.ReferenceItem(f"{word}#{pron_id}", word, word, family or word, pron_id, "", "en-US-cmudict", accepted,
                            accepted[0], ["test"], "attested", split, flags or [])


class TestNormalization(unittest.TestCase):
    def test_variants(self):
        for raw in ("Coffee", " coffee ", "COFFEE", "coffée", "cof-fee", "cof'fee", "ｃｏｆｆｅｅ"):
            self.assertEqual(ps.normalize_word(raw), ("coffee", "ok"), raw)

    def test_refusals(self):
        self.assertEqual(ps.normalize_word("")[1], "empty")
        self.assertEqual(ps.normalize_word("  ")[1], "empty")
        self.assertEqual(ps.normalize_word("new york")[1], "unsupported_multiword")
        self.assertEqual(ps.normalize_word("b2b")[1], "unsupported_characters")
        self.assertEqual(ps.normalize_word("កាហ្វេ")[1], "unsupported_characters")
        self.assertEqual(ps.normalize_word("a" * (ps.MAX_LETTERS + 1))[1], "too_long")
        self.assertEqual(ps.normalize_word("a" * ps.MAX_LETTERS)[1], "ok")


class TestPronunciation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cmu = Path(self.tmp.name) / "cmu.dict"
        self.cmu.write_text("record R AH0 K AO1 R D\nrecord(2) R EH1 K ER0 D\nsofa S OW1 F AH0\nsofa(2) S OW1 F AH2\n"
                            "o'neil OW0 N IY1 L\n", encoding="utf-8")
        ps._cmudict.cache_clear()

    def tearDown(self):
        ps._cmudict.cache_clear()
        self.tmp.cleanup()

    def test_default_ambiguous_and_supplied(self):
        r = ps.get_pronunciation("Record", path=self.cmu)
        self.assertEqual((r.status, r.phones, r.ambiguous), ("dictionary", ("R", "AH0", "K", "AO1", "R", "D"), True))
        self.assertEqual(r.alternatives, (("R", "EH1", "K", "ER0", "D"),))
        s = ps.get_pronunciation("record", supplied="r eh1 k er0 d", path=self.cmu)
        self.assertEqual((s.status, s.provider, s.phones[1], s.dictionary_hit), ("provided", "caller", "EH1", True))
        self.assertNotEqual(ps.encode_source("record", r, "phones"), ps.encode_source("record", s, "phones"))
        self.assertEqual(ps.encode_source("record", r, "letters"), ps.encode_source("record", s, "letters"))

    def test_stress_only_variants_not_ambiguous(self):
        self.assertFalse(ps.get_pronunciation("sofa", path=self.cmu).ambiguous)

    def test_apostrophe_headword_and_miss(self):
        self.assertEqual(ps.get_pronunciation("O'Neil", path=self.cmu).status, "dictionary")
        miss = ps.get_pronunciation("zzqx", path=self.cmu)
        self.assertEqual((miss.status, miss.phones, miss.provider), ("unavailable", (), "none"))
        toks = [ps.ENGLISH_VOCAB.tokens[i] for i in ps.encode_source("zzqx", miss, "phones")]
        self.assertEqual(toks[-2:], ["<sep>", "<no_pron>"])

    def test_unknown_phone_and_refused_word(self):
        with self.assertRaises(ps.InputError) as e:
            ps.get_pronunciation("record", supplied="R EH K", path=self.cmu)
        self.assertEqual(e.exception.status, "unknown_phone")
        with self.assertRaises(ps.InputError) as e:
            ps.get_pronunciation("", path=self.cmu)
        self.assertEqual(e.exception.status, "empty")

    def test_missing_resource(self):
        with self.assertRaises(FileNotFoundError):
            ps.get_pronunciation("record", path=Path(self.tmp.name) / "absent.dict")

    def test_replaced_file_is_not_served_from_cache(self):
        f = Path(self.tmp.name) / "versioned.dict"
        f.write_text("tomato T AH0 M EY1 T OW2\n", encoding="utf-8")
        first = ps.get_pronunciation("tomato", path=f)
        f.write_text("tomato T AH0 M AA1 T OW2\n", encoding="utf-8")
        os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 10**9))
        second = ps.get_pronunciation("tomato", path=f)
        self.assertEqual((first.phones[3], second.phones[3]), ("EY1", "AA1"))
        self.assertNotEqual(first.resource_sha256, second.resource_sha256)


class TestEncoding(unittest.TestCase):
    def test_phone_tokens_are_whole(self):
        p = ps.PronunciationResult("one", ("W", "AH1", "N"), "provided", "caller")
        toks = [ps.ENGLISH_VOCAB.tokens[i] for i in ps.encode_source("one", p, "phones")]
        self.assertIn("en:AH1", toks)
        self.assertEqual(len(toks), 1 + 3 + 1 + 3)

    def test_dropout_and_too_long(self):
        p = ps.PronunciationResult("one", ("W", "AH1", "N"), "provided", "caller")
        toks = [ps.ENGLISH_VOCAB.tokens[i] for i in ps.encode_source("one", p, "phones", drop_phones=True)]
        self.assertEqual(toks[-1], "<no_pron>")
        long = ps.PronunciationResult("ab", ("AH0",) * ps.MAX_SRC, "provided", "caller")
        with self.assertRaises(ps.InputError) as e:
            ps.encode_source("ab", long, "phones")
        self.assertEqual(e.exception.status, "too_long")

    def test_english_ids_do_not_depend_on_khmer_inventory(self):
        p = ps.PronunciationResult("one", ("W", "AH1", "N"), "provided", "caller")
        v = ps.SourceVocab(["k", "a", "."])
        self.assertEqual(ps.encode_source("one", p, "phones", v), ps.encode_source("one", p, "phones"))

    def test_p2g_homophones_keep_both_targets(self):
        v = ps.SourceVocab(["k", "a"])
        self.assertEqual(ps.encode_p2g(["k", "a"], v), ps.encode_p2g(["k", "a"], v))
        with self.assertRaises(ps.InputError):
            ps.encode_p2g(["zz"], v)

    def test_khmer_word_check(self):
        self.assertTrue(ps.is_khmer_word("កាហ្វេ"))
        self.assertFalse(ps.is_khmer_word("កាហ្វេ។"))
        self.assertFalse(ps.is_khmer_word("កា ហ្វេ"))


class TestReferences(unittest.TestCase):
    def test_families_and_splits(self):
        fam = rd.group_families(["share", "sharing", "shared", "shares", "table"])
        self.assertEqual(len({fam[w] for w in ("share", "sharing", "shared", "shares")}), 1)
        self.assertNotEqual(fam["table"], fam["share"])

    def test_two_readings_same_word_are_distinct_items(self):
        a = item("record", ["រ៉េខត"], "record", pron_id="default")
        b = item("record", ["រីខត"], "record", pron_id="alt1")
        rd.validate_items([a, b])

    def test_duplicates_and_bad_items_refused(self):
        a = item("ab", ["អាប"])
        with self.assertRaises(ValueError):
            rd.validate_items([a, a])
        with self.assertRaises(ValueError):
            rd.validate_items([item("ab", [""])])
        bad = item("ab", ["អាប"])
        bad.tier = "gold"
        with self.assertRaises(ValueError):
            rd.validate_items([bad])

    def test_malformed_file_row(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"
            rd.save_reference_items([item("ab", ["អាប"])], p)
            p.write_text(p.read_text() + '{"id": "x"}\n')
            with self.assertRaises(ValueError):
                rd.load_reference_items(p)

    def test_collision_detected(self):
        items = [item("sharing", ["ក"], "share", split="final"), item("share", ["ខ"], "share", split="dev")]
        rep = rd.audit_splits(items, training_words=["shared"], training_targets=["ក"])
        self.assertEqual(rep["families_in_several_splits"], ["share"])
        self.assertEqual(rep["train_word_related"]["final"], 1)
        self.assertEqual(rep["train_target_shared"]["final"], 1)


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.items = [item("coffee", ["កាហ្វេ", "ខហ្វី"], "coffee"), item("table", ["តេប៊ល"], "table"),
                      item("tables", ["តេប៊លស៍"], "table")]

    def test_multi_reference_and_nfc(self):
        o = ev.item_outcomes(self.items, {"coffee#default": ["ខហ្វី"], "table#default": ["x"],
                                          "tables#default": [unicodedata.normalize("NFD", "តេប៊លស៍")]})
        self.assertEqual([o[i.id]["top1"] for i in self.items], [1, 0, 1])

    def test_alignment_errors(self):
        with self.assertRaises(ValueError):
            ev.item_outcomes(self.items, {"coffee#default": ["a"]})
        full = {i.id: ["a"] for i in self.items}
        with self.assertRaises(ValueError):
            ev.item_outcomes(self.items, {**full, "zzz#default": ["a"]})
        with self.assertRaises(TypeError):
            ev.item_outcomes(self.items, {**full, "coffee#default": "កាហ្វេ"})

    def test_abstention_is_failure_and_order_irrelevant(self):
        preds = {"tables#default": ["តេប៊លស៍"], "coffee#default": [], "table#default": ["តេប៊ល"]}
        sc = ev.score_predictions(self.items, preds)
        self.assertEqual((sc["correct_items"], sc["abstained"]), (2, 1))
        self.assertAlmostEqual(sc["primary_macro"], 0.5)   # coffee 0, table family (1 + 1) / 2 = 1
        self.assertAlmostEqual(sc["item_accuracy"], 2 / 3)

    def test_measurement_controls(self):
        self.assertTrue(ev.measurement_controls(self.items)["ok"])

    def test_paired_comparison_and_fields(self):
        ids = [i.id for i in self.items]
        c = [{i: 0 for i in ids}] * 3
        t = [{i: 1 for i in ids}] * 3
        res = ev.compare_paired_runs(self.items, c, t, n_boot=200)
        self.assertEqual(res["seed_gains_pp"], [100.0] * 3)
        self.assertEqual(res["ci95_pp"], [100.0, 100.0])
        f = ev.evidence_fields(res["ci95_pp"])
        self.assertTrue(f["positive_effect_supported"])
        self.assertEqual(ev.summary_status(False, f), "inconclusive")
        self.assertEqual(ev.summary_status(True, f), "supported_improvement")
        self.assertFalse(ev.adoption(False, res, True, True)["adopt"])
        self.assertTrue(ev.adoption(True, res, True, True)["adopt"])
        self.assertFalse(ev.adoption(True, res, True, None)["adopt"])

    def test_evidence_field_overlap(self):
        f = ev.evidence_fields([-1.0, 7.0])
        self.assertEqual(list(f.values()), [False, False, False])
        self.assertEqual(ev.summary_status(True, f), "inconclusive")
        self.assertEqual(ev.summary_status(True, ev.evidence_fields([-2.0, 3.0])), "meaningful_gain_not_supported")


if __name__ == "__main__":
    unittest.main()
