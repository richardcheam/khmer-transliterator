"""Tests of E08 (`scripts/e08.py`): recovery rules, input selection rules, prompt rendering and the AI Mode importer.

Every response text below is a hand-written TEST FIXTURE that exercises formatting cases. None of them is, or
imitates, a real Google AI Mode response."""
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
import e08  # noqa: E402

MANIFEST = [{"id": f"e08s{n:03d}", "english": w, "category": "common", "pronunciation_arpabet": ph,
             "pronunciation_ipa": e08.ipa(ph), "batch": b}
            for n, (w, ph, b) in enumerate([("coffee", "K AA1 F IY0", 1), ("london", "L AH1 N D AH0 N", 1),
                                            ("taxi", "T AE1 K S IY0", 1), ("bus", "B AH1 S", 2)], 1)]
NO_PROTECTION = {"words": set(), "stems": set(), "held_targets": set(), "counts": {}}

FIXTURE_BATCH_01 = """TEST FIXTURE (not a real response)
```
| id | english | preferred_khmer | alternative_1 | alternative_2 | uncertain | note |
|---|---|---|---|---|---|---|
| **e08s001** | coffee | `កាហ្វេ` | ខូហ្វី | | no | |
| e08s002 | london | ឡុងដ៍ (London) | | | no | explanation in the spelling cell |
| e08s003 | taxi | តាក់ស៊ី | | | yes | not sure |
| e08s004 | bus | ប៊ុស | | | no | belongs to batch 2 |
| e08s099 | ghost | ហ្គោស | | | no | not in the manifest |
```
"""


class Env:
    """Patch every E08 output path into a temporary folder and use the small fixture manifest."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "exp" / "reports").mkdir(parents=True)
        (d / "exp" / "synthetic" / "prompts").mkdir(parents=True)
        self.patch = mock.patch.multiple(e08, EXP=d / "exp", REPORTS=d / "exp" / "reports", SYN=d / "exp" / "synthetic",
                                         PROMPTS=d / "exp" / "synthetic" / "prompts", IMPORTED=d / "imported",
                                         RESPONSES=d / "responses", load_manifest=lambda: MANIFEST,
                                         protected=lambda: NO_PROTECTION)
        self.patch.start()
        (d / "responses").mkdir()
        self.d = d
        return self

    def __exit__(self, *a):
        self.patch.stop()
        self.tmp.cleanup()


class TestRules(unittest.TestCase):
    def test_doctests(self):
        self.assertEqual(doctest.testmod(e08, verbose=False).failed, 0)

    def test_recovery_rules(self):
        base = {"word": "anna", "target": "អានណា", "possible_translation": False, "compound_extra_component": None,
                "screen": {"category": "suspected_conflict", "level": "1", "ops": [["del_b", "", "n"]], "reasons": []}}
        self.assertEqual(e08.recovery_decision(base), ("recovered", ["doubled_letter"]))
        self.assertEqual(e08.recovery_decision({**base, "word": "ana"})[0], "uncertain")        # no doubled letter
        switch = {**base, "screen": {**base["screen"], "reasons": [e08.READING_SWITCH]}}
        self.assertEqual(e08.recovery_decision(switch)[0], "uncertain")                          # no silent switch
        many = {**base, "word": "class", "target": "កណ្តាល",
                "screen": {"category": "suspected_conflict", "level": "3+", "reasons": [],
                           "ops": [["sub", "K", "k"], ["sub", "L", "ɗ"], ["sub", "S", "l"]]}}
        self.assertEqual(e08.recovery_decision(many)[0], "error_or_translation")
        self.assertEqual(e08.recovery_decision({**base, "screen": {"category": "unknown"}})[0], "uncertain")
        self.assertEqual(e08.explain(["sub", "K", "c"], "alec", "អាឡិច"), "final_ic")
        self.assertEqual(e08.explain(["del_b", "", "j"], "anatolia", "អាណាតូលីយ៉ា"), "y_glide")
        self.assertIsNone(e08.explain(["sub", "S", "h"], "ace", "អេហ"))                        # h not written ស

    def test_selection_rules(self):
        self.assertTrue(e08.clusters(("S", "T", "R", "IY1", "T")))
        self.assertTrue(e08.clusters(("T", "EH1", "K", "S", "T")))
        self.assertFalse(e08.clusters(("K", "AE1", "T")))
        self.assertTrue(e08.is_abbreviation("ftp", ("EH1", "F", "T", "IY1", "P", "IY1")))
        self.assertEqual(e08.stressed_vowel(("K", "AA1", "F", "IY0")), "AA")

    def test_prompt_has_policy_and_no_references(self):
        p = e08.render_prompt(MANIFEST[:3], "test")
        for must in ("not translation", "Never give a Khmer translation", "uncertain", "Keep every id exactly",
                     "up to two alternative", "Do not invent sources or confidence scores",
                     "| id | english | preferred_khmer | alternative_1 | alternative_2 | uncertain | note |"):
            self.assertIn(must, p)
        self.assertNotRegex(p, "[ក-៿]")                    # no Khmer at all: no predictions or references


class TestImporter(unittest.TestCase):
    def test_cells_and_spellings(self):
        self.assertEqual(e08.check_spelling("**កាហ្វេ**"), ("កាហ្វេ", "ok"))
        self.assertEqual(e08.check_spelling("កាហ្វេ​"), ("កាហ្វេ", "ok"))            # zero-width space removed
        self.assertEqual(e08.check_spelling("ក" * 41)[1], "too_long")
        self.assertEqual(e08.check_spelling("кафе")[1], "non_khmer")
        self.assertEqual(e08.check_spelling("កាហ្វេ ឬ ខូហ្វី")[1], "explanatory_text")
        self.assertEqual(e08.check_spelling("ក៊ាា")[1], "malformed_orthography")
        rec = e08.judge(["e08s001", "coffee", " `កាហ្វេ` ", "", "", "No", ""], MANIFEST[0])
        self.assertEqual((rec["preferred"], rec["raw_cells"][2]), ("កាហ្វេ", " `កាហ្វេ` "))   # raw form kept

    def test_parse_detects_problems(self):
        man = {r["id"]: r for r in MANIFEST}
        res = e08.parse_response(FIXTURE_BATCH_01, {"e08s001", "e08s002", "e08s003"}, man)
        self.assertEqual(set(res["records"]), {"e08s001", "e08s002", "e08s003"})
        self.assertEqual(sorted(p["problem"] for p in res["problems"]), ["id_from_another_batch", "unknown_id"])
        dup = "| e08s001 | coffee | កាហ្វេ | | | no | |\n| e08s001 | coffee | ខូហ្វី | | | no | |"
        r2 = e08.parse_response(dup, {"e08s001", "e08s002"}, man)
        self.assertEqual(r2["problems"][0]["problem"], "duplicate_conflicting_rows")
        self.assertEqual(r2["missing"], ["e08s002"])
        tabs = "e08s001\tcoffee\tកាហ្វេ\t\t\tno\t"
        self.assertIn("e08s001", e08.parse_response(tabs, {"e08s001"}, man)["records"])

    def test_judge(self):
        m = MANIFEST[0]
        ok = e08.judge(["e08s001", "coffee", "កាហ្វេ", "ខូហ្វី", "កាហ្វេ", "no", ""], m)
        self.assertEqual((ok["status"], ok["alternatives"], ok["source"]), ("accepted", ["ខូហ្វី"],
                                                                            "google_ai_mode_synthetic"))
        self.assertEqual(e08.judge(["e08s001", "tea", "កាហ្វេ", "", "", "no", ""], m)["reason"], "english_mismatch")
        self.assertEqual(e08.judge(["e08s001", "coffee", "កាហ្វេ", "", "", "maybe?", ""], m)["reason"],
                         "uncertain_field_unreadable")
        self.assertEqual(e08.judge(["e08s001", "coffee", "", "ខូហ្វី", "", "no", ""], m)["reason"],
                         "preferred_empty")                         # an alternative is never promoted
        self.assertEqual(e08.judge(["e08s001", "coffee"], m)["reason"], "too_few_columns")

    def test_import_end_to_end_and_regeneration(self):
        with Env() as env:
            self.assertEqual(e08.import_responses(env.d / "responses")["state"], "awaiting_ai_mode_responses")
            (env.d / "responses" / "batch_01.md").write_text(FIXTURE_BATCH_01, encoding="utf-8")
            s = e08.import_responses(env.d / "responses")
            self.assertEqual(s["accepted"], 1)
            self.assertEqual(s["quarantined"], {"preferred_explanatory_text": 1, "marked_uncertain": 1,
                                              "id_from_another_batch": 1})
            self.assertEqual(s["state"], "awaiting_regeneration_responses")
            self.assertTrue((env.d / "imported" / "raw" / "batch_01.md").exists())     # raw kept
            regen = json.loads((env.d / "exp" / "synthetic" / "prompts" / "regen_01.ids.json").read_text())
            self.assertEqual(regen, ["e08s002", "e08s004"])        # malformed + missing; not uncertain, not accepted
            # a replacement response fixes e08s002 and tries to overwrite the accepted e08s001
            fix = ("| e08s002 | london | ឡុងដ៍ | | | no | |\n| e08s004 | bus | ប៊ុស | | | no | |\n"
                   "| e08s001 | coffee | ខុហ្វេ | | | no | |")
            (env.d / "responses" / "regen_01.md").write_text(fix, encoding="utf-8")
            s2 = e08.import_responses(env.d / "responses")
            acc = {r["id"]: r for r in e08.read_jsonl(env.d / "imported" / "accepted.jsonl")}
            self.assertEqual(acc["e08s001"]["preferred"], "កាហ្វេ")                    # accepted entry kept
            self.assertEqual(set(acc), {"e08s001", "e08s002", "e08s004"})
            self.assertEqual(s2["state"], "imported_ready_to_freeze")              # uncertain stays quarantined
            self.assertEqual(s2["quarantined"], {"marked_uncertain": 1})

    def test_empty_files_are_skipped(self):
        with Env() as env:
            ok = "| e08s001 | coffee | កាហ្វេ | | | no | |\n| e08s002 | london | ឡុងដ៍ | | | no | |\n" \
                 "| e08s003 | taxi | តាក់ស៊ី | | | no | |"                       # TEST FIXTURE
            (env.d / "responses" / "batch_01.md").write_text(ok, encoding="utf-8")
            (env.d / "responses" / "batch_02.md").write_text("", encoding="utf-8")       # not filled in yet
            s = e08.import_responses(env.d / "responses")
            self.assertEqual(s["files"]["batch_02.md"], {"skipped": "empty file (not filled in yet)"})
            self.assertEqual(s["state"], "awaiting_ai_mode_responses")
            self.assertEqual(s["missing"], 0)                       # unfilled batches are not "missing"

    def test_refuses_protected_overlap(self):
        with Env() as env:
            prot = {"words": {"coffee"}, "stems": set(), "held_targets": set(), "counts": {}}
            with mock.patch.object(e08, "protected", lambda: prot):
                with self.assertRaises(RuntimeError):
                    e08.import_responses(env.d / "responses")


if __name__ == "__main__":
    unittest.main()
