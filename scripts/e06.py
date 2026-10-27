"""E06 — reference-model study: can a stronger pretrained model add English → Khmer phonetic knowledge beyond the
small letters model, and (only if established by independent review) could a compact student inherit it?

Steps (resumable; progress in `state/e06/status.json` and `PROGRESS.md`):
    pool        fresh-word pilot pool (56 words) and the overlap audit, before any new candidate output
    register    hashes of commit, code, data, checkpoints, pool, frozen prompt, recipe and gate → registration.json
    baselines   letters-model candidates (E04 G seeds, product default E02 letters_s0) for the pool and discovery
    compare     reference agreement (historical, unreviewed), automated diagnostics, pool tables → results.json
    queue       blinded review queue + separate origin key (data/e06/review/, git-ignored, local)
    gate        teacher-quality gate (registered procedure); `teacher_advantage_unresolved` without judgments
    discovery   read-only constrained-discovery feasibility audit over local corpora (≤ 200 sampled words)

The ByT5 model steps are in `scripts/e06_byt5.py` (smoke, train, predict).

Example:
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e06.py pool
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e05  # noqa: E402
from khmer_translit import phonetic_source as ps  # noqa: E402
from khmer_translit import reference_data as rd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "E06-reference-model"
EXP = ROOT / "experiments" / EXPERIMENT
REPORTS = EXP / "reports"
PRED = EXP / "predictions"
STATE = ROOT / "state" / "e06"
E03 = ROOT / "data" / "e03"
REVIEW = ROOT / "data" / "e06" / "review"        # git-ignored: local only
JUDGMENTS = REVIEW / "judgments.tsv"             # only an independent reviewer may create this file
POOL = EXP / "pool.jsonl"
PROMPT = EXP / "prompts" / "route_a_prompt.txt"
PRIMARY_EXCLUDE = {"pronunciation_ambiguous", "possible_translation"}
FORBIDDEN = e05.FORBIDDEN
N_BOOT, BOOT_SEED = 10_000, 20261027
POOL_SEED, DISCOVERY_SEED, REVIEW_SEED = "e06-pool", "e06-discovery", "e06-review"
DISCOVERY_N = 200
PER_CATEGORY = 8
DIFFICULT_ANYWHERE = ("ough", "augh", "eigh", "eau", "sch", "rh", "ght")
DIFFICULT_START = ("kn", "wr", "ps", "pn", "gn")
UNSUITABLE: dict[str, str] = {}                  # words removed from the pool before outputs, with reasons
# Historical material (any word appearing here is not fresh). Read in full and tokenised, except the files in
# INPUT_ONLY, of which only the Latin inputs are read (their answers are never parsed or stored).
HISTORICAL = ["data/e02/train.tsv", "data/e02/dev_references.jsonl", "data/e03/candidates.jsonl", "data/e03/train.tsv",
              "data/e03/dev_references.jsonl", "data/e03/eval_references.jsonl", "data/e03/exclusions.json",
              "data/e04/phone_seen.json", "data/e05/review/review_queue.tsv", "data/mined_kmwiki.tsv",
              "data/mined_fineweb.tsv", "data/extra_train.tsv", "data/pairs.tsv", "data/roman/pairs.tsv",
              "data/llm_drafts.tsv", "data/llm_drafts_r2.tsv", "data/owner_picks.tsv", "data/names_dev.tsv",
              "data/names_test.tsv", "data/names_test2.tsv", "data/list_split.json", "data/pick/items.json",
              "data/rerank/cands.json", "data/rerank/web_counts.json", "data/e2/dev.json",
              "data/lexicon/km_lexicon.tsv", "datasets/build/latin_audit.json",
              "../pocket-tts-khmer/configs/latin_khmer.tsv"]
HISTORICAL_GLOBS = ["data/synth/*", "experiments/**/*.md", "experiments/**/*.json", "experiments/**/*.jsonl",
                    "experiments/**/*.yaml", "docs/**/*.md", "handoff/**/*.md", "README.md"]
INPUT_ONLY = {"data/benchmark/v1/dev41.tsv": "tsv_col0", "data/benchmark/v1/list_test.tsv": "tsv_col0",
              "data/benchmark/v1/names_test2.tsv": "tsv_col0", "data/benchmark/v1/pick_test.tsv": "tsv_col0",
              "data/benchmark/v1/romanised_test.tsv": "tsv_col0", "data/e2/reserve.json": "json_string_list"}
NOT_OPENED = {"data/e02/references.jsonl": "E02 final references: never opened; every input word comes from the "
                                           "excluded sources listed in data/e02/manifest.json"}
WORDLIST_LOWER = ROOT / "data" / "en_common_words.txt"     # SCOWL lower-case entries (data/SOURCES.md)
WORDLIST_FULL = Path("/usr/share/dict/american-english")    # same SCOWL release, with capitalised proper nouns


def guard(path: Path | str) -> Path:
    """Refuse forbidden paths (E02 final references/results, benchmark panels, the reserve).

    Raises:
        PermissionError: For a forbidden path.
    """
    return e05.guard(path)


def sha(path: Path) -> str:
    return hashlib.sha256(guard(path).read_bytes()).hexdigest()


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def items(name: str) -> list[rd.ReferenceItem]:
    return rd.load_reference_items(guard(E03 / f"{name}_references.jsonl"))


def primary(its):
    return [i for i in its if not PRIMARY_EXCLUDE & set(i.flags)]


def macro(its, values: dict) -> float:
    fam = defaultdict(list)
    for i in its:
        fam[i.family].append(values[i.id])
    return mean(mean(v) for v in fam.values()) if fam else 0.0


def norm_word(s: str) -> str:
    """Lower-case ASCII letters only (accents stripped).

    Examples:
        >>> norm_word("Café-Noir")
        'cafenoir'
    """
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if "a" <= c <= "z")


def latin_tokens(text: str) -> set[str]:
    """Lower-cased Latin letter runs (after accent stripping) of a text."""
    t = unicodedata.normalize("NFKD", text.lower())
    return set(re.findall(r"[a-z]+", t))


# ------------------------------------------------------------------ status (durable progress)
def status() -> dict:
    p = STATE / "status.json"
    return json.loads(p.read_text()) if p.exists() else {"steps": {}, "budget": {}}


def mark(step: str, outputs: list[Path], **extra) -> None:
    st = status()
    st["steps"][step] = {"done": True, "at": now(), "outputs": {str(p.relative_to(ROOT)): sha(p) for p in outputs},
                         **extra}
    _write_status(st)


def mark_progress(step: str, note: str) -> None:
    st = status()
    st.setdefault("progress", {})[step] = {"at": now(), "note": note}
    _write_status(st)


def _write_status(st: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "status.json").write_text(json.dumps(st, indent=1) + "\n")
    lines = [f"# E06 progress ({now()})", ""] + [f"- {k}: done {v['at']}" for k, v in st["steps"].items()]
    lines += [f"- {k} (in progress): {v['note']} ({v['at']})" for k, v in st.get("progress", {}).items()
              if k not in st["steps"]]
    lines += ["", "Resume: `CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e06.py run` (model steps: "
              "`scripts/e06_byt5.py train|predict`, which resume from their own checkpoints)"]
    (STATE / "PROGRESS.md").write_text("\n".join(lines) + "\n")


def is_done(step: str) -> bool:
    s = status()["steps"].get(step)
    return bool(s) and all((ROOT / p).exists() and sha(ROOT / p) == h for p, h in s["outputs"].items())


def registration() -> dict:
    return json.loads((EXP / "registration.json").read_text())


# ------------------------------------------------------------------ historical exclusion universe
def historical_files() -> list[Path]:
    files = [ROOT / p for p in HISTORICAL]
    for g in HISTORICAL_GLOBS:
        files += [p for p in ROOT.glob(g) if p.is_file() and EXPERIMENT not in str(p)]
    return sorted({p.resolve() for p in files if p.exists() and not forbidden(p)})


def forbidden(p: Path) -> bool:
    return any(f in str(Path(p).resolve()) for f in FORBIDDEN)


def skipped_forbidden() -> list[str]:
    """Glob matches that the guard forbids (never opened; recorded in the ledger)."""
    out = []
    for g in HISTORICAL_GLOBS:
        out += [str(p.relative_to(ROOT)) for p in ROOT.glob(g) if p.is_file() and forbidden(p)]
    return sorted(set(out))


def input_only_words(rel: str) -> set[str]:
    """Latin inputs of a benchmark panel or the reserve, without reading their answers into memory as data."""
    p = ROOT / rel
    mode = INPUT_ONLY[rel]
    if mode == "tsv_col0":
        with open(p, encoding="utf-8") as f:                 # deliberate guard bypass: first column only
            return {norm_word(line.split("\t", 1)[0]) for line in f if line.strip()}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not (isinstance(data, list) and all(isinstance(x, str) for x in data)):
        raise ValueError(f"{rel}: expected a list of Latin inputs")
    return {norm_word(x) for x in data}


def historical_universe() -> tuple[set[str], dict]:
    """All historical Latin tokens and a per-source count ledger."""
    words: set[str] = set()
    ledger = {}
    for p in historical_files():
        toks = latin_tokens(guard(p).read_text(encoding="utf-8", errors="replace"))
        words |= toks
        ledger[str(p.relative_to(ROOT.parent)) if not str(p).startswith(str(ROOT)) else str(p.relative_to(ROOT))] = \
            {"tokens": len(toks), "sha256": sha(p), "access": "full text, tokenised"}
    for rel in INPUT_ONLY:
        toks = input_only_words(rel)
        words |= toks
        ledger[rel] = {"tokens": len(toks), "access": "Latin inputs only"}
    for rel, why in NOT_OPENED.items():
        ledger[rel] = {"access": "not opened", "why": why}
    for rel in skipped_forbidden():
        ledger[rel] = {"access": "not opened", "why": "forbidden by the evaluation guard (E02 final material)"}
    return {w for w in words if w}, ledger


def stem_set(words) -> set[str]:
    out: set[str] = set()
    for w in words:
        out |= rd.stems(w)
    return out


def is_fresh(word: str, hist_stems: set[str]) -> bool:
    """A word is fresh when none of its stems (lexical family keys) occurs among historical stems.

    Examples:
        >>> is_fresh("sharing", stem_set(["share"])), is_fresh("table", stem_set(["share"]))
        (False, True)
    """
    return not (rd.stems(word) & hist_stems)


# ------------------------------------------------------------------ pool (pure helpers + build)
def strip_stress(r: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(re.sub(r"\d", "", p) for p in r)


def is_ambiguous(readings) -> bool:
    """Two or more CMUdict readings that differ after removing stress."""
    return len({strip_stress(r) for r in readings}) > 1


def is_difficult(word: str, reading: tuple[str, ...]) -> bool:
    """Registered spelling-to-sound difficulty rule.

    Examples:
        >>> is_difficult("knight", ("N", "AY1", "T")), is_difficult("table", ("T", "EY1", "B", "AH0", "L"))
        (True, False)
    """
    return (len(word) - len(reading) >= 3 or any(x in word for x in DIFFICULT_ANYWHERE)
            or word.startswith(DIFFICULT_START) or word.endswith("mb"))


def order_key(seed: str, word: str) -> str:
    return hashlib.sha256(f"{seed}:{word}".encode()).hexdigest()


def candidate_words() -> tuple[set[str], set[str]]:
    """(lower-case dictionary words, proper names) from SCOWL; names are capitalised entries not also lower-case."""
    lower = {w.strip() for w in WORDLIST_LOWER.read_text(encoding="utf-8").splitlines() if re.fullmatch(r"[a-z]+", w.strip())}
    names = set()
    if WORDLIST_FULL.exists():
        for w in WORDLIST_FULL.read_text(encoding="utf-8", errors="replace").splitlines():
            if re.fullmatch(r"[A-Z][a-z]+", w) and w.lower() not in lower:
                names.add(w.lower())
    return lower, names


def categorise(word: str, kind: str, cmu: dict) -> str | None:
    """The single pool category of a fresh candidate (priority order), or None if it fits none."""
    rs = cmu.get(word)
    n = len(word)
    if rs and len(rs) > 1:
        return "ambiguous" if is_ambiguous(rs) and 4 <= n <= 9 else None
    if not rs:
        return "dictionary_miss" if kind == "lower" and 5 <= n <= 10 else None
    r = rs[0]
    if kind == "name":
        return "name" if 4 <= n <= 9 else None
    if is_difficult(word, r) and 5 <= n <= 10:
        return "difficult"
    if 3 <= n <= 4:
        return "short"
    if n >= 11:
        return "long"
    if 5 <= n <= 10:
        return "ordinary"
    return None


CATEGORIES = ("ordinary", "name", "difficult", "short", "long", "dictionary_miss", "ambiguous")


def select_pool(cands: list[tuple[str, str]], cmu: dict, hist_stems: set[str], per: int = PER_CATEGORY,
                seed: str = POOL_SEED) -> tuple[list[dict], dict]:
    """Deterministic selection: candidates in sha256(seed:word) order, first `per` per category, one word per family.

    Args:
        cands: `(word, kind)` with kind "lower" or "name".

    Returns:
        `(pool rows, exclusion counts)`.
    """
    take = {c: [] for c in CATEGORIES}
    used: set[str] = set()
    excl = Counter()
    for w, kind in sorted(cands, key=lambda x: order_key(seed, x[0])):
        if all(len(v) >= per for v in take.values()):
            break
        if w in UNSUITABLE:
            excl["unsuitable"] += 1
            continue
        if not is_fresh(w, hist_stems):
            excl["historical_family"] += 1
            continue
        c = categorise(w, kind, cmu)
        if c is None or len(take[c]) >= per:
            excl["no_open_category"] += 1
            continue
        if rd.stems(w) & used:
            excl["family_already_in_pool"] += 1
            continue
        used |= rd.stems(w)
        rs = cmu.get(w) or ()
        take[c].append({"word": w, "category": c, "kind": kind, "track": "ambiguous" if c == "ambiguous" else "primary",
                        "default_phones": " ".join(rs[0]) if rs else None,
                        "all_readings": [" ".join(r) for r in rs], "in_cmudict": bool(rs),
                        "flags": (["pronunciation_ambiguous"] if c == "ambiguous" else [])
                        + ([] if rs else ["dictionary_miss"])})
    rows = []
    for c in CATEGORIES:
        for n, r in enumerate(take[c]):
            rows.append({"id": f"p{len(rows) + 1:02d}", **r})
    return rows, dict(excl)


def pool() -> dict:
    """Build the pilot pool and the overlap audit (no model outputs involved)."""
    cmu, cmu_sha = ps.cmudict()
    hist, ledger = historical_universe()
    hist_stems = stem_set(hist)
    lower, names = candidate_words()
    cands = [(w, "lower") for w in lower] + [(w, "name") for w in names]
    rows, excl = select_pool(cands, cmu, hist_stems)
    for r in rows:                                   # self-check
        assert is_fresh(r["word"], hist_stems) and r["word"] not in hist
    EXP.mkdir(parents=True, exist_ok=True)
    with open(POOL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    audit = {"rule": "fresh = no stem of the word (reference_data.stems, the lexical-family key) occurs among the "
                     "stems of any Latin token in the historical material; SCOWL candidates in sha256('e06-pool:'+word) "
                     "order; 8 per category; one word per family",
             "historical_tokens": len(hist), "historical_stems": len(hist_stems), "sources": ledger,
             "candidates": {"lower": len(lower), "names": len(names)},
             "fresh_candidates": sum(is_fresh(w, hist_stems) for w, _ in cands), "exclusions_during_walk": excl,
             "pool_size": len(rows), "per_category": dict(Counter(r["category"] for r in rows)),
             "pool_sha256": sha(POOL), "cmudict_sha256": cmu_sha, "unsuitable_removed": UNSUITABLE,
             "wordlists": {"lower": sha(WORDLIST_LOWER), "full": sha(WORDLIST_FULL) if WORDLIST_FULL.exists() else None},
             "note": "inspected examples in reports, docs and handoff files are included as historical material"}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "overlap_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1) + "\n")
    mark("pool", [POOL, REPORTS / "overlap_audit.json"])
    return {k: v for k, v in audit.items() if k != "sources"}


def load_pool() -> list[dict]:
    return [json.loads(x) for x in POOL.read_text(encoding="utf-8").splitlines()]


def discovery_words(cmu=None) -> list[str]:
    """200 fresh words (not in the pool), sha256('e06-discovery:'+word) order, 4–12 letters."""
    hist, _ = historical_universe()
    hist_stems = stem_set(hist)
    pool_stems = stem_set(r["word"] for r in load_pool())
    lower, names = candidate_words()
    out = []
    for w in sorted(lower | names, key=lambda x: order_key(DISCOVERY_SEED, x)):
        if 4 <= len(w) <= 12 and is_fresh(w, hist_stems) and not (rd.stems(w) & pool_stems):
            out.append(w)
            pool_stems |= rd.stems(w)
        if len(out) == DISCOVERY_N:
            break
    return out


def prediction_sets() -> dict[str, list[tuple[str, str]]]:
    """`{set: [(id, word)]}` for the reference-model predictions."""
    sets = {"dev": [(i.id, i.word) for i in items("dev")], "eval": [(i.id, i.word) for i in items("eval")],
            "pool": [(r["id"], r["word"]) for r in load_pool()]}
    dw = DISCOVERY / "words.json"
    if dw.exists():
        sets["discovery"] = [(f"d{n:03d}", w) for n, w in enumerate(json.loads(dw.read_text()), 1)]
    return sets


DISCOVERY = REPORTS / "discovery"


# ------------------------------------------------------------------ frozen prompt (Route A; recorded, not run)
ROUTE_A_PROMPT = """Task: write the English word below in Khmer script so that a Khmer reader would pronounce it like the
English pronunciation given. This is phonetic transliteration, not translation: never give a Khmer word that means
the same thing, unless it is also pronounced like the English word.
English word: {word}
Intended English pronunciation (CMUdict ARPAbet, US): {phones}
Answer with JSON only, no other text: {{"khmer": ["<up to 4 spellings, best first>"], "status": "ok" | "unsupported" | "uncertain"}}
"""


def write_prompt() -> Path:
    PROMPT.parent.mkdir(parents=True, exist_ok=True)
    PROMPT.write_text(ROUTE_A_PROMPT, encoding="utf-8")
    return PROMPT


def render_prompt(word: str, phones: str | None) -> str:
    """The frozen prompt for one word (pronunciation 'not available' for dictionary misses).

    Raises:
        RuntimeError: If the saved prompt file differs from the frozen text.
    """
    if PROMPT.exists() and PROMPT.read_text(encoding="utf-8") != ROUTE_A_PROMPT:
        raise RuntimeError("route A prompt changed after freezing")
    return ROUTE_A_PROMPT.format(word=word, phones=phones or "not available")


# ------------------------------------------------------------------ registration
def register() -> dict:
    """Hashes and rules, committed before full adaptation and before any reference-model generation."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import e06_byt5 as B
    write_prompt()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    feas = json.loads((REPORTS / "feasibility.json").read_text())
    out = {"experiment": EXPERIMENT, "registered_at": now(), "commit_at_registration": commit,
           "label": "operational comparison on historical (unreviewed) and fresh (reference-free) material; "
                    "teacher quality requires independent judgments",
           "byt5": {"model": "google/byt5-small", "revision": B.REVISION, "base_sha256": B.BASE_SHA256,
                    "license": "apache-2.0", "recipe": B.RECIPE, "interface": "letters only (no pronunciation input)",
                    "feasibility": {k: feas[k] for k in ("params", "update_s_median_after_first",
                                                         "estimated_train_hours_1000_updates", "peak_rss_mb")}},
           "route_a": {"status": "unavailable", "prompt_sha256": sha(PROMPT)},
           "baselines": {"E04 G_s0..s2": {m: sha(ROOT / "models/e04" / m / "model.npz") for m in ("G_s0", "G_s1", "G_s2")},
                         "product default E02 letters_s0": sha(ROOT / "models/e02/letters_s0/model.npz")},
           "data": {p: sha(ROOT / p) for p in ("data/e03/train.tsv", "data/e03/dev_references.jsonl",
                                                "data/e03/eval_references.jsonl", "data/src/cmudict/cmudict.dict")},
           "pool": {"path": str(POOL.relative_to(ROOT)), "sha256": sha(POOL)},
           "gate": GATE, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
           "code": {p: sha(ROOT / p) for p in ("scripts/e06.py", "scripts/e06_byt5.py")}}
    (EXP / "registration.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


# ------------------------------------------------------------------ baselines (letters models)
def g_predict(model: str, words: list[str], width: int = 4) -> list[list[str]]:
    import e04
    npm, meta = e04.load_np(ROOT / "models" / "e04" / model / "model.npz")
    voc, _ = e04.vocab()
    return [[t for t, _ in r] for r in e04.beam(npm, [e04.encode(w, (), "g", voc) for w in words], width)]


def default_predict(words: list[str]) -> list[dict]:
    from khmer_translit.english import Transliterator
    tl = Transliterator.from_pretrained(ROOT / "models/e02/letters_s0/model.npz")
    return [{"status": r.status, "candidates": [c.text for c in r.candidates][:4]}
            for r in tl.predict_many(words, top_k=4)]


def baselines() -> dict:
    """Letters-model top-4 (beam 4) on the pool and the discovery sample; E04's saved eval outputs are reused."""
    PRED.mkdir(parents=True, exist_ok=True)
    sets = {k: v for k, v in prediction_sets().items() if k in ("pool", "discovery")}
    outs = []
    for name, rows in sets.items():
        words = [w for _, w in rows]
        systems = {m: g_predict(m, words) for m in ("G_s0", "G_s1", "G_s2")}
        if name == "pool":
            systems["default_e02"] = [r["candidates"] if r["status"] == "ok" else [] for r in default_predict(words)]
        for s, preds in systems.items():
            p = PRED / f"{s}_{name}.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                for (i, w), c in zip(rows, preds):
                    f.write(json.dumps({"id": i, "word": w, "system": s, "width": 4, "candidates": e05.dedup(c)},
                                       ensure_ascii=False) + "\n")
            outs.append(p)
    mark("baselines", outs)
    return {"files": [p.name for p in outs]}


def load_pred(system: str, name: str) -> dict[str, list[str]]:
    p = PRED / f"{system}_{name}.jsonl"
    return {r["id"]: e05.dedup(r["candidates"]) for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}


def e04_eval(model: str) -> dict[str, list[str]]:
    """E04's saved beam-4 eval outputs (same decoder settings as here)."""
    p = ROOT / "experiments/E04-pronunciation-pathways/results/eval_per_item.jsonl"
    return {r["id"]: e05.dedup(r["top4"]) for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())
            if r["model"] == model}


# ------------------------------------------------------------------ automated diagnostics (level 3 proxies)
CONS = (0x1780, 0x17A2)
INDEP = (0x17A3, 0x17B3)
DEP_V = (0x17B6, 0x17C5)
SIGNS = (0x17C6, 0x17D1)
COENG = 0x17D2


def orthography(text: str | None) -> str:
    """Rule-based well-formedness of a Khmer word: `ok` or the first violated rule.

    Rules: Khmer letters/signs only; starts with a consonant or independent vowel; coeng is followed by a consonant;
    no two dependent vowels in a row; a dependent vowel or sign follows a base. This is a screening proxy: a
    well-formed string can still be a wrong or unreadable spelling.

    Examples:
        >>> orthography("កាហ្វេ"), orthography("ាក"), orthography("ក្"), orthography(""), orthography("ab")
        ('ok', 'starts_with_dependent', 'dangling_coeng', 'empty', 'non_khmer')
    """
    if text is None:
        return "malformed_utf8"
    if not text:
        return "empty"
    if not ps.is_khmer_word(text):
        return "non_khmer"
    cps = [ord(c) for c in text]
    within = lambda c, r: r[0] <= c <= r[1]  # noqa: E731
    if not (within(cps[0], CONS) or within(cps[0], INDEP)):
        return "starts_with_dependent"
    for a, b in zip(cps, cps[1:] + [None]):
        if a == COENG and (b is None or not within(b, CONS)):
            return "dangling_coeng"
        if within(a, DEP_V) and b is not None and within(b, DEP_V):
            return "double_dependent_vowel"
    return "ok"


def consonant_screen(word: str, text: str, phones: str | None, km_lex, lat_lex) -> str:
    """The frozen E02-diagnostics consonant screen (automated proxy): level 0/1/2/3+/na, or `no_phones`."""
    import e02_diagnostics as DG
    if not phones or not text:
        return "no_phones" if not phones else "na"
    ph = tuple(phones.split())
    return DG.compat_row(word, text, ph, (ph,), km_lex, lat_lex)["level"]


# ------------------------------------------------------------------ comparison
def topk_cov(its, preds: dict[str, list[str]], k: int) -> dict[str, int]:
    return {i.id: int(any(c in {nfc(a) for a in i.accepted} for c in preds.get(i.id, [])[:k])) for i in its}


def compare() -> dict:
    """Reference agreement on the historical E03 sets (level 2) and level-3 diagnostics on every set."""
    import e02_diagnostics as DG
    import numpy as np
    km_lex, lat_lex = DG.load_lexicon()
    cmu, _ = ps.cmudict()
    res = {"evidence_levels": {"1_independent_review": "none available (no judgments collected)",
                               "2_attested_unreviewed": "E03 dev/eval mined, consonant-screened references (historical)",
                               "3_automated_proxy": "orthography rules, frozen consonant screen, lexicon membership, "
                                                    "agreement between systems"}}
    # level 2: historical eval (primary) and dev (selection set, descriptive)
    ev = primary(items("eval"))
    by = {"byt5_lora": load_pred("byt5", "eval"), **{m: e04_eval(m) for m in ("G_s0", "G_s1", "G_s2")}}
    tab = {}
    for s, p in by.items():
        e05.check_ids({i.id for i in items("eval")}, {k: v for k, v in p.items()})
        tab[s] = {"top1_macro": round(macro(ev, topk_cov(ev, p, 1)), 4), "top4_macro": round(macro(ev, topk_cov(ev, p, 4)), 4),
                  "empty_or_failed": sum(not p.get(i.id) for i in ev)}
    idx = e05.boot_indices(len({i.family for i in ev}), N_BOOT, BOOT_SEED)
    fm = {f"{s}@{k}": e05.family_means(ev, topk_cov(ev, p, k)) for s, p in by.items() for k in (1, 4)}
    gs = ["G_s0", "G_s1", "G_s2"]
    res["historical_eval"] = {
        "label": "level 2 (attested, unreviewed); historical set; ByT5 one adaptation run, G three seeds",
        "items": len(ev), "families": len({i.family for i in ev}), "systems": tab,
        "byt5_minus_G_mean_top1": e05.contrast(fm, ["byt5_lora@1"], [f"{g}@1" for g in gs], idx),
        "byt5_minus_G_mean_top4": e05.contrast(fm, ["byt5_lora@4"], [f"{g}@4" for g in gs], idx),
        "per_seed_top1": {g: e05.contrast(fm, ["byt5_lora@1"], [f"{g}@1"], idx) for g in gs}}
    dv = primary(items("dev"))
    bd = load_pred("byt5", "dev")
    g0d = {}
    for line in (ROOT / "experiments/E04-pronunciation-pathways/reports/pilot_dev_per_item.jsonl").read_text().splitlines():
        r = json.loads(line)
        g0d[r["id"]] = e05.dedup(r["G"]["top4"])
    res["dev_selection_set"] = {"label": "descriptive; ByT5 checkpoint selected on these items",
                                "byt5_top1": round(macro(dv, topk_cov(dv, bd, 1)), 4),
                                "byt5_top4": round(macro(dv, topk_cov(dv, bd, 4)), 4),
                                "G_s0_top1": round(macro(dv, topk_cov(dv, g0d, 1)), 4),
                                "G_s0_top4": round(macro(dv, topk_cov(dv, g0d, 4)), 4)}
    # level 3 diagnostics on eval (all items) and the pool
    def diag(rows_words: dict[str, str], preds: dict[str, list[str]], raw_status: dict | None, phones_of):
        orth, screen, lex, n_empty = Counter(), Counter(), 0, 0
        for i, w in rows_words.items():
            c = preds.get(i, [])
            top = c[0] if c else None
            if not c:
                n_empty += 1
            orth[orthography(top) if c else (raw_status or {}).get(i, "empty")] += 1
            if top:
                screen[consonant_screen(w, top, phones_of(w), km_lex, lat_lex)] += 1
                lex += top in km_lex
        return {"orthography_top1": dict(orth), "consonant_screen_top1": dict(screen),
                "top1_is_lexicon_word": lex, "no_candidate": n_empty, "words": len(rows_words)}
    default_ph = lambda w: " ".join(cmu[w][0]) if cmu.get(w) else None  # noqa: E731
    evw = {i.id: i.word for i in items("eval")}
    raw_e = _raw_status("eval")
    res["diagnostics_eval"] = {s: diag(evw, p, raw_e if s == "byt5_lora" else None, default_ph) for s, p in by.items()}
    pool_rows = load_pool()
    pw = {r["id"]: r["word"] for r in pool_rows}
    ps_ = {"byt5_lora": load_pred("byt5", "pool"), **{m: load_pred(m, "pool") for m in ("G_s0", "G_s1", "G_s2", "default_e02")}}
    raw_p = _raw_status("pool")
    res["pool"] = {"label": "fresh words; no qualified or attested references exist; level 3 only",
                   "words": len(pool_rows), "per_category": dict(Counter(r["category"] for r in pool_rows)),
                   "diagnostics": {s: diag(pw, p, raw_p if s == "byt5_lora" else None, default_ph) for s, p in ps_.items()},
                   "by_category": {}}
    for c in CATEGORIES:
        ids = [r["id"] for r in pool_rows if r["category"] == c]
        res["pool"]["by_category"][c] = {
            s: {"orthography_ok_top1": sum(orthography((p.get(i) or [None])[0]) == "ok" for i in ids),
                "screen_level0_top1": sum(bool(p.get(i)) and consonant_screen(pw[i], p[i][0], default_ph(pw[i]),
                                                                              km_lex, lat_lex) == "0" for i in ids)}
            for s, p in ps_.items()} | {"words": len(ids)}
    agree = {}
    for a in ps_:
        for b in ps_:
            if a < b:
                agree[f"{a}|{b}"] = sum(bool(ps_[a].get(i)) and bool(ps_[b].get(i)) and ps_[a][i][0] == ps_[b][i][0]
                                        for i in pw)
    res["pool"]["top1_agreement_between_systems"] = agree
    res["pool"]["examples"] = [{"word": r["word"], "category": r["category"], "phones": r["default_phones"],
                                **{s: (p.get(r["id"]) or [None])[0] for s, p in ps_.items()}} for r in pool_rows]
    _ = np
    (EXP / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    mark("compare", [EXP / "results.json"])
    return {k: v for k, v in res.items() if k in ("historical_eval", "dev_selection_set")}


def _raw_status(name: str) -> dict[str, str]:
    """ByT5 failure status per id: refused / malformed / empty when no usable candidate exists."""
    out = {}
    for r in map(json.loads, (PRED / f"byt5_{name}.jsonl").read_text(encoding="utf-8").splitlines()):
        if not r["candidates"]:
            out[r["id"]] = r["status"] if r["status"] != "ok" else (r["raw"][0]["status"] if r["raw"] else "empty")
    return out


# ------------------------------------------------------------------ blinded review queue
QUEUE_SYSTEMS = ("byt5_lora", "G_s0", "default_e02")


def build_queue(pool_rows: list[dict], preds: dict[str, dict[str, list[str]]], seed: str = REVIEW_SEED):
    """Deduplicated candidates per word in random order; returns `(rows, origin key)`. Origin never in rows."""
    rows, key = [], {"note": "origin key: keep separate from the queue; never show to reviewers", "items": {}}
    order = sorted(pool_rows, key=lambda r: order_key(seed, r["word"]))
    for n, r in enumerate(order, 1):
        origins: dict[str, list[str]] = {}
        for s in QUEUE_SYSTEMS:
            for rank, c in enumerate(preds[s].get(r["id"], [])[:4], 1):
                origins.setdefault(nfc(c), []).append(f"{s}@{rank}")
        texts = list(origins)
        random.Random(f"{seed}:{r['word']}").shuffle(texts)
        ik = f"w{n:03d}"
        key["items"][ik] = {"pool_id": r["id"], "word": r["word"], "category": r["category"], "candidates": {}}
        pron = r["default_phones"] or "not in CMUdict: use the usual US English pronunciation"
        if r["track"] == "ambiguous":
            pron = r["default_phones"] + " (default; other readings: " + "; ".join(r["all_readings"][1:]) + ")"
        for j, t in enumerate(texts, 1):
            cid = f"{ik}-c{j:02d}"
            key["items"][ik]["candidates"][cid] = {"text": t, "origins": origins[t]}
            rows.append({"item": ik, "english": r["word"], "pronunciation": pron, "candidate_id": cid,
                         "candidate": t, "judgment": "", "comment": ""})
    return rows, key


def queue() -> dict:
    pool_rows = load_pool()
    preds = {"byt5_lora": load_pred("byt5", "pool"), "G_s0": load_pred("G_s0", "pool"),
             "default_e02": load_pred("default_e02", "pool")}
    rows, key = build_queue(pool_rows, preds)
    REVIEW.mkdir(parents=True, exist_ok=True)
    cols = ["item", "english", "pronunciation", "candidate_id", "candidate", "judgment", "comment"]
    with open(REVIEW / "review_queue.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(r[c] for c in cols) + "\n")
    (REVIEW / "review_queue.json").write_text(json.dumps(
        {"policy": "Phonetic: the English word, pronounced as English (the given US reading), written so a Khmer "
                   "reader says something close to it. Not a translation. Mark each spelling acceptable, unacceptable "
                   "or uncertain; several may be acceptable; comments optional. Order is random; origin hidden.",
         "judgment_values": ["acceptable", "unacceptable", "uncertain"], "rows": rows}, ensure_ascii=False, indent=1)
        + "\n", encoding="utf-8")
    (REVIEW / "origin_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    man = {"words": len(pool_rows), "candidates": len(rows), "systems": list(QUEUE_SYSTEMS),
           "files": {p.name: sha(p) for p in (REVIEW / "review_queue.tsv", REVIEW / "review_queue.json",
                                              REVIEW / "origin_key.json")},
           "location": "data/e06/review/ (git-ignored; local only; not published; nobody contacted)",
           "judgments": "none collected"}
    (REPORTS / "review_queue_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    mark("queue", [REPORTS / "review_queue_manifest.json", REVIEW / "review_queue.tsv", REVIEW / "origin_key.json"])
    return man


# ------------------------------------------------------------------ teacher-quality gate (registered before judgments)
GATE = {"needs": "independent judgments (data/e06/review/judgments.tsv by a reviewer who is not a model or the "
                  "agent) on the blinded queue",
        "unit": "primary-track pool word; a system's top-1 is accepted if judged acceptable ('uncertain' = not "
                "accepted; sensitivity: uncertain = accepted)",
        "comparison": "paired: reference (byt5_lora) top-1 accepted − baseline (default_e02) top-1 accepted, per word",
        "interval": "family bootstrap (pool words are one family each), 10,000 resamples, seed 20261027, 95% percentile",
        "pass_if": ["at least 40 primary words judged", "reference accepted count > baseline accepted count",
                    "interval lower bound > 0",
                    "net advantage not confined: positive difference in at least 3 categories and no category "
                    "supplies more than half of the net wins",
                    "malformed, translations (judged comments) and unsupported cases reported",
                    "teacher-output use compatible (ByT5: Apache-2.0)"],
        "otherwise": "teacher_advantage_unresolved (insufficient sample or interval) or teacher_not_better"}


def read_judgments(path: Path) -> dict[str, str]:
    """`{candidate_id: judgment}` from a reviewer TSV with the queue's columns.

    Raises:
        ValueError: For unknown judgment values.
    """
    out = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    for line in lines[1:]:
        r = dict(zip(head, line.split("\t")))
        j = r.get("judgment", "").strip().lower()
        if j:
            if j not in ("acceptable", "unacceptable", "uncertain"):
                raise ValueError(f"unknown judgment {j!r}")
            out[r["candidate_id"]] = j
    return out


def gate_decision(per_word: list[dict], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Apply the registered gate to per-word rows `{"category", "ref": 0/1, "base": 0/1}` (primary track)."""
    import numpy as np
    n = len(per_word)
    if n < 40:
        return {"status": "teacher_advantage_unresolved", "reason": f"only {n} primary words judged (< 40)"}
    d = np.array([r["ref"] - r["base"] for r in per_word], dtype=float)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    b = np.sort(d[idx].mean(1)) * 100
    lo, hi = float(b[int(0.025 * n_boot)]), float(b[int(0.975 * n_boot) - 1])
    ref, base = sum(r["ref"] for r in per_word), sum(r["base"] for r in per_word)
    by_cat = defaultdict(int)
    for r in per_word:
        by_cat[r["category"]] += r["ref"] - r["base"]
    net = ref - base
    spread = sum(v > 0 for v in by_cat.values()) >= 3 and (net <= 0 or max(by_cat.values()) <= net / 2)
    out = {"words": n, "ref_accepted": ref, "base_accepted": base, "diff_pp": round(float(d.mean() * 100), 2),
           "ci95_pp": [round(lo, 2), round(hi, 2)], "by_category_net": dict(by_cat), "spread_ok": spread}
    if ref > base and lo > 0 and spread:
        out["status"] = "teacher_passes"
    elif ref <= base and hi < 0:
        out["status"] = "teacher_not_better"
    else:
        out["status"] = "teacher_advantage_unresolved"
    return out


def gate() -> dict:
    """Gate result; `teacher_advantage_unresolved` when no independent judgments exist."""
    if not JUDGMENTS.exists():
        out = {"status": "teacher_advantage_unresolved",
               "reason": "no independent judgments exist (queue prepared, not collected); unreviewed outputs and "
                         "automated proxies cannot unlock distillation", "procedure": GATE}
    else:
        j = read_judgments(JUDGMENTS)
        key = json.loads((REVIEW / "origin_key.json").read_text(encoding="utf-8"))
        per_word = []
        for ik, it in key["items"].items():
            if it["category"] == "ambiguous":
                continue
            top = {s: next((cid for cid, c in it["candidates"].items() if f"{s}@1" in c["origins"]), None)
                   for s in ("byt5_lora", "default_e02")}
            if all(t is not None and t in j for t in top.values()):
                per_word.append({"category": it["category"], "ref": int(j[top["byt5_lora"]] == "acceptable"),
                                 "base": int(j[top["default_e02"]] == "acceptable")})
        out = {**gate_decision(per_word), "procedure": GATE}
    (REPORTS / "gate.json").write_text(json.dumps(out, indent=1) + "\n")
    mark("gate", [REPORTS / "gate.json"])
    return out


# ------------------------------------------------------------------ constrained-discovery feasibility (read-only)
NEAR_CHARS = 60


def scan_doc(text: str, targets: dict[str, list[str]]) -> dict[str, dict]:
    """For each English target word found as a Latin token in `text`: which of its candidate Khmer strings occur in
    the same document, and whether one occurs within NEAR_CHARS characters of the English word."""
    out = {}
    toks = latin_tokens(text) & targets.keys()
    if not toks:
        return out
    low = unicodedata.normalize("NFKD", text.lower())
    for w in toks:
        pos = [m.start() for m in re.finditer(rf"(?<![a-z]){w}(?![a-z])", low)]
        found, near, snippet = [], False, None
        for c in targets[w]:
            for m in re.finditer(re.escape(c), text):
                found.append(c)
                if any(abs(m.start() - p) <= NEAR_CHARS for p in pos):
                    near = True
                    s0 = max(0, min(pos, key=lambda p: abs(p - m.start())) - 40)
                    snippet = snippet or text[s0:s0 + 140].replace("\n", " ")
                break
        out[w] = {"latin_hits": len(pos), "candidates_in_doc": sorted(set(found)), "near": near, "snippet": snippet}
    return out


def discovery_words_step() -> dict:
    """Freeze the 200 discovery words (before any candidate is generated for them)."""
    DISCOVERY.mkdir(parents=True, exist_ok=True)
    words = discovery_words()
    (DISCOVERY / "words.json").write_text(json.dumps(words, indent=1) + "\n")
    mark("discovery_words", [DISCOVERY / "words.json"])
    return {"words": len(words)}


def discovery() -> dict:
    """Read-only scan of the local kmwiki dump and FineWeb-2 shard for the discovery words and their candidates."""
    import pyarrow.parquet as pq
    words = json.loads((DISCOVERY / "words.json").read_text())
    rows = prediction_sets()["discovery"]
    cand = {w: [] for w in words}
    for s in ("byt5", "G_s0"):
        for i, c in load_pred(s, "discovery").items():
            w = dict(rows)[i]
            cand[w] += [x for x in c if len(x) >= 3]          # one- and two-character strings match everywhere
    cand = {w: e05.dedup(c) for w, c in cand.items()}
    agg = {w: {"docs": 0, "docs_with_candidate": 0, "near_docs": 0, "candidates_seen": Counter(), "snippets": []}
           for w in words}
    t0 = time.time()

    def absorb(res, source):
        for w, r in res.items():
            a = agg[w]
            a["docs"] += 1
            if r["candidates_in_doc"]:
                a["docs_with_candidate"] += 1
                a["candidates_seen"].update(r["candidates_in_doc"])
            if r["near"]:
                a["near_docs"] += 1
                if len(a["snippets"]) < 2:
                    a["snippets"].append({"source": source, "text": r["snippet"]})
    n_docs = Counter()
    wiki = ROOT / "data/src/kmwiki/kmwiki-latest-pages-articles.xml.bz2"
    with bz2.open(guard(wiki), "rt", encoding="utf-8", errors="replace") as f:
        buf = []
        for line in f:
            if "<text" in line:
                buf = [line]
            elif buf:
                buf.append(line)
            if buf and "</text>" in line:
                n_docs["kmwiki"] += 1
                absorb(scan_doc("".join(buf), cand), "kmwiki")
                buf = []
    shard = next((ROOT / "data/src/fineweb2_km").rglob("*.parquet"))
    pf = pq.ParquetFile(guard(shard))
    for rg in range(pf.num_row_groups):
        for text in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
            n_docs["fineweb2"] += 1
            absorb(scan_doc(text, cand), "fineweb2")
    table = []
    for w in words:
        a = agg[w]
        cls = ("contextual_near" if a["near_docs"] else "co_document" if a["docs_with_candidate"]
               else "english_only" if a["docs"] else "absent")
        table.append({"word": w, "class": cls, "docs_with_word": a["docs"], "docs_with_candidate": a["docs_with_candidate"],
                      "near_docs": a["near_docs"], "candidates_seen": dict(a["candidates_seen"].most_common(3)),
                      "snippets": a["snippets"], "n_candidates_searched": len(cand[w])})
    out = {"label": "read-only feasibility audit; existing-target discovery (candidate strings must already occur in "
                    "local Khmer text near the English word); NOT gold labels; no target invented or kept as training data",
           "words": len(words), "docs_scanned": dict(n_docs), "seconds": round(time.time() - t0),
           "classes": dict(Counter(r["class"] for r in table)), "near_chars": NEAR_CHARS,
           "candidate_sources": "ByT5-LoRA top-4 and E04 G_s0 top-4 (strings of ≥ 3 characters)", "table": table}
    (DISCOVERY / "audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    mark("discovery", [DISCOVERY / "audit.json"])
    return {k: v for k, v in out.items() if k != "table"}


# ------------------------------------------------------------------ command line
STEPS = {"pool": pool, "register": register, "baselines": baselines, "compare": compare, "queue": queue,
         "gate": gate, "discovery_words": discovery_words_step, "discovery": discovery}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=[*STEPS, "run"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    todo = ["baselines", "compare", "queue", "gate", "discovery"] if a.step == "run" else [a.step]
    for s in todo:
        if a.step == "run" and is_done(s) and not a.force:
            print(f"{s}: done, skipped")
            continue
        print(s, json.dumps(STEPS[s](), ensure_ascii=False, default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
