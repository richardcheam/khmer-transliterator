#!/usr/bin/env python3
"""One unified Latin → Khmer transliterator, with and without a Khmer-pronunciation side task (D173).

Experiment E01 (`experiments/E01-unified/`: PLAN.md, config.yaml, RESULTS.md). Run from the repository root:

    .venv/bin/python scripts/translit_unified.py data                    # -> models/unified/ + manifest
    .venv/bin/python scripts/translit_unified.py train --arm base --seed 0   # config: experiments/E01-unified/config.yaml
    .venv/bin/python scripts/translit_unified.py train --arm aux --seed 0
    .venv/bin/python scripts/translit_unified.py export --arm aux --seed 0
    .venv/bin/python scripts/translit_unified.py score                   # -> experiments/E01-unified/results.json
    # smoke run (never into a registered folder):
    .venv/bin/python scripts/translit_unified.py train --arm base --seed 0 --set train.epochs=1 --out /tmp/smoke

Each run folder gets `config.json`, `provenance.json`, `metrics.jsonl`, `summary.json` and a W&B mirror
(`khmer_translit.experiment`); scoring uses the frozen benchmark (`khmer_translit.benchmark`, v1).

**Question:** does an auxiliary task, predicting Google's Khmer-adapted pronunciation of Latin words, improve one
Latin → Khmer model that covers romanised Khmer, foreign names and English loans?

**Spelling data (identical in both arms):**
- the rule-romanised Google-lexicon pairs (`data/roman/pairs.tsv`, train split);
- the web-mined name pairs (`extra_train.tsv`);
- the frozen loanword-list train split (live spellings);
- the 150 owner example picks;
- the 2,399 existing LLM drafts.

The last three are the "loans" and are repeated `LOAN_REPEAT` times, as in arm E. No Chhunneng pair,
`known_targets.txt` or old-model output is used.

**Leak filter (`build`):**
- Held out: every evaluation panel (romanised test, Wikidata test2 names, pick test, dev-41, reserve, list test)
  and every validation set (list val, Wikidata dev names, romanised val).
- A training row is dropped if its Latin form is related to a held-out Latin word (equal, or a simple suffix
  variant: `stems`), or if its Khmer target is a held-out Khmer target. This applies to the pronunciation rows too
  (Latin side).
- Counts are recorded per source and per held-out set, each set counted on its own.

**Side task:** Google lexicon Latin entries (a–z, non-empty pronunciation). The target is the phone-token sequence
with "." syllable boundaries. Alternative pronunciations of one word are kept as separate rows. No Khmer spelling
is needed.

**Model:**
- Arm `base`: the letters-only encoder–decoder of `translit_train.Model` (d 192, 3 + 3 layers, 4 heads, FF 512,
  dropout 0.1, `MAX_IN` 48).
- Arm `aux`: the same model plus `AuxDecoder`, a second decoder of the same size over phone tokens, reading the
  shared encoder. It is discarded at export.
- Both arms use the same initial shared parameters: the aux modules are built after the shared ones, and the
  global RNG is restored after building them.

**Training (registered in D173):**
- AdamW lr 1e-3, wd 0.01; warm-up 500 steps, then cosine; batch 128; 40 epochs; label smoothing 0.1.
- Cross-entropy is averaged over non-pad target tokens.
- Spelling batches come from a private `random.Random(seed)`, so both arms see identical batches in identical
  order, with the same number of optimizer steps.
- Aux arm, on every `AUX_EVERY`-th spelling step:
  - one aux batch of `AUX_BATCH` rows is drawn from its own `random.Random(10_000 + seed)`;
  - the loss of that step is `spell + AUX_WEIGHT · aux`, with one backward pass and one optimizer step;
  - the aux forward runs under its own torch RNG state (saved and restored around it), so the spelling dropout
    stream is unchanged.
- Gradient clipping at norm 1.0 over all trained parameters.
- Checkpoint selection, after every epoch: the equally weighted mean of greedy exact match on the three
  validation sets (romanised, names, English). Spelling only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from khmer_translit.loanwords import LIST_TSV as LOANWORDS  # noqa: E402
D = ROOT / "models" / "unified"
EXPERIMENT = ROOT / "experiments" / "E01-unified"
CONFIG = EXPERIMENT / "config.yaml"
REPORT = EXPERIMENT / "results.json"
LEX = ROOT / "data/lexicon/km_lexicon.tsv"
MAX_IN, AUX_MAX = 48, 48
LOAN_REPEAT, BATCH, EPOCHS, LR, WARMUP = 4, 128, 40, 1e-3, 500
AUX_WEIGHT, AUX_EVERY, AUX_BATCH = 0.5, 8, 128
# Config schema (`experiment.load_config`): every key must be in the config file. The values here are E01's.
CONFIG_KEYS = {
    "paths": {"runs_dir": "models/unified"},
    "data": {"max_in": MAX_IN, "aux_max": AUX_MAX, "loan_repeat": LOAN_REPEAT, "roman_val_n": 1000},
    "model": {"d": 192, "layers": 3, "heads": 4, "ff": 512, "dropout": 0.1},
    "train": {"seeds": [0, 1, 2], "epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": 0.01, "warmup": WARMUP,
              "label_smoothing": 0.1, "grad_clip": 1.0, "threads": 6},
    "aux": {"weight": AUX_WEIGHT, "every": AUX_EVERY, "batch": AUX_BATCH},
    "eval": {"benchmark": "v1", "beam": 4},
}
DEFAULT_CONFIG = {**CONFIG_KEYS, "overrides": []}
# Settings this code implements but does not read from the config yet: a config that changes them is refused.
FIXED = {"paths": CONFIG_KEYS["paths"], "data": CONFIG_KEYS["data"], "model": CONFIG_KEYS["model"]}


def check_implemented(cfg: dict) -> None:
    """Refuse a config whose data, model or path settings differ from what this script implements.

    Those values are built into the data stage and the model classes; changing them needs a code change, which
    a config edit alone must not pretend to make.

    Args:
        cfg: A config from `experiment.load_config`.

    Raises:
        ValueError: Naming each differing setting.
    """
    diff = [f"{s}.{k}={cfg[s][k]!r} (implemented: {v!r})" for s, keys in FIXED.items() for k, v in keys.items()
            if cfg[s][k] != v]
    if diff:
        raise ValueError("not configurable in this script yet: " + ", ".join(diff))
ROMAN_VAL_N = 1000
SUFFIXES = ("ings", "ing", "ers", "er", "ies", "ied", "es", "ed", "ly", "s")
PAD, BOS, EOS = 0, 1, 2


# ---------------------------------------------------------------- leak filter
def stems(word: str) -> set[str]:
    """The word and its simple suffix-stripped stems (with a restored final e), for variant matching.

    Args:
        word: Lower-case Latin word.

    Returns:
        A set of strings of length ≥ 3 (the word itself always included, whatever its length).

    Examples:
        >>> sorted(stems("sharing"))
        ['shar', 'share', 'sharing']
        >>> sorted(stems("ok"))
        ['ok']
    """
    out = {word}
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            base = word[: -len(suf)]
            out |= {base, base + "e", base + "y"} if suf in ("ies", "ied") else {base, base + "e"}
            if len(base) >= 4 and base[-1] == base[-2]:
                out.add(base[:-1])
    return {s for s in out if len(s) >= 3} | {word}


ENGLISH_SETS = ("pick_test", "dev41", "reserve", "list_test", "val_english")


def norm(word: str) -> str:
    """Matching form of a Latin word: Unicode NFKD with accents removed, lower case, letters a–z only (so
    "Wi-Fi", "wi fi", "Wí'fi" and "wifi" all match).

    Examples:
        >>> norm("Wi-Fi"), norm("Café's")
        ('wifi', 'cafes')
    """
    import unicodedata
    w = unicodedata.normalize("NFKD", word).lower()
    return re.sub("[^a-z]", "", "".join(c for c in w if not unicodedata.combining(c)))


def related(word: str, held_stems: set[str]) -> bool:
    """Whether a Latin word equals or is a simple variant of a held-out word.

    Examples:
        >>> related("tablet", stems("tablets")), related("table", stems("tablets"))
        (True, False)
    """
    return bool(stems(word) & held_stems)


# ---------------------------------------------------------------- data
def _tsv(path: Path, a: int = 0, b: int = 1) -> list[tuple[str, str]]:
    """Two columns of a TSV with a header row (comment and blank lines skipped)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if line and not line.startswith("#") and "\t" in line:
            c = line.split("\t")
            out.append((c[a].strip(), c[b].strip()))
    return out


def _names(path: Path) -> list[tuple[str, str]]:
    """Word-aligned `(latin, khmer)` pairs of a Wikidata name file, unique and in file order.

    Each name is aligned word by word in the Khmer word order (same or reversed) whose total rule similarity is
    higher (D135): a fixed alignment that does not depend on any model being tested.

    Args:
        path: A names TSV (`qid<TAB>latin<TAB>khmer<TAB>seen_in_train`, with a header row), e.g.
            `data/names_dev.tsv`.

    Returns:
        The unique lower-cased word pairs, e.g. `[("sok", "សុខ"), ...]`.
    """
    from khmer_translit import latin_g2p
    pairs = []
    for row in path.read_text(encoding="utf-8").splitlines()[1:]:
        _qid, lat, km, _seen = row.split("\t")
        lw, kw = lat.split(), km.split()
        orders = [kw, kw[::-1]] if len(kw) > 1 else [kw]
        best = max(orders, key=lambda o: sum(latin_g2p.similarity(latin_g2p.transliterate(a), b) for a, b in zip(lw, o)))
        pairs += list(zip(lw, best))
    return list(dict.fromkeys((a.lower(), b) for a, b in pairs))


def live_list() -> dict[str, str]:
    """The live loanword list: lower-case Latin → Khmer."""
    out = {}
    for line in LOANWORDS.read_text(encoding="utf-8").splitlines():
        if "\t" in line and not line.startswith("#") and not line.startswith("word\t"):
            w, k = line.split("\t")[:2]
            out[w.strip().lower()] = k.strip()
    return out


EVAL_SETS = ("romanised_test", "names_test2", "pick_test", "dev41", "reserve", "list_test")
VAL_SETS = ("val_english", "val_names", "val_romanised")


def _raw_heldout() -> dict[str, dict]:
    """Every evaluation panel and validation set before validation pruning (see `heldout`)."""
    T = ROOT / "data"
    rows = [l.split("\t") for l in (T / "pairs.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    attested: dict[str, set[str]] = {}
    for r in rows:
        attested.setdefault(r[0], set()).add(r[1])
    split = json.loads((T / "list_split.json").read_text(encoding="utf-8"))
    live = live_list()
    cands = json.loads((T / "rerank/cands.json").read_text(encoding="utf-8"))
    roman_val = [(a, b) for a, b, _s, sp in (l.split("\t") for l in
                 (T / "roman/pairs.tsv").read_text(encoding="utf-8").splitlines()[1:]) if sp == "val"]
    return {
        # every Chhunneng-attested form of a test Latin word is accepted (and held out), as in D133–D137
        "romanised_test": {"role": "eval", "pairs": sorted({(r[0], k) for r in rows if r[3] == "test"
                                                            for k in attested[r[0]]})},
        "names_test2": {"role": "eval", "pairs": _names(T / "names_test2.tsv")},
        "pick_test": {"role": "eval", "pairs": _tsv(T / "synth/test.tsv")},
        "dev41": {"role": "eval", "pairs": [(w, x["ref"]) for w, x in cands["dev"].items()]},
        "reserve": {"role": "eval", "pairs": [(w, "") for w in json.loads((T / "e2/reserve.json").read_text(encoding="utf-8"))]},
        "list_test": {"role": "eval", "pairs": [(w, live[w]) for w in split["test"] if w in live]},
        "val_english": {"role": "val", "pairs": [(w, live[w]) for w in split["val"] if w in live]},
        "val_names": {"role": "val", "pairs": _names(T / "names_dev.tsv")},
        "val_romanised": {"role": "val", "pairs": roman_val},
    }


def match_sets(held: dict[str, dict]) -> dict[str, tuple[set[str], set[str]]]:
    """Per held-out set, the Latin match set and the Khmer targets used by the leak filter.

    English sets (`ENGLISH_SETS`) hold each word's suffix stems; romanised and name sets hold the exact normalised
    word only. Matching against them is done by `latin_hits` / `row_hits` only.

    Returns:
        `{set name: (latin match set, khmer targets)}`.

    Examples:
        >>> m = match_sets({"pick_test": {"pairs": [("Tablets", "ក")]}, "val_names": {"pairs": [("Sokha", "ខ")]}})
        >>> "tablet" in m["pick_test"][0], sorted(m["val_names"][0])
        (True, ['sokha'])
    """
    return {n: ({s for a, _ in h["pairs"] for s in (stems(norm(a)) if n in ENGLISH_SETS else {norm(a)})},
                {b for _, b in h["pairs"] if b}) for n, h in held.items()}


def latin_hits(word: str, sets: dict[str, tuple[set[str], set[str]]]) -> list[str]:
    """Held-out sets whose Latin side matches a word, under each set's own policy.

    English sets: the word's suffix stems meet the set's stems (inflections match). Name and romanised sets: the
    exact normalised word only (no suffix stripping of either side).

    Examples:
        >>> sets = match_sets({"val_names": {"pairs": [("sokha", "")]}, "pick_test": {"pairs": [("share", "")]}})
        >>> latin_hits("sokhas", sets), latin_hits("Sokha", sets), latin_hits("sharing", sets)
        ([], ['val_names'], ['pick_test'])
    """
    w = norm(word)
    st = stems(w)
    return [n for n, (ms, _km) in sets.items() if (st & ms if n in ENGLISH_SETS else w in ms)]


def row_hits(latin: str, khmer: str, sets: dict[str, tuple[set[str], set[str]]]) -> tuple[list[str], list[str]]:
    """`(sets hit by the Latin side, sets hit by the Khmer target)` for one training row (the one filter rule)."""
    return latin_hits(latin, sets), ([n for n, (_ms, km) in sets.items() if khmer and khmer in km])


def val_eval_audit(raw: dict[str, dict]) -> dict:
    """Overlap of every validation set with every evaluation panel (reserve: Latin only; its answers unopened).

    For each (validation, evaluation) pair it counts validation items with an exact normalised Latin match, a
    variant-only match (suffix stems meet, not exact), and a shared Khmer target. For the name sets it also
    counts shared Wikidata person IDs.

    Returns:
        `{"pairs": {"val|eval": {...}}, "shared_name_qids": n}`.
    """
    out = {}
    for v in VAL_SETS:
        for e in EVAL_SETS:
            e_exact = {norm(a) for a, _ in raw[e]["pairs"]}
            e_stems = {s for a, _ in raw[e]["pairs"] for s in stems(norm(a))}
            e_km = {b for _, b in raw[e]["pairs"] if b}
            exact = sum(norm(a) in e_exact for a, _ in raw[v]["pairs"])
            variant = sum(norm(a) not in e_exact and bool(stems(norm(a)) & e_stems) for a, _ in raw[v]["pairs"])
            km = sum(bool(b) and b in e_km for _, b in raw[v]["pairs"])
            out[f"{v}|{e}"] = {"exact_latin": exact, "variant_only_latin": variant, "shared_khmer": km}
    q = lambda f: {l.split("\t")[0] for l in (ROOT / "data" / f).read_text(encoding="utf-8").splitlines()[1:]}
    return {"pairs": out, "shared_name_qids": len(q("names_dev.tsv") & q("names_test2.tsv"))}


def heldout() -> dict[str, dict]:
    """Evaluation panels (unchanged) and validation sets pruned of evaluation material.

    A validation item is removed if its Latin side hits any evaluation panel under that panel's policy
    (`latin_hits`), or its Khmer target is an evaluation target. Evaluation panels are never changed, and
    validation is not replenished.

    Returns:
        `{name: {"pairs": [(latin, khmer or "")], "role"}}`.
    """
    raw = _raw_heldout()
    ev = match_sets({n: raw[n] for n in EVAL_SETS})
    out = {n: raw[n] for n in EVAL_SETS}
    for v in VAL_SETS:
        keep = [(a, b) for a, b in raw[v]["pairs"] if not any(row_hits(a, b, ev))]
        out[v] = {"role": "val", "pairs": keep}
    return out


def google_latin() -> list[tuple[str, list[str]]]:
    """Google lexicon Latin entries with a non-empty pronunciation: `(word, phone tokens)`; alternatives kept."""
    out = []
    for line in LEX.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "\t" not in line:
            continue
        w, p = line.split("\t")[:2]
        if all("ក" <= c <= "៿" for c in w):
            continue
        out.append((w.strip().lower(), p.split()))
    return out


def sha(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    """Write the filtered training files, validation sets and `manifest.json` into `D`.

    Returns:
        The manifest (counts, exclusions, vocabularies, hashes, rules).
    """
    T = ROOT / "data"
    list_split = lambda: json.loads((T / "list_split.json").read_text(encoding="utf-8"))  # frozen (D146)
    raw = _raw_heldout()
    held = heldout()
    per_set = match_sets(held)
    live = live_list()
    roman = [(a, b) for a, b, _s, sp in (l.split("\t") for l in (T / "roman/pairs.tsv").read_text(encoding="utf-8").splitlines()[1:])
             if sp == "train"]
    sources = {
        "roman": roman,
        "mined": _tsv(T / "extra_train.tsv"),
        "list_train": [(w, live[w]) for w in list_split()["train"] if w in live],
        "owner_examples": _tsv(T / "synth/examples.tsv"),
        "llm_drafts": _tsv(T / "llm_drafts.tsv"),
    }
    kept, excl = {}, {}
    for name, rows in sources.items():
        good, by = [], {"latin_related": 0, "khmer_target": 0, "bad_chars_or_length": 0, "duplicate": 0}
        by_set = {n: 0 for n in held}
        seen = set()
        for a, b in rows:
            a = a.lower()
            lat_hit, km_hit = row_hits(a, b, per_set)
            for n in set(lat_hit) | set(km_hit):
                by_set[n] += 1
            if not re.fullmatch("[a-z]+", a) or len(a) > MAX_IN or not b or len(b) > 40 \
                    or not all("ក" <= c <= "៿" for c in b):
                by["bad_chars_or_length"] += 1
            elif lat_hit:
                by["latin_related"] += 1
            elif km_hit:
                by["khmer_target"] += 1
            elif (a, b) in seen:
                by["duplicate"] += 1
            else:
                seen.add((a, b))
                good.append((a, b))
        kept[name], excl[name] = good, {"rows_in": len(rows), "kept": len(good), "dropped": by,
                                        "hits_per_heldout_set": by_set}
    aux_rows, aux_by = [], {"empty": 0, "not_az_or_length": 0, "latin_related": 0, "duplicate": 0}
    aux_set = {n: 0 for n in held}
    seen = set()
    for w, toks in google_latin():
        lat_hit = latin_hits(w, per_set)
        for n in lat_hit:
            aux_set[n] += 1
        if not toks:
            aux_by["empty"] += 1
        elif not re.fullmatch("[a-z]+", w) or len(w) > MAX_IN or len(toks) > AUX_MAX:
            aux_by["not_az_or_length"] += 1
        elif lat_hit:
            aux_by["latin_related"] += 1
        elif (w, tuple(toks)) in seen:
            aux_by["duplicate"] += 1
        else:
            seen.add((w, tuple(toks)))
            aux_rows.append((w, " ".join(toks)))
    spell = [(a, b, n) for n, rows in kept.items() for a, b in rows]
    src_chars = sorted({c for a, _b, _n in spell for c in a} | {c for w, _p in aux_rows for c in w})
    tgt_chars = sorted({c for _a, b, _n in spell for c in b})
    aux_vocab = sorted({t for _w, p in aux_rows for t in p.split()})
    D.mkdir(parents=True, exist_ok=True)
    (D / "spell_train.tsv").write_text("latin\tkhmer\tsource\n" + "".join(f"{a}\t{b}\t{n}\n" for a, b, n in spell),
                                       encoding="utf-8")
    (D / "aux_train.tsv").write_text("latin\tphones\n" + "".join(f"{w}\t{p}\n" for w, p in aux_rows), encoding="utf-8")
    rv = held["val_romanised"]["pairs"]
    rv = sorted(rv, key=lambda p: hashlib.sha1(p[1].encode()).hexdigest())[:ROMAN_VAL_N]
    vals = {"romanised": rv, "names": held["val_names"]["pairs"], "english": held["val_english"]["pairs"]}
    val_words = {norm(a) for n in VAL_SETS for a, _ in held[n]["pairs"]}
    pron_diag = [(w, " ".join(t)) for w, t in google_latin()
                 if t and re.fullmatch("[a-z]+", w) and w in val_words and len(t) <= AUX_MAX]
    (D / "pron_diag.tsv").write_text("latin\tphones\n" + "".join(f"{w}\t{p}\n" for w, p in pron_diag), encoding="utf-8")
    for n, pairs in vals.items():
        (D / f"val_{n}.tsv").write_text("latin\tkhmer\n" + "".join(f"{a}\t{b}\n" for a, b in pairs), encoding="utf-8")
    inputs = [LEX, T / "roman/pairs.tsv", T / "extra_train.tsv", T / "llm_drafts.tsv", T / "synth/examples.tsv",
              T / "list_split.json", LOANWORDS, T / "pairs.tsv", T / "names_test2.tsv",
              T / "names_dev.tsv", T / "synth/test.tsv", T / "rerank/cands.json", T / "e2/reserve.json"]
    manifest = {
        "rules": {"latin": "row_hits(): norm() both sides (NFKD, no accents, lower, a-z only); English sets (pick, dev41, "
                           "reserve, list test, list val) match suffix stems of both words; romanised and name sets match the "
                           "exact normalised word only",
                  "validation": "validation items hitting any evaluation panel (same rule, plus shared Khmer target) are removed; "
                                "panels unchanged; validation not replenished",
                  "khmer": "drop if khmer = any held-out khmer target (the reserve has no answers: its Khmer targets cannot be filtered)",
                  "aux": "Google Latin a–z, non-empty pronunciation, same Latin rule; alternatives kept",
                  "loans": f"list_train + owner_examples + llm_drafts repeated {LOAN_REPEAT}x"},
        "heldout_sizes": {n: len(h["pairs"]) for n, h in held.items()},
        "val_sizes_before_pruning": {n: len(raw[n]["pairs"]) for n in VAL_SETS},
        "val_eval_audit_before_pruning": val_eval_audit(raw),
        "val_eval_audit_after_pruning": val_eval_audit({**raw, **{n: held[n] for n in VAL_SETS}}),
        "sources": excl, "aux": {"rows_in": len(google_latin()), "kept": len(aux_rows), "dropped": aux_by,
                                 "hits_per_heldout_set": aux_set,
                                 "words_with_alternatives": sum(1 for w in {r[0] for r in aux_rows}
                                                                if sum(r[0] == w for r in aux_rows) > 1)},
        "val_sizes": {n: len(p) for n, p in vals.items()},
        "pron_diag_size": len(pron_diag),
        "vocab": {"src": "".join(src_chars), "tgt": "".join(tgt_chars), "aux": aux_vocab},
        "lengths": {"max_in": MAX_IN, "max_latin_seen": max(len(a) for a, _b, _n in spell + [(w, "", "") for w, _ in aux_rows]),
                    "max_out": 40, "max_khmer_seen": max(len(b) for _a, b, _n in spell),
                    "aux_max": AUX_MAX, "max_aux_seen": max(len(p.split()) for _w, p in aux_rows)},
        "input_sha256": {os.path.relpath(p, ROOT): sha(p) for p in inputs},   # the list lives in the speech repo
        "output_sha256": {f: sha(D / f) for f in ("spell_train.tsv", "aux_train.tsv", "val_romanised.tsv",
                                                    "val_names.tsv", "val_english.tsv")},
    }
    (D / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest


def load_data() -> tuple[list[tuple[str, str]], list[tuple[str, list[str]]], dict, dict]:
    """Training pairs (loans repeated), aux rows, validation sets and the manifest, as written by `build`."""
    manifest = json.loads((D / "manifest.json").read_text(encoding="utf-8"))
    rows = [l.split("\t") for l in (D / "spell_train.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    loans = {"list_train", "owner_examples", "llm_drafts"}
    spell = [(a, b) for a, b, n in rows if n not in loans] + [(a, b) for a, b, n in rows if n in loans] * LOAN_REPEAT
    aux = [(w, p.split()) for w, p in (l.split("\t") for l in (D / "aux_train.tsv").read_text(encoding="utf-8").splitlines()[1:])]
    vals = {n: _tsv(D / f"val_{n}.tsv") for n in ("romanised", "names", "english")}
    return spell, aux, vals, manifest


# ---------------------------------------------------------------- model (torch)
def encode_batch(items: list[str], vocab: str) -> "torch.Tensor":
    """Pad Latin strings to ids (+3 for the specials). Raises if any string is longer than `MAX_IN` (no truncation)."""
    import torch
    if any(len(s) > MAX_IN for s in items):
        raise ValueError("source longer than MAX_IN: truncation is not allowed")
    S = torch.zeros(len(items), max(map(len, items)), dtype=torch.long)
    for r, s in enumerate(items):
        S[r, :len(s)] = torch.tensor([vocab.index(c) + 3 for c in s])
    return S


def target_batch(seqs: list[list[str]], vocab: list[str] | str, max_len: int) -> "torch.Tensor":
    """BOS + ids + EOS, padded. Raises if a sequence is longer than `max_len`."""
    import torch
    if any(len(s) > max_len for s in seqs):
        raise ValueError("target longer than its cap: truncation is not allowed")
    index = {t: i + 3 for i, t in enumerate(vocab)}
    rows = [[BOS] + [index[t] for t in s] + [EOS] for s in seqs]
    T = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long)
    for r, s in enumerate(rows):
        T[r, :len(s)] = torch.tensor(s)
    return T


def make_aux_decoder(n_aux: int, d: int = 192, layers: int = 3, heads: int = 4, ff: int = 512, dropout: float = 0.1):
    """The auxiliary pronunciation decoder: same block design as the spelling decoder, its own embeddings.

    Args:
        n_aux: Phone vocabulary size including the 3 specials.

    Returns:
        A torch `nn.Module` with `forward(memory, pad, tgt_in) -> logits`.
    """
    import torch
    from torch import nn
    from translit_train import Block

    class AuxDecoder(nn.Module):
        """Decoder over phone tokens that reads the shared encoder's memory."""

        def __init__(self) -> None:
            super().__init__()
            self.emb, self.pos = nn.Embedding(n_aux, d), nn.Embedding(AUX_MAX + 2, d)
            self.dec = nn.ModuleList(Block(d, heads, ff, dropout, True) for _ in range(layers))
            self.ln, self.out, self.drop = nn.LayerNorm(d), nn.Linear(d, n_aux), nn.Dropout(dropout)

        def forward(self, memory: torch.Tensor, pad: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
            """Teacher-forced phone logits `(B, Tt, n_aux)`."""
            L = tgt_in.shape[1]
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)[None]
            g = self.drop(self.emb(tgt_in) + self.pos.weight[:L])
            for blk in self.dec:
                g = blk(g, causal, memory, pad)
            return self.out(self.ln(g))

    return AuxDecoder()


def encoder_memory(model, src):
    """Run only the shared encoder of a `translit_train.Model`: `(memory, pad mask)`."""
    from translit_train import PAD as _P
    pad = (src == _P)[:, None, :]
    h = model.drop(model.src_emb(src) + model.src_pos.weight[: src.shape[1]])
    for blk in model.enc:
        h = blk(h, pad)
    return model.enc_ln(h), pad


def build_models(n_src: int, n_tgt: int, n_aux: int, arm: str, seed: int):
    """Shared model (identical across arms for a seed) and, for the aux arm, the aux decoder.

    The global torch RNG is restored after the aux decoder is built, so the training-time dropout stream is the
    same in both arms.

    Returns:
        `(model, aux_or_None)`.
    """
    import torch
    from translit_train import Model
    torch.manual_seed(seed)
    model = Model(n_src, n_tgt, max_in=MAX_IN)
    aux = None
    if arm == "aux":
        state = torch.get_rng_state()
        aux = make_aux_decoder(n_aux)
        torch.set_rng_state(state)
    return model, aux


def greedy_exact(model, pairs: list[tuple[str, str]], src_v: str, tgt_v: str) -> int:
    """Number of pairs the spelling decoder gets exactly right by greedy decoding (eval mode)."""
    import torch
    from translit_train import greedy
    model.eval()
    hit = 0
    with torch.no_grad():
        for i in range(0, len(pairs), 512):
            chunk = pairs[i:i + 512]
            S = encode_batch([a for a, _ in chunk], src_v)
            for ids, (_a, b) in zip(greedy(model, S), chunk):
                hit += "".join(tgt_v[t - 3] for t in ids if t >= 3) == b
    model.train()
    return hit


def aux_exact(model, aux, rows: list[tuple[str, list[str]]], src_v: str, aux_v: list[str]) -> float:
    """Diagnostic only: greedy phone-sequence exact match of the aux decoder on held-out words (never used for
    checkpoint selection or any recipe choice).

    Returns:
        Exact-match rate in [0, 1] (0.0 for an empty list).
    """
    import torch
    if not rows:
        return 0.0
    model.eval(), aux.eval()
    hit = 0
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            chunk = rows[i:i + 256]
            memory, pad = encoder_memory(model, encode_batch([w for w, _ in chunk], src_v))
            out = torch.full((len(chunk), 1), BOS, dtype=torch.long)
            for _ in range(AUX_MAX + 1):
                nxt = aux(memory, pad, out)[:, -1].argmax(-1)
                out = torch.cat([out, nxt[:, None]], 1)
                if bool((out == EOS).any(1).all()):
                    break
            for row, (_w, ref) in zip(out[:, 1:].tolist(), chunk):
                row = row[: row.index(EOS)] if EOS in row else row
                hit += [aux_v[t - 3] for t in row if t >= 3] == ref
    model.train(), aux.train()
    return hit / len(rows)


def train(arm: str, seed: int, epochs: int | None = None, out: Path | None = None, data: tuple | None = None,
          trace: bool = False, cfg: dict | None = None, run=None) -> dict:
    """Train one arm for one seed; keep the best checkpoint by the mean spelling-validation exact match.

    Args:
        arm: "base" or "aux".
        seed: Paired seed (0, 1, 2).
        epochs: Epochs; None = the config's (40 registered; smaller only in tests and smoke runs).
        out: Output folder (default `models/unified/<arm>_s<seed>`).
        data: `(spell, aux_rows, vals, manifest)` instead of `load_data()` (tests only).
        trace: Record a hash of the global torch RNG state after every spelling forward pass (tests only).
        cfg: The resolved config (default `DEFAULT_CONFIG`, = E01).
        run: An `experiment.Run` that receives every epoch record and the final summary (None: no records).

    Returns:
        The training log (also written as `train_log.json`).
    """
    import torch
    from torch import nn
    c = cfg or DEFAULT_CONFIG
    tc, ac = c["train"], c["aux"]
    epochs = tc["epochs"] if epochs is None else epochs
    torch.set_num_threads(tc["threads"])
    spell, aux_rows, vals, manifest = data if data is not None else load_data()
    src_v, tgt_v, aux_v = manifest["vocab"]["src"], manifest["vocab"]["tgt"], manifest["vocab"]["aux"]
    model, aux = build_models(len(src_v) + 3, len(tgt_v) + 3, len(aux_v) + 3, arm, seed)
    params = list(model.parameters()) + (list(aux.parameters()) if aux is not None else [])
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    batch, warmup = tc["batch"], tc["warmup"]
    steps = epochs * math.ceil(len(spell) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=tc["label_smoothing"])
    rng_spell, rng_aux = random.Random(seed), random.Random(10_000 + seed)
    saved = torch.get_rng_state()
    torch.manual_seed(10_000 + seed)
    aux_state = torch.get_rng_state()
    torch.set_rng_state(saved)
    aux_order, aux_pos = [], 0
    diag_path = D / "pron_diag.tsv"
    pron_diag = ([(w, p.split()) for w, p in (l.split("\t") for l in diag_path.read_text(encoding="utf-8").splitlines()[1:])]
                 if diag_path.exists() and data is None else [])
    out = out or D / f"{arm}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    log, best, best_epoch, t0, step, aux_steps, rng_trace = [], Fraction(-1), 0, time.time(), 0, 0, []
    batch_digest = hashlib.sha256()
    for ep in range(epochs):
        order = list(range(len(spell)))
        rng_spell.shuffle(order)
        tot_s = tot_a = 0.0
        gn = {"aux_steps": [], "spell_only_steps": []}
        for i in range(0, len(order), batch):
            chunk = [spell[j] for j in order[i:i + batch]]
            batch_digest.update(",".join(map(str, order[i:i + batch])).encode())
            S = encode_batch([a for a, _ in chunk], src_v)
            Tt = target_batch([list(b) for _, b in chunk], tgt_v, 40)
            logits = model(S, Tt[:, :-1])
            loss = ce(logits.reshape(-1, logits.shape[-1]), Tt[:, 1:].reshape(-1))
            tot_s += loss.item() * len(chunk)
            if trace:
                rng_trace.append(hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:16])
            if aux is not None and step % ac["every"] == 0:
                if aux_pos + ac["batch"] > len(aux_order):
                    aux_order = list(range(len(aux_rows)))
                    rng_aux.shuffle(aux_order)
                    aux_pos = 0
                ach = [aux_rows[j] for j in aux_order[aux_pos:aux_pos + ac["batch"]]]
                aux_pos += ac["batch"]
                saved = torch.get_rng_state()
                torch.set_rng_state(aux_state)
                Sa = encode_batch([w for w, _ in ach], src_v)
                Ta = target_batch([p for _, p in ach], aux_v, AUX_MAX)
                memory, pad = encoder_memory(model, Sa)
                la = aux(memory, pad, Ta[:, :-1])
                loss_a = ce(la.reshape(-1, la.shape[-1]), Ta[:, 1:].reshape(-1))
                aux_state = torch.get_rng_state()
                torch.set_rng_state(saved)
                loss = loss + ac["weight"] * loss_a
                tot_a += loss_a.item()
                aux_steps += 1
            opt.zero_grad()
            loss.backward()
            norm_ = float(nn.utils.clip_grad_norm_(params, tc["grad_clip"]))
            gn["aux_steps" if aux is not None and step % ac["every"] == 0 else "spell_only_steps"].append(norm_)
            opt.step()
            sched.step()
            step += 1
        hits = {n: greedy_exact(model, p, src_v, tgt_v) for n, p in vals.items()}
        score = sum(Fraction(hits[n], len(p)) for n, p in vals.items()) / 3       # exact; rounded only for display
        n_aux_ep = len(gn["aux_steps"]) if aux is not None else 0
        log.append({"epoch": ep + 1, "spell_loss": round(tot_s / len(spell), 4),
                    "aux_loss": round(tot_a / max(1, n_aux_ep), 4) if aux is not None else None,
                    "grad_norm_mean": {k: round(sum(v) / len(v), 4) if v else None for k, v in gn.items()},
                    "pron_diag_exact": round(aux_exact(model, aux, pron_diag, src_v, aux_v), 4) if aux is not None else None,
                    "val_hits": hits, "val_n": {n: len(p) for n, p in vals.items()},
                    "val": {n: round(hits[n] / len(p), 4) for n, p in vals.items()},
                    "val_mean": round(float(score), 4), "minutes": round((time.time() - t0) / 60, 1)})
        print(json.dumps(log[-1]), flush=True)
        if run is not None:
            run.log(log[-1], step=ep + 1)
        if score > best:
            best, best_epoch = score, ep + 1
            torch.save({"state": model.state_dict(), "aux": aux.state_dict() if aux is not None else None,
                        "src_vocab": src_v, "tgt_vocab": tgt_v, "aux_vocab": aux_v, "d": 192, "layers": 3, "heads": 4,
                        "epoch": ep + 1, "phones": False, "max_in": MAX_IN, "tags": False}, out / "best.pt")
    result = {"arm": arm, "seed": seed, "epochs": epochs, "spell_steps": step, "aux_steps": aux_steps,
              "spell_pairs": len(spell), "aux_rows": len(aux_rows), "batch_order_sha256": batch_digest.hexdigest(),
              "params_shared": sum(p.numel() for p in model.parameters()),
              "params_aux": sum(p.numel() for p in aux.parameters()) if aux is not None else 0,
              "best_val_mean": round(float(best), 6), "best_epoch": best_epoch,
              "wall_minutes": round((time.time() - t0) / 60, 1), "manifest_sha256": sha(D / "manifest.json"), "log": log}
    if trace:
        result["rng_trace"] = rng_trace
    (out / "train_log.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    if run is not None:
        run.finish({k: v for k, v in result.items() if k not in ("log", "rng_trace")})
    return result


def export(folder: Path) -> Path:
    """Write the spelling part of `best.pt` as `translit.npz` (the aux decoder is discarded).

    Returns:
        The npz path.
    """
    import numpy as np
    import torch
    ck = torch.load(folder / "best.pt")
    arrays = {k: v.numpy().astype(np.float32) for k, v in ck["state"].items()}
    meta = {"src_vocab": ck["src_vocab"], "tgt_vocab": ck["tgt_vocab"], "heads": ck["heads"], "layers": ck["layers"],
            "epoch": ck["epoch"], "phones": False, "max_in": ck["max_in"], "tags": False}
    path = folder / "translit.npz"
    np.savez(path, __meta__=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    return path


# ---------------------------------------------------------------- scoring
def panels() -> dict[str, list[tuple[str, frozenset[str]]]]:
    """The D173 panels rebuilt from the data (never the reserve), one item per Latin word with all its attested
    Khmer forms. Frozen as benchmark v1 (`khmer_translit.benchmark`), which `score` uses; kept to rebuild or audit
    the panels.

    A word counts as right if the output equals any attested form (the project's long-standing rule, D133).

    Returns:
        `{panel: [(latin, {khmer, ...}), ...]}` for romanised test, names test2, pick test, dev-41 (diagnostic) and
        list test, sorted by Latin word.
    """
    held = heldout()
    out = {}
    for n in ("romanised_test", "names_test2", "pick_test", "dev41", "list_test"):
        forms: dict[str, set[str]] = {}
        for a, b in held[n]["pairs"]:
            forms.setdefault(a.lower(), set()).add(b)
        out[n] = [(a, frozenset(forms[a])) for a in sorted(forms)]
    return out


def predict(npz: Path, words: list[str], known: frozenset[str]) -> tuple[list[str], list[str]]:
    """Raw (beam-4 top) and reranked (`translit.pick` with `known`) spellings for each word."""
    from khmer_translit import translit
    model = translit.load(npz)
    raw, rer = [], []
    for nb in model.nbest_batch(words, 4):
        raw.append(nb[0][0] if nb else "")
        rer.append(translit.pick(nb, known))
    return raw, rer


def gate(counts: dict[int, dict[str, tuple[int, int, int]]]) -> dict:
    """The registered D173 gate, from integer correctness counts (no rounding in any decision).

    Args:
        counts: `{seed: {panel: (aux_correct, base_correct, n)}}` for panels "pick_test", "romanised_test" and
            "names_test2".

    Returns:
        Exact per-seed and mean changes (as floats for display, computed from fractions) and `passes`, which
        requires exactly seeds {0, 1, 2}.

    Examples:
        >>> c = {s: {"pick_test": (60, 50, 150), "romanised_test": (1000 - l, 1000, 2226), "names_test2": (100, 100, 228)}
        ...      for s, l in zip((0, 1, 2), (22, 22, 23))}
        >>> g = gate(c); round(g["romanised_mean_pp"], 6), g["passes"]
        (-1.003294, False)
    """
    seeds = sorted(counts)
    pp = {p: [Fraction(100) * (counts[s][p][0] - counts[s][p][1]) / counts[s][p][2] for s in seeds]
          for p in ("pick_test", "romanised_test", "names_test2")}
    mean = {p: sum(v) / len(v) for p, v in pp.items()} if seeds else {}
    ok = (seeds == [0, 1, 2] and mean["pick_test"] >= 5 and sum(d > 0 for d in pp["pick_test"]) >= 2
          and mean["romanised_test"] >= -1 and mean["names_test2"] >= -1)
    return {"seeds": seeds, "english_pp_per_seed": [float(d) for d in pp["pick_test"]],
            "romanised_pp_per_seed": [float(d) for d in pp["romanised_test"]],
            "names_pp_per_seed": [float(d) for d in pp["names_test2"]],
            "english_mean_pp": float(mean.get("pick_test", 0)), "romanised_mean_pp": float(mean.get("romanised_test", 0)),
            "names_mean_pp": float(mean.get("names_test2", 0)), "passes": bool(ok)}


def valid_runs() -> list[str]:
    """Run folders that may enter the gate: exactly `base_s{0,1,2}` / `aux_s{0,1,2}`, exported, and trained on the
    current manifest (stale exports, superseded or unrelated folders are ignored)."""
    current = sha(D / "manifest.json")
    out = []
    for arm in ("base", "aux"):
        for seed in (0, 1, 2):
            run = D / f"{arm}_s{seed}"
            logf = run / "train_log.json"
            if (run / "translit.npz").exists() and logf.exists() \
                    and json.loads(logf.read_text(encoding="utf-8")).get("manifest_sha256") == current:
                out.append(run.name)
    return out


def served_as_if_unlisted(words: list[str]) -> list[str]:
    """The served fallback router for words as if they were not in the loanword list (`translit.guess_many`):
    English-dictionary words → E (with its CMUdict-phone input), f/q/w/x/z leftovers → rules, others → B; served
    re-ranking resources (lexicon + `known_targets.txt`, unlike the new models' Google-only list)."""
    from khmer_translit import translit
    return [k for k, _route in translit.guess_many(words)]


def score() -> dict:
    """Score the valid runs and the served references on every panel; apply the D173 gate from integer counts.

    Returns:
        The report (also written to `REPORT`): per-run raw / reranked accuracy and correct counts, per-item
        outcomes, the served reference two ways (actual fallback routing "as if unlisted", and the old
        panel-assigned B/E), the gate and a joint word bootstrap.
    """
    from khmer_translit import translit
    from khmer_translit.lexicon import keys as _lexicon_keys
    from khmer_translit import benchmark
    google = frozenset(_lexicon_keys())
    P = benchmark.load(DEFAULT_CONFIG["eval"]["benchmark"])
    out: dict = {"panels": {n: len(p) for n, p in P.items()}, "models": {}, "items": {}}
    runs = valid_runs()
    for run in runs:
        res = {}
        for n, pairs in P.items():
            raw, rer = predict(D / run / "translit.npz", [a for a, _ in pairs], google)
            c_raw, c_rer = sum(r in b for r, (_a, b) in zip(raw, pairs)), sum(r in b for r, (_a, b) in zip(rer, pairs))
            res[n] = {"raw": round(c_raw / len(pairs), 4), "reranked": round(c_rer / len(pairs), 4),
                      "raw_correct": c_raw, "reranked_correct": c_rer, "n": len(pairs)}
            out["items"].setdefault(n, {})[run] = [int(r in b) for r, (_a, b) in zip(rer, pairs)]
        out["models"][run] = res
    served, panel_assigned = {}, {}
    known_served = translit._resources()[3]
    for n, pairs in P.items():
        words = [a for a, _ in pairs]
        got = served_as_if_unlisted(words)
        c = sum(g in b for g, (_a, b) in zip(got, pairs))
        served[n] = {"reranked": round(c / len(pairs), 4), "correct": c, "n": len(pairs)}
        out["items"].setdefault(n, {})["served_as_if_unlisted"] = [int(g in b) for g, (_a, b) in zip(got, pairs)]
        mdir = "model_e" if n in ("pick_test", "dev41", "list_test") else "model_b"
        raw, rer = predict(ROOT / "models" / mdir / "translit.npz", words, known_served)
        panel_assigned[n] = {"model": mdir, "raw": round(sum(r in b for r, (_a, b) in zip(raw, pairs)) / len(pairs), 4),
                             "reranked": round(sum(r in b for r, (_a, b) in zip(rer, pairs)) / len(pairs), 4)}
    out["served_reference"] = {"label": "served fallback routing, as if unlisted (served re-ranking resources)",
                               "panels": served}
    out["panel_assigned_reference"] = {"label": "panel-assigned B/E reference (not the served router)",
                                       "panels": panel_assigned}
    seeds = [s for s in (0, 1, 2) if f"aux_s{s}" in runs and f"base_s{s}" in runs]
    counts = {s: {p: (out["models"][f"aux_s{s}"][p]["reranked_correct"], out["models"][f"base_s{s}"][p]["reranked_correct"],
                      out["models"][f"aux_s{s}"][p]["n"]) for p in ("pick_test", "romanised_test", "names_test2")}
              for s in seeds}
    g = gate(counts)
    if seeds:
        dev = [Fraction(100) * (out["models"][f"aux_s{s}"]["dev41"]["reranked_correct"]
                                - out["models"][f"base_s{s}"]["dev41"]["reranked_correct"]) / len(P["dev41"]) for s in seeds]
        g["dev41_pp_per_seed_diagnostic"] = [float(d) for d in dev]
        items = out["items"]["pick_test"]
        n = len(P["pick_test"])
        diff = [Fraction(sum(items[f"aux_s{s}"][i] - items[f"base_s{s}"][i] for s in seeds), len(seeds)) for i in range(n)]
        g["english_joint_word_bootstrap_95ci_pp"] = list(benchmark.paired_bootstrap(diff))
        g["english_words_aux_better_worse_tied"] = [sum(d > 0 for d in diff), sum(d < 0 for d in diff), sum(d == 0 for d in diff)]
    out["gate"] = g
    REPORT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


def main() -> int:
    """CLI entry point (`data`, `train`, `export`, `score`); see the module docstring.

    Returns:
        0.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["data", "train", "export", "score"])
    ap.add_argument("--arm", choices=["base", "aux"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=Path, default=CONFIG, help="experiment config (default E01's)")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override one config value (recorded in the run); needs --out")
    ap.add_argument("--out", type=Path, default=None, help="run folder (default <runs_dir>/<arm>_s<seed>)")
    args = ap.parse_args()
    if args.mode == "data":
        m = build()
        print(json.dumps({k: m[k] for k in ("heldout_sizes", "val_sizes", "lengths")}, ensure_ascii=False))
        print(json.dumps({n: (s["rows_in"], s["kept"], s["dropped"]) for n, s in m["sources"].items()}))
        print(json.dumps({k: m["aux"][k] for k in ("rows_in", "kept", "dropped", "words_with_alternatives")}))
    elif args.mode == "train":
        from khmer_translit.experiment import Run, load_config
        cfg = load_config(args.config, CONFIG_KEYS, args.set)
        check_implemented(cfg)
        if args.set and args.out is None:
            raise SystemExit("--set changes the registered recipe: write the run elsewhere with --out")
        if args.seed not in cfg["train"]["seeds"]:
            raise SystemExit(f"seed {args.seed} is not one of the registered seeds {cfg['train']['seeds']}")
        out = args.out or ROOT / cfg["paths"]["runs_dir"] / f"{args.arm}_s{args.seed}"
        inputs = [args.config, D / "manifest.json", D / "spell_train.tsv", D / "aux_train.tsv"] + \
                 [D / f"val_{n}.tsv" for n in ("romanised", "names", "english")]
        group = cfg.get("experiment", args.config.parent.name) + ("-smoke" if args.set else "")   # tests never mix in
        run = Run(out, group, f"{args.arm}_s{args.seed}", cfg, inputs, job_type="smoke" if args.set else "train")
        r = train(args.arm, args.seed, out=out, cfg=cfg, run=run)
        print(json.dumps({k: v for k, v in r.items() if k != "log"}))
    elif args.mode == "export":
        print(export(D / f"{args.arm}_s{args.seed}"))
    else:
        from khmer_translit import benchmark
        from khmer_translit.experiment import Run
        cfg = DEFAULT_CONFIG
        version = cfg["eval"]["benchmark"]
        inputs = [benchmark.MANIFESTS / f"{version}.json"] + sorted((benchmark.FILES / version).glob("*.tsv")) + \
                 [D / run / "translit.npz" for run in valid_runs()]
        rec = Run(D / "score", "E01-unified", "score", cfg, inputs, job_type="eval")
        r = score()
        rec.finish({"gate": r["gate"], "reranked": {m: {p: v["reranked"] for p, v in res.items()}
                                                   for m, res in r["models"].items()},
                    "served_reference": {p: v["reranked"] for p, v in r["served_reference"]["panels"].items()}})
        print(json.dumps({"models": r["models"], "served_reference": r["served_reference"], "gate": r["gate"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
