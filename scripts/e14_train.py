#!/usr/bin/env python3
"""E14 (romanised Khmer): build the training data, train, export and score one model against served model B.

Experiment `experiments/E14-romanised-typing/` (PLAN.md, REGISTRATION.md, config.yaml). Run from the repository root:

    .venv/bin/python scripts/e14_train.py data                 # -> models/e14/ (training files + manifest.json)
    .venv/bin/python scripts/e14_train.py train --seed 0       # -> models/e14/aux_s0/
    .venv/bin/python scripts/e14_train.py export --seed 0
    .venv/bin/python scripts/e14_train.py score                # -> experiments/E14-romanised-typing/results.json
    # smoke run (never into a registered folder):
    .venv/bin/python scripts/e14_train.py train --seed 0 --set train.epochs=1 --out /tmp/smoke

The recipe is E01's `aux` arm (`scripts/translit_unified.py`: same model, side task, optimiser and schedule), with
three changes fixed in the registration:

- **More spelling data:** the Chhunneng pairs (train split, dictionary and informal rows), the owner's own typing
  (train half of the frozen 320-word list) and the Gemini spellings the owner checked ("LLM-suggested,
  owner-verified"). The two owner sources are repeated like the loans (`LOAN_REPEAT`).
- **One more held-out set,** `e14_test`: the 160 owner-typed test words. A row is dropped if its Khmer target is a
  test word, or its Latin side equals a typed test spelling (spaces, hyphens and apostrophes removed).
- **Romanised validation** is the Chhunneng val split (real typing) instead of rule-romanised dictionary words.

Owner answers with spaces (*puk ke*) give the joined row (*pukke* → ពួកគេ) and, when the Khmer word splits into as
many known parts as there are typed words, one row per part (*puk* → ពួក, *ke* → គេ).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import translit_unified as tu  # noqa: E402

D = ROOT / "models" / "e14"
EXPERIMENT = ROOT / "experiments" / "E14-romanised-typing"
CONFIG = EXPERIMENT / "config.yaml"
REPORT = EXPERIMENT / "results.json"
TYPING = ROOT / "data" / "e14" / "typing"
GEMINI = ROOT / "data" / "e14" / "gemini"
TYPIST = "Richard"
NAME = "e14"                 # candidate prefix in the report (E15 reuses this script with its own settings)
DROP: set[str] = set()       # training sources left out (E15: {"roman"})
REFS = {"e01_aux": ROOT / "models" / "unified"}   # owner-test-only references, folders with aux_s<seed>/translit.npz
EXTRA: dict = {}             # extra training sources: name -> (file, loader returning [(latin, khmer)]) (E16)
REPEATED = {"list_train", "owner_examples", "llm_drafts", "owner_typing", "owner_checked_llm"}
CONFIG_KEYS = {**tu.CONFIG_KEYS, "paths": {"runs_dir": "models/e14"}}


def clean(spelling: str) -> list[str]:
    """The typed words of one answer, lower-case a-z only (hyphens and apostrophes removed).

    >>> clean("Puk ke"), clean("sa-at"), clean("k'ot")
    (['puk', 'ke'], ['saat'], ['kot'])
    """
    return [re.sub(r"[^a-z]", "", w) for w in spelling.lower().split() if re.sub(r"[^a-z]", "", w)]


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def owner_answers() -> dict[str, list[tuple[str, list[str]]]]:
    """The owner's answers by source: `{source: [(khmer, [spelling, ...]), ...]}`; test words under `"test"`."""
    typed = _read(TYPING / f"typed_{TYPIST}.tsv")
    checked = _read(GEMINI / f"verified_{TYPIST}.tsv")
    var = lambda r: [v.strip() for v in r["typed"].split(",") if v.strip()]
    return {"test": [(r["khmer"], var(r)) for r in typed if r["split"] == "test" and r["skipped"] != "1"],
            "owner_typing": [(r["khmer"], var(r)) for r in typed if r["split"] == "train" and r["skipped"] != "1"],
            "owner_checked_llm": [(r["khmer"], var(r)) for r in checked if r["skipped"] != "1" and var(r)]}


def owner_rows(answers: list[tuple[str, list[str]]], vocab: set[str]) -> list[tuple[str, str]]:
    """Training rows from owner answers: the joined spelling, plus per-part rows when the parts line up.

    Args:
        answers: `[(khmer, [spelling, ...])]`.
        vocab: Khmer words used to split compounds (`lexicon.segment`, greedy longest match).

    Returns:
        `[(latin, khmer)]`, e.g. `("pukke", "ពួកគេ"), ("puk", "ពួក"), ("ke", "គេ")` for *puk ke*.
    """
    from khmer_translit.lexicon import segment
    out = []
    for km, spellings in answers:
        seg = segment(km, {k: 1 for k in vocab if k != km})
        parts = [p for p, _ in seg] if len(seg) > 1 and all(ok for _, ok in seg) else []
        for s in spellings:
            words = clean(s)
            if not words:
                continue
            out.append(("".join(words), km))
            if len(words) > 1 and len(words) == len(parts):
                out += list(zip(words, parts))
    return out


def test_panel() -> list[tuple[str, list[str]]]:
    """The owner-typed test: `[(khmer, [typed spelling, ...])]`, 160 words, in word-list order."""
    return owner_answers()["test"]


def heldout() -> dict[str, dict]:
    """E01's held-out sets plus `e14_test`; validation pruned of every evaluation set; Chhunneng val as romanised val."""
    held = tu.heldout()
    held["e14_test"] = {"role": "eval", "pairs": [("".join(clean(s)), km) for km, sp in test_panel() for s in sp]}
    rows = [l.split("\t") for l in (ROOT / "data/pairs.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    held["val_romanised"] = {"role": "val", "pairs": [(r[0], r[1]) for r in rows if r[3] == "val"]}
    ev = tu.match_sets({n: h for n, h in held.items() if h["role"] == "eval"})
    for v in tu.VAL_SETS:
        held[v] = {"role": "val", "pairs": [(a, b) for a, b in held[v]["pairs"] if not any(tu.row_hits(a, b, ev))]}
    return held


def build() -> dict:
    """Write `spell_train.tsv`, `aux_train.tsv`, `val_*.tsv` and `manifest.json` into `D`.

    Returns:
        The manifest (counts per source, drops per reason and per held-out set, vocabularies, hashes).
    """
    T = ROOT / "data"
    held = heldout()
    per_set = tu.match_sets(held)
    live = tu.live_list()
    split = json.loads((T / "list_split.json").read_text(encoding="utf-8"))
    pairs = [l.split("\t") for l in (T / "pairs.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    ans = owner_answers()
    vocab = {km for src in ans.values() for km, _ in src} | {r[1] for r in pairs if r[3] == "train"}
    sources = {
        "roman": [(a, b) for a, b, _s, sp in (l.split("\t") for l in (T / "roman/pairs.tsv").read_text(
            encoding="utf-8").splitlines()[1:]) if sp == "train"],
        "mined": tu._tsv(T / "extra_train.tsv"),
        "chhunneng": [(r[0], r[1]) for r in pairs if r[3] == "train"],
        "list_train": [(w, live[w]) for w in split["train"] if w in live],
        "owner_examples": tu._tsv(T / "synth/examples.tsv"),
        "llm_drafts": tu._tsv(T / "llm_drafts.tsv"),
        "owner_typing": owner_rows(ans["owner_typing"], vocab),
        "owner_checked_llm": owner_rows(ans["owner_checked_llm"], vocab),
    }
    sources.update({k: load() for k, (_f, load) in EXTRA.items()})
    sources = {k: v for k, v in sources.items() if k not in DROP}
    kept, excl = {}, {}
    for name, rows in sources.items():
        good, by, by_set, seen = [], {"latin_related": 0, "khmer_target": 0, "bad_chars_or_length": 0,
                                      "duplicate": 0}, {n: 0 for n in held}, set()
        for a, b in rows:
            a = a.lower()
            lat_hit, km_hit = tu.row_hits(a, b, per_set)
            for n in set(lat_hit) | set(km_hit):
                by_set[n] += 1
            if not re.fullmatch("[a-z]+", a) or len(a) > tu.MAX_IN or not b or len(b) > 40 \
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
    aux_rows, seen, aux_drop = [], set(), {"empty": 0, "not_az_or_length": 0, "latin_related": 0, "duplicate": 0}
    for w, toks in tu.google_latin():
        if not toks:
            aux_drop["empty"] += 1
        elif not re.fullmatch("[a-z]+", w) or len(w) > tu.MAX_IN or len(toks) > tu.AUX_MAX:
            aux_drop["not_az_or_length"] += 1
        elif tu.latin_hits(w, per_set):
            aux_drop["latin_related"] += 1
        elif (w, tuple(toks)) in seen:
            aux_drop["duplicate"] += 1
        else:
            seen.add((w, tuple(toks)))
            aux_rows.append((w, " ".join(toks)))
    spell = [(a, b, n) for n, rows in kept.items() for a, b in rows]
    vals = {"romanised": held["val_romanised"]["pairs"], "names": held["val_names"]["pairs"],
            "english": held["val_english"]["pairs"]}
    src_chars = sorted({c for a, _b, _n in spell for c in a} | {c for w, _p in aux_rows for c in w}
                       | {c for p in vals.values() for a, _ in p for c in a.lower()})
    tgt_chars = sorted({c for _a, b, _n in spell for c in b})
    D.mkdir(parents=True, exist_ok=True)
    (D / "spell_train.tsv").write_text("latin\tkhmer\tsource\n" + "".join(f"{a}\t{b}\t{n}\n" for a, b, n in spell),
                                       encoding="utf-8")
    (D / "aux_train.tsv").write_text("latin\tphones\n" + "".join(f"{w}\t{p}\n" for w, p in aux_rows), encoding="utf-8")
    for n, p in vals.items():
        (D / f"val_{n}.tsv").write_text("latin\tkhmer\n" + "".join(f"{a}\t{b}\n" for a, b in p), encoding="utf-8")
    test = test_panel()
    inputs = [tu.LEX, T / "roman/pairs.tsv", T / "extra_train.tsv", T / "pairs.tsv", T / "llm_drafts.tsv",
              T / "synth/examples.tsv", T / "list_split.json", TYPING / "words.json"]
    manifest = {
        "rules": {"e14_test": "drop a row if its Khmer target is an owner-typed test word, or its Latin side equals a "
                              "typed test spelling (a-z only, spaces removed); applied to every source and the side task",
                  "owner_rows": "each spelling variant: joined row; plus one row per typed word when the Khmer word "
                                "splits into as many known parts (lexicon.segment over the owner and Chhunneng words)",
                  "repeated": f"{sorted(REPEATED)} x{tu.LOAN_REPEAT}",
                  "val_romanised": "Chhunneng val split, pruned of every evaluation set (E01 rule)",
                  "other": "E01 rules unchanged (tu.row_hits, tu.heldout)"},
        "heldout_sizes": {n: len(h["pairs"]) for n, h in held.items()},
        "test_words": len(test), "test_items": sum(len(s) for _, s in test),
        "sources": excl, "aux": {"kept": len(aux_rows), "dropped": aux_drop},
        "val_sizes": {n: len(p) for n, p in vals.items()},
        "spell_pairs_per_epoch": sum(len(r) * (tu.LOAN_REPEAT if n in REPEATED else 1) for n, r in kept.items()),
        "vocab": {"src": "".join(src_chars), "tgt": "".join(tgt_chars),
                  "aux": sorted({t for _w, p in aux_rows for t in p.split()})},
        "input_sha256": {str(p.relative_to(ROOT)): tu.sha(p) for p in inputs},
        "owner_files_sha256": {f.name: tu.sha(f) for f in (TYPING / f"typed_{TYPIST}.tsv",
                                                            GEMINI / f"verified_{TYPIST}.tsv")},
        **({"extra_inputs_sha256": {str(f.relative_to(ROOT)): tu.sha(f) for f, _l in EXTRA.values()}} if EXTRA else {}),
        "output_sha256": {f: tu.sha(D / f) for f in ("spell_train.tsv", "aux_train.tsv", "val_romanised.tsv",
                                                    "val_names.tsv", "val_english.tsv")},
    }
    (D / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest


def load_data():
    """`(spell pairs with repeats, aux rows, validation sets, manifest)` as written by `build` (replaces E01's)."""
    manifest = json.loads((D / "manifest.json").read_text(encoding="utf-8"))
    rows = [l.split("\t") for l in (D / "spell_train.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    spell = [(a, b) for a, b, n in rows if n not in REPEATED] + \
            [(a, b) for a, b, n in rows if n in REPEATED] * tu.LOAN_REPEAT
    aux = [(w, p.split()) for w, p in (l.split("\t") for l in (D / "aux_train.tsv").read_text(encoding="utf-8").splitlines()[1:])]
    vals = {n: tu._tsv(D / f"val_{n}.tsv") for n in ("romanised", "names", "english")}
    return spell, aux, vals, manifest


# ---------------------------------------------------------------- scoring
def spell_items(npz: Path, items: list[str], known: frozenset[str]) -> list[str]:
    """Each typed item → Khmer, word by word (as the speech front end splits on spaces), re-ranked with `known`."""
    from khmer_translit import translit
    model = translit.load(npz)
    toks = sorted({t for s in items for t in clean(s)})
    best = dict(zip(toks, (translit.pick(nb, known) for nb in model.nbest_batch(toks, 4))))
    return ["".join(best[t] for t in clean(s)) for s in items]


def word_scores(outputs: list[str], test: list[tuple[str, list[str]]]) -> list[Fraction]:
    """Per test word, the share of its typed spellings spelled exactly right (outputs in item order)."""
    out, i = [], 0
    for km, sp in test:
        out.append(Fraction(sum(o == km for o in outputs[i:i + len(sp)]), len(sp)))
        i += len(sp)
    return out


def gate(e14: dict[int, list[Fraction]], b: list[Fraction], guards: dict[int, dict[str, tuple[int, int, int]]]) -> dict:
    """The registered E14 rule (REGISTRATION.md), from exact per-word scores and integer guard counts.

    Args:
        e14: `{seed: per-word owner-test scores}` for seeds 0, 1, 2.
        b: Served model B's per-word owner-test scores.
        guards: `{seed: {panel: (e14_correct, b_correct, n)}}` for "romanised_test" and "names_test2".

    Returns:
        Per-seed and mean differences in points and `passes`.

    Examples:
        >>> one, half = [Fraction(1)] * 2, [Fraction(1), Fraction(1, 2)]
        >>> g = gate({s: one for s in (0, 1, 2)}, half, {s: {"romanised_test": (95, 100, 100),
        ...           "names_test2": (50, 50, 100)} for s in (0, 1, 2)})
        >>> g["primary_mean_pp"], g["romanised_mean_pp"], g["passes"]
        (25.0, -5.0, True)
    """
    seeds = sorted(e14)
    n = len(b)
    prim = [Fraction(100) * (sum(e14[s]) - sum(b)) / n for s in seeds]
    gp = {p: [Fraction(100) * (guards[s][p][0] - guards[s][p][1]) / guards[s][p][2] for s in seeds]
          for p in ("romanised_test", "names_test2")}
    mean = lambda v: sum(v) / len(v) if v else Fraction(0)
    ok = (seeds == [0, 1, 2] and mean(prim) >= -3 and mean(gp["romanised_test"]) >= -5
          and mean(gp["names_test2"]) >= -3)
    return {"seeds": seeds, "primary_pp_per_seed": [float(x) for x in prim], "primary_mean_pp": float(mean(prim)),
            "romanised_pp_per_seed": [float(x) for x in gp["romanised_test"]],
            "romanised_mean_pp": float(mean(gp["romanised_test"])),
            "names_pp_per_seed": [float(x) for x in gp["names_test2"]], "names_mean_pp": float(mean(gp["names_test2"])),
            "passes": bool(ok)}


def runs() -> list[int]:
    """Seeds whose run is exported and trained on the current manifest."""
    cur = tu.sha(D / "manifest.json")
    return [s for s in (0, 1, 2) if (D / f"aux_s{s}" / "translit.npz").exists()
            and json.loads((D / f"aux_s{s}" / "train_log.json").read_text(encoding="utf-8")).get("manifest_sha256") == cur]


def score() -> dict:
    """Score the E14 runs, served B and the clean E01 aux runs on the owner test and the benchmark v1 panels."""
    from khmer_translit import benchmark, translit
    from khmer_translit.lexicon import keys
    google = frozenset(keys())
    known_served = translit._resources()[3]
    test = test_panel()
    items = [s for _, sp in test for s in sp]
    P = benchmark.load("v1")
    models = {f"{NAME}_aux_s{s}": (D / f"aux_s{s}" / "translit.npz", google) for s in runs()}
    models["served_b"] = (ROOT / "models/model_b/translit.npz", known_served)
    for ref, folder in REFS.items():
        for s in (0, 1, 2):
            models[f"{ref}_s{s}"] = (folder / f"aux_s{s}" / "translit.npz", google)
    out: dict = {"owner_test": {"words": len(test), "items": len(items)}, "models": {}, "per_word": {}}
    for name, (npz, known) in models.items():
        ws = word_scores(spell_items(npz, items, known), test)
        res = {"owner_test": round(float(100 * sum(ws) / len(ws)), 2)}
        out["per_word"][name] = [str(x) for x in ws]
        if not any(name.startswith(ref) for ref in REFS):
            for p in ("romanised_test", "names_test2", "pick_test", "dev41", "list_test"):
                _raw, rer = tu.predict(npz, [a for a, _ in P[p]], known)
                c = sum(r in f for r, (_a, f) in zip(rer, P[p]))
                res[p] = {"correct": c, "n": len(P[p]), "pct": round(100 * c / len(P[p]), 2)}
        out["models"][name] = res
    sys_out = [k for k, _r in translit.guess_many(sorted({t for s in items for t in clean(s)}))]
    lookup = dict(zip(sorted({t for s in items for t in clean(s)}), sys_out))
    ws = word_scores(["".join(lookup[t] for t in clean(s)) for s in items], test)
    out["models"]["served_system_router"] = {"owner_test": round(float(100 * sum(ws) / len(ws)), 2)}
    b_train = {l.split("\t")[1] for l in (ROOT / "data/pairs.tsv").read_text(encoding="utf-8").splitlines()[1:]
               if l.split("\t")[3] == "train"}
    out["served_b_saw_test_words"] = sum(km in b_train for km, _ in test)
    unseen = [km not in b_train for km, _ in test]           # owner-test words B was never trained on
    out["b_unseen_owner_test_pct"] = {
        name: round(float(100 * sum(Fraction(x) for x, u in zip(ws, unseen) if u) / sum(unseen)), 2)
        for name, ws in out["per_word"].items()}
    seeds = runs()
    if seeds:
        b = [Fraction(x) for x in out["per_word"]["served_b"]]
        e = {s: [Fraction(x) for x in out["per_word"][f"{NAME}_aux_s{s}"]] for s in seeds}
        guards = {s: {p: (out["models"][f"{NAME}_aux_s{s}"][p]["correct"], out["models"]["served_b"][p]["correct"],
                          len(P[p])) for p in ("romanised_test", "names_test2")} for s in seeds}
        g = gate(e, b, guards)
        diff = [sum(e[s][i] for s in seeds) / len(seeds) - b[i] for i in range(len(b))]
        g["primary_word_bootstrap_95ci_pp"] = list(benchmark.paired_bootstrap(diff))
        g[f"words_{NAME}_better_worse_tied"] = [sum(d > 0 for d in diff), sum(d < 0 for d in diff),
                                                sum(d == 0 for d in diff)]
        for ref in REFS:
            rw = [[Fraction(x) for x in out["per_word"][f"{ref}_s{s}"]] for s in (0, 1, 2)]
            g[f"diagnostic_{NAME}_minus_{ref}_pp"] = float(Fraction(100) * (
                sum(sum(v) for v in e.values()) / len(seeds) - sum(map(sum, rw)) / 3) / len(b))
            if ref != "e01_aux":
                rd = [sum(e[s][i] for s in seeds) / len(seeds) - sum(r[i] for r in rw) / 3 for i in range(len(b))]
                g[f"diagnostic_{NAME}_minus_{ref}_word_bootstrap_95ci_pp"] = list(benchmark.paired_bootstrap(rd))
        out["gate"] = g
    REPORT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


def main(doc: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=(doc or __doc__).splitlines()[0])
    ap.add_argument("mode", choices=["data", "train", "export", "score"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    tu.D, tu.load_data = D, load_data          # E01's training loop reads its data and run folder through these
    if a.mode == "data":
        m = build()
        print(json.dumps({k: m[k] for k in ("heldout_sizes", "test_words", "test_items", "val_sizes",
                                            "spell_pairs_per_epoch")}, ensure_ascii=False))
        print(json.dumps({n: (s["rows_in"], s["kept"], s["dropped"]) for n, s in m["sources"].items()}))
    elif a.mode == "train":
        from khmer_translit.experiment import Run, load_config
        cfg = load_config(a.config, CONFIG_KEYS, a.set)
        if a.set and a.out is None:
            raise SystemExit("--set changes the registered recipe: write the run elsewhere with --out")
        if a.seed not in cfg["train"]["seeds"]:
            raise SystemExit(f"seed {a.seed} is not registered")
        out = a.out or ROOT / cfg["paths"]["runs_dir"] / f"aux_s{a.seed}"
        inputs = [a.config, D / "manifest.json", D / "spell_train.tsv", D / "aux_train.tsv"] + \
                 [D / f"val_{n}.tsv" for n in ("romanised", "names", "english")]
        run = Run(out, EXPERIMENT.name + ("-smoke" if a.set else ""), f"aux_s{a.seed}", cfg, inputs,
                  job_type="smoke" if a.set else "train")
        r = tu.train("aux", a.seed, out=out, cfg=cfg, run=run)
        print(json.dumps({k: v for k, v in r.items() if k != "log"}))
    elif a.mode == "export":
        print(tu.export(D / f"aux_s{a.seed}"))
    else:
        r = score()
        print(json.dumps({"models": r["models"], "gate": r.get("gate"), "b_saw": r["served_b_saw_test_words"]},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
