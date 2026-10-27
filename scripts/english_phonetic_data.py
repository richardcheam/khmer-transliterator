#!/usr/bin/env python3
"""Experiment E02 data: audit the sources, build the English training pairs, the P2G pairs and the new benchmark.

Run from the repository root:

    .venv/bin/python scripts/english_phonetic_data.py build    # -> data/e02/*, experiments/E02-*/reports/*.json
    .venv/bin/python scripts/english_phonetic_data.py check    # re-hash data/e02 against its manifest

**What goes where** (the policy is `experiments/E02-english-phonetic/data_policy.json`, written here):

- **Benchmark pool:** single-word entries of the loanword list whose provenance is owner-verified (owner added,
  confirmed, chosen or corrected the spelling, or verified an LLM suggestion). Tier `attested`: reviewed by the
  owner for the speech front-end, **not** under the E02 phonetic policy, so no item is `independent_reviewed`.
  Lexical families are split by hash: 35% `dev` (checkpoint and arm selection), 65% `final` (opened once).
- **Exploratory proxy track (`final_proxy`):** a hash sample of 400 single-word list entries marked
  `llm_suggested_spotchecked` (LLM spelling, owner spot check of a sample). Never pooled with `final`.
- **Historical:** the English panels of benchmark v1 (pick test, dev-41, list test), E01's list validation words
  and the reserve's Latin words (answers never opened). Unchanged, scored only by the v1 scorer after selection;
  kept out of training and of the new splits.
- **Supervised training:** (1) web-mined bracket pairs (Khmer Wikipedia, FineWeb-2) whose Latin word has a CMUdict
  entry (English-readable) and is not classed Khmer-like, after the D135 noise filter, tier `proxy`; (2) the
  owner's 150 example picks (`synth/examples.tsv`), tier `attested`. Rows whose Latin word is related (shared
  suffix stem) to any held-out word, or whose Khmer is an accepted dev / final / proxy spelling, are removed.
- **P2G pretraining:** Google Khmer lexicon Khmer headwords → their pronunciation, reversed; any spelling that is
  an accepted dev / final / proxy form is removed (so every final item is target-unseen there); split 95 / 5 by
  connected spelling–pronunciation component.
- **Excluded:** LLM drafts and Gemini spellings (unverified), loanword-list drafts and `attested` conventional
  forms (often translations or French-style names), multi-word entries, the Chhunneng pairs and the rule-romanised
  Khmer pairs (another task; Chhunneng origin undocumented), Google-lexicon Latin headwords (phones only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from khmer_translit import en_g2p, latin_g2p  # noqa: E402
from khmer_translit import phonetic_source as ps  # noqa: E402
from khmer_translit import reference_data as rd  # noqa: E402
from khmer_translit.loanwords import LIST_TSV, read as read_list  # noqa: E402

T = ROOT / "data"
OUT = T / "e02"
EXP = ROOT / "experiments" / "E02-english-phonetic"
REPORTS = EXP / "reports"
SALT = "e02-english-phonetic"
OWNER = ("owner_added", "owner_confirmed", "owner_chosen", "owner_corrected", "llm_suggested_owner_verified")
FINAL_SHARE = 65          # % of benchmark-pool families in `final`; the rest are `dev`
PROXY_N = 400             # size of the exploratory proxy track
P2G_DEV_SHARE = 5         # % of P2G components in P2G dev
ACCENT = "en-US-cmudict"
TRANSLATION_SIM = 0.15    # both rule spellings this far from the reference: flag `possible_translation`


def sha(path: Path) -> str:
    """sha256 of a file."""
    return ps.sha256_file(path)


def _tsv(path: Path) -> list[list[str]]:
    """Rows of a TSV with a header (blank and # lines skipped)."""
    return [l.split("\t") for l in path.read_text(encoding="utf-8").splitlines()[1:] if l and not l.startswith("#")]


def keep_forms(forms: dict[str, tuple[int, float]]) -> list[str]:
    """The D135 noise filter for the mined Khmer forms of one Latin word (as `scripts/translit_extra.py`).

    Keep a form seen ≥ 2 times or once with similarity ≥ 0.7; drop a form that is a strict suffix of another kept
    form; keep at most 3, most frequent first.

    Examples:
        >>> keep_forms({"ដារា": (5, 0.8), "រា": (6, 0.6), "ដារ៉ា": (7, 0.9)})
        ['ដារ៉ា', 'ដារា']
    """
    ok = {k: v for k, v in forms.items() if v[0] >= 2 or v[1] >= 0.7}
    ok = {k: v for k, v in ok.items() if not any(o != k and o.endswith(k) for o in ok)}
    return sorted(ok, key=lambda k: (-ok[k][0], k))[:3]


def rule_similarity(word: str, khmer: str) -> float:
    """Best character similarity of the two rule spellings (CMUdict sound-out, letter sound-out) to `khmer`.

    A descriptive screen for translations, not a correctness measure.
    """
    cands = [en_g2p.transliterate(word) or "", latin_g2p.transliterate(word)]
    return max(latin_g2p.similarity(c, khmer) for c in cands if c) if any(cands) else 0.0


ANCHORED: set[str] = set()   # words whose owner spelling was picked among model E's candidates (synth/examples)


def reference_item(word: str, khmer: list[str], preferred: str, sources: list[str], tier: str, split: str,
                   family: str, review: dict) -> rd.ReferenceItem:
    """One `ReferenceItem` with its default reading and audit flags."""
    pron = ps.get_pronunciation(word)
    flags = ["candidate_anchored"] if word in ANCHORED else []
    if pron.status == "unavailable":
        flags.append("dictionary_miss")
    if pron.ambiguous:
        flags.append("pronunciation_ambiguous")
    if max(rule_similarity(word, k) for k in khmer) < TRANSLATION_SIM:
        flags.append("possible_translation")
    acc = sorted({ps.nfc(k) for k in khmer})
    return rd.ReferenceItem(f"{word}#default", word, word, family, "default", " ".join(pron.phones), ACCENT, acc,
                            ps.nfc(preferred), sources, tier, split, flags, review)


def read_reserve_latin() -> list[str]:
    """The reserve's Latin words only (the file holds no answers; nothing else of it is read)."""
    return [ps.normalize_word(w)[0] for w in json.loads((T / "e2" / "reserve.json").read_text(encoding="utf-8"))]


def build() -> dict:
    """Build every E02 data file and the audit summaries.

    Returns:
        The data manifest.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    audit = json.loads((ROOT / "datasets" / "build" / "latin_audit.json").read_text(encoding="utf-8"))
    prov = audit["provenance_by_word"]
    live = read_list()
    counts: dict[str, dict] = {}
    ANCHORED.update(ps.normalize_word(r[0])[0] for r in _tsv(T / "synth" / "examples.tsv"))

    # ---- historical sets (v1 English panels, reserve Latin): held out of everything new
    hist: dict[str, list[tuple[str, str]]] = {}
    for name in ("pick_test", "dev41", "list_test"):
        hist[name] = [(ps.normalize_word(r[0])[0], r[1]) for r in _tsv_noheader(T / "benchmark" / "v1" / f"{name}.tsv")]
    hist["reserve_latin"] = [(w, "") for w in read_reserve_latin()]
    list_val = json.loads((T / "list_split.json").read_text(encoding="utf-8"))["val"]
    hist["e01_list_val"] = [(w, live.get(w, "")) for w in list_val]
    hist_words = {w for rows in hist.values() for w, _ in rows}
    hist_stems = {s for w in hist_words for s in rd.stems(w)}

    # ---- loanword list: classify every entry
    list_rows = Counter()
    pool, proxy_pool = {}, {}
    for w, k in live.items():
        p = prov.get(w, "unknown")
        nw, st = ps.normalize_word(w)
        if st != "ok":
            list_rows[f"excluded_{st}:{p}"] += 1
        elif not ps.is_khmer_word(ps.nfc(k)):
            list_rows[f"excluded_non_khmer_target:{p}"] += 1
        elif rd.related(nw, hist_stems):
            list_rows[f"historical_related:{p}"] += 1
        elif p in OWNER:
            pool[nw] = (ps.nfc(k), p)
            list_rows[f"benchmark_pool:{p}"] += 1
        elif p == "llm_suggested_spotchecked":
            proxy_pool[nw] = (ps.nfc(k), p)
            list_rows[f"proxy_candidates:{p}"] += 1
        else:
            list_rows[f"excluded_policy:{p}"] += 1
    counts["loanword_list"] = dict(sorted(list_rows.items()))

    # ---- benchmark splits by lexical family (pool and proxy grouped together, so no family is in both)
    fam = rd.group_families(list(pool) + list(proxy_pool))
    items: list[rd.ReferenceItem] = []
    review = {"reviewer": "project owner", "policy": "speech front-end loanword list (D110-D118), not the E02 "
              "phonetic policy", "evidence": "datasets/build/latin_audit.json provenance_by_word"}
    pool_fams = {fam[w] for w in pool}
    for w, (k, p) in sorted(pool.items()):
        split = "final" if rd.split_bucket(fam[w], SALT) < FINAL_SHARE else "dev"
        items.append(reference_item(w, [k], k, [f"loanword_list:{p}"], "attested", split, fam[w], review))
    proxy_cands = sorted((w for w in proxy_pool if fam[w] not in pool_fams),
                         key=lambda w: hashlib.sha256(f"{SALT}:proxy:{w}".encode()).hexdigest())
    chosen_fams: set[str] = set()
    for w in proxy_cands:
        if len([i for i in items if i.split == "final_proxy"]) >= PROXY_N:
            break
        if fam[w] in chosen_fams:
            continue
        chosen_fams.add(fam[w])
        for v in sorted(x for x in proxy_cands if fam[x] == fam[w]):
            k, p = proxy_pool[v]
            items.append(reference_item(v, [k], k, [f"loanword_list:{p}"], "proxy", "final_proxy", fam[v],
                                        {"reviewer": "none (LLM spelling; owner spot check of a sample, D111)"}))
    rd.validate_items(items)
    held_words = {it.word for it in items} | hist_words
    held_stems = {s for w in held_words for s in rd.stems(w)}
    held_targets = {a for it in items if it.split in ("dev", "final", "final_proxy") for a in it.accepted}

    # ---- supervised training rows
    cmu, cmu_sha = ps.cmudict()
    forms: dict[str, dict[str, list]] = defaultdict(dict)
    mined_in = Counter()
    for name in ("mined_kmwiki.tsv", "mined_fineweb.tsv"):
        for r in _tsv(T / name):
            lat, km, cls, n, sim = r[:5]
            mined_in[name] += 1
            km = ps.nfc(km)
            o = forms[lat].setdefault(km, [0, 0.0, set()])
            o[0] += int(n)
            o[1] = max(o[1], float(sim))
            o[2].add(cls)
    train, drop = [], Counter()
    for lat in sorted(forms):
        w, st = ps.normalize_word(lat)
        kept = keep_forms({k: (v[0], v[1]) for k, v in forms[lat].items()})
        for km in sorted(forms[lat]):
            cls = forms[lat][km][2]
            if st != "ok" or w != lat:
                drop["mined:not_plain_a_z"] += 1
            elif "khmer_like" in cls:
                drop["mined:khmer_like_romanisation"] += 1
            elif not ps.is_khmer_word(km) or len(km) > ps.MAX_TGT:
                drop["mined:bad_khmer_or_length"] += 1
            elif km not in kept:
                drop["mined:noise_filter"] += 1
            elif w not in cmu:
                drop["mined:no_english_reading"] += 1
            elif rd.related(w, held_stems):
                drop["mined:heldout_related"] += 1
            elif km in held_targets:
                drop["mined:heldout_target"] += 1
            else:
                train.append((w, km, "mined_web", "proxy"))
    for r in _tsv(T / "synth" / "examples.tsv"):
        w, st = ps.normalize_word(r[0])
        km = ps.nfc(r[1])
        if st != "ok" or not ps.is_khmer_word(km):
            drop["owner_examples:bad_row"] += 1
        elif rd.related(w, held_stems):
            drop["owner_examples:heldout_related"] += 1
        elif km in held_targets:
            drop["owner_examples:heldout_target"] += 1
        else:
            train.append((w, km, "owner_examples", "attested"))
    train = sorted(set(train))
    counts["mined_rows_in"] = dict(mined_in)
    counts["supervised_dropped"] = dict(sorted(drop.items()))
    counts["supervised_kept"] = dict(Counter(s for _w, _k, s, _t in train))

    # training-validation split: 3% of training families, for the loss used in checkpoint tie-breaks only
    tfam = rd.group_families(w for w, *_ in train)
    rows_out = []
    for w, km, src, tier in train:
        pron = ps.get_pronunciation(w)
        part = "train_val" if rd.split_bucket(tfam[w], SALT + ":tv") < 3 else "train"
        rows_out.append((w, km, src, tier, " ".join(pron.phones), part))

    # ---- P2G pairs from the Google lexicon (Khmer headwords)
    lex = ROOT / "data" / "lexicon" / "km_lexicon.tsv"
    p2g, p2g_drop = set(), Counter()
    for line in lex.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "\t" not in line:
            continue
        head, phones = line.split("\t")[:2]
        km = ps.nfc(head.strip())
        if not ps.is_khmer_word(km):
            p2g_drop["not_khmer_headword"] += 1
        elif not phones.strip():
            p2g_drop["empty_pronunciation"] += 1
        elif len(km) > ps.MAX_TGT or len(phones.split()) + 1 > ps.MAX_SRC:
            p2g_drop["too_long"] += 1
        elif km in held_targets:
            p2g_drop["heldout_target"] += 1
        else:
            p2g.add((km, " ".join(phones.split())))
    p2g = sorted(p2g)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for km, ph in p2g:
        a, b = find("s:" + km), find("p:" + ph)
        if a != b:
            parent[max(a, b)] = min(a, b)
    p2g_rows = [(ph, km, "p2g_dev" if rd.split_bucket(find("s:" + km), SALT + ":p2g") < P2G_DEV_SHARE else "p2g_train")
                for km, ph in p2g]
    counts["p2g"] = {"kept": len(p2g_rows), "dropped": dict(p2g_drop),
                     "homophone_groups": sum(1 for c in Counter(ph for ph, _k, _s in p2g_rows).values() if c > 1),
                     "dev": sum(s == "p2g_dev" for *_x, s in p2g_rows)}

    # ---- vocabularies (from training material only, never from references)
    km_phones = sorted({p for ph, _k, _s in p2g_rows for p in ph.split()})
    vocab = ps.SourceVocab(km_phones)
    tgt_chars = sorted({c for _w, km, *_r in rows_out for c in km} | {c for _p, km, _s in p2g_rows for c in km})
    ref_chars = {c for it in items for a in it.accepted for c in a}
    lengths = {"max_letters": max(len(w) for w, *_ in rows_out),
               "max_en_source": max(len(ps.encode_source(w, ps.get_pronunciation(w), "phones")) for w, *_ in rows_out),
               "max_p2g_source": max(len(ph.split()) + 1 for ph, _k, _s in p2g_rows),
               "max_target": max(max(len(k) for _w, k, *_r in rows_out), max(len(k) for _p, k, _s in p2g_rows)),
               "registered_max_src": ps.MAX_SRC, "registered_max_tgt": ps.MAX_TGT}

    # ---- write
    (OUT / "train.tsv").write_text("word\tkhmer\tsource\ttier\tphones\tpart\n" + "".join(
        "\t".join(r) + "\n" for r in rows_out), encoding="utf-8")
    (OUT / "p2g.tsv").write_text("phones\tkhmer\tpart\n" + "".join("\t".join(r) + "\n" for r in p2g_rows),
                                 encoding="utf-8")
    rd.save_reference_items(items, OUT / "references.jsonl")
    rd.save_reference_items([i for i in items if i.split == "dev"], OUT / "dev_references.jsonl")   # trainer reads only this
    (OUT / "vocab.json").write_text(json.dumps({"encoder_version": ps.ENCODER_VERSION, "src_tokens": vocab.tokens,
                                                "km_phones": km_phones, "tgt_chars": "".join(tgt_chars)},
                                               ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    sep = rd.audit_splits(items, [r[0] for r in rows_out], [r[1] for r in rows_out])
    strata = strata_report(items)
    inputs = [LIST_TSV, ROOT / "datasets/build/latin_audit.json", T / "mined_kmwiki.tsv", T / "mined_fineweb.tsv",
              T / "synth/examples.tsv", lex, ps.CMUDICT, T / "list_split.json", T / "e2/reserve.json"] + [
              T / "benchmark/v1" / f"{n}.tsv" for n in ("pick_test", "dev41", "list_test")]
    manifest = {
        "experiment": "E02-english-phonetic", "encoder_version": ps.ENCODER_VERSION,
        "source_vocab_sha256": vocab.sha256(), "n_src_tokens": len(vocab.tokens), "n_tgt_chars": len(tgt_chars),
        "reference_chars_outside_target_vocab": sorted(ref_chars - set(tgt_chars)),
        "counts": counts, "separation": sep, "strata": strata, "lengths": lengths,
        "train_rows": {p: sum(r[5] == p for r in rows_out) for p in ("train", "train_val")},
        "train_dictionary_hits": sum(bool(r[4]) for r in rows_out),
        "input_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in inputs},
        "output_sha256": {f: sha(OUT / f) for f in ("train.tsv", "p2g.tsv", "references.jsonl", "dev_references.jsonl",
                                                       "vocab.json")},
        "cmudict_sha256": cmu_sha,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (REPORTS / "data_build.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                                             encoding="utf-8")
    write_sample(rows_out, items)
    return manifest


def _tsv_noheader(path: Path) -> list[list[str]]:
    """Rows of a header-less TSV (the frozen v1 panels)."""
    return [l.split("\t") for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def strata_report(items: list[rd.ReferenceItem]) -> dict:
    """Per split: item counts by flag, dictionary hit, length and family size.

    Args:
        items: Every reference item.

    Returns:
        `{split: {...counts}}`.
    """
    out = {}
    for sp in rd.SPLITS:
        sub = [i for i in items if i.split == sp]
        if not sub:
            continue
        fam_size = Counter(i.family for i in sub)
        out[sp] = {"items": len(sub), "families": len(fam_size),
                   "multi_item_families": sum(1 for v in fam_size.values() if v > 1),
                   "flags": dict(Counter(f for i in sub for f in i.flags)),
                   "primary_eligible": sum(not ({"pronunciation_ambiguous", "possible_translation"} & set(i.flags))
                                           for i in sub),
                   "short_le4": sum(len(i.word) <= 4 for i in sub), "long_ge9": sum(len(i.word) >= 9 for i in sub)}
    return out


def write_sample(rows: list[tuple], items: list[rd.ReferenceItem]) -> None:
    """A fixed-seed audit sample (training rows by source, dev items), drawn before any new model exists.

    Writes `reports/audit_sample.tsv`: 60 training rows (stratified by source) and 40 dev items, with the rule
    similarity as a descriptive column. No final item is sampled.
    """
    rng = random.Random(20261027)
    by_src = defaultdict(list)
    for r in rows:
        by_src[r[2]].append(r)
    lines = ["kind\tword\tkhmer\tsource\ttier\trule_similarity\tflags"]
    for src in sorted(by_src):
        for r in rng.sample(by_src[src], min(30, len(by_src[src]))):
            lines.append(f"train\t{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{rule_similarity(r[0], r[1]):.2f}\t")
    dev = [i for i in items if i.split == "dev"]
    for i in rng.sample(dev, min(40, len(dev))):
        lines.append(f"dev\t{i.word}\t{'|'.join(i.accepted)}\t{i.sources[0]}\t{i.tier}\t"
                     f"{max(rule_similarity(i.word, a) for a in i.accepted):.2f}\t{','.join(i.flags)}")
    (REPORTS / "audit_sample.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check() -> int:
    """Re-hash `data/e02` against its manifest.

    Returns:
        0 if every output matches, 1 otherwise.
    """
    m = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    bad = [f for f, h in m["output_sha256"].items() if sha(OUT / f) != h]
    print(json.dumps({"ok": not bad, "mismatched": bad}))
    return 1 if bad else 0


def main() -> int:
    """CLI entry point.

    Returns:
        Exit status.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["build", "check"])
    args = ap.parse_args()
    if args.mode == "check":
        return check()
    m = build()
    print(json.dumps({k: m[k] for k in ("counts", "separation", "strata", "lengths", "train_rows")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
