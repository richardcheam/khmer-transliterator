#!/usr/bin/env python3
"""E16 data (romanised Khmer from Gemini in bulk): word list, fresh owner test, prompts and the automatic filter.

    .venv/bin/python scripts/e16.py freq                      # count Khmer words in the local FineWeb-2 shard
    .venv/bin/python scripts/e16.py select                    # once: freeze the fresh test words (owner types them)
    .venv/bin/python scripts/e16.py prompts                   # Gemini prompt files in data/e16/gemini/
    .venv/bin/python scripts/e16.py type --typist "Richard"   # typing page for the test words, http://127.0.0.1:8769
    .venv/bin/python scripts/e16.py fetch                     # send the prompts to the Gemini API (free key)
    .venv/bin/python scripts/e16.py filter                    # read the replies, apply the automatic filter

`fetch` sends each `prompt_<nn>.md` to the Gemini API (key in GEMINI_API_KEY, owner's free tier) and saves the answer as
`reply_<nn>.txt` next to it; the owner may also paste replies by hand. No reply is
checked by hand: `filter` keeps or drops every suggested spelling by fixed rules (see `filter_rows`). Test words are
chosen before any prompt is written and never appear in a prompt.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14  # noqa: E402  (typing page, Store, Chhunneng word pool)

from khmer_translit.lexicon import keys  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "e16"
GEMINI = DATA / "gemini"
EXP = ROOT / "experiments" / "E16-gemini-bulk"
SHARD = ROOT / "data" / "src" / "fineweb2_km" / "train_000_00000.parquet"
SEED = "e16-test-v1"
N_DOCS = 200_000          # documents read for the frequency count
N_PROMPT = 10_000         # words sent to Gemini
N_TEST = 150              # fresh owner test words
TEST_FROM_TOP = 3_000     # test words are drawn from the most common words
BATCH = 250               # words per prompt
MIN_ALONE = 5             # a frequency-list word must occur at least this often as a stand-alone run
KHMER_RUN = re.compile(r"[ក-៓៝]+")
LATIN = re.compile(r"[a-z]+")
PROMPT = """You are helping build a dataset of romanised Khmer: how Cambodians type Khmer words with Latin letters when
chatting (Facebook, Telegram, SMS). For each Khmer word below, give the 1 to 3 most common chat spellings,
the most common first. Use only lower-case a-z. Do not translate. Do not explain.
Answer one line per word, exactly in this format:  KHMER=spelling1,spelling2

{words}
"""


def segment_fast(run: str, prefixes: set[str], vocab: frozenset[str]) -> list[str]:
    """Greedy longest-match split of a Khmer run into vocabulary words (unknown characters are skipped).

    Same rule as `khmer_translit.lexicon.segment`, but it stops extending a match once no word starts with it.

    Examples:
        >>> v = frozenset({"កខ", "គ"}); p = {"ក", "កខ", "គ"}
        >>> segment_fast("កខគឃ", p, v)
        ['កខ', 'គ']
    """
    out, i = [], 0
    while i < len(run):
        best, j = 0, i + 1
        while j <= len(run) and run[i:j] in prefixes:
            if run[i:j] in vocab:
                best = j
            j += 1
        if best:
            out.append(run[i:best])
            i = best
        else:
            i += 1
    return out


def freq() -> dict:
    """Count lexicon words in the first N_DOCS documents of the local FineWeb-2 Khmer shard.

    Two counts per word: inside segmented text, and as a whole stand-alone run (between spaces or punctuation).
    Real words also occur alone (titles, lists, spaced text); segmentation fragments almost never do.
    """
    vocab = keys()
    prefixes = {w[:k] for w in vocab for k in range(1, len(w) + 1)}
    counts: collections.Counter = collections.Counter()
    alone: collections.Counter = collections.Counter()
    n = 0
    for b in pq.ParquetFile(SHARD).iter_batches(batch_size=2000, columns=["text"]):
        for t in b.column("text").to_pylist():
            for run in KHMER_RUN.findall(t):
                counts.update(segment_fast(run, prefixes, vocab))
                if run in vocab:
                    alone[run] += 1
            n += 1
            if n >= N_DOCS:
                break
        if n >= N_DOCS:
            break
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "wordfreq.tsv", "w", encoding="utf-8") as f:
        for w, c in counts.most_common():
            f.write(f"{w}\t{c}\t{alone[w]}\n")
    return {"docs": n, "distinct_words": len(counts), "tokens": sum(counts.values()), "top20": counts.most_common(20)}


def pool() -> list[str]:
    """Words in priority order: E14's pool (chat words, extra chat list, 1000 common), then web frequency order."""
    out = [w for w, _ in e14.word_pool()]
    seen = set(out)
    for line in (DATA / "wordfreq.tsv").read_text(encoding="utf-8").splitlines():
        w, _, n_alone = line.split("\t")
        if w not in seen and e14.is_khmer_word(w) and len(w) > 1 and int(n_alone) >= MIN_ALONE:
            seen.add(w)
            out.append(w)
    return out


def labelled() -> set[str]:
    """Words that already have owner labels (E14 typing list, both halves, and the 400 checked Gemini words)."""
    done = {w["khmer"] for w in json.loads((e14.DATA / "words.json").read_text(encoding="utf-8"))}
    with open(e14.GEMINI / "verified_Richard.tsv", encoding="utf-8") as f:
        done |= {r["khmer"] for r in csv.DictReader(f, delimiter="\t")}
    return done


def select() -> dict:
    """Freeze N_TEST fresh test words from the most common unlabelled words (refuses to re-freeze)."""
    frozen = EXP / "test_frozen.json"
    if frozen.exists():
        raise SystemExit(f"{frozen} exists: the test list is frozen")
    done = labelled()
    top = [w for w in pool() if w not in done][:TEST_FROM_TOP]
    test = sorted(top, key=lambda w: hashlib.sha256(f"{SEED}:{w}".encode()).hexdigest())[:N_TEST]
    rows = [{"word_id": f"t{i + 1:03d}", "khmer": w, "split": "test"} for i, w in enumerate(test)]
    DATA.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    (DATA / "test_words.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    out = {"created": e14.now(), "seed": SEED, "words": len(rows),
           "rule": f"pool = E14 word pool then FineWeb-2 frequency order; drop words with owner labels; first "
                   f"{TEST_FROM_TOP}; lowest {N_TEST} by sha256(seed:word)",
           "test_words_sha256": e14.sha(DATA / "test_words.json"),
           "policy": "test words never enter a prompt or any training file; typed answers stay local"}
    frozen.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


def test_words() -> list[dict]:
    return json.loads((DATA / "test_words.json").read_text(encoding="utf-8"))


def prompts() -> dict:
    """Prompt files for the first N_PROMPT pool words that are neither test words nor already labelled."""
    test = {w["khmer"] for w in test_words()}
    words = [w for w in pool() if w not in test and w not in labelled()][:N_PROMPT]
    GEMINI.mkdir(parents=True, exist_ok=True)
    files = []
    for k in range(0, len(words), BATCH):
        f = GEMINI / f"prompt_{k // BATCH + 1:02d}.md"
        f.write_text(PROMPT.format(words="\n".join(words[k:k + BATCH])), encoding="utf-8")
        files.append(f.name)
    out = {"created": e14.now(), "words": len(words), "batch": BATCH, "excluded_test_words": len(test),
           "files": files, "replies": "paste each Gemini reply into reply_<nn>.txt next to its prompt"}
    (GEMINI / "prompts.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


def read_replies() -> tuple[list[tuple[str, list[str]]], dict]:
    """(Khmer word, Gemini spellings) for every answered prompt; reports partial and skipped files.

    Rows are aligned by position when the reply has one line per word (Gemini may garble a key). Otherwise lines
    are matched by their Khmer key, unmatched words are skipped, and the file is skipped if under 90% match.
    Spaced spellings (`rieb chom`) are joined, as for the owner's spaced answers.
    """
    test = {w["khmer"] for w in test_words()}
    rows, report = [], {"answered": [], "missing": [], "bad_length": []}
    for pf in sorted(GEMINI.glob("prompt_*.md")):
        pw = [x.strip() for x in pf.read_text(encoding="utf-8").split("\n\n", 1)[1].splitlines() if x.strip()]
        rf = pf.with_name(pf.name.replace("prompt_", "reply_").replace(".md", ".txt"))
        if not rf.exists():
            report["missing"].append(pf.name)
            continue
        rl = [x.strip().strip("`").strip() for x in rf.read_text(encoding="utf-8").splitlines() if "=" in x]
        if any(w in test for w in pw):
            raise SystemExit(f"test word in {pf.name}")
        if len(rl) == len(pw):
            pairs = list(zip(pw, rl))
        else:                       # a line is missing or extra: match by the Khmer key, skip unmatched words
            by_key = {}
            for line in rl:
                by_key.setdefault(line.partition("=")[0].strip(), line)
            pairs = [(w, by_key[w]) for w in pw if w in by_key]
            report["bad_length"].append(f"{rf.name}: {len(rl)} rows for {len(pw)} words, {len(pairs)} matched by key")
            if len(pairs) < 0.9 * len(pw):
                continue
        report["answered"].append(rf.name)
        for w, line in pairs:
            rows.append((w, list(dict.fromkeys(v.strip().lower().replace(" ", "")
                                               for v in line.partition("=")[2].split(",") if v.strip()))))
    return rows, report


def filter_rows(rows: list[tuple[str, list[str]]], trusted: dict[str, str], rank: dict[str, int]) -> tuple[list, dict]:
    """The automatic filter: keep a Gemini spelling only if it passes every rule.

    1. format: only a-z, 1 to 3 x (Khmer length) + 2 letters;
    2. trusted clash: dropped if trusted data (owner typing, Chhunneng chat) maps the same spelling to another word;
    3. Gemini clash: a spelling suggested for several words is kept only for the most common of them.

    Args:
        rows: (Khmer word, spellings) from `read_replies`.
        trusted: Latin spelling -> Khmer word from owner-checked and Chhunneng chat data.
        rank: Khmer word -> frequency rank (lower = more common).

    Returns:
        (kept [(latin, khmer)], counts of dropped spellings by rule).

    Examples:
        >>> filter_rows([("ក", ["ka", "k1"]), ("កា", ["ka", "kaa"]), ("គ", ["ko"])], {"ko": "ខ"}, {"ក": 0, "កា": 1})
        ([('ka', 'ក'), ('kaa', 'កា')], {'format': 1, 'trusted_clash': 1, 'gemini_clash': 1})
    """
    drop = collections.Counter({"format": 0, "trusted_clash": 0, "gemini_clash": 0})
    cand = []
    for kh, sps in rows:
        for s in sps:
            if not re.fullmatch(r"[a-z]+", s) or len(s) > 3 * len(kh) + 2:
                drop["format"] += 1
            elif s in trusted and trusted[s] != kh:
                drop["trusted_clash"] += 1
            else:
                cand.append((s, kh))
    owner: dict[str, str] = {}
    for s, kh in cand:
        if s not in owner or rank.get(kh, 10 ** 9) < rank.get(owner[s], 10 ** 9):
            owner[s] = kh
    kept = []
    for s, kh in cand:
        if owner[s] == kh:
            kept.append((s, kh))
        else:
            drop["gemini_clash"] += 1
    return kept, dict(drop)


def trusted_map() -> dict[str, str]:
    """Latin -> Khmer from the owner's typing (train half), the 400 checked words and the Chhunneng chat list."""
    out: dict[str, str] = {}
    words = json.loads((e14.DATA / "words.json").read_text(encoding="utf-8"))
    test = {w["khmer"] for w in words if w["split"] == "test"} | {w["khmer"] for w in test_words()}
    for f in (e14.DATA / "typed_Richard.tsv", e14.GEMINI / "verified_Richard.tsv"):
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r["khmer"] not in test:
                    for v in r["typed"].split(","):
                        if v.strip():
                            out.setdefault(v.replace(" ", "").strip(), r["khmer"])
    for f in sorted((e14.CHH / "khmer-english-transliteration").glob("*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            kh, _, lat = line.partition("=")
            if kh.strip() and kh.strip() not in test:
                for v in lat.split(","):
                    if LATIN.fullmatch(v.strip().lower() or "-"):
                        out.setdefault(v.strip().lower(), kh.strip())
    return out


def run_filter() -> dict:
    rows, report = read_replies()
    rank = {w: i for i, w in enumerate(pool())}
    kept, drop = filter_rows(rows, trusted_map(), rank)
    with open(DATA / "gemini_filtered.tsv", "w", encoding="utf-8") as f:
        f.write("latin\tkhmer\n")
        for s, kh in kept:
            f.write(f"{s}\t{kh}\n")
    return {**report, "words": len(rows), "spellings": sum(len(s) for _, s in rows), "kept": len(kept),
            "dropped": drop, "kept_words": len({kh for _, kh in kept})}


GEMINI_API_MODEL = "gemini-3.6-flash"
GEMINI_THINKING = "medium"   # as in the owner's app runs for E14


def fetch(limit: int | None = None, pause: float = 20.0) -> dict:
    """Send every unanswered prompt to the Gemini API and save the reply text; resumable and rate-limit tolerant.

    The key is read from the environment (GEMINI_API_KEY) and sent in a header, never written anywhere. Each call is
    logged (file, model version, token counts, time) to `fetch_log.jsonl`. On HTTP 429/5xx it waits and retries.

    Args:
        limit: send at most this many prompts (None = all).
        pause: seconds between calls.

    Raises:
        SystemExit: no key in the environment, or a non-retryable API error.
    """
    import os
    import time
    import urllib.error
    import urllib.request
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_API_MODEL}:generateContent"
    done, sent = [], 0
    for pf in sorted(GEMINI.glob("prompt_*.md")):
        rf = pf.with_name(pf.name.replace("prompt_", "reply_").replace(".md", ".txt"))
        if rf.exists():
            continue
        if limit is not None and sent >= limit:
            break
        body = json.dumps({"contents": [{"parts": [{"text": pf.read_text(encoding="utf-8")}]}],
                           "generationConfig": {"thinkingConfig": {"thinkingLevel": GEMINI_THINKING}}}).encode()
        for attempt in range(8):
            req = urllib.request.Request(url, body, {"Content-Type": "application/json", "x-goog-api-key": key})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    d = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 7:
                    time.sleep(min(600, 30 * 2 ** attempt))
                    continue
                raise SystemExit(f"{pf.name}: HTTP {e.code} {e.read()[:300]!r}")
            except (urllib.error.URLError, TimeoutError):
                time.sleep(60)
        else:
            raise SystemExit(f"{pf.name}: gave up")
        cand = (d.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
        rf.write_text(text, encoding="utf-8")
        with open(GEMINI / "fetch_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"file": rf.name, "model": d.get("modelVersion"), "finish": cand.get("finishReason"),
                                "usage": d.get("usageMetadata"), "at": e14.now()}) + "\n")
        done.append(rf.name)
        sent += 1
        time.sleep(pause)
    return {"fetched": done, "model": GEMINI_API_MODEL}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("freq", "select", "prompts", "filter"):
        sub.add_parser(c)
    fe = sub.add_parser("fetch")
    fe.add_argument("--limit", type=int)
    t = sub.add_parser("type")
    t.add_argument("--typist", required=True)
    t.add_argument("--port", type=int, default=8769)
    a = ap.parse_args()
    if a.cmd == "fetch":
        print(json.dumps(fetch(a.limit), ensure_ascii=False, indent=1))
        return 0
    if a.cmd != "type":
        fn = {"freq": freq, "select": select, "prompts": prompts, "filter": run_filter}[a.cmd]
        print(json.dumps(fn(), ensure_ascii=False, indent=1))
        return 0
    store = e14.Store(a.typist, test_words(), DATA, "typed")
    store.log = DATA / f"typed_log_{store.typist}.jsonl"
    srv = e14.ThreadingHTTPServer(("127.0.0.1", a.port), e14.make_handler(store, e14.PAGE))
    print(f"typing page: open http://127.0.0.1:{a.port}  ({len(store.answers)}/{len(store.words)} answered)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
