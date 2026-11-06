#!/usr/bin/env python3
"""E17 data (romanised Khmer, as much LLM data as the free tools allow): more words, Gemini plus Claude spellings.

    .venv/bin/python scripts/e17.py freq       # count Khmer words in the whole local FineWeb-2 shard
    .venv/bin/python scripts/e17.py prompts    # Gemini prompts (new words) and Claude prompts (E16 + new words)
    .venv/bin/python scripts/e17.py gemini     # send unanswered Gemini prompts (owner's free key, resumable)
    .venv/bin/python scripts/e17.py claude     # send unanswered Claude prompts (`claude -p`, owner's subscription)
    .venv/bin/python scripts/e17.py gpt        # send unanswered GPT prompts (`codex exec`, owner's ChatGPT login)
    .venv/bin/python scripts/e17.py filter     # merge both sources and apply E16's automatic filter

Word list: E16's 10,000 prompted words, then new words from the whole shard (stand-alone count >= MIN_ALONE),
excluding every owner-labelled word and both test lists. Gemini spells the new words (E16 already has the first
10,000); Claude and GPT spell all of them. All E17 prompts use `PROMPT` (E16's instructions made explicit, plus
examples from the owner's own typing, training half only). Nothing is checked by hand: `filter` merges the three
sources per word (Gemini, Claude, GPT) and keeps or drops each spelling with `e16.filter_rows`.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14  # noqa: E402
import e16  # noqa: E402

from khmer_translit.lexicon import keys  # noqa: E402

ROOT = e16.ROOT
DATA = ROOT / "data" / "e17"
GEMINI = DATA / "gemini"
CLAUDE = DATA / "claude"
GPT = DATA / "gpt"
EXP = ROOT / "experiments" / "E17-llm-max"
MIN_ALONE = 3             # stand-alone occurrences in the whole shard
MIN_ALONE_SHARE = 0.003   # and stand-alone in at least 0.3% of all occurrences (drops segmentation fragments)
N_NEW = 20_000            # at most this many new words
GEMINI_BATCH = 500        # words per Gemini call (the free key allows few calls a day)
CLAUDE_BATCH = 250
CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_EFFORT = "medium"  # all three models at a medium thinking level (Gemini: e16.GEMINI_THINKING)
GPT_BATCH = 250
GPT_MODEL = "gpt-6.1-sol"
GPT_REASONING = "medium"
# Few-shot examples: the owner's own typing, training half only (never a test word); spaces joined as in training.
EXAMPLES = """ខ្ញុំ=nhom
គាត់=kort,koit
នៅ=nov
ជាមួយ=cheamouy,cheamuy
ចំនួន=jomnoun,jumnoun
អ្នក=nek,neak
យើង=yerng
ចេញ=jenh
បន្ថែម=bonthaem
ពន្លឺ=punleu,ponleu
ខ្លួន=klun,kloun
កន្លែង=kornlaeng,kunlaeng
វិញ=vinh,venh
ឈ្មោះ=chmous
ខ្លាំង=klang
ទឹក=terk
ឃើញ=khernh
ម្ខាង=mkharng,mkhang
រៀន=rean
សិក្សា=serksa
ដើម=derm
ប្រាកដ=prakot
មិត្តភក្តិ=mitphaek
ញ៉ាំ=nham
ហើយ=hoy,houy
ណាស់=nas
អរគុណ=orkun
ប្អូន=paoun,poun
ម៉ោង=moang
ទិញ=tinh,tenh"""
PROMPT = """You are helping build a dataset of romanised Khmer: how Cambodians actually type Khmer words with Latin
letters when chatting (Facebook, Telegram, Messenger, SMS). This is informal chat spelling, not a formal or academic
romanisation.
For each Khmer word below, give the 1 to 3 most common chat spellings, the most common first.
Rules:
- lower-case a-z only: no accents, digits, hyphens or spaces (write a compound as one word);
- spell the word as it sounds in everyday Khmer speech;
- do not translate and do not explain.
Examples of how a native speaker types:
""" + EXAMPLES + """
Answer one line per word, in the same order, exactly in this format:  KHMER=spelling1,spelling2

{words}
"""


def freq() -> dict:
    """Word counts over the whole shard (same rule as `e16.freq`, which read the first 200,000 documents)."""
    vocab = keys()
    prefixes = {w[:k] for w in vocab for k in range(1, len(w) + 1)}
    counts: collections.Counter = collections.Counter()
    alone: collections.Counter = collections.Counter()
    n = 0
    for b in pq.ParquetFile(e16.SHARD).iter_batches(batch_size=2000, columns=["text"]):
        for t in b.column("text").to_pylist():
            for run in e16.KHMER_RUN.findall(t):
                counts.update(e16.segment_fast(run, prefixes, vocab))
                if run in vocab:
                    alone[run] += 1
            n += 1
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "wordfreq.tsv", "w", encoding="utf-8") as f:
        for w, c in counts.most_common():
            f.write(f"{w}\t{c}\t{alone[w]}\n")
    return {"docs": n, "distinct_words": len(counts), "tokens": sum(counts.values())}


def e16_words() -> list[str]:
    """The 10,000 words of E16's Gemini prompts, in prompt order."""
    out = []
    for pf in sorted(e16.GEMINI.glob("prompt_*.md")):
        out += [x.strip() for x in pf.read_text(encoding="utf-8").split("\n\n", 1)[1].splitlines() if x.strip()]
    return out


def excluded() -> set[str]:
    """Owner-labelled words (both halves of the typing list, the 400 checked words) and both test lists."""
    return e16.labelled() | {w["khmer"] for w in e16.test_words()}


def new_words() -> list[str]:
    """Words after E16's, in frequency order: E14's pool first, then the whole-shard list.

    A shard word must occur at least MIN_ALONE times as a whole stand-alone run, and in at least MIN_ALONE_SHARE of
    its occurrences: fragments of the greedy segmentation (for example តល, inside ~1M runs but alone 32 times) fail.
    """
    skip = excluded() | set(e16_words())
    out, seen = [], set()
    cand = [w for w, _ in e14.word_pool()]
    for line in (DATA / "wordfreq.tsv").read_text(encoding="utf-8").splitlines():
        w, n, n_alone = line.split("\t")
        if (e14.is_khmer_word(w) and len(w) > 1 and int(n_alone) >= MIN_ALONE
                and int(n_alone) >= MIN_ALONE_SHARE * int(n)):
            cand.append(w)
    for w in cand:
        if w not in skip and w not in seen:
            seen.add(w)
            out.append(w)
    return out[:N_NEW]


def _write(folder: Path, words: list[str], batch: int) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for k in range(0, len(words), batch):
        f = folder / f"prompt_{k // batch + 1:03d}.md"
        f.write_text(PROMPT.format(words="\n".join(words[k:k + batch])), encoding="utf-8")
        files.append(f.name)
    return files


def prompts() -> dict:
    """Gemini prompts for the new words; Claude and GPT prompts for E16's words plus the new words (refuses to redo)."""
    if GEMINI.exists() or CLAUDE.exists() or GPT.exists():
        raise SystemExit("prompt folders exist: the word lists are fixed")
    new = new_words()
    old = e16_words()
    bad = excluded() & (set(new) | set(old))
    if bad:
        raise SystemExit(f"{len(bad)} excluded words in the prompts")
    out = {"created": e14.now(), "new_words": len(new), "e16_words": len(old),
           "gemini_files": len(_write(GEMINI, new, GEMINI_BATCH)),
           "claude_files": len(_write(CLAUDE, old + new, CLAUDE_BATCH)),
           "gpt_files": len(_write(GPT, old + new, GPT_BATCH))}
    (DATA / "prompts.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


def _ask_claude(prompt: str, cwd: Path) -> tuple[str, dict]:
    p = subprocess.run(["claude", "-p", "--model", CLAUDE_MODEL, "--effort", CLAUDE_EFFORT, "--tools", "",
                        "--no-session-persistence", "--output-format", "json"], input=prompt, cwd=cwd,
                       capture_output=True, text=True, timeout=1800)
    meta = {"tool": "claude -p " + CLI_VERSION["claude"], "requested_model": CLAUDE_MODEL, "effort": CLAUDE_EFFORT}
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError:
        return "", {**meta, "error": True, "stderr": p.stderr[-300:]}
    return ("" if d.get("is_error") else d.get("result") or ""), {
        **meta, "error": bool(d.get("is_error")), "models": list(d.get("modelUsage") or {}), "usage": d.get("usage")}


def _ask_gpt(prompt: str, cwd: Path) -> tuple[str, dict]:
    out = cwd / "last_message.txt"
    out.unlink(missing_ok=True)
    p = subprocess.run(["codex", "exec", "-m", GPT_MODEL, "-s", "read-only", "--skip-git-repo-check", "--ephemeral",
                        "-c", f"model_reasoning_effort={GPT_REASONING}", "-o", str(out), "-"],
                       input=prompt, cwd=cwd, capture_output=True, text=True, timeout=1800)
    text = out.read_text(encoding="utf-8") if out.exists() and p.returncode == 0 else ""
    log = p.stdout + p.stderr
    head = dict(re.findall(r"^(model|provider|reasoning effort|sandbox): (.+)$", log, re.M))  # Codex's run header
    tok = re.search(r"tokens used\s*\n?\s*([\d,]+)", log)
    return text, {"tool": "codex exec " + CLI_VERSION["codex"], "requested_model": GPT_MODEL,
                  "effort": GPT_REASONING, "reported": head, "tokens": tok and int(tok.group(1).replace(",", "")),
                  "error": p.returncode != 0, "returncode": p.returncode}


def _version(cmd: str) -> str:
    try:
        return subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


CLI_VERSION: dict[str, str] = {}


def llm_fetch(folder: Path, ask, pause: float = 5.0) -> dict:
    """Send every unanswered prompt in `folder` through `ask` (Claude or GPT CLI) and save the reply.

    Each call runs from an empty folder outside the repository, so no project instructions are loaded. A reply with
    fewer than 90% of the expected lines, or an error (for example a usage limit), is retried with a growing wait.
    """
    cwd = Path.home() / ".cache" / f"e17_{folder.name}"
    cwd.mkdir(parents=True, exist_ok=True)
    CLI_VERSION.update(claude=_version("claude"), codex=_version("codex"))
    done = []
    for pf in sorted(folder.glob("prompt_*.md")):
        rf = pf.with_name(pf.name.replace("prompt_", "reply_").replace(".md", ".txt"))
        if rf.exists():
            continue
        prompt = pf.read_text(encoding="utf-8")
        n = len([x for x in prompt.split("\n\n", 1)[1].splitlines() if x.strip()])
        for attempt in range(12):
            try:
                text, info = ask(prompt, cwd)
            except subprocess.TimeoutExpired:
                text, info = "", {"error": True, "timeout": True}
            lines = sum("=" in x for x in text.splitlines())
            with open(folder / "fetch_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"file": rf.name, "attempt": attempt, "lines": lines, "expected": n, **info,
                                    "at": e14.now()}) + "\n")
            if lines >= 0.9 * n:
                rf.write_text(text, encoding="utf-8")
                done.append(rf.name)
                break
            time.sleep(min(3600, 60 * 2 ** attempt))
        time.sleep(pause)
    return {"fetched": len(done)}


def run_filter() -> dict:
    """Merge Gemini (E16 and E17 replies), Claude and GPT spellings per word, then apply `e16.filter_rows`.

    Writes `llm_filtered.tsv` (latin, khmer) for training and `llm_sources.tsv` (which model gave each kept
    spelling) for the analysis.
    """
    g16, r16 = e16.read_replies()
    g17, r17 = e16.read_replies(GEMINI)
    cl, rc = e16.read_replies(CLAUDE)
    gp, rg = e16.read_replies(GPT)
    merged: dict[str, list[str]] = {}
    src: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for name, rows in (("gemini", g16 + g17), ("claude", cl), ("gpt", gp)):
        for w, sp in rows:
            for s in sp:
                if s not in merged.setdefault(w, []):
                    merged[w].append(s)
                src[(s, w)].add(name)
    rank = {w: i for i, w in enumerate(e16_words() + new_words())}
    kept, drop = e16.filter_rows(list(merged.items()), e16.trusted_map(), rank)
    with open(DATA / "llm_filtered.tsv", "w", encoding="utf-8") as f, \
            open(DATA / "llm_sources.tsv", "w", encoding="utf-8") as g:
        f.write("latin\tkhmer\n")
        g.write("latin\tkhmer\tsources\n")
        for s, kh in kept:
            f.write(f"{s}\t{kh}\n")
            g.write(f"{s}\t{kh}\t{'+'.join(sorted(src[(s, kh)]))}\n")
    by = collections.Counter("+".join(sorted(src[k])) for k in kept)
    return {"gemini_files": {"e16": len(r16["answered"]), "e17": len(r17["answered"]), "e17_missing": len(r17["missing"])},
            "claude_files": {"answered": len(rc["answered"]), "missing": len(rc["missing"])},
            "gpt_files": {"answered": len(rg["answered"]), "missing": len(rg["missing"])},
            "bad_length": r17["bad_length"] + rc["bad_length"] + rg["bad_length"], "words": len(merged),
            "spellings": sum(len(v) for v in merged.values()), "kept": len(kept), "dropped": drop,
            "kept_by_source": dict(by), "kept_words": len({kh for _, kh in kept})}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["freq", "prompts", "gemini", "claude", "gpt", "filter"])
    a = ap.parse_args()
    fn = {"freq": freq, "prompts": prompts, "filter": run_filter,
          "claude": lambda: llm_fetch(CLAUDE, _ask_claude), "gpt": lambda: llm_fetch(GPT, _ask_gpt),
          "gemini": lambda: e16.fetch(folder=GEMINI)}[a.cmd]
    print(json.dumps(fn(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
