#!/usr/bin/env python3
"""E14 (romanised Khmer): freeze the typing word list, then serve a local page where the owner types each Khmer
word the way they would write it in Latin letters when chatting.

    .venv/bin/python scripts/e14.py select                      # once: freeze the 320 words and the test/train split
    .venv/bin/python scripts/e14.py type --typist "Richard"     # open http://127.0.0.1:8767
    .venv/bin/python scripts/e14.py prompts                     # Gemini prompt files (owner pastes the replies)
    .venv/bin/python scripts/e14.py verify --typist "Richard"   # check Gemini's spellings, http://127.0.0.1:8768

The page shows one Khmer word at a time. The typist enters one or more spellings separated by commas (the first is
the one they would normally type) or presses "skip" when they would not type the word. Every answer rewrites
`data/e14/typing/typed_<typist>.tsv` and is appended to `review_log_<typist>.jsonl`; the page resumes at the first
unanswered word. Half of the words (`split = test`) never enter training; their typed answers stay local.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from khmer_translit.lexicon import segment

ROOT = Path(__file__).resolve().parents[1]
CHH = ROOT / "data" / "src" / "chhunneng"
DATA = ROOT / "data" / "e14" / "typing"
EXP = ROOT / "experiments" / "E14-romanised-typing"
SEED = "e14-typing-v1"
N_WORDS = 320
# everyday chat words missing from both Chhunneng lists (chosen by the agent, not taken from any test set)
EXTRA = ["អត់", "ទេ", "ញ៉ាំ", "បាយ", "អី", "ហ្នឹង", "ហើយ", "ចឹង", "ណាស់", "ផង", "ដែរ", "ទៀត", "បង", "ប្អូន",
         "អរគុណ", "សុខសប្បាយ", "ឆ្ងាញ់", "ទិញ", "លុយ", "ម៉ោង"]
KHMER_WORD = re.compile(r"^[ក-៿᧠-᧿]+$")
FIELDS = ("word_id", "khmer", "split", "typed", "skipped", "answered_at")
GEMINI_MODEL = "Gemini 3.6 Flash, medium thinking, free web app (owner-run, 2026-11-01)"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def is_khmer_word(s: str) -> bool:
    """One Khmer-script token with no spaces or Latin letters.

    >>> is_khmer_word("ខ្ញុំ"), is_khmer_word("ជា root"), is_khmer_word("")
    (True, False, False)
    """
    return bool(KHMER_WORD.match(s))


def word_pool() -> list[tuple[str, str]]:
    """(Khmer word, origin) in priority order: Chhunneng chat words, the extra list, then the common-words list."""
    seen, out = set(), []

    def add(w: str, origin: str) -> None:
        w = w.strip()
        if is_khmer_word(w) and w not in seen:
            seen.add(w)
            out.append((w, origin))

    for f in sorted((CHH / "khmer-english-transliteration").glob("*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            add(line.split("=")[0], "chhunneng_chat")
    for w in EXTRA:
        add(w, "agent_chat_list")
    for i in range(1, 11):                                     # files _1 … _10 in list order
        for line in (CHH / "1000-most-common-khmer-words" / f"1000-most-common-khmer-words_{i}.txt").read_text(
                encoding="utf-8").splitlines():
            add(line, "chhunneng_common")
    return out


def select() -> dict:
    """Freeze the first N_WORDS words of the pool and a seeded 50/50 test/train split (refuses to re-freeze)."""
    frozen = EXP / "typing_frozen.json"
    if frozen.exists():
        raise SystemExit(f"{frozen} exists: the word list is frozen")
    pool = word_pool()[:N_WORDS]
    order = sorted(range(len(pool)), key=lambda i: hashlib.sha256(f"{SEED}:{pool[i][0]}".encode()).hexdigest())
    test = set(order[: len(pool) // 2])
    rows = [{"word_id": f"w{i + 1:03d}", "khmer": w, "origin": o, "split": "test" if i in test else "train"}
            for i, (w, o) in enumerate(pool)]
    DATA.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    (DATA / "words.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    out = {"created": now(), "seed": SEED, "words": len(rows),
           "by_origin": {o: sum(r["origin"] == o for r in rows) for o in {r["origin"] for r in rows}},
           "by_split": {s: sum(r["split"] == s for r in rows) for s in ("test", "train")},
           "words_json_sha256": sha(DATA / "words.json"),
           "rule": "pool = Chhunneng chat words, then the agent's extra chat list, then the Chhunneng 1000-common list "
                   "in file order; first 320 unique single Khmer tokens; split by sha256(seed:word), lowest half = test",
           "test_policy": "test words never enter any training file (Chhunneng, Gemini drafts or typing rows are "
                          "filtered by Khmer word); their typed answers stay local"}
    frozen.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (EXP / "typing_words.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


GEMINI = DATA.parent / "gemini"
PROMPT = """You are helping build a dataset of romanised Khmer: how Cambodians type Khmer words with Latin letters when
chatting (Facebook, Telegram, SMS). For each Khmer word below, give the 1 to 3 most common chat spellings,
the most common first. Use only lower-case a-z. Do not translate. Do not explain.
Answer one line per word, exactly in this format:  KHMER=spelling1,spelling2

{words}
"""


def prompts(batch: int = 100) -> dict:
    """Gemini prompt files for the common-word list minus every frozen test word (the owner pastes the replies)."""
    words = json.loads((DATA / "words.json").read_text(encoding="utf-8"))
    test = {w["khmer"] for w in words if w["split"] == "test"}
    pool = [w for w, o in word_pool() if w not in test]
    GEMINI.mkdir(parents=True, exist_ok=True)
    files = []
    for k in range(0, len(pool), batch):
        f = GEMINI / f"prompt_{k // batch + 1:02d}.md"
        f.write_text(PROMPT.format(words="\n".join(pool[k:k + batch])), encoding="utf-8")
        files.append(f.name)
    out = {"created": now(), "words": len(pool), "excluded_test_words": len(test), "files": files,
           "replies": "paste each Gemini reply into reply_<nn>.txt next to its prompt"}
    (GEMINI / "prompts.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


def drafts() -> list[dict]:
    """Gemini's suggested spellings, read from reply_<nn>.txt and aligned with prompt_<nn>.md line by line.

    Gemini occasionally garbles the Khmer key (e.g. a Latin letter inside it), so rows are matched by position, and a
    file whose line count differs from its prompt is an error. Words the owner already typed (train half of the
    typing list) are left out: their labels exist. Test words were never in the prompts; this is checked again.

    Returns:
        [{"word_id": "g001", "khmer": ..., "split": "train", "gemini": ["knhom", "khnhom"], "key_ok": bool}, ...]

    Raises:
        SystemExit: a reply file is missing or has a different number of rows than its prompt, or a test word leaked.
    """
    words = json.loads((DATA / "words.json").read_text(encoding="utf-8"))
    test = {w["khmer"] for w in words if w["split"] == "test"}
    typed = {w["khmer"] for w in words if w["split"] == "train"}
    out = []
    for pf in sorted(GEMINI.glob("prompt_*.md")):
        pw = [x.strip() for x in pf.read_text(encoding="utf-8").split("\n\n", 1)[1].splitlines() if x.strip()]
        rf = pf.with_name(pf.name.replace("prompt_", "reply_").replace(".md", ".txt"))
        if not rf.exists():
            raise SystemExit(f"missing {rf.name}")
        rl = [x.strip().strip("`").strip() for x in rf.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(rl) != len(pw):
            raise SystemExit(f"{rf.name}: {len(rl)} rows for {len(pw)} prompt words")
        for w, line in zip(pw, rl):
            if w in test:
                raise SystemExit(f"test word {w} in a Gemini prompt")
            key, _, val = line.partition("=")
            sp = [v.strip().lower() for v in val.split(",") if v.strip()]
            if w not in typed:
                out.append({"word_id": f"g{len(out) + 1:03d}", "khmer": w, "split": "train",
                            "gemini": list(dict.fromkeys(sp)), "key_ok": key.strip() == w})
    vocab = {k: 1 for k in {x["khmer"] for x in words} | {x["khmer"] for x in out}}
    for x in out:                                       # split compounds into known words, for the spaced suggestion
        seg = segment(x["khmer"], {k: 1 for k in vocab if k != x["khmer"]})
        x["pieces"] = [p for p, _ in seg] if len(seg) > 1 and all(ok for _, ok in seg) else []
    return out


def known_spellings(typist: str) -> dict[str, str]:
    """The owner's usual spelling (first variant) of each *train* typing word; test words are never included."""
    f = DATA / f"typed_{re.sub(r'[^A-Za-z0-9_-]', '_', typist)}.tsv"
    if not f.exists():
        return {}
    with open(f, encoding="utf-8") as fh:
        return {r["khmer"]: r["typed"].split(",")[0] for r in csv.DictReader(fh, delimiter="\t")
                if r["split"] == "train" and r["typed"]}


class Store:
    """The typist's answers, rewritten to TSV on every save (resume-safe).

    Args:
        typist: name used in the file names.
        words: the word list (default: the frozen typing list).
        folder, prefix: where to write `<prefix>_<typist>.tsv` (default: the typing folder, `typed`).
    """

    def __init__(self, typist: str, words: list[dict] | None = None, folder: Path = DATA, prefix: str = "typed") -> None:
        self.typist = re.sub(r"[^A-Za-z0-9_-]", "_", typist)
        self.words = words if words is not None else json.loads((DATA / "words.json").read_text(encoding="utf-8"))
        self.path = folder / f"{prefix}_{self.typist}.tsv"
        self.log = folder / f"{prefix}_log_{self.typist}.jsonl" if prefix != "typed" else DATA / f"review_log_{self.typist}.jsonl"
        self.extra: dict = {}
        self.lock = threading.Lock()
        self.answers: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    if r["typed"] or r["skipped"] == "1":
                        self.answers[r["word_id"]] = r

    def state(self) -> dict:
        return {"typist": self.typist,
                "words": [{"word_id": w["word_id"], "khmer": w["khmer"], "gemini": w.get("gemini", []),
                           "pieces": w.get("pieces", [])} for w in self.words], **self.extra,
                "answers": {k: {"typed": v["typed"], "skipped": v["skipped"] == "1"} for k, v in self.answers.items()}}

    def save(self, word_id: str, typed: str, skipped: bool) -> dict:
        """Store one answer: comma-separated Latin spellings, or a skip.

        Raises:
            ValueError: unknown word, non-Latin text, or neither spellings nor a skip.
        """
        ids = {w["word_id"]: w for w in self.words}
        if word_id not in ids:
            raise ValueError(f"unknown word {word_id}")
        variants = [v.strip().lower() for v in typed.split(",") if v.strip()]
        if not skipped and not variants:
            raise ValueError("type a spelling or press skip")
        if any(not re.fullmatch(r"[a-z'\- ]+", v) for v in variants):
            raise ValueError("Latin letters only (a-z, apostrophe, hyphen, space)")
        w = ids[word_id]
        row = {"word_id": word_id, "khmer": w["khmer"], "split": w["split"],
               **({"gemini": ",".join(w["gemini"])} if "gemini" in w else {}),
               "typed": ",".join(dict.fromkeys(variants)) if not skipped else "", "skipped": "1" if skipped else "0",
               "answered_at": now()}
        with self.lock:
            self.answers[word_id] = row
            with open(self.path, "w", encoding="utf-8", newline="") as f:
                wr = csv.DictWriter(f, FIELDS + (("gemini",) if self.words and "gemini" in self.words[0] else ()),
                                    delimiter="\t")
                wr.writeheader()
                for x in self.words:
                    if x["word_id"] in self.answers:
                        wr.writerow(self.answers[x["word_id"]])
            with open(self.log, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Romanised Khmer typing</title><style>
:root{--bg:#fafaf7;--fg:#1d1d1b;--mut:#6b6b66;--acc:#2d5bd7;--card:#fff;--bd:#ddd}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#eee;--mut:#9a9a93;--acc:#7ea2ff;--card:#202020;--bd:#333}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:560px;margin:0 auto;padding:24px 16px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:28px 20px;text-align:center}
.km{font-size:64px;line-height:1.3;font-family:"Noto Sans Khmer","Noto Serif Khmer",sans-serif;margin:8px 0 20px}
input{width:100%;box-sizing:border-box;font-size:22px;padding:10px 12px;border:1px solid var(--bd);border-radius:8px;
background:var(--bg);color:var(--fg)}
.row{display:flex;gap:8px;margin-top:12px}button{flex:1;font-size:16px;padding:10px;border-radius:8px;border:1px solid var(--bd);
background:var(--bg);color:var(--fg);cursor:pointer}button.pri{background:var(--acc);color:#fff;border-color:var(--acc)}
.mut{color:var(--mut);font-size:14px}.bar{height:6px;background:var(--bd);border-radius:3px;margin:12px 0}
.bar>div{height:100%;background:var(--acc);border-radius:3px}.err{color:#c0392b;min-height:1.5em}
</style></head><body><main>
<p class="mut">Type each Khmer word the way <b>you</b> would write it in Latin letters when chatting. Several ways? Separate
them with commas, the usual one first (e.g. <i>knhom, khnhom</i>). Press <b>Skip</b> if you would never type it.</p>
<div class="bar"><div id="bar"></div></div><p class="mut" id="prog"></p>
<div class="card"><div class="km" id="km"></div><input id="t" autocomplete="off" autocapitalize="off" spellcheck="false">
<div class="err" id="err"></div><div class="row"><button id="prev">&larr; Back</button><button id="skip">Skip</button>
<button class="pri" id="save">Save &rarr;</button></div></div>
<p class="mut">Enter = save. Answers are saved on every click; you can close the page and come back.</p></main>
<script>
let S,i=0;const $=x=>document.getElementById(x);
async function load(){S=await (await fetch('/api/state')).json();i=S.words.findIndex(w=>!S.answers[w.word_id]);if(i<0)i=S.words.length-1;show()}
function show(){const w=S.words[i],a=S.answers[w.word_id];$('km').textContent=w.khmer;$('t').value=a&&!a.skipped?a.typed.split(',').join(', '):'';
const n=Object.keys(S.answers).length;$('prog').textContent=`Word ${i+1} of ${S.words.length} · ${n} answered`+(a&&a.skipped?' · (skipped)':'');
$('bar').style.width=(100*n/S.words.length)+'%';$('err').textContent='';$('t').focus()}
async function send(skipped){const w=S.words[i];const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({word_id:w.word_id,typed:$('t').value,skipped})});const j=await r.json();if(j.error){$('err').textContent=j.error;return}
S.answers[w.word_id]={typed:j.saved.typed,skipped};if(i<S.words.length-1)i++;show()}
$('save').onclick=()=>send(false);$('skip').onclick=()=>send(true);$('prev').onclick=()=>{if(i>0)i--;show()};
$('t').addEventListener('keydown',e=>{if(e.key==='Enter')send(false)});load();
</script></body></html>"""


VERIFY_PAGE = (PAGE
    .replace("<title>Romanised Khmer typing</title>", "<title>Check Gemini spellings</title>")
    .replace('''<p class="mut">Type each Khmer word the way <b>you</b> would write it in Latin letters when chatting. Several ways? Separate
them with commas, the usual one first (e.g. <i>knhom, khnhom</i>). Press <b>Skip</b> if you would never type it.</p>''',
'''<p class="mut">Gemini suggested the spellings under each word. Click (or press 1, 2, 3) the ones a Cambodian would
really type; they go into the box, and you can edit or add your own. Only what is in the box is kept. Box empty and
<b>Skip</b> = drop the word. A <b>★</b> button joins your own spellings of the word's parts with spaces.</p>''')
    .replace('<div class="km" id="km"></div>', '<div class="km" id="km"></div><p class="mut" id="pieces"></p>')
    .replace('<div class="err" id="err">', '<div id="chips" class="row" style="flex-wrap:wrap"></div><div class="err" id="err">')
    .replace("$('bar').style.width", "chips(w);$('bar').style.width")
    .replace("$('t').addEventListener", '''function known(){const k={...S.known};S.words.forEach(x=>{const a=S.answers[x.word_id];if(a&&!a.skipped&&a.typed)k[x.khmer]=a.typed.split(',')[0]});return k}
function chips(w){const c=$('chips');c.innerHTML='';$('pieces').textContent=w.pieces.length?'parts: '+w.pieces.join(' · '):'';
const K=known();if(w.pieces.length&&w.pieces.every(p=>K[p])){const sp=w.pieces.map(p=>K[p]).join(' ');const b=document.createElement('button');
b.textContent='★  '+sp;b.onclick=()=>add(sp);c.appendChild(b)}
w.gemini.forEach((g,k)=>{const b=document.createElement('button');
b.textContent=(k+1)+'  '+g;b.onclick=()=>add(g);c.appendChild(b)})}
function add(g){const v=$('t').value.split(',').map(x=>x.trim()).filter(x=>x);if(!v.includes(g))v.push(g);$('t').value=v.join(', ');$('t').focus()}
document.addEventListener('keydown',e=>{const k=parseInt(e.key);const g=S&&S.words[i].gemini;
if(g&&k>=1&&k<=g.length){e.preventDefault();add(g[k-1])}});
$('t').addEventListener'''))


def make_handler(store: Store, page: str = PAGE):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def json(self, obj, code=HTTPStatus.OK):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/api/state":
                return self.json(store.state())
            b = page.encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_POST(self):
            if self.path != "/api/save":
                return self.json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                row = store.save(body["word_id"], body.get("typed", ""), bool(body.get("skipped")))
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                return self.json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            self.json({"saved": row})

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    sub.add_parser("prompts")
    t = sub.add_parser("type")
    t.add_argument("--typist", required=True)
    t.add_argument("--port", type=int, default=8767)
    v = sub.add_parser("verify")
    v.add_argument("--typist", required=True)
    v.add_argument("--port", type=int, default=8768)
    a = ap.parse_args()
    if a.cmd in ("select", "prompts"):
        print(json.dumps(select() if a.cmd == "select" else prompts(), ensure_ascii=False, indent=1))
        return 0
    if a.cmd == "verify":
        words = drafts()
        (GEMINI / "drafts.json").write_text(json.dumps({"model": GEMINI_MODEL, "words": words}, ensure_ascii=False,
                                                       indent=1) + "\n", encoding="utf-8")
        store = Store(a.typist, words, GEMINI, "verified")
        store.extra = {"known": known_spellings(a.typist)}
        page = VERIFY_PAGE
    else:
        store, page = Store(a.typist), PAGE
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(store, page))
    print(f"typing page: open http://127.0.0.1:{a.port}  ({len(store.answers)}/{len(store.words)} answered; Ctrl+C stops)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
