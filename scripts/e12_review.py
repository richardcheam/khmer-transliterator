#!/usr/bin/env python3
"""Local review page for the E12 blinded test: one word per screen, answers saved straight into the import files.

Run from the repository root:

    .venv/bin/python scripts/e12_review.py --reviewer "Richard"

then open http://127.0.0.1:8766. Another blinded set (same `batch_<n>.tsv` format) is served with `--data <folder>`
and `--title`, e.g. E13's collection and test reviews. The page shows each word, its US reading and its blinded spellings (A/B), with
three answers per spelling: acceptable, not acceptable, uncertain. Every click rewrites `judgments_<n>.tsv` (the
`batch_<n>.tsv` template with the judgment column filled; unanswered rows stay blank, which the importer counts as
missing) and appends the click to `review_log.jsonl`, all in `data/e12/test/`. Stop with Ctrl+C at any time; the
page resumes at the first unanswered word. The server binds to 127.0.0.1, never reads `origin_key.json` and has
no route that shows model identity, references or scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "data" / "e12" / "test"
UI = Path(__file__).with_name("e12_review_ui.html")
POLICY = ROOT / "docs" / "PHONETIC_POLICY.md"
JUDGMENTS = ("acceptable", "not_acceptable", "uncertain")      # = e10.JUDGMENTS (checked in tests/test_e12.py)
FIELDS = ("item_id", "word", "intended_reading", "label", "candidate", "judgment")


def now() -> str:
    """Current UTC time, ISO 8601 to the second."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_tsv(p: Path) -> list[dict]:
    """Rows of a batch or judgments TSV."""
    return list(csv.DictReader(io.StringIO(p.read_text(encoding="utf-8")), delimiter="\t"))


class Store:
    """The four batches and their answers, kept in `judgments_<n>.tsv` beside the templates.

    Args:
        data: The test folder holding `batch_<n>.tsv` (normally `data/e12/test/`).
        reviewer: The name written to `judgments.provenance.json` and the click log.
        title: The page heading.

    Raises:
        FileNotFoundError: No `batch_<n>.tsv` in `data`.

    Example:
        >>> import tempfile, pathlib
        >>> d = pathlib.Path(tempfile.mkdtemp())
        >>> _ = (d / "batch_1.tsv").write_text("\\t".join(FIELDS) + "\\nt1\\tcat\\tK AE1 T\\tA\\tខាត\\t\\n", encoding="utf-8")
        >>> s = Store(d, "tester")
        >>> s.save("t1", "A", "acceptable")["judgment"]
        'acceptable'
        >>> read_tsv(d / "judgments_1.tsv")[0]["judgment"]
        'acceptable'
    """

    def __init__(self, data: Path, reviewer: str, title: str = "E12 spelling review") -> None:
        self.data, self.reviewer, self.title, self.lock = data, reviewer, title, threading.Lock()
        self.batches: dict[str, list[dict]] = {}
        for p in sorted(data.glob("batch_*.tsv"), key=lambda p: int(p.stem.split("_")[1])):
            n = p.stem.split("_")[1]
            rows = read_tsv(p)
            done = data / f"judgments_{n}.tsv"
            if done.exists():                           # resume: keep earlier answers for unchanged rows
                old = {(r["item_id"], r["label"]): r for r in read_tsv(done)}
                for r in rows:
                    o = old.get((r["item_id"], r["label"]))
                    if o and o["candidate"] == r["candidate"] and o.get("judgment") in JUDGMENTS:
                        r["judgment"] = o["judgment"]
            self.batches[n] = rows
        if not self.batches:
            raise FileNotFoundError(f"no batch_<n>.tsv in {data}")
        prov = data / "judgments.provenance.json"
        if not prov.exists():
            prov.write_text(json.dumps({
                "judge": reviewer, "started": now(), "tool": "scripts/e12_review.py (local page, click answers)",
                "policy": "docs/PHONETIC_POLICY.md; explicit acceptable / not_acceptable / uncertain per spelling",
                "policy_sha256": hashlib.sha256(POLICY.read_bytes()).hexdigest() if POLICY.exists() else None,
                "set": title, "timestamps": "every click in review_log.jsonl (UTC)",
                "blinded": True, "independent": False,
                "note": "the owner judged each spelling on the page; nothing typed or filled in by an agent"},
                indent=1) + "\n", encoding="utf-8")

    def state(self) -> dict:
        """Batches as items (word, reading, spellings with their current answers) for the page."""
        out = {}
        for n, rows in self.batches.items():
            items: dict[str, dict] = {}
            for r in rows:
                it = items.setdefault(r["item_id"], {"item_id": r["item_id"], "word": r["word"],
                                                     "reading": r["intended_reading"], "candidates": []})
                it["candidates"].append({"label": r["label"], "text": r["candidate"],
                                         "judgment": r.get("judgment") or ""})
            out[n] = list(items.values())
        return {"reviewer": self.reviewer, "title": self.title, "judgments": list(JUDGMENTS), "batches": out}

    def save(self, item_id: str, label: str, judgment: str) -> dict:
        """Record one answer and rewrite that batch's judgments file.

        Args:
            item_id: For example "t001".
            label: "A" or "B", as shown.
            judgment: One of `JUDGMENTS`, or "" to clear the answer.

        Returns:
            The updated row.

        Raises:
            KeyError: Unknown item or label.
            ValueError: Invalid judgment.
        """
        if judgment and judgment not in JUDGMENTS:
            raise ValueError(f"invalid judgment {judgment!r}")
        with self.lock:
            for n, rows in self.batches.items():
                for r in rows:
                    if r["item_id"] == item_id and r["label"] == label:
                        r["judgment"] = judgment
                        buf = io.StringIO()
                        w = csv.DictWriter(buf, FIELDS, delimiter="\t", lineterminator="\n", extrasaction="ignore")
                        w.writeheader()
                        w.writerows(rows)
                        tmp = self.data / f".judgments_{n}.tsv.tmp"
                        tmp.write_text(buf.getvalue(), encoding="utf-8")
                        tmp.replace(self.data / f"judgments_{n}.tsv")
                        with (self.data / "review_log.jsonl").open("a", encoding="utf-8") as f:
                            f.write(json.dumps({"utc": now(), "reviewer": self.reviewer, "item_id": item_id,
                                                "label": label, "judgment": judgment}) + "\n")
                        return dict(r)
        raise KeyError(f"{item_id}:{label}")


def make_handler(store: Store) -> type[BaseHTTPRequestHandler]:
    """HTTP handler: `GET /` (page), `GET /api/state`, `POST /api/judge` {item_id, label, judgment}.

    Args:
        store: The answers store.

    Returns:
        The handler class; bad requests get HTTP 400/404 with `{"error": ...}` and nothing is written.
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            """Silence the per-request console log."""

        def send(self, body: bytes, ctype: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send(json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self.send(UI.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.json(store.state())
            else:
                self.json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/judge":
                return self.json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            try:
                b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                row = store.save(str(b["item_id"]), str(b["label"]), str(b.get("judgment") or ""))
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                return self.json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            self.json({"saved": row})

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--data", type=Path, default=TEST_DATA, help="folder with batch_<n>.tsv (default: E12 test)")
    ap.add_argument("--title", default="E12 spelling review")
    a = ap.parse_args()
    store = Store(a.data if a.data.is_absolute() else ROOT / a.data, a.reviewer, a.title)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(store))
    print(f"{a.title}: open http://127.0.0.1:{a.port}  (Ctrl+C to stop; answers are saved on every click)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
