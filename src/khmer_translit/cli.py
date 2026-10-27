"""`khmer-translit`: command line for the English → Khmer phonetic transliterator (experiment E02).

    khmer-translit prepare --manifest bundle/resources.json     # check local files and hashes (no downloads)
    khmer-translit predict --model model.npz --word coffee --json
    khmer-translit predict --model model.npz --input words.txt --jsonl
    khmer-translit predict --model model.npz --word record --pronunciation "R EH1 K ER0 D"
    khmer-translit inspect --model model.npz

Everything runs offline from local files. Phone-input models need CMUdict: `--cmudict PATH` or the environment
variable `KHMER_TRANSLIT_CMUDICT`; letters-only models need nothing else. Exit status: 0 on success, 1 if any
input was refused or a check failed, 2 for usage or file errors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from khmer_translit import phonetic_source as ps


def _load(args: argparse.Namespace):
    from khmer_translit.english import Transliterator
    return Transliterator.from_pretrained(args.model, cmudict_path=args.cmudict or ps.CMUDICT,
                                          expected_sha256=getattr(args, "sha256", None))


def cmd_prepare(args: argparse.Namespace) -> int:
    """Check every file of a resource manifest (`{"files": [{"path", "sha256", "role"}]}`; paths relative to it)."""
    manifest = Path(args.manifest)
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    report, ok = [], True
    for f in spec["files"]:
        p = (manifest.parent / f["path"]).resolve()
        status = "missing" if not p.exists() else ("ok" if ps.sha256_file(p) == f["sha256"] else "sha256_mismatch")
        ok &= status == "ok"
        report.append({"path": str(p), "role": f.get("role", ""), "status": status})
    print(json.dumps({"ok": ok, "files": report}, ensure_ascii=False, indent=1))
    return 0 if ok else 1


def cmd_predict(args: argparse.Namespace) -> int:
    """Predict one word or one word per line of a file; results keep input order."""
    tl = _load(args)
    if args.word is not None:
        words = [args.word]
    else:
        text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
        words = text.splitlines()
    prons = [args.pronunciation] * len(words) if args.pronunciation else None
    results = tl.predict_many(words, top_k=args.top_k, pronunciations=prons)
    for n, r in enumerate(results, 1):
        d = {"id": n, **r.to_dict()}
        if args.json or args.jsonl:
            print(json.dumps(d, ensure_ascii=False, indent=1 if args.json else None))
        else:
            print(f"{r.input}\t{r.text if r.status == 'ok' else '[' + r.status + ']'}")
    return 0 if all(r.status == "ok" for r in results) else 1


def cmd_inspect(args: argparse.Namespace) -> int:
    """Print the model's metadata, checksum, evidence status and required resources."""
    tl = _load(args)
    info = tl.info()
    info["model_sha256"] = ps.sha256_file(Path(args.model) if Path(args.model).is_file() else Path(args.model) / "model.npz")
    info["requires"] = (["CMUdict (sha256 " + info["pronunciation"]["cmudict_sha256"] + ")"]
                        if info.get("pronunciation") else ["nothing besides numpy"])
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point of the `khmer-translit` console script."""
    ap = argparse.ArgumentParser(prog="khmer-translit", description="English word → Khmer phonetic spelling (offline)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="check local resources against a manifest (no downloads)")
    p.add_argument("--manifest", required=True)
    for name in ("predict", "inspect"):
        q = sub.add_parser(name)
        q.add_argument("--model", required=True, help="model.npz or its folder")
        q.add_argument("--cmudict", default=None, help="CMUdict file (phone-input models)")
        q.add_argument("--sha256", default=None, help="refuse the model unless its sha256 matches")
        if name == "predict":
            src = q.add_mutually_exclusive_group(required=True)
            src.add_argument("--word")
            src.add_argument("--input", help="file with one word per line, or - for stdin")
            fmt = q.add_mutually_exclusive_group()
            fmt.add_argument("--json", action="store_true")
            fmt.add_argument("--jsonl", action="store_true")
            q.add_argument("--pronunciation", help='ARPAbet with stress digits, e.g. "R EH1 K ER0 D"')
            q.add_argument("--top-k", type=int, default=1)
    args = ap.parse_args(argv)
    try:
        return {"prepare": cmd_prepare, "predict": cmd_predict, "inspect": cmd_inspect}[args.cmd](args)
    except (FileNotFoundError, ValueError) as e:
        print(f"khmer-translit: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
