"""`khmer-translit`: command line for the English → Khmer phonetic transliterator.

    khmer-translit prepare --manifest bundle/resources.json     # check local files and hashes (no downloads)
    khmer-translit predict --model model.npz --word coffee --json
    khmer-translit predict --model model.npz --input words.txt --jsonl
    khmer-translit predict --model model.npz --word record --pronunciation "R EH1 K ER0 D"
    khmer-translit inspect --model model.npz
    khmer-translit predict --backend byt5 --model ADAPTER_DIR --base BASE_DIR --offline --input words.txt --jsonl
    khmer-translit predict --backend byt5 --model ADAPTER_DIR --word coffee --top-k 4 --json

Backends: `e02` (default; the historical NumPy model, offline from local files; phone-input models need CMUdict via
`--cmudict PATH` or `KHMER_TRANSLIT_CMUDICT`) and `byt5` (research preview; needs `pip install
'khmer-translit[byt5]'`, an adapter folder and the pinned `google/byt5-small` base, from `--base FOLDER` or the
Transformers cache). The model is loaded once per process. Exit status: 0 on success, 1 if any input was refused or
a check failed, 2 for usage or file errors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from khmer_translit import phonetic_source as ps


def _load(args: argparse.Namespace):
    if getattr(args, "backend", "e02") == "byt5":
        from khmer_translit.byt5 import ByT5Transliterator
        return ByT5Transliterator.from_pretrained(args.model, base=args.base, offline=args.offline or None,
                                                  threads=args.threads, expected_sha256=getattr(args, "sha256", None))
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
    if args.backend == "byt5":
        if args.pronunciation:
            raise ValueError("--pronunciation is not used by the byt5 backend (it reads the letters only)")
        results = tl.predict_many(words, top_k=args.top_k)
    else:
        prons = [args.pronunciation] * len(words) if args.pronunciation else None
        results = tl.predict_many(words, top_k=args.top_k, pronunciations=prons)
    for n, r in enumerate(results, 1):
        d = {"id": n, "backend": args.backend, **r.to_dict()}
        if args.json or args.jsonl:
            print(json.dumps(d, ensure_ascii=False, indent=1 if args.json else None))
        else:
            print(f"{r.input}\t{r.text if r.status == 'ok' else '[' + r.status + ']'}")
    return 0 if all(r.status == "ok" for r in results) else 1


def cmd_inspect(args: argparse.Namespace) -> int:
    """Print the model's metadata, checksum, evidence status and required resources."""
    tl = _load(args)
    info = tl.info()
    if args.backend == "byt5":
        print(json.dumps(info, ensure_ascii=False, indent=1))
        return 0
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
        q.add_argument("--backend", choices=("e02", "byt5"), default="e02",
                       help="e02: historical NumPy model (default); byt5: ByT5 + LoRA research preview")
        q.add_argument("--model", required=True, help="e02: model.npz or its folder; byt5: the adapter folder")
        q.add_argument("--base", default=None, help="byt5: local google/byt5-small folder or Hub id (pinned revision)")
        q.add_argument("--offline", action="store_true", help="byt5: never download (also HF_HUB_OFFLINE=1)")
        q.add_argument("--threads", type=int, default=None, help="byt5: CPU threads")
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
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"khmer-translit: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
