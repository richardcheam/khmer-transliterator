#!/usr/bin/env python3
"""Pin the data: sha256 and size of every file under `data/`, in the tracked `datasets/manifest.json`.

The data is not in git, so this manifest is how the repository records which data version an experiment used.

    .venv/bin/python scripts/data_manifest.py check    # compare data/ with the manifest (exit 1 on any change)
    .venv/bin/python scripts/data_manifest.py write    # rewrite the manifest after an intended data change

After `write`, commit the manifest together with a note in `datasets/README.md` (what changed and why).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from khmer_translit.experiment import sha256  # noqa: E402

DATA = ROOT / "data"
MANIFEST = ROOT / "datasets" / "manifest.json"


def scan(folder: Path = DATA) -> dict[str, dict]:
    """Hash every file under a folder.

    Args:
        folder: The data folder (default `data/`).

    Returns:
        `{relative path: {"bytes", "sha256"}}`, sorted by path.
    """
    return {str(p.relative_to(folder)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(folder.rglob("*")) if p.is_file()}


def compare(old: dict[str, dict], new: dict[str, dict]) -> dict[str, list[str]]:
    """Differences between two scans.

    Args:
        old: The manifest's files.
        new: A fresh `scan()`.

    Returns:
        `{"changed": [...], "missing": [...], "added": [...]}` (paths).

    Examples:
        >>> compare({"a": {"sha256": "1"}, "b": {"sha256": "2"}}, {"a": {"sha256": "9"}, "c": {"sha256": "3"}})
        {'changed': ['a'], 'missing': ['b'], 'added': ['c']}
    """
    return {"changed": sorted(p for p in old.keys() & new.keys() if old[p]["sha256"] != new[p]["sha256"]),
            "missing": sorted(old.keys() - new.keys()), "added": sorted(new.keys() - old.keys())}


def main() -> int:
    """CLI: `check` or `write` (see the module docstring).

    Returns:
        0 if the data matches (check) or after writing; 1 if check finds a difference.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "write"])
    mode = ap.parse_args().mode
    files = scan()
    if mode == "write":
        MANIFEST.parent.mkdir(exist_ok=True)
        MANIFEST.write_text(json.dumps({"files": len(files), "bytes": sum(f["bytes"] for f in files.values()),
                                        "data": files}, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(files)} files")
        return 0
    diff = compare(json.loads(MANIFEST.read_text(encoding="utf-8"))["data"], files)
    for kind, paths in diff.items():
        for p in paths:
            print(f"{kind}: {p}")
    print("data matches the manifest" if not any(diff.values()) else "data differs from the manifest")
    return 1 if any(diff.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
