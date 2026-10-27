#!/usr/bin/env python3
"""Experiment E03: train / export / verify one arm on the DACS data, with E02's trainer unchanged.

    .venv/bin/python scripts/e03_train.py train --arm phones --seed 0 --out models/e03/phones_s0
    .venv/bin/python scripts/e03_train.py export --out models/e03/phones_s0
    .venv/bin/python scripts/e03_train.py verify --out models/e03/phones_s0

`english_phonetic_train` is reused as is (model, optimizer, schedule, pairing, private RNG streams, phone dropout,
checkpoint rule, export and parity check). Only its module-level paths are pointed at E03: data `data/e03/`,
config `experiments/E03-assessable-phones/config.yaml`, arms `letters` / `phones`, the run records' experiment id,
and the exported model's evidence note. E02's files and code are not modified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import english_phonetic_train as T  # noqa: E402
from khmer_translit import experiment  # noqa: E402
from khmer_translit import phonetic_source as ps  # noqa: E402

EXPERIMENT = "E03-assessable-phones"
DATA = ROOT / "data" / "e03"
CONFIG = ROOT / "experiments" / EXPERIMENT / "config.yaml"
ARMS = ("letters", "phones")
EVIDENCE = "mechanism study on screened mined pairs; references unqualified; not a product default (E03)"


class _Experiment:
    """`experiment` as seen by the trainer: the same functions, with E03's experiment id on every run record."""

    load_config = staticmethod(experiment.load_config)

    @staticmethod
    def Run(folder, _experiment, name, config, inputs=(), job_type="train"):
        return experiment.Run(folder, EXPERIMENT, name, config, inputs, job_type)


def configure() -> None:
    """Point the shared trainer at E03 (idempotent)."""
    T.DATA, T.CONFIG, T.ARMS = DATA, CONFIG, ARMS
    T.experiment = _Experiment
    T.data_inputs = lambda: [CONFIG, DATA / "manifest.json", DATA / "train.tsv", DATA / "dev_references.jsonl",
                             DATA / "vocab.json", ps.CMUDICT]


def export(out: Path, cfg: dict) -> Path:
    """E02's export, then the E03 evidence note and experiment id in the metadata (weights untouched)."""
    path = T.export(out, cfg)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files if k != "__meta__"}
        meta = json.loads(str(z["__meta__"]))
    meta["evidence_status"], meta["experiment"] = EVIDENCE, EXPERIMENT
    tmp = out / "model.tmp.npz"
    np.savez(tmp, __meta__=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    os.replace(tmp, path)
    return path


def main() -> int:
    """CLI: train, export or verify one E03 job."""
    configure()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["train", "export", "verify"])
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = T.load_cfg([])
    out = Path(args.out)
    if args.mode == "train":
        if not args.arm:
            ap.error("--arm is required")
        T.train(args.arm, args.seed, out, cfg)
    elif args.mode == "export":
        export(out, cfg)
    else:
        return 0 if T.verify(out)["ok"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
