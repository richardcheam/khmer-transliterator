"""Research bookkeeping for training runs: strict configs, provenance, and run logs mirrored to W&B.

Every run folder written through this module holds:
- `config.json`: the resolved config (the experiment's `config.yaml` plus any `--set` overrides, which are listed);
- `provenance.json`: code version (git commit, uncommitted changes and a hash of their diff), Python and package
  versions, machine, start time, and the sha256 of every input file;
- `metrics.jsonl`: one JSON object per logged step, **the source of truth** for the run;
- `summary.json`: final values (best epoch, test results, ...).

`Run` mirrors the same records to Weights & Biases (team `richardcheam912-ensiie`, project `khmer-transliterator`,
group = experiment id). The team is fixed here on purpose: the laptop's shell sets `WANDB_ENTITY` to a name W&B
does not know, so only `KHMER_WANDB_ENTITY` can change it. W&B is a viewer, not a store: if it is unavailable, the
run continues and the error is noted in `provenance.json`. Set `WANDB_MODE=offline` to log offline, or
`WANDB_MODE=disabled` to skip W&B.

Standard library only, except `pyyaml` (configs) and `wandb` (optional).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WANDB_ENTITY = "richardcheam912-ensiie"
WANDB_PROJECT = "khmer-transliterator"


def sha256(path: Path) -> str:
    """The sha256 of a file, read in 1 MB blocks.

    Args:
        path: Any file.

    Returns:
        The hex digest.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path: Path, expected: Mapping[str, Mapping[str, Any]], overrides: Iterable[str] = ()) -> dict:
    """Read an experiment config and check it has exactly the expected sections and keys.

    No value is filled in silently: the file must name every key, so the config alone describes the run.

    Args:
        path: A YAML file with sections (e.g. `data`, `model`, `train`) of scalar values.
        expected: `{section: {key: example_value}}`; the example's type is the required type (an int is accepted
            where a float is expected).
        overrides: `"section.key=value"` strings applied after reading (e.g. `"train.epochs=1"` for a smoke run).
            Values are parsed as YAML scalars.

    Returns:
        `{section: {key: value}}`, plus `"overrides": [...]` listing what was overridden.

    Raises:
        ValueError: On a missing or unknown section or key, a wrong type, or a malformed override.

    Example:
        `load_config(Path("experiments/E01-unified/config.yaml"), EXPECTED, ["train.epochs=1"])`
    """
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    meta = {k: cfg.pop(k) for k in ("experiment", "description") if k in cfg}
    for item in overrides:
        key, sep, raw = item.partition("=")
        section, dot, name = key.partition(".")
        if not sep or not dot:
            raise ValueError(f"override must look like section.key=value, not {item!r}")
        cfg.setdefault(section, {})[name] = yaml.safe_load(raw)
    if set(cfg) != set(expected):
        raise ValueError(f"config sections {sorted(cfg)} != expected {sorted(expected)}")
    for section, keys in expected.items():
        got = cfg[section] or {}
        if set(got) != set(keys):
            missing, unknown = sorted(set(keys) - set(got)), sorted(set(got) - set(keys))
            raise ValueError(f"[{section}] missing {missing}, unknown {unknown}")
        for k, example in keys.items():
            want = (int, float) if isinstance(example, float) else type(example)
            if isinstance(got[k], bool) != isinstance(example, bool) or not isinstance(got[k], want):
                raise ValueError(f"[{section}] {k} must be {type(example).__name__}, got {got[k]!r}")
    return {**meta, **cfg, "overrides": list(overrides)}


def _git(*args: str) -> str:
    """Output of one git command in the repository root ("" if git fails)."""
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def provenance(inputs: Iterable[Path] = ()) -> dict:
    """What produced a run: code version, environment and input file hashes.

    Args:
        inputs: Files the run reads (data, configs); each is hashed. Paths are recorded relative to the repository
            root (`../` for files outside it, such as the speech repo's loanword list).

    Returns:
        `{"git_commit", "git_dirty", "git_diff_sha256", "python", "packages", "platform", "cpu_count", "argv",
        "started_utc", "inputs"}`. A dirty tree means the run used uncommitted code: the diff hash identifies it,
        but only a commit makes it reproducible.
    """
    diff = _git("diff", "HEAD")
    packages = {}
    for name in ("numpy", "torch", "wandb"):
        mod = sys.modules.get(name)
        if mod is None:
            try:
                mod = __import__(name)
            except ImportError:
                continue
        packages[name] = getattr(mod, "__version__", "?")
    return {"git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
            "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "python": sys.version.split()[0], "packages": packages, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "argv": sys.argv,
            "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "inputs": {os.path.relpath(p, ROOT): sha256(Path(p)) for p in inputs}}


class Run:
    """One training or evaluation run: local records first, W&B mirror second.

    Args:
        folder: The run folder (created if needed).
        experiment: The experiment id, e.g. "E01-unified" (the W&B group).
        name: The run name, e.g. "aux_s0".
        config: The resolved config (`load_config`).
        inputs: Files to hash into `provenance.json`.
        job_type: "train" or "eval" (W&B job type).

    Example:
        `run = Run(out, "E01-unified", "aux_s0", cfg, inputs); run.log({"val_mean": 0.31}, step=1); run.finish({...})`
    """

    def __init__(self, folder: Path, experiment: str, name: str, config: Mapping[str, Any],
                 inputs: Iterable[Path] = (), job_type: str = "train") -> None:
        """Write `config.json` and `provenance.json`, then start the W&B mirror (see the class docstring)."""
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.prov = provenance(inputs)
        (self.folder / "config.json").write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
        self.metrics = self.folder / "metrics.jsonl"
        self.wb = None
        if os.environ.get("WANDB_MODE") != "disabled":
            id_file = self.folder / "wandb_run_id.txt"
            if not id_file.exists():
                id_file.write_text(f"{experiment}-{name}-{hashlib.sha1(os.urandom(8)).hexdigest()[:8]}\n")
            try:
                import wandb
                self.wb = wandb.init(entity=os.environ.get("KHMER_WANDB_ENTITY", WANDB_ENTITY), project=WANDB_PROJECT,
                                     group=experiment, name=name, job_type=job_type, id=id_file.read_text().strip(),
                                     resume="allow", config={**config, "provenance": self.prov}, dir=str(self.folder))
            except Exception as exc:   # W&B is a mirror only: never let it stop a run
                self.prov["wandb_error"] = repr(exc)
        (self.folder / "provenance.json").write_text(json.dumps(self.prov, indent=1) + "\n", encoding="utf-8")

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        """Append one record to `metrics.jsonl` and mirror it to W&B.

        Args:
            metrics: JSON-serialisable values (nested dicts are flattened with "/" for W&B).
            step: The step (e.g. the epoch number).
        """
        with open(self.metrics, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": step, **metrics}) + "\n")
        if self.wb is not None:
            try:
                self.wb.log(_flat(metrics), step=step)
            except Exception as exc:
                self._note(exc)

    def finish(self, summary: Mapping[str, Any]) -> None:
        """Write `summary.json`, set it as the W&B run summary and close the W&B run.

        Args:
            summary: Final values, e.g. `{"best_epoch": 20, "best_val_mean": 0.40}`.
        """
        (self.folder / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        if self.wb is not None:
            try:
                self.wb.summary.update(_flat(summary))
                self.wb.finish()
            except Exception as exc:
                self._note(exc)

    def _note(self, exc: Exception) -> None:
        """Record a W&B failure in `provenance.json` and stop mirroring."""
        self.prov["wandb_error"] = repr(exc)
        (self.folder / "provenance.json").write_text(json.dumps(self.prov, indent=1) + "\n", encoding="utf-8")
        self.wb = None


def _flat(d: Mapping[str, Any], prefix: str = "") -> dict:
    """Flatten nested dicts into "a/b" keys for W&B; lists and None are skipped.

    Examples:
        >>> _flat({"val": {"names": 0.4}, "loss": 1.2, "x": None})
        {'val/names': 0.4, 'loss': 1.2}
    """
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(_flat(v, key + "/"))
        elif isinstance(v, (int, float, str, bool)):
            out[key] = v
    return out
