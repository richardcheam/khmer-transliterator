"""Download the published model files from Hugging Face, pinned to one commit and checked by sha256.

Needs the optional extra `khmer-translit[hub]` (huggingface_hub; the `byt5` extra includes it). Nothing is
downloaded at import or install time; files are fetched on the first call and then read from the local cache.

    >>> from khmer_translit import hub                    # doctest: +SKIP
    >>> hub.download("small")                             # doctest: +SKIP  (the folder holding model.npz)

The model files are licensed CC BY-SA 4.0 (see each model repository), separately from this package's MIT code.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

MODELS = {
    "small": {
        "repo": "ric912/en-km-phonetic-transliterator",
        "revision": "beed6741281de7fbe698e5fce2151f4a0f80bc49",
        "files": {"model.npz": "8901050b1093cf90f60523c7941d6c12dbd8571d71698e37abcf874ba437904b"},
    },
    "byt5": {
        "repo": "ric912/en-km-phonetic-byt5-lora",
        "revision": "8e5f42353a607268adeeae29c8696ae276f358b1",
        "files": {"adapter.safetensors": "698a23ae1526d3da8f3b2bd00f0177799a939d2005c165569fcba7d58e76cc3f",
                  "adapter_config.json": "1424c1422603413ea71ad891d255a9ef865de8f5aec5676e182f1d6481363309"},
    },
}
NOTICES = ("LICENSE", "README.md")


class IntegrityError(RuntimeError):
    """A downloaded model file does not have the expected sha256."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(folder: str | Path, files: dict[str, str]) -> Path:
    """Check that `folder` holds every file with its expected sha256.

    Raises:
        FileNotFoundError: A file is missing.
        IntegrityError: A file's sha256 differs.
    """
    folder = Path(folder)
    for name, want in files.items():
        p = folder / name
        if not p.exists():
            raise FileNotFoundError(f"model file missing: {p}")
        got = sha256_file(p)
        if got != want:
            raise IntegrityError(f"{p}: sha256 {got} differs from the expected {want}")
    return folder


def download(model: str = "small", cache_dir: str | Path | None = None, local_files_only: bool = False) -> Path:
    """The local folder of one published model (`"small"` or `"byt5"`), downloading it on first use.

    Args:
        model: "small" (model.npz) or "byt5" (the LoRA adapter; the base model is fetched separately).
        cache_dir: Hugging Face cache folder (default: the user's standard cache).
        local_files_only: Use only files already in the cache (no network).

    Returns:
        The verified folder.

    Raises:
        KeyError: Unknown model name.
        ImportError: huggingface_hub is not installed (`pip install "khmer-translit[hub]"`).
        IntegrityError: A downloaded file has an unexpected sha256.
    """
    spec = MODELS[model]
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError('downloading the model needs huggingface_hub: pip install "khmer-translit[hub]" '
                          "(or download the files yourself and pass their path)") from e
    folder = snapshot_download(repo_id=spec["repo"], revision=spec["revision"], cache_dir=cache_dir,
                               local_files_only=local_files_only, allow_patterns=[*spec["files"], *NOTICES])
    return verify(folder, spec["files"])
