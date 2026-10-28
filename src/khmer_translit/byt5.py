"""ByT5 backend (research preview): English word → Khmer phonetic spelling with ByT5-small + a LoRA adapter.

A separate, explicitly selected profile. The historical E02 NumPy model (`khmer_translit.english`) stays the
package default, and the legacy `khmer_translit.translit` functions are untouched. Needs the optional extra
`khmer-translit[byt5]` (PyTorch, Transformers, safetensors).

What is loaded:
- **Base:** `google/byt5-small` at a pinned revision. Pass a local folder holding `config.json` and
  `pytorch_model.bin`, or let Transformers resolve the Hub id from its cache. The download happens on first use
  only, and never with `offline=True`.
- **Adapter:** a folder with `adapter.safetensors` (the LoRA A/B tensors) and `adapter_config.json`, which records
  the format, the base revision, the adapter targets, the decoding settings and provenance.

Decoding:
- **Single answer (`top_k=1`):** greedy, the E09/E10 frozen setting.
- **Candidate lists (`top_k > 1`):** canonical beam search, `early_stopping="never"` and `length_penalty=0`, with
  width `max(top_k, candidate_beam)`. E09 showed that `early_stopping=True` truncates byte-level beams.
- **Scores:** summed log-probabilities. They rank candidates and are **not** calibrated confidence.

Example (needs the adapter folder and a cached or local base):

    >>> from khmer_translit.byt5 import ByT5Transliterator                    # doctest: +SKIP
    >>> tl = ByT5Transliterator.from_pretrained("adapter/", base="models/e06/byt5-small", offline=True)  # doctest: +SKIP
    >>> tl.predict("coffee").text                                            # doctest: +SKIP
    'កាហ្វេ'
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

from khmer_translit import phonetic_source as ps
from khmer_translit.english import MAX_TOP_K, Candidate, ModelFormatError, PredictionResult

FORMAT = "khmer-translit-byt5-adapter-v1"
BASE_REPO = "google/byt5-small"
BASE_REVISION = "68377bdc18a2ffec8a0533fef03b1c513a4dd49d"
OFFSET, PAD, EOS = 3, 0, 1                  # ByT5 ids: byte b → b + 3; pad 0 (also decoder start), eos 1
MAX_SRC_BYTES = 32                           # longer inputs are refused ("too_long"), never cut
MAX_NEW_TOKENS = 121                         # every training target (≤ 120 bytes) plus EOS
ADAPTER_FILE, CONFIG_FILE = "adapter.safetensors", "adapter_config.json"
BASE_WEIGHTS = ("pytorch_model.bin", "model.safetensors")   # the pinned revision ships pytorch_model.bin


class MissingDependencyError(ImportError):
    """The optional `byt5` extra (torch, transformers, safetensors) is not installed."""


def _deps():
    try:
        import torch
        import transformers  # noqa: F401
        import safetensors.torch  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError("the ByT5 backend needs the optional extra: pip install 'khmer-translit[byt5]' "
                                     f"(missing: {e.name})") from None
    return torch


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encode(word: str) -> list[int]:
    """Normalised word → ByT5 input ids (UTF-8 bytes + 3, then EOS), exactly as in training.

    Examples:
        >>> encode("ab")
        [100, 101, 1]
    """
    return [b + OFFSET for b in word.encode("utf-8")] + [EOS]


def decode(ids: Sequence[int]) -> tuple[str | None, str]:
    """Generated ids (decoder start removed) → `(text, status)`: "ok", "empty", "malformed_utf8" or "no_eos".

    Examples:
        >>> decode([228, 161, 131, 1]), decode([228, 161, 131]), decode([1]), decode([228, 1])
        (('ក', 'ok'), (None, 'no_eos'), ('', 'empty'), (None, 'malformed_utf8'))
    """
    ids = list(ids)
    if EOS not in ids:
        return None, "no_eos"
    out = [i - OFFSET for i in ids[:ids.index(EOS)] if i >= OFFSET]
    if not out:
        return "", "empty"
    try:
        return ps.nfc(bytes(out).decode("utf-8")), "ok"
    except UnicodeDecodeError:
        return None, "malformed_utf8"


def _lora_cls():
    torch = _deps()
    from torch import nn

    class LoRALinear(nn.Module):
        """`base(x) + (alpha / r) · B(A(x))` on a frozen linear layer (inference only; no dropout)."""

        def __init__(self, base: nn.Linear, r: int, alpha: float):
            super().__init__()
            self.base = base
            self.A = nn.Parameter(torch.zeros(r, base.in_features), requires_grad=False)
            self.B = nn.Parameter(torch.zeros(base.out_features, r), requires_grad=False)
            self.scale = alpha / r

        @property
        def weight(self):                    # T5's gated FFN reads `wo.weight.dtype`
            return self.base.weight

        def forward(self, x):
            return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

    return LoRALinear


TARGETS = {"attn": ("q", "k", "v", "o"), "ffn": ("wi_0", "wi_1", "wo")}


class ByT5Transliterator:
    """A loaded ByT5 + LoRA model; load once per process and reuse.

    Args:
        model: The adapted `T5ForConditionalGeneration` (eval mode).
        config: The adapter's `adapter_config.json`.
        info: Load details (paths, hashes, threads).
    """

    def __init__(self, model, config: dict, info: dict) -> None:
        self.model, self.config, self._info = model, config, info
        dec = config["decode"]
        self.candidate_beam = int(dec["candidate_beam"])
        self.batch_size = int(dec.get("batch_size", 32))

    @classmethod
    def from_pretrained(cls, adapter: str | Path, base: str | Path | None = None, *, offline: bool | None = None,
                        threads: int | None = None, expected_sha256: str | None = None,
                        verify_base: bool = True) -> "ByT5Transliterator":
        """Load the base model and the adapter.

        Args:
            adapter: Folder with `adapter.safetensors` and `adapter_config.json`.
            base: Local base folder, or a Hub id (default `google/byt5-small` at the pinned revision, resolved
                from the Transformers cache or downloaded once unless offline).
            offline: Never download. Default: true if `HF_HUB_OFFLINE` or `KHMER_TRANSLIT_OFFLINE` is set.
            threads: CPU threads for PyTorch (default: PyTorch's choice).
            expected_sha256: If given, the adapter tensor file must have this sha256.
            verify_base: Check a local base folder's weights file against the sha256 the adapter records.

        Returns:
            The transliterator.

        Raises:
            MissingDependencyError: The `byt5` extra is not installed.
            FileNotFoundError: Missing adapter files, or a base that is neither a folder nor in the cache.
            ModelFormatError: Wrong format, checksum, base revision or adapter tensors.
        """
        torch = _deps()
        from safetensors.torch import load_file
        from transformers import T5ForConditionalGeneration
        adapter = Path(adapter)
        cfg_p, ten_p = adapter / CONFIG_FILE, adapter / ADAPTER_FILE
        for p in (cfg_p, ten_p):
            if not p.exists():
                raise FileNotFoundError(f"adapter file not found: {p} (an adapter folder holds {CONFIG_FILE} and "
                                        f"{ADAPTER_FILE})")
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        if cfg.get("format") != FORMAT:
            raise ModelFormatError(f"{cfg_p}: format {cfg.get('format')!r}, expected {FORMAT!r}")
        if cfg["base"]["revision"] != BASE_REVISION:
            raise ModelFormatError(f"adapter was trained on base revision {cfg['base']['revision']}, this code pins "
                                   f"{BASE_REVISION}")
        got = sha256_file(ten_p)
        if got != cfg["adapter_sha256"] or (expected_sha256 and got != expected_sha256):
            raise ModelFormatError(f"{ten_p}: sha256 {got} differs from the recorded/expected value")
        if offline is None:
            offline = bool(os.environ.get("HF_HUB_OFFLINE") or os.environ.get("KHMER_TRANSLIT_OFFLINE"))
        if threads:
            torch.set_num_threads(int(threads))
        src = Path(base) if base is not None and Path(base).is_dir() else None
        if src is not None:
            weights = next((src / f for f in BASE_WEIGHTS if (src / f).exists()), None)
            if not (src / "config.json").exists() or weights is None:
                raise FileNotFoundError(f"base folder {src} needs config.json and {' or '.join(BASE_WEIGHTS)} "
                                        f"({BASE_REPO} @ {BASE_REVISION})")
            if verify_base and sha256_file(weights) != cfg["base"]["weights_sha256"]:
                raise ModelFormatError(f"{weights} is not {BASE_REPO} @ {BASE_REVISION} (sha256 differs)")
            model = T5ForConditionalGeneration.from_pretrained(str(src), dtype=torch.float32, local_files_only=True)
            base_desc = str(src)
        else:
            repo = str(base) if base is not None else BASE_REPO
            try:
                model = T5ForConditionalGeneration.from_pretrained(repo, revision=BASE_REVISION, dtype=torch.float32,
                                                                   local_files_only=offline)
            except OSError as e:
                raise FileNotFoundError(f"base model {repo} @ {BASE_REVISION} is not available"
                                        f"{' offline (not in the cache)' if offline else ''}: pass --base FOLDER or "
                                        f"run once online to cache it ({e.__class__.__name__})") from None
            base_desc = f"{repo}@{BASE_REVISION}"
        L = _lora_cls()
        from transformers.models.t5.modeling_t5 import T5Attention, T5DenseGatedActDense
        a = cfg["adapter"]
        for p in model.parameters():
            p.requires_grad_(False)
        for m in list(model.modules()):
            names = (TARGETS["attn"] if isinstance(m, T5Attention) and "attn" in a["targets"] else
                     TARGETS["ffn"] if isinstance(m, T5DenseGatedActDense) and "ffn" in a["targets"] else ())
            for n in names:
                setattr(m, n, L(getattr(m, n), a["r"], a["alpha"]))
        sd = load_file(str(ten_p))
        want = {k for k in model.state_dict() if k.endswith((".A", ".B"))}
        if set(sd) != want:
            raise ModelFormatError(f"adapter tensors do not match the targets: {len(set(sd) - want)} unknown, "
                                   f"{len(want - set(sd))} missing")
        model.load_state_dict(sd, strict=False)
        model.eval()
        return cls(model, cfg, {"adapter": str(adapter), "adapter_sha256": got, "base": base_desc,
                                "offline": offline, "threads": torch.get_num_threads()})

    # ------------------------------------------------------------------ inference
    def _generate(self, srcs: list[list[int]], beam: int) -> list[list[tuple[str | None, str, float]]]:
        torch = _deps()
        out = []
        bs = self.batch_size if beam == 1 else max(1, self.batch_size // 2)
        with torch.no_grad():
            for i in range(0, len(srcs), bs):
                chunk = srcs[i:i + bs]
                S = max(map(len, chunk))
                ids = torch.tensor([s + [PAD] * (S - len(s)) for s in chunk])
                kw = (dict(num_beams=beam, num_return_sequences=beam, length_penalty=0.0, early_stopping="never")
                      if beam > 1 else dict(num_beams=1))
                g = self.model.generate(input_ids=ids, attention_mask=(ids != PAD).long(),
                                        max_new_tokens=MAX_NEW_TOKENS, do_sample=False, output_scores=True,
                                        return_dict_in_generate=True, **kw)
                seqs = g.sequences.tolist()
                if beam > 1:
                    scores = g.sequences_scores.tolist()
                else:
                    lp = torch.stack([torch.log_softmax(s, -1) for s in g.scores], 1)
                    tok = g.sequences[:, 1:]
                    chosen = lp.gather(2, tok.unsqueeze(-1)).squeeze(-1)
                    alive = torch.ones_like(chosen, dtype=torch.bool)
                    alive[:, 1:] = (tok[:, :-1] != EOS).cumprod(1).bool()
                    scores = (chosen * alive).sum(1).tolist()
                for n in range(len(chunk)):
                    res = []
                    for j in range(beam):
                        raw = seqs[n * beam + j][1:]
                        if EOS in raw:
                            raw = raw[:raw.index(EOS) + 1]
                        t, st = decode(raw)
                        res.append((t, st, round(float(scores[n * beam + j]), 6)))
                    out.append(res)
        return out

    def predict(self, word: str, *, top_k: int = 1) -> PredictionResult:
        """Transliterate one word (see `predict_many`)."""
        return self.predict_many([word], top_k=top_k)[0]

    def predict_many(self, words: Sequence[str], *, top_k: int = 1) -> list[PredictionResult]:
        """Transliterate independent words; results in input order, one per input.

        Args:
            words: English words or names, one per entry (multi-word entries are refused, not split).
            top_k: 1 → the greedy single answer; 2..8 → up to `top_k` distinct canonical-beam candidates.

        Returns:
            `PredictionResult`s; refused inputs carry a status ("empty", "unsupported_multiword",
            "unsupported_characters", "too_long") and no candidates; "no_candidate" if decoding finished nothing.

        Raises:
            ValueError: `top_k` out of range.
        """
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        beam = 1 if top_k == 1 else max(top_k, self.candidate_beam)
        results: list[PredictionResult] = []
        todo: list[tuple[int, list[int]]] = []
        for i, raw in enumerate(words):
            w, status = ps.normalize_word(raw)
            if status == "ok" and len(w.encode("utf-8")) > MAX_SRC_BYTES:
                status = "too_long"
            if status != "ok":
                results.append(PredictionResult(raw, w, status, message=f"input refused: {status}"))
                continue
            results.append(PredictionResult(raw, w, "ok"))
            todo.append((i, encode(w)))
        if todo:
            for (i, _), cands in zip(todo, self._generate([s for _, s in todo], beam)):
                seen, keep = set(), []
                for t, st, sc in cands:
                    if st == "ok" and t and t not in seen:
                        seen.add(t)
                        keep.append(Candidate(t, sc))
                results[i].candidates = keep[:top_k]
                if not keep:
                    results[i].status = "no_candidate"
                    results[i].message = f"no well-formed spelling finished ({', '.join(sorted({s for _, s, _ in cands}))})"
        return results

    def info(self) -> dict:
        """Adapter metadata, load details and the decoding settings (for `khmer-translit inspect`)."""
        return {**{k: v for k, v in self.config.items()}, "loaded": self._info,
                "decoding": {"top_k=1": "greedy", "top_k>1": f"canonical beam (early_stopping='never', "
                             f"length_penalty 0), width max(top_k, {self.candidate_beam})",
                             "max_new_tokens": MAX_NEW_TOKENS, "max_input_bytes": MAX_SRC_BYTES,
                             "scores": "summed log-probabilities; not calibrated confidence"}}


def export_adapter(state: dict, out: Path, config: dict) -> dict:
    """Write an adapter folder from LoRA tensors (`{name: tensor}` ending in .A/.B) and a config (without hash).

    Returns the written config (with `adapter_sha256`)."""
    _deps()
    from safetensors.torch import save_file
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in sorted(state.items())}, str(out / ADAPTER_FILE))
    cfg = {**config, "format": FORMAT, "adapter_sha256": sha256_file(out / ADAPTER_FILE)}
    (out / CONFIG_FILE).write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return cfg


__all__ = ["ByT5Transliterator", "MissingDependencyError", "export_adapter", "encode", "decode", "FORMAT",
           "BASE_REPO", "BASE_REVISION", "MAX_SRC_BYTES", "MAX_NEW_TOKENS"]
