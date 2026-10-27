"""English → Khmer phonetic transliteration: the standalone API of experiment E02.

Given one English word (or a name meant to be read as English), return Khmer script that approximates its
English pronunciation. It keeps sound, not meaning: `coffee` is spelled as heard, not translated.

This API is separate from the legacy `khmer_translit.translit` functions (`guess`, `guess_many`, ...), which keep
serving the speech front-end with models B and E. Nothing here routes to those models or reads the speech
repository: the model is loaded explicitly from a local file, and inference needs only NumPy (plus the local
CMUdict file for phone-input models).

Example (with a trained model on disk):

    >>> from khmer_translit.english import Transliterator              # doctest: +SKIP
    >>> tl = Transliterator.from_pretrained("models/e02/letters_s0/model.npz")   # doctest: +SKIP
    >>> r = tl.predict("coffee", top_k=2)                              # doctest: +SKIP
    >>> r.status, [c.text for c in r.candidates]                       # doctest: +SKIP
    ('ok', [...])

Scores are model log-probabilities of whole spellings (≤ 0). They rank candidates; they are **not** calibrated
probabilities of being correct.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from khmer_translit import phonetic_source as ps
from khmer_translit.translit import Transliterator as _NumpyModel

FORMAT = "khmer-translit-english-v1"
MAX_TOP_K = 8


@dataclass(frozen=True)
class Candidate:
    """One candidate spelling.

    Attributes:
        text: Khmer spelling (NFC).
        log_score: Model log-probability of the whole spelling (not a calibrated correctness probability).
    """

    text: str
    log_score: float


@dataclass
class PredictionResult:
    """The result for one input word.

    Attributes:
        input: The input as given.
        normalized: The normalised word ("" if refused before normalisation).
        status: "ok", or why there is no prediction: "empty", "unsupported_multiword", "unsupported_characters",
            "too_long", "unknown_phone", or "no_candidate" (the search finished no spelling).
        candidates: Distinct spellings, best first (empty unless status is "ok").
        pronunciation: The pronunciation used (`PronunciationResult.to_dict()`), None for letters-only models or
            refused inputs.
        message: Human-readable detail for a refused input.
    """

    input: str
    normalized: str
    status: str
    candidates: list[Candidate] = field(default_factory=list)
    pronunciation: dict | None = None
    message: str = ""

    @property
    def text(self) -> str:
        """The best spelling, or "" if there is none."""
        return self.candidates[0].text if self.candidates else ""

    def to_dict(self) -> dict:
        """JSON-ready form."""
        return {"input": self.input, "normalized": self.normalized, "status": self.status,
                "candidates": [{"text": c.text, "log_score": c.log_score} for c in self.candidates],
                "pronunciation": self.pronunciation, "message": self.message}


class ModelFormatError(ValueError):
    """A model file that this code cannot load correctly (wrong format, encoder version or vocabulary)."""


class Transliterator:
    """A loaded E02 model.

    Args:
        model: The NumPy encoder-decoder.
        meta: The model's metadata (`__meta__` of the `.npz`).
        cmudict_path: CMUdict file for phone-input models.
    """

    def __init__(self, model: _NumpyModel, meta: dict, cmudict_path: Path = ps.CMUDICT) -> None:
        self.model, self.meta, self.cmudict_path = model, meta, Path(cmudict_path)
        self.mode = meta["mode"]
        self.vocab = ps.SourceVocab([t[3:] for t in meta["src_tokens"] if t.startswith("km:")])
        self.beam = int(meta.get("beam", 4))

    @classmethod
    def from_pretrained(cls, path: str | Path, cmudict_path: str | Path = ps.CMUDICT,
                        expected_sha256: str | None = None) -> "Transliterator":
        """Load a model exported by `scripts/english_phonetic_train.py export`.

        Args:
            path: The `model.npz` file, or a folder holding it.
            cmudict_path: CMUdict file (phone-input models only; checked against the hash the model records).
            expected_sha256: If given, the file's sha256 must match.

        Returns:
            The transliterator.

        Raises:
            FileNotFoundError: Missing model file (or CMUdict for a phone-input model).
            ModelFormatError: Checksum, format, encoder version or vocabulary mismatch.
        """
        path = Path(path)
        if path.is_dir():
            path = path / "model.npz"
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        if expected_sha256 and ps.sha256_file(path) != expected_sha256:
            raise ModelFormatError(f"{path}: sha256 differs from the expected {expected_sha256}")
        try:
            data = np.load(path)
            meta = json.loads(str(data["__meta__"]))
        except Exception as e:   # corrupt zip, missing meta, bad JSON
            raise ModelFormatError(f"{path}: not a readable model file ({e})") from None
        if meta.get("format") != FORMAT:
            raise ModelFormatError(f"{path}: format {meta.get('format')!r}, expected {FORMAT!r}")
        if meta.get("encoder_version") != ps.ENCODER_VERSION:
            raise ModelFormatError(f"{path}: encoder {meta.get('encoder_version')!r}, code has {ps.ENCODER_VERSION!r}")
        vocab = ps.SourceVocab([t[3:] for t in meta["src_tokens"] if t.startswith("km:")])
        if vocab.tokens != meta["src_tokens"]:
            raise ModelFormatError(f"{path}: source vocabulary does not match this code's English tokens")
        if meta["mode"] not in ps.MODES:
            raise ModelFormatError(f"{path}: unknown mode {meta['mode']!r}")
        weights = {k: data[k].astype(np.float64) for k in data.files if k != "__meta__"}
        if weights["src_emb.weight"].shape[0] != len(meta["src_tokens"]):
            raise ModelFormatError(f"{path}: embedding size does not match the vocabulary")
        if meta["mode"] == "phones":
            cmu = Path(cmudict_path)
            if not cmu.exists():
                raise FileNotFoundError(f"this phone-input model needs CMUdict (sha256 "
                                        f"{(meta.get('pronunciation') or {}).get('cmudict_sha256', '?')}); not found "
                                        f"at {cmu}. Pass --cmudict PATH (API: cmudict_path=) or set "
                                        f"KHMER_TRANSLIT_CMUDICT")
            want = (meta.get("pronunciation") or {}).get("cmudict_sha256")
            if want and ps.sha256_file(cmu) != want:
                raise ModelFormatError(f"CMUdict at {cmu} differs from the one the model was trained with")
        model = _NumpyModel(weights, "", meta["tgt_vocab"], meta["heads"], meta["layers"], False, meta["max_src"],
                            False)
        return cls(model, meta, Path(cmudict_path))

    def _pronunciation(self, word: str, pronunciation: str | None) -> ps.PronunciationResult | None:
        """The pronunciation for a phone-input model (None for letters-only)."""
        if self.mode == "letters":
            return None
        return ps.get_pronunciation(word, supplied=pronunciation, path=self.cmudict_path)

    def encode_ids(self, word: str, pronunciation: str | None = None) -> tuple[list[int], ps.PronunciationResult | None]:
        """Source ids for one normalised word, exactly as in training.

        Raises:
            InputError: If the word or the supplied pronunciation is refused.
        """
        pron = self._pronunciation(word, pronunciation)
        return ps.encode_source(word, pron, self.mode, self.vocab), pron

    def next_logprobs(self, src_ids: list[int], prefix: list[int]) -> np.ndarray:
        """Log-probabilities of the next Khmer token after `prefix` (BOS first): for export parity checks."""
        memory, pad = self.model.encode_ids([src_ids])
        return self.model.decode_step(memory, pad, np.array([prefix]))[0]

    def predict(self, word: str, *, pronunciation: str | None = None, top_k: int = 1,
                beam: int | None = None) -> PredictionResult:
        """Transliterate one word.

        Args:
            word: One English word or name.
            pronunciation: Optional ARPAbet reading (e.g. "R EH1 K ER0 D"); used by phone-input models instead of
                the dictionary default. A letters-only model ignores it and reports `pronunciation: None`.
            top_k: Number of distinct candidates (1 to `MAX_TOP_K`).
            beam: Beam width (default: the model's, 4; at least `top_k`).

        Returns:
            A `PredictionResult`; refused inputs have a status other than "ok" and no candidates.

        Raises:
            ValueError: If `top_k` or `beam` is out of range.
        """
        return self.predict_many([word], top_k=top_k, beam=beam, pronunciations=[pronunciation])[0]

    def predict_many(self, words: Sequence[str], *, top_k: int = 1, beam: int | None = None,
                     pronunciations: Sequence[str | None] | None = None) -> list[PredictionResult]:
        """Transliterate several words (one batched beam search); results in input order.

        Args:
            words: Words.
            top_k: As `predict`.
            beam: As `predict`.
            pronunciations: Optional per-word ARPAbet readings (None entries use the default).

        Returns:
            One `PredictionResult` per word.

        Raises:
            ValueError: If `top_k` or `beam` is out of range, or `pronunciations` has the wrong length.
        """
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        beam = self.beam if beam is None else beam
        if not 1 <= beam <= 16:
            raise ValueError("beam must be between 1 and 16")
        beam = max(beam, top_k)
        pronunciations = list(pronunciations) if pronunciations is not None else [None] * len(words)
        if len(pronunciations) != len(words):
            raise ValueError("pronunciations must have one entry per word")
        results: list[PredictionResult] = []
        todo: list[tuple[int, list[int]]] = []
        for i, (raw, sup) in enumerate(zip(words, pronunciations)):
            w, status = ps.normalize_word(raw)
            if status != "ok":
                results.append(PredictionResult(raw, w, status, message=f"input refused: {status}"))
                continue
            try:
                ids, pron = self.encode_ids(w, sup)
            except ps.InputError as e:
                results.append(PredictionResult(raw, w, e.status, message=str(e)))
                continue
            results.append(PredictionResult(raw, w, "ok", pronunciation=pron.to_dict() if pron else None))
            todo.append((i, ids))
        if todo:
            memory, pad = self.model.encode_ids([ids for _i, ids in todo])
            for (i, _ids), nb in zip(todo, self.model.beam_search(memory, pad, beam)):
                seen, cands = set(), []
                for text, score in nb:
                    text = ps.nfc(text)
                    if text and text not in seen:
                        seen.add(text)
                        cands.append(Candidate(text, float(score)))
                results[i].candidates = cands[:top_k]
                if not cands:   # the search ended no hypothesis within MAX_TGT characters
                    results[i].status, results[i].message = "no_candidate", "no finished spelling within the length limit"
        return results

    def info(self) -> dict:
        """Model metadata plus parameter count (for `khmer-translit inspect`)."""
        n = sum(int(v.size) for v in self.model.w.values())
        return {**{k: v for k, v in self.meta.items() if k not in ("src_tokens", "tgt_vocab")},
                "parameters": n, "n_src_tokens": len(self.meta["src_tokens"]), "n_tgt_chars": len(self.meta["tgt_vocab"])}
