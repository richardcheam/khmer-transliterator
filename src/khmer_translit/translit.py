"""Learned Latin → Khmer transliterator for romanised Khmer words (V02-N1).

Romanised Khmer (sophea, chanthou, panha) follows Khmer conventions (ph = ភ, ch = ច, nh = ញ, ea = ា) that the English
sound-out rules in `latin_g2p` get wrong. This module holds a small character-level transformer (encoder-decoder)
trained by `scripts/translit_train.py` on ~24k romanised-Khmer word pairs (`scripts/translit_data.py`).

**Inference is pure NumPy** (no torch at serving time): `load()` reads the exported weights (`.npz`) and
`Transliterator.nbest()` runs beam search. The NumPy forward pass mirrors the torch model in
`scripts/translit_train.py` exactly (pre-LayerNorm blocks, learned positions, tied nothing); a test compares the two.

**Shapes:** d = model width, Latin input ≤ `MAX_IN` letters, Khmer output ≤ `MAX_OUT` code points. Token 0 = pad,
1 = BOS, 2 = EOS; the rest are characters (`src_vocab`, `tgt_vocab` stored in the weights file).
"""
from __future__ import annotations

from collections import OrderedDict

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

MAX_IN = 32   # source length of letters-only models (arms A, B); a phone-aware model stores its own `max_in`
PHONES = ("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z "
          "ZH").split()
SEP = "|"
TAGS = {"en": "\ue200", "other": "\ue201"}   # group tags (D149): one private-use character before the letters
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"       # word lists, pairs and raw sources (not in git)
MODELS = ROOT / "models"   # trained models, one folder each with `translit.npz` (not in git)
EN_WORDS = DATA / "en_common_words.txt"


@lru_cache(maxsize=1)
def _en_words() -> frozenset[str]:
    """The lower-case English word list used by `word_tag` (SCOWL, `data/SOURCES.md`).

    Returns:
        The words; empty if the file is missing.
    """
    try:
        return frozenset(EN_WORDS.read_text(encoding="utf-8").split())
    except FileNotFoundError:
        return frozenset()


def word_tag(word: str) -> str:
    """The group tag a tag-aware model gets at inference (D149): "en" for ordinary English words, else "other".

    A word counts as English when it is a lower-case entry of the SCOWL list, where proper nouns are capitalised.
    On the 240 labelled D149 words this sent 82 of 126 English words and no foreign name to "en".

    Args:
        word: A Latin word (any case).

    Returns:
        "en" or "other".

    Examples:
        >>> word_tag("Shadow"), word_tag("joaquin")
        ('en', 'other')
    """
    return "en" if word.lower() in _en_words() else "other"


def source(word: str, use_phones: bool, tag: str | None = None) -> str:
    """The model's source string for a word: an optional group tag, its letters, and for a phone-aware model (D146)
    a separator plus its CMUdict pronunciation, one private-use character per phone and per stress digit.

    Words without a CMUdict entry get the separator only, so the model learns to rely on the letters for them.

    Args:
        word: A Latin word (any case).
        use_phones: Append the pronunciation.
        tag: None (no tag; arms A–E) or a key of `TAGS` ("en" / "other"), prepended as one character (D149).

    Returns:
        The source string.

    Raises:
        KeyError: If `tag` is not None and not a key of `TAGS`.

    Examples:
        >>> source("One", False)
        'one'
        >>> s = source("one", True); s[:4], len(s)
        ('one|', 8)
        >>> source("one", False, "en") == TAGS["en"] + "one"
        True
    """
    w = word.lower()
    if tag is not None:
        return TAGS[tag] + source(w, use_phones)
    if not use_phones:
        return w
    from khmer_translit import en_g2p
    ph = en_g2p.phones(w) or []
    enc = []
    for p in ph:
        enc.append(chr(0xE000 + PHONES.index(p.rstrip("012"))))
        if p[-1].isdigit():
            enc.append(chr(0xE100 + int(p[-1])))
    return w + SEP + "".join(enc)
MAX_OUT = 40
PAD, BOS, EOS = 0, 1, 2
DEFAULT_WEIGHTS = MODELS / "model" / "translit.npz"


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """LayerNorm over the last axis (matches `torch.nn.LayerNorm`).

    Args:
        x: Array `(..., d)`.
        w: Scale `(d,)`.
        b: Shift `(d,)`.
        eps: Variance floor (torch default 1e-5).

    Returns:
        Normalised array, same shape as `x`.

    Examples:
        >>> import numpy as np
        >>> _layer_norm(np.array([[1.0, 3.0]]), np.ones(2), np.zeros(2)).round(3).tolist()
        [[-1.0, 1.0]]
    """
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis.

    Args:
        x: Scores; `-inf` entries get probability 0.

    Returns:
        Probabilities, same shape.

    Examples:
        >>> import numpy as np
        >>> _softmax(np.array([0.0, 0.0])).tolist()
        [0.5, 0.5]
    """
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


@dataclass
class Transliterator:
    """A loaded model: weight arrays plus vocabularies and sizes.

    Attributes:
        w: Parameter name → array (torch `state_dict` names, see `scripts/translit_train.py`).
        src_vocab: Latin characters; index i + 3 is the token id.
        tgt_vocab: Khmer characters; index i + 3 is the token id.
        heads: Attention heads.
        layers: Encoder layers = decoder layers.
        phones: Whether the source carries the CMUdict pronunciation (`source`, D146).
        tags: Whether the source starts with a group tag chosen by `word_tag` (D149).
        max_in: Longest source in tokens (learned positions).
    """

    w: dict[str, np.ndarray]
    src_vocab: str
    tgt_vocab: str
    heads: int
    layers: int
    phones: bool = False
    max_in: int = MAX_IN
    tags: bool = False

    def _linear(self, x: np.ndarray, name: str) -> np.ndarray:
        """`x @ W.T + b` for the torch Linear `name`.

        Args:
            x: Input `(..., in)`.
            name: Parameter prefix, e.g. "enc.0.ff1".

        Returns:
            Output `(..., out)`.
        """
        return x @ self.w[name + ".weight"].T + self.w[name + ".bias"]

    def _attn(self, q_in: np.ndarray, kv_in: np.ndarray, name: str, mask: np.ndarray | None) -> np.ndarray:
        """Multi-head attention (separate q/k/v/o projections).

        Args:
            q_in: Queries `(B, Tq, d)`.
            kv_in: Keys/values source `(B, Tk, d)`.
            name: Parameter prefix, e.g. "dec.1.cross".
            mask: Boolean `(B or 1, Tq or 1, Tk)`; True = blocked. None = no mask.

        Returns:
            `(B, Tq, d)`.
        """
        B, Tq, d = q_in.shape
        h, dh = self.heads, d // self.heads
        q = self._linear(q_in, name + ".q").reshape(B, Tq, h, dh).transpose(0, 2, 1, 3)
        k = self._linear(kv_in, name + ".k").reshape(B, -1, h, dh).transpose(0, 2, 1, 3)
        v = self._linear(kv_in, name + ".v").reshape(B, -1, h, dh).transpose(0, 2, 1, 3)
        s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)
        if mask is not None:
            s = np.where(mask[:, None], -1e9, s)
        out = (_softmax(s) @ v).transpose(0, 2, 1, 3).reshape(B, Tq, d)
        return self._linear(out, name + ".o")

    def _ff(self, x: np.ndarray, name: str) -> np.ndarray:
        """Position-wise feed-forward with GELU (tanh-free erf form, as torch default).

        Args:
            x: `(..., d)`.
            name: Block prefix, e.g. "enc.0".

        Returns:
            `(..., d)`.
        """
        from math import sqrt
        hid = self._linear(x, name + ".ff1")
        hid = 0.5 * hid * (1.0 + _erf(hid / sqrt(2.0)))
        return self._linear(hid, name + ".ff2")

    def encode(self, words: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Encode a batch of lower-case Latin words.

        Args:
            words: Lower-case Latin words; each becomes `source(word, self.phones, tag)` (tag from `word_tag`
                when `self.tags`), cut to `max_in` tokens
                (characters outside `src_vocab` are skipped).

        Returns:
            `(memory, pad_mask)`: `(B, T, d)` encoder states and `(B, T)` True where padded.
        """
        ids = [[self.src_vocab.index(c) + 3 for c in source(w, self.phones, word_tag(w) if self.tags else None) if c in self.src_vocab][:self.max_in]
               for w in words]
        T = max(1, max(len(i) for i in ids))
        x = np.zeros((len(ids), T), dtype=np.int64)
        for r, i in enumerate(ids):
            x[r, :len(i)] = i
        pad = x == PAD
        h = self.w["src_emb.weight"][x] + self.w["src_pos.weight"][:T]
        for n in range(self.layers):
            p = f"enc.{n}"
            a = _layer_norm(h, self.w[p + ".ln1.weight"], self.w[p + ".ln1.bias"])
            h = h + self._attn(a, a, p + ".self", pad[:, None, :])
            h = h + self._ff(_layer_norm(h, self.w[p + ".ln2.weight"], self.w[p + ".ln2.bias"]), p)
        return _layer_norm(h, self.w["enc_ln.weight"], self.w["enc_ln.bias"]), pad

    def decode_step(self, memory: np.ndarray, pad: np.ndarray, prefix: np.ndarray) -> np.ndarray:
        """Log-probabilities of the next Khmer token for each prefix (full recompute; outputs are short).

        Args:
            memory: Encoder states `(B, T, d)`.
            pad: Encoder pad mask `(B, T)`.
            prefix: Token ids `(B, L)` starting with BOS.

        Returns:
            `(B, V)` log-probabilities.
        """
        L = prefix.shape[1]
        h = self.w["tgt_emb.weight"][prefix] + self.w["tgt_pos.weight"][:L]
        causal = np.triu(np.ones((L, L), dtype=bool), 1)[None]
        for n in range(self.layers):
            p = f"dec.{n}"
            a = _layer_norm(h, self.w[p + ".ln1.weight"], self.w[p + ".ln1.bias"])
            h = h + self._attn(a, a, p + ".self", causal)
            a = _layer_norm(h, self.w[p + ".ln2.weight"], self.w[p + ".ln2.bias"])
            h = h + self._attn(a, memory, p + ".cross", pad[:, None, :])
            h = h + self._ff(_layer_norm(h, self.w[p + ".ln3.weight"], self.w[p + ".ln3.bias"]), p)
        h = _layer_norm(h[:, -1], self.w["dec_ln.weight"], self.w["dec_ln.bias"])
        logits = self._linear(h, "out")
        shifted = logits - logits.max(-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))

    def nbest(self, word: str, beam: int = 4) -> list[tuple[str, float]]:
        """Beam-search the best Khmer spellings for one Latin word (`nbest_batch([word], beam)[0]`).

        Args:
            word: A Latin word (any case; non-letters are ignored).
            beam: Beam width (default 4); also the number of results.

        Returns:
            Up to `beam` `(khmer, logprob)` pairs, best first. `logprob` is the total log-probability (≤ 0);
            `exp(logprob)` near 1 means a confident reading.

        Examples:
            >>> from khmer_translit import translit
            >>> model = translit.load()          # doctest: +SKIP
            >>> model.nbest("sophea")[0][0]      # doctest: +SKIP
            'សុភា'
        """
        return self.nbest_batch([word], beam)[0]

    def _heads(self, x: np.ndarray) -> np.ndarray:
        """Split the last axis into attention heads: `(R, T, d)` → `(R, h, T, d / h)`.

        Args:
            x: Projected keys, values or queries `(R, T, d)`.

        Returns:
            The per-head view.
        """
        R, T, d = x.shape
        return x.reshape(R, T, self.heads, d // self.heads).transpose(0, 2, 1, 3)

    def _cached_attn(self, q_in: np.ndarray, k: np.ndarray, v: np.ndarray, name: str,
                     mask: np.ndarray | None) -> np.ndarray:
        """Attention of one new position over precomputed per-head keys/values.

        Args:
            q_in: Normalised input of the new position `(R, 1, d)`.
            k: Keys `(R, h, T, dh)`.
            v: Values `(R, h, T, dh)`.
            name: Parameter prefix of the attention block (its q and o projections are used).
            mask: Boolean `(R, T)`, True = blocked; None = attend to all.

        Returns:
            `(R, 1, d)`.
        """
        R = q_in.shape[0]
        q = self._heads(self._linear(q_in, name + ".q"))
        s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(k.shape[-1])
        if mask is not None:
            s = np.where(mask[:, None, None, :], -1e9, s)
        out = (_softmax(s) @ v).transpose(0, 2, 1, 3).reshape(R, 1, -1)
        return self._linear(out, name + ".o")

    def nbest_batch(self, words: list[str], beam: int = 4) -> list[list[tuple[str, float]]]:
        """Beam-search several words at once, with cached decoder keys/values (D148).

        All live beams of all words advance in lock-step and are decoded in one call per step. Each step runs the
        decoder for the **new** character only: earlier positions' self-attention keys/values are kept per beam
        (re-gathered when beams are re-ranked), and the cross-attention keys/values of each word are computed once.
        A word leaves the batch when its search is finished. Results equal a full recompute (`decode_step`) up to
        float rounding, and each word's result is what a search of that word alone gives.

        Args:
            words: Latin words (any case; non-letters are ignored). May be empty.
            beam: Beam width per word (default 4); also the number of results per word.

        Returns:
            One list per input word, as `nbest` returns for it.

        Examples:
            >>> from khmer_translit import translit
            >>> model = translit.load()                          # doctest: +SKIP
            >>> [r[0][0] for r in model.nbest_batch(["sophea", "dara"])]   # doctest: +SKIP
            ['សុភា', 'តារា']
        """
        if not words:
            return []
        memory, pad = self.encode([w.lower() for w in words])
        cross = [(self._heads(self._linear(memory, f"dec.{n}.cross.k")), self._heads(self._linear(memory, f"dec.{n}.cross.v")))
                 for n in range(self.layers)]
        d = memory.shape[-1]
        h, dh = self.heads, d // self.heads
        cache_k = [np.zeros((len(words), h, 0, dh)) for _ in range(self.layers)]
        cache_v = [np.zeros((len(words), h, 0, dh)) for _ in range(self.layers)]
        # one row per live beam: (word index, tokens, score); row r of the caches belongs to rows[r]
        rows: list[tuple[int, list[int], float]] = [(i, [BOS], 0.0) for i in range(len(words))]
        done: list[list[tuple[list[int], float]]] = [[] for _ in words]
        for step in range(MAX_OUT):
            src = np.array([r[0] for r in rows])
            x = self.w["tgt_emb.weight"][np.array([r[1][-1] for r in rows])][:, None] + self.w["tgt_pos.weight"][step]
            for n in range(self.layers):
                p = f"dec.{n}"
                a = _layer_norm(x, self.w[p + ".ln1.weight"], self.w[p + ".ln1.bias"])
                cache_k[n] = np.concatenate([cache_k[n], self._heads(self._linear(a, p + ".self.k"))], axis=2)
                cache_v[n] = np.concatenate([cache_v[n], self._heads(self._linear(a, p + ".self.v"))], axis=2)
                x = x + self._cached_attn(a, cache_k[n], cache_v[n], p + ".self", None)
                a = _layer_norm(x, self.w[p + ".ln2.weight"], self.w[p + ".ln2.bias"])
                x = x + self._cached_attn(a, cross[n][0][src], cross[n][1][src], p + ".cross", pad[src])
                x = x + self._ff(_layer_norm(x, self.w[p + ".ln3.weight"], self.w[p + ".ln3.bias"]), p)
            logits = self._linear(_layer_norm(x[:, 0], self.w["dec_ln.weight"], self.w["dec_ln.bias"]), "out")
            shifted = logits - logits.max(-1, keepdims=True)
            logp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
            cand: dict[int, list[tuple[list[int], float, int]]] = {}
            for r, ((i, toks, score), row) in enumerate(zip(rows, logp)):
                for t in np.argsort(row)[::-1][:beam]:
                    cand.setdefault(i, []).append((toks + [int(t)], score + float(row[t]), r))
            new_rows, parents = [], []
            for i, cs in cand.items():
                cs.sort(key=lambda c: c[1], reverse=True)
                kept = []
                for toks, score, r in cs:
                    if toks[-1] == EOS:
                        done[i].append((toks, score))
                    else:
                        kept.append((toks, score, r))
                    if len(kept) == beam:
                        break
                done[i].sort(key=lambda c: c[1], reverse=True)
                if kept and not (len(done[i]) >= beam and done[i][beam - 1][1] >= kept[0][1]):
                    new_rows += [(i, toks, score) for toks, score, _r in kept]
                    parents += [r for _t, _s, r in kept]
            if not new_rows:
                break
            rows = new_rows
            idx = np.array(parents)
            cache_k = [c[idx] for c in cache_k]
            cache_v = [c[idx] for c in cache_v]
        return [[("".join(self.tgt_vocab[t - 3] for t in toks[1:-1] if t >= 3), score) for toks, score in d_[:beam]]
                or [("", float("-inf"))] for d_ in done]

    def __call__(self, word: str) -> str:
        """Best Khmer spelling for one Latin word (`nbest(word)[0][0]`).

        Args:
            word: A Latin word.

        Returns:
            The Khmer spelling (may be empty for garbage input).

        Examples:
            >>> model = load()        # doctest: +SKIP
            >>> model("panha")        # doctest: +SKIP
            'បញ្ញា'
        """
        return self.nbest(word)[0][0]


def pick(nbest: list[tuple[str, float]], known: set[str] | frozenset[str], margin: float = 3.0) -> str:
    """Choose a spelling from the beam: the best-scoring **known Khmer word** within `margin` of the top, else the top.

    Romanisation is ambiguous (dara → ដរា, តារា, ដារា); a real word or attested name is usually the intended one.

    Args:
        nbest: `Transliterator.nbest()` output, best first.
        known: Known Khmer spellings (e.g. lexicon headwords plus training targets).
        margin: Largest log-probability gap from the top hypothesis that a known word may have (default 3.0,
            i.e. up to ~20× less likely).

    Returns:
        The chosen Khmer spelling.

    Examples:
        >>> pick([("ដរា", -0.78), ("តារា", -2.23)], {"តារា"})
        'តារា'
        >>> pick([("ដរា", -0.78), ("តារា", -9.0)], {"តារា"})
        'ដរា'
    """
    top = nbest[0][1]
    for word, score in nbest:
        if word in known and top - score <= margin:
            return word
    return nbest[0][0]


NON_KHMER_LETTERS = frozenset("fqwxz")   # each occurs in ≤ 9 of 24k romanised-Khmer training forms
MODEL_DIR = DEFAULT_WEIGHTS.parent
CMUDICT = DATA / "src" / "cmudict" / "cmudict.dict"
SERVED = {"model": "model_b", "router": "all", "en_model": "model_e"}   # what is served (D137, D150); see configure


def configure(model: str | None = None, router: str | None = None, en_model: str | None | bool = False) -> None:
    """Choose the models `guess` / `guess_many` use (the caller's serving config; defaults = `SERVED`).

    The speech front-end calls this once with the values from its `configs/serving.json`. Changing the choice
    clears the loaded models and the per-word cache.

    Args:
        model: Model folder under `models/` for names and romanised Khmer (e.g. "model_b"); None keeps the current.
        router: "all" (only f/q/w/x/z words go to the rules) or "cmudict" (also English dictionary words); None
            keeps the current.
        en_model: Model folder for English dictionary words (e.g. "model_e"); None or "" switches it off; False
            (the default) keeps the current.

    Raises:
        ValueError: On an unknown router.

    Example:
        `configure(model="model_b", router="all", en_model="model_e")`
    """
    if router is not None and router not in ("all", "cmudict"):
        raise ValueError(f"router must be 'all' or 'cmudict', not {router!r}")
    if model is not None:
        SERVED["model"] = model
    if router is not None:
        SERVED["router"] = router
    if en_model is not False:
        SERVED["en_model"] = en_model or None
    _resources.cache_clear()
    _en_model.cache_clear()
    _GUESS_CACHE.clear()


def route(word: str, english: set[str] | frozenset[str], romanised: set[str] | frozenset[str],
          mode: str = "cmudict") -> str:
    """Pick the route for one unknown Latin word: English sound-out rules or the learned model.

    Args:
        word: Lower-case Latin word.
        english: English dictionary words (CMUdict headwords).
        romanised: Latin forms seen as romanised Khmer in the training pairs.
        mode: "cmudict" (D133, arm A) or "all" (D137, arm B: the model also learned foreign names, so only
            `NON_KHMER_LETTERS` words keep the rules).

    Returns:
        "rules" for a word with a letter romanised Khmer does not use (`NON_KHMER_LETTERS`), or (mode "cmudict")
        for a dictionary English word that is not a known romanised-Khmer form; otherwise "model".

    Examples:
        >>> route("phone", {"phone", "dara"}, {"dara"}), route("dara", {"phone", "dara"}, {"dara"})
        ('rules', 'model')
        >>> route("sophea", set(), set()), route("zando", set(), set())
        ('model', 'rules')
        >>> route("phone", {"phone"}, set(), mode="all")
        'model'
    """
    if NON_KHMER_LETTERS & set(word):
        return "rules"
    if mode == "all":
        return "model"
    return "rules" if word in english and word not in romanised else "model"


def served_choice() -> tuple[Path, str]:
    """The configured model for names and romanised Khmer, and the router (`SERVED`, set by `configure`).

    Returns:
        `(weights path, router mode)`, e.g. `(models/model_b/translit.npz, "all")`.

    Examples:
        >>> served_choice()[1] in {"cmudict", "all"}
        True
    """
    return MODELS / SERVED["model"] / "translit.npz", SERVED["router"]


@lru_cache(maxsize=1)
def _resources() -> tuple[Transliterator, frozenset[str], frozenset[str], frozenset[str]]:
    """Load the served model and its word lists once: `(model, english, romanised, known)`.

    The model is the one `served_choice()` names (read once per process: restart the server after changing it).
    `known` = Google lexicon headwords + arm A's train/val Khmer targets (`known_targets.txt`), for `pick`; arm B was
    chosen with this same list (D137).

    Returns:
        The cached tuple.

    Raises:
        FileNotFoundError: If the model or a list is missing (`guess` then falls back to rules).
    """
    from khmer_translit import lexicon
    english = frozenset(line.split()[0].split("(")[0] for line in CMUDICT.read_text(encoding="utf-8").splitlines()
                        if line[:1].isalpha())
    romanised = frozenset((MODEL_DIR / "romanised_forms.txt").read_text(encoding="utf-8").split())
    known = frozenset((MODEL_DIR / "known_targets.txt").read_text(encoding="utf-8").split()) | lexicon.keys()
    return load(served_choice()[0]), english, romanised, known


@lru_cache(maxsize=1)
def _en_model() -> Transliterator | None:
    """The model for English dictionary words, if one is configured (`SERVED["en_model"]`, D150).

    Words that `word_tag` calls "en" get this model's spelling (+ `pick`) instead of the main route; every other
    word keeps the main path.

    Returns:
        The loaded model, or None when switched off.

    Raises:
        FileNotFoundError: If the folder has no `translit.npz`.
    """
    name = SERVED["en_model"]
    return load(MODELS / name / "translit.npz") if name else None


@lru_cache(maxsize=1)
def _name_counts() -> dict[str, int]:
    """Latin forms mined as names from web text, with document counts ≥ 5 (`models/model_b/name_counts.tsv`).

    Returns:
        Latin form → count.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    path = MODELS / "model_b" / "name_counts.tsv"
    rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    return {w: int(n) for w, n in rows}


def caps_reads_as_word(word: str) -> bool:
    """Whether an unlisted all-caps word (1–6 letters) is a name typed in capitals rather than an acronym (D139).

    Rule, checked on FineWeb-2 Khmer: of the 400 most frequent unlisted all-caps words, 8 are read as words (ADHOC,
    COVAX, SARS, TODAY, NET, SONG, LAN, SWIFT), and 20 of 24 common Khmer names are:
    - 1–2 letters, or 3 letters starting with a vowel → letters (UN, TV, ABA, USB);
    - no vowel, or 3+ consonants in a row → letters (CPP, HTML, UNDP);
    - 3 letters → a word only if consonant + vowel and mined as a name ≥ 10 times (SOK, HUN, KIM; not NGO, CEO);
    - 4–6 letters → a word only if a known romanised-Khmer form or mined as a name ≥ 5 times (DARA, SOPHEA).
    (y counts as a vowel.)

    Args:
        word: The word as written.

    Returns:
        True to read it as a word.

    Raises:
        FileNotFoundError: If the model word lists are missing.

    Examples:
        >>> caps_reads_as_word("ABA"), caps_reads_as_word("HTML"), caps_reads_as_word("UNDP")
        (False, False, False)
        >>> caps_reads_as_word("DARA"), caps_reads_as_word("SOK")
        (True, True)
    """
    w = word.lower()
    vowels = "aeiouy"
    if len(w) <= 2 or (len(w) == 3 and w[0] in vowels):
        return False
    if not any(c in vowels for c in w) or re.search(f"[^{vowels}]{{3}}", w):
        return False
    counts = _name_counts()
    if len(w) == 3:
        return w[0] not in vowels and w[1] in vowels and counts.get(w, 0) >= 10
    return w in _resources()[2] or counts.get(w, 0) >= 5


def guess(word: str) -> tuple[str, str]:
    """Khmer spelling for an unknown Latin word, by route: English dictionary words → the English-word model when
    configured (D150); else romanised Khmer → model (+ `pick`), else the rules.

    This is the front-end fallback after the loanword list, letter spelling and compound split (V02-N1).
    If the model files are missing, every word goes to the rules. Results are cached per lower-case word
    (`_guess_lower`), so a name repeated in a long text costs one beam search.

    Args:
        word: A Latin word (any case).

    Returns:
        `(khmer, route)` with route "model_en", "model" or "rules".

    Examples:
        >>> from khmer_translit import translit
        >>> translit.guess("Sophea")       # doctest: +SKIP
        ('សុភា', 'model')
        >>> translit.guess("zando")[1]
        'rules'
    """
    return _guess_lower(word.lower())


_GUESS_CACHE: OrderedDict[str, tuple[str, str]] = OrderedDict()
GUESS_CACHE_SIZE = 4096


def _remember(word: str, result: tuple[str, str]) -> tuple[str, str]:
    """Store one `guess` result in the bounded cache (oldest dropped first).

    Args:
        word: Lower-case Latin word.
        result: `(khmer, route)`.

    Returns:
        `result`, unchanged.

    Examples:
        >>> _remember("zzzexample", ("ហ្ស", "rules"))
        ('ហ្ស', 'rules')
    """
    _GUESS_CACHE[word] = result
    _GUESS_CACHE.move_to_end(word)
    while len(_GUESS_CACHE) > GUESS_CACHE_SIZE:
        _GUESS_CACHE.popitem(last=False)
    return result


def _guess_lower(word: str) -> tuple[str, str]:
    """Cached body of `guess` for a lower-case word (`guess_many([word])[0]`).

    Args:
        word: Lower-case Latin word.

    Returns:
        `(khmer, route)`, as `guess`.

    Examples:
        >>> _guess_lower("zando")[1]
        'rules'
    """
    return guess_many([word])[0]


def guess_many(words: list[str]) -> list[tuple[str, str]]:
    """`guess` for several words, with all model-routed, uncached words spelled in one batch (`nbest_batch`).

    The front-end calls this once per text before its word-by-word pass, so a sentence with several unknown
    English words costs one batched search instead of one search per word (D148). Results are cached per
    lower-case word, as for `guess`.

    Args:
        words: Latin words (any case; repeats allowed).

    Returns:
        `(khmer, route)` per input word, in input order. Route "model_en" = the English-word model (D150).

    Examples:
        >>> from khmer_translit import translit
        >>> [r[1] for r in translit.guess_many(["zando", "Zando"])]
        ['rules', 'rules']
    """
    from khmer_translit import latin_g2p
    lower = [w.lower() for w in words]
    found = {w: _GUESS_CACHE[w] for w in lower if w in _GUESS_CACHE}
    todo = list(dict.fromkeys(w for w in lower if w not in found))
    if todo:
        try:
            model, english, romanised, known = _resources()
        except FileNotFoundError:
            found.update((w, (latin_g2p.transliterate(w), "rules")) for w in todo)
        else:
            mode = served_choice()[1]
            en_model = _en_model()
            to_model, to_en = [], []
            for w in todo:
                if en_model is not None and word_tag(w) == "en":
                    to_en.append(w)
                elif route(w, english, romanised, mode) == "rules":
                    found[w] = (latin_g2p.transliterate(w), "rules")
                else:
                    to_model.append(w)
            for w, nb in zip(to_model, model.nbest_batch(to_model)):
                found[w] = (pick(nb, known), "model")
            if to_en:
                for w, nb in zip(to_en, en_model.nbest_batch(to_en)):
                    found[w] = (pick(nb, known), "model_en")
        for w in todo:
            _remember(w, found[w])
    return [found[w] for w in lower]


def _erf(x: np.ndarray) -> np.ndarray:
    """Element-wise error function, Abramowitz–Stegun 7.1.26 (absolute error ≤ 1.5e-7, ample for GELU).

    Args:
        x: Any array.

    Returns:
        erf(x), same shape.

    Examples:
        >>> import numpy as np
        >>> np.round(_erf(np.array([0.0, 1.0, -1.0])), 6).tolist()
        [0.0, 0.842701, -0.842701]
    """
    sign = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-a * a))


def load(path: str | Path = DEFAULT_WEIGHTS) -> Transliterator:
    """Load exported weights (`scripts/translit_train.py export`).

    Args:
        path: The `.npz` file (default `models/model/translit.npz`). It holds every parameter plus a
            `__meta__` JSON string (vocabularies, heads, layers).

    Returns:
        A ready `Transliterator`.

    Raises:
        FileNotFoundError: If the file does not exist.

    Examples:
        >>> from khmer_translit import translit
        >>> model = translit.load()      # doctest: +SKIP
    """
    data = np.load(path)
    meta = json.loads(str(data["__meta__"]))
    weights = {k: data[k].astype(np.float64) for k in data.files if k != "__meta__"}
    return Transliterator(weights, meta["src_vocab"], meta["tgt_vocab"], meta["heads"], meta["layers"],
                          meta.get("phones", False), meta.get("max_in", MAX_IN), meta.get("tags", False))
