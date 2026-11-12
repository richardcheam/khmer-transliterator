"""The character encoder-decoder in NumPy (inference only): the network of the small English model.

Pre-LayerNorm Transformer: character embeddings plus learned positions, encoder and decoder blocks, beam search with
cached keys and values. Parameter names match the PyTorch model in `training/nets.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_IN = 32
MAX_OUT = 40
PAD, BOS, EOS = 0, 1, 2


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
        w: Parameter name → array (PyTorch `state_dict` names of `training/nets.py`).
        src_vocab: Latin characters; index i + 3 is the token id.
        tgt_vocab: Khmer characters; index i + 3 is the token id.
        heads: Attention heads.
        layers: Encoder layers = decoder layers.
        phones: Unused by this package (always False); kept so older weight files load unchanged.
        tags: Unused by this package (always False).
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

    def encode_ids(self, ids: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        """Run the encoder on source token ids (the part of `encode` after tokenisation).

        Args:
            ids: One list of source ids per item (PAD = 0 never inside), each at most `max_in` long.

        Returns:
            `(memory, pad_mask)` as `encode` returns them.
        """
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

    def beam_search(self, memory: np.ndarray, pad: np.ndarray, beam: int = 4) -> list[list[tuple[str, float]]]:
        """The beam search of `nbest_batch`, from encoder states (`encode` or `encode_ids`).

        Args:
            memory: `(B, T, d)` encoder states.
            pad: `(B, T)` pad mask.
            beam: Beam width; also the number of results per item.

        Returns:
            One list of `(khmer, logprob)` per item, best first.
        """
        n_items = memory.shape[0]
        cross = [(self._heads(self._linear(memory, f"dec.{n}.cross.k")), self._heads(self._linear(memory, f"dec.{n}.cross.v")))
                 for n in range(self.layers)]
        d = memory.shape[-1]
        h, dh = self.heads, d // self.heads
        cache_k = [np.zeros((n_items, h, 0, dh)) for _ in range(self.layers)]
        cache_v = [np.zeros((n_items, h, 0, dh)) for _ in range(self.layers)]
        # one row per live beam: (word index, tokens, score); row r of the caches belongs to rows[r]
        rows: list[tuple[int, list[int], float]] = [(i, [BOS], 0.0) for i in range(n_items)]
        done: list[list[tuple[list[int], float]]] = [[] for _ in range(n_items)]
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


