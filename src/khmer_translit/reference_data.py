"""Provenance-aware reference items and lexical-family grouping for the English phonetic benchmark (E02).

A `ReferenceItem` is one English word **with one intended reading**: two readings of `record` would be two items
(different `pron_id`) in the same lexical group. Accepted Khmer spellings are a set (several can be right); the
preferred form is a separate field and is not needed for scoring.

**Quality tiers** (`TIERS`), from strongest to weakest:
- `independent_reviewed`: documented acceptance under the E02 phonetic policy by a competent reviewer independent
  of the models and of candidate generation. Only these items can qualify a benchmark.
- `attested`: a human-written or human-reviewed spelling from a reliable source, not reviewed under this policy
  and with no claim that every acceptable form is listed.
- `proxy`: automatic, LLM-drafted or only spot-checked material.

**Lexical families** (`family_key`, `group_families`): words that share a simple suffix stem (share / sharing /
shared) are one family, joined transitively, so a family never straddles a split and is resampled as one unit.

Standard library only.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

TIERS = ("independent_reviewed", "attested", "proxy")
SPLITS = ("train", "dev", "final", "final_proxy", "historical")
SUFFIXES = ("ings", "ing", "ers", "er", "ies", "ied", "es", "ed", "ly", "s")


def stems(word: str) -> set[str]:
    """The word and its simple suffix-stripped stems (with a restored final e), as in E01's leak filter.

    Args:
        word: A normalised word.

    Returns:
        Stems of length ≥ 3, plus the word itself.

    Examples:
        >>> sorted(stems("sharing"))
        ['shar', 'share', 'sharing']
        >>> sorted(stems("ok"))
        ['ok']
    """
    out = {word}
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            base = word[: -len(suf)]
            out |= {base, base + "e", base + "y"} if suf in ("ies", "ied") else {base, base + "e"}
            if len(base) >= 4 and base[-1] == base[-2]:
                out.add(base[:-1])
    return {s for s in out if len(s) >= 3} | {word}


def group_families(words: Iterable[str]) -> dict[str, str]:
    """Group words into lexical families: two words are related when their `stems` meet; families are the
    connected components (union-find), named by their alphabetically first word.

    Args:
        words: Normalised words (duplicates allowed).

    Returns:
        `{word: family id}`.

    Examples:
        >>> fam = group_families(["share", "sharing", "shared", "table", "tablets"])
        >>> fam["sharing"], fam["shared"], fam["tablets"], fam["table"]
        ('share', 'share', 'tablets', 'table')
    """
    words = sorted(set(words))
    parent = {w: w for w in words}

    def find(w: str) -> str:
        while parent[w] != w:
            parent[w] = parent[parent[w]]
            w = parent[w]
        return w

    by_stem: dict[str, str] = {}
    for w in words:
        for s in stems(w):
            if s in by_stem:
                a, b = find(w), find(by_stem[s])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                by_stem[s] = w
    return {w: find(w) for w in words}


def related(word: str, held_stems: set[str]) -> bool:
    """Whether a word equals or is a simple variant of a held-out word (its stems meet `held_stems`).

    Examples:
        >>> related("tablet", stems("tablets")), related("table", stems("tablets"))
        (True, False)
    """
    return bool(stems(word) & held_stems)


def split_bucket(key: str, salt: str) -> int:
    """A stable 0–99 bucket for a key (sha256 of `salt:key`), used for hash splits.

    Examples:
        >>> split_bucket("share", "e02") == split_bucket("share", "e02")
        True
    """
    return int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest(), 16) % 100


@dataclass
class ReferenceItem:
    """One benchmark item: an English word with one intended reading and its accepted Khmer spellings.

    Attributes:
        id: Unique id, `"<word>#<pron_id>"`.
        input: The word as it appears in the source.
        word: Normalised word (`phonetic_source.normalize_word`).
        family: Lexical family id (`group_families`), the split and resampling unit.
        pron_id: Reading id: "default" (first CMUdict reading or none) or "alt1", "alt2", ...
        intended_phones: The intended English reading (ARPAbet), "" if the dictionary has none.
        accent: Accent of the intended reading ("en-US-cmudict").
        accepted: Accepted Khmer spellings, NFC, sorted.
        preferred: The source's own spelling (one of `accepted`).
        sources: Source ids (e.g. "loanword_list:owner_added").
        tier: One of `TIERS`.
        split: One of `SPLITS`.
        flags: Audit flags (e.g. "pronunciation_ambiguous", "dictionary_miss").
        review: Reviewer / policy / date evidence, `{}` if none.
    """

    id: str
    input: str
    word: str
    family: str
    pron_id: str
    intended_phones: str
    accent: str
    accepted: list[str]
    preferred: str
    sources: list[str]
    tier: str
    split: str
    flags: list[str] = field(default_factory=list)
    review: dict = field(default_factory=dict)

    def validate(self) -> None:
        """Check one item's own fields.

        Raises:
            ValueError: Empty id/word/references, unknown tier or split, or a preferred form not accepted.
        """
        if not self.id or not self.word:
            raise ValueError(f"item without id or word: {self!r}")
        if not self.accepted or any(not a for a in self.accepted):
            raise ValueError(f"{self.id}: no accepted spelling")
        if self.tier not in TIERS:
            raise ValueError(f"{self.id}: unknown tier {self.tier!r}")
        if self.split not in SPLITS:
            raise ValueError(f"{self.id}: unknown split {self.split!r}")
        if self.preferred and self.preferred not in self.accepted:
            raise ValueError(f"{self.id}: preferred form not among accepted forms")


def validate_items(items: Sequence[ReferenceItem]) -> None:
    """Check a whole set: each item valid, ids unique, compound identities (word, reading, accent) unique.

    Args:
        items: The items.

    Raises:
        ValueError: On the first problem found.

    Examples:
        >>> a = ReferenceItem("ab#default", "ab", "ab", "ab", "default", "", "en-US-cmudict", ["អាប"], "អាប", [],
        ...                   "attested", "dev")
        >>> validate_items([a, a])
        Traceback (most recent call last):
        ...
        ValueError: duplicate item id 'ab#default'
    """
    ids, identities = set(), set()
    for it in items:
        it.validate()
        if it.id in ids:
            raise ValueError(f"duplicate item id {it.id!r}")
        key = (it.word, it.pron_id, it.accent)
        if key in identities:
            raise ValueError(f"duplicate compound identity {key}")
        ids.add(it.id)
        identities.add(key)


def save_reference_items(items: Sequence[ReferenceItem], path: Path) -> None:
    """Write items as JSON lines (sorted by id), after `validate_items`.

    Args:
        items: The items.
        path: Output `.jsonl`.
    """
    validate_items(items)
    lines = [json.dumps(asdict(it), ensure_ascii=False, sort_keys=True) for it in sorted(items, key=lambda i: i.id)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_reference_items(path: Path) -> list[ReferenceItem]:
    """Read and validate a `.jsonl` written by `save_reference_items`.

    Args:
        path: The file.

    Returns:
        The items, file order.

    Raises:
        ValueError: If the file is malformed or the set is invalid (unknown fields, duplicates, ...).
    """
    items = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append(ReferenceItem(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"{path}:{n}: malformed item ({e})") from None
    validate_items(items)
    return items


def audit_splits(items: Sequence[ReferenceItem], training_words: Iterable[str] = (),
                 training_targets: Iterable[str] = ()) -> dict:
    """Separation checks between splits and against training material.

    Args:
        items: Every reference item (all splits).
        training_words: Normalised Latin words of the supervised training rows.
        training_targets: Khmer targets of the supervised training rows.

    Returns:
        `{"families_in_several_splits": [...], "train_word_related": {split: n}, "train_target_shared":
        {split: n}, "sizes": {split: {"items", "families"}}}`; a clean benchmark has the first list empty and
        zero counts for every evaluation split.
    """
    by_family: dict[str, set[str]] = {}
    for it in items:
        by_family.setdefault(it.family, set()).add(it.split)
    held_stems = {s for w in set(training_words) for s in stems(w)}
    targets = set(training_targets)
    out = {"families_in_several_splits": sorted(f for f, s in by_family.items() if len(s) > 1),
           "train_word_related": {}, "train_target_shared": {}, "sizes": {}}
    for sp in SPLITS:
        sub = [it for it in items if it.split == sp]
        if not sub:
            continue
        out["sizes"][sp] = {"items": len(sub), "families": len({it.family for it in sub})}
        out["train_word_related"][sp] = sum(related(it.word, held_stems) for it in sub)
        out["train_target_shared"][sp] = sum(any(a in targets for a in it.accepted) for it in sub)
    return out
