"""English input processing for the English → Khmer phonetic transliterator (experiment E02).

One module owns every step between a raw English word and the model's source token ids, so training, validation,
NumPy inference and the CLI cannot drift apart:

1. **Normalisation** (`normalize_word`): NFKC, lower case, accents removed, apostrophes and hyphens dropped; the
   result must be one word of `a`–`z` letters, at most `MAX_LETTERS` long. Anything else gets an explicit status
   (`empty`, `unsupported_multiword`, `unsupported_characters`, `too_long`) instead of a guess.
2. **Pronunciation** (`get_pronunciation`): a pronunciation supplied by the caller (ARPAbet, CMUdict style, stress
   digit on every vowel), else the **first-listed** CMUdict reading of the normalised word, else none. There is no
   letter-to-sound fallback: a dictionary miss is reported as `unavailable` and encoded as `<no_pron>`, never
   replaced by a silent rule guess. Other CMUdict readings are kept as `alternatives` and flag `ambiguous`.
3. **Encoding** (`encode_source`): one token per letter and one token per phone (a phone such as `AH0` is one
   token, never split into characters):

   - `letters`: `<task:en>` + letters;
   - `phones`:  `<task:en>` + letters + `<sep>` + phones (or `<no_pron>`).

   Khmer phones → spelling pretraining (`encode_p2g`) uses `<task:km_p2g>` + Khmer phone tokens (`km:` namespace,
   from the Google Khmer lexicon). English and Khmer phones are separate tokens: the two alphabets are not mapped
   onto each other.

The English part of the vocabulary is fixed in code (`EN_TOKENS`), so its ids never depend on data; the Khmer
phone inventory is appended after it and stored with each model. `ENCODER_VERSION` names this contract and is
recorded in every model; a model with another version is refused.

Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CMUDICT = Path(os.environ.get("KHMER_TRANSLIT_CMUDICT", ROOT / "data" / "src" / "cmudict" / "cmudict.dict"))
ENCODER_VERSION = "en-phonetic-src-v1"
ALPHABET = "arpabet-cmudict-stress"
MAX_LETTERS = 32     # longest accepted normalised word (the longest eligible training word has 20 letters)
MAX_SRC = 80         # learned source positions; a longer encoded source is refused, never cut
MAX_TGT = 40         # Khmer code points, as in the existing models (translit.MAX_OUT)

PAD, BOS, EOS = 0, 1, 2
SPECIALS = ["<pad>", "<bos>", "<eos>", "<task:en>", "<task:km_p2g>", "<sep>", "<no_pron>"]
LETTERS = [chr(c) for c in range(ord("a"), ord("z") + 1)]
EN_VOWELS = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
EN_CONSONANTS = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
EN_PHONES = [v + s for v in EN_VOWELS for s in "012"] + EN_CONSONANTS
EN_TOKENS = SPECIALS + LETTERS + ["en:" + p for p in EN_PHONES]
STATUSES = ("provided", "dictionary", "g2p", "unavailable")   # pronunciation status; "g2p" has no provider yet


@dataclass(frozen=True)
class PronunciationResult:
    """The English pronunciation chosen for one word, with where it came from.

    Attributes:
        word: The normalised word (`normalize_word`), or "" if the input was refused.
        phones: The chosen phones (ARPAbet with stress digits), empty if none.
        status: One of `STATUSES`.
        provider: "caller", "cmudict" or "none".
        alternatives: Every other CMUdict reading, in file order.
        ambiguous: True when CMUdict lists readings that differ in more than stress and no reading was supplied.
        alphabet: `ALPHABET`.
        resource_sha256: sha256 of the CMUdict file used ("" if it was not needed).
        dictionary_hit: Whether the word has a CMUdict entry (also reported for supplied readings).
    """

    word: str
    phones: tuple[str, ...]
    status: str
    provider: str
    alternatives: tuple[tuple[str, ...], ...] = ()
    ambiguous: bool = False
    alphabet: str = ALPHABET
    resource_sha256: str = ""
    dictionary_hit: bool = False

    def to_dict(self) -> dict:
        """JSON-ready form (phones as space-joined strings).

        Returns:
            A dict with every field.

        Examples:
            >>> PronunciationResult("one", ("W", "AH1", "N"), "provided", "caller").to_dict()["phones"]
            'W AH1 N'
        """
        return {"word": self.word, "phones": " ".join(self.phones), "status": self.status, "provider": self.provider,
                "alternatives": [" ".join(a) for a in self.alternatives], "ambiguous": self.ambiguous,
                "alphabet": self.alphabet, "resource_sha256": self.resource_sha256,
                "dictionary_hit": self.dictionary_hit}


class InputError(ValueError):
    """An input the transliterator refuses, with a machine-readable `status`.

    Args:
        status: e.g. "empty", "unsupported_characters", "too_long", "unknown_phone".
        message: Human-readable explanation.
    """

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def normalize_word(raw: str) -> tuple[str, str]:
    """Normalise one English word, or say why it is refused.

    Steps: NFKC; strip surrounding space; lower case; remove accents (NFKD, combining marks dropped); drop
    apostrophes (' and ’) and hyphens. The result must match `[a-z]+` and be at most `MAX_LETTERS` long.

    Args:
        raw: The input as typed.

    Returns:
        `(normalised, status)`: status "ok", or "empty", "unsupported_multiword", "unsupported_characters",
        "too_long" (then `normalised` is still the cleaned string, for reporting).

    Examples:
        >>> normalize_word(" Café "), normalize_word("O'Neil"), normalize_word("wi-fi")
        (('cafe', 'ok'), ('oneil', 'ok'), ('wifi', 'ok'))
        >>> normalize_word("new york")[1], normalize_word("")[1], normalize_word("ខ្មែរ")[1]
        ('unsupported_multiword', 'empty', 'unsupported_characters')
    """
    w = unicodedata.normalize("NFKC", raw).strip().lower()
    w = "".join(c for c in unicodedata.normalize("NFKD", w) if not unicodedata.combining(c))
    w = re.sub("['’‘-]", "", w)
    if not w:
        return w, "empty"
    if re.search(r"\s", w):
        return w, "unsupported_multiword"
    if not re.fullmatch("[a-z]+", w):
        return w, "unsupported_characters"
    if len(w) > MAX_LETTERS:
        return w, "too_long"
    return w, "ok"


def sha256_file(path: Path) -> str:
    """sha256 of a file.

    Args:
        path: Any file.

    Returns:
        Hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=4)
def cmudict(path: Path = CMUDICT) -> tuple[dict[str, tuple[tuple[str, ...], ...]], str]:
    """Every CMUdict reading, keyed by the headword normalised like an input (`normalize_word`).

    Headwords that normalise to the same key (its / it's) share one entry, readings in file order.

    Args:
        path: The `cmudict.dict` file.

    Returns:
        `(readings, sha256)`: `{word: (reading, ...)}` and the file's sha256.

    Raises:
        FileNotFoundError: If the file is missing (run `khmer-translit prepare` to check resources).
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"CMUdict not found at {path}; it is needed for phone-input models")
    out: dict[str, list[tuple[str, ...]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line[:1].isalpha():
            continue
        head, *ph = line.split("#")[0].split()
        key, status = normalize_word(head.split("(")[0])
        if status != "ok" or not ph or any(p not in EN_PHONES for p in ph):
            continue
        if tuple(ph) not in out.setdefault(key, []):
            out[key].append(tuple(ph))
    return {k: tuple(v) for k, v in out.items()}, sha256_file(Path(path))


def parse_phones(text: str) -> tuple[str, ...]:
    """Parse a supplied pronunciation: space-separated ARPAbet, a stress digit (0/1/2) on every vowel.

    Args:
        text: e.g. "R EH1 K ER0 D".

    Returns:
        The phone tuple.

    Raises:
        InputError: status "unknown_phone" for an unknown or unstressed phone, "empty" for no phones.

    Examples:
        >>> parse_phones("r eh1 k er0 d")
        ('R', 'EH1', 'K', 'ER0', 'D')
        >>> parse_phones("R EH K")
        Traceback (most recent call last):
        ...
        khmer_translit.phonetic_source.InputError: unknown phone 'EH' (ARPAbet; vowels need a stress digit 0/1/2)
    """
    ph = tuple(text.upper().split())
    if not ph:
        raise InputError("empty", "empty pronunciation")
    for p in ph:
        if p not in EN_PHONES:
            raise InputError("unknown_phone", f"unknown phone {p!r} (ARPAbet; vowels need a stress digit 0/1/2)")
    return ph


def _strip_stress(ph: tuple[str, ...]) -> tuple[str, ...]:
    """Phones without stress digits (to tell real alternatives from stress variants)."""
    return tuple(p.rstrip("012") for p in ph)


def get_pronunciation(word: str, *, supplied: str | None = None, path: Path = CMUDICT) -> PronunciationResult:
    """The pronunciation used for one word: supplied, else the first CMUdict reading, else none.

    Args:
        word: The raw word (normalised here).
        supplied: Optional ARPAbet pronunciation (`parse_phones`); it always wins.
        path: CMUdict file (tests pass a small one).

    Returns:
        A `PronunciationResult`.

    Raises:
        InputError: If the word is refused by `normalize_word`, or `supplied` does not parse.

    Examples:
        >>> r = get_pronunciation("Coffee")              # doctest: +SKIP
        >>> r.status, r.phones                           # doctest: +SKIP
        ('dictionary', ('K', 'AA1', 'F', 'IY0'))
    """
    w, status = normalize_word(word)
    if status != "ok":
        raise InputError(status, f"input {word!r} refused: {status}")
    if supplied is not None:
        ph = parse_phones(supplied)
        try:
            table, digest = cmudict(path)
            hit = w in table
        except FileNotFoundError:
            digest, hit = "", False
        return PronunciationResult(w, ph, "provided", "caller", (), False, ALPHABET, digest, hit)
    table, digest = cmudict(path)
    readings = table.get(w)
    if not readings:
        return PronunciationResult(w, (), "unavailable", "none", (), False, ALPHABET, digest, False)
    ambiguous = len({_strip_stress(r) for r in readings}) > 1
    return PronunciationResult(w, readings[0], "dictionary", "cmudict", readings[1:], ambiguous, ALPHABET, digest,
                               True)


@dataclass
class SourceVocab:
    """Source token vocabulary: the fixed English part (`EN_TOKENS`) plus the Khmer phone inventory.

    Attributes:
        km_phones: Khmer phone symbols in id order (appended after `EN_TOKENS`).
        tokens: Every token string, index = id.
    """

    km_phones: list[str] = field(default_factory=list)
    tokens: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = EN_TOKENS + ["km:" + p for p in self.km_phones]
        self._ids = {t: i for i, t in enumerate(self.tokens)}

    def id(self, token: str) -> int:
        """Id of a token string.

        Raises:
            InputError: status "unknown_token" if absent.
        """
        try:
            return self._ids[token]
        except KeyError:
            raise InputError("unknown_token", f"token {token!r} not in the source vocabulary") from None

    def sha256(self) -> str:
        """Hash of the token list (recorded with every model and data build)."""
        return hashlib.sha256(json.dumps(self.tokens, ensure_ascii=False).encode()).hexdigest()


ENGLISH_VOCAB = SourceVocab()
MODES = ("letters", "phones")


def encode_source(word: str, pronunciation: PronunciationResult | None, mode: str,
                  vocab: SourceVocab = ENGLISH_VOCAB, drop_phones: bool = False) -> list[int]:
    """Source token ids for one English word.

    Args:
        word: The normalised word (`normalize_word` status "ok").
        pronunciation: Its `PronunciationResult` (required for mode "phones"; ignored for "letters").
        mode: "letters" or "phones".
        vocab: The source vocabulary (the English ids are the same in every vocabulary).
        drop_phones: Encode `<no_pron>` instead of the phones (the registered training-time phone dropout).

    Returns:
        Token ids, at most `MAX_SRC`.

    Raises:
        InputError: "unsupported_characters" for a non-normalised word, "too_long" if the source exceeds
            `MAX_SRC`; ValueError for an unknown mode or a missing pronunciation in mode "phones".

    Examples:
        >>> p = PronunciationResult("one", ("W", "AH1", "N"), "provided", "caller")
        >>> [ENGLISH_VOCAB.tokens[i] for i in encode_source("one", p, "phones")]
        ['<task:en>', 'o', 'n', 'e', '<sep>', 'en:W', 'en:AH1', 'en:N']
        >>> [ENGLISH_VOCAB.tokens[i] for i in encode_source("one", p, "letters")]
        ['<task:en>', 'o', 'n', 'e']
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if not re.fullmatch("[a-z]+", word):
        raise InputError("unsupported_characters", f"{word!r} is not a normalised word")
    ids = [vocab.id("<task:en>")] + [vocab.id(c) for c in word]
    if mode == "phones":
        if pronunciation is None:
            raise ValueError("mode 'phones' needs a pronunciation")
        ids.append(vocab.id("<sep>"))
        if drop_phones or not pronunciation.phones:
            ids.append(vocab.id("<no_pron>"))
        else:
            ids += [vocab.id("en:" + p) for p in pronunciation.phones]
    if len(ids) > MAX_SRC:
        raise InputError("too_long", f"source of {word!r} has {len(ids)} tokens > {MAX_SRC}")
    return ids


def encode_p2g(km_phones: list[str], vocab: SourceVocab) -> list[int]:
    """Source ids for Khmer phones → Khmer spelling pretraining: `<task:km_p2g>` + one token per Khmer phone.

    Args:
        km_phones: Google-lexicon phone symbols, syllable dots included (e.g. `["s", "ɑ", "w", ".", "r", "ɨ"]`).
        vocab: A vocabulary whose `km_phones` cover them.

    Returns:
        Token ids.

    Raises:
        InputError: "unknown_token" for a phone outside the vocabulary, "too_long" beyond `MAX_SRC`.

    Examples:
        >>> v = SourceVocab(["k", "a"])
        >>> [v.tokens[i] for i in encode_p2g(["k", "a"], v)]
        ['<task:km_p2g>', 'km:k', 'km:a']
    """
    ids = [vocab.id("<task:km_p2g>")] + [vocab.id("km:" + p) for p in km_phones]
    if len(ids) > MAX_SRC:
        raise InputError("too_long", f"P2G source has {len(ids)} tokens > {MAX_SRC}")
    return ids


def nfc(text: str) -> str:
    """Canonical Unicode form (NFC) used for every Khmer reference and prediction.

    Examples:
        >>> nfc("ក") == "ក"
        True
    """
    return unicodedata.normalize("NFC", text)


def is_khmer_word(text: str) -> bool:
    """Whether a string is one Khmer word: Khmer letters, vowels and signs only (U+1780–U+17D3, U+17DD), no
    punctuation (។ ៕ ...), digits, spaces or zero-width characters.

    Examples:
        >>> is_khmer_word("ថូម៉ាស់"), is_khmer_word("ថូម៉ាស់។"), is_khmer_word("")
        (True, False, False)
    """
    return bool(text) and all("ក" <= c <= "៓" or c == "៝" for c in text)
