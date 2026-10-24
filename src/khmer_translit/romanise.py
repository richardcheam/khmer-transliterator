"""Khmer script → the common Latin typing of Khmer (names in the press, chat), from the written form (D170).

Popular romanisation follows the **spelling and consonant series**, not only the sound: កំពត → kampot,
មង្គល → mongkol, សុភា → sophea, ចាន់ → chan, វាសនា → veasna. This module parses a word into orthographic
syllables (onset cluster, vowel sign, final consonant) and writes each one with series-dependent vowel values.
`romanise_variants` also returns common alternative typings (ea / a, t / th, u / o …), because real users vary.

Rules only, no data: the clean way to make romanised-Khmer training pairs from a Khmer word list (D170).
"""
from __future__ import annotations

import re

COENG, MUUS, TRIISAP, BANTOC, TOAN = "្", "៉", "៊", "់", "៍"
# consonant → (Latin, series 1 or 2)
CONS = {
    "ក": ("k", 1), "ខ": ("kh", 1), "គ": ("k", 2), "ឃ": ("kh", 2), "ង": ("ng", 2),
    "ច": ("ch", 1), "ឆ": ("chh", 1), "ជ": ("ch", 2), "ឈ": ("chh", 2), "ញ": ("nh", 2),
    "ដ": ("d", 1), "ឋ": ("th", 1), "ឌ": ("d", 2), "ឍ": ("th", 2), "ណ": ("n", 1),
    "ត": ("t", 1), "ថ": ("th", 1), "ទ": ("t", 2), "ធ": ("th", 2), "ន": ("n", 2),
    "ប": ("b", 1), "ផ": ("ph", 1), "ព": ("p", 2), "ភ": ("ph", 2), "ម": ("m", 2),
    "យ": ("y", 2), "រ": ("r", 2), "ល": ("l", 2), "វ": ("v", 2), "ឝ": ("s", 1), "ឞ": ("s", 1),
    "ស": ("s", 1), "ហ": ("h", 1), "ឡ": ("l", 1), "អ": ("", 1),
}
FINAL = {"ក": "k", "ខ": "k", "គ": "k", "ឃ": "k", "ង": "ng", "ច": "ch", "ឆ": "ch", "ជ": "ch", "ឈ": "ch",
         "ញ": "nh", "ដ": "t", "ឋ": "t", "ឌ": "t", "ឍ": "t", "ណ": "n", "ត": "t", "ថ": "t", "ទ": "t", "ធ": "t",
         "ន": "n", "ប": "b", "ផ": "p", "ព": "p", "ភ": "p", "ម": "m", "យ": "y", "រ": "r", "ល": "l", "វ": "v",
         "ស": "s", "ហ": "h", "ឡ": "l", "អ": ""}
SONORANT = set("ងញនមយរលវណ")
# vowel sign(s) → (series-1 value, series-2 value); longest match first
VOWELS = {
    "ាំង": ("ang", "eang"), "ុំ": ("om", "um"), "ាំ": ("am", "oam"), "ោះ": ("aoh", "uoh"), "េះ": ("eh", "eh"),
    "ុះ": ("oh", "uh"), "ាះ": ("ah", "eah"), "ិះ": ("eh", "ih"), "ំ": ("am", "um"), "ះ": ("ah", "eah"),
    "ៈ": ("ak", "eak"), "ា": ("a", "ea"), "ិ": ("e", "i"), "ី": ("ey", "i"), "ឹ": ("oe", "oe"),
    "ឺ": ("eu", "eu"), "ុ": ("o", "u"), "ូ": ("ou", "ou"), "ួ": ("uo", "uo"), "ើ": ("aeu", "eu"),
    "ឿ": ("oeu", "oeu"), "ៀ": ("ie", "ie"), "េ": ("e", "e"), "ែ": ("ae", "e"), "ៃ": ("ai", "ey"),
    "ោ": ("ao", "o"), "ៅ": ("ao", "ov"), "័": ("a", "oa"),
}
INHERENT_OPEN, INHERENT_CLOSED = ("a", "o"), ("a", "o")
INDEP = {"ឥ": "e", "ឦ": "ei", "ឧ": "u", "ឩ": "ou", "ឪ": "ov", "ឫ": "rue", "ឬ": "rueu", "ឭ": "lue", "ឮ": "lueu",
         "ឯ": "ae", "ឰ": "ai", "ឱ": "ao", "ឳ": "ao", "ឲ": "ao"}
VSIGN = set("ាិីឹឺុូួើឿៀេែៃោៅំះៈ័")
# (vowel value, final) pairs written as one group
GROUP = {("am", "ng"): "ang", ("oam", "ng"): "eang", ("om", "ng"): "ong", ("um", "ng"): "ung", ("ae", "v"): "aev", ("e", "v"): "eo", ("a", "y"): "ay", ("ea", "y"): "eay", ("ea", "v"): "eav",
         ("o", "y"): "oy", ("u", "y"): "uy", ("ou", "y"): "ouy", ("uo", "y"): "uoy"}
ASPIRATE = {("ត", "ថ"), ("ទ", "ធ"), ("ក", "ខ"), ("គ", "ឃ"), ("ច", "ឆ"), ("ជ", "ឈ"), ("ប", "ផ"), ("ព", "ភ")}
HSUB = {"វ": "f", "គ": "g", "ស": "z", "ល": "l", "ម": "m", "ន": "n", "ក": "g", "យ": "y", "វ៉": "f"}
PALI_FINAL = {"ត": "th", "ថ": "th", "ទ": "th", "ធ": "th"}
VARIANTS = [("ea", "a"), ("o", "a"), ("b", "p"), ("t", "th"), ("u", "o"), ("ey", "i"), ("ou", "u"), ("ch", "j"), ("v", "w"),
            ("ae", "e"), ("oe", "eu"), ("ao", "au"), ("aeu", "oeu"), ("k", "g")]


def units(word: str) -> list[dict]:
    """Split a word into consonant units: base consonant, subscripts, vowel signs, series marks.

    Args:
        word: Khmer letters (other characters are skipped).

    Returns:
        `[{"base", "subs", "vowel", "series", "silent", "indep"}]`.

    Examples:
        >>> [(u["base"], u["subs"], u["vowel"]) for u in units("កម្ពុជា")]
        [('ក', '', ''), ('ម', 'ព', 'ុ'), ('ជ', '', 'ា')]
    """
    out: list[dict] = []
    i = 0
    while i < len(word):
        ch = word[i]
        if ch in INDEP:
            out.append({"base": "", "subs": "", "vowel": "", "series": 1, "silent": False, "indep": INDEP[ch]})
            i += 1
            continue
        if ch not in CONS:
            if out and ch in VSIGN:
                out[-1]["vowel"] += ch
            elif out and ch == TOAN:
                out[-1]["silent"] = True
            i += 1
            continue
        u = {"base": ch, "subs": "", "vowel": "", "series": CONS[ch][1], "silent": False, "indep": "", "muus": False}
        i += 1
        while i < len(word):
            c = word[i]
            if c == COENG and i + 1 < len(word) and word[i + 1] in CONS:
                u["subs"] += word[i + 1]
                i += 2
            elif c == MUUS:
                u["series"], u["muus"] = 1, True
                i += 1
            elif c == TRIISAP:
                u["series"] = 2
                i += 1
            elif c in VSIGN:
                u["vowel"] += c
                i += 1
            elif c == BANTOC:
                i += 1
            elif c == TOAN:
                u["silent"] = True
                i += 1
            else:
                break
        out.append(u)
    for u in out:   # a non-sonorant subscript sets the series of the cluster (ស្ព → 2nd)
        if u["subs"] and u["base"] and CONS[u["base"]][1] == 1 and CONS[u["subs"][-1]][1] == 2 \
                and u["subs"][-1] not in SONORANT and MUUS not in word:
            u["series"] = 2
    return out


def vowel_value(sign: str, series: int) -> str:
    """Latin value of a vowel-sign sequence for a series (longest known match; unknown signs dropped).

    Examples:
        >>> vowel_value("ា", 2), vowel_value("ុំ", 1), vowel_value("ោះ", 2)
        ('ea', 'om', 'uoh')
    """
    for k in sorted(VOWELS, key=len, reverse=True):
        if sign.startswith(k):
            return VOWELS[k][series - 1] + vowel_value(sign[len(k):], series) if sign[len(k):] else VOWELS[k][series - 1]
    return ""


def romanise(word: str) -> str:
    """Canonical popular romanisation of one Khmer word.

    Args:
        word: A Khmer word (no spaces).

    Returns:
        Lower-case Latin.

    Examples:
        >>> [romanise(w) for w in ["សុភា", "កំពត", "មង្គល", "កម្ពុជា", "ចាន់", "សុខា", "ប្រាក់", "អាំង"]]
        ['sophea', 'kampot', 'mongkol', 'kampuchea', 'chan', 'sokha', 'prak', 'ang']
        >>> romanise("ហ្វេសប៊ុក"), romanise("ហ្គេម")
        ('fesbuk', 'gem')
    """
    us = units(word)
    syl: list[list] = []        # [onset, vowel, final]
    for j, u in enumerate(us):
        if u["silent"]:
            continue
        if u["indep"]:
            syl.append(["", u["indep"], ""])
            continue
        last = j == len(us) - 1
        base, subs, series = u["base"], u["subs"], u["series"]
        latin = CONS[base][0]
        if base == "ប" and (u["muus"] or subs[:1] in ("រ", "ល")):
            latin = "p"                                # ប៉, ប្រ, ប្ល sound p
        if base == "ហ" and subs[:1] in HSUB:           # loan clusters ហ្វ f, ហ្គ g, ហ្ស z, ហ្ល l
            latin, subs = HSUB[subs[0]], subs[1:]
        gem = bool(subs) and (subs[0] == base or (base, subs[0]) in ASPIRATE)
        prev = syl[-1] if syl else None
        if gem and prev is not None and (not u["vowel"] or (last and u["vowel"] == "ិ")) and last:
            prev[2] = prev[2] or (PALI_FINAL.get(base, FINAL[base]) * (2 if subs[0] == base and base in SONORANT else 1))
            continue                                   # final geminate / Pali ending: វណ្ណ, សម្បត្តិ, ប្រសិទ្ធ
        if gem and subs[0] == base:
            onset = latin + "".join(CONS[x][0] for x in subs[1:])
        elif gem:
            if prev is not None and not prev[2]:
                prev[2] = FINAL[base]
            onset = CONS[subs[0]][0] + "".join(CONS[x][0] for x in subs[1:])
            series = CONS[subs[0]][1] if not u["muus"] else 1
        else:
            onset = latin + "".join(CONS[x][0] for x in subs)
        if u["vowel"]:
            syl.append([onset, vowel_value(u["vowel"], series), ""])
            continue
        nxt = us[j + 1] if j + 1 < len(us) else None
        nxt_final = (nxt is not None and j + 1 == len(us) - 1 and nxt["subs"] and not nxt["indep"]
                     and (nxt["subs"][0] == nxt["base"] or (nxt["base"], nxt["subs"][0]) in ASPIRATE)
                     and nxt["vowel"] in ("", "ិ"))
        if nxt_final:                                  # វ + ណ្ណ: this consonant opens the closing syllable
            syl.append([onset, INHERENT_CLOSED[series - 1], ""])
            continue
        closes = prev is not None and not prev[2] and (
            not prev[1].endswith(("m", "h", "k")) or (base == "ង" and prev[1].endswith("m")))
        if closes and not gem:
            if subs and base in SONORANT | {"ក", "ច", "ត", "ប", "ស"} and not last:
                prev[2] = FINAL[base]            # Pali cluster: base closes, subscript opens
                sub = subs[-1]
                syl.append([CONS[sub][0], INHERENT_OPEN[CONS[sub][1] - 1], ""])
            else:
                prev[2] = FINAL[base]
            continue
        syl.append([onset, (INHERENT_CLOSED if not last else INHERENT_OPEN)[series - 1], ""])
    out = []
    for onset, v, fin in syl:
        out.append(onset + GROUP.get((v, fin), v + fin))
    return re.sub(r"(.)\1\1+", r"\1\1", "".join(out))


def romanise_variants(word: str, n: int = 2) -> list[str]:
    """The canonical form plus up to `n - 1` common alternatives (first applicable `VARIANTS` substitutions).

    Examples:
        >>> romanise_variants("ដារា")
        ['darea', 'dara']
    """
    base = romanise(word)
    out = [base]
    for a, b in VARIANTS:
        if len(out) >= n:
            break
        if a in base:
            v = base.replace(a, b)
            if v not in out:
                out.append(v)
    return out
