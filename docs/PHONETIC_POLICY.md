# Phonetic policy: English word → Khmer spelling (authoritative, 2026-10-28)

This file is the single reference for what counts as a good output of the English → Khmer phonetic transliterator.
It applies to:
- review judgments;
- training-data acceptance (E08's `DATA_POLICY.md` is consistent with it);
- documentation claims.

## The task
- **Input:** one English word or name, together with its intended English reading.
- **Output:** Khmer script that a Khmer reader would read aloud as a recognisable approximation of that
  pronunciation.
- **Sound, not meaning:** *coffee* is spelled as heard, and the Khmer word for coffee is not the answer unless it is
  also a sound-compatible loan spelling.

## Reading and accent
1. **Default accent:** General American, as recorded by the first CMUdict reading (`en-US-cmudict`). Where a
   pronunciation is shown next to the word, that reading is the one to judge.
2. **Ambiguous words:** words with several readings that differ beyond stress (*record* as noun or verb, *live*) are
   judged against the reading shown. If none is shown, the item is excluded or marked **uncertain**, never guessed.
3. **Names:** a name follows the English reading specified for it, not its original-language pronunciation (*Jose*
   as read in English when that reading is specified). An output that clearly follows another language's reading is
   **not acceptable** for that item; if it is unclear which reading the output follows, mark it **uncertain**.

## What is acceptable
4. **Approximation is expected.** Khmer has no exact match for many English sounds. Acceptable adaptations include:
   - the usual consonant substitutions (English *f* → ហ្វ or ផ, *g* → ហ្គ or ក, *v* → វ);
   - unwritten or simplified final clusters (*-st*, *-nd*);
   - dropped or weakened final *r* in non-rhotic-style spellings;
   - a common vowel approximation;
   - the choice between Khmer consonant series, with or without ៉ / ៊, when the reading stays recognisable.
5. **Established loan spellings** count when they are compatible with the intended pronunciation (e.g. a common
   Khmer spelling of *taxi*).
6. **Several spellings can all be acceptable.** A reviewer may accept every candidate that qualifies.

## What is not acceptable
7. **Changes to what is pronounced:**
   - **omitted syllables or endings** (*leonard* → លីណាត drops a syllable);
   - **added syllables**;
   - **a changed consonant** that changes the word heard (*grade* → ក្រេដ is acceptable only if ក for *g* is
     judged recognisable);
   - **a vowel** that makes a different word.
8. **Translations, explanations and descriptions,** including correct translations.
9. **Defective script:** Latin or other-script letters, look-alike letters from other scripts (Lao ໂ for ោ), or
   broken script.

## Judgment values (per candidate)
| Value | Meaning |
|---|---|
| `acceptable` | Meets 4–6 for the reading shown |
| `not_acceptable` | Fails 7–9 |
| `uncertain` | The reviewer cannot decide (unclear reading, unfamiliar name, borderline adaptation) |

Per word, a reviewer may also mark **`none_acceptable`**. This must agree with the candidate judgments: no candidate
can then be `acceptable`. Reviewers never type a new spelling; the task needs no reviewer-authored transcription.

## Evidence tracks (never pooled into one accuracy figure)
| Track | Source | What it can support |
|---|---|---|
| 1. Historical mined-reference agreement | E03 dev/eval: web-mined, consonant-screened, unreviewed, repeatedly examined | Exploratory comparison of systems on the same items; not phonetic accuracy |
| 2. Agent / AI-assisted exploratory judgments | Inspection by an AI agent, or AI-drafted labels | Hypotheses and error categories only; never ground truth |
| 3. Independent human-reviewed judgments | A Khmer-literate reviewer using this policy, blind to system identity | Phonetic acceptability, in the role declared before the review |

**Required wording:** a document that reports a number must say which track it comes from. Without track-3
evidence, quality is described as "unverified phonetic quality".
