# Benchmark card: English → Khmer phonetic spelling (khmer-translit-byt5 0.2.0)

The phonetic policy in `docs/PHONETIC_POLICY.md` defines what counts as acceptable. This card describes the evidence
available for this release. The three evidence tracks are reported separately and **never pooled**.

## Track 1: historical mined-reference agreement (numbers exist)
| Split | Items | Word families | Use |
|---|---|---|---|
| dev | 253 | 250 | Checkpoint selection, the E10 gate |
| eval | 228 | 222 | Historical scoring, after candidates and decoding were frozen |

- **Origin:** English words in brackets after a Khmer spelling, mined from FineWeb-2 `khm_Khmr` and Khmer
  Wikipedia, and accepted by an automatic consonant screen against CMUdict and Google's Khmer lexicon. A word can
  have several accepted spellings, and any one of them counts as a match.
- **Reference quality is unknown:**
  - the screen checks consonants only, so wrong vowels pass;
  - some references are respellings or brand forms rather than phonetic spellings.
- **Reuse:** the eval split has been examined in E03–E10. It is no longer a held-out test, and its results are
  exploratory. The sealed E01 reserve and the E02 final reference spellings were not opened.
- **Metrics:**
  - exact match of the greedy answer (ACC), exact match among 4 canonical-beam candidates, and character error
    rate (CER, normalised by reference length, best over references);
  - paired word-family bootstrap, 10,000 resamples, seed 20261027.
  - The NEWS 2018 metrics (W18-2408; shared-task report W18-2409) assume reviewed references. Mean F-score, MRR
    and MAP_ref would add precision without validity, so they are not reported.
  - Candidate coverage at 4 is diagnostic: a reviewer would still have to choose.
- **Context:** Kirov et al. 2024 (2024.cl-2.2) show that transliteration quality depends on context and on
  reference conventions. Single-word mined references capture neither.

## Confirmation set: operational diagnostics only
- **Sampling:**
  - 120 English words (40 ordinary, 40 irregular, 40 names), sampled by family before any new model output;
  - all evaluation splits, training words and synthetic inputs excluded;
  - manifest sha256 `3daec5e6…` (full hash in `evidence/frozen.json`).
- **No references and no independent judgments**, so it gives **no quality number**. It is "fresh", but that does
  not qualify it as a benchmark.
- **What it does report:** output health for every candidate model.
  - No empty, invalid-orthography, repeated or truncated outputs.
  - 120 distinct answers.
  - The two E10 models give the same greedy answer on only 28% of these words. Where the models disagree on new
    words, at least one of them is wrong or the word has several readings. It is a reason for review, not a score.

## Track 2: agent or AI-assisted exploratory judgments
None are counted for this release. Unreviewed LLM outputs are never gold labels.

## Track 3: independent human-reviewed phonetic judgments (not collected)
A blinded review batch is prepared but not collected. No reviewer was contacted and no judgments were fabricated.

- **Contents:** 60 dev words (40 representative, 20 sampled from model disagreements, reported separately) with
  163 candidate spellings, at most 4 per word.
- **Blinding:** model identities and reference origins are hidden, and the order is randomised deterministically.
  The key stays local.
- **Reviewer task:** pick a judgment per candidate under the phonetic policy. No transcription is typed.
- **Import:** a validating importer rejects missing IDs, duplicates and inconsistent judgments, and requires a
  provenance file. `experiments/E10-release-cycle/review/README.md` has the instructions.

Until track 3 exists, **the phonetic quality of this model is unverified.**
