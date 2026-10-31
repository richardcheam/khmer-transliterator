# Evaluation of the E11 adapter (khmer-translit-byt5-e11 0.2.0)

**One checkpoint everywhere:** E11 `r_s0`, update 3,000, sha256 `286292a4…9713`. The numbers below come from the
saved score files in `evidence/`. Four kinds of evidence are kept apart and are not pooled.

## 1. Mined-reference agreement (exploratory spelling agreement)
- **References:** web-mined Khmer spellings that passed an automatic consonant screen. They were **never
  reviewed**.
- **Repeated use:** the dev and eval splits have been examined many times since E03.
- **What it measures:** agreement with one mined spelling. It does not measure phonetic accuracy.
- **Source:** `evidence/e11_results.json`.

| Split | Items (families) | Greedy exact agreement | Mean CER |
|---|---|---|---|
| dev | 276 (253) | 53.0% (macro) | 0.157 |
| eval | 228 (222) | 52.3% | 0.163 |

- **E11 − E07** on eval: +2.5 points [95% CI −2.7, +7.7] (paired family bootstrap, 10,000 resamples). E11 − E08:
  −0.1 [−4.7, +4.6].
- **Registered class:** "uncertain alternative".
- **How weak this measure is:** the owner's review found 28 of 72 reference "misses" acceptable and 8 of 37
  reference matches not acceptable (E11 report).

## 2. Owner-reviewed development comparison (development evidence)
- **Words:** 60 historical-dev words, blinded, judged by the owner.
  - 40 are a representative sample.
  - 20 are words where the E07/E08/E11 greedy outputs disagreed, so they are failure-enriched.
- **Method:** the judgments were transcribed from the owner's chat replies under three confirmed interpretation
  rules (`data/e11/review/judgments.tsv.provenance.json`). This was the only review not collected on the click
  page.
- **Source:** `evidence/e11_dev_review_scores.json`.

| Sample | E11 | E07 | E08 |
|---|---|---|---|
| Representative (40) | 29 (72.5%) | 24 (60.0%) | 30 (75.0%) |
| Disagreement (20) | 10 (50.0%) | 5 (25.0%) | 8 (40.0%) |

These dev words also informed E11's checkpoint selection (through mined references), so this is not a test.

## 3. Blinded tests on two different 120-family samples
- **Samples:** both tests use 120 English word families, 40 ordinary / 40 irregular spelling / 40 names, frozen
  before the compared model was trained.
  - **Freshness:** each sample is disjoint from every training, dev and earlier review word family.
  - **Second sample:** it was drawn later and also excludes the first sample's families.
- **Review:** greedy outputs only. Spellings shared by both compared models were shown once. Model identity was
  hidden behind seeded labels.
- **Answers:** one project-affiliated judge (the owner) gave an explicit acceptable / not acceptable / uncertain
  for every spelling on the click page, under `docs/PHONETIC_POLICY.md` (General American reading, shown as CMUdict
  ARPAbet).

| Test | Spellings judged | E11 acceptable | Not acceptable | Uncertain | Confirmed-acceptable share |
|---|---|---|---|---|---|
| E12 test (2026-10-30) | 185 | 77 | 40 | 3 | **64.2%** |
| E13 test (2026-10-31) | 169 | 90 | 29 | 1 | **75.0%** |

Exploratory breakdown by declared group (40 each; no test):

| Group | E12 test | E13 test |
|---|---|---|
| Ordinary | 27 | 34 |
| Irregular spelling | 14 | 20 |
| Names | 36 | 36 |

How to read the two tests:
- **Same checkpoint:** the checkpoint and decoding are identical in both tests. The 64.2% and 75.0% are not two
  versions or an improvement. They are two samples of the same declared design.
- **Why they differ:** sampling variation (each share has roughly ±8 points of uncertainty at n = 120), different
  words and, possibly, the judge's drift between sessions.
- **No pooled figure:** no single headline accuracy is given. Pooling would need equal-weight sampling from one
  frame and one judging condition, and that isn't established.
- **Not a population benchmark:** the 40/40/40 design over-represents irregular spellings and names relative to
  everyday English text.
- **Uncertainty:** it was low (3 and 1 spellings), and the confirmed-acceptable share counts "uncertain" as not
  confirmed.
- **Judgments stay local:** the judgment files stay local; `evidence/` holds the aggregate score files.

## 4. What this evidence does not show
- **No independent validation:** there is no independent or second judge and no production validation.
- **No superiority claim:** E11 is not better than every earlier model. In paired comparisons on these tests,
  E12 = E11 (0.0 [−8.3, +8.3]) and E13 = E11 (0.0 [−7.5, +7.5]).
- **Unassessed spellings:** a spelling missing from an accepted list was never judged and stays unassessed. It is
  not automatically wrong.

## 5. Candidate coverage on irregular development words (development diagnostic)
COVERAGE_PENDING

## 6. Runtime
RUNTIME_PENDING

## 7. Methods and references
- **Multiple accepted outputs:** the NEWS 2018 whitepaper (Chen et al., W18-2408) defines ACC, mean F-score, MRR
  and MAP_ref over reference lists that may hold several accepted transliterations per word.
  - This project's mined data has one unreviewed spelling per item, so only exact agreement (ACC) and CER are
    reported against it.
  - The human tests record a judgment per spelling instead, so several spellings can be acceptable.
- **Candidate rescoring:** Grundkiewicz and Heafield (W18-2413) improved NEWS 2018 transliteration with
  ensembling and rescoring n-best lists with right-to-left models.
  - This motivates measuring whether acceptable spellings are present in E11's beam (§5).
  - It is not evidence that a reranker would help here. No reranker was built or measured.
- **Byte-level models:** Xue et al. (2022, TACL, ByT5) report byte-level models are robust to noise and strong on
  spelling- and pronunciation-sensitive tasks. That's why this base was chosen; it's not a measured Khmer result.
- **Model card:** follows the Hugging Face model-card guidance (metadata with `base_model`, `base_model_relation:
  adapter`, and a licence entry once one is chosen).
