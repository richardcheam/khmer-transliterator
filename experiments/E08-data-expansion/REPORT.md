# E08 — data expansion for ByT5-small: report of the preparation phase (2026-10-28)

**Update 3 (2026-10-28): the single exploratory R+S run (`rs_s0`, training amendment 1).**
**Classification: Small or uncertain change** (rule pre-registered in `TRAINING_AMENDMENT_1.md`).

- **What ran:** one fresh LoRA on the pinned ByT5-small, E07 recipe unchanged, seed 0, frozen arm R+S (5,148 rows,
  3,778 words; 384 synthetic rows = 193 words). 3,000 updates in 5.18 h, peak RSS 5,105 MB (limit 5,120 MB), no
  guard stop. Row-level sampling as E07, no oversampling.
  - Draws by source over 96,000: control 43.8%, recovered 25.1%, ALT 23.7%, synthetic 7.5%; 18.65 passes over the
    rows (E07: 42.6). 0 eval words in the training data.
- **Selection (dev only):** best at the last check, update 3,000.

  | Dev greedy (276 items) | 600 | 1,200 | 1,800 | 2,400 | 3,000 |
  |---|---|---|---|---|---|
  | rs_s0 top-1 / CER | 32.2 / .361 | 46.6 / .195 | 51.0 / .172 | 53.8 / .162 | **54.2 / .151** |
  | E07 top-1 / CER | 37.8 / .287 | **52.2 / .186** | 51.4 / .166 | 51.0 / .175 | 50.6 / .168 |

  E07 peaked early; rs_s0 was still improving when the schedule ended.
- **Predictions frozen before scoring:** `reports/rs_predictions_frozen.json` (sha256 per file), then
  `scripts/e08_train.py analyse`.
- **Historical E03 eval (228 primary items, 222 families).** This set has been examined repeatedly across E03–E08
  and its references are mined and unreviewed: the numbers are exploratory agreement with those references, not
  phonetic accuracy.

  | System | Top-1 (beam 4) | Top-4 oracle | Mean CER | Greedy top-1 |
  |---|---|---|---|---|
  | **rs_s0** | **47.5%** | 65.2% | 0.191 | 52.4% |
  | E07 | 47.2% | 64.7% | 0.189 | 49.9% |
  | default E02 letters | 46.2% | 75.6% | 0.207 | — |
  | G mean (s0–s2) | 41.3% | — | 0.217 | — |

  | Paired family bootstrap (10,000) | Δ (points) | 95% interval |
  |---|---|---|
  | rs_s0 − E07, top-1 (primary) | **+0.3** | [−5.7, +6.5] |
  | rs_s0 − E07, greedy top-1 | +2.6 | [−2.4, +7.5] |
  | rs_s0 − E07, top-4 oracle | +0.5 | [−5.4, +6.3] |
  | rs_s0 − E07, CER reduction | +0.1 | [−2.5, +2.8] |
  | rs_s0 − G mean | +6.1 | [+0.1, +12.2] |
  | rs_s0 − default | +1.3 | [−6.1, +8.8] |

  Rule check: Δ = +0.3 < +3.0, and eval CER 0.191 > 0.189 → not promising; the interval includes 0 and Δ > −3 →
  not a regression.
- **Validity:** every greedy output is valid Khmer (245/245); beam-4 candidates: 895 ok, 83 empty, 2 malformed
  UTF-8 (lower-ranked only; no top-1 is empty or invalid). Top-1 diversity is the same as E07 (223 vs 220 distinct).
- **Where they differ** (`results_rs_per_item.jsonl`): rs_s0 alone right on 24 items, E07 alone right on 25, both
  right 84, both wrong 95 (56 of them with the *same* wrong answer). Of the 91 items whose top-1 differ: 22 vowels
  only, 31 consonants only, 38 both.
  - By length: rs_s0 better on 5–6 characters (51.5 vs 47.5%), worse on 7–8 (31.0 vs 37.9%) and ≥ 9 (2/9 vs 5/9;
    tiny).
  - Words in the default model's training data: 39.4 vs 42.1%; words outside it: 54.7 vs 53.0%.
- **Blinded examples** (`reports/rs_blinded_examples.json`, 10 per group, A/B random; key in
  `rs_blinded_key.json`). **Agent inspection, not review** (`reports/rs_agent_inspection.json`): judged before
  unblinding, the agent preferred rs_s0 in 7, E07 in 17, tie 6.
  - Where the mined reference split the two, the agent mostly agreed with the reference (rs-only correct: 6–1;
    E07-only correct: 0–8).
  - Where both were wrong and different (no reference help), the agent preferred E07 8 of 10. rs_s0 there dropped a
    syllable (*leonard* → លីណាត), truncated (*yap* → យ៉) and turned g into j or k (*berger* → ប៊ើជើ,
    *grade* → ក្រេដ).
  - Vowels: rs_s0's vowel choices were not systematically better (e.g. *beats* → បៀត, *marx* → មក).
- **Cost:** load 6.3 s; batch-1 greedy median 485 ms; inference peak RSS 1,868 MB; adapter 25.1 MB (same as E07).

**Strongest evidence about the remaining limitation.** Doubling the data (2.3× the rows, three new sources)
changed 49 eval answers, but the changes cancel: 24 gained, 25 lost, and 56 items share the same wrong answer in
both models. On this evaluation, more noisy data of the same kind moves *which* words are right, not *how many*;
the bottleneck looks like label quality (consonant-screened, vowel errors pass) and reference quality, not data
quantity. A second, smaller signal: for both models greedy beats beam-4 top-1 (rs_s0 52.4 vs 47.5%), so the
sum-log-prob beam ranking costs about 5 points. That is noted only; greedy was not the registered primary, and no
decoding or data fix was launched.

**Not done (by the amendment):** no replication (the result is not promising), no C or R arm, no other training,
no serving, distillation, merge, release or publication. Model files stay local in `models/e08/rs_s0/`.

**Update 2 (2026-10-28): data frozen** (state `data_frozen_ready_for_registration`; `reports/data_frozen.json`).
- **Replacement round:** `regen_01` fixed 3 of the 8 look-alike-script items. The other 5 used Lao ໂ again despite
  the added warning, so they stay quarantined. `regen_02.md` is optional.
- **Synthetic total:** 193 words accepted, 384 rows (preferred + alternatives).
- **Arm R+S:** 5,148 optimisation rows over 3,778 words. The C and R hashes match `training_registration.json`.
- **Importer fix:** an answered replacement round's id list is no longer rewritten by a later import (regression
  test added).
- **Training:** still not run; it needs the owner's go-ahead.

**Update 1 (2026-10-28, after the owner's AI Mode answers):** state `awaiting_regeneration_responses` (optional).
- **Accepted:** 190/200 words, with 188 alternatives (`google_ai_mode_synthetic`). By stratum: common 37,
  irregular 38, vowels 39, clusters 39, names 37.
- **Quarantined:**
  - 8 preferred spellings with Lao/Tibetan look-alike letters (ໂ for ោ). Their replacement prompt is
    `synthetic/prompts/regen_01.md`.
  - 2 that AI Mode itself marked uncertain.
- **The owner's extra round:** `additional.md`, 130 words, kept separate. It filled 2 structural quarantines.
- **Self-consistency:** where both rounds were accepted, the preferred spelling was the same for only 48 of 122
  words; 22 more shared some spelling and 52 were disjoint. These labels are unstable, and that is reported as a
  property of the synthetic data, not validation.

**Original state: `awaiting_ai_mode_responses`.** This was an external input dependency.
- All preparation that does not need AI Mode answers is done.
- **No training ran in this phase.**
- Product default, serving, previous results and checkpoints, protected families, E02 final references and the
  reserve are unchanged.

## Files
- **Policy:** `DATA_POLICY.md`, frozen before any example was accepted.
- **Training plan (not executed):** `TRAINING_PLAN.md` + `training_registration.json`.
- **Recovery:** `reports/recovery_audit.md`, `recovery_summary.json`.
- **Sources:**
  - `reports/source_audit.md`, `source_pilots.json`, `alt_attested_summary.json`.
- **Synthetic inputs and prompts:**
  - `reports/input_manifest_frozen.json`, `prompts_frozen.json`;
  - `synthetic/` (manifest, 10 prompts, README).
- **Code and tests:** `scripts/e08.py`, `tests/test_e08.py`.
- **Local only (git-ignored, data policy):**
  - every pair manifest (`data/e08/recovery/`, `data/e08/attested/`);
  - the arms (`data/e08/training/`);
  - the downloads (`data/e08/sources/`);
  - AI Mode responses (`data/e08/ai_mode_responses/`) and imports (`data/e08/synthetic/`).

## New eligible pairs, by provenance (after every exclusion)
| Provenance | Pairs | Words new to control | Acceptance | Agent inspection |
|---|---|---|---|---|
| `control` (E03 optimisation split, unchanged) | 2,255 (+93 train-val) | — | E03's build | — |
| `recovered` (E03-excluded mined pairs) | **1,292** | 813 (of 1,112 words) | Documented screen-weakness rules | About 16/20 plausible; vowel errors pass the consonant screen |
| `attested_alt` (ALT Parallel Corpus, CC BY 4.0) | **1,217** | 947 (of 1,097 words) | Verbatim match in the human translation + the same screen | About 34/40 plausible; truncated endings are the main error |
| `google_ai_mode_synthetic` | **0 so far** (200 inputs frozen) | 200 by construction | Structural import only | — |

**Quarantined, kept apart and not trained on:**
- recovery: 1,693 uncertain, 109 errors or translations;
- ALT: 288 uncertain, 5 errors.

## Exclusions enforced before any synthetic input was chosen
- **Protected English families** (stems): 4,502 words / 5,674 stems from Latin inputs only:
  - E03 exclusions (E02 held-out sets, v1 panels, reserve Latin words, E01 list validation);
  - E03 dev/eval words;
  - the E06 pool and discovery words;
  - the E05/E06 review queues;
  - the benchmark panels and the reserve through E06's input-only reader.
- **Not opened:** E02's final references and the reserve targets. Reads go through `e06.guard`, which raises on
  forbidden paths.
- **Excluded from the additions:**
  - recovered pairs in protected families (368), and ALT pairs in protected families (506);
  - targets equal to a dev/eval or held-out target (156 recovered, 150 ALT; cross-split reuse recorded, not
    called leakage);
  - exact duplicates of existing pairs (615 ALT);
  - rows from train-val families (22).

## Synthetic input manifest (frozen; `synthetic/input_manifest.jsonl`, sha256 `9dfb2fdb…cb30`)
- **Size:** 200 words, 40 per stratum, chosen before any AI Mode output and not from any test failures.
  - **Common:** count of the word as a Latin token in the local FineWeb-2 Khmer shard, ≥ 5, descending.
  - **Irregular:** E06's registered spelling-to-sound patterns.
  - **Vowels:** at most 3 per stressed vowel.
  - **Clusters:** initial or final runs of ≥ 2 consonants.
  - **Names:** SCOWL capitalised entries with a CMUdict reading.
- **Every word is:**
  - fresh against all historical Latin tokens (E06's rule) and outside protected families;
  - not a recovered or ALT word;
  - not a romanised-Khmer form;
  - one per family, with a single CMUdict reading up to stress (2,194 ambiguous candidates quarantined).
- **Amendment 1 (before any output):** abbreviation-like inputs excluded (9; e.g. *etc*, *ftp*, *blvd*).
- **Coverage:** every English phoneme appears. OY is primary-stressed only once, and ZH and DH once each.
- **Strata are not difficulty labels.** The names stratum includes brand and place names, such as *crisco* and
  *colosseum*.

## Prompts and importer
- **Prompts:** ten standalone prompts of 20 words each. Each contains:
  - the policy: sound, not meaning; the given English reading, including for names;
  - stable ids and the required table;
  - up to two same-reading alternatives;
  - an `uncertain` option, and no sources or confidence scores.

  No model predictions or benchmark references appear in them; a test checks that a prompt contains no Khmer at
  all.
- **Importer:** it keeps raw files and handles pipes, tabs, code fences and emphasis. It matches by id and detects
  missing, duplicate-conflicting, unknown and other-batch ids. It also:
  - normalises to NFC and keeps the raw cells;
  - checks script, length and orthography;
  - quarantines explanations in spelling cells, unreadable `uncertain` fields and English mismatches;
  - never promotes an alternative;
  - refuses protected overlap;
  - writes regeneration prompts for missing or malformed rows only, so accepted rows are never re-asked or
    overwritten.

## Limitations
- **The screen is consonant-only:** recovered and ALT pairs can carry vowel errors (about 20% questionable in agent
  samples).
- **ALT translators:** some may follow a name's original-language pronunciation, and the screen catches consonant
  conflicts only.
- **Agent inspection is not independent review.** Nothing here is a validated reference.
- **Synthetic spellings:** once imported, they are unverified LLM output. They stay training data only.

## Owner handoff (the only manual step)
Paste `synthetic/prompts/batch_01.md` … `batch_10.md` into Google AI Mode, and save each complete answer as
`data/e08/ai_mode_responses/batch_01.md` … `batch_10.md`. Then run:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e08.py import --responses data/e08/ai_mode_responses
```

## Reproduce
```bash
uv sync && uv pip install --python .venv/bin/python -r research/requirements-byt5.txt
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e08.py prepare   # needs data/e08/sources/ (URLs and sha256 in source_audit.md / source_pilots.json)
.venv/bin/python scripts/e08.py status
.venv/bin/python -m unittest tests.test_e08
```
`prepare` is resumable. It recomputes recovery, sources and attested pairs only when their summaries are absent, and
verifies the frozen manifest's hash instead of rebuilding it.
