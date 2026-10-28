# E09 — ByT5 decoding: why does greedy beat beam-4, and can decoding improve E07/E08? (inference only)

**Status:** planned 2026-10-28.
- This file, `config.yaml` and `registration.json` are committed **before** any new generation.
- The selected settings (`reports/selected_settings.json`) are committed **before** any historical-eval scoring of
  a new setting.

**Fixed for this phase:**
- **No training:** no adapter updates and no synthetic labels. The checkpoints are E07 `models/e07/full/best.pt`
  (update 1,200) and E08 `models/e08/rs_s0/best.pt` (update 3,000), both hash-checked.
- **No downloads or paid services:** CPU only, 6 threads, one model loaded at a time.
- **Unchanged:** E01–E08 results, predictions and checkpoints, the product default, serving, protected references.
  E02 final references and the sealed reserve are not opened (`e06.guard`).
- **Budget:** at most 4 h of active experiment time (prediction + analysis), counted in the ledger. RSS ≤ 5 GiB
  with ≥ 1.5 GiB available (`e07.Guard`, 3 consecutive bad readings → graceful stop between batches).

## What was known before this plan (stated so it is not mistaken for a result)
- **E08 update 3:** eval greedy top-1 beat beam-4 top-1 by about 5 points (E08 52.4 vs 47.5%; E07 49.9 vs 47.2%).
- **Saved dev beam lists:** before writing this plan, the saved dev beam lists
  (`experiments/E0{7,8}*/predictions/*_dev.jsonl`) were read. E08's lists often hold an empty candidate (≈ −29)
  and a one-character candidate (e.g. *arab* → `អ`, −9.7), while greedy gives a longer output.
- **Hypothesis H1** (from those lists, untested): the registered beam call uses `early_stopping=True`. In
  Transformers this ends the search as soon as `num_beams` finished hypotheses exist. With byte targets, an early
  EOS can enter the finished set in the first few steps, so junk short hypotheses could fill it and stop the search
  before the better long hypotheses finish.
  - Transformers' own note says canonical beam search is `early_stopping="never"` with `length_penalty=0.0`.
  - H1 predicts that beam top-1 often has a **lower** model log-probability than the greedy output, which
    canonical beam search would rarely allow.

## 1. Starting point and reproduction (`reports/environment.json`, `reports/reproduction.json`)
- **Recorded:** versions, the git revision, and hashes of sources, checkpoints, base model, references and saved
  predictions; tokenizer and special ids.
- **Reproduction:** both models, all 276 dev items. The registered code (`e07.words_gen`) is run with the
  registered settings:
  - greedy, batch 32;
  - beam 4 with 4 returned, `length_penalty 0.0`, `early_stopping True`, batch 16;
  - `max_new_tokens` 121, no sampling, no logits processors.

  The results are compared item by item with the saved files (texts, statuses and scores to 1e-4). A mismatch is
  investigated before anything is interpreted. Saved files are never overwritten.

## 2. Audit (`reports/generation_audit.md`)
Checks, each pass or fail with evidence:
- **Model state:** adapter keys loaded strictly and checkpoint hashes; eval mode; dropout off (same output twice).
- **Inputs:** identical input ids and masks for greedy and beam.
- **Generation config:** `do_sample False`, decoder start 0, pad 0, EOS 1, no extra processors; the effective
  `generation_config` is dumped.
- **Lengths:** EOS and padding handling; byte budget (every reference fits in 121 bytes; count of `no_eos`).
- **Decoding output:** UTF-8 decoding and malformed counts; NFC on outputs and references; the top-1 extraction
  rule (the first non-empty, well-decoded candidate).
- **Scores:** `sequences_scores` versus the teacher-forced sum log-prob including EOS (lp 0 → equal), and the greedy
  score versus its teacher-forced log-prob.
- **Alignment:** item-id alignment of predictions and references.
- **H1:**
  - the share of items where beam top-1's teacher-forced log-prob is below greedy's;
  - the rank and status of empty or one-character candidates;
  - whether search stops before greedy's length.

**Classification:** implementation defect / configuration effect / model preference (ranking) / reference-spelling
mismatch. A confirmed defect gets a failing regression test first, then the smallest fix. Original and corrected
results are reported separately.

## 3. Development comparison (`reports/dev_comparison.json`)
- **Data:** E03 dev, primary items, the same set as checkpoint selection.
- **Settings for each model** (everything else fixed as in §1):

  | Id | Setting |
  |---|---|
  | `greedy` | registered |
  | `beam2_reg`, `beam4_reg`, `beam8_reg` | registered call (lp 0, `early_stopping True`), k returned |
  | `beam2_can`, `beam4_can`, `beam8_can` | the same with `early_stopping "never"` (canonical); only these differ |

  The `_can` rows exist as a pre-specified contingency for H1. They change exactly one flag and are run whether or
  not H1 is confirmed, so the comparison is not chosen after the fact.
- **Metrics:**
  - top-1 macro and mean CER (`khmer_translit.evaluation`);
  - raw rank-1 status (ok, empty, malformed, no EOS);
  - top-1 validity (`e06.orthography`), empty and truncated counts;
  - repeated output (a 1–4 code-point unit repeated ≥ 3 times in a row);
  - mean top-1 length in code points and bytes;
  - generation seconds;
  - distinct candidates per item and top-k oracle, as diagnostics only (not quality).
- **Score reconstruction:** teacher-forced log-probs of greedy, every beam candidate and every reference
  alternative.
- **Conditional follow-up (at most one extra beam-4 setting):**
  - **Trigger:** only if, after the canonical runs, `beam4_can` top-1 is **shorter** than greedy on ≥ 60% of items
    where they differ **and** `beam4_can` dev top-1 is below greedy. That would indicate a length problem.
  - **Setting:** `beam4_can` with `length_penalty 1.0`, i.e. scores per generated byte.
  - **Before running it:** the evidence, value and expected effect are written to the ledger. Otherwise it is
    omitted.
- **Bounds:** no other settings, and no combined sweeps.

## 4. Disagreements (`reports/disagreements.json`)
- **Sample:** dev, each model's greedy versus registered beam-4 top-1. 5 items per group per model (40 in total),
  seeded (`e09:sample:{model}:{group}`).
- **Groups:** greedy right / beam wrong; beam right / greedy wrong; both wrong and different; agreement.
- **Saved per item:**
  - the English word and the reference alternatives as stored;
  - the greedy output and every beam candidate (registered and canonical), with rank, beam score, teacher-forced
    log-prob, bytes and EOS status;
  - the references' teacher-forced log-probs.
- **Agent inspection** (labelled non-independent): missing syllables or endings, consonant or vowel/diacritic
  changes, shorter outputs, plausible spellings the reference rejects, and invalid or truncated bytes.
- **No changes to the data:** no reference alternatives are invented, and no references are changed.

## 5. Selection (frozen before eval; `reports/selected_settings.json`)
- **Per model, among all settings run on dev:** highest dev top-1 macro, then lower dev mean CER, then lower
  generation seconds.
- **Freezing:** the setting, its hash and the checkpoint hashes are committed before eval generation or scoring.

## 6. Historical eval (E03 eval primary, 228 items, 222 families; exploratory)
- **Settings per model:** registered beam-4, greedy, and the selected setting if different.
  - The registered beam-4 and greedy eval predictions are the saved E07/E08 files: no rerun, as long as the dev
    reproduction is exact.
  - A new setting is generated once, then frozen and hashed before scoring.
- **Contrasts:** paired family bootstrap (10,000, seed 20261027), each model's checkpoint held fixed.
  - Decoding:
    - E07 selected − beam4_reg, and E07 greedy − beam4_reg;
    - E08 selected − beam4_reg, and E08 greedy − beam4_reg.
  - Models:
    - E08 − E07, both greedy;
    - E08 − E07, both selected;
    - E08 − E07, both beam4_reg (for reference).
- **Not a headline:** a mixed comparison (E08 greedy vs E07 beam-4).
- **Limits:** intervals are conditional on these two checkpoints and say nothing about seed variance. Selection on
  dev adds uncertainty that is not included.
  - A decoding gain is not evidence that E08's data helped.
  - Agreement with mined references is not phonetic acceptability.
  - This eval set has been examined repeatedly.

## Not authorised
Training, a product-default or serving change, distillation, merge, release or publication, and another experiment
started automatically.
