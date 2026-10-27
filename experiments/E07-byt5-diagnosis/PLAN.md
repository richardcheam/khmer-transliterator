# E07 — ByT5 diagnosis: why were E06's outputs nearly constant, and can this setup learn the task?

**Status:** planned 2026-10-27.
- This file, `config.yaml` and the tiny-subset configuration (`scripts/e07.py TINY`, `TINY_GATE`) are committed
  **before** any E07 training.
- The full-adaptation recipe (`full_recipe.json`) is written and committed with `registration.json` **after** the
  tiny gate passes and **before** full adaptation. If the gate fails, it is never written.

**Fixed for this phase:**
- E01–E06 results and checkpoints, the product default (E02 `letters_s0`), serving, E02's final references, the
  sealed reserve and the E03 split boundaries stay unchanged;
- no new downloads (the pinned `google/byt5-small` from E06 is reused), no paid services, GPU, TTS/audio,
  synthetic labels or distillation.

## Questions
1. What caused E06's near-constant generation?
   - **Verified defect:** confirmed by a check that fails and is fixed, with a regression test.
   - **Adaptation limitation:** the pipeline is correct but the recipe or budget did not learn the task.
   - **Unresolved:** neither shown.
2. Can the existing pipeline learn an input-dependent mapping at all? Tested by memorising 32 pairs.
3. Only if (2) passes: does one corrected full adaptation learn input-dependent transliteration, and how does it
   compare with the letters baselines on the historical E03 eval set?

## 1. Starting point (`reports/environment.json`, `reports/e06_reproduction.json`)
- **Recorded:**
  - git revision;
  - source and data hashes, E03 sizes and split ids;
  - base revision and checkpoint hash, E06 adapter hash;
  - Python / PyTorch / Transformers versions; PEFT is not installed, because the LoRA is the repository's own
    module;
  - threads and limits.
- **Research environment:** `research/requirements-byt5.txt` (pins) and `research/requirements-byt5.lock.txt`
  (full freeze), outside `uv.lock`, so serving and `uv sync` are not affected.
- **Reproduction:** E06's selected adapter on a fixed 16-row sample (8 optimisation + 8 train-val rows,
  sha256 order; no dev/eval). It saves raw input ids, masks, decoded inputs, labels, decoded targets, generated ids
  and outputs.

## 2. Pipeline audit (`reports/pipeline_audit.json`, `reports/pipeline_audit.md`)
Ten checks:
1. distinct unmasked inputs;
2. target alignment;
3. masks and padding;
4. −100 label padding ignored (manual loss = model loss);
5. decoder shift, start id and EOS read from the model config;
6. adapter locations and optimiser parameters;
7. finite gradients and actual updates, with base weights unchanged;
8. save → reload → identical ids, and a zero-initialised adapter gives exactly the base logits;
9. byte limits vs generation length;
10. greedy and beam scores equal teacher-forced sums.

**E06 loss decomposition:** E06's loss is split by byte class, comparing correct, shuffled and empty inputs.
- Byte classes: the Khmer UTF-8 lead byte, the block byte, the final byte, and EOS.
- Hypothesis: the per-byte average is dominated by near-deterministic lead and block bytes, so it can fall while
  the informative final byte stays input-independent.

## 3. Tiny memorisation diagnostic (frozen; `scripts/e07.py TINY`)
- **Data:** 32 pairs from the E03 optimisation split (`train` part).
  - Each input word occurs once with one target, and each target occurs once.
  - 8 per word-length bin (≤ 4, 5–6, 7–8, ≥ 9 letters), in sha256(`e07-tiny:`+word) order.
  - The selection never depends on model outputs, and it is checked disjoint from dev/eval inputs.
- **Model:** starts from the downloaded base, not E06's adapter. It uses the same encoding, labels, adapter
  module, saving and generation as the full run.
- **Configuration:**
  - LoRA r 16, α 32 (scale 2, as E06), on attention q/k/v/o **and** the gated FFN wi_0/wi_1/wo of every block:
    6.26M trainable parameters;
  - adapter and base dropout 0;
  - AdamW, lr 1e-3, 20-update warm-up, then constant; batch 16, clip 1.0, seed 0.
- **Why this configuration (from the audit):** memorising 32 arbitrary pairs needs capacity and no regularisation.
  - E06's attention-only adapter was designed for generalisation (dropout 0.1, decaying LR).
  - PEFT's documentation notes that targeting all linear layers can approach full fine-tuning.
  - The Transformers T5 page recommends 1e-4 to 3e-4 for full fine-tuning; LoRA commonly uses a higher rate.
    E06's 1e-3 is kept, so the diagnostic tests the pipeline rather than a new rate.
- **Budget:** ≤ 1,000 updates or 45 minutes per attempt. Greedy evaluation of the 32 every 100 updates; stop early
  at 32/32.
- **At most two attempts.** A second attempt needs a recorded diagnosis and exactly one changed setting
  (`reports/tiny_attempt2_change.json`, enforced in code). No sweep.
- **Gate (all required):**
  - greedy, eval mode, NFC exact match on **≥ 30 of 32**;
  - outputs depend on inputs: ≥ 30 distinct outputs; a shuffled input reproduces the *other* word's target for
    ≥ 30 of 32; shuffled-input loss > correct-input loss;
  - a fresh reload of the saved adapter gives identical ids;
  - all audit checks pass.
- **Reported:** the complete 32-row table, byte-level loss by class, output diversity and the input-dependence
  measurements.
- **Limits of a pass:** it shows only that the setup can memorise this subset, not generalisation and not
  bug-freedom.
- **If the gate fails within budget:** full adaptation is skipped and the phase ends with the investigation.

## 4. One corrected full adaptation (only after a pass)
- Registered in `full_recipe.json` + `registration.json` and committed before it starts:
  - recipe and fixes;
  - adapter targets, rank, scaling and dropout;
  - optimiser, lr, schedule and accumulation;
  - maximum updates and limits;
  - dev selection rule;
  - metrics and decoding;
  - data and source hashes.
- **Data:**
  - the E03 optimisation split (2,255 pairs, verified);
  - train-val (93) stays a loss/input-dependence diagnostic only;
  - dev (primary items) is used only for checkpoint selection;
  - no eval targets in any tuning, screening or selection.
- **Run:** one run from the base model, never from the tiny adapter. ≤ 3,000 updates or 6 hours. Throughput is
  measured early, and the registered time cap applies as is.
- **Checkpoint selection:** dev evaluated at ≤ 5 predetermined updates including the final one. Highest dev greedy
  top-1 macro (exact reference agreement), then lower mean CER, then earlier.
- **Fixed after registration:** the recipe and budget; nothing is extended after seeing results.

## 5. Evaluation (only after a full run)
- **Frozen first:** predictions for dev, eval and the E06 pool (greedy, and beam 4 top-4 sum log-prob) are written
  and hashed before eval scoring.
- **Historical E03 eval** (228 primary items; mined, unreviewed; already used by E03–E06; a historical research
  comparison, not a fresh benchmark):
  - **Measures:**
    - top-1 exact (beam-4 top-1, the same decoding as the G baselines; greedy also reported);
    - mean CER (`khmer_translit.evaluation`: min edit distance / reference length over accepted spellings);
    - validity (`ok` / `empty` / `malformed_utf8` / `no_eos`, orthography rules);
    - diversity (distinct top-1, most frequent top-1 share);
    - strata by word length and source, with the ambiguous track separate.
  - **Compared with:**
    - E06 ByT5 (its saved beam-4 predictions);
    - the product default (E02 `letters_s0`, its own decoder, top-4);
    - E04 G_s0, G_s1, G_s2 individually and their mean (saved beam-4 predictions).
  - **Uncertainty:** paired family bootstrap, 10,000 resamples, seed 20261027. Intervals are conditional on these
    checkpoints, since one ByT5 run measures no seed variability.
  - **Equal budgets:** oracle top-4 is reported only at equal budgets (beam 4 vs beam 4).
- **Input dependence (diagnostic, not quality):**
  - samples: train-val 93 and the fixed 16-row sample;
  - inputs: correct vs shuffled (fixed derangement) vs empty (EOS only);
  - measured: loss per byte class and greedy-output changes.
- **Fresh E06 pool (56 words):**
  - membership unchanged;
  - operational checks and labelled heuristics only (orthography, consonant screen), never accuracy.
- **Cost:**
  - load time and batch-1 latency (greedy, beam 4);
  - greedy throughput and peak RSS;
  - adapter and base size;
  - hardware.

## 6. Outcome classes (one is reported)
- **Diagnostic failure:** tiny memorisation or pipeline checks failed; full adaptation skipped.
- **Pipeline repaired, adaptation still unsuccessful:** the gate passed, but the full run is still collapsed or
  substantially unsuccessful (dev top-1 < 10%, or fewer than half as many distinct eval top-1 outputs as the G
  mean).
- **Task learned, no clear advantage:** input-dependent (the shuffled-input loss on final bytes exceeds the
  correct-input loss by ≥ 0.5 nats on train-val, and the diversity condition is met); the eval top-1 interval vs
  the G mean includes 0 or is negative.
- **Promising ByT5 research candidate:** input-dependent, and the eval top-1 interval vs the G mean is above 0.
  Independent phonetic quality remains unverified.

No outcome authorises distillation, publication or deployment. Memorisation or agreement with mined references does
not make ByT5 a qualified teacher.

## 7. Limits (registered in `scripts/e07.py LIMITS`; enforced by `Guard`, `budget_left` and `fit`)
- **Compute:** CPU only, one training job at a time, ≤ 6 threads.
- **Runs:** 2 tiny attempts (≤ 1,000 updates / 45 min each) and 1 full run (≤ 3,000 updates / 6 h).
- **Time:** ≤ 8 training hours total and ≤ 12 wall hours, from the ledger's first entry.
- **Memory:** RSS ≤ 5 GiB and MemAvailable ≥ 1.5 GiB. Three consecutive bad readings stop the run gracefully and
  leave it resumable.
- **No downloads or requests.**
- **Ledger:** `runs/ledger.jsonl` (start, segment, done).
- **Resume:** from `last.pt`, refused unless the config, data and base hashes match. Optimiser, scheduler and RNG
  state are restored, and completed updates are never repeated.

## Sources (recorded in `SOURCE_VERIFICATION.md`)
- the ByT5-small model card;
- ByT5 (Xue et al., TACL 2022);
- Kirov et al. 2024 (CL 50(2));
- Transformers v5.19.0 T5 docs;
- PEFT LoRA docs.
