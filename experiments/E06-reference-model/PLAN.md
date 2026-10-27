# E06 — Reference model: does a pretrained byte model add English → Khmer phonetic knowledge?

**Status:** registered 2026-10-27, after the feasibility check and before full adaptation and before any
reference-model generation. The registration commit is the first commit containing this file; its hashes are in
`registration.json`.

**Fixed for this phase:**
- previous results, checkpoints, benchmarks, the sealed reserve, the product default (E02 `letters_s0`) and
  serving stay unchanged;
- no paid requests, GPU, cloud, TTS/audio or third-party messages.

## Questions
1. Can a stronger reference model give useful English → Khmer phonetic knowledge beyond the current small letters
   model?
2. Only if (1) is established by independent review: can a compact student inherit it?

## Reference-model routes (`SOURCE_VERIFICATION.md`)
- **Route A, instruction model (Gemini): unavailable.**
  - There is no authorised free programmatic interface here: no API key in the environment or the project, and
    no Gemini client installed.
  - The owner's favourable experience was through the chat interface. Scraping it, creating accounts or making
    paid calls are not allowed.
  - The frozen prompt is still recorded (`prompts/route_a_prompt.txt`), so a later authorised run would use the
    same text. No small development check of the prompt was possible.
- **Route B, `google/byt5-small`: feasible.**
  - Pinned revision `68377bdc…`, Apache-2.0, 300M parameters (12 encoder / 4 decoder layers, d_model 1472).
  - The model card says it must be fine-tuned before use.
  - Downloaded: `config.json`, `generation_config.json`, `pytorch_model.bin` (1.12 GiB; sha256 verified). The TF
    and Flax copies were not downloaded.
  - Python packages: transformers 5.19.0 + 17 small dependencies, 62 MiB in the uv cache.
  - Total downloads: about 1.18 GiB of the 2 GiB budget.

## Task definition (frozen)
- **Input:** one English word or name, pronounced as English. **Output:** readable Khmer script approximating that
  pronunciation. Semantic translation is excluded; an established Khmer equivalent counts only if pronounced
  similarly.
- **Pronunciation policy:**
  - the default English reading is CMUdict's first listed (US) reading;
  - words with several CMUdict readings that differ beyond stress are flagged `pronunciation_ambiguous` and
    reported separately;
  - words without a CMUdict entry are flagged `dictionary_miss`.
- **Reference interface: letters only.** ByT5 receives the lower-cased English word as UTF-8 bytes and **no**
  pronunciation, the same input as the letters baseline. The difference tested is pretraining and capacity, not
  extra input.
- **Practical baseline:**
  - the product default E02 `letters_s0`;
  - for same-data comparisons, E04's `G_s0`, `G_s1`, `G_s2`, trained on the same E03 partition as ByT5.

## Evidence levels (always reported separately)
1. **Independently reviewed acceptable outputs** under this policy: none exist. The blinded queue below is
   prepared, not collected.
2. **Attested but unreviewed spellings:** E03 dev/eval (mined, consonant-screened). Historical, not fresh.
3. **Model-generated or automated proxies:**
   - orthographic well-formedness rules;
   - the frozen E02-diagnostics consonant screen;
   - Khmer-lexicon membership;
   - agreement between systems.

   These are checks, not correctness.

## Fresh pilot pool (`pool.jsonl`, `reports/overlap_audit.json`; built before any new output)
- **Candidates:** SCOWL (`data/en_common_words.txt`, and `/usr/share/dict/american-english` capitalised entries as
  names) with CMUdict readings.
- **Freshness rule:** none of the word's stems (`reference_data.stems`, the lexical-family key) occurs among the
  stems of any Latin token in the historical material. That material covers:
  - every E02–E05 data file;
  - the mined pairs, the LLM-drafting rounds (`data/synth/`, `llm_drafts*`), owner picks, names and romanised sets;
  - the speech repo's loanword list;
  - every report, doc and handoff file, which covers inspected examples;
  - only the Latin inputs of the benchmark panels and the reserve.

  E02's final references and forbidden results were never opened. Their inputs come from the excluded sources.
- **Categories:** 8 words each in sha256(`e06-pool:`+word) order, one word per family. The primary track is
  ordinary, name, difficult (a registered spelling-to-sound rule), short (3–4 letters), long (≥ 11) and
  dictionary miss: 48 words. A separate ambiguous track adds 8.
- **Use:** the pool has no attested or qualified references by construction, so it supports only level-3
  diagnostics and the review queue. Once reviewed and used for any selection, it would become development
  evidence. Confirmation would need separate, untouched material.

## Adaptation recipe (one recipe, registered; `scripts/e06_byt5.py RECIPE`)
- **LoRA:** r 16, alpha 32, dropout 0.05, on q/k/v/o of every attention block (encoder self, decoder self,
  decoder cross).
  - The base is frozen, with its own dropout 0.1.
  - Trainable parameters: 2,375,680 of 302,013,440.
- **Optimiser:** AdamW, lr 1e-3, no weight decay, gradient clip 1.0.
- **Batch and schedule:** batch 16 (no accumulation), linear warm-up over 50 updates, then linear decay to 0 at
  1,000.
- **Stop:** 1,000 updates or 4 elapsed hours, whichever comes first.
- **Data:**
  - E03 training partition: 2,255 optimisation rows, 93 train-val rows (loss tie-break);
  - byte limits: source ≤ 32, target ≤ 120 bytes;
  - oversized rows are rejected and counted, never truncated (0 at feasibility).
- **Selection:** every 100 updates by greedy exact-match macro on E03 dev primary items; ties go to the lower
  train-val loss, then the earlier update.
- **Decoding:** beam 4, top 4, sum log-prob (length_penalty 0, checked against teacher-forced scores),
  ≤ 120 new bytes.
  - Byte sequences that are not valid UTF-8 count as `malformed_utf8`; empty outputs as failures.
- **Feasibility check** (`reports/feasibility.json`): 1.9 s per update, about 0.7 h for 1,000 updates, peak RSS
  3.9 GB. Unadapted zero-shot outputs were malformed UTF-8, as expected for an unadapted model. That is not an
  assessment of transfer.
- **Not established by an inadequate run:** a weak result from this small, short adaptation does not show that
  pretrained transfer is ineffective. Kirov et al. fine-tuned ByT5-*base* for 50,000 steps.

## Comparisons (`scripts/e06.py compare`)
- **Historical E03 eval** (level 2, 228 primary items):
  - ByT5 top-1 and top-4 vs each G seed and their mean;
  - paired family bootstrap, 10,000 resamples, seed 20261027;
  - dev is reported descriptively, because ByT5 is selected on it.
- **Pool** (level 3), for ByT5, G_s0–s2 and the product default:
  - orthography, consonant screen, lexicon membership, refusals/malformed outputs and agreement, by category;
  - the ambiguous track separately.

## Blinded review queue (`data/e06/review/`, git-ignored, local)
- **Content:** all 56 pool words. Candidates are the deduplicated union of ByT5, G_s0 and the product default
  top-4, in random order with the origin hidden. Several may be acceptable; "uncertain" is allowed.
- **Origin key:** a separate file.
- **Not done:** nobody is contacted, nothing is published, no judgment is filled in by the agent.

## Teacher-quality gate (frozen before any judgment; `scripts/e06.py GATE`, `gate_decision`)
- **Needs:** independent judgments in `data/e06/review/judgments.tsv` by a reviewer who is neither a model nor
  the agent.
- **Unit:**
  - one primary pool word;
  - a system's top-1 counts as accepted if judged acceptable ("uncertain" = not accepted; a sensitivity analysis
    treats it as accepted);
  - paired ByT5 − product default.
- **Interval:** bootstrap over words, 10,000 resamples, seed 20261027, 95% percentile.
- **Pass requires all of:**
  - ≥ 40 judged primary words;
  - more ByT5 acceptances;
  - interval lower bound > 0;
  - a positive difference in ≥ 3 categories, with no category supplying more than half of the net wins;
  - malformed outputs, translations and unsupported cases reported;
  - teacher-output use permitted (Apache-2.0).
- **Otherwise:** `teacher_advantage_unresolved`, or `teacher_not_better`. Without judgments the result is
  `teacher_advantage_unresolved`, and distillation (Task 8) is skipped.

## Distillation (Task 8; only if a teacher passes)
- At most 1,000 eligible inputs, generated once.
- Control vs treatment students: seed 0 first, seeds 1–2 only if dev improves by ≥ 2 points.
- At most 6 runs, no architecture change.
- Not run unless the gate passes.

## Constrained-discovery feasibility (Task 9; read-only)
- **Words:** 200 fresh words (not in the pool), frozen in `reports/discovery/words.json` before any candidate
  for them was generated.
- **Candidates:** ByT5 and G_s0 top-4 (strings ≥ 3 characters) serve only as search constraints.
- **Search:** the local kmwiki dump and the FineWeb-2 shard, for documents that contain the English word and a
  candidate Khmer string, within 60 characters (contextual) or anywhere in the document.
- **Rules:** existing-target discovery is kept distinct from invention. Nothing found is kept as a label.

## Budgets
- **Compute:**
  - CPU, one model job at a time, ≤ 6 threads;
  - ByT5 adaptation ≤ 4 h;
  - ≤ 12 aggregate training hours;
  - ≤ 6 student runs.
- **Downloads:** ≤ 2 GiB (about 1.18 GiB used).
- **Requests:** Route A has no requests (unavailable); the cap of 1,100 word requests is unused.
- **Progress:** `state/e06/status.json`, `PROGRESS.md`. Training resumes from `models/e06/byt5_lora/last.pt`
  (recipe and base hash validated).
