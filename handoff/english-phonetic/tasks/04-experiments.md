# Task 4 — Run the registered experiment to completion

**Outputs:** run ledger, bounded-job runner, selected candidate, paired results, final evaluation, budget accounting.

## Queue and runner

- [ ] Create/extend `scripts/english_phonetic_run.py`. The dry-run prints exact phases, IDs, commands, inputs, expected artifact paths and total caps without starting training. Actual `--resume` initializes or resumes that same queue. Use subprocess argument arrays rather than shell interpolation of words/paths.
- [ ] State transitions: `pending`, `running`, `interrupted`, `failed`, `completed`, `unavailable`, with reason and timestamps. Success requires expected outputs and a successful verification command, not merely process exit 0 or a log sentence. Write state atomically. Track per-job elapsed time including resumed segments and cumulative attempts/time/downloads.
- [ ] Test budget exhaustion, a live prior process, stale PID, failed export, partial checkpoint, changed config/hash, explicit interrupted resume and an already completed job. Never execute the same job twice because a monitor lost its last response. Checkpoint model/optimizer/scheduler/step/epoch/RNG/data generator state for deterministic continuation.
- [ ] Before launching, verify frozen data, benchmark, config, source encoder and inference-resource hashes. Start W&B mirroring only if existing credentials/network work; local JSONL is authoritative. W&B failure does not abort training or require the owner to log in.
- [ ] Run one full seed-0 pilot for `letters`, `phones`, `phones_p2g` (skip unavailable C). Run the C pretraining immediately before its supervised run. Keep per-arm per-seed folders distinct. A code/data correction after a full launch consumes an attempt; preserve superseded evidence and remain within the cap.
- [ ] Choose the candidate on validation primary score only. Ties prefer the simpler recipe: letters, then phones, then phones_p2g. Do not inspect final or v1 predictions to choose. If letters wins, package it and report no development-selected treatment; do not manufacture a confirmatory improvement comparison by selecting the best final score.
- [ ] If a treatment wins, run the selected treatment and letters control at seeds 1 and 2 automatically. Together with seed 0 these are 3 paired seeds, using at most 7 full supervised attempts in total. If C wins, use its corresponding bounded pretraining at each confirmation seed. Other treatments do not get extra seeds.
- [ ] Limit the confirmatory claim to the development-selected recipe versus letters. B versus C has only one pilot seed in this budget; it is exploratory. Even if C wins the three-seed comparison, do not claim a confirmed incremental P2G benefit over B, because B was not confirmed at seeds 1 and 2. The C recipe also combines pretraining and replay.
- [ ] If a job fails, diagnose/fix a routine implementation issue, preserving logs. Reuse completed valid checkpoints only when input/config semantics remain unchanged. No hidden spare run budget. An inability to complete all paired seeds means the confirmatory result is incomplete/inconclusive, not an invitation to ignore a failed seed.

## Reporting while jobs run

- [ ] Record supervised/P2G loss, development scores, gradient norms, update counts, effective source lengths, dictionary miss/ambiguity rates, samples/sec and elapsed time. Report P2G and replay overhead separately. Check finite values and operational health; do not tune the recipe from interim scores.
- [ ] Keep concise progress updates and durable resume state. If an external interruption occurs, the next resume must pick up the exact unfinished phase. Do not ask whether seeds 1 and 2 should run.

## Final evaluation

- [ ] Freeze chosen checkpoints and resource hashes before opening final predictions. Evaluate the full fixed final set once after selection/training; score all required paired seeds in that evaluation event. No final-score choice of checkpoint, reranker, phone reading or treatment.
- [ ] Save actual inputs, intended/provider phones, raw ordered candidates and scores, selected outputs, accepted-reference IDs, quality tiers, per-word/group outcomes and explicit abstentions/errors. Keep any restricted references in their permitted local location; reports may use IDs/aggregate results when redistribution is disallowed.
- [ ] Keep primary default-provider and optional supplied-pronunciation reports separate. A reference's intended phones are metadata for primary scoring, not a hidden source feature. Count erroneous provider pronunciations as pipeline errors; never inject oracle phones only for evaluation or retarget references after seeing outputs.
- [ ] Run exact metrics and paired group bootstrap defined in Task 2; report every seed. Report known/missing-pronunciation and seen/unseen lexical/target strata. A small subgroup with inadequate precision is descriptive, not a standalone superiority claim.
- [ ] Report fixed-reranker results separately and make clear if a conventional-word prior changes the apparent phonetic gain. Do not change reranker after seeing final results.
- [ ] Run v1 historical diagnostics only after selection is frozen, with its old scorer/rules unchanged. Do not pool romanised Khmer/names scores into the new primary score or add new adoption guards after seeing them.
- [ ] Apply reference qualification, effect evidence and adoption rules exactly. Mean below 5 is not proof of no benefit. A wide interval is `inconclusive`. A high automatic phonetic score is not independent human validation. Distinguish conditional exclusion of +5 from proof of equivalence or zero effect.
- [ ] If final references are unqualified, use exploratory/reference-agreement results and retain the scientific limitation. Do not call a preference-matching improvement a demonstrated pronunciation improvement.
- [ ] Choose the packaged default: promote treatment only if the frozen adoption rule and runtime guard pass; otherwise use the eligible letters baseline. Keep candidate weights as research artifacts if permitted. No old Chhunneng-derived model may quietly become the fallback.
- [ ] Record the decision and hashes, then continue to packaging even for a negative/inconclusive result.
