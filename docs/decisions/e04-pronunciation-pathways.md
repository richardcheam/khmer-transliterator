# E04 decisions (pronunciation pathways)

Routine decisions made while executing the owner's E04 instruction (2026-10-27). Consequential results go to
`docs/DECISIONS.md` (D181).

| # | Date | Decision | Why |
|---|---|---|---|
| 1 | 2026-10-27 | Experiment id E04, folder `experiments/E04-pronunciation-pathways/` | Next unused id |
| 2 | 2026-10-27 | Views built from the existing tokens (P = `<task:en>` `<sep>` phones); no new token | Identical vocabulary and parameter counts in every arm; the structure identifies the view, nothing identifies the word |
| 3 | 2026-10-27 | A new trainer in `scripts/e04.py` reusing E02's model, targets, padding, greedy decode, loss and checkpoint helpers | The new schedule, the update budget and per-example view sampling do not fit the E02 trainer; E02/E03 code untouched |
| 4 | 2026-10-27 | Research export format `e04-research-v1` with its own NumPy loader | The product API must not load research checkpoints |
| 5 | 2026-10-27 | Sanity schedule: 50-update warm-up over 500 updates, batch 64 (the whole subset) | The 500-update sanity budget; the same sequential shape |
| 6 | 2026-10-27 | Branch metrics use beam-4 dev top-1 macro; checkpoint selection uses greedy | Instruction: greedy for selection, fixed beam-4 for evaluation |
| 7 | 2026-10-27 | "GP strongest" means GP ≥ G and GP ≥ P on dev | A tie keeps GP, the simpler of the two joint options |
| 8 | 2026-10-27 | Donor perturbations not repeated in E04 | Secondary in E03; the instruction asks for legitimate alternatives |
| 9 | 2026-10-27 | Sources verified on abstract pages; the CVPR and IEEE pages were unavailable, so their arXiv versions were used | Recorded in SOURCE_VERIFICATION.md |
| 10 | 2026-10-27 | E03 vowel-count erratum added to its report; its data is unchanged | The 3,008 figure came from a stale build |
| 11 | 2026-10-27 | Commits omit Claude attribution lines | Owner's standing rule |
| 12 | 2026-10-27 | Added a post-hoc `retention` step (saved predictions only) with a seed-churn baseline | Question 2 asks whether GP keeps P's capability; one seed of P cannot separate that from retraining churn without a baseline |
| 13 | 2026-10-27 | The example table in REPORT.md comes from the saved eval predictions (seed 0) | Plain explanation for the owner; no new predictions |
