# E05 decisions log (candidate diversity)

Consequential choices made while executing E05. Results summary: `docs/DECISIONS.md` (D182).

| # | Date | Decision | Why |
|---|---|---|---|
| 1 | 2026-10-27 | Id E05, folder `experiments/E05-candidate-diversity/` | Next unused experiment id |
| 2 | 2026-10-27 | E04 report clarified in place with a dated section; numbers unchanged; D181 annotated | Instruction: correct three interpretive statements without overwriting registered results |
| 3 | 2026-10-27 | Beam-8 generated for all four models on dev and eval; beam-4 generated only for dev G_s1/G_s2 (primary) | E04 saved beam-4 only (eval all models; dev primary G_s0/P_s0); generate only where saved outputs cannot support a configuration |
| 4 | 2026-10-27 | Dev beam-4 analysis restricted to primary items | E04's saved dev outputs cover primary items only |
| 5 | 2026-10-27 | Dictionary-hit stratum replaced by "G_s0 top-1 is a lexicon word" | Every DACS primary reference is a lexicon headword (228/228 eval, 253/253 dev), so a reference-based dictionary stratum is degenerate |
| 6 | 2026-10-27 | Bootstrap with NumPy `default_rng(20261027)`, one shared family-draw matrix for all systems | Paired comparisons need identical draws across systems; same seed as earlier experiments (generator differs from `evaluation.bootstrap_ci`) |
| 7 | 2026-10-27 | Review queue includes the mined reference(s) as blinded candidates; stored in git-ignored `data/e05/review/` | Reviewing the mined references is the missing prerequisite named in E03/E04; "local" and "do not publish" |
| 8 | 2026-10-27 | Commits omit Claude attribution lines | Owner's standing rule |
| 9 | 2026-10-27 | `G8` at beam-4 reported as under-filled rather than built from beam-8 | Beam-4 returns at most 4 candidates; mixing widths is not allowed |
| 10 | 2026-10-27 | Strata reported as item coverage without intervals; the flagged track as family macro | Groups are small (14–135 items); descriptive only |
