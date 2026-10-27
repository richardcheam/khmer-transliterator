# E03 decisions (assessable supervision, letters vs phones)

Routine decisions made while executing the owner's E03 instruction (2026-10-27). Consequential results go to
`docs/DECISIONS.md` (D180).

| # | Date | Decision | Why |
|---|---|---|---|
| 1 | 2026-10-27 | Experiment id E03, folder `experiments/E03-assessable-phones/` | Next unused id |
| 2 | 2026-10-27 | The repair lives in `scripts/e03_data.py` (`classify`, `recovery_reason`); the miner and E02's build are not edited | Owner: do not change E02's historical build in place; the mined files keep their stored class, which the new rule reproduces exactly |
| 3 | 2026-10-27 | Latin-side romanised Khmer stays excluded, including detector collisions with common English words (recorded separately) | Declared policy kept from E02; the detector is a heuristic list derived from the Chhunneng data, so collisions are not settled either way |
| 4 | 2026-10-27 | Train only on DACS pairs, not E02's 10,031 rows | The question is phone input under assessable supervision; E02's rows have no exact readings |
| 5 | 2026-10-27 | Held-out sets recomputed by an isolated step from the files E02 held out | No cached inventory existed; only sets and counts leave the step, and the read is recorded |
| 6 | 2026-10-27 | A training pair whose target is accepted in dev or eval is removed (250) | E02's held-out-target rule, applied across E03's own splits |
| 7 | 2026-10-27 | Splits 80/10/10 by family (`split_bucket`, salt `e03-assessable-phones`); 4% of training families form train_val | Enough eval families for a paired family bootstrap; train_val only breaks checkpoint ties, as in E02 |
| 8 | 2026-10-27 | Quarantine: compound extra component = established misalignment; other suspected conflicts = unresolved; the 4 unreadable = unknown | Owner: quarantine clear misalignments and unresolved contradictions; keep compatible alternatives |
| 9 | 2026-10-27 | Vowel information reported only (syllable count vs English vowels) | Owner: no unvalidated vowel-adaptation rules |
| 10 | 2026-10-27 | E02's trainer and runner reused through thin wrappers (`e03_train.py`, `e03_run.py`) | Same model, optimizer, schedule, pairing and parity code; E02 files untouched |
| 11 | 2026-10-27 | The unchanged 500-step warm-up covers 69% of E03's 720 steps; kept | Owner: established schedule, no sweep; declared before training |
| 12 | 2026-10-27 | No re-ranker in E03 | Every reference is a lexicon word: a lexicon re-ranker would leak answers |
| 13 | 2026-10-27 | E03 eval items use the split label `final` inside `eval_references.jsonl` | `reference_data` has no `eval` split; documented as E03's eval partition everywhere |
| 14 | 2026-10-27 | Commits omit Claude attribution lines | Owner's standing rule |
