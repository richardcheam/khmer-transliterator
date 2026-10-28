# E10 decisions log (release cycle)

Results summary: `docs/DECISIONS.md` (D188).

| # | Date | Decision | Why |
|---|---|---|---|
| 1 | 2026-10-28 | Id E10, one folder for the whole cycle (`experiments/E10-release-cycle/`) with state.json and a ledger | Next unused id; one cycle, one record |
| 2 | 2026-10-28 | One authoritative policy in `docs/PHONETIC_POLICY.md`, with three evidence tracks that are never pooled | Instruction §2 |
| 3 | 2026-10-28 | The dev-review batch and its key stay local (`data/e10/review/`); only instructions and hashes are committed | The batch contains mined references (data policy: pair manifests stay local) |
| 4 | 2026-10-28 | The confirmation manifest also excludes every E08 training word and synthetic input, by family | Fresh means unseen by the candidate as well as unprotected |
| 5 | 2026-10-28 | One candidate recipe (A: word-level sampling + LR 2e-3), not two | A second recipe would vary A's components (an ablation); capacity is not indicated |
| 6 | 2026-10-28 | `e07.fit` gains an optional `order` hook; the default is unchanged | Word-level sampling without changing E07/E08 reproduction |
| 7 | 2026-10-28 | Commits omit Claude attribution lines | Owner's standing rule |
