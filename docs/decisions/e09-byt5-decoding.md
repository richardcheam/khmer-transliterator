# E09 decisions log (ByT5 decoding, inference only)

Results summary: `docs/DECISIONS.md` (D187).

| # | Date | Decision | Why |
|---|---|---|---|
| 1 | 2026-10-28 | Id E09, folder `experiments/E09-byt5-decoding/`; plan, config and registration committed before any generation | Next unused id; protocol first |
| 2 | 2026-10-28 | H1 (early stopping with byte targets) stated in PLAN.md from the saved dev beam lists, before any new generation; canonical (`early_stopping="never"`) settings pre-specified as a contingency and run regardless | No after-the-fact choice of comparison |
| 3 | 2026-10-28 | Reproduction on all 276 dev items, both models; saved files never overwritten | Instruction §2 |
| 4 | 2026-10-28 | Classified as a configuration defect (search error), not an implementation defect; regression test with a scripted model failed first; fix = `e07.gen` arguments with registered defaults | Saved E07/E08 predictions must still reproduce |
| 5 | 2026-10-28 | Length-penalty follow-up not run (trigger: canonical beam shorter on ≥ 60% of differing items; observed 39% / 47%) | Pre-registered trigger not met |
| 6 | 2026-10-28 | Selection (dev top-1, CER, seconds): E07 beam2 canonical, E08 beam8 canonical; committed before eval | Instruction §6 |
| 7 | 2026-10-28 | Saved eval files reused for registered beam-4 and greedy (dev reproduction exact); only selected settings generated | Deduplicate identical configurations |
| 8 | 2026-10-28 | E08's pre-registered "small or uncertain" verdict (D186) left as recorded; the matched comparison is reported, not re-classified | The rule was registered for beam-4; no retroactive re-scoring |
| 9 | 2026-10-28 | Commits omit Claude attribution lines | Owner's standing rule |
