# E08 decisions log (data expansion)

Results summary: `docs/DECISIONS.md` (D185).

| # | Date | Decision | Why |
|---|---|---|---|
| 1 | 2026-10-28 | Id E08, folder `experiments/E08-data-expansion/`; no earlier E08 existed | Next unused id; this instruction replaces the earlier E08 training plan, so no E08 training was registered or run |
| 2 | 2026-10-28 | Data policy frozen before accepting any example (`DATA_POLICY.md`) | Instruction §2 |
| 3 | 2026-10-28 | Recovery only by rules for screen weaknesses documented before this phase; noise-filtered forms screened with the unchanged frozen screen | No model preference as acceptance authority |
| 4 | 2026-10-28 | Possible reading switches left uncertain; possible translations and unexplained 3+ conflicts → errors | No silent switch from the English reading |
| 5 | 2026-10-28 | Sources: ALT Parallel (CC BY 4.0) used; ECCC permission-blocked (signed form); Khmerologie lead only (no reuse terms; garbled extraction); Google lexicon auxiliary only (already local, circular with the screen) | Instruction §4; 20.4 MiB of downloads |
| 6 | 2026-10-28 | ALT acceptance: verbatim translator spelling + the frozen screen; letters model only proposes the search strings; stoplist, ≥ 3 characters, no dependent sign after the match (pilot fixes registered before full extraction) | Pilot found spurious substring hits |
| 7 | 2026-10-28 | Target equal to a dev/eval or held-out target → excluded (as E03); counts recorded | Keeps the R arm comparable with C's construction |
| 8 | 2026-10-28 | New rows from E03 train-val families excluded (22) | Train-val is a diagnostic set |
| 9 | 2026-10-28 | "Common" stratum ranked by FineWeb-2 Khmer-shard Latin-token counts | No frequency list is local; this counts English words Khmer web text uses |
| 10 | 2026-10-28 | Amendment 1, before any AI Mode output: abbreviation-like inputs excluded (9) and the manifest re-frozen | *etc*, *ftp*, *blvd* read as expansions or letter names, which the letters-only model cannot see |
| 11 | 2026-10-28 | Synthetic import validates structure only; accepted rows never re-asked; uncertain rows not regenerated | Instruction §7 |
| 12 | 2026-10-28 | Phase-B plan: word-level sampler for every arm, including C (so C is a new run, not E07's checkpoint); equal provenance weights; 3,000 updates per run | Alternatives must not multiply exposure; equal updates keep the comparison budget-matched |
| 13 | 2026-10-28 | Commits omit Claude attribution lines | Owner's standing rule |
| 14 | 2026-10-28 | The owner's extra round (`additional.md`, 130 rows with ids `e08sNNNb`) is kept as a separate generation. It fills only records quarantined for structural reasons (2: *nutrients*, *overview*) and never replaces or extends an accepted record; its agreement with round 1 is reported as self-consistency | No frozen prompt produced it; a repeated generation is not validation |
| 15 | 2026-10-28 | Amendment 2 (replacement prompts only, before their answers): a note to use Khmer Unicode only (ោ, not Lao ໂ) | 9 first-round preferred spellings mixed Lao/Tibetan look-alike letters; the frozen batch prompts are unchanged |
| 16 | 2026-10-28 | Importer fix: answered replacement rounds keep their prompt and id list; new rounds take the next number | A re-import had shrunk `regen_01`'s list and flagged its own answers as another batch's |
| 17 | 2026-10-28 | Data frozen with 193 synthetic words; the 5 remaining look-alike-script items and 2 uncertain stay quarantined; no third round requested | The same 5 failed again after the explicit warning; quarantine is the policy for malformed records |
| 18 | 2026-10-28 | Training amendment 1 (owner instruction): one exploratory R+S candidate vs the saved E07 checkpoint; C and R not trained; E07 row-level sampling kept; decision rule pre-registered | Discovery first, attribution later; avoid a simultaneous objective change |
| 19 | 2026-10-28 | Exposure check applies the protected-family rule to added rows only; 35 control words related to the wider E08 protected set are reported, not removed | The control is E03's immutable baseline, identical to E07's training data |
| 20 | 2026-10-28 | rs_s0 classified **small or uncertain change** (eval top-1 +0.3 [−5.7, +6.5] vs E07; CER not lower); no replication; no further runs | Pre-registered rule in TRAINING_AMENDMENT_1.md |
| 21 | 2026-10-28 | Remaining limitation recorded as label/reference quality, not quantity (24 gains vs 25 losses; 56 shared wrong answers); greedy > beam-4 for both models noted, not acted on | Amendment: identify the strongest evidence, launch no speculative fix |
