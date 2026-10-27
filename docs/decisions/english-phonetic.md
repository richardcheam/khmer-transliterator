# E02 English phonetic: engineering decisions

Routine choices made while executing the owner-approved handoff (`handoff/english-phonetic/`), each with its
reason. Consequential results go to `docs/DECISIONS.md` (D178).

| # | Date | Decision | Reason |
|---|---|---|---|
| 1 | 2026-10-27 | Work on branch `english-phonetic`; handoff placed at `handoff/english-phonetic/` | Handoff placement rule; main stays untouched |
| 2 | 2026-10-27 | Experiment id **E02** (`experiments/E02-english-phonetic`) | First unused id. The earlier "next candidate E02" (romanised-typing data) was never created; it becomes a later id |
| 3 | 2026-10-27 | Benchmark pool = owner-verified single words of the loanword list; tier `attested`, not `independent_reviewed` | They are the only human-checked English→Khmer spellings; they were reviewed under the TTS-list policy, not E02's, and 94 final items were picked among model E's candidates |
| 4 | 2026-10-27 | Benchmark declared **unqualified**; evidence status fixed to `inconclusive` before training | No independently reviewed references exist; the handoff forbids self-certification and new owner annotation |
| 5 | 2026-10-27 | Supervised data = web-mined pairs with a CMUdict entry, not Khmer-like, after the D135 noise filter (10,031 rows) | The only eligible clean English/name pairs; CMUdict hit = "has an English reading"; LLM drafts and conventional forms excluded by default |
| 6 | 2026-10-27 | Owner example picks end up in no training set | All 150 are also owner-verified list words, so they sit in the benchmark pool and the leak filter removes them |
| 7 | 2026-10-27 | Historical exclusions: v1 English panels, E01 list-val words and reserve Latin are kept out of training and the new splits (reserve answers never opened) | Keeps v1 diagnostics meaningful; reserve stays sealed |
| 8 | 2026-10-27 | Pronunciation default = first CMUdict reading; ambiguous (readings differ beyond stress) items flagged before prediction and kept in a diagnostic track | Handoff rule for ambiguous items without per-reading references |
| 9 | 2026-10-27 | No English G2P fallback; misses → `<no_pron>` | No eligible local G2P; a ByT5 download would add a runtime dependency; explicit missing is preferred over a silent rule |
| 10 | 2026-10-27 | English and Khmer phones are separate token namespaces (`en:`, `km:`) | Different alphabets; no validated map |
| 11 | 2026-10-27 | `MAX_SRC` = 80 positions, `MAX_TGT` = 40, `MAX_LETTERS` = 32; longer inputs refused | Longest eligible sources 34 (EN) / 39 (P2G); no truncation |
| 12 | 2026-10-27 | Validation = greedy top-1 on dev (final scoring beam-4) | Cheap per-epoch selection; difference declared |
| 13 | 2026-10-27 | P2G replay = separate optimizer update at the current LR, same loss (label smoothing 0.1), schedule not stepped | Handoff: replay must not advance the supervised schedule |
| 14 | 2026-10-27 | Pairing by `torch.manual_seed(seed)` + recorded initial-state hash; batch order by string-seeded private RNG per epoch | Same tensors and batch ids in every arm; resume-safe without saving generator state |
| 15 | 2026-10-27 | Supplied-pronunciation track not run | Every item's intended reading equals the default reading, so it duplicates the primary track |
| 16 | 2026-10-27 | Trained weights `restricted_artifacts_local_only` | Training rows carry CC BY-SA / ODC-By terms; derivative status of weights unresolved; publishing needs the owner anyway |
| 17 | 2026-10-27 | Smoke-run count: 8 of 10 used (4 jobs that failed at start-up because a data file was missing, then 4 good ones) | Counted conservatively |
| 18 | 2026-10-27 | MekongPhon not downloaded | The local Google lexicon already gives 61,809 eligible P2G pairs; MekongPhon would add a second phone alphabet |
| 19 | 2026-10-27 | Commits omit Claude attribution lines | Owner's standing rule |
| 20 | 2026-10-27 | `state/` is git-ignored; a copy of the final run ledger goes to the experiment folder | The runner's live state files are untracked and made every run's `git_dirty` true; `letters_s0` started that way (tracked diff empty, so its code is exactly `b9c7ea4`) |
| 21 | 2026-10-27 | CLI and `KHMER_TRANSLIT_CMUDICT` committed while `letters_s0` ran | Product-only change; training code paths unchanged (env var unset during runs) |
| 22 | 2026-10-27 | Runs after `ac5a157` are also flagged `git_dirty` | Uncommitted tests (`TestCLI`), `SOURCE_VERIFICATION.md` and the bench script were in the tree; none is imported by training; diff hash in each `provenance.json` |
| 23 | 2026-10-27 | The CMUdict parse is cached per (path, size, mtime) | Task 5 "cache version separation": a replaced file must not be served from the old parse |
| 24 | 2026-10-27 | Runtime bench: no pronunciation lookups for letters models; hit/miss from the items' `dictionary_miss` flag; the first measurement was replaced | The first run parsed CMUdict inside the letters process, inflating its lookup time (13 ms) and RSS (129 → 65 MiB) |
| 25 | 2026-10-27 | `report` reads the treatment's measured guards from `reports/runtime.json` | It passed `runtime_ok=None` (counted as a failure) before Task 5 existed; the guards are the registered ones |
| 26 | 2026-10-27 | The clean-venv check uses Python 3.10 and the uv cache (else PyPI for NumPy) | The cached NumPy wheels are for 3.10; the package itself installs from the local wheel |
| 27 | 2026-10-27 | The bundle carries the owner-list references and per-item outcomes | Per-item outcomes are a Task 6 deliverable; the whole bundle is local only (NOTICES.md) |
