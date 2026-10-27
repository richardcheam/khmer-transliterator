# Task 6 — Final verification, artifacts and handover

- [ ] Review the whole diff against the approved scope. Ensure the documentation describes the actual selected model, not an abandoned candidate. Fix material issues and rerun checks relevant to the fix. Do not start a new architecture/schedule experiment.
- [ ] Run the existing project suite plus the new meaningful tests. Record exact command, exit status and pass/skip/failure counts. Distinguish full-data checks from skipped checks when files are absent. Do not claim scientific verification because unit tests pass.
- [ ] Verify source data/resource hashes, new benchmark hashes, selected checkpoint/export parity, all-run ledger, queue budget, prediction ID coverage, exact-count decision, CLI/API installed-wheel operation and offline behavior. Record limitations instead of deleting failed evidence.
- [ ] Verify E01, v1 files and old model artifacts were preserved. Confirm no reserve answer/prediction was opened; use the recorded access policy and paths, not an invented guarantee about unobservable history. State what this execution actually did.
- [ ] Build `dist/english-phonetic-<version>/` with wheel, eligible checkpoint(s), README quick start, source/license notices, model card, benchmark card, configs/manifests, run summary, final report and per-item outcomes where redistribution permits. Add `SHA256SUMS`; exclude keys, raw private notes, unrestricted source corpora, caches, temporary environments and undocumented-data models.
- [ ] Build a ZIP of that bundle and test archive integrity and safe relative paths. Check the wheel from a clean temporary environment, independent of source checkout and sibling speech repo.
- [ ] Write final reports using `templates/FINAL_REPORT.md`. Include positive, negative and inconclusive findings; report whether references qualify and whether full paired confirmation completed. State estimated gain, interval, seed variability, evidence classification and adoption decision separately.
- [ ] Include `SOURCE_VERIFICATION.md`: each SOURCES entry, checked URL/local version, what the source actually supports, whether used, permission evidence and unresolved gaps. Cite sources in the model/benchmark cards and explain which parts are this project's hypotheses rather than proven prior results.
- [ ] Include `RESUME.md` only if something truly cannot finish: exact state, completed deliverables, genuine blocker, smallest missing prerequisite and exact command. This is not a routine “shall I continue?” invitation. If all work completed, provide rerun commands instead.
- [ ] Remove only task-created expendable scratch/cache files when safe. Preserve reusable checkpoints, configs, failed/superseded run records, audit reports and unrelated user work. Show final git status with expected ignored artifact paths.
- [ ] Commit final tracked work. If configured credentials allow, push the dedicated feature branch; never force-push or merge main. If push is unavailable, leave the commit and exact branch/HEAD in the report without blocking local artifacts.
- [ ] Return one concise self-contained final message: product bundle path, install/predict commands, best model/default, primary results and qualifications, tests, time/memory/cost, preserved reserve/E01 status, branch/commit and any real blocker. No routine continuation question.

## Checklist that must not be silently collapsed

| Dimension | Possible final states |
| --- | --- |
| Product | runnable_local / runnable_research_preview / tooling_only_weights_unavailable |
| Benchmark | qualified / partially_qualified / unqualified |
| Experiment | complete_paired / pilot_only / incomplete / not_trainable |
| Evidence | supported_improvement / meaningful_gain_not_supported / inconclusive |
| Adoption | treatment_promoted / letters_baseline_retained / no_eligible_default |
| Redistribution | eligible_artifacts_bundled / restricted_artifacts_local_only / unresolved |

Use the actual combination. For example, a runnable research preview with an unqualified benchmark is a useful engineering deliverable, but it is not a validated breakthrough.
