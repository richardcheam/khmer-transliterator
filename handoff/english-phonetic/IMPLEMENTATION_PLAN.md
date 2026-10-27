# English phonetic transliterator implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` if installed, or the native task runner workflow. Execute this owner-approved handoff without a new approval round. Steps use checkboxes for tracking.

**Goal:** Deliver a tested local English-to-Khmer phonetic transliterator and a qualified, uncertainty-aware comparison of three small-model recipes.

**Architecture:** Reuse the compact Transformer and NumPy deployment path. Compare letters, letters plus English phones, and phone-input training initialized by Khmer P2G. Keep benchmark qualification independent from model implementation.

**Tech stack:** Existing Python project, PyTorch training, NumPy inference, TSV/JSON/JSONL/YAML records, current unittest suite, wheel packaging.

**Spec:** `DESIGN.md` and `AUTONOMY.md` in this handoff.

## Global constraints

- English phonetic transcription into Khmer script; one policy; no audio/TTS or chat romanisation expansion.
- Preserve E01, benchmark v1, historical checkpoints and sealed reserve contents.
- Three registered arms, at most seven full supervised attempts; all bounds in `AUTONOMY.md` apply.
- No paid calls, GPU jobs, undocumented-data model promotion or post-test tuning.
- Independent-quality claims require qualified references; otherwise explicit `inconclusive` status.
- Exact decisions use counts/full precision; round display only.
- Agent-owned engineering defaults must be frozen before full training; do not ask the owner to approve routine details.

## Suggested file map

Follow existing naming patterns; if a named file already exists, extend it rather than creating a duplicate subsystem. Record justified preflight substitutions.

| File | Responsibility |
| --- | --- |
| `src/khmer_translit/english.py` | Public English-focused API and result types |
| `src/khmer_translit/phonetic_source.py` | Phone lookup, normalization, explicit pronunciations and shared encoding |
| `src/khmer_translit/reference_data.py` | Provenance-aware records, filtering, group splits |
| `src/khmer_translit/evaluation.py` | New benchmark validation, scoring, comparison and controls |
| `scripts/english_phonetic_data.py` | Audit/build/freeze CLI |
| `scripts/english_phonetic_train.py` | Train/pretrain/export/verify CLI |
| `scripts/english_phonetic_run.py` | Fixed-queue orchestration, budgets and resume |
| `scripts/english_phonetic_eval.py` | Locked final evaluation CLI |
| `tests/test_english_*.py` | Task-specific behavioral tests |
| `experiments/<next-id>-english-phonetic/` | Plan, config, protocol, resources, reports |
| `state/english-phonetic/` | Durable execution state; transient machine-specific files ignored by git |

Do not overwrite existing arm names `A`, `B`, `E` or imply these new arm IDs are old served models. Choose the next unused experiment ID after inspecting current repository history; do not assume E02 is still free.

## Cross-task interfaces

- `PronunciationResult`: normalized word, phone token sequence(s), alphabet/version, provider, dictionary hit/miss, ambiguity, resource hash, status.
- `get_pronunciation(word: str, *, supplied: str | None = None) -> PronunciationResult` in `phonetic_source.py`.
- `ReferenceItem`: stable ID, input, normalized/family group IDs, intended English phones/accent, accepted Khmer forms, source IDs, quality tier, split and flags.
- `load_reference_items(path: Path) -> list[ReferenceItem]` and `audit_splits(...) -> dict` in `reference_data.py`.
- `encode_source(word: str, pronunciation: PronunciationResult, mode: str) -> list[int]`: modes `letters` and `phones`; version/hash recorded and reused in training, validation, export and inference.
- `Transliterator.from_pretrained(path: str | Path) -> Transliterator` in `english.py`.
- `Transliterator.predict(word: str, *, pronunciation: str | None = None, top_k: int = 1) -> PredictionResult`; `predict_many(words: Sequence[str], *, top_k: int = 1) -> list[PredictionResult]`.
- `PredictionResult`: original input, normalized input, status, ordered candidates with model log-scores, selected pronunciation and source provenance. Log-scores are not calibrated correctness probabilities.
- `score_predictions(items: Sequence[ReferenceItem], predictions: Mapping[str, PredictionResult]) -> dict` and `compare_paired_runs(...) -> dict` in `evaluation.py`. Require exact ID coverage; no silent dropped predictions.
- Durable runner command: `python scripts/english_phonetic_run.py --config PATH --resume`; first invocation also uses `--resume` and initializes state if absent.

## Review focus

1. A translation-equivalent label has different sounds: reference audit flags/excludes it rather than declaring model outputs wrong (Task 1/2).
2. Same English spelling has two pronunciations: supplied pronunciation and recorded default produce reproducible behavior (Task 3).
3. A valid Khmer alternative differs from the chosen favorite: multi-reference scoring accepts it, while uncertain alternatives remain explicitly unqualified (Task 2).
4. Dictionary miss, unknown phone, oversized input or missing resource: train/inference agree on a documented fallback or explicit error; never truncate silently (Task 3/5).
5. Runner restarts while training is live or after partial artifact writes: no duplicate job, corrupted promotion or exceeded budget (Task 4/6).

## Ordered tasks

- [ ] **Task 1 — Audit actual state and define the coherent data.** Execute `tasks/01-audit.md`. Deliver audit, source ledger, actual source-encoding trace and data eligibility decisions. Commit documentation and any necessary audit tooling.
- [ ] **Task 2 — Qualify the benchmark before model selection.** Execute `tasks/02-benchmark.md`. Deliver development/final split, controls, power analysis and qualified-or-inconclusive status; freeze before training. Commit the protocol/config and manifests, not private/unlicensed raw data.
- [ ] **Task 3 — Implement comparable arms and source/P2G transfer.** Execute `tasks/03-models.md`. Write focused failing behavioral tests, implement, run the tests, prove source/export parity, and commit.
- [ ] **Task 4 — Execute the fixed experiment queue.** Execute `tasks/04-experiments.md`. Test dry-run/budget/resume first, run bounded jobs, select only on validation, confirm if required, then evaluate once. Commit reports and configs with data/model artifact hashes.
- [ ] **Task 5 — Deliver the standalone product.** Execute `tasks/05-product.md`. Deliver CLI/API, source preparation, packaged checkpoint where eligible, offline behavior and compatibility. Verify the installed wheel in a clean environment and commit.
- [ ] **Task 6 — Integrate, verify and finish.** Execute `tasks/06-release.md`. Review the whole diff, run appropriate integration checks, build the release-ready bundle, write final evidence and clean task-owned scratch files only. Commit final docs and return the final report.

## Self-review before full jobs

- [ ] Every design requirement maps to a task and an output.
- [ ] Interfaces agree across the source pipeline, export, API and evaluation.
- [ ] The five review-focus cases have meaningful tests in their owning task.
- [ ] Data selection, primary metric, seed queue, candidate tie-break and decision rules are committed before full training.
- [ ] The runner enforces resource bounds and preserves experimental failures.
- [ ] No benchmark status is inferred solely from unit tests, hashes or model agreement.

Do not ask for another execution-method choice. The owner chose autonomous native execution through completion.
