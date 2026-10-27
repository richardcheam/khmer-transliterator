# Task 5 — Standalone local product

**Outputs:** usable Python API, console CLI, explicit resource preparation, model packaging, examples, model/data cards and wheel.

## Public behavior

- [ ] Add `Transliterator` and result types from the plan without breaking existing imports/callers. Model loading is explicit and local. Avoid silently selecting the old B/E router. Maintain the old API as legacy behavior with clear docs; the new English API follows this experiment's phonetic policy.
- [ ] Provide `khmer-translit` console commands, adjusting only if that entry-point name is already occupied:
  - `khmer-translit prepare --manifest PATH`: explicit eligible resource preparation/checks, no hidden downloads during import or predict.
  - `khmer-translit predict --model PATH --word coffee --json`: one word, result/provenance.
  - `khmer-translit predict --model PATH --input words.txt --jsonl`: one input per line; stable ordered output with IDs/errors.
  - `khmer-translit predict --model PATH --word record --pronunciation '...'`: explicit phones in the documented alphabet, not arbitrary unparsed IPA.
  - `khmer-translit inspect --model PATH`: config, checksum, model/evidence status and required resources.
- [ ] Provide top-k distinct candidate strings and model log-scores. Do not label beam softmax values “confidence of correctness.” Validate `top_k` and beam limits; report the chosen deterministic default pronunciation and any unresolved ambiguity.
- [ ] Input contract: one English word/name; supported apostrophes/hyphens handled by registered normalization. Empty/oversized/unsupported multiword or non-English-script inputs get explicit errors/status, not an English guess presented as universally valid Latin transliteration. Preserve original inputs in output.
- [ ] Missing phones use the registered `<NO_PRON>` pathway. A missing model/resource file produces an actionable path/hash error. Optional G2P must be explicitly prepared; no network surprise during inference. Support offline prediction after preparation, including a clean baseline model that does not need G2P.

## Packaging and resource rights

- [ ] Keep runtime inference light: use existing NumPy path if parity is proven; training dependencies remain optional. Do not make W&B, TTS, browser access or paid services runtime requirements.
- [ ] Package versioned metadata with source encoder, pronunciation provider/inventory, accent, input policy, max lengths, model and resource checksums, beam settings, training source ledger and evidence status. Fail clearly on incompatible metadata rather than loading wrong vocabulary weights.
- [ ] Create a wheel/sdist using the repository's build backend. Use a separate checkpoint bundle if bundling eligible weights into the wheel is impractical. Supply an exact local install and model-path example. A public package license choice and public publishing are not assumed; preserve existing rights and mark any unresolved code-license decision in release documentation without blocking local build/testing.
- [ ] Include only data/weights with adequate permission/attribution evidence. A permissive code license does not clear imported datasets. Keep nonredistributable evaluation data out of distributable bundles while retaining their local hashes/metric reports.
- [ ] No public PyPI upload, main-branch merge, serving switch or release tag. Deliver release-ready local artifacts and a dedicated branch if configured access permits.

## Product verification

- [ ] Test single/batch API, CLI JSON/JSONL schema, input ordering, supplied pronunciation, missing provider, cache version separation, unsupported input, repeat-run determinism and missing/corrupt checkpoint. Test outside the checkout through the installed wheel to expose accidental `sys.path`, sibling TTS-repo and cwd dependencies.
- [ ] Confirm an offline example works without the sibling speech repository or its loanword list. Import/predict must not synthesize audio or contact a service.
- [ ] Measure cold-load time and warm prediction latency on the actual laptop with at least 100 development-only words and at least 3 repeated timing passes. Separate dictionary hit/miss and batch1/batch16, and include pronunciation lookup/G2P costs. Measure peak RSS in a fresh process; record threads and machine info.
- [ ] Handoff engineering targets: cold load <=30 seconds, warm batch1 p95 <=1 second/word, peak RSS <=1.5 GiB on this laptop. A promoted treatment must also have warm batch1 p95 <=2x the matched letters baseline. These are deployment guards chosen for this project, not published performance facts. If missed, optimize inference/export/caching without altering trained model semantics or final results; otherwise retain the baseline default and disclose the limitation.
- [ ] Include an exact copy-paste demo using an eligible completed checkpoint, plus output marked as measured rather than a handwritten expected answer. If weights could not be trained, provide the training/resume command and explicit `weights_unavailable`; do not misrepresent a stub as a working trained model.
- [ ] Commit code/docs/tests and advance automatically to final integration.
