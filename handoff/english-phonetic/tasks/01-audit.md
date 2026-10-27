# Task 1 — Audit repository, task policy and usable resources

**Inputs:** design, current checkout/local data, SOURCES references R1–R8 and P1–P7.
**Outputs:** `reports/audit.md`, `resources.json`, `data_policy.json`, source encoding trace, retained/rejected counts and decisions in the new experiment folder.

## Execution

- [ ] Inspect current `AGENTS.md`/`CLAUDE.md`, branch/status, repo docs, configuration, training model, export and inference paths. Preserve ongoing work and old results. Record current commit instead of assuming the inspected public commit is current.
- [ ] Locate actual local panel files, manifests, models, raw sources and English pronunciation resources. Read only reserve Latin inputs if the prior protocol permits them for exclusion; do not open reserve answers or generate reserve predictions. Treat the handoff as authority for the new experiment, not an instruction to rewrite E01 rules.
- [ ] Trace model E and E01 source tensors from raw input through `encode_batch`/`source` and export metadata. The public E01 code has `phones=False` despite a contradictory documentation sentence. Capture at least one development-only input trace for letters and phone-aware paths. Correct stale documentation separately from historical results.
- [ ] Inventory each source's actual files, hashes, row counts, origin, license text/attribution, derivative restrictions, and permitted role: training, inference, evaluation or excluded. A repository-level license does not prove rights in imported data. Do not write legal certainty that the source materials do not support.
- [ ] Audit English training records for semantic translations, conventional forms with shifted pronunciation, Khmer-person-name romanisation, mixed-language strings, questionable alignment, duplicates and conflicting labels. Assign `eligible`, `uncertain`, `excluded` with reasons. Do not treat an automated pronunciation-distance score or LLM agreement as an independent verification.
- [ ] Use existing clean English/name transliteration pairs that meet the phonetic policy. Do not pad the corpus with rule-romanised Khmer dictionary words; those are a different source task. Preserve source tier in the records and use the same eligible supervised rows and sample weights in every arm.
- [ ] Keep independently verified/attested labels separate from LLM drafts. Unverified drafts are excluded from the first controlled supervised comparison by default; if this leaves no viable corpus, take the data-blocker fallback rather than quietly changing the experiment.
- [ ] Inspect English pronunciation lookup: model E's CMUdict handling, missing-word rate, alternative pronunciations, stress/alphabet conventions, token coverage and input lengths. Choose a local English G2P fallback if already available and documented, or an eligible small free resource within budget. CMUdict lookup and G2P must not invoke a synthesizer or audio. Explicitly missing output is preferable to a silent unreliable rule substitution.
- [ ] Inspect the existing Google Khmer lexicon for P2G usable entries, pronunciation conventions and attribution. Deduplicate actual spelling/pronunciation pairs, preserve alternatives and homophones, and build a shared phone vocabulary with language/task markers. Inconsistent phone alphabets require a documented conversion map or separate tokens; do not assume identical strings mean identical phones.
- [ ] Inspect MekongPhon only if it supplies genuinely useful, eligible additional material within budget. Its paper's 1.3M Khmer sentence pairs are automatically expanded from 8,026 word–IPA seed pairs. Start with the word lexicon if eligible; do not download/train on the full corpus merely because its headline count is large. The published checkpoint is Khmer→IPA, not an invertible IPA→Khmer model.

## Tests and evidence

- [ ] Add targeted checks for a malformed row, normalization variants, a known training/evaluation collision, a multi-target word and an explicitly excluded source. Tests verify plumbing, not linguistic truth.
- [ ] Emit per-source counts before/after each exclusion and make a small reproducible development sample with reasons. Avoid cherry-picking only clean successes.
- [ ] Emit missing-resource and ambiguity rates. Identify every assumption not actually verified on the local machine.
- [ ] Record audit status and proceed to Task 2 automatically. If some labels are uncertain, exclude/quarantine them and proceed with usable material; if none remain, finish tooling/product paths under the documented blocker status.
