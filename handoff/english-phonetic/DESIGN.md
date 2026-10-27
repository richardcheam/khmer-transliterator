# Design: a small English-to-Khmer phonetic transliterator

## Intent and acceptance policy

Given a written English word or a name intended to be pronounced as English, return readable Khmer script that approximates the specified English pronunciation. Preserve sound, not meaning. Use a deterministic documented English accent default, initially CMUdict-compatible US English; an optional supplied pronunciation disambiguates words such as `record` and names with multiple readings.

Established Khmer spellings are admissible when compatible with this phonetic policy. Do not automatically substitute a translation or a conventional name with a substantially different pronunciation. Multiple Khmer spellings may be valid. Phonetic approximation and orthographic preference must not be conflated.

No audio or TTS belongs in the training, runtime, benchmark or acceptance gate for this work. No requirement for the owner to author transcriptions or become a linguistic adjudicator. A later optional independent text review may increase confidence; it is not a reason to stall otherwise authorized engineering work.

## Selected approach and alternatives

The selected approach is one small character/phone sequence model, with one default output policy, tested in three arms:

| Arm ID | Source | Prior training |
| --- | --- | --- |
| `letters` | English letters | None |
| `phones` | The same English letters plus English phone tokens | None |
| `phones_p2g` | Same as `phones` | Khmer phones to Khmer spelling, then supervised transliteration |

The architecture is the existing compact Transformer encoder-decoder, unless the local audit establishes a necessary compatibility repair. Reuse the current NumPy inference/export where feasible. Keep existing stable public interfaces intact; add a separate English-focused API. A combined tagged source is sufficient for the first phone-input experiment; do not introduce dual encoders, adapters or mixture-of-experts in this experiment.

Alternatives considered: collecting large amounts of owner typing addresses another task and is rejected here; a large pretrained ByT5 system is a possible later reference but is not the default deployment model; conventional-versus-phonetic modes are deferred because this product has one coherent phonetic target. Articulatory features, hard phonology constraints, minimum-risk training, reinforcement learning and teacher distillation are research extensions, not extra experiment arms to add automatically.

## Existing project evidence that must be reconciled

At the inspected commit, `docs/TRANSLITERATOR.md` describes model E and E01 as phone-aware. The actual E01 training/save/export code in `scripts/translit_unified.py` uses `phones=False` and encodes Latin strings. Model E already has a CMUdict-based source pathway. Verify actual local batch contents and exported metadata before asserting that phone inputs are new, absent or correctly applied. If local code has changed, update this finding with evidence rather than blindly preserving it.

The English sources include owner preferences, LLM drafts and automatically mined bracket pairs. Their existence does not prove compatibility with the revised phonetic objective. E01 also mixed romanised Khmer and Cambodian names into training; this experiment must have a coherent English-focused supervised dataset. Do not use the old mixed dataset unexamined.

## Data and benchmark architecture

Use source and target provenance, explicit English pronunciations, accepted references, normalized input groups and reference quality tiers. Keep E01 and v1 unchanged. Create a new benchmark version if contents or policies change, with a separate development partition and a locked final partition. Select models on development only.

Evaluation tracks are: multi-reference exact spelling match; character error as secondary diagnosis; automated phonetic diagnostics; and independently assessed acceptability where trustworthy judgments already exist. Only qualified references support a final improvement claim. Do not let a pronunciation proxy silently become the gold metric.

Each qualified-primary item requires documented independent acceptance under the phonetic policy. Primary inference uses the deployed default pronunciation provider, not reference phones; explicitly supplied pronunciation is a separate conditional track. Multiple readings of the same spelling have distinct pronunciation/accent IDs within one lexical split group.

Train/test distinctions include unseen English lexical families and known-versus-missing English dictionary entries. Target-side pretraining overlap is measured and disclosed: seeing a Khmer spelling in a monolingual lexicon is not the same as seeing its English-to-Khmer pair, but it changes the generalization claim. Reserve a stricter target-unseen subgroup where enough examples exist.

## Product architecture

- Source processor: English normalization and explicitly versioned pronunciation lookup/G2P with provenance.
- Inference model: local checkpoint, source encoding, beam generation and stable top-k results.
- API: batch prediction with pronunciation metadata and explicit unsupported/ambiguous-input behavior.
- CLI: one word or batch file; text and JSON output; offline operation after explicit preparation.
- Reproducibility: frozen config/data/resource hashes, per-run records, artifact checksums and export parity.

The final bundle contains code/wheel, eligible local model artifacts, data/source notices, model card, benchmark qualification, per-word comparison outputs, final report and commands. Models derived from undocumented Chhunneng material must not become the new default or be bundled as clean models.

## Resources and evidence standard

At most seven full supervised runs: three seed-0 pilots plus the selected treatment and letters control at seeds 1 and 2. If letters is best on development, no treatment confirmation is needed. At most three bounded P2G pretraining jobs, one for each used `phones_p2g` seed; no separate architecture sweeps. Count all prior-training cost. Other explicit limits are in `AUTONOMY.md`.

Confirmatory evidence is for the selected recipe versus letters. Phone input versus P2G-transfer is a one-seed exploratory comparison in this budget. C includes fixed pretraining and replay; their individual effects are not isolated.

A practical adoption gain is 5 percentage points on the qualified primary metric, inherited as a decision threshold for this new protocol rather than a retroactive change to E01. Distinguish an estimated gain, uncertainty about that gain, and the decision to package/promote a model. Inconclusive evidence remains inconclusive. These proposals have not been demonstrated to work for this language pair.
