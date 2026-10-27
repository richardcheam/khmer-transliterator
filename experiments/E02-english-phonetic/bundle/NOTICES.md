# Notices and redistribution status

**Status: `restricted_artifacts_local_only`.** This bundle is for local use on the owner's machine. Do not publish
or share the weights, the references or the outcomes without the owner's explicit decision.

## Code
The `khmer_translit` package (wheel and sdist in `wheel/`): written for this project. **No code licence has been
chosen yet**; that is the owner's decision, open.

## Data behind the weights (`checkpoints/`)
Every checkpoint was trained on the same supervised pairs; the `phones_p2g` runs also on Khmer lexicon pairs.

| Source | Licence | What it means for the weights |
|---|---|---|
| Bracket pairs mined from Khmer Wikipedia (dump 2026-10-01) | CC BY-SA 4.0 / GFDL | attribution required; whether trained weights are share-alike adaptations is **unresolved** |
| Bracket pairs mined from FineWeb-2 (`khm_Khmr`, shard 0) | ODC-By 1.0 + Common Crawl terms of use | attribution; terms of the underlying web pages |
| Google Khmer lexicon (`google/language-resources`, km, @ `bc72dd65`) | CC BY 4.0 | attribution below; used for P2G pretraining/replay and the diagnostic re-ranker list |

No Chhunneng data, no LLM output and no owner list entry was used for training any checkpoint here.

## Google Khmer lexicon attribution (CC BY 4.0)
Khmer pronunciation lexicon from Google's `language-resources` repository (https://github.com/google/language-resources),
Copyright 2018 Google Inc., licensed under the Creative Commons Attribution 4.0 International License. Used here reversed
(pronunciation → spelling) and filtered (Khmer headwords; accepted benchmark spellings removed). Full notice:
`resources/google_km_lexicon_ATTRIBUTION.md`.

## CMUdict (`resources/cmudict/`)
The Carnegie Mellon Pronouncing Dictionary, BSD-style licence: `resources/cmudict/LICENSE` (sha256 of the
dictionary in `prepare_manifest.json`). Required by phone-input models only.

## References and outcomes (`outcomes/`)
The benchmark spellings come from the owner's speech front-end list (the owner's choices and LLM drafts the owner
reviewed; the proxy track is LLM output with a spot check). They are the owner's material: local only.
