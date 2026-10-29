# E02 release preparation (2026-10-29)

This is a release-preparation task, not an experiment: no training and no new evaluation.

- **Prepares:** the original E02 letters-only model (`models/e02/letters_s0/model.npz`, sha256 `8901050b…904b`) as
  a local Hugging Face release candidate.
- **Documents:** `release/e02/`. These are the bundle's README, model card, benchmark card, provenance audit,
  notices and permissions, licensing decision and publication checklist.
- **Script:** `scripts/e02_release.py`, steps `audit` → `build` → `verify` → `bundle`.
- **Evidence:** `reports/`, holding provenance facts, the build record, the isolated install check and the bundle
  manifest.
- **Bundle:** `dist/khmer-translit-e02-letters-0.2.0/` and `.tar.gz` (local, git-ignored). The historical
  `dist/english-phonetic-0.1.0/` is untouched.

Publication is unresolved. The owner's decisions are listed in `release/e02/PUBLICATION_CHECKLIST.md`.
