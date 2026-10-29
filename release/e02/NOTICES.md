# Notices, attribution and publication permissions: E02 letters-only model

**Publication status: UNRESOLVED.** Nothing here may be published without the owner's explicit decision.

This is a factual record of stated terms, checked 2026-10-29 against primary sources. It is not legal advice, and
it does not conclude that publication is permitted. The E08/E10 Google AI Mode rows were **not** used by this model
(`PROVENANCE_AUDIT.md`). Their absence does not by itself make it eligible.

## Permissions by item
| Item | Exact source and revision | Licence / terms (authoritative link) | Attribution required | Redistribution proposed? | Unresolved implications |
|---|---|---|---|---|---|
| **Package code** (`khmer_translit`, wheel + sdist) | This repository; the build commit is recorded in `evidence/build.json` | **PolyForm Noncommercial 1.0.0** (owner, 2026-10-29; [text](https://polyformproject.org/licenses/noncommercial/1.0.0); `LICENSE`) | Pass on the terms or their URL and the `Required Notice:` line | Proposed (wheel/sdist) | Commercial use needs the owner's permission |
| **Model weights** (`model.npz`) | `letters_s0`, sha256 `8901050b…904b` | **No licence chosen** | Upstream attribution below | Proposed (bundle, HF folder) | Depends on the data rows below and on the owner's licence choice |
| Khmer Wikipedia text → 693 training rows | `kmwiki-20261001-pages-articles.xml.bz2`, sha1 `9ed47fb3…d1a1` (matches Wikimedia's published sum) | CC BY-SA 4.0 and GFDL ([Wikipedia:Copyrights](https://en.wikipedia.org/wiki/Wikipedia:Copyrights); [CC BY-SA 4.0 legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)) | Yes, when sharing the material or adaptations: creator identification, licence notice and link, indication of changes (BY-SA §3(a)(1)) | **No**: no mined pair or page text is bundled | **Open:** whether the weights are "Adapted Material" of the Wikipedia text, and whether training needed permission at all. CC: "CC licenses apply only when copyright permission is required"; if models "are based on ShareAlike content and they will be shared publicly", SA would mean the same licence ([CC, Using CC-licensed works for AI training](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/)). Whether word-and-name spellings carry protectable expression is also open. Neither "SA applies" nor "SA does not apply" is concluded |
| FineWeb-2 (`HuggingFaceFW/fineweb-2`, `khm_Khmr`) → 9,807 training rows | File `data/khm_Khmr/train/000_00000.parquet`, sha256 `6f4506f6…e993` = Hub LFS hash at revision `af9c1333` (2025-10-27) | **ODC-By 1.0** for the database ([dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2): "The use of this dataset is also subject to CommonCrawl's Terms of Use"; [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/)) | Yes for a Produced Work used publicly. ODC-By §4.3 example notice: "Contains information from DATABASE NAME which is made available under the ODC Attribution License." | **No**: no row or document is bundled | ODC-By §2.4: "this License does not cover any rights (other than Database Rights or in contract) in individual Contents". The **database licence does not grant rights in the underlying web pages**: they remain with each page's owner. The dataset offers a [PII removal/opt-out form](https://forms.gle/VyNT3ZAUPZjPuWp39). Pages removed after 2026-10-12 would still be reflected in these weights |
| Common Crawl (FineWeb-2's source) | Snapshots 2013 to April 2024 (per the FineWeb-2 card) | [Common Crawl Terms of Use](https://commoncrawl.org/terms-of-use), last updated 2024-03-07: crawled content "may be subject to separate terms of use or terms of service from the owners"; "you agree to respect the copyrights and other applicable rights of third parties" | Not specified for models | No | Same third-party question as above. Attribution alone does not settle it |
| Owner loanword list (speech repo `configs/latin_khmer.tsv`, sha256 `69bf45f4…384e`) | Used for checkpoint selection (86 dev items), the evaluation tracks and leak filters; **not training labels** | Owner's own work. Some rows are LLM suggestions the owner verified (17 of the 86 selection items) | n/a | **No** (references and review data stay local) | Whether checkpoint selection on LLM-suggested, owner-verified spellings engages the LLM provider's terms is **not determined**. The LLM service and its terms for those rows are not recorded in the list's provenance |
| Chhunneng pairs (`github.com/Chhunneng/khmer-text-transliteration`, commit `fe8028f1`) and models trained on them | Mining-time aligner and classifier only (`PROVENANCE_AUDIT.md` §3); **no label text in training** | Repository MIT licence; origin of the pairs undocumented (`data/SOURCES.md`) | MIT notice when redistributing that code or data (not done) | No | Indirect influence on which web pairs were kept. Whether that matters is the owner's decision |
| CMUdict (`cmusphinx/cmudict`, sha256 `81917843…3d22`) | Data-build filter only; **not needed at inference and not bundled** | BSD-style (`data/src/cmudict/LICENSE`) | Only when redistributing the dictionary | No | None for this bundle |
| Google Khmer lexicon (`google/language-resources` km, commit `bc72dd65`) | Mining-class filter (§3) and a diagnostic re-ranker; not training data for this model; not bundled | CC BY 4.0 ([legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en)), Copyright 2018 Google Inc. | When sharing it (not done); credited below as good practice | No | None identified |
| NumPy (runtime dependency) | Installed by the user from PyPI (tested 2.2.6) | BSD-3-Clause | Not bundled | No | None |
| Documents and evidence in the bundle | Project-authored; aggregate counts, hashes and model outputs on test words only | Owner's decision | n/a | Proposed | Part of the owner's licence decision |

## Attribution text (for any publication the owner approves)
> This model was trained on English–Khmer word pairs mined from Khmer Wikipedia (dump of 2026-10-01; text available
> under CC BY-SA 4.0, https://km.wikipedia.org) and from FineWeb-2 (HuggingFaceFW/fineweb-2, `khm_Khmr`). Contains
> information from FineWeb-2, which is made available under the ODC Attribution License (ODC-By 1.0); its use is
> also subject to Common Crawl's Terms of Use. Data selection used the CMU Pronouncing Dictionary and the Google
> `language-resources` Khmer lexicon (CC BY 4.0, Copyright 2018 Google Inc.).

## What is not bundled
- Training pairs, mined files, raw web documents and dumps.
- Owner references, the review queues, the reserve.
- CMUdict and the Google lexicon.
- ByT5 adapters and base weights; phone-input and other E02 checkpoints; torch checkpoints; W&B folders; caches.
