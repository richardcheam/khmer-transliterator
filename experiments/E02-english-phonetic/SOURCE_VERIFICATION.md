# E02 source verification

Checked on 2026-10-27 during execution. Per handoff source, this file records:
- `checked`: what was checked;
- `local_path_or_snapshot` and `hash`;
- what the source **supports** here, and what it **does not support**;
- `permission_evidence`, `used_for`, `availability` and what remains **unresolved**.

**No paper was re-downloaded in this run.** Paper entries rely on the handoff's reading of the primary documents
(they are cited, not re-verified), and the project's own claims are marked as hypotheses. Repository entries were
checked against the local checkout and the local data.

## Repository evidence

| Id | Checked | Local version / hash | Supports | Does not support | Used for | Unresolved |
|---|---|---|---|---|---|---|
| R1 benchmark v1 | `benchmarks/v1.json` and `data/benchmark/v1/*.tsv`; hashes match (`benchmark.load`) | commit `b3f2096`; panel hashes in `benchmarks/v1.json` | 5 frozen panels; pick picks mostly from E's top 5 | independent phonetic validity | historical diagnostics after selection; leak exclusion | — |
| R2 benchmark code | `src/khmer_translit/benchmark.py` | same | literal-equality outcomes, word bootstrap | NFC policy, family grouping (new in `evaluation.py`) | v1 diagnostics scored with `outcomes` unchanged | — |
| R3 data card | `datasets/README.md` and `data/SOURCES.md` vs local files | `datasets/manifest.json` | origins, licences declared per source, Chhunneng caveat | blanket redistribution rights | `resources.json` | weight licensing (CC BY-SA mined pairs) |
| R4 protocol | `docs/RESEARCH_PROTOCOL.md` | same | plan before training, paired seeds, leak filtering | TTS listening (excluded by the handoff) | PLAN.md structure | — |
| R5 E01 code | `scripts/translit_unified.py`, exported `translit.npz` meta of all six models | `phones: false` in base_s0..aux_s2 | E01 = letters-only input + Latin → Google-phones side decoder | E01 phone input | audit §2 | — |
| R6 TRANSLITERATOR.md / en_g2p | prose vs code; model E meta `phones: true`, `max_in: 48` | same | E is the only phone-input model (Chhunneng-trained) | E as a clean baseline | audit §2; stale sentence corrected | — |
| R7 E01 results | `experiments/E01-unified/RESULTS.md` | same | +4.67 pts (reference agreement), CI [+2.0, +7.6], gate missed | phonetic verification | discordance for the power grid (29/150) | — |
| R8 owner picks | `experiments/E00-history/reports/translit_pick.json` (not re-opened); `data/synth/*.tsv` | same | the procedure anchors picks on E's candidates | an exhaustive list of acceptable forms | the `candidate_anchored` flag (94 final items) | — |

## Primary research and resources

| Id | Checked | Supports (per the handoff's reading) | Does not support | Used | Permission | Unresolved |
|---|---|---|---|---|---|---|
| P1 He & Cohen 2020 | not re-fetched | a phonetic auxiliary task can help English–Chinese transliteration | Khmer; the E02 phone-input or P2G recipe | motivation only | — | whether it transfers to Khmer |
| P2 ByT5 (TACL 2022) | not re-fetched | byte models handle spelling well | a fit on this CPU budget | no | — | — |
| P3 CharsiuG2P | not fetched | multilingual G2P models exist | English→Khmer spellings | no (no G2P fallback, decision 9) | — | — |
| P4 MekongPhon (LREC 2026) | not fetched; the repository was unavailable to the handoff as well | a Khmer word–IPA lexicon (8,026 seeds) | an IPA→Khmer model; loanword labels | no (decision 18) | model card Apache-2.0; data rights unknown | everything about local rights |
| P5 Ngo et al. 2015 | not re-fetched | target phonology helps English→Vietnamese | Khmer rules | no | — | — |
| P6 PanPhon | not re-fetched | articulatory features | E02 gains | no | — | — |
| P7 Dror 2018; NLPStatTest | not re-fetched | paired tests, effect sizes, power thinking | label validity; seed uncertainty | power grid and bootstrap design (project choices) | — | — |
| P8 CMUdict | local `data/src/cmudict/cmudict.dict`, sha256 `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22`, with `LICENSE` | US English phones, with alternatives | Khmer-adapted phones; correctness | pronunciation provider | BSD-style notice (local file) | errors and omissions in names |
| P9 Shurtz 2025 (MRL) | not re-fetched | orthographic plus phonetic input in translation | word transcription | motivation only | — | — |
| Google Khmer lexicon | `data/lexicon/km_lexicon.tsv` @ `bc72dd65`, `ATTRIBUTION.md` | Khmer headword → pronunciation | English loan spellings | P2G pretraining and replay; re-ranker list | CC BY 4.0 (attribution kept) | — |
| Mined bracket pairs | `data/mined_{kmwiki,fineweb}.tsv`; hashes in `data/e02/manifest.json` | human-written Latin–Khmer co-occurrences | correct alignment; an English reading | supervised training (proxy) | CC BY-SA 4.0 / GFDL; ODC-By 1.0 + CC terms | share-alike status of trained weights |

## What is this project's hypothesis, not a sourced finding
- That English phone input, or Khmer P2G transfer plus replay, helps this small model.
- The three-arm design, the 7-run budget, the replay schedule and the runtime guards.
- That owner-verified loanword-list spellings approximate the E02 phonetic policy.
