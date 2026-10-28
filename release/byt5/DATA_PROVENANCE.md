# Training-data provenance and attribution ledger (khmer-translit-byt5 0.2.0)

Every adapter in this release was trained on the frozen E08 arm R+S: `data/e08/training/arm_RS.tsv`, sha256
`03a4988f84cdc97aff6b8356e019d1076c8d09a1ecec4004a2a909ed3e229b0f`. It has 5,148 optimisation rows over 3,778
English words, plus 93 diagnostic train-val rows that are never optimised. The pair files stay local: they are not
part of the bundle.

| Provenance | Rows | Words | Where it comes from | Acceptance | Stated terms of the source |
|---|---|---|---|---|---|
| `control` | 2,255 | 1,883 | E03 build: bracket pairs mined from FineWeb-2 `khm_Khmr` (shard `000_00000.parquet`) and the Khmer Wikipedia dump 2026-10-01. Over E03's full set the sources were FineWeb-2 2,643, both 296 and Wikipedia 28 pairs. | E03 dictionary-assessable consonant screen (automatic, unreviewed) | FineWeb-2: `license: odc-by` (dataset card, checked 2026-10-28); `data/SOURCES.md` also records the Common Crawl terms of use. Wikipedia text: CC BY-SA 4.0 / GFDL. |
| `recovered` | 1,292 | 1,098 | Mined pairs that E03 had excluded, from the same sources: FineWeb 1,233, FineWeb + Wikipedia 44, Wikipedia 32 (by unique pair, before dedup) | Documented screen-weakness rules (E08 `recovery_audit.md`) | As above |
| `attested_alt` | 1,217 | 1,092 | Names and loanwords found verbatim in the human Khmer translations of the ALT Parallel Corpus (NICT, `ALT-Parallel-Corpus-20191206.zip`, sha256 `05f7b31b…`) | Verbatim match + the same consonant screen | CC BY 4.0 for NICT's translations; the English is Wikinews (CC BY 2.5). Cite Riza et al. 2016, "Introduction of the Asian Language Treebank", Oriental COCOSDA. |
| `google_ai_mode_synthetic` | 384 | 193 | Google AI Mode answers to 10 frozen prompts, pasted by the owner (E08) | Structural import only: **unverified LLM output** | Google's Generative AI Additional Terms of Service (checked 2026-10-28): "You may not use the Services to develop machine learning models or related technology." **Whether these rows may be used for a released model is unresolved**; see `NOTICES.md`. |

## Resources used without being training pairs
| Resource | Use | Terms |
|---|---|---|
| `google/byt5-small` @ `68377bdc18a2ffec8a0533fef03b1c513a4dd49d` | Frozen base model (not redistributed in this bundle) | Hub metadata `license: apache-2.0`; pre-trained on mC4 |
| CMUdict (`cmusphinx/cmudict`) | Readings for the data screen, input selection and the reviewer's pronunciation column; not a model input | BSD-style (CMU) |
| SCOWL `american-english` | Candidate word lists for the confirmation manifest | SCOWL permissive notice |
| Google `language-resources` km lexicon | The consonant screen's Khmer readings | CC BY 4.0 (Copyright 2018 Google Inc.) |

## Not used
- Chhunneng data, E02 final reference spellings, the sealed reserve, ECCC (permission form not signed) and
  Khmerologie (no reuse terms).
- No reviewer-authored labels exist.

## Attribution text (for any future publication the owner approves)
> Training pairs include material mined from FineWeb-2 (HuggingFaceFW/fineweb-2, ODC-By 1.0) and Khmer Wikipedia
> (CC BY-SA 4.0), and Khmer spellings attested in the ALT Parallel Corpus (NICT, CC BY 4.0; Riza et al. 2016).
> The base model is google/byt5-small (Apache 2.0; Xue et al. 2022, TACL 10:291–306).
