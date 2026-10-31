# Training-data provenance and attribution (khmer-translit-byt5-e11 0.2.0)

**What the adapter was trained on:**
- **Training file:** the frozen E08 arm R, `data/e08/training/arm_R.tsv`, sha256
  `d00706cf1e16ccb81f9d367301d9a3fa637a46f79e2e8c8f0b2e501b57f8ec88`.
- **Optimised rows:** 4,764 pairs over 3,585 words. The ordered-pair hash `5dd38022…c1d61` equals the run's
  recorded data identity.
- **Not optimised:** the file has 93 more train-val rows, which are never optimised.
- **Kept local:** the pair files aren't part of the bundle.
- **Re-checked from the files:** `evidence/identity.json` and E11's `reports/data_checks.json` re-derive these
  facts from the data itself.

| Provenance | Rows | Origin | Acceptance | Terms of the source (checked 2026-10-31) |
|---|---|---|---|---|
| `control` | 2,255 | **E03 build:** bracketed Latin forms next to Khmer spellings, mined from FineWeb-2 `khm_Khmr` (shard `000_00000.parquet`, bytes identical to dataset revision `af9c1333`) and the Khmer Wikipedia dump of 2026-10-01. Over E03's full set: FineWeb-2 2,643, both 296, Wikipedia 28 pairs | E03 automatic consonant screen (unreviewed) | **FineWeb-2:** ODC-By 1.0, and "also subject to CommonCrawl's Terms of Use" (dataset card). **Wikipedia text:** CC BY-SA 4.0 / GFDL (Wikimedia Terms of Use §7) |
| `recovered` | 1,292 | Mined pairs from the same sources that E03 had excluded: FineWeb 1,233, FineWeb + Wikipedia 44, Wikipedia 32 (unique pairs, before dedup) | Documented screen-weakness rules (E08 `recovery_audit.md`) | As above |
| `attested_alt` | 1,217 | Names and loanwords found verbatim in the human Khmer translations of the **ALT Parallel Corpus** (NICT, `ALT-Parallel-Corpus-20191206.zip`) | Verbatim match + the same consonant screen | "NICT released the ALT Parallel Corpus as a Creative Commons Attribution 4.0 International (CC BY 4.0) License"; the English side is Wikinews (CC BY 2.5). **The separate "Khmer ALT" annotation set** (`km-nova`, tokenised/POS-tagged) is CC BY-NC-SA 4.0 and **was not used** |

## Verified absences
- **No AI-generated rows.** There are no Google AI Mode or other LLM-generated rows:
  - `ai_mode_rows_in_R` = 0;
  - no R pair is identical to an AI Mode spelling;
  - R is a subset of E08's R+S, and R+S minus R is exactly the 384 AI Mode rows.
- **No other adapter's weights.** E11 started from a fresh LoRA adapter on the pinned base. Its checkpoint
  differs from E08's (`d5da8c29…`), E10's (`35b450e2…`), E12's (`566a673e…`) and E13's (`e32cd817…`).
- **No quarantined pairs** from the earlier screens, no test or dev word family, no E02 final reference spelling
  and no reserve item.
- **No Chhunneng pairs as labels.** However, see `NOTICES.md` §3 on indirect influence during mining.
- **No human-reviewed labels:** none were used in training.

## Resources used without being training pairs
| Resource | Use | Terms |
|---|---|---|
| `google/byt5-small` @ `68377bdc18a2ffec8a0533fef03b1c513a4dd49d` | Frozen base (not redistributed) | Hub metadata `license: apache-2.0`; pre-trained on mC4 |
| CMUdict (`cmusphinx/cmudict`) | Readings for the data screen and the reviewer's pronunciation column; **not a model input** | BSD-style |
| Google `language-resources` km lexicon | The consonant screen's Khmer readings | CC BY 4.0 (Copyright 2018 Google Inc.) |

## Lineage
- **Run:** `scripts/e11.py train`, registered 2026-10-29 (`cfg_hash` `f5d7cf92…`).
- **Code state:** the run started from commit `c234de26` with uncommitted changes (diff sha256 `d1f969bb…`).
- **What pins the code:** the scripts that ran are pinned by the registered source hashes (`scripts/e11.py`,
  `e07.py`, `e06_byt5.py`, `e06.py`, `e09.py`, `e10.py` and the research requirements). The full list is in
  `evidence/identity.json`.
- **Environment:** PyTorch 2.14.0, Transformers 5.19.0, Python 3.10, CPU.

## Attribution text
> Training pairs include material mined from FineWeb-2 (HuggingFaceFW/fineweb-2, ODC-By 1.0) and Khmer Wikipedia
> (CC BY-SA 4.0), and Khmer spellings attested in the ALT Parallel Corpus (NICT, CC BY 4.0; Riza et al. 2016,
> "Introduction of the Asian Language Treebank", Oriental COCOSDA). The base model is google/byt5-small
> (Apache 2.0; Xue et al. 2022, TACL 10:291–306).
