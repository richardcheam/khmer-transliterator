# Provenance audit: E02 letters-only model (`letters_s0`)

**Audited:** 2026-10-29 by re-hashing files and reading the code that produced them, not by repeating reports.
Machine-checkable facts are in `evidence/provenance_facts.json`, produced by `scripts/e02_release.py audit`.

**Not opened:** the E02 final reference spellings (`data/e02/references.jsonl`) and the sealed reserve targets.
The reserve's Latin words are read only by the E02 data build, to exclude them.

**Raw training pairs stay local.** Only counts and hashes appear here.

## 1. Checkpoint identity and lineage (verified)
| Artifact | sha256 | Check |
|---|---|---|
| `models/e02/letters_s0/model.npz` (NumPy weights + `__meta__`) | `8901050b1093cf90f60523c7941d6c12dbd8571d71698e37abcf874ba437904b` | Equals the hash in `FINAL_REPORT.md` and `parity.json` |
| ← exported from `best.pt` (torch) | `199df48c4efc4d2f79f5a4a205173ada84eb37349972934f5bcf0ab1e8dad848` | Equals `done.json:best_sha256` and the `checkpoint_sha256` embedded in `model.npz` |
| ← training config `experiments/E02-english-phonetic/config.yaml` | `941a8453…dacaf` | Equals the npz metadata and `provenance.json` |
| ← data manifest `data/e02/manifest.json` | `6cdc7c8a…ffad` | Equals the npz metadata; every input and output hash it lists re-verified |
| ← training rows `data/e02/train.tsv` | `ee473aa0…7aa` | Equals the manifest and `provenance.json` |

**The run:**
- Arm `letters`, seed 0; best epoch 27 of 40; 3,080 optimiser steps; 0.25 h on CPU.
- 2,609,169 parameters; d 192, 3 + 3 layers, 4 heads, FF 512.
- **Export parity:** NumPy vs torch on 12 fixtures, max |Δ log p| 4.1e-6 (`parity.json`).
- **Training code:** commit `b9c7ea4`, recorded with `git_dirty: true` and no diff hash. The exact state of the
  uncommitted training-script edits is therefore **not recoverable**. The package source at that commit reproduces
  the installed wheel's outputs exactly on all 103 dev inputs (`install_verify.json`), so the *inference*
  implementation is verified. The training-time script state is not.
- **Run ledger:** `experiments/E02-english-phonetic/run_ledger.json`, job `letters_s0`: completed, 1 attempt,
  `init: null`. Metrics are mirrored to the owner's W&B project.

## 2. What the checkpoint was trained on (verified)
| Question | Finding | Evidence |
|---|---|---|
| **Initialisation** | Random, from `torch.manual_seed(0)` (initial-state hash `40dba049…`). No pretrained or P2G weights; `train()` refuses `--init` for the letters arm | `done.json` (`p2g_init_file_sha256: null`), `config.json` (`init: null`), `english_phonetic_train.py` (`train`, `build_model`) |
| **Auxiliary training** | None. `replay_steps: 0`. P2G replay is loaded only for `phones_p2g`. No phones: `mode: letters`, `phone_dropout_examples: 0` | `done.json`; `english_phonetic_train.py` lines 317–339 |
| **Optimisation rows** | 9,735 rows (`part=train`), every row `source=mined_web`, `tier=proxy` | `train.tsv` aggregate |
| **Train-val rows** (loss for tie-breaks only; never optimised) | 296 rows, also `mined_web` | as above |
| **Which mined file each of the 10,031 rows comes from** | FineWeb-2 only 9,338; Khmer Wikipedia only 224; both 469 (exact match on Latin + NFC Khmer; 0 unmatched) | `provenance_facts.json:train_tsv.by_mined_file` |
| **Wikipedia-derived pairs** | **Yes**: 693 rows (224 + 469) | as above |
| **FineWeb-2-derived pairs** | **Yes**: 9,807 rows (9,338 + 469) | as above |
| **Chhunneng examples** | **None in the training rows**: 0 identical pairs with `data/pairs.tsv` (27,766 `chh_dict` + 301 `chh_informal`). 16 Latin strings occur in both, with different Khmer. But see §3: Chhunneng-derived resources shaped the *mining* | set intersection; data builder reads only the two mined files and `synth/examples.tsv` |
| **LLM-generated labels** (including historical drafts) | **None as training labels.** Every row traces to a mined web file. Identical pairs also occur in the LLM draft files (`llm_drafts.tsv` 251, `llm_drafts_r2.tsv` 69, Gemini `synthetic_pairs.tsv` 92); those rows were taken from the web files, where the same spelling appears independently. Whether any *web page* text was itself machine-generated cannot be determined | `exclusion_checks`; `english_phonetic_data.py` (`build`) never opens the LLM files |
| **Owner-authored or owner-selected labels** | **None as training labels.** The builder would have added the owner's 150 example picks (`synth/examples.tsv`), but all 150 were dropped as related to held-out words. The manifest records `owner_examples:heldout_related: 150`, and `supervised_kept` is `mined_web` only. 0 pair or word overlap with `examples.tsv` / `owner_picks.tsv` | manifest counts; `train.tsv` source column |

**Filters applied when building the rows**, all by `english_phonetic_data.py`, counts from the manifest. Of 67,337
mined rows in, 10,031 were kept.

| Filter | Rows dropped |
|---|---|
| D135 noise filter | 23,326 |
| No CMUdict entry | 17,641 |
| Classed `khmer_like` romanisation | 13,433 |
| Related to a held-out word | 940 |
| Target equal to a held-out spelling | 22 |
| Bad Khmer or length | 29 |

The held-out set used by these filters includes the owner's loanword list, the LLM proxy words, benchmark v1 and the
reserve's Latin words. Those sources are used **negatively**: they remove rows and never add any.

## 3. Upstream: how the mined files were made (partially verified)
| Item | Finding |
|---|---|
| Khmer Wikipedia dump | Local `kmwiki-latest-pages-articles.xml.bz2` has sha1 `9ed47fb3…d1a1`, **equal to Wikimedia's published sha1 for `kmwiki-20261001-pages-articles.xml.bz2`** (fetched 2026-10-29) |
| FineWeb-2 shard | Local `train_000_00000.parquet` has sha256 `6f4506f6…e993`, **equal to the Hub LFS hash of `data/khm_Khmr/train/000_00000.parquet` at dataset revision `af9c1333`** (2025-10-27, the current main as of 2026-10-29). Which revision was downloaded on 2026-10-12 was not recorded, but the bytes are identical |
| Mining | `scripts/translit_mine.py`, 2026-10-12: bracketed Latin forms next to Khmer tokens; the Khmer side is verbatim web text, cut at a syllable boundary |
| **Chhunneng-derived influence on selection** | Mining used the learned transliterator then served (`translit._resources()`) for two things: (a) choosing the cut point and the similarity ≥ 0.5 keep-test; (b) the `khmer_like` class, through `romanised_forms.txt` and `known_targets.txt` plus Google-lexicon keys. The served models of that period were trained on the Chhunneng pairs (`model_cards/model_b.md`: "re-ranks with `known_targets.txt` (also Chhunneng)"). So **Chhunneng-derived resources influenced which web pairs were kept and how they were cut. They contributed no label text.** Which exact model file was loaded at mining time is **not recorded** |
| Rules sound-out | `latin_g2p` rules, part of the keep-test; project code |

## 4. Resource roles
| Resource | Optimisation / initialisation | Validation and checkpoint selection | Evaluation | Inference |
|---|---|---|---|---|
| FineWeb-2 mined pairs | **yes** (9,807 rows) | train-val loss tie-break | — | — |
| Khmer Wikipedia mined pairs | **yes** (693 rows) | train-val loss tie-break | — | — |
| Owner loanword list (`../pocket-tts-khmer/configs/latin_khmer.tsv`), owner-verified singles | no (filter only) | **yes**: 86 primary dev items select the epoch (owner added 43, confirmed 21, chosen 5, **LLM-suggested then owner-verified 17**) | yes (final, not opened here) | — |
| LLM proxy track (`llm_suggested_spotchecked`) | no (filter only) | — | yes (exploratory, 400) | — |
| Benchmark v1 panels, reserve Latin words | no (filter only) | — | v1 scored historically; reserve never scored | — |
| CMUdict | no (filter: English-readable words) | — | — | **not used** by the letters model |
| Google Khmer lexicon | no (letters arm); a mining-class filter (§3) | — | diagnostic re-ranker only (not in the product) | — |
| Chhunneng pairs and Chhunneng-trained models | no; a mining-time filter and aligner (§3) | — | — | — |
| `model.npz` + NumPy | — | — | — | **yes**: pure model decoding, beam 4, no lookup lists |

## 5. Missing evidence and unresolved questions
1. **Training-script state at `b9c7ea4`** (dirty, no diff recorded). The weights, data and config are hash-verified;
   the exact script edits are not.
2. **Model used during mining:** the file was not recorded in `datasets/build/translit_mine_*.json`. It was a
   Chhunneng-trained model of that date. Whether that indirect influence matters for release is the owner's call.
3. **FineWeb-2 download revision:** not recorded at the time. The byte identity with `af9c1333` is verified now.
4. **Machine-generated web text:** whether some mined spellings come from pages that were themselves produced by
   machine translation or LLMs is not knowable from the data.
5. **Selection used LLM-suggested, owner-verified spellings** (17 of 86 dev items). They influenced which epoch was
   kept, not the gradient.
6. **Rights in the mined pairs themselves** (short name/word spellings) and in the underlying pages: see
   `NOTICES.md`.
