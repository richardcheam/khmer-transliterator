# E02 audit: repository state, task policy and usable resources (Task 1)

Audited 2026-10-27 on branch `english-phonetic`, starting from `main` at `b3f2096` (the commit the handoff
inspected; the tree was clean). Numbers below come from `scripts/english_phonetic_data.py build`
(`reports/data_build.json`), unless stated otherwise.

## 1. Code paths actually used

- **Model:** `scripts/translit_train.py` `Model`: a pre-LN Transformer encoder-decoder, d 192, 3 + 3 layers,
  4 heads, FF 512, GELU, dropout 0.1, learned positions. E02 reuses it unchanged in all three arms. The source
  positions are `MAX_SRC = 80`, set before training. The longest eligible encoded source has 34 tokens (English)
  and 39 tokens (P2G).
- **NumPy inference:** `khmer_translit.translit.Transliterator`. It was refactored without any change in
  behaviour: `encode` → `encode_ids`, and `nbest_batch` → `beam_search`. The new API feeds token ids through it.
  The 29 existing tests pass unchanged.
- **Old source encoding (`translit.source`) cuts input silently.** The NumPy `encode` truncates to `max_in` and
  drops unknown characters. The E02 encoder (`phonetic_source.encode_source`) refuses an over-long source
  instead (`too_long`). Its phones are whole tokens (`en:AH1`).

## 2. The E01 phone-input question (stale documentation)

Traced in `reports/encoding_trace.json`:

| Model | Exported `phones` | What the encoder sees for *cookie* |
|---|---|---|
| E (served) | `true`, `max_in` 48 | `cookie` + `|` + one private-use character per CMUdict phone and per stress digit (13 characters) |
| E01 base / aux (all seeds) | `false`, `max_in` 48 | `cookie`: Latin letters only (`translit_unified.encode_batch`) |
| B (served) | absent (letters only) | `cookie` |

**Finding:** E01 never received English phones as input. Its auxiliary decoder predicted the Google-lexicon
(Khmer) pronunciation of Latin words, from letters. Two sentences in `docs/TRANSLITERATOR.md` were wrong and
have been corrected (see §8):
- the sentence calling E01 "phone-aware";
- the sentence describing its side task as "spell Khmer → its Google pronunciation".

Model E is the only earlier model with phone input. It was trained with Chhunneng data, so it is no clean
baseline. E02's `phones` arm is the first controlled test of English phone input on clean data.

## 3. Task policy

The task: an English word, or a name meant to be read as English, written phonetically in Khmer script. It keeps
the sound, not the meaning. The reading is the first CMUdict reading (US English) unless the caller supplies one.
Out of scope: translation, romanised Khmer chat, audio and TTS.

## 4. Source ledger and roles

Full ledger with hashes, licences and limits: `resources.json`. Summary:

| Source | Role in E02 | Tier |
|---|---|---|
| Loanword list (speech repo), owner-verified single words | benchmark pool → `dev` / `final` | attested |
| Loanword list, `llm_suggested_spotchecked` | 400-item `final_proxy` exploratory track | proxy |
| Loanword list, drafts / `attested` conventional forms / multi-word | excluded | — |
| Mined bracket pairs (Khmer Wikipedia, FineWeb-2) | supervised training, filtered | proxy |
| Owner example picks (`synth/examples.tsv`) | **none**: all 150 are also owner-verified list words, so they sit in the benchmark pool | — |
| Google Khmer lexicon, Khmer headwords | P2G pretraining and replay; fixed re-ranker word list | attested resource |
| CMUdict | English pronunciation provider (inference resource) | resource |
| Benchmark v1 English panels, E01 list validation, reserve Latin | historical; excluded from training and from the new splits | — |
| LLM drafts, Gemini spellings, Chhunneng pairs, romanised Khmer pairs | excluded | — |
| MekongPhon, ByT5 G2P | not used (see `resources.json`) | — |

**Rights.** Nothing here establishes legal certainty.
- Mined pairs carry CC BY-SA 4.0 (Wikipedia) and ODC-By plus Common Crawl terms (FineWeb-2). Whether weights
  trained on them are share-alike derivatives is **unresolved**. Trained weights therefore stay local
  (`restricted_artifacts_local_only`) until the owner decides.
- The Google lexicon is CC BY 4.0: attribution required.
- The loanword list is the owner's own work. It contains LLM-drafted rows.

## 5. Supervised data: counts and exclusions

- Mined rows in: 67,337 (3,532 Khmer Wikipedia, 63,805 FineWeb-2). These are merged per Latin–Khmer pair.
- Dropped:

  | Reason | Rows |
  |---|---|
  | D135 noise filter | 23,326 |
  | no CMUdict entry (no English reading) | 17,641 |
  | Khmer-like romanisation (Cambodian names) | 13,433 |
  | related to a held-out word | 940 |
  | Khmer target equal to a held-out reference | 22 |
  | bad Khmer or length | 29 |

- **Kept: 10,031 rows** of 7.5k Latin words; 9,735 train and 296 training-validation (loss tie-break only).
  Every row has a CMUdict reading, by construction. Tier: proxy.
- **Owner examples: 150 in, 0 kept.** All 150 are benchmark-pool words, so the leak filter removes them.
- **What is known to be imperfect:**
  - the alignment is automatic;
  - many entries are non-English names (Japanese, Spanish, Arabic) that Khmer writers spell from the
    original language, not from the CMUdict reading;
  - some forms follow French conventions.

  These rows stay **proxy**. The fixed-seed audit sample (`reports/audit_sample.tsv`: 30 training rows and 40
  dev items, drawn before any E02 model existed) was **not independently adjudicated**, and no pass rate is
  claimed for it.
- Every arm trains on the same rows with the same weights (1 each).

## 6. Pronunciation resource

- CMUdict sha256 `81917843…3d22`, 126k headwords. Headwords are normalised like inputs.
- **Coverage:** 100% of training rows (by construction). On the benchmark, misses are 8 of 103 dev items,
  23 of 191 final items and 78 of 400 proxy items.
- **Ambiguity** (readings differing beyond stress): 17 dev, 24 final and 46 proxy items. These are flagged
  before any prediction, and the flagged final items form a separate diagnostic track.
- **No local English G2P fallback is used.** A miss is encoded `<no_pron>`, explicitly. Both phone arms see
  `<no_pron>` through the registered 10% training dropout.

## 7. P2G resource

- Source: the Google lexicon, 69,415 lines.
  - Dropped: 7,261 Latin or other non-Khmer headwords; 61 empty pronunciations; 284 spellings that equal an
    accepted dev/final/proxy form (so every final item is unseen as a P2G target).
  - **Kept: 61,809** (phones → spelling) pairs, with 3,872 homophone groups kept. 3,093 go to dev, split by
    connected spelling–pronunciation component.
- **Phone alphabet:** Google's Khmer IPA-like symbols, syllable dots included (55 tokens, namespace `km:`; the source vocabulary has 157 tokens, target 78 Khmer characters).
  English ARPAbet phones are separate tokens (namespace `en:`). No conversion map is assumed.

## 8. Changes to earlier documentation

The `docs/TRANSLITERATOR.md` row for the unified E01 models now says letters-only input and a Latin → Google
pronunciation side task. E01's registered results are untouched.

## 9. Assumptions not verified on this machine

- That the owner-verified spellings follow the E02 phonetic policy. They were reviewed for the speech front-end;
  conventional forms may be among them, and 2 final items are flagged `possible_translation`.
- That CMUdict's first reading is the reading the owner meant.
- That mined names are "intended as English": many are not.
- The licence status of trained weights (see §4).
