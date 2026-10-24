# Latin → Khmer transliterator

The speech model reads **Khmer script only**. Any Latin-script word in the input must be rewritten into Khmer
spelling before synthesis. This document covers what the transliterator does, the data, the model, every version
trained, where it stands (2026-10-24) and how to work on it.

**Goal now:** **one** clean (releasable) model for all Latin input, replacing the two served today (B for names and
romanised Khmer, E for English words) and dropping the Chhunneng data, whose origin is undocumented.

Decision numbers up to D174 refer to the archived decision log of the speech repo (`git -C ../pocket-tts-khmer show archive/checkpoints-p0-p6:state/DECISIONS.md`); later ones are in `docs/DECISIONS.md`.

## 1. The job

| Kind of input | Example in | Wanted out | Convention |
|---|---|---|---|
| **Romanised Khmer** (Khmer words or names typed in Latin) | sophea, panha, knhom, nham bay | សុភា, បញ្ញា, ខ្ញុំ, ញ៉ាំបាយ | ph = ភ/ផ, ch = ច/ជ, nh = ញ, ea = ា (2nd series), ou = ូ |
| **Foreign names** | ronaldo, chelsea | រ៉ូណាល់ដូ, ឆែលស៊ី | how Khmer writers spell them (web-attested) |
| **English words, read the Khmer way** | safety, thunder | សេហ្វធី, ថាន់ឌ័រ | Khmer-adapted reading, not native English: ៊/៉ series shifts, ៍ on silent finals, ហ្វ for f, ័រ for -er |

The output must be a valid Khmer spelling the TTS can read. Exact match is the main metric; for English words the
owner's ear is the final judge, and several spellings can be acceptable.

### Where it sits (speech repo: `src/khmer_tts/frontend.py::resolve_latin_words`)
1. **Loanword list** `configs/latin_khmer.tsv` in the speech repo (4,901 entries, exact lookup). If found, done.
2. Capitals of 1–6 letters (ABA) are spelled by letter names, unless seen as a name (`translit.caps_reads_as_word`, `models/model_b/name_counts.tsv`).
3. Run-together listed words are split (phoneshop → ហ្វូន សប).
4. **Transliterator** `src/khmer_translit/translit.py::guess_many`:
   - an English dictionary word (`word_tag` = "en": in `data/en_common_words.txt`, SCOWL) → **model E**;
   - otherwise a word with f, q, w, x or z → English sound-out rules (`latin_g2p`);
   - otherwise → **model B**.

   Each model's beam-4 output is re-ranked by `pick` (§3.4). Results are cached per word.

The speech repo's `configs/serving.json` chooses the models (`translit_model=model_b`, `translit_router=all`,
`translit_en_model=model_e`) and passes them to `translit.configure`; without a call, the package uses the same
defaults (`translit.SERVED`).

## 2. Data (`data/`, not in git; trained models in `models/`)

| File | Rows | Contents | Origin / licence | Notes |
|---|---|---|---|---|
| `pairs.tsv` (from `src/chhunneng/`) | 28,067 (train 24,184 / val 1,643 / test 2,240, grouped by Khmer word) | Romanised Khmer → Khmer: mostly dictionary words, 301 informal chat spellings (knhom → ខ្ញុំ). No names, no English | github.com/Chhunneng/khmer-text-transliteration @ fe8028f1; MIT declared, **origin of the pairs undocumented** | **Release blocker.** Its test split is the only large real-typing test |
| `mined_kmwiki.tsv`, `mined_fineweb.tsv` | 3,532 / 63,805 | Latin ↔ Khmer pairs mined from brackets ("ឈ្មោះ (Name)") in Khmer Wikipedia and FineWeb-2 | CC BY-SA 4.0 / ODC-By 1.0 | `scripts/translit_mine.py` |
| `extra_train.tsv` | 35,291 | Filtered union of the mined pairs (`scripts/translit_extra.py`) | as above | Held 99 of the 150 pick-test words (D163); filtered copy `synth/extra_notest.tsv` |
| loanword list (speech repo, `khmer_translit.loanwords`) + frozen split `list_split.json` (train 1,556 / val 182 / test 125) | 4,901 | English and foreign words → Khmer. Provenance in `reports/latin_audit.json` | ours; LLM drafts owner-spot-checked | New list words do not enter training on their own |
| `llm_drafts.tsv`, `llm_drafts_r2.tsv` | 2,399 / 1,856 | English words spelled by LLM agents in the owner's style (D147, D151) | LLM output | r2 helped nothing (E2a) |
| `synth/examples.tsv` / `synth/test.tsv` | 150 / 150 | The owner's 300 spelling picks, split into training examples and a **test-only** half | owner-verified | test half = the "pick test" |
| `roman/pairs.tsv` | 118,308 | **Clean** replacement for Chhunneng: 61,032 Google-lexicon headwords romanised by rules (`src/khmer_translit/romanise.py`) + one variant each | Google lexicon CC BY 4.0 + our rules | Used by the unified model (§4) |
| `names_dev.tsv` / `names_test2.tsv` | 184 / 131 names | Wikidata Cambodian person names (CC0), dev and a frozen test | CC0 | `src/wikidata/persons_kh.csv` |
| `e2/reserve.json` | 120 English words | Untouched reserve for a final owner A/B | — | **Never score it** |
| `src/` | 2.2 GB | Raw sources: chhunneng, cmudict, fineweb2_km, kmwiki, wikidata, … | see `SOURCES.md` | |

Inference resources: Google lexicon `data/lexicon/km_lexicon.tsv` (62,175 headwords, CC BY 4.0; the
`known` list for `pick`), `models/model/known_targets.txt` (Chhunneng-derived, must go for a release), CMUdict
`src/cmudict/cmudict.dict` (phone-aware source encoding), SCOWL `en_common_words.txt` (the router).

### Evaluation sets
| Set | Size | Caveat |
|---|---|---|
| Romanised test (Chhunneng test split) | 2,226 scored | Same style as Chhunneng training: easy for Chhunneng-trained models |
| Wikidata names test2 | 223 words | Official spellings are idiosyncratic (Norodom) |
| Owner-pick test | 150 English words | Answers come mostly from E's top 5 |
| dev-41 | 41 English words | Development only; a single retrain moves it ±5 words |
| List test split | 125 | Small |
| Owner blinded listening / text A/B | 10–40 items | **Decides what is served** |

## 3. Model

### 3.1 Architecture (`scripts/translit_train.py::Model`; served in NumPy by `src/khmer_translit/translit.py`)
- Character-level Transformer encoder–decoder, pre-LayerNorm: d 192, 3 + 3 layers, 4 heads, FF 512, GELU,
  dropout 0.1, learned positions. About **2.58M parameters**.
- Vocabularies: the characters seen in training + PAD 0 / BOS 1 / EOS 2.
- Source encoding (`translit.source`): letters only (B), or **phone-aware** (`--phones`, E): letters + "|" + the
  CMUdict pronunciation, one private-use character per phone; `--max-in 48`.

### 3.2 Training (`scripts/translit_train.py`, CPU)
Cross-entropy with label smoothing 0.1; AdamW lr 1e-3, weight decay 0.01; 500 warm-up steps then cosine; batch 128;
40 epochs. The checkpoint with the best greedy exact match on validation is kept. `export --out DIR` writes
`translit.npz` (float32 arrays + JSON meta), which serving loads without torch.

### 3.3 Inference
NumPy re-implementation (matches torch to 1e-6); batched beam search with a key/value cache, beam 4. 10 new words
take about 160 ms on CPU.

### 3.4 Re-ranking (`translit.pick`)
From the beam, choose the best-scoring candidate that is a **known Khmer word** within 3.0 log-prob of the top one;
otherwise the top candidate.

## 4. Versions trained

| Model | Recipe | Key results | Status |
|---|---|---|---|
| **A** `models/model/` | Chhunneng only, letters | Romanised 69.2%; owner A/B vs rules 8 / 2 / 2 (D134) | default weights of `translit.load()` |
| **B** `models/model_b/` | A + mined pairs | Romanised 70.0%; **names test2 48.5%**; owner A/B vs A 9 / 2 / 1 (D138) | **Served** (names, romanised Khmer) |
| **E** `models/model_e/` | B + loanword list + LLM drafts ×4, phone-aware | Better on English (10 / 4), worse on names; owner A/B 14 / 8 / 8 (D150) | **Served** (English dictionary words) |
| C, D, T, E2a/E2c, s0, s1 | list, drafts, tags, fine-tunes, Gemini synthetic data | None beat E or B by its registered rule (D146–D164) | weights archived |
| **Unified** `models/unified/{base,aux}_s{0,1,2}/` (D173/D174) | **One clean model** (no Chhunneng): `roman/` + mined + list + drafts + picks, phone-aware; **aux** adds a side task (spell Khmer → its Google pronunciation) | Side task: pick test **+4.67 pts** over base (every seed positive, CI [+2.0, +7.6]); no harm to romanised or names. **Missed the registered 5-pt gate.** vs served: romanised 33–35% vs **70%**, names 35–44% vs 48%, pick test 21–29% vs 31% | not served |

**Owner listening, clean `aux_s1` vs served (10 sentences, `reports/translit_listen_clean.json`):** served
only right 3, clean only right 0, both 5, neither 2. The clean model matched served on names and English words and
lost only on everyday romanised **chat words** (mok, ot, te, rean, nis, laor).

## 5. What has been learned
1. **The remaining gap is data, not the model:** a clean source of real romanised typing (chat words, as people
   text them). Rule-romanised dictionary words do not transfer to how people type. Options:
   - ask the Chhunneng author where their pairs came from (if clean, the problem disappears);
   - collect real typed examples (owner and testers typing common words; Khmer web text with romanised brackets;
     the speech repo's `data/demo/unknown_latin.jsonl` from real use);
   - ship with weaker romanised reading, the loanword list covering common words.
2. **The pronunciation side task is worth keeping** in any future recipe: a consistent small gain at about +7% compute.
3. **Two models exist today** because adding English loans made one model better on English but worse on names.
   One tagged model (T) came close but missed the romanised bar.
4. **More LLM-written data stopped helping** (D, E2a, s1). English errors are mostly convention choices (vowel
   length, series marks, ័រ vs ឺរ), not missing sounds.
5. **Noise floor:** one retrain moves dev-41 by about ±5 words; compare with paired seeds, as the unified runs did.
6. **Leakage:** mined pairs contained 99 of the 150 pick-test words. Filter every test set out of every training
   file (`translit_unified.py` does this).
7. **The speech model also misreads some correctly spelled loanwords** (fiberglass, application): loan sounds are
   only 1.7% of its training audio. No transliterator can fix those; front-end respellings or new recordings can.

## 6. Working on it

| Task | Command (from the repository root) |
|---|---|
| Compare served vs clean on any words | `.venv/bin/python scripts/translit_try.py nham bay safety ronaldo` |
| Build the source pairs and splits | `.venv/bin/python scripts/translit_data.py` |
| Mine bracket pairs (Wikipedia, FineWeb-2) | `.venv/bin/python scripts/translit_mine.py --source kmwiki` (or `--source fineweb`), then `scripts/translit_extra.py` |
| Rule-romanised clean pairs | `.venv/bin/python scripts/translit_roman.py build` (`tune` scores the rule variants) |
| Train / export a served-style model | `.venv/bin/python scripts/translit_train.py train [flags] --out models/NAME`, then `export --out models/NAME` |
| Unified clean model: data, train, export, score | `.venv/bin/python scripts/translit_unified.py data`, then `train --arm aux --seed 0`, `export --arm aux --seed 0`, `score` |
| Mirror a training run to W&B | `.venv/bin/python scripts/track_translit.py models/unified/aux_s0` |
| Blind listening A/B for the owner (needs the speech model) | in the speech repo: `scripts/translit_listen_clean.py gen` / `queue` / `report`, answered in `scripts/review_app.py` |
| Tests | `.venv/bin/python -m unittest discover -s tests` |

**Rules that kept results trustworthy:** write the adoption rule down before seeing results; never tune on the test
sets or the reserve; compare with paired seeds; the owner's blinded A/B decides what is served.

**Release constraints:** no Chhunneng pairs or `known_targets.txt`; attribution for the Google lexicon (CC BY 4.0),
FineWeb-2 (ODC-By) and Wikipedia (CC BY-SA). The evaluation scripts for older arms are archived in the speech
repo's tag `archive/checkpoints-p0-p6` (`archive/scripts/transliterator/`); their model weights are on the HDD
(`/mnt/hdd/data/khmer-tts-archive/data/translit/`).
