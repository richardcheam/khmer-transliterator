# E03 — English phones under dictionary-assessable supervision (letters vs phones, after the selection repair)

**Status:** registered 2026-10-27, before any E03 training. The registration commit is the first commit that
contains this file. Its hash, and those of the frozen files, are in `state/e03/status.json` (copied to
`run_ledger.json` at the end). Owner's instruction: a bounded two-arm experiment, no architecture search.

**Fixed inputs:**
- Code: branch `english-phonetic`, from `0e5daf3` (E02 and its diagnostics, D178/D179). E02's code, data, results,
  checkpoints, benchmark v1, the product default and serving are not changed.
- Data: `data/e03/` from `scripts/e03_data.py exclusions` + `build`. Hashes are in `data/e03/manifest.json`. The
  audit (counts only) is `reports/data_audit.json`; overlaps are in `reports/overlap.json`.
- Frozen evaluation: `frozen_eval.json` + `frozen_eval.sha256`.
- Detector resources (heuristics; also hashed in the manifest):
  - `models/model/romanised_forms.txt`;
  - `models/model/known_targets.txt`.

  Both are lists derived from the Chhunneng data, used only to *detect*, never as training data.

## 1. The repaired selection rule
**The bug:** `scripts/translit_mine.py` writes `khmer_like` if `latin ∈ romanised_forms` **or**
`khmer ∈ lexicon ∪ known_targets`, and E02 dropped every `khmer_like` row. `e03_data.classify` recomputes the two
conditions separately, and reproduces the stored class for all 65,422 mined pairs.

**The repaired policy:**
- **Latin side romanised Khmer:** still excluded, the declared policy, kept from E02. The detector is a word list,
  so it can collide with real English: 549 such rows have a Latin form that is also a common English word (SCOWL).
  They are recorded as `excluded:latin_romanised_collision`, not declared settled, and stay excluded.
- **Khmer side a known word:** not a reason to reject. These rows are the recovered candidates: 11,377 merged
  (latin, khmer) rows. Of them, 7,714 have a CMUdict word and an exact Google reading before any filter. The
  diagnostics' 8,725 counted both conditions together; the build reproduces exactly 8,725 when it does the same.
- **Unchanged E02 checks, in E02's order** (`filter_reason`):
  - plain a–z;
  - well-formed Khmer target ≤ 40 characters;
  - the D135 noise filter over all mined forms of the word;
  - a CMUdict reading;
  - held-out family;
  - held-out target;
  - target characters in E02's target vocabulary.
- **Held-out sets:** produced by an isolated step (`e03_data.py exclusions`) that mechanically reads the files E02
  held out and writes only the sets to `data/e03/exclusions.json`. It reported counts only: 1,312 words, 1,908
  stems, 674 targets. The files it read:
  - `data/e02/references.jsonl`, all tracks including final;
  - the v1 English panels;
  - the reserve (Latin words only, as in E02);
  - E01's list-validation words.

  No reference was inspected, scored or used for selection.

## 2. The dataset: dictionary-assessable, consonant-screened pairs (DACS)
- **Assessable track:** the target has an exact Google-lexicon reading. 4,239 of the 4,253 pairs that pass the
  filters; 14 have no reading and are kept outside.
- **Screen:** the frozen E02-diagnostics consonant comparison (`compat_row`; rules frozen at `e4c0857`, a test
  checks they are unchanged). It is a **screening proxy, not a certification**:
  - 3,218 `compatible_evidence`;
  - 1,017 `suspected_conflict`;
  - 4 `unknown`.
- **Quarantine** (kept on file in `data/e03/candidates.jsonl`, never trained on):
  - `established_misalignment`: the target is a loan that fits the English better, plus extra lexicon word(s), as
    in +គណនី "account" (98);
  - `unresolved_conflict`: any other suspected conflict (920);
  - `unknown_screen` (4).
- **Retained:** the `compatible_evidence` pairs.
  - Alternative targets of one word are kept together: 356 words have 2 targets, 63 have 3.
  - 137 words keep some targets and have others quarantined.
- **Vowels:** reported separately (syllable count vs English vowel count), never used to accept or reject. Among
  the assessable pairs, 3,862 have equal counts, 291 have one Khmer syllable fewer, and 63 have one more.
- **Splits** by English lexical family (the E02 stems + union-find, salt `e03-assessable-phones`), 80/10/10:

  | Partition | Pairs | Words | Families |
  |---|---|---|---|
  | train (of which 93 train_val, loss tie-break only) | 2,348 | 1,964 | 1,917 |
  | dev (checkpoint selection) | 323 | 276 | 270 |
  | eval (the new mechanism-evaluation partition) | 296 | 245 | 239 |

  A training pair whose target is accepted in dev or eval is removed (250).
- **E02's dev words** (examined in the diagnostics) are held out of all E03 partitions. They are used only as
  secondary development material (`e02_dev`).
- **Overlap with history** (`reports/overlap.json`): E03 targets are lexicon words, so E02's P2G pretraining saw
  them as spellings. The other overlaps (E02 training, E01's mined set, the A/B/E targets, the loanword list) are
  reported as counts.
- **Name:** dictionary-assessable, consonant-screened pairs. Not "phone-consistent": vowels are unscreened, and the
  screen is the same family of rules as the mechanism reader.

## 3. Arms and training (`config.yaml`)
- **The arms:**
  - `letters`: `<task:en>` + letters;
  - `phones`: + `<sep>` + the first CMUdict reading, with 10% `<no_pron>` dropout (the private RNG is drawn in
    both arms).
- **Training:** E02's trainer, unchanged (`scripts/e03_train.py` only points it at E03):
  - d192, 3 + 3 layers;
  - AdamW 1e-3, 500 warm-up steps then cosine;
  - label smoothing 0.1, batch 128, clip 1.0, 40 epochs, 6 threads, CPU FP32.
- **Pairing:** `torch.manual_seed(seed)` with the initial-state hash and the batch-order hash recorded, and the
  same rows and weights in both arms.
- **Seeds:** 0, 1, 2 for both arms. **Six supervised attempts at most, failures included**; 12 aggregate hours;
  one process at a time; no P2G or replay.
- **Declared before training:** 2,255 training rows make 18 batches per epoch and 720 steps, so the unchanged
  500-step warm-up covers 69% of training. The registered recipe is kept, without a schedule change.
- **Checkpoint:** per run, the best epoch by greedy top-1 macro on the primary-eligible E03 dev items; ties go to
  the lower train-val loss, then the earlier epoch. Final scoring is beam-4.

## 4. Measurement (`scripts/e03_eval.py`, frozen in `frozen_eval.json` before training)
- **Item sets:**
  - `eval`: primary = items without the `pronunciation_ambiguous` / `possible_translation` flags;
  - `dev`: secondary; it was used for checkpoint selection;
  - `e02_dev`: secondary.

  E02's final set and the reserve are refused by the path guard.
- **Quality proxies (registered primary):** raw beam-4 top-1 multi-reference exact match (NFC), macro over lexical
  families, on the primary `eval` items; phones − letters per seed and the mean.
  - 95% paired family-bootstrap interval: 10,000 resamples, seed 20261027, per-family differences averaged over
    the seeds.
  - Also: item accuracy, top-4, CER (paired family interval of the CER difference), and the flagged track.
  - E02's evidence fields with the +5-point threshold are reported for continuity.
  - **No re-ranker:** every E03 reference is a lexicon word, so a lexicon re-ranker would leak the answer.
- **Mechanism (the E02-diagnostics protocol):**
  - Letters fixed. Conditions: default phones, `<no_pron>`, a donor's phones (another word of the same set, phone
    count ±1, different reading, seeded), and the other CMUdict readings.
  - Per row: top-1 change, top-4 overlap, rank of the default top-1, teacher-forced length-normalised log-prob
    change, and consonant movement towards the given phones (frozen reader: closer / farther / same /
    unreadable).
  - The change type: consonant-preserving means a vowel or diacritic change.
  - Where both outputs are lexicon words: syllable-count movement towards the given vowel count. Reader coverage is
    reported.
  - **Registered primary mechanism measure:** under donor phones, phones arm, the share of items whose top-1
    consonants move closer minus the share moving farther, mean over seeds. Family-bootstrap interval; letters as
    the negative control (expected 0).
  - E02's diagnostics found 2–7% of donor cases moving closer; that figure is historical context only.
- **Controls:**
  - encoded phones exactly as intended;
  - letters outputs identical under every condition;
  - teacher-forced score equal to the beam score;
  - measurement controls on the eval references;
  - export parity per run.

## 5. Interpretation, fixed now
- **The eval partition is unqualified:** mined spellings, screened by the same family of consonant rules that the
  mechanism reader uses. Agreement on it is a quality proxy only. Steering on consonants is partly circular with
  the screen, so neither validates phonetic quality.
- **Sensitivity is not correctness;** resisting deliberately wrong phones is not a defect.
- **No product default changes,** whatever the result. The E02 comparisons are context only: the data changed, so
  no cross-experiment difference is attributed to the repair alone.
- No threshold changes, reference edits, extra arms, seeds or schedule changes after training.
