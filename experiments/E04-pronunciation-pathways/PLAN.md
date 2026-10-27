# E04 — Pronunciation pathways: G (letters) vs P (phones only) vs GP (both), conditional three-view model

**Status:** registered 2026-10-27, before any full training. The registration commit is the first commit that
contains this file; its hashes are in `registration.json`. The owner approved this phase in their instruction.
Research only:
- the product default (E02 `letters_s0`) and serving are unchanged;
- E01–E03, their results, checkpoints and benchmarks are preserved;
- no TTS, paid services, GPU, downloads, synthetic labels or owner-written spellings.

## Questions
1. Does English pronunciation alone (P) support a useful English → Khmer prediction pathway on this data?
2. Does joint letter + phone training (GP) fail to keep that capability?

## Starting state (reconciled)
- **Branch and code:** branch `english-phonetic` at `98b501f` (E03, D180).
- **Data:** E03's DACS partitions in `data/e03/`, used unchanged; hashes in `registration.json`.

  | Partition | Pairs | Words | Families |
  |---|---|---|---|
  | train | 2,255 optimisation + 93 train-val (tie-break) | 1,964 | 1,917 |
  | dev (253 primary items) | 323 | 276 | 270 |
  | eval (228 primary items) | 296 | 245 | 239 |

  E03's false-positive filters are not repaired here.
- **Reconciliation:** E03's "3,008 syllable-matched retained pairs" came from a stale build output. The correct
  count is 2,774 of 2,967; an erratum was added to E03's report.
- **Vocabularies:**
  - source: 157 tokens (E02 encoder `en-phonetic-src-v1`: specials, a–z, 69 ARPAbet-with-stress tokens, 55 Khmer
    phones unused here);
  - target: 78 Khmer characters.
- **Pronunciation provider:** CMUdict sha `81917843…3d22`, first listed reading with stress, no G2P. Every DACS
  word has a reading by construction.
- **Checkpoints available:** E02 (`models/e02`), E03 (`models/e03`), untouched.
- **Historical overlap:** E03 `reports/overlap.json`. E01's mined set and models B/E saw most eval pairs; E02's P2G
  saw every target.

## Phone-ambiguity audit (train/dev only; `reports/ambiguity_audit.json`)
- **Material:** 2,098 distinct phone sequences over 2,240 words.
- **Shared sequences:** 131 sequences belong to more than one word (273 words). In 43 of them the words have
  disjoint Khmer targets, e.g. *be/bea/bee*, *bill/bil*. In the others they share a target (*abbot/abbott*).
- **Spelling collisions:** the same Khmer target for different English words occurs too (*accra/agra*). Different
  spellings are not automatically contradictory.
- **Coverage:** unknown phone tokens: 0.
- **Dev primary:** 15 of 253 items have the phones of a training word with disjoint targets, a distinction G can
  keep and P cannot under exact match. Inside dev there are no such conflicts (dev-only ceiling 100%). Both are
  properties of these references, not limits on phonetic correctness.
- **Phone-seen in training:** dev 16/276, eval 15/245. Evaluation is almost entirely on unseen phone sequences.

## Source views (`scripts/e04.py encode`; the same vocabulary and parameter count in every arm)
- **G:** `<task:en>` + letters.
- **P:** `<task:en>` `<sep>` + phones. No letters, word id or spelling length; same phones give identical input
  (tested).
- **GP:** `<task:en>` + letters + `<sep>` + phones, identical to E03's phones encoding.
- **V3** (conditional): one view per example, drawn uniformly from {G, P, GP} by a private RNG; main inference view
  GP.

## Training (`config.yaml`; model, optimizer, batch, dropout 0.1 and label smoothing 0.1 as in E03)
- **Updates:** 3,000 per full run, about 170 passes over the 2,255 rows.
- **Schedule:** linear warm-up to the base rate 1e-3 over 150 updates, then cosine over the remaining 2,850.
  - Tested on the rates the optimizer actually applies: base reached at update 149, decay from update 150.
  - E03's old product schedule peaked at 0.379 × base (at update 299 of 720). This is a correction for E04, not a
    claimed cause of E03's result.
- **No phone or view dropout** in G, P or GP.
- **Pairing:** `torch.manual_seed(seed)` before building the model. The initial-state hash and the batch-order
  hash are recorded (private order RNG `{seed}:order:{epoch}`; the view RNG is drawn in every arm). The same rows
  and weights in every arm.
- **Checkpoint:** every 100 updates, by greedy top-1 macro on the primary dev items in the arm's main view; ties go
  to the lower train-val loss, then the earlier update.
- **Logged:** learning curves, gradient norms, training/dev agreement, the applied learning rate per update, source
  lengths and elapsed time.

## Sanity checks (before full runs; `scripts/e04.py sanity`)
- **Subset:** ≤ 64 training-part words whose phone sequence belongs to one word only and which have one target,
  ordered by sha256(`e04-sanity:`+word). No dev or eval words. The rule and the excluded collisions are recorded.
- **Runs:** G, P and GP for ≤ 500 updates, with dropout 0 and label smoothing 0. Each must reach ≥ 95% greedy exact
  training agreement. This checks implementation and optimisation, not generalisation.
- **On failure:** if P fails while G passes, inspect encoding, targets, gradients, masks and the decoder, and fix
  only demonstrated bugs. At most 6 sanity attempts. If the checks remain unresolved, stop the full experiments
  and report the blocker.

## Pilots and the registered branch (dev only, beam-4, primary dev items)
- **Pilots:** G_s0, P_s0, GP_s0.
- **Useful signal:** P's dev macro > G's, **or** adding P's top-4 to G's top-4 raises dev reference coverage by
  ≥ 5 points.
- **The branches:**
  - **A**, no useful signal: no fusion and no confirmation; report the limitation for this dataset and recipe.
  - **B**, useful signal and GP ≥ G and GP ≥ P: GP is the candidate.
  - **C**, useful signal but GP below G or P: one V3_s0 pilot (same 3,000 updates). Plus an exploratory G/P score
    combination:
    - union of their top-4 dev candidates, rescored under both models (mean log-prob per token, EOS included);
    - weights w ∈ {0, 0.25, 0.5, 0.75, 1} selected on dev;
    - its two-model runtime recorded. It is not a deployable or confirmable candidate.
- **Screen for confirmation** (B and C only):
  - eligible: P, GP, V3;
  - a candidate needs seed-0 dev macro ≥ G + 2 points;
  - the highest wins, ties go to the simpler recipe (P, then GP, then V3);
  - the screen is an engineering rule, not a significance test.
- **Confirmation:** the candidate and G at seeds 1 and 2. Nothing is added for a disappointing result.
- **Budget:**
  - at most 8 full attempts (3 pilots + 1 conditional + 4 confirmation), failures included;
  - 6 sanity attempts;
  - 12 aggregate training hours;
  - one process at a time, 6 threads, CPU only.

## Evaluation (after selection; the historical E03 eval partition)
- **The set:** it informed E03, so it is not fresh evidence; its references are unqualified. E02's final set is
  not read, and no reserve predictions are made.
- **Measures:**
  - raw beam-4 top-1, top-4 and CER for every model and seed;
  - a paired family bootstrap (10,000 resamples, seed 20261027) of candidate − G over 3 seeds, labelled
    confirmatory only if confirmation ran; otherwise one-seed pilot comparisons, labelled exploratory;
  - strata: phone-seen/unseen, the flagged (ambiguous) track;
  - failure types: generation (no reference in the top 4) vs ranking (in the top 4, not top 1);
  - documented alternative CMUdict readings (E03's frozen eval sample) for the phone-using models.
- **Vowels:** assessed only where both outputs are lexicon words (coverage reported); otherwise a stated gap.
- **Donor perturbations:** not repeated (secondary in E03). The consonant screen and reader are not used as
  validation of each other.
- **No improvement here is phonetic accuracy,** and no product default changes.
