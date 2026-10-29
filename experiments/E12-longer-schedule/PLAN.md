# E12: longer schedule for ByT5-small (6,000 updates): plan (written before training)

**Question.** Does E11's working recipe, given a longer optimisation schedule (6,000 updates with linear decay over
6,000), produce better English → Khmer phonetic spellings than E11?

This is a performance step, not an ablation. Architecture, real training data and sampling stay fixed. One run.

## Starting state (`reports/start_checks.json`, all pass)
| Check | Value |
|---|---|
| E11 selected checkpoint | update 3,000, sha256 `286292a4…9713` (= E11 `frozen.json`) |
| Base | `google/byt5-small` revision `68377bdc…d49d`, weights `5c5aaf56…931a` |
| Arm R | `data/e08/training/arm_R.tsv` `d00706cf…ec88`: 4,764 optimisation rows (+ 93 train-val), hash `5dd38022…` |
| E11 recipe | E07 `full_recipe.json`, cfg hash `f5d7cf92…` |
| Owner dev judgments (E11 review) | 60 words, 109 candidates, none blank; single owner judge; dev evidence only; none of the 60 families in R |
| E09 decoding | greedy `num_beams=1`, `max_new_tokens=121`; canonical beams use `early_stopping="never"` |

E08 appears only as a historical column. It gives no labels and no initialisation.

## Phonetic policy (`docs/PHONETIC_POLICY.md`, unchanged; clarifications for this phase)
- **Rendering:** render the shown English reading (CMUdict, General American) in readable Khmer script. A
  translation is never acceptable.
- **Several spellings:** more than one adaptation may be acceptable.
- **Missing sounds:** missing sounds or syllables are not acceptable. This is distinct from an acceptable
  approximation (policy rules 4 and 7).
- **Names:** follow the shown English reading.
- **Explicit answers:** each judgment is `acceptable`, `not_acceptable` or `uncertain`. A skipped answer is
  **missing, never accepted**. This changes the convenience rule used in the E11 dev review, where skipped
  single-spelling words were counted as acceptable at the owner's instruction.
- **The judge:** the owner is a legitimate single judge, but not independent of the project. Results are described
  that way.

## Test inputs (frozen before training: `test/input_manifest.jsonl`, `test/frozen.json`)
- **Size:** 120 word families, one item each: 40 ordinary, 40 irregular spelling-to-sound, 40 names. Each has its
  one CMUdict reading (ambiguous readings excluded).
- **Length quota per category:** 13 words of ≤ 4 letters, 14 of 5–8 and 13 of ≥ 9. Irregular words have only six
  of ≤ 4 letters; the shortfall is filled from the other bins and reported.
- **Balance:** stressed-vowel caps (≤ 5 per vowel per category) and consonant-cluster caps (≤ 24 per category) in
  a first pass.
- **Seed:** seeded order (`e12-test`). Words are not chosen by model behaviour: no model has been run on them.
- **Exclusions:**
  - protected families: E03 dev/eval, E02 held-out sets, the benchmark panels and the reserve, read as Latin
    inputs only;
  - every historical Latin token in the repository;
  - every arm R word and E08 synthetic input;
  - the E10 confirmation words and the E10/E11 review words;
  - romanised forms, abbreviations and ambiguous readings.
- **E10 confirmation set not reused:** its usage audit is clean (operational diagnostics only), but its lengths
  are unbalanced (6/120 words ≤ 4 letters, 61 ≥ 9) and its names include derived forms.
- **Amendments before freezing** (on reading the inputs; no outputs existed):
  - name filter extended to *-ize* forms and plurals;
  - one slur excluded;
  - stricter one-per-family keys (*lighten*/*lightest*).
  Each redraw used the same seed.
- **Scope:** a single-judge test of this declared sample under this policy, not a universal benchmark.

## Recipe (`registration.json`)
- **Unchanged from E11:**
  - ByT5-small with the pinned base and a fresh LoRA adapter, seed 0;
  - LoRA r16 / α32 / dropout 0.05 on attention + FFN, base dropout 0.1;
  - AdamW lr 1e-3, betas (0.9, 0.999), weight decay 0, gradient clip 1.0;
  - batch 32 (no accumulation), warm-up 100;
  - row-level sampling on arm R only.
- **Changed:** `max_updates` 6,000, linear decay over the new 6,000-update horizon, `max_hours` 12, checkpoints at
  1,200 / 2,400 / 3,000 / 3,600 / 4,800 / 6,000, and the selection rule below.
- **Fresh start:** this is a new schedule from the base, not a resume of E11's exhausted scheduler.
- **Discrepancies:** listed in `discrepancies_vs_e11`. E07's recipe still names beam 4 as its headline decode;
  E09 superseded that, and both E11 and E12 use greedy.

## Training safety
- **Compute:** CPU only, 6 threads, one process, one attempt.
- **Memory guard:** 3 consecutive readings with RSS > 6.5 GiB or MemAvailable < 0.4 GiB trigger a resumable stop.
- **Resumable state:** `last.pt` holds the adapter, optimiser, scheduler, torch RNG, epoch/position and elapsed
  time. It is saved every 100 updates and at each checkpoint.
- **Resume checks:** config, data and base hashes are checked before the model is built. A started run without a
  resumable state is refused rather than restarted as an uncounted attempt.
- **Time accounting:** active time per segment is in the ledger.
- **Time cap:** if the 12 h cap stops the run between checkpoints, the final state is also evaluated and is
  eligible.

## Checkpoint selection (automatic, fixed now)
- **Rule:** lowest dev greedy mean CER, then highest dev exact agreement, then the earlier update.
- **Data:** historical dev set, mined references.
- **An imperfect surrogate:** chosen so the run finishes unattended. It does not select for phonetic quality.
- **Owner dev judgments:** they cover only old outputs. They are reported as coverage (known acceptable, not
  acceptable, uncertain, unknown) and never rank checkpoints.

## Historical diagnostics (exploratory, after freezing)
- **Freeze first:** the selected checkpoint and the greedy dev/eval predictions of E12, E11, E07 and E08 are hashed
  before scoring.
- **Reported:**
  - exact agreement and CER;
  - paired family bootstrap (10,000, seed 20261027): E12 − E11, with E07/E08 historical;
  - validity, missing endings, length bins, error kinds;
  - curves, learning rates, resources.
- **No follow-up:** no tuning or further run follows from these.

## Prospective blinded test (after selection)
- **Outputs:** greedy E12 and E11 on the 120 inputs, hashed before any judgment.
- **Shown to the judge:**
  - the word and its reading;
  - deduplicated outputs with seeded labels (one candidate when both agree).
- **Hidden:** model identity, scores and references.
- **Batches:** four batches of 30, local in `data/e12/test/`, with the origin key in a separate file.
- **Importer:** rejects unknown ids, edited candidates, invalid values, conflicting duplicates and missing
  answers.

## Test analysis (fixed now; `TEST_ANALYSIS` in `scripts/e12.py`)
- **Primary:** E12 − E11 in the confirmed-acceptable share over all 120 items. `uncertain` counts as not
  confirmed.
- **Also:**
  - counts per value; uncertainty rates;
  - paired both / E12 only / E11 only / neither;
  - family-level paired bootstrap 95% interval;
  - two-sided exact McNemar test on discordant clear pairs (pairs with an `uncertain` are not in this test).
- **Sensitivity:** uncertain counted as acceptable (n = 120); items with any uncertain excluded (n stated).
- **Wording:** a non-significant difference is not equivalence. Test judgments never enter training or checkpoint
  choice in this cycle.
- **Next step:** replication is recommended only if the test evidence warrants it, and is never launched
  automatically.
