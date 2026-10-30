# E13: pronunciation-guided adaptation of ByT5-small: plan (written before candidates and training)

**Question.** Does one combined change produce better English → Khmer phonetic spellings than E11? The change has
three parts:
- a small set of owner-accepted examples of difficult English readings;
- the English pronunciation written next to the word in every input;
- replay of the real-data pool R.

**Type:** a performance-discovery step, not an ablation. If it helps, it will not show which part helped.
**Baseline:** E11 (`models/e11/r_s0/best.pt`, update 3,000, sha256 `286292a4…9713`), word-only input. E12
showed no benefit from a longer schedule (D195), so that approach is not repeated.

## Rationale and prior evidence (kept in view)
**Our own earlier evidence is negative or inconclusive for phone input:**
- **E02 (D178):** small-model phones vs letters +1.65 points [−1.65, +4.94]; inconclusive.
- **E03 (D180):** +0.60 [−2.40, +3.75], no detectable effect. Donor phones changed 34% of outputs, but net
  consonant steering was only +1.4% [−0.8, +3.5].
- **E04 (D181):** letters + phones vs letters +2.20 [−1.48, +6.01], not detectable. Phones alone were weaker
  (−8 points) but complementary.
- **E05 (D182):** pooling letter and phone candidates showed no clear advantage over more letter candidates.

**What differs here:**
- **Model:** ByT5-small adapted from E11, which already writes the task, instead of a small model trained from
  scratch.
- **Supervision:** owner-accepted targets for difficult readings, where the spelling alone is the weakest guide.
- **Judging:** a blinded owner judgment on fresh words, not agreement with mined references.

A negative result here would add to that evidence, not overturn it.

**External motivation (hypotheses only; none shows a Khmer ByT5 benefit):**

| Source | What it supports | Limit for us |
|---|---|---|
| Oh & Choi 2005, *An Ensemble of Grapheme and Phoneme for Machine Transliteration*, IJCNLP ([I05-1040](https://aclanthology.org/I05-1040/)) | Combining grapheme and phoneme information can beat either alone for English → Korean/Japanese | Statistical models of 2005, other scripts; the abstract is not on the landing page, so this summary is from the paper's known content |
| Ngo et al. 2015, *Phonology-augmented statistical transliteration for low-resource languages*, Interspeech ([ngo15](https://www.isca-archive.org/interspeech_2015/ngo15_interspeech.html)) | Target-language phonology helps when training pairs are few (Vietnamese, 94 pairs) | Statistical system, Vietnamese; Khmer phonology is not modelled here |
| CMUdict ([cmusphinx/cmudict](https://github.com/cmusphinx/cmudict)) | US-English ARPAbet readings with stress; free use | US accent; names and rare words missing; one reading chosen per word |
| Xue et al. 2022, *ByT5*, TACL ([2022.tacl-1.17](https://aclanthology.org/2022.tacl-1.17/)) | Byte models do well on spelling- and pronunciation-sensitive tasks | No transliteration into Khmer; nothing about phone strings as input |

## Input and pronunciation policy
- **Task:** render the specified English pronunciation in readable Khmer script (`docs/PHONETIC_POLICY.md`,
  unchanged, sha256 `9847667e…`).
- **Source format:** `word: <word>\npronunciation: <ARPAbet with stress>`. It is lower-cased by the existing byte
  encoder, which is lossless for ARPAbet.
  - Example: `word: yacht` / `pronunciation: Y AA1 T`.
  - A word without a reading gets `pronunciation: <no_pron>`. Phones are never invented.
- **Lookup:** the pinned local CMUdict (`data/src/cmudict/cmudict.dict`, sha256 `81917843…`) through
  `khmer_translit.phonetic_source.get_pronunciation`. The first reading is the established default (policy rule 1).
  No pronunciation model and no external API.
- **New collections and the test:** the reading is fixed in the manifest before any output exists, with its id
  (`cmudict:<word>`).
  - Words whose readings differ beyond stress (heteronyms) are excluded.
  - Names keep the English CMUdict reading, never an original-language reading.
- **R rows:** the same lookup. Ambiguous words take the first reading, and missing words take `<no_pron>`; both
  are counted.
- **Limits:** source ≤ 192 bytes, target ≤ 120 bytes. Longer inputs are refused, never truncated.
- **Interfaces:** the inference interface still takes English text, and the lookup is automatic. E11's
  word-only interface is unchanged.

## Frozen inputs (`collection/`, `test/`; frozen before any candidate or output)
- **Collection:** 100 new families.

  | Split | Irregular spelling | Ordinary | Names | Total |
  |---|---|---|---|---|
  | Train | 40 | 10 | 10 | 60 |
  | Dev | 20 | 10 | 10 | 40 |

  - **Irregular words:** chosen by E06's difficulty rule, round-robin over six declared patterns: silent letter,
    unusual vowel, consonant digraph, spelling-dependent consonant, cluster or ending, other. Each pattern is split
    2:1 between train and dev.
  - **Ordinary words and names:** seeded order, at most 3 per stressed vowel.
- **Test:** 120 new families (40 ordinary, 40 irregular, 40 names) from E12's balanced sampler. This is a declared
  balanced distribution, not a sample of real usage.
- **Disjoint from:**
  - R and E08 inputs;
  - every historical Latin token, the protected families (E03 dev/eval, E02 held-out, benchmark panels and the
    reserve, read as Latin inputs only);
  - E10 confirmation, E10/E11 review and E12 test words and their families;
  - each other. One family per item; the overlap audits in both `frozen.json` files are 0.
- **No outcome-based selection:** words are not chosen by how any model does on them; no model had been run on
  them when they were frozen.
- **Limitation noted at freezing:** SCOWL's capitalised entries include demonyms and terms among the names
  (*guatemalan*, *neapolitan*, *paleocene*). They were kept to keep E12's declared sampler unchanged.

## Collection review (first owner step)
- **Candidates (E11, word-only interface):** at most two per word, the greedy output and the first well-formed
  canonical beam-4 candidate (`early_stopping="never"`) that differs from it. Identical spellings are shown once.
  No hosted model.
- **Page:** the local click page (`scripts/e12_review.py --data data/e13/collection`). It shows the word, its
  reading and seeded A/B labels.
  - Each spelling gets acceptable, not acceptable or uncertain; both or neither may be acceptable.
  - No typing, no scores or ranks. The origin key is a separate file.
- **Saved:** raw judgments, click timestamps (`review_log.jsonl`), the policy hash and provenance. The judge is the
  owner: single and project-affiliated, not independent.
- **Datasets:**
  - accepted spellings become targets, with all alternatives kept;
  - rejected spellings never become targets, and uncertain ones are quarantined;
  - a family with no accepted spelling stays unlabelled;
  - mined R spellings are untouched.
- **Feasibility minimum before training** (not power):
  - ≥ 30 train families with an accepted spelling, of which ≥ 20 irregular;
  - ≥ 15 dev families with an accepted spelling.
  - **If short:** the audit is completed and replacement items are prepared by the same rules. Nothing is weakened.
- **Existing judgments:** the about 300 E11/E12 judgments are not reused as training data. E12's test stays a
  regression set only.

## Recipe (frozen in `registration.json` before training)
- **Initialisation:** E11's verified adapter on the pinned base (`google/byt5-small` `68377bdc…`). This is task
  adaptation, with a fresh optimiser and scheduler.
- **Unchanged from E11:** LoRA r16 / α32 / dropout 0.05 on attention + FFN, base dropout 0.1.
- **Optimiser:** AdamW with E11's betas (0.9, 0.999), weight decay 0 and clip 1.0. Peak learning rate 3e-4, 100
  warm-up updates, linear decay over 2,000 updates.
- **Run:** batch 32, seed 0, at most 2,000 updates or 6 active hours.
- **Sampling, every block of 100 updates (3,200 draws):**
  - exactly 2,880 draws from R, in R's convention (successive seeded permutations), and 320 from the reviewed
    train families (family uniform, then an accepted alternative uniform);
  - exactly 320 draws (10%), in independent seeded positions, get `<no_pron>` in place of the pronunciation, with
    the target unchanged. This is part of the combined recipe, not a separately validated improvement.
  - Actual draw counts are in `registration.json` → `exposure`.
- **A bounded hypothesis:** these learning-rate and mix values are not claimed optimal, and there is no search.

## Pipeline validation (before the run; `reports/pipeline_checks.json`)
- **Inputs and targets:** exact source format and lookup, target encoding and EOS, masking that changes only the
  pronunciation field, byte limits with refusal on overlength.
- **Adapter:** strict E11 loading, which must reproduce E11's cached dev outputs.
- **Sampling:** block quotas and determinism.
- **Smoke:** 3 updates (the longest batch first) check gradients, parameter changes, valid generation, time and
  memory. The smoke adapter is saved separately in `models/e13/smoke/`; the registered run starts again from
  E11's file.
- **Defects:** a pipeline defect gets a focused regression test before training. A falling loss is not evidence
  that the pronunciation is used.

## Training and selection
- **Compute:** CPU only, 6 threads, one process, one attempt.
- **Memory guard:** 3 consecutive readings with RSS > 6.5 GiB or MemAvailable < 0.4 GiB give a resumable stop.
- **Resume:** `last.pt` every 100 updates, with hash-checked resume. A lost resumable state is refused rather than
  restarted.
- **Checkpoints:** 400 / 800 / 1,200 / 1,600 / 2,000.
- **Selection (automatic, fixed now):**
  1. lowest mean CER of the greedy output against the accepted spellings of the reviewed dev collection;
  2. then highest exact agreement with any accepted spelling;
  3. then the earlier update.
- **Selection caveat:** those references are reviewed E11 candidates and are incomplete. CER and exact agreement
  are surrogates; a novel spelling is unjudged, not wrong. Known-judgment coverage is reported separately.
- **Monitoring:** historical dev (mined references) at each checkpoint, for regressions only. It never selects.
- **Freeze:** the selected checkpoint and the generation settings (greedy, `max_new_tokens` 121) are frozen and
  hashed before any test output.

## Mechanism check (dev items only, after freezing)
- **Conditions:** letters fixed; correct, missing (`<no_pron>`) and a frozen mismatched (deranged) pronunciation.
- **Reported:**
  - output changes;
  - whether mismatched outputs move toward the donor word's output;
  - examples;
  - missing-pronunciation scores;
  - dictionary coverage.
- **Limits:** sensitivity is not correctness, and mismatched inputs are not scored as tasks. No run follows from
  this check.

## Prospective blinded test (second owner step)
- **Outputs:** greedy E13 (word + looked-up reading, asserted equal to the frozen intended reading) and E11 (word
  only), hashed before judging.
- **Shown:** outputs deduplicated, with seeded labels. Hidden: model identity, scores and references.
- **Batches:** at most 150 spellings each; the origin key is separate.
- **Judgments:** explicit acceptable / not acceptable / uncertain. A blank is missing, never accepted. No typing.

## Test analysis (fixed now; `TEST_ANALYSIS` in `scripts/e13.py`)
- **Primary:** E13 − E11 in owner-confirmed top-1 acceptability over all 120 items. Uncertain counts as not
  confirmed.
- **Also:**
  - counts and uncertainty rates;
  - the paired table (both, E13 only, E11 only, neither);
  - a family bootstrap 95% interval (10,000, seed 20261027);
  - an exact McNemar test on clear discordant pairs.
- **Sensitivity:** uncertain counted as acceptable; items with any uncertain excluded.
- **Secondary:** the same analysis for ordinary, irregular and names (40 each). A gain in one group with losses
  elsewhere is shown, and subgroup differences are not confirmed findings.
- **Decision:**

  | Class | Rule |
  |---|---|
  | Promising improvement | Primary interval lower bound > 0 |
  | Regression | Primary interval upper bound < 0 |
  | Uncertain change | Anything else |

  - Any group with a point difference ≤ −10 points is flagged as a trade-off.
  - Missing-pronunciation behaviour is reported beside the decision.
- **Wording:** non-significance is not equivalence. The combined result does not identify a cause, and there is no
  ablation yet.
- **Next steps:** replication only if warranted, never launched automatically.
- **Isolation:** test judgments never enter training, recipe or checkpoint choice. Mined-reference results are
  exploratory only.
