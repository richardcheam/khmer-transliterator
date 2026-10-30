# E13 report: pronunciation-guided adaptation of ByT5-small (D196)

**Result:** uncertain change. On the owner's blinded test of 120 fresh words, E13 and E11 are both acceptable on
90/120 (75.0%), difference 0.0 [−7.5, +7.5], 11 vs 11 discordant words, exact p = 1.0. The pronunciation input is
used only weakly. E13 is not adopted. No second seed.

## What was changed (one combined change, no ablations)
- **Input:** the word plus its first CMUdict reading with stress, `word|ARPAbet` (for example `fire|F AY1 ER0`).
  When the word has no reading the input is `word|<no_pron>`; phones are never invented.
- **Data:** 90% replay of E11's arm R (4,764 rows) and 10% owner-accepted spellings of difficult words. Those come
  from 42 train families (23 irregular, 56 spellings), sampled uniformly by family and then by alternative.
- **Masking:** a fixed 10% of draws had the reading replaced by `<no_pron>`; targets were unchanged.
- **Recipe:** initialised from E11's adapter (`286292a4…`). AdamW with lr 3e-4, warm-up 100, linear decay over
  2,000 updates. Effective batch 32 (2 × 16 micro-batches). Seed 0, CPU, 6 threads.
- **Amendments before training** (PLAN.md): the labelled source format ran out of memory, so the compact `word|phones`
  format and micro-batches replaced it. Recorded with a regression test before the run.

## Prior evidence and references
- **Earlier experiments:** E02–E05 gave phone input on small models no detectable benefit: E02 +1.65, E03 +0.60,
  E04 +2.20 points, all CIs including 0. This was contrary evidence before E13 started.
- **References:** I05-1040 and ngo15 (pronunciation-informed transliteration), CMUdict and ByT5. Applicability is
  recorded in PLAN.md.
- **E13 result:** it agrees with that negative evidence.

## Collection (owner judgments)
- **Reviewed:** 100 families frozen before candidates, 200 spellings (E11 greedy plus one distinct canonical-beam
  alternative).
- **Answers:** 92 acceptable, 100 not, 8 uncertain.
- **Feasibility:** passed, with 42 train families (23 irregular) and 30 dev families having accepted spellings.

## Training and selection
- **Run:** `pron_s0` stopped on the 6 h cap at update 1,969.
- **Collection-dev CER** (registered rule, lower is better):

  | Update | 400 | 800 | 1,200 | 1,600 | 1,969 |
  |---|---|---|---|---|---|
  | CER | 0.152 | **0.085** | 0.141 | 0.133 | 0.135 |
  | Exact | 40.0% | 56.7% | 43.3% | 43.3% | 40.0% |

- **Selected:** update 800 (checkpoint `e32cd817…`), frozen with all predictions before test generation (05f7f37).
- **Historical dev top-1 (monitoring only):** flat at 53–55% throughout.
- **Selection noise:** the collection dev has only 30 scored words, so one word is worth about 3 points.

## Dev diagnostics (exploratory)

| Measure | E13 | E11 | E13 − E11 |
|---|---|---|---|
| Collection dev CER (30 words) | 0.085 | 0.125 | E11's greedy was itself a candidate there |
| Historical dev top-1 (253) | 54.6% | 53.0% | +1.6 [−2.4, +5.6] |
| Historical eval top-1 (228) | 51.1% | 52.3% | −1.3 [−4.8, +2.3] |

Operational checks were clean on every set: no empty outputs, no truncation, no repeats, valid orthography.

## Mechanism: is the reading used? (dev only, diagnostic)
- **Output changes:**

  | Condition | Collection dev (40) | Historical dev (253) |
  |---|---|---|
  | Reading removed (`<no_pron>`) | 10% of outputs change | 7.1% change |
  | Another word's reading | 15% change | 7.9% change |
  | Output moves towards the donor word | 0 | 0.4% |

- **Missing-pronunciation behaviour:** quality barely changes without the reading. Collection dev CER goes from
  0.085 to 0.106 (exact 56.7% to 46.7%); historical dev CER from 0.165 to 0.167 (exact unchanged at 54.9%).
- **Coverage:** CMUdict covered all R, dev and test words, so `<no_pron>` never occurred at test time.
- **Conclusion:** the model still mostly reads the letters. The owner's example *proffered* (`P R AA1 F ER0 D`) still
  sounds the "-ed".

## Primary test (owner, blinded, predeclared analysis)
- **Set:** 120 families (40 ordinary / 40 irregular / 40 names), frozen before training; greedy E13 (with reading)
  vs E11 (word only).
- **Review:** 169 deduplicated spellings in 2 blinded batches, all judged explicitly; one non-independent judge.
- **Results:**

  | | E13 | E11 |
  |---|---|---|
  | Acceptable | 90 | 90 |
  | Not acceptable | 29 | 29 |
  | Uncertain | 1 | 1 |
  | Share acceptable | 75.0% | 75.0% |

- **Paired:**
  - both acceptable 79, E13 only 11, E11 only 11, neither 19;
  - difference 0.0 [−7.5, +7.5] (family bootstrap, 10,000 resamples);
  - exact paired test p = 1.0;
  - sensitivity: uncertain counted as acceptable or excluded, 0.0 either way.
- **Secondary groups** (exploratory, not confirmed):

  | Group | E13 | E11 | Difference | Discordant |
  |---|---|---|---|---|
  | Ordinary | 32/40 | 34/40 | −5.0 [−22.5, +12.5] | 5 vs 7 |
  | Irregular | 21/40 | 20/40 | +2.5 [−10.0, +15.0] | 4 vs 3 |
  | Names | 37/40 | 36/40 | +2.5 [−5.0, +10.0] | 2 vs 1 |

  No group crosses the registered −10-point trade-off flag.

## Decision
- **Class:** uncertain change (registered rule: the CI includes 0). Not equivalence: about ±7.5 points remain
  possible.
- **What the combination achieved:**
  - closer agreement with the owner's accepted dev spellings (CER 0.085 vs 0.125);
  - no change in the owner's blinded acceptability on fresh words;
  - no loss on the historical sets.
- **Cause:** not identifiable without ablations, which were excluded by design. The mechanism check shows the
  reading has only a weak influence, so any dev gain more likely comes from the accepted examples than from the
  pronunciation.
- **Not done:** E13 is not adopted over E11 and gets no replication or second seed. Nothing is published, packaged,
  served, merged or tagged.
- **Data stays local:** weights in `models/e13/`, raw spellings and judgments in `data/e13/`.
- **Next:** probably not more phone input to this model. Irregular spellings remain the weakest group (≈ 50%).

## Files
PLAN.md, registration.json, collection/ and test/ (manifests, frozen hashes, batch hashes), runs/ledger.jsonl,
predictions/ (dev and eval), reports/ (start checks, collection, pipeline checks, frozen selection, mechanism,
test_scores.json), results.json. Code: `scripts/e13.py`; review page: `scripts/e12_review.py --data data/e13/test`;
tests: `tests/test_e13.py`.
