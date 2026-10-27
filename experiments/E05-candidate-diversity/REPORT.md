# E05 — Candidate diversity: report

**Status:** done 2026-10-27. Registered at `06ef30a` (protocol and hashes committed before any new prediction).
**Label:** exploratory on historical data: the dev and eval partitions were already examined in E03/E04. The
references are mined, consonant-screened and unreviewed. No training, no reranker, no product or serving change.

## In plain words
E04 found that pooling four guesses from the letters model (G) with four from the pronunciation-only model (P)
contains the right spelling more often than G's four guesses alone. But the pool has up to **8** guesses, against
4. E05 asks the fair question: is P's pool better than other ways of getting 8 guesses?
- **G8:** the same letters model's top-8;
- **G+G:** four guesses each from two letters models trained with different seeds.

Small example (eval, beam-8; ✓ = the mined reference is among the guesses):

| Word (phones) → reference | G's top-8 | P's top-4 |
|---|---|---|
| guinea (`G IH1 N IY0`) → ហ្គីនី | ហ្គូណា, ហ្គូណេ, … all "gu-n…" ✗ | ហ្គីនី ✓ |
| pearl (`P ER1 L`) → ភើល | ភាល, ភាល់, ភាលែល, … ✗ | ភើល ✓ (it "hears" *er*) |

Such words exist, but only 5–7 eval words per G model are found by G+P and by neither control. Meanwhile 15–18
words are found by a control but not by G+P. Over the whole eval set:

| System (eval, 228 items, beam-8) | Macro coverage, mean of 3 | Distinct guesses per word |
|---|---|---|
| G4 | 66.4 | 4.0 |
| G8 | 72.2 | 7.97 |
| G+G (3 pairs) | 73.1 | 6.0 |
| G+P (3 combinations, shared P_s0) | 73.9 | 6.6 |

**Answer:** on the registered primary set, G+P does not clearly beat either control: +1.6 points over G8
[−1.1, +4.5], +0.8 over G+G [−2.3, +3.9]. Most of E04's "+8 points" came from simply having more guesses. Coverage
is an oracle: it says the right answer is somewhere in the list. It does not say a system could pick it, and it
says nothing about how good the pronunciation is.

## What ran
- **Predictions** (`predictions/`, 10 files; 48 s, one process):
  - beam-8 for G_s0, G_s1, G_s2 and P_s0 on all dev (276) and eval (245) items;
  - beam-4 for dev G_s1/G_s2 (primary items), which E04 had not saved.
  - Checkpoint hashes were checked against `registration.json` before decoding (they equal E04's `results.json`).
- **Analysis:** `results.json` and `results/per_item_{dev,eval}_w{4,8}.jsonl`.
- **Cost:** `reports/cost.json`, 5 fresh processes, 52 s.
- **Review queue:** `reports/review_queue_manifest.json`; the files are local in `data/e05/review/`.
- **Total computation:** about 2 minutes of the 2-hour budget. CPU only, one job at a time, 6 threads.

## Reproduction of E04 (beam-4, saved outputs)
- **Eval:** G_s0 top-4 64.56, G_s0 ∪ P top-4 72.67.
- **Dev:** 64.2 and 76.0.
- All four equal E04's reported numbers (`results.json` → `e04_reproduction`).
- At beam-4, "G8" cannot exist: the beam returns at most 4 candidates. The beam-4 configuration shows it as
  under-filled (duplicate rate 0.50) instead of mixing in beam-8 candidates.
- **Search width alone** (G's top-4 at beam-8 vs beam-4): −1.2 to +1.2 points. This is reported separately and
  not attributed to pronunciation (`search_width_effect`).

## Primary comparisons (eval primary, beam-8, macro coverage, paired family bootstrap)
Settings: 222 families, 10,000 resamples, seed 20261027, the same family draws for every system.

| Per G checkpoint | G+P − G4 | G+P − G8 | G+P − G+G (first pair) | G+P − G+G (second pair) |
|---|---|---|---|---|
| G_s0 | +8.7 [5.1, 12.6] | +2.3 [−1.8, 6.3] | +0.6 [−3.9, 5.0] (s0,s1) | +0.6 [−3.2, 4.4] (s0,s2) |
| G_s1 | +7.4 [4.1, 11.0] | +1.1 [−1.8, 4.1] | +2.0 [−2.1, 6.0] (s0,s1) | +0.6 [−3.2, 4.2] (s1,s2) |
| G_s2 | +6.5 [3.6, 9.9] | +1.5 [−1.7, 4.7] | +1.1 [−2.4, 4.7] (s0,s2) | −0.3 [−4.1, 3.6] (s1,s2) |

| Summary over the fixed combinations | Points [95%] |
|---|---|
| G+P − G8 | **+1.63 [−1.08, +4.51]** |
| G+P − G+G | **+0.75 [−2.25, +3.90]** |
| G+P − G4 | +7.51 [+4.66, +10.51] |
| G8 − G4 (more guesses, same model) | +5.88 [+3.85, +8.13] |
| G+G − G4 (second letters seed) | +6.76 [+5.11, +8.56] |
| Equal 4 slots: interleaved G+P@4, G first − G4 | +1.13 [−2.18, +4.66] |
| Equal 4 slots: interleaved G+P@4, P first − G4 | +0.53 [−3.00, +4.28] |
| Equal 4 slots: G+P@4 (G first) − G+G@4 | −0.38 [−3.53, +2.85] |
| Equal 4 slots: G+P@4 (P first) − G+G@4 | −0.98 [−4.43, +2.63] |
| Equal 4 slots: G+G@4 − G4 | +1.50 [−0.23, +3.23] |

**What the intervals mean:**
- they are conditional on these four trained checkpoints;
- the comparisons share checkpoints, and every G+P shares the single P_s0;
- they do not estimate how a newly trained G+P ensemble would vary across training runs;
- the three G+P rows are not independent replicas.

**Registered decision rule** (`results.json` → `verdict`): **no clear advantage over G8 and G+G**.
- Both summary lower bounds are below zero.
- The uncertainty flag is set because per-combination signs are mixed (eight of nine estimates are positive, one
  is negative).
- The intervals are all narrower than 10 points.
- Point estimates of +0.6 to +2.3 are not evidence of complementarity.

## Coverage versus candidate budget (eval, beam-8, means over checkpoints / pairs / both orders)
| k | G (own rank) | G+P, G first | G+P, P first | G+G, both orders (range) |
|---|---|---|---|---|
| 1 | 41.3 | 41.3 | 34.7 | 41.3 (40.0–42.7) |
| 2 | 55.3 | 56.6 | 53.9 | 55.7 (55.1–56.2) |
| 4 | 66.4 | 67.5 | 66.9 | 67.9 (65.9–69.7) |
| 8 | 72.2 | 73.9 | 73.9 | 73.1 (72.7–74.0) |

- Interleaving is a fixed, reference-free diagnostic order, not a trained reranker.
- Neither order was chosen. Putting P first costs 6.6 points at k = 1.

## Candidate sets (eval, beam-8)
- **Distinct candidates:**
  - G8: 7.97 per word, 0.3% duplicates;
  - G+P: 6.6 per word, 17% duplicate slots (162–168 of 228 unions are under 8);
  - G+G: 6.0 per word, 25% duplicate slots.
- **Union sizes:** "8 slots" is not 8 distinct guesses. Two letters seeds overlap more than G and P do.
- **Top-4 overlap with P:** 1.3–1.4 shared candidates (Jaccard 0.23–0.26) between G and P. Between two G seeds,
  2.0 shared (Jaccard 0.36–0.39).
  - P is more different from G than a second G seed is.
  - That extra difference turns into almost no extra coverage.
- **Pool of all four models** (top-8 each, about 21 distinct guesses): the reference is present for 188 of 228
  items and absent from every list for 40.
- **Words found by G+P and by neither control:** 6 / 5 / 7 for G_s0 / G_s1 / G_s2 (e.g. *guinea*, *pearl*,
  *breeze*, *austen*, *intel*, *michele*).
- **Reverse:** words found by some control and not by G+P: 18 / 15 / 16. Words found by every control and not by
  G+P: 1 / 1 / 0.

## Strata (eval, beam-8, G_s0 systems; item coverage except the flagged row, which is family macro; no intervals)
| Stratum | Items | G4 | G8 | G+P | G+G (s0,s1) | G+G (s0,s2) |
|---|---|---|---|---|---|---|
| phone-unseen | 214 | 64.0 | 70.1 | 72.9 | 71.5 | 72.0 |
| phone-seen | 14 | 78.6 | 92.9 | 85.7 | 92.9 | 85.7 |
| G_s0 top-1 is a lexicon word | 135 | 90.4 | 92.6 | 92.6 | 94.8 | 92.6 |
| G_s0 top-1 is not a lexicon word | 93 | 28.0 | 40.9 | 46.2 | 40.9 | 44.1 |
| flagged (ambiguous reading), macro | 17 | 29.4 | 47.1 | 64.7 | 52.9 | 58.8 |

- **Dictionary-hit stratum:** every primary reference is a lexicon headword by construction (228/228). That
  stratum is therefore degenerate and was replaced by the reference-free flag shown.
- **Where G+P helps:** its largest gaps appear where G's best guess is not a real Khmer word, and on the 17
  flagged items. Both groups are small, have no intervals and were not the registered test.

## Dev (descriptive; 253 primary items, 250 families)
On dev, G+P beats both controls:
- G+P − G8 +4.87 [+1.67, +8.27];
- G+P − G+G +5.13 [+1.87, +8.67];
- all nine per-combination estimates are +3.8 to +6.8;
- at an equal 4 slots, G+P@4 − G+G@4 is +3.8 to +3.9 (lower bounds about +0.4).

Eval, the registered primary set, does not reproduce this. All four checkpoints were selected on these dev items
(greedy top-1 in E04), and dev informed earlier decisions. The dev–eval gap is reported, not explained, and it is
not used to change the conclusion.

## Practical cost (`reports/cost.json`)
Settings: 100 primary dev items, batch 1, 6 threads, 3 timed repetitions, a fresh process per system, on an
i7-8750H (12 logical CPUs, 7.6 GB, Python 3.10, NumPy 2.2.6).

| System | Warm ms/word (median) | Cold load | Peak RSS | Resident models | Distinct guesses |
|---|---|---|---|---|---|
| G4 (beam-4) | 14.7 | 0.04 s | 131 MB | 1 | 4.0 |
| G8 (beam-8) | 21.9 | 0.04 s | 131 MB | 1 | 7.9 |
| G+P (beam-4 each) | 30.2 | 0.08 s | 151 MB | 2 | 6.4 |
| G+P (beam-8 each, matched protocol) | 47.2 | 0.08 s | 151 MB | 2 | 6.3 |
| G+G (beam-4 each) | 31.9 | 0.08 s | 151 MB | 2 | 6.0 |

- **Memory:** peak RSS includes about 35 MB of imports. Each model adds about 21 MB once loaded. Both models must
  stay resident for a two-model system.
- **Speed:** G8 is about 27% faster than G+P at beam-4 and gives more distinct guesses. G+P also needs a
  CMUdict reading for every word, and there is no G2P for words outside the dictionary.
- **Equal slots do not mean equal compute.**

## Conclusion and next direction
- **Candidate-generation opportunity:** on the registered eval set, pooling P's candidates with G's adds no
  clear coverage beyond G8 or a second G seed. The E04 pooling result does not establish special value from
  pronunciation. Dev shows an advantage that eval does not reproduce, so the evidence is mixed at best and
  conditional on one P checkpoint.
- **Ranking without references:** not measured. The k = 1 rows only show that putting P first loses 6.6 points.
- **Phonetic quality:** not measured. The references are unreviewed.
- **Direction (registered "no clear advantage" branch):**
  - keep the simpler candidate generator: one letters model, with beam-8 if more guesses are needed;
  - do not build a pronunciation-specific reranker;
  - do not launch more phone or fusion training.
- **Optional next step:** an independent review of the queue below. It would turn mined references into
  reviewed ones, which is the missing prerequisite named in E03/E04. Any later ranking study should wait for that
  evidence.

## Blinded review queue (prepared, not collected)
- **Files** (local only, git-ignored; nothing is published and no reviewer has been contacted):
  - `data/e05/review/review_queue.tsv` and `review_queue.json`;
  - `data/e05/review/origin_key.json`, kept separate;
  - hashes in `reports/review_queue_manifest.json`.
- **Content:** 56 eval words, 726 rows.
  - Samples by pattern: G+P only 6 (all there are), controls only 10, all cover 10, none cover 10, mixed 10, and
    10 where G and P share their top-1.
  - The candidates are the deduplicated union of G_s0 top-8, G_s1/G_s2 top-4, P_s0 top-4 and the mined
    reference(s), in random order with the origin hidden.
  - Fields: acceptable / unacceptable / uncertain (several may be acceptable) and a comment.
- **Not done:** no judgments have been filled in.

## Limitations
- **References:** mined, screened and unreviewed. Coverage is reference agreement, not phonetic correctness.
- **Historical data:** the eval and dev sets were examined before. Dev was also used for checkpoint selection.
- **Single P model:** one P checkpoint (P_s0) underlies every G+P figure. The intervals are conditional on the
  four checkpoints and say nothing about retrained ensembles.
- **Small strata:** 14 phone-seen, 17 flagged, 93 / 135 lexicon-flag items, all without intervals.
- **Unmatched distinct counts:** G+P averages 6.6 distinct guesses, G8 7.97 and G+G 6.0. The comparison is over
  equal slots, not equal distinct candidates.
- **Timings:** wall-clock on one laptop; they vary from run to run.

## Reproduction
```bash
cd ~/projects/khmer-transliterator
export CUDA_VISIBLE_DEVICES=
.venv/bin/python scripts/e05.py register      # already done and committed (06ef30a); rerunning rewrites registration.json
.venv/bin/python scripts/e05.py run           # predict → analyse → cost → queue, skipping finished steps
.venv/bin/python scripts/e05.py analyse       # or any single step; `run --force` reruns all
.venv/bin/python -m unittest tests.test_e05
```
State and progress: `state/e05/status.json`, `state/e05/PROGRESS.md`.

## Artifacts
- **Plan and registration:**
  - `PLAN.md`, `config.yaml`, `registration.json` (commit `99723ed` at registration time, checkpoint, data and
    code hashes, decoder);
  - decisions log `docs/decisions/e05-candidate-diversity.md`.
- **Predictions:** `predictions/{dev,eval}_w8_{G_s0,G_s1,G_s2,P_s0}.jsonl`, `predictions/dev_w4_{G_s1,G_s2}.jsonl`.
  Unchanged E04 files are reused: `E04-pronunciation-pathways/results/eval_per_item.jsonl` and
  `reports/pilot_dev_per_item.jsonl`.
- **Results:** `results.json` (all systems, contrasts, verdict, strata, overlap, pool) and
  `results/per_item_*.jsonl`.
- **Reports:** `reports/cost.json`, `reports/review_queue_manifest.json`.
- **Code:** `scripts/e05.py`, `tests/test_e05.py`.
