# E10 report: one bounded improvement attempt and a local ByT5 release candidate (2026-10-29)

**Outcome.**
- **No quality improvement was shown.** The one candidate recipe (A) did not pass the registered gate.
- **E08 `rs_s0` remains the release candidate.** It is packaged as a local research preview,
  `khmer-translit-byt5` 0.2.0, with unverified phonetic quality.
- **Nothing was published, merged, tagged or served.**

## 1. Diagnosis (E08 dev errors and training behaviour; no final-test failures)
`reports/diagnosis.json` has the details.

- **E08 was under-trained.** Dev was still rising at update 3,000, by which point the learning rate had decayed to 0.
  Its final training loss was 0.147, twice E07's 0.074. It reproduced only 57% of the control training rows.
- **Exposure was skewed:** the 1,031 words with several spellings took 46.6% of the training draws (one word was
  drawn 147 times, the median word 19 times).
- **Dev errors** (115 of 253 wrong):

  | Kind | Count |
  |---|---|
  | vowels only | 47 |
  | consonants changed | 22 |
  | consonant added | 17 |
  | consonant dropped | 12 |
  | marks only | 10 |
  | both | 7 |

  Capacity limits were not indicated.

## 2. Candidate recipe (frozen before training)
- **Recipe A:** word-level sampling (each epoch visits every word once and draws one of its spellings uniformly)
  plus learning rate 2e-3.
- **Everything else as E08:** data, base, LoRA, 3,000 updates, batch 32, seed 0.
- **One recipe, not two:** a second recipe would only have varied A's components, which would be an ablation.
- **Gate:** dev greedy exact match at least +3.0 points over E08 (0.542), with CER not higher.
- Registered in `registration.json` and `recipes/A.json` (decision log 1–7).

## 3. Training
| Run | Updates | Hours (6 threads) | Selected update | Dev exact match | Dev CER |
|---|---|---|---|---|---|
| E08 `rs_s0` (baseline, existing) | 3,000 | 5.2 | 3,000 | 54.2% | 0.151 |
| **A_s0** | 3,000 | 5.39 | 3,000 | **55.4%** | **0.141** |

- **A_s0's dev curve:** 41.4 → 47.0 → 53.0 → 55.4 → 55.4% at updates 600 → 3,000. A learned faster, but levelled
  off at 2,400–3,000.
- **Three clean memory pauses**, at updates 992, 1,075 and 1,399, each resumed from `last.pt`. Other programs on
  the laptop were using memory. The owner relaxed the guard and later allowed stopping the metrey-voice workers
  (decisions 8–9).

**Gate: failed (+1.2 points < +3.0).** CER improved by 0.010. As registered: no replication (A_s1, E08_s1 not
trained), E08 kept, and no further searches. One of the four allowed training attempts was used.

## 4. Historical evaluation (track 1; frozen first: `reports/candidates_frozen.json`, commit 4e27f11)
| | Dev greedy | Dev CER | Eval greedy | Eval CER | Eval 4-candidate coverage |
|---|---|---|---|---|---|
| E08 `rs_s0` | 54.2% | 0.151 | 52.4% | 0.161 | 79.1% |
| A_s0 | 55.4% | 0.141 | 53.0% | 0.166 | 78.7% |

**A_s0 − E08**, paired family bootstrap, 95% CI:

| | Exact match | CER reduction |
|---|---|---|
| dev | +1.2 [−3.2, +5.6] | +1.0 [−1.0, +3.1] |
| eval (228 items, 222 families) | +0.6 [−4.5, +5.9] | −0.5 [−2.5, +1.6] |

The eval split has been reused since E03, and its references are mined and unreviewed, so these numbers are
exploratory. Per the protocol they cannot trigger more tuning.

**Error kinds** (eval, greedy):

| Model | vowels only | consonants and vowels | consonants changed | consonant dropped | consonant added | marks only |
|---|---|---|---|---|---|---|
| E08 | 53 | 18 | 12 | 10 | 8 | 7 |
| A_s0 | 47 | 20 | 12 | 11 | 9 | 7 |

**Output health:**
- 0 invalid-orthography, empty, repeated or truncated outputs, for both models and both decodings, on dev, eval and
  the confirmation set.
- **Runtime:** generation took 23–24 s greedy and about 48 s for beam 4, per model per split.

**Confirmation set** (120 fresh words, sampled before any new output): **operational diagnostics only.**
- Every output is valid and distinct.
- The two models give the same greedy answer on only 28% of these words. That shows how much spellings of unseen
  words still vary between runs. It is not a quality score.

## 5. Independent review (track 3)
**Prepared, not collected.**
- **Batch:** 60 dev words with 163 blinded candidates, from both models and the mined references with origins
  hidden; 40 representative words plus 20 from disagreements, reported separately.
- **Importer:** validating; it rejects missing IDs, duplicates and inconsistent judgments, and requires
  provenance.
- **Location:** `data/e10/review/`, kept local because it contains mined references; its manifest and instructions
  are in `review/`.
- No reviewer was contacted and nothing was fabricated.

## 6. Release candidate (E08 `rs_s0`) and packaging
- **Backend:** `khmer_translit.byt5`, CLI `--backend byt5`. The default is still E02.
- **Adapter:** sha256 `d514a51f…88f8`.
- **`reports/release_verify.json`:** 0 mismatches against the evaluated predictions on 276 dev items (greedy and
  top-4); fully offline; load 6.1 s; one word 0.58 s greedy / 0.94 s top-4; peak RSS 2.0 GB.
- **`reports/install_check.json`:** a fresh venv outside the repository, with the wheel installed offline. CLI
  batch, JSONL alignment, top-4, refusals (exit code 1), the missing-base error (exit code 2), no downloads, and
  outputs identical to the repository.
  - **Caveat:** uv's cache could not resolve torch offline, and fetching it (1–3 GB) was out of budget. The extra's
    three packages were borrowed from the research environment via a `.pth` path; `khmer_translit` itself came from
    the wheel.
- **Bundle:** `dist/khmer-translit-byt5-0.2.0/`, plus `.tar.gz` and `.sha256`. It contains the wheel and sdist, the
  adapter, documents, evidence, a prepared Hugging Face folder (not uploaded) and `SHA256SUMS`. It holds no `.pt`
  or `.bin` files, base weights, mined pairs, references or review key. The wheel sha256 is stable across rebuilds.

## 7. Compute
| Item | Amount |
|---|---|
| Training | 5.39 h, one attempt of four (limits: 24 h, 4 attempts) |
| Inference and evaluation | about 0.3 h (diagnosis 515 s, evaluation 314 s, verification and install about 3 min; limit 4 h) |
| Hardware | CPU only, 6 threads |
| Peak training RSS | 5.2 GB |

No paid services, downloads or GPU were used.

## 8. Remaining dependencies (owner)
1. **Independent phonetic review:** without it, quality is unverified.
2. **Publication blockers** (see `release/byt5/NOTICES.md`):
   - the Google AI Mode terms clause on the 384 synthetic rows (the fallback is a retrain on arm R);
   - the CC BY-SA question for the Wikipedia-derived pairs;
   - no code licence has been chosen.
3. **Owner decisions:** whether to publish the HF folder, and what is served. Nothing changed in serving.
