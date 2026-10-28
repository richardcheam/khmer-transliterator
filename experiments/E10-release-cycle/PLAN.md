# E10 — release cycle for the ByT5 English → Khmer transliterator (master plan, 2026-10-28)

**Status:** this plan, `registration.json`, `recipes/A.json`, the review batch manifest and the confirmation manifest
are committed **before** any E10 training or any output on the confirmation words. Progress lives in `state.json`;
budgets and events in `runs/ledger.jsonl`; decisions in `docs/decisions/e10-release-cycle.md`.

## Starting point (verified, not assumed)
| Assumption | Verified by |
|---|---|
| **E08 is the leading research candidate.** `models/e08/rs_s0/best.pt`, update 3,000, sha256 `d5da8c29…`. Its dev greedy top-1 is 54.2% (E07: 52.2%). | `done.json`; E09 `results.json` |
| **Greedy is the working single-answer decoder.** On historical eval, E08 greedy is 52.4% against 47.5% for the registered beam-4. | E09 REPORT §5 |
| **Candidate lists use canonical beam search,** `early_stopping="never"`, length penalty 0. | E09 `selected_settings.json`, D187 |
| **The old configuration stays reproducible:** `e07.gen` defaults still equal the registered call. | E09 reproduction (exact); `test_registered_call_unchanged` |
| **The E03 references are weak evidence:** mined, consonant-screened, unreviewed and repeatedly examined. | E03 `REPORT.md`; reference `review` fields (`reviewer: none`) |

- **Untouched:** E02 final reference spellings and the reserve stay unopened (every read goes through `e06.guard`).
- **Preserved:** old results, checkpoints and registrations are never overwritten.

## Phases
1. **Policy** (`docs/PHONETIC_POLICY.md`): one authoritative phonetic policy and the three evidence tracks, which
   are never pooled.
2. **Review preparation:**
   - **Dev-review batch** (`review/`, data in `data/e10/review/`, local because it holds mined references):
     - 60 dev words: 40 representative, plus 20 where E07 and E08 greedy disagree, reported separately;
     - ≤ 4 candidates per word, origins hidden, deterministic order.
   - **Import:** `scripts/e10.py import --judgments FILE` validates the file (missing ids, invalid values,
     duplicates, edited candidates, NONE contradictions) and needs a `.provenance.json` declaring the track.
   - **Confirmation manifest** (`confirmation/`): 120 fresh families (ordinary, irregular, names; 40 each), frozen
     before any E10 output. Without independent judgments, only operational diagnostics are reported for it.
3. **Diagnosis** (`reports/diagnosis.json`, dev and training data only):
   - **Under-fitting:** dev was still rising at the last checkpoint with the LR decayed to 0. Final train loss is
     0.147, against 0.074 for E07. The model reproduces 57% of its control rows and 11% of the synthetic rows.
   - **Exposure imbalance:** the 1,031 words with several spellings take 46.6% of the draws (up to 147 draws per
     word, against a median of 19).
   - **Dev errors (115 wrong of 253):** vowels only 47, consonants changed 22, consonant added 17, consonant
     dropped 12, ៉ / ៊ only 10, both 7. These categories are automatic, against mined references.
4. **One candidate recipe, A** (`recipes/A.json`): word-level sampling plus peak LR 2e-3; everything else is E07's
   recipe.
   - **Why one:** capacity is not indicated, because E07 fitted its data with the same adapter. A second recipe
     would only vary A's components, which would be an ablation.
   - **Not separable:** the two changes' contributions stay unresolved.
5. **Training:** the A_s0 schedule is fixed (≤ 3,000 updates or 6 h; checkpoints at 600/1,200/1,800/2,400/3,000).
   - **Selection:** dev greedy top-1 macro, then CER, then the earlier checkpoint. This is a proxy, because no
     reviewed dev judgments exist.
   - **Gate:** A_s0 dev top-1 ≥ E08 + 3.0 points with dev CER not higher.
   - **If the gate passes:** train A_s1 and E08_s1 (the E08 recipe at seed 1).
   - **If it fails:** no replication; E08 stays the release candidate.
   - **Limit:** at most 3 training attempts are planned, against a budget of 4.
6. **Freeze, then evaluate** (frozen before historical scoring):
   - candidates: E08 rs_s0, plus A runs if trained;
   - decoding: greedy for the single answer, and canonical beam-4 for candidate lists (equal budget);
   - reported: dev, historical eval (exploratory), seed-specific results, paired family bootstrap, validity, error
     categories, runtime and memory. The confirmation set gets operational diagnostics only.
7. **Product:** a `khmer_translit.byt5` backend and a CLI profile.
   - The historical default (the NumPy E02 model) is unchanged.
   - Torch and Transformers are an optional extra.
   - Adapter export to safetensors; the base model pinned by revision.
8. **Release verification and documents:**
   - fresh-environment install and offline check, with parity, length limits and error messages tested;
   - model card, benchmark card, provenance and permission ledger, README, reproduction guide, changelog;
   - Hugging Face directory contents, local archive and SHA256SUMS. **No publishing.**

## Budgets (enforced by `scripts/e10.py LIMITS` and the ledger)
- **Machine:** CPU only, 6 threads, one training process at a time; no other heavy model job while training.
- **Training:** ≤ 4 attempts and ≤ 24 training hours.
- **Inference:** ≤ 4 h of inference/evaluation.
- **Memory:** RSS ≤ 5 GiB with ≥ 1.5 GiB available.
- **Not allowed:** paid services, model or corpus downloads, GPU, publication, merge, tag, or serving change.
