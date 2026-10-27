# E04 — Pronunciation pathways: report

**Status:** done 2026-10-27, branch B (GP candidate, confirmed with 3 paired seeds). Research only: the product
default (E02 `letters_s0`) and serving are unchanged. Plan: `PLAN.md` (registered at `d82be16`, before training).

## In plain words
We trained the same small model three ways on the same 2,255 English → Khmer pairs:
- **G** sees only the English letters (`buffet`);
- **P** sees only the CMUdict pronunciation (`B AH1 F AH0 T`), no letters at all;
- **GP** sees both.

Small example from the eval set (seed 0, top-1 guesses; the reference is the mined Khmer spelling):

| Word (phones) → reference | G | P | GP |
|---|---|---|---|
| accel (`AH0 K S EH1 L`) → អាកសែល | អាសែល ✗ (k lost) | អាកសែល ✓ | អាកសែល ✓ |
| guinea (`G IH1 N IY0`) → ហ្គីនី | ហ្គូណា ✗ (reads "gu-na") | ហ្គីនី ✓ | ហ្គូណា ✗ |
| big (`B IH1 G`) → ប៊ីក | ប៊ីជ ✗ (g read as j) | ប៊ីក ✓ | ប៊ីក ✓ |

With the phones, the model "hears" the k in *accel* and the hard g in *big*. GP sometimes uses the phones
(accel, big) and sometimes follows the letters (guinea).

What we found:
1. **Pronunciation alone works, but less well than letters.** P gets about 35% exact on eval vs G's 43%. It is
   a real pathway (it gets 20 eval items right that G misses), not noise and not a broken model.
2. **Letters and pronunciation get different words right.** If we pool G's and P's four best guesses, the right
   spelling is among them for 73% of eval words, vs 65% for G alone.
3. **Feeding both (GP) is a little better than letters, but not detectably.** Over three paired seeds, GP − G is
   +2.2 points, with a 95% interval of −1.5 to +6.0 points. This interval includes zero.
4. **GP does not collect the whole pooled benefit** (65.8% top-4 vs the pooled 72.7%). But we cannot show that GP
   specifically "forgets" P's wins: GP keeps 8 of P's 20 extra items, and it keeps exactly as many (8 of 17) of
   the items that simply a *second G seed* happened to get right. Most of the "P-only" wins are ordinary
   retraining churn.

None of this measures phonetic correctness: the references are mined, unreviewed spellings.

## What ran (budget)
- Sanity: 3 of 6 attempts. G, P and GP each reached 100% exact training agreement on 64 unambiguous words within
  500 updates (`reports/sanity.json`), so P's lower score is not an implementation failure.
- Full runs: 7 of 8 attempts, 2.01 of 12 hours, one process, 6 threads, CPU only, no failures
  (`run_ledger.json`).
  - pilots G_s0, P_s0, GP_s0;
  - confirmation G_s1, GP_s1, G_s2, GP_s2.
- **Schedule check:** in every run the rate the optimizer actually applied started at 6.7e-6, reached the base
  1e-3 at update 149 and decayed from 150 to 3e-10 at update 3,000 (`models/e04/*/done.json`, `lr_trace.json`).
- **Pairing:** the same initial weights hash and batch-order hash for every arm at a given seed (`done.json`).
- **Parity:** NumPy export vs PyTorch on 12 fixtures for all 7 models: greedy outputs equal, max log-prob gap
  ≤ 6e-6 (`models/e04/*/parity.json`).

## Pilots and the registered branch (dev, beam-4, 253 primary items; `reports/pilot_dev.json`)
| Model | Top-1 macro | Top-4 macro | CER |
|---|---|---|---|
| G_s0 | 43.8 | 64.2 | 0.218 |
| P_s0 | 38.8 | 65.2 | 0.235 |
| GP_s0 | 46.2 | 65.0 | 0.219 |

- **Useful signal:** P's macro does not beat G's, but G ∪ P top-4 coverage is 76.0 vs 64.2 (+11.8 points ≥ 5).
  So yes.
- GP ≥ G and GP ≥ P → **branch B**: no V3 run, no score combination.
- **Screen:** GP is +2.4 points over G (≥ 2) → candidate GP; confirmed at seeds 1 and 2.
- **Dev complementarity:**
  - P top-4 hits that G misses: 30; the reverse: 27;
  - top-1: 23 and 36.
- **Learning curves** (`models/e04/*/metrics.jsonl`):
  - greedy dev macro rises to about 0.40–0.45 by update 600 and then stays flat within about ±2 points;
  - training agreement reaches 100% (P: 97%);
  - the selected checkpoints (updates 300–2,100) therefore mostly pick a noise peak on the plateau.

## Evaluation (historical E03 eval set: 228 primary items, 222 families; `results.json`)
This set already informed E03, so it is **not fresh evidence**, and its references are unqualified (mined,
consonant-screened, unreviewed). E02's final set was not read, and no reserve predictions were made.

| Model | Top-1 macro | Top-4 macro | CER | ok / ranking / generation | phone-seen (14) | unseen (214) | flagged (17) |
|---|---|---|---|---|---|---|---|
| G_s0 | 42.7 | 64.6 | 0.204 | 99 / 49 / 80 | 57.1 | 41.8 | 17.6 |
| G_s1 | 41.2 | 66.8 | 0.219 | 94 / 58 / 76 | 35.7 | 41.4 | 17.6 |
| G_s2 | 40.0 | 66.8 | 0.227 | 93 / 60 / 75 | 42.9 | 39.9 | 29.4 |
| P_s0 | 34.7 | 54.1 | 0.242 | 80 / 43 / 105 | 7.1 | 36.4 | 41.2 |
| GP_s0 | 41.4 | 65.8 | 0.225 | 96 / 54 / 78 | 35.7 | 42.1 | 41.2 |
| GP_s1 | 44.5 | 66.8 | 0.216 | 103 / 50 / 75 | 50.0 | 44.4 | 23.5 |
| GP_s2 | 44.6 | 67.7 | 0.209 | 103 / 52 / 73 | 42.9 | 45.0 | 5.9 |

Failure types (counts of items): *ranking* = a reference is in the top 4 but not first; *generation* = no
reference in the top 4.

**Paired family bootstrap** (10,000 resamples, seed 20261027, top-1 macro, points):
- **GP − G, confirmatory, 3 seeds:** per seed −1.28, +3.30, +4.58; mean **+2.20 [−1.48, +6.01]**. No detectable
  effect.
- **P − G, exploratory, seed 0 only:** **−8.03 [−14.64, −1.35]**.

**Reading the strata:**
- **Phone-seen** (14 items whose phones occur in training): P falls to 1 of 14. It cannot tell apart words that
  share a pronunciation but have different reference spellings, which the audit predicted (15 such dev items).
  With 14 items, the other differences are noise.
- **Flagged track** (17 items): too small to read.
- **Failures:** about half of each model's misses are ranking failures, i.e. the reference is already among the
  model's four guesses. P mostly fails at generation (105).

## Post-hoc retention analysis (not registered; saved predictions only; `reports/retention_posthoc.json`)
Seed 0 is the only seed with all three arms.

| Eval top-1 | Items | GP_s0 keeps |
|---|---|---|
| right in P, wrong in G | 20 | 8 |
| right in G, wrong in P | 39 | 27 |
| right in both | 60 | 50 |
| churn baseline: right in G_s1, wrong in G_s0 | 17 | 8 |

- **Top-4:**
  - P-only hits: 18, GP keeps 10;
  - G ∪ P coverage 72.7 vs G 64.6 and GP 65.8.
- **Reading:** GP keeps P's unique wins at the same rate as it keeps a second G seed's unique wins. The data do
  not separate "joint training loses the phone pathway" from ordinary seed-to-seed churn. The pooled top-4
  advantage is real on both dev and eval, but it is a two-model union. GP does not reach it.

## Alternative CMUdict readings (E03's frozen eval sample, 21 cases; `results.json` → `alternative_readings`)
| Model | Top-1 changed | Agrees, default reading | Agrees, alternative | Both outputs in the lexicon |
|---|---|---|---|---|
| P_s0 | 16 | 9 | 3 | 6 |
| GP_s0 | 10 | 8 | 5 | 5 |
| GP_s1 | 7 | 4 | 4 | 3 |
| GP_s2 | 16 | 1 | 4 | 1 |

- P follows the pronunciation, as it must, e.g. *buffet* `B AH0 F EY1` → បាហ្វេ (the final t is dropped).
- GP's response varies a lot by seed (7 to 16 of 21 changed).
- **Vowel gap:** vowel correctness could only be judged where both outputs are lexicon words: 1 to 6 cases per
  model. That is too few to assess vowels, so vowel quality remains unassessed.

## Answers to the two questions
1. **Standalone pathway:** on this data and recipe, P is a usable but weaker pathway: −8 points top-1 vs G
   (exploratory, one seed). Its errors differ from G's enough that the union of their candidates covers about 8
   to 12 more points of references.
2. **Does joint training lose it?** Partly. GP's candidates do not reach the G ∪ P coverage. But GP is not
   detectably worse than G (+2.2 [−1.5, +6.0]), and the per-item loss of P's wins is indistinguishable from seed
   churn. There is no evidence of a specific joint-training failure, and the conditional three-view recipe was
   correctly not run (branch B).

## Limitations
- The references are mined and consonant-screened, not reviewed. Exact match against them is a proxy for
  spelling agreement, not for phonetic accuracy. The vowels are effectively unassessed.
- The eval set is historical: it informed E03's design, and E01/B/E's training saw most of its pairs (E03
  `reports/overlap.json`).
- P was run with one seed only, so its comparison is exploratory.
- 228 items and 222 families: the confirmatory interval is ±3.7 points wide. An effect of about 2 points cannot
  be confirmed or excluded.
- Checkpoint selection on a flat, noisy dev curve adds variance (best updates between 300 and 2,100).
- The pronunciations are CMUdict's first reading (US). There is no G2P for unknown words, so P cannot run on
  out-of-dictionary words, which a product would need.
- The retention analysis is post-hoc and uses one seed.

## Recommendation (not launched)
- The cheapest lead is the ranking failures: about 50 items per model have the reference in the top 4.
- The G ∪ P union shows that candidate diversity exists, but turning it into a product needs reviewed references
  first. Otherwise any reranker is tuned to mined spellings.
- The prerequisite named in E03 still stands: independent review of a reference set.

## Reproduction
```bash
cd ~/projects/khmer-transliterator
export CUDA_VISIBLE_DEVICES=
.venv/bin/python scripts/e04.py audit                     # reports/ambiguity_audit.json, data/e04/phone_seen.json
.venv/bin/python scripts/e04.py sanity                    # reports/sanity.json (state/e04/sanity/)
.venv/bin/python scripts/e04_run.py --config experiments/E04-pronunciation-pathways/config.yaml --resume
#   runs pilots → devcheck (reports/pilot_dev.json) → selection → confirmation → evaluate (results.json)
.venv/bin/python scripts/e04.py retention                 # reports/retention_posthoc.json (post-hoc)
.venv/bin/python -m unittest tests.test_e04
```
Single steps: `scripts/e04.py train --arm GP --seed 1 --out models/e04/GP_s1`, then `export` / `verify` with the
same `--out`, and `devcheck` / `evaluate`.

## Artifacts
- **Plan and registration:**
  - `PLAN.md`, `config.yaml`, `registration.json` (data and code hashes at `d82be16`);
  - `SOURCE_VERIFICATION.md`;
  - decisions log `docs/decisions/e04-pronunciation-pathways.md`.
- **Reports:**
  - `reports/ambiguity_audit.json`, `reports/sanity.json`;
  - `reports/pilot_dev.json` with `pilot_dev_per_item.jsonl`;
  - `reports/retention_posthoc.json`.
- **Results:**
  - `results.json` (sha256 prefix `cc14a19246c770d4`);
  - `results/eval_per_item.jsonl` (`1920dfb31812f448`).
- **Run ledger:** `run_ledger.json` (copy of `state/e04/status.json`).
- **Models** (local, git-ignored): `models/e04/{G,P,GP}_s0`, `{G,GP}_s{1,2}`. Each has `model.npz` (sha256 in
  `results.json`), `best.pt`, `metrics.jsonl`, `lr_trace.json`, `done.json`, `parity.json`, `provenance.json`.
- **Code:** `scripts/e04.py`, `scripts/e04_run.py`, `tests/test_e04.py`.
