# E01 — One clean transliterator, with and without a Khmer-pronunciation side task

**Status:** done (registered 2026-10-15 as D173, before any training; result D174, 2026-10-16).
This plan was written in the decision log before training and is copied here unchanged in substance.
Config: `config.yaml`. Code: `scripts/translit_unified.py`. Results: `RESULTS.md`, `results.json`.

## Question
Does auxiliary supervision with Google's Khmer-adapted pronunciations of Latin words improve **one** Latin →
Khmer model covering romanised Khmer, foreign names and English loans? (New to this project, not claimed as novel.)

## Hypothesis
The side task teaches the shared encoder how Latin letter strings sound in Khmer, which should help English loans
most, without hurting romanised Khmer or names.

## Data (identical in both arms; no Chhunneng pair, `known_targets.txt` or old-model output)
- Rule-romanised Google-lexicon pairs, train split (`data/roman/pairs.tsv`): 107,745 kept.
- Web-mined name pairs (`extra_train.tsv`): 33,064.
- The frozen loanword-list train split with live spellings: 1,365.
- The owner's example picks: 135. Existing LLM drafts: 2,177.
- The last three are loans, repeated ×4 as in arm E: 155,517 spelling pairs per epoch.
- **Side task:** Google lexicon Latin entries (a–z, non-empty pronunciation), 6,244 rows; target = phone tokens
  with "." syllable marks. Alternative pronunciations kept as separate rows.

## Held out (never trained on)
- Evaluation panels: romanised test (Chhunneng test split, all attested forms accepted), Wikidata test2 name
  words, pick test 150, dev-41, list test 125, and the 120-word reserve (Latin words filtered, answers never
  opened). Frozen as benchmark `v1`.
- Validation (checkpoint selection only): list val 182 (English), Wikidata dev words 325 (names), 1,000
  hash-chosen romanised val rows (synthetic rule typing).
- **Leak filter:** a training row is dropped if its normalised Latin form meets a held-out word (English sets also
  by simple suffix variants), or its Khmer target is any held-out Khmer target. Applied to every source,
  including the side task. Counts per source and per set are in `models/unified/manifest.json`, with input hashes.

## Arms
- **base:** letters-only encoder–decoder (d 192, 3 + 3 layers, 4 heads, FF 512, dropout 0.1, max input 48).
- **aux:** the same, plus a pronunciation decoder of the same design reading the shared encoder; discarded at
  export.

## Training (both arms)
AdamW lr 1e-3, wd 0.01; 500 warm-up steps then cosine; batch 128; 40 epochs; label smoothing 0.1; cross-entropy
over non-pad tokens; gradient clip 1.0. Spelling batches from a private `random.Random(seed)`: identical order,
steps and schedule in both arms. Shared initial parameters identical. Aux: on every 8th spelling step, one aux
batch of 128 (its own `random.Random(10_000 + seed)` and its own saved torch RNG state, so the spelling dropout
stream is unchanged); loss = spell + 0.5 · aux. Seeds 0, 1, 2, paired. Extra compute recorded, not equalised.

**Checkpoint selection:** after each epoch, the equally weighted mean of greedy exact match on the three validation
sets. Spelling only, never an evaluation panel.

**Re-ranking at inference:** known-word list = Google lexicon Khmer headwords only, identical for both arms. Raw
and re-ranked results both reported.

## Primary measure and adoption rule (written before any result)
- **Primary:** re-ranked exact match on the 150 pick-test words.
- **Advance to independent listening only if all hold:**
  - mean paired improvement over the 3 seeds ≥ **5 points**;
  - positive in ≥ 2 of 3 seeds;
  - mean regression ≤ 1 point, separately, on the romanised test and on test2 names.
- dev-41 is diagnostic only. Reported: every seed, paired word outcomes, and a word-bootstrap 95% CI of the mean
  over seeds (words resampled jointly across arms and seeds).

## Execution plan
Seed 0 for both arms, then a technical review; seeds 1 and 2 only if it is sound. Bugs are documented and the
affected pairs rerun. No recipe change after seeing results.

## Limitations stated up front
- Romanised validation is synthetic rule typing: it measures generalisation from rules, not from real typing.
- The romanised test is Chhunneng's own test split, the same style the served models were trained on.
- The lexicon used for re-ranking contains some held-out Khmer words, as any dictionary would (disclosed, allowed).
