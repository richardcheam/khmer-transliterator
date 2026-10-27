# Benchmark card — `en-phonetic-v1` (E02)

| | |
|---|---|
| **Task** | one English word or name → its English pronunciation written in Khmer script (not a translation) |
| **Files** | `data/e02/references.jsonl` (all tracks), `data/e02/dev_references.jsonl` (dev only, the one file the trainer reads) |
| **Hashes** | references `11d89a0a…f0b5`, dev `58541efd…2989` (full values and every input hash in `data/e02/manifest.json`) |
| **Built by** | `scripts/english_phonetic_data.py build` (deterministic; `check` re-verifies the hashes) |
| **Qualification** | **unqualified** (see below): results on it are *reference agreement*, not phonetic accuracy |
| **Redistribution** | local only. The references come from the owner's speech front-end list (owner's work + LLM drafts); not published without the owner's approval |

## Tracks

| Track | Items / families | Source of the accepted spellings | Use |
|---|---|---|---|
| dev | 103 / 101 (86 primary-eligible) | owner-verified list words | checkpoint and arm selection only |
| **final → primary** | 165 / 162 | owner-verified list words, no ambiguity or translation flag | the registered primary measure, opened once |
| final → flagged | 26 | `pronunciation_ambiguous` (24) or `possible_translation` (2) | diagnostic |
| final_proxy | 400 / 375 | LLM-suggested list spellings, owner spot check only | exploratory; never pooled with primary |
| v1 English panels | pick_test 150, dev41 41, list_test 125 | benchmark v1 (frozen, `benchmarks/v1.json`) | historical, old scoring rule, never pooled |

"Owner-verified" covers the list's `owner_added` (149), `owner_confirmed` (78), `llm_suggested_owner_verified`
(53), `owner_chosen` (10) and `owner_corrected` (4) single words; split into dev and final by a hash of the lexical
family (salt `e02-english-phonetic`). Rows the list marks `attested`, `attested_web` or as LLM drafts were kept out
by policy (they mix conventional and translated forms). Multi-word entries (327) are out of scope.

## Item format (`khmer_translit.reference_data.ReferenceItem`)
`id`, `input`, `word` (normalised), `family`, `pron_id`, `intended_phones` (the registered default reading: the
first CMUdict reading, or none), `accent` (US), `accepted` (all accepted Khmer spellings, NFC), `preferred`,
`sources`, `tier` (independent_reviewed / attested / proxy), `split`, `flags`, `review`. Two readings of one spelling
would be two items in one family; none occur.

## Scoring (`khmer_translit.evaluation`)
- A prediction is correct when its NFC form equals the NFC form of **any** accepted spelling (multi-reference exact
  match). An empty or refused prediction counts as wrong.
- **Primary number:** macro over lexical families (each family's mean, then the unweighted mean), plus plain item
  accuracy. Families join words with a shared simple suffix stem, transitively (*connect / connected / connection*).
- Predictions are aligned by id; a missing, duplicate or unknown id is an error, not a silent skip.
- **Paired comparison:** per-family differences averaged over seeds, 10,000 family-bootstrap resamples, seed
  20261027, 95% percentile interval. The interval is conditional on the trained seeds.
- **Measurement controls** (run in tests and in the final report): every accepted form and its NFD form score as
  correct, a corrupted form scores as wrong, missing/unknown ids are refused.

## Separation from training (checked by the build, `reports/data_build.json`)
- No family in two splits; no training word related to a dev/final/proxy word (940 mined rows removed); no training
  target equal to a held-out accepted spelling (22 removed).
- The 150 owner pick examples are all benchmark-pool words, so none is used for training.
- P2G (Khmer phones → spelling) data drops the 284 Google-lexicon entries whose spelling is a held-out accepted
  form, so every final item is unseen as a P2G target too.

## Why it is unqualified
Qualification needed independent acceptance of each primary item **under the phonetic policy** (English sound
written in Khmer). None exists:
- The owner verified these spellings for the speech front-end, a related but different policy; the tier is
  `attested`, not `independent_reviewed`.
- 94 of the 191 final items are `candidate_anchored`: the owner picked them among historical model E's outputs, so
  the references lean towards E's conventions.
- Every final item has exactly one accepted spelling, while Khmer often allows several (vowel length, series
  marks), so exact match under-counts correct answers.
- The proxy track is LLM output with an owner spot check only.

The registered consequence (PLAN.md): the evidence status of E02 is `inconclusive` whatever the numbers, and no
treatment can be adopted on this benchmark.

## Known strata (final track)
23 dictionary misses (no CMUdict reading: the phone arms get `<no_pron>`), 24 ambiguous readings, 37 words of ≤ 4
letters, 29 of ≥ 9 letters. Power for a +5-point gain with 162 families: 0.29 at a discordance of 0.20, 0.50 at 0.10
(`reports/power.json`).

## What would qualify it
An independent reviewer (not the owner, not an LLM, not the model) who accepts or rejects each primary item's
spellings under the written phonetic policy and adds the alternatives they also accept; their decisions stored per
item in `review` with tier `independent_reviewed`. The scoring code needs no change.
