# E05 — Candidate diversity: G+P vs G8 and G+G (inference only)

**Status:** registered 2026-10-27, before any new prediction and before any new comparison result was seen. The
registration commit is the first commit containing this file; hashes are in `registration.json`.

**Label:** exploratory on historical data. The E03/E04 dev and eval partitions have already been examined by
earlier work, so this is not fresh, independent validation.

**Scope:**
- inference only;
- no training, architecture change, reranker, paid call, download, TTS/audio, product-default or serving change;
- CPU, one inference job at a time, ≤ 6 threads, ≤ 2 aggregate elapsed computation hours;
- E01–E04 artifacts and the sealed reserve are preserved;
- E02's final references are not read and no reserve predictions are made (`scripts/e05.py guard`, tested).

## Question
Does pooling a letters model's candidates with a pronunciation-only model's candidates (G+P) add reference coverage
beyond simply generating more candidates from one letters model (G8), or pooling two letters models (G+G)?

E04 compared G+P (up to 8 slots) only with G4 (4 slots). That is not an equal-budget comparison (E04 report,
clarification of 2026-10-27).

## Inputs (frozen; hashes in `registration.json`)
- **Models:** E04's `G_s0`, `G_s1`, `G_s2`, `P_s0` (`models/e04/*/model.npz`, format `e04-research-v1`).
- **Items:**
  - E03/E04 dev: 276 items, 253 primary, 250 families;
  - E03/E04 eval: 245 items, 228 primary, 222 families;
  - multi-reference items are scored against all accepted spellings;
  - `pronunciation_ambiguous` items are reported separately as the flagged track.
- **Source encoding:** each model's main E04 view (`scripts/e04.py encode`). P sees CMUdict's first listed reading
  with stress, as in E04.
- **Saved E04 beam-4 outputs:**
  - eval, all four models: `results/eval_per_item.jsonl`;
  - dev primary, G_s0 and P_s0: `reports/pilot_dev_per_item.jsonl`.

## Decoder (identical for every constituent)
- The NumPy beam search used in E04.
- Score: the sum of token log-probabilities, EOS included, with no length normalisation.
- At most 40 output tokens, unchanged source processing.
- **Primary matched configuration:** beam width 8 for every model; top-4 or top-8 are then taken from that list.
- **New predictions:** beam-8 for the four models on dev and eval (all items). Beam-4 only for dev G_s1/G_s2
  (primary items), which E04 did not save.
- **Beam-4 reproduction:** saved E04 beam-4 outputs plus those two files; dev beam-4 is primary-only. Beam-4 and
  beam-8 candidates are never mixed within a configuration.
- **Search-width effect:** reported separately (G4 at width 4 vs top-4 of width 8). It is not attributed to
  pronunciation.

## Candidate rules
- **Normalisation:** NFC; empty strings dropped; identical strings deduplicated, keeping the first (best) rank.
- **No filling:** an undersized union is reported as such, never filled with extra candidates.
- **Failures:** an empty or refused output counts as not covered.
- **Ids:** prediction ids must equal reference ids; missing or unknown ids are an error.

## Systems
- **G4[g]:** top-4 of G model g. **G8[g]:** top-8 of the same model.
- **G+G[g,h]:** union of top-4 of two different G seeds (all three unordered pairs).
- **G+P[g]:** union of G model g's top-4 and P_s0's top-4, for g = G_s0, G_s1, G_s2. P_s0 is reused, so these are
  not three independent G+P replicas.
- **Budget curve (secondary)** at k = 1, 2, 4, 8, with a reference-independent order:
  - single models use their own rank;
  - two-model systems alternate ranked candidates (first model first), skipping duplicates;
  - both model-priority orders are reported, and no order is selected;
  - interleaving is a diagnostic ordering, not a trained reranker.

## Metrics
- **Coverage (oracle, reference-dependent):** an item is covered if any candidate equals an accepted reference
  (NFC). Reported as item coverage and as macro coverage over lexical families (primary metric).
- **Candidate counts:** distinct candidates per word (mean/min/max), duplicate rate (1 − distinct/slots),
  under-filled and empty counts.
- **Pool:** reference present vs absent in the pooled candidates of all four models.
- **Overlap:** shared top-4 candidates between constituents (mean count, Jaccard, equal top-1).
- **Item sets:** items covered by G+P[g] but by neither control (G8[g] and the G+G pairs containing g), and the
  reverse.
- **Strata:**
  - phone-seen vs unseen;
  - the flagged track separately;
  - a dictionary-hit stratum. Every primary reference is a lexicon headword by DACS construction, so this
    stratum is degenerate. It is replaced by a reference-free flag: whether G_s0's top-1 is a lexicon word.
- **Cost:** see below.

## Comparisons and uncertainty (primary: eval primary items, beam-8, macro coverage)
- **Per combination g:** G+P[g] − G4[g], G+P[g] − G8[g], and G+P[g] − G+G[g,h] for the two pairs containing g.
- **Summary over the fixed combinations:**
  - mean G+P − mean G8, mean G+P − mean G+G (three pairs), mean G+P − mean G4;
  - for scale: G8 − G4 and G+G − G4;
  - at the equal 4-slot budget: interleaved G+P@4 (both orders) − G4, and − interleaved G+G@4.
- **Bootstrap:**
  - paired family bootstrap, 10,000 resamples, seed 20261027 (NumPy `default_rng`);
  - every system is resampled with the same family draws;
  - 95% percentile interval.
- **What the interval means:**
  - it is conditional on these trained checkpoints;
  - comparisons share checkpoints, and every G+P shares P_s0;
  - it does not estimate the training-population uncertainty of a newly trained ensemble.
- **Dev:** the same numbers are reported descriptively. Nothing is tuned on dev or eval: there is no combination
  setting to tune.

## Decision rule (registered)
- **Consistent advantage:** the summary lower bounds of G+P − G8 **and** G+P − G+G are both > 0, **and** every
  per-combination estimate against G8 and against each G+G is > 0.
  - Conclusion: pronunciation-specific diversity is plausible on these historical references, conditional on
    P_s0.
  - Recommendation: a separately registered ranking study, contingent on better evaluation evidence. Nothing is
    trained here.
- **Otherwise:** no clear advantage. The pooling result does not establish special value from pronunciation.
  - Recommendation: keep the simpler candidate generator; no pronunciation-specific reranker, and no more
    phone/fusion training.
- **Uncertainty flag:** set if per-combination signs are mixed or any summary interval is wider than 10 points. No
  favourable combination is singled out.
- **The report always separates three things:**
  - candidate-generation opportunity (what coverage measures);
  - the ability to rank without references (not measured; the interleaved k = 1 / 2 rows only describe a fixed
    order);
  - actual phonetic quality (not measured).

## Practical cost (`reports/cost.json`)
- **Sample:** 100 primary dev items (sha256 order).
- **Process:** each configuration runs in a fresh process, one at a time, 6 threads, batch 1. Measured: cold load,
  first call, warm ms per word (3 repetitions), peak RSS (including imports), distinct candidates, and whether
  the models must stay resident.
- **Configurations:** G4 (beam-4), G8 (beam-8), G+P (beam-4 each), G+P (beam-8 each, the matched protocol), G+G
  (beam-4 each).
- **Machine information** is recorded.

## Blinded review queue (prepared only; `data/e05/review/`, git-ignored, local)
- **Sample:** eval primary items, sampled reproducibly (sha256 order, seed `e05-review`), 10 per pattern of the
  G_s0 systems: G+P only, controls only, all cover, none cover, mixed, plus 10 where G and P have the same top-1.
  This covers agreements and disagreements, not only G+P wins.
- **Candidates per word:** the deduplicated union of G_s0 top-8, G_s1 and G_s2 top-4, P_s0 top-4 and the mined
  reference(s), in random order with the origin hidden.
- **Fields:** acceptable / unacceptable / uncertain (several may be acceptable) and a comment.
- **Origin key:** a separate file.
- **Not done:** no judgments are filled in, nothing is published, no one is contacted.

## Commands
See `REPORT.md` after the run; the steps are `scripts/e05.py register | predict | analyse | cost | queue`, or
`run` for all steps after registration.
