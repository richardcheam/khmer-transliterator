# Benchmark card: `en-phonetic-v1` (E02 historical evaluation)

The benchmark spellings themselves are **not** included in this bundle: they are the owner's material and stay
local. This card describes them so the numbers in `MODEL_CARD.md` can be read correctly. The full card is in the
source repository at `experiments/E02-english-phonetic/BENCHMARK_CARD.md`.

| | |
|---|---|
| **Task** | One English word or name → its English pronunciation written in Khmer script (not a translation) |
| **Qualification** | **Unqualified.** No item has independent acceptance under the phonetic policy. Results are *reference agreement*, not phonetic accuracy |
| **Metric** | Exact match of the model's top answer (raw beam 4) against the accepted spelling, macro-averaged over word families. Also top-4 coverage and CER, as diagnostics |
| **Hashes** | All tracks `11d89a0a…f0b5`; dev `58541efd…2989`. Every input hash is in `data/e02/manifest.json` |

## Tracks
| Track | Items / families | Source of the accepted spellings | Use |
|---|---|---|---|
| dev | 103 / 101 (86 primary-eligible) | Owner-verified loanword-list words | Checkpoint and arm selection |
| final → primary | 165 / 162 | Owner-verified list words without ambiguity or translation flags | The registered measure, opened once (2026-10-27), not again since |
| final → flagged | 26 | `pronunciation_ambiguous` (24), `possible_translation` (2) | Diagnostic |
| final_proxy | 400 / 375 | LLM-suggested list spellings with an owner spot check | Exploratory; never pooled |

## Known limitations
- **One accepted spelling per final item,** so correct-sounding variants count as wrong. Exact match is a floor on
  usefulness.
- **The references follow the speech front-end's conventions**, reviewed by the owner for that purpose, not under
  the E02 phonetic policy.
  - 94 of the 191 final items were picked by the owner among an older model's candidates (`candidate_anchored`).
  - 53 owner-verified list words began as LLM suggestions.
- **Statistical power:** at 162 families, differences of a few points are within noise. Seed-to-seed spread of the
  letters model is ±0.8 points.
- **Leakage:**
  - no word family is in two splits;
  - no training word is related to a held-out word;
  - no training target equals a held-out accepted spelling.
- **Comparisons:** these numbers are not comparable with NEWS shared-task results or with the ByT5 model's
  mined-reference evaluation, which uses different references and items.
