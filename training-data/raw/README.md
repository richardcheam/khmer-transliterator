# Raw inputs of the training data

`pipeline/prepare.py` turns these files into the training files of both published models (`../small/`,
`../byt5/`), byte for byte (`--check` verifies it). It also downloads, pinned by commit and sha256, the CMU
Pronouncing Dictionary (to check that a word has an English reading) and Google's Khmer lexicon (for the character
and pronunciation lists). Licences and notices: `../NOTICE.md`.

| File | Rows | What it is |
|---|---|---|
| `mined_kmwiki.tsv`, `mined_fineweb.tsv` | 3,521 and 63,627 | every Khmer–English bracket pair mined from Khmer Wikipedia (dump 2026-10-01) and FineWeb-2 (`khm_Khmr`, shard 000_00000), e.g. "ចនសុន (Johnson)": `latin`, `khmer`, `class` (`foreign` or `khmer_like`, from an older in-house classifier: is the Latin a romanised Khmer word?), `count` (times seen), `similarity` (to a rule spelling of the Latin), `source_phrase` |
| `author_examples.tsv` | 149 | the author's example spellings (all removed by the held-out rules in the end) |
| `historical_panels.tsv` | 498 | English words of earlier test panels (`pick_test`, `dev41`, `list_test`) and of an earlier validation list, kept out of training |
| `byt5_pairs.tsv` | 4,857 (4,764 `train`, 93 `train_val`) | the ByT5 adapter's pairs, already screened: `control` (2,348) and `recovered` (1,292) web-mined pairs, `attested_alt` (1,217) names and loanwords from the ALT Parallel Corpus; `phones` is the CMU reading |

**Boundaries.** Mining itself is not part of the pipeline: the miner used an older in-house model, which is not
published, so the mined pairs are the starting point. The ByT5 pairs went through earlier automatic screens (a
consonant check against the English pronunciation, documented recovery rules for wrongly rejected pairs, and the
ALT extraction); `prepare.py` assembles and checks them but does not re-run those screens.

**What is not here.** The released data also removed rows related to a sealed reserve of English test words, kept
unpublished so it can still serve as a test: the rows of 81 Latin words (11 Wikipedia, 178 FineWeb-2 rows) and one
example are absent from these files, so `prepare.py` reproduces the release without knowing the reserve. This keeps
the reserve's words unpublished; it cannot fully hide them, since comparing these files with their public sources can
hint at which rows were removed.
