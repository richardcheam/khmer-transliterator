# Training data: sources and licences

**Licence of these files:** CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), because part of the
data is derived from Khmer Wikipedia (CC BY-SA). This covers the author's rights only; the sources below keep their
own terms. The trained model weights are also CC BY-SA 4.0 (see their model cards), and the code is MIT (`LICENSE`).

| Source | Used for | Licence and notice |
|---|---|---|
| Khmer Wikipedia, dump of 2026-10-01 (https://dumps.wikimedia.org/kmwiki/20261001/) | mined English–Khmer word pairs (both models) | CC BY-SA 4.0, © Khmer Wikipedia contributors (article histories at https://km.wikipedia.org) |
| FineWeb-2, subset `khm_Khmr` (https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | mined English–Khmer word pairs (both models) | ODC-By 1.0 (https://opendatacommons.org/licenses/by/1-0/); also subject to Common Crawl's terms of use (https://commoncrawl.org/terms-of-use). Contains information from FineWeb-2, made available under the ODC Attribution License. ODC-By covers the database, not the individual web pages, whose rights remain with their authors. |
| ALT Parallel Corpus (NICT; https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/) | names and loanwords spelled in its human Khmer translations (`byt5/train.tsv`, source `alt`) | CC BY 4.0; Riza et al. (2016), "Introduction of the Asian Language Treebank" |
| CMU Pronouncing Dictionary | filtering and checking pairs against the English pronunciation (not included) | BSD-style licence of the CMU Pronouncing Dictionary |
| Google `language-resources`, Khmer lexicon `km/data/lexicon.tsv` | data checks (not included) | CC BY 4.0, © 2018 Google Inc. |
| The author's checked spelling list | `small/dev.jsonl` | the author's |

If you represent a source and have concerns, please open an issue on this repository.
