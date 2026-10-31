# Notices and publication status (khmer-translit-byt5-e11 0.2.0)

**Status: local release candidate. Publication is pending the owner's decisions** (`LICENSING_DECISION.md`). This
file records the stated terms of each asset as checked on 2026-10-31. It is not a legal opinion, and nothing here
concludes that publication is permitted.

| Asset | In the bundle | Stated terms | Status |
|---|---|---|---|
| Package code (`khmer_translit`, wheel + sdist) | yes | **PolyForm Noncommercial 1.0.0** (owner, D190; `LICENSE-code`) | Settled |
| Adapter weights (`adapter.safetensors`) | yes | Derived from the data in `DATA_PROVENANCE.md` | **Licence undecided** |
| Base model `google/byt5-small` | **no** (pinned by revision, fetched by the user) | Apache 2.0 (Hub metadata) | Not redistributed |
| Documents and aggregate evidence | yes | Project-authored | Owner's decision with the release |
| Training pairs, mined references, review batches, keys, judgments | **no** | Mixed | Kept local |

## Questions that still apply after removing the AI Mode rows
Leaving out the Google AI Mode labels removed the one restrictive terms clause on E08/E10's data
("You may not use the Services to develop machine learning models…"). It doesn't settle these:

1. **Khmer Wikipedia, CC BY-SA 4.0.**
   - Some of the control and recovered pairs come from Wikipedia text.
   - Whether trained weights are "adapted material" that would have to carry a share-alike licence is unresolved.
   - For E02's weights the owner accepted this risk with CC BY-NC-SA 4.0 (D191). That decision covers E02 only.
2. **FineWeb-2: ODC-By 1.0 plus the Common Crawl Terms of Use** (updated 2024-03-07).
   - Attribution is required.
   - Common Crawl's terms say crawled content "may be subject to separate terms of use or terms of service from
     the owners", and ask users to respect third parties' copyrights.
   - So the rights in the underlying web pages are not granted by either licence.
3. **Indirect Chhunneng influence during mining** (E02 `PROVENANCE_AUDIT.md` §3).
   - The mining step that produced the control and recovered pairs used the then-served transliterator, which was
     trained on Chhunneng pairs, to choose cut points and to keep or drop pairs.
   - It contributed no label text. Which model file was loaded isn't recorded.
   - The owner published E02 with the same influence, but hasn't recorded a decision for this adapter.
4. **ALT Parallel Corpus: CC BY 4.0.** Attribution and the citation are required. The CC BY-NC-SA Khmer
   annotation set was not used.
5. **Evaluation word lists.** The test manifests (English words drawn from the project's CMUdict-based word pools) are not in the bundle,
   only aggregate scores. Publishing them is optional.

## Wording rules for any public text
- Say: one non-independent judge, two different 120-family samples (64.2% and 75.0% confirmed acceptable), not
  pooled.
- Never say: "independently validated", "production-ready", "state of the art" or "better than all previous
  models".
- Mined-reference numbers are spelling agreement, not phonetic accuracy.
- The adapter needs `google/byt5-small` at the pinned revision and `khmer-translit[byt5]`. It isn't a PEFT or
  AutoModel checkpoint.
