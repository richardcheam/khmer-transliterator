# Notices and publication status (khmer-translit-byt5 0.2.0)

**Status: local research preview. Nothing here may be published without the owner's explicit decision.** This file
records the stated terms of each asset. It is not a legal opinion, and nothing here concludes that publication is
permitted.

| Asset | In the bundle | Stated terms | Publication status |
|---|---|---|---|
| Package code (`khmer_translit`, wheel + sdist) | yes | **PolyForm Noncommercial 1.0.0** (owner, 2026-10-29; `LICENSE`) | Code licence settled; publication still blocked by the adapter questions |
| Adapter weights (`adapter/adapter.safetensors`) | yes | Derived from the data in `DATA_PROVENANCE.md` | **Blocked** (see below) |
| Base model `google/byt5-small` | no (pinned by revision; fetched by the user) | Apache 2.0 (Hub metadata) | Not redistributed; users obtain it from the Hub |
| Documents, results and evaluation summaries | yes | Project-authored | Needs the owner's decision (part of the repository) |
| Confirmation input manifest (120 English words) | yes | Words selected from SCOWL/CMUdict lists | Low risk; owner's decision |
| Training pairs, mined references, review batch and key | **no** | Mixed (see provenance) | Kept local |

## Open questions that block publishing the adapter
1. **Google AI Mode output** (384 training rows, 193 words): Google's Generative AI Additional Terms say "You may
   not use the Services to develop machine learning models or related technology."
   - Every E08/E10 adapter was trained with these rows.
   - Whether that clause covers them, and therefore the weights, needs the owner's (or counsel's) decision.
   - **Fallback:** an adapter trained without them, the frozen arm R (`arm_R.tsv`, sha256 `d00706cf…`), which
     would be a new training run.
2. **Wikipedia (CC BY-SA 4.0):** whether trained weights are adaptations subject to share-alike is unresolved, as
   already recorded in 0.1.0's notices.
3. **FineWeb-2 (ODC-By 1.0) and the Common Crawl terms of use:** attribution is required; the terms of the
   underlying web pages also apply to the mined pairs.
4. **ALT (CC BY 4.0):** attribution is required (text in `DATA_PROVENANCE.md`).
5. **Code licence:** PolyForm Noncommercial 1.0.0, chosen 2026-10-29. The adapter weights have no licence yet.

## Wording rules for any public text
- This is a **research preview with unverified phonetic quality**.
- Historical exact-match numbers are agreement with mined, unreviewed references. They are not phonetic accuracy.
- The package is not production-validated.
