# Licence decisions needed (owner): E11 ByT5 adapter

**Already settled:**
- **Code:** PolyForm Noncommercial 1.0.0 (D190). It applies to the wheel and sdist unchanged.
- **Base model:** Apache 2.0 (Google). It's not redistributed, and users download it at the pinned revision.

- **Adapter weights (owner, 2026-10-31, D198):** CC BY-NC-SA 4.0, the same terms as E02 (D191), chosen by the
  owner for this adapter. Licence text: `LICENSE-weights.md`, shipped as `huggingface/LICENSE`. The owner accepts the
  same share-alike risk as for E02 (row (a) below).

## Decision 1: licence for `adapter.safetensors` (decided: option (a))
The training data constrains the choice (`DATA_PROVENANCE.md`, `NOTICES.md`):

| Option | Fits research/non-commercial intent | Wikipedia share-alike, if it reaches the weights | Hub metadata |
|---|---|---|---|
| (a) CC BY-NC-SA 4.0 (same as E02) | yes | **not** BY-SA-compatible; accepted risk, as for E02 | `license: cc-by-nc-sa-4.0` |
| (b) CC BY-SA 4.0 | allows commercial use | satisfies it | `license: cc-by-sa-4.0` |
| (c) CC BY-NC 4.0 or another NC licence | yes | not compatible; same risk as (a) without share-alike | `license: cc-by-nc-4.0` |
| (d) do not publish yet / retrain without Wikipedia-derived rows | — | removes the direct rows, though FineWeb pages may still carry SA terms | — |

- **Attribution:** whichever is chosen, the attribution text in `DATA_PROVENANCE.md` must ship with the weights.
- **Licence limits:** no licence can grant more rights in the underlying web pages than the owner holds.

## Decision 2: accept the recorded data questions for this adapter
These are the same kinds of question as for E02, but each needs a decision for E11:
- page rights under the Common Crawl terms;
- the Wikipedia share-alike question;
- the indirect Chhunneng influence on mining (`NOTICES.md` §3).

## Decision 3: publication itself
- **Repository name and visibility:** the proposed repository is `ric912/en-km-phonetic-byt5-lora`, private first
  as for E02 (D192).
- **Wheel:** whether to put the PolyForm-licensed wheel in that repository, or only point to it.

After the owner decides, the steps are in `PUBLICATION_CHECKLIST.md`.
