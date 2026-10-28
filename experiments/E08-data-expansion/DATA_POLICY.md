# E08 data policy (frozen 2026-10-28, before any new example was accepted)

## What a training example is
- **Input:** one English word or name, with a specified English pronunciation:
  - the CMUdict first reading (US), kept as provenance (`phones` / `pronunciation_arpabet`);
  - the ByT5 model input stays the lower-cased English letters only.
- **Output:** readable Khmer script that approximates that pronunciation.

## Eligibility
1. **No translations.** A translation is not eligible merely because it conveys the meaning (*class* → កណ្តាល
   "middle" is not eligible).
2. **Loan spellings.** An established Khmer loan spelling is eligible when it is compatible with the intended
   English reading.
3. **Alternatives.** Several spellings can be legitimate alternatives for one reading. All are kept, and exact
   duplicate pairs are removed.
4. **No silent reading switch.** Foreign names must not switch from the specified English reading to their
   original-language pronunciation. Pairs that fit a non-English reading better are quarantined as **uncertain**,
   not declared wrong.
5. **Quarantine.** Uncertain examples are quarantined without being declared incorrect. Quarantine is a manifest
   of its own, never silently dropped or silently used.
6. **Ambiguous English words.** The intended reading is fixed before generation (CMUdict first reading). Words
   whose readings differ beyond stress are quarantined from the synthetic manifest, so no conflicting targets exist
   for one input.
7. **Romanised-Khmer inputs stay excluded** (`models/model/romanised_forms.txt`).
8. **Protected families stay excluded**, matched by stems, for both training additions and synthetic inputs:
   - E03 dev/eval;
   - E02 held-out sets, the benchmark panels and the reserve (Latin inputs only);
   - the E06 fresh pool and E06 discovery words;
   - the E05/E06 review queues.

## Provenance (never pooled without a label)
| Provenance | Origin | Acceptance authority |
|---|---|---|
| `control` | `data/e03/train.tsv` unchanged (2,255 optimisation + 93 train-val rows) | E03's registered build |
| `recovered` | E03-excluded mined pairs | A documented screen-weakness rule (`scripts/e08.py explain`/`recovery_decision`), never a model's preference |
| `attested` | Newly audited public sources | Only where reuse permission is explicit; the same frozen screen |
| `google_ai_mode_synthetic` | Google AI Mode responses to the frozen prompts | Structural import only. This is **not** linguistic validation |

## What validation means here
- **Not validation:** valid Khmer Unicode, repeated generations, model agreement and self-reported confidence.
- **Agent inspection:** linguistic inspection by the agent is labelled agent inspection, not independent review.
- **Synthetic labels:** they are training data only. They are never benchmark references, and never promoted to
  reviewed status.
