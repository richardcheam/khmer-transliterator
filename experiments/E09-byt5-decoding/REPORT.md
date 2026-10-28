# E09 — ByT5 decoding: report (2026-10-28; inference only)

**In one sentence:** greedy beat beam-4 because the registered beam call stopped searching too early (a
configuration defect, `early_stopping=True`). With canonical beam search, beam is about as good as greedy or
slightly better, so decoding gives no large gain beyond greedy.

Plan `PLAN.md` (committed `1bb9f4d` before any generation). Selection frozen in `reports/selected_settings.json`
(committed `a81b29b` before eval).

## 1. Did the greedy advantage reproduce?
**Yes.**
- Registered greedy and beam-4 regenerated on all 276 dev items match the saved E07/E08 files exactly: texts,
  statuses and scores (`reports/reproduction.json`).
- On dev primary (253 items, 250 families), greedy beat registered beam-4:
  - E07: by 1.2 points [−4.4, +2.0];
  - E08: by 4.6 points [−9.0, −0.2].
- On the historical eval, the margins were +2.7 (E07) and +5.0 (E08) points (table in §5).

## 2. Defect or configuration issue?
**A configuration defect; no implementation defect.** Details are in `reports/generation_audit.md`.
- **The flag:** the registered beam call used `early_stopping=True`, so Transformers ends the search once
  `num_beams` hypotheses have finished.
- **Why it bites with bytes:** junk hypotheses finish within the first steps, and they fill the finished set
  before longer, better spellings end.
  - Junk means EOS at once (seen 111 times in E08's dev beams), or one letter + EOS.
- **How it shows:**
  - beam top-1 had a lower model log-prob than greedy's output on 52/71 (E08) and 24/45 (E07) of the differing dev
    items;
  - beam top-1 was byte-shorter on 54/63 and 28/39 of the dev primary items where it differed;
  - it often drops ៉ / ៊ or writes ក for ហ្គ.
- **Everything else checked out:** adapter loading, eval mode, inputs, special ids, the byte budget, UTF-8, NFC,
  score reconstruction (≤ 3e-5) and alignment.
- **Order of work:** a regression test failed first. The fix is then an `early_stopping` / `length_penalty`
  argument on `e07.gen`, with the registered defaults kept. Original results stay as saved.
- **Not a length-penalty problem:** after the fix, canonical beam-4 top-1 was shorter than greedy on only 39%
  (E07) and 47% (E08) of differing items, against the 60% trigger. So the one allowed length-penalty follow-up was
  **not run**.

## 3. Development comparison (dev primary; `reports/dev_comparison.json`)
| Setting | E07 top-1 | E07 CER | E08 top-1 | E08 CER | Seconds E07 / E08 (276 words) |
|---|---|---|---|---|---|
| greedy | 52.2 | .186 | 54.2 | .151 | 20 / 20 |
| beam2 registered | 50.2 | .190 | 48.2 | .171 | 32 / 32 |
| beam4 registered | 51.0 | .181 | 49.6 | .159 | 41 / 40 |
| beam8 registered | 52.2 | .175 | 50.4 | .157 | 67 / 65 |
| **beam2 canonical** | **52.6** | **.182** ← E07 selected | 55.4 | .151 | 32 / 33 |
| beam4 canonical | 52.6 | .183 | 55.4 | .148 | 44 / 45 |
| **beam8 canonical** | 52.6 | .183 | **55.4** | **.147** ← E08 selected | 74 / 73 |

- **Validity:**
  - every top-1 is non-empty and decodes; `no_eos` 0 in every setting;
  - one E07 greedy top-1 fails the orthography check and one is repeated. E07 canonical has 1 repeated, and E08 has
    0 in every setting.
- **Output length:** mean top-1 length is 5.1 code points for greedy and canonical, and 4.8–5.0 for registered
  beams.
- **Candidates per item:** canonical keeps k distinct candidates per item. Registered beam-4 kept 3.99 (E07) and
  3.60 (E08).
  - Top-4 oracle (diagnostic, not quality): E08 canonical 82.8% vs registered 68.0%.
- **Paired contrasts (dev):**
  - canonical − registered beam-4: E07 +1.6 [−1.2, +4.4], E08 +5.8 [+1.8, +9.8];
  - canonical − greedy (each width): E07 +0.4 [−1.2, +2.0], E08 +1.2 [−0.4, +3.2].
- **Selection rule** (top-1, then CER, then seconds):
  - E07: the three canonical widths tie on top-1, so the lowest CER wins → **beam2 canonical**;
  - E08: the same → **beam8 canonical**.
  - Both differences from greedy are 1–3 items.

## 4. Disagreements (`reports/disagreements.json`; 40 dev items, 5 per group per model; agent inspection, non-independent)
- **Greedy right, registered beam wrong:** always the configuration defect.
  - Examples: *mani* ម៉ានី → មានី; *mora* ម៉ូរ៉ា → មូរ៉ា; *bee* ប៊ី → បៃ; *gavin* ហ្គាវីន → កាវីន.
  - Greedy's output was much more probable (e.g. −0.02 vs −4.69); canonical beam returns greedy's spelling.
- **Registered beam right, greedy wrong:** mostly lucky lower-probability outputs that matched the mined reference.
  - Examples: *walmart* វ៉លម៉ាត −2.61 vs greedy −0.76; *falcon*; *garay* ការ៉េ vs ហ្គារ៉េ.
  - Canonical beam keeps greedy's spelling there. A few are close calls between vowel signs (*lauren*, *hulme*).
- **Both wrong:** mostly vowel or final-consonant variants (*booth*, *bagdad*, *sacramento*, *clean*).
  - Several references look doubtful as English readings: *esther* → ស្តើរ (log-prob −16.9), *saw* → ស, *roi* →
    រ៉យ.
  - References were not changed, and no alternatives were added.

## 5. Historical eval (exploratory; `results.json`, `results_per_item.jsonl`)
This set (228 primary items, 222 families) has been **examined repeatedly** (E03–E09), and its references are
mined and unreviewed. Intervals are paired family-bootstrap (10,000), **conditional on these two checkpoints**:
they do not measure seed variance and exclude the extra uncertainty of dev selection. The registered beam-4 and
greedy predictions are the saved files; the selected settings were generated once and frozen
(`reports/eval_frozen.json`).

| System | Top-1 | Mean CER | Top-4 oracle |
|---|---|---|---|
| E07 beam-4 registered | 47.2 | .189 | 64.7 |
| E07 greedy | 49.9 | .178 | — |
| E07 beam-2 canonical (selected) | 49.4 | .176 | 61.1 (top-2) |
| E08 beam-4 registered | 47.5 | .191 | 65.2 |
| E08 greedy | 52.4 | .161 | — |
| E08 beam-8 canonical (selected) | **53.3** | .163 | 79.6 |

| Question | Contrast | Top-1 Δ [95%] | CER reduction Δ [95%] | Items +/− |
|---|---|---|---|---|
| Does decoding improve E07? | selected − registered beam-4 | +2.3 [−0.9, +5.4] | +1.2 [+0.1, +2.5] | 9 / 4 |
| | greedy − registered beam-4 | +2.7 [−0.9, +6.3] | +1.1 [−0.5, +2.6] | 11 / 5 |
| | selected − greedy | −0.5 [−2.3, +0.9] | +0.2 [−0.5, +1.0] | 1 / 2 |
| Does decoding improve E08? | selected − registered beam-4 | **+5.9 [+1.4, +10.4]** | +2.6 [+1.1, +4.1] | 20 / 6 |
| | greedy − registered beam-4 | +5.0 [0.0, +9.9] | +2.8 [+1.0, +4.6] | 22 / 10 |
| | selected − greedy | +0.9 [−1.4, +3.6] | −0.2 [−1.6, +1.1] | 5 / 3 |
| Does E08 beat E07, both greedy? | E08 − E07 | +2.6 [−2.4, +7.5] | +1.8 [−0.6, +4.4] | 19 / 14 |
| Does E08 beat E07, both selected? | E08 − E07 | +3.9 [−1.1, +9.0] | +1.4 [−0.9, +3.9] | 21 / 13 |
| (reference) both registered beam-4 | E08 − E07 | +0.3 [−5.7, +6.5] | +0.1 [−2.5, +2.8] | 24 / 25 |

## 6. What this means
- **A practical improvement was found, and it is a correction rather than a new trick.** Use greedy, or canonical
  beam search (`early_stopping="never"`), instead of the registered beam call.
  - Against the registered beam-4 that E07/E08 reported as primary, this gains about +2 (E07) and +5–6 (E08) top-1
    points.
  - Beyond greedy, canonical beam adds little: dev +0.4 / +1.2, eval −0.5 / +0.9, all within noise. Greedy is
    2–4× cheaper.
  - Canonical beam's real extra is a clean candidate list (E08 top-4 oracle 82.8% on dev), useful if several
    candidates are needed.
- **E08 vs E07:** with matched, working decoding, E08 is ahead by +2.6 (greedy) and +3.9 (selected) points, but
  both intervals include 0.
  - The earlier "+0.3, no change" (D186) was depressed by the defect, which hurt E08 more: its beams held more
    junk.
  - E08 reached +3.9 with dev-selected settings, but on eval this fails E08's pre-registered "promising" rule: the
    interval lower bound is below 0. That rule was registered for beam-4, so it is not re-applied here; the
    comparison is reported, not re-classified.
- **No claim about the data:** a decoding gain is not evidence that the expanded dataset helped.
- **No claim about phonetics:** higher agreement with mined references is not shown to be better phonetic
  acceptability.

## 7. Unresolved
- **Phonetic quality is unmeasured.** No independent review exists, and the references contain doubtful entries
  (§4). Whether ៉ / ៊ spellings or vowel choices are better for an English reading is not established by reference
  agreement.
- **One checkpoint per model:** no seed variance.
- **Repeated looks:** dev was also the checkpoint-selection set, and eval has been examined repeatedly.
- **E08's pre-registered verdict:** it was made with defective beam decoding. A matched re-check would need a
  replication seed, which is not launched here.

## Compute and safety
- **Active time:** 0.47 h by ledger (a conservative sum: step totals plus the generation inside them).
  - Wall time 18:08–18:30 UTC.
  - Peak RSS 2.1 GB, one model at a time, 6 threads, CPU only.
- **Not done:** no training, downloads or paid services; E02 final references and the reserve not opened.
- **Unchanged:** saved E07/E08 predictions, checkpoints, serving and the product default.

## Reproduce
```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e09.py register     # hashes → registration.json
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e09.py reproduce    # vs saved E07/E08 dev predictions
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e09.py audit
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e09.py dev          # 7 settings × 2 models (cached, resumable)
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e09.py diagnose
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e09.py select       # commit reports/selected_settings.json next
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e09.py eval         # refuses an uncommitted selection
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e09.py analyse
.venv/bin/python -m unittest tests.test_e09
```
Caches live in `state/e09/cache/` (local; keyed by checkpoint, setting and data hashes). Checkpoints are local in
`models/e07/full/` and `models/e08/rs_s0/`.
