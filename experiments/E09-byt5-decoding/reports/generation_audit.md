# E09 generation audit (2026-10-28)

**Data:** `generation_audit.json`, `reproduction.json`, `disagreements.json`.
**Scope:** both checkpoints, E07 `models/e07/full/best.pt` (update 1,200) and E08 `models/e08/rs_s0/best.pt`
(update 3,000).

## Verdict
- **No implementation defect** was found in our generation or scoring code.
- **A confirmed configuration defect** explains the greedy advantage: the registered beam call used
  `early_stopping=True`.
  - With byte-level targets, short "junk" hypotheses finish in the first steps. These are EOS at once, or one Khmer
    letter (3 bytes) plus EOS. Once `num_beams` hypotheses have finished, Transformers stops.
  - Better hypotheses that need more bytes are still running at that point and are discarded.
- **Effect:** beam top-1 was often **less probable** under the model than greedy's output, and byte-shorter. It
  drops ៉ / ៊ and writes ក for ហ្គ.

## Checks
| Check | Result |
|---|---|
| Reproduction: registered greedy and beam-4, all 276 dev items, both models | **Exact**: 0 text/status mismatches; beam scores identical (max diff 0.0) |
| Adapter | strict key check (`e07.load_adapter`); checkpoint sha256 = `done.json` (E07 `24c9d200…`, E08 `d5da8c29…`) |
| Eval mode / dropout | `model.training False`; the same outputs twice |
| Input encoding | the same `encode_src` and padding/mask code for every setting (`e07.gen`) |
| Sampling / processors | `do_sample=False` is passed explicitly; the checkpoint's generation config sets none of temperature, top-k/p, repetition penalty, no-repeat n-gram, bad words, forced EOS, min length or suppression (all null); no `logits_processor` |
| Special ids | decoder start 0, pad 0, EOS 1 (base `generation_config.json` and `T5Config`) |
| Byte budget | longest dev/eval reference = 46 ids incl. EOS, against `max_new_tokens` 121; `no_eos` = 0 in every setting |
| UTF-8 | malformed: greedy 0; beam-4 3 lower-ranked E07 candidates; top-1 never malformed |
| NFC | references stored NFC; outputs NFC in `decode_ids`; `evaluation.item_outcomes` normalises both sides |
| Score reconstruction | teacher-forced sum log-prob incl. EOS vs generation score: max \|Δ\| 1.6e-5 (greedy), 2.8e-5 (beam, `length_penalty` 0 → raw sum) |
| Item alignment | dev and eval ids equal the saved files; `evaluation.item_outcomes` refuses missing or extra ids |
| Top-1 extraction | first non-empty, well-decoded candidate (E07/E08 convention); no raw rank-1 was empty |

## The mechanism (H1, stated in PLAN.md before any generation)
**Search error.** On the items where registered beam-4 top-1 ≠ greedy (all 276 dev items), the model's own
teacher-forced log-prob was higher for greedy's output in:
- **E07:** 24 of 45;
- **E08:** 52 of 71.

Canonical search, with only this flag changed, ranks by the same quantity, so these are hypotheses the search
failed to keep.

**Junk fills the beam.** Over 253 dev primary items, beam-4 candidates of ≤ 4 ids (EOS alone, or one letter + EOS),
registered → canonical:
- **E07:** 17 → 1;
- **E08:** 177 → 3, including 111 empty candidates.

E08's lists hold more of this junk, which is why E08 lost more from beam search (dev −4.6 points vs greedy) than
E07 did (−1.2).

**Byte-shorter outputs.** Where registered beam-4 top-1 differs from greedy (253 dev primary items):

| | E07 | E08 |
|---|---|---|
| Items that differ | 39 | 63 |
| Shorter in bytes | 28 | 54 |
| Of which: greedy with ៉ / ៊ removed | 7 | 12 |
| Same bytes | 9 | 7 |
| Longer | 2 | 2 |

**Minimal reproduction:** the test `tests/test_e09.py::test_registered_early_stopping_ends_beam_before_the_best_hypothesis`.
- **Set-up:** a scripted model where "aaaaaa" + EOS has log-prob −0.63, but EOS, "b" and "c" rank 2nd–4th at the
  first step.
- **`early_stopping=True`** returns `''` (−3.00), `a`, `aa`, `b`.
- **`"never"`** returns `aaaaaa` first.
- **Order:** the test failed before the fix.

**Fix:** `e07.gen` gains `early_stopping` and `length_penalty` arguments. The defaults are the registered call, so
saved predictions still reproduce (`test_registered_call_unchanged`). E09 passes `"never"`. No saved file was
overwritten.

## Classification of the remaining differences (agent inspection of `disagreements.json`, non-independent)
1. **Implementation defect:** none.
2. **Configuration effect:** confirmed (above). It accounts for most greedy-vs-beam disagreements.
   - Typical registered-beam losses: *mani* ម៉ានី → មានី, *mora* ម៉ូរ៉ា → មូរ៉ា, *bee* ប៊ី → បៃ,
     *gavin* ហ្គាវីន → កាវីន.
   - In each, greedy's output had a far higher log-prob (e.g. −0.02 vs −4.69).
3. **Model preference / ranking:** after the fix, canonical beam-4 differs from greedy on only 18 (E07) and 19 (E08)
   dev items.
   - There it finds a somewhat more probable spelling: *saint* សាន (−0.93 vs −1.00), *bagdad* បាក់ដាត,
     *sacramento* សាក្រាមិនតូ (−1.03 vs −2.31). These are as often right as wrong against the reference.
   - Some of the registered beam's "wins" over greedy were lower-probability outputs that happened to match the
     mined reference (*walmart* វ៉លម៉ាត, *falcon* ហ្វាលខុន, *garay* ការ៉េ). With the fix, these items return
     greedy's more probable spelling.
4. **Reference-spelling mismatch:** some references look doubtful as English readings.
   - Examples: *esther* → ស្តើរ, log-prob −16.9; *saw* → ស; *roi* → រ៉យ.
   - In several both-wrong items, the outputs differ from the reference only by a vowel or a final consonant
     (*booth*, *bagdad*, *sacramento*).
   - The references were not changed, and no alternatives were added.
