# Evidence ledger

Where every important number and claim in `main.tex` comes from. Each empirical value is stored in
`data/results_extract.json` with a `kind`:
- `json`: read by `tools/extract.py` with `git show <revision>:<file>`;
- `derived`: computed from `json` values, with the rule recorded;
- `transcribed`: copied from a committed report's text or table, with file and section (§2 lists these).

The key column below is the `\val{...}` key used in the report. Architecture and recipe settings, sample-design
constants (120 families, 40/40/40, 60/40/20), dates, section numbers and bibliographic data are written
literally.

- **Revision:** `d7e7d80c4187b363219343d9e088a01ad6ae56fa`, the git tag `archive/research-e02-e13`, which is the
  repository before the 2026-10-31 cleanup.
- **Exceptions:** the small model's three timing runs are read at commits `3ea7019`, `03452be` and `1422153`.
- **Not read:** no protected final reference spelling, reserve target, judgment file or training pair was opened.
  The report uses the committed aggregate files only.

## 1. Headline claims

| Claim in the report | Value | Source file (at the revision) | Field | Metric, denominator | Limitations |
|---|---|---|---|---|---|
| Released adapter acceptable, test sample 1 | 77/120 = 64.2% (`t1.e11.*`) | `experiments/E12-longer-schedule/reports/test_scores.json` | `analysis.counts.e11`, `analysis.confirmed_acceptable_share.e11` | Confirmed acceptable (uncertain = not confirmed); 120 families, one word each | One project-affiliated judge; 40/40/40 design; not a population estimate |
| 6,000-update model, same sample | 77/120 = 64.2%; Δ 0.0 [−8.3, +8.3]; 14 vs 14 discordant; p = 1.00 (`t1.*`) | same | `analysis.counts.e12`, `analysis.primary`, `analysis.paired`, `analysis.exact_test` | Paired family bootstrap, 10,000 resamples, seed 20261027 | One seed; non-significance is not equivalence |
| Released adapter acceptable, test sample 2 | 90/120 = 75.0% (`t2.e11.*`) | `experiments/E13-pronunciation-guided/reports/test_scores.json` | `analysis.counts.e11` | As above, different 120 families | A different sample from test 1: not a later improvement |
| Pronunciation-guided model, same sample | 90/120; Δ 0.0 [−7.5, +7.5]; 11 vs 11 (`t2.*`) | same | `analysis.primary`, `analysis.paired` | As above | Several ingredients changed together; no attribution |
| Groups, test sample 2 | ordinary 34/32, irregular 20/21, names 36/37 (E11/E13) | same | `analysis.secondary_groups.<g>.counts` | 40 words per group | Exploratory; not tested |
| Groups, test sample 1 | ordinary 27/27, irregular 14/19, names 36/31 (E11/E12) | `experiments/E12-longer-schedule/REPORT.md` §5.1 | **table only (no JSON field)**, transcribed | 40 words per group | Transcribed; `extract.py` asserts that they sum to the JSON totals (77, 77) |
| Coverage: 23 words; greedy 8; beam top-1 10; any of 4: 15; none: 8 | `cov.*` | `experiments/E11-real-data-byt5/coverage/results.json` | `overall.words`, `overall.greedy_top1.acceptable`, `overall.beam_top1.acceptable`, `overall.any_acceptable_in_beam4`, `overall.acceptable_per_word.0` | Words of 23 irregular development words; 93 distinct candidates, 100% judged | Development diagnostic; oracle, not reranker accuracy |
| Oracle adds 7 over greedy, 5 over beam top-1 | `cov.gain_vs_greedy`, `cov.gain_vs_beam1` | derived | 15 − 8; 15 − 10 | Valid because `any_acceptable_in_greedy_or_beam4` = `any_acceptable_in_beam4` = 15, and `recoverable_words` = 7 (asserted) | As above |
| Development review | representative 29/24/30 of 40; disagreement 10/5/8 of 20 (E11/E07/E08) | `experiments/E11-real-data-byt5/reports/review_scores.json` | `acceptable_items.<sample>.<model>.acceptable` | Greedy spelling acceptable; samples reported separately | Post hoc dev review; labels given in chat and transcribed under 3 confirmed rules plus clarifications (`provenance.method`; see §5) |
| Reference vs judgment cross-table: 28/72 differing spellings acceptable; 8/37 matching not acceptable | — | `experiments/E11-real-data-byt5/REPORT.md` §5 | **report table only** | 109 distinct judged candidate spellings; the 4 uncertain ones stay in the denominators (72 = 28 + 41 + 3, 37 = 28 + 8 + 1) | Same dev review; no JSON field |
| Small model, phones vs letters | +1.6 (1.646) [−1.6, +4.9]; seeds +0.6, +1.5, +2.8 (`e02.phones`, `s.seed.*`) | `experiments/E02-english-phonetic/results.json` | `comparisons.confirmatory.{mean_gain_pp,ci95_pp,seed_gains_pp}` | Family-macro beam-4 top-1 vs curated references; 162 families; conditional bootstrap | Unqualified benchmark; simulated power for +5 is 0.22–0.50 over the registered discordance grid (`s.power.*`, `power.power_primary`); inconclusive by registration |
| Small model scores | primary 16.7%, top-4 40.4%, CER 0.353 (165 items); proxy 30.1%, top-4 51.3% (400) | same | `runs.letters_s0.primary.*`, `runs.letters_s0.final_proxy.*` | Agreement with one spelling per item | Candidate-anchored references: 82 of 165 primary items, 12 of 26 flagged items, 94 of 191 across both final tracks (`s.anch.*`; `strata.candidate_anchored.n`) |
| Phone sensitivity (small model) | donor phones change top-1 for 33.0–44.7% of 94 words (seeds 0/1/2: 36.2, 33.0, 44.7); letters control 0% (`sens.min`, `sens.max`) | `experiments/E02-diagnostics-2026-10-27/sensitivity/summary.json` | `aggregate.phones_s{0,1,2}.donor.top1_change_rate` (min and max over all seeds) | Share of words whose top-1 changes | Moves toward donor consonants 2–7 per seed: `sensitivity/posthoc.json` `toward_given_phones.phones_s*\|donor.closer_to_given_phones` (post hoc) |
| Phones on screened pairs | +0.6 [−2.4, +3.8] (`e03.phones`) | `experiments/E03-assessable-phones/results.json` | `quality.comparisons.eval` | Mined references; 3 paired seeds | Exploratory set |
| Letters + phones vs letters | +2.2 [−1.5, +6.0]; phones only −8.0 (`e04.*`) | `experiments/E04-pronunciation-pathways/results.json` | `comparisons.GP_vs_G`, `comparisons.P_vs_G` | As above | P has one seed. The report's later clarification (equal budgets, retention) was followed |
| Equal-budget pooling | +1.6 [−1.1, +4.5] (`e05.gp_g8`) | `experiments/E05-candidate-diversity/results.json` | `configs.eval_w8.contrasts.summary.G+P_vs_G8` | Top-8 oracle | Oracle, not accuracy |
| Pronunciation-guided, historical | eval −1.3 [−4.8, +2.3]; reading removed 7.1%, wrong 7.9% of outputs change | `experiments/E13-pronunciation-guided/results.json`, `reports/mechanism.json` | `eval.e13_minus_e11.top1`; `historical_dev.output_changed_vs_correct.*` | 228 / 253 items | Collection-dev rates are 10% / 15% (40 items) |
| Corrected adapter vs small letters models | +5.8 [+0.05, +11.6] (`e07.vs_g`) | `experiments/E07-byt5-diagnosis/results.json` | `historical_eval.contrasts.byt5_e07_minus_G_mean_top1` | Registered beam-4 top-1; 228 items | Registered beam had the early-stopping defect; one adapter run |
| More data (with synthetic rows) | +0.3 [−5.7, +6.5]; greedy +2.6 [−2.4, +7.5] (`e08.*`) | `experiments/E08-data-expansion/results_rs.json` | `historical_eval.contrasts.rs_minus_e07_*` | 228 items | One run |
| Real data vs E07 / vs E08 | +2.5 [−2.7, +7.7]; −0.1 [−4.7, +4.6] (`e11.*`) | `experiments/E11-real-data-byt5/results.json` | `eval.contrasts.*.top1` | Greedy; 222 families | Historical set examined repeatedly |
| Greedy agreement, historical eval | E07 49.9, E08 52.4, E11 52.3, E12 53.0, E13 51.1 (`h.*.top1`) | E11/E12/E13 `results.json` | `eval.table.<m>.top1_macro` | 228 items, 222 families | Figure 1a; exploratory |
| Decoding | canonical − registered beam +5.9 [+1.4, +10.4]; canonical − greedy +0.9 [−1.4, +3.6] (`e09.*`) | `experiments/E09-byt5-decoding/results.json` | `contrasts.e08:selected_minus_beam4_reg.top1`, `contrasts.e08:selected_minus_greedy.top1` | 228 items | "3.60 candidates per item" is from E09 REPORT §3, the *mani* example from §4 |
| Longer schedule, historical | +0.7 [−4.1, +5.4] (`e12.vs_e11.hist`) | `experiments/E12-longer-schedule/results.json` | `eval.contrasts.e12_minus_e11.top1` | 222 families | "65 of 120 spellings changed" is from E12 REPORT §5.2 |
| Release parity | 0/276 dev, 0/245 eval greedy, 0 top-4 (`par.*`) | `experiments/E11-real-data-byt5/release/verify.json` | `parity.*` | All dev/eval rows, including non-primary items | — |
| Isolated install | 19/19 checks, dev parity 0, load 5.5 s, 14.1 words/s (`b.i.*`) | `.../release/install_check.json` | `checks`, `api.*` | Fresh uv venv, CPU torch | Network blocked by a Python socket guard (unprivileged namespaces refused); one machine |
| Adapter runtime | 554 ms greedy, 820 ms top-4, 14.5 words/s, peak 2,019 MB, load 7.1 s (`b.*`) | `.../release/verify.json` | `latency_ms.*`, `parity.dev.greedy_words_per_second`, `peak_rss_mb`, `load_seconds` | i7-8750H, 6 threads | Single machine |
| Small model runtime | load 0.15–0.16 s; p50 17–19 ms; batch-16 13 ms/word; peak 65 MiB (`s.t1..t3.*`) | `experiments/E02-release/reports/install_verify.json` at 3 commits | `performance.*` | 103 dev words, fresh process | Range over 3 runs (see §3) |

## 2. Numbers taken from report prose or tables (no JSON field)

| Number | Source |
|---|---|
| 11,377 rows recovered; 1,507 excluded; 65,422 mined pairs reconciled; 2,967 DACS pairs | `experiments/E03-assessable-phones/REPORT.md` Q1 and summary |
| 0 of 10,031 targets with exact lexicon lookup; 88.6% consonant agreement on 30,980 held-out headwords | `experiments/E02-diagnostics-2026-10-27/REPORT.md` §1 |
| 1,292 recovered, 1,217 ALT-attested; 384 synthetic rows / 193 words; no synthetic rows in arm R | `experiments/E08-data-expansion/REPORT.md`; `release/byt5-e11/DATA_PROVENANCE.md` "Verified absences" (archived at the revision) |
| Arm R: 4,764 rows, 3,585 words; control 2,255, recovered 1,292, ALT 1,217 | `experiments/E11-real-data-byt5/release/identity.json` and the adapter config `provenance.rows_by_provenance` |
| Small model: 9,735 pairs, 296 validation; epoch 27 of 40; 86 dev items; architecture | released card (`release/en-km-phonetic-transliterator/README.md`); E02 RESULTS "Selection" |
| E06 output ហ្វេន for 204 of 228 words; UTF-8 byte explanation | `experiments/E07-byt5-diagnosis/REPORT.md` §1 |
| (now JSON) Vowel-only share 41–49% of mismatches | derived from E11 `results.json` `eval.error_kinds` (46/113, 47/108, 53/108) |
| (now JSON) Overlap: 110 of 228 primary eval items (E07 `results.json` `strata.in_default_e02_training=True.items`); 122 of 245 words in the full partition (E03 `reports/overlap.json` `eval.e02_train_words`, `eval.words`) | — |
| (now JSON) Power 0.22–0.50 (0.29 at discordance 0.20) | E02 `results.json` `power.power_primary` |
| Released file hashes: model.npz `8901050b…` (Hub, tag v0.2.0) = `letters_s0`; adapter `698a23ae…` (Hub, revision `77f36282`) = local export; training checkpoint `286292a4…` | Hub API `files_metadata` (checked 2026-10-31); `experiments/E02-release/reports/provenance_facts.json`; E11 `results.json` `selected.checkpoint_sha256` |
| Khmer examples: ខូហ្វី, ឡុនដុន (release smoke), កាហ្វេ (card), ចនសុន (mining example), ខេមឺរ៉ុន, បេន, បាន (dev review, E11 REPORT §5), ម៉ានី → មានី (E09 REPORT §4), ហ្វេន (E07 REPORT §1) | all recorded in the project; none comes from a protected reference set |

## 3. Discrepancies found and how they were resolved

1. **Small model latency.** The public card gives 17 ms p50 / 30 ms p95, which is the first of three recorded
   install checks (`3ea7019`: 16.7 / 29.6 ms). The two later runs measured 18.6 / 33.6 ms (`03452be`) and
   19.2 / 38.0 ms (`1422153`), and the experiment report gives 18 / 32 ms. The report gives the 17–19 ms range and
   notes this in its appendix.
2. **Pooled development review.** Project notes (decision D194, the E11 report's "All 60" row) quote 65% / 48% /
   63% over 60 words. These figures pool the representative and the disagreement-enriched samples, so they are
   not used.
3. **E07 eval 47.2% vs 49.9%.** The E07 report's 47.2% is the registered beam-4 with `early_stopping=True`. The
   49.9% is greedy decoding (E09, E11). The report uses greedy figures for adapters and explains the defect.
4. **E13 eval 51.05%.** The stored value is 0.5105. Float formatting would print 51.0; the project reports 51.1.
   The extract rounds half up from the decimal text, so 51.1 is used.
5. **E02 interval.** The records quote +1.65 [−1.65, +4.94]. With one decimal the report shows +1.6 [−1.6, +4.9].
   The E07 lower bound, +0.05, is kept at two decimals so it does not read as +0.1.
6. **Parity denominators.** The release parity used 276 dev and 245 eval rows, while scoring used 253 and 228
   primary items. Both counts are correct; parity covers every row, including non-primary items.
7. **E12 test groups.** These exist only as a REPORT table. The values were transcribed, and `extract.py` asserts
   that the group sums equal the JSON totals.
8. **"Recovered" pairs.** Two different recoveries are involved: E03's selection repair (11,377 rows back into the
   candidate pool) and E08's recovery of 1,292 screen-rejected pairs. The report keeps them apart.
9. **No cross-model comparison.** The historical eval set overlaps the small model's training words (110 of 228
   primary items; 122 of 245 words in the full partition), so the two released models are not compared. E08's table
   row for the "default E02 letters" model (46.2%) is therefore not used.

## 4. Release relationship (verified 2026-10-31)

| Release | Hub revision | File | sha256 | Recorded checkpoint |
|---|---|---|---|---|
| `ric912/en-km-phonetic-transliterator` | tag `v0.2.0` (`ab88afe3`) | `model.npz` | `8901050b…904b` | `models/e02/letters_s0/model.npz` (same hash) |
| `ric912/en-km-phonetic-byt5-lora` | `77f36282` | `adapter.safetensors` | `698a23ae…cc3f` | export of `models/e11/r_s0/best.pt` (`286292a4…`), bit-identical tensors (`release/export.json`) |

## 5. Correction pass (2026-10-31, decision D203)

These errors in the first version of the report were found in review and corrected. The historical records were
not changed.

| Item | First version | Corrected | Source |
|---|---|---|---|
| Candidate-anchored references | "94 items chosen from an earlier model's candidates", placed inside the 165-item primary set | 82 of 165 primary; 12 of 26 flagged; 94 of 191 across both final tracks | E02 `results.json` `runs.letters_s0.{primary,final_flagged}.strata.candidate_anchored.n` |
| Donor-phone sensitivity range | 36.2–44.7% (seed 0 to seed 2) | 33.0–44.7% (minimum seed 1, maximum seed 2) | `sensitivity/summary.json`; now min/max over all seeds |
| Historical-eval overlap | "122 of its words" (the full 245-word partition, not the 228-item primary set) | 110 of 228 primary items; 122 of 245 words in the full partition, each labelled | E07 `results.json` strata; E03 `reports/overlap.json` |
| Review procedure | "Every spelling required an explicit click" | Tests and new coverage judgments: clicks on the review page. Development review: chat, transcribed under three confirmed rules. Coverage: 90 of 93 judgments clicked, 3 reused from the development review | `review_scores.json` `provenance.method`; test and coverage `provenance.tool`; `coverage/batches.json` |
| Transcription rules | "four interpretation rules" | three rules (unnamed candidates not acceptable; "ok" without a label = all acceptable; skipped single-candidate item = acceptable), plus the judge's clarification of skipped multi-candidate items, two second-reply conventions ("not sure", "maybe C too") and three confirmations | `review_scores.json` `provenance.method` |
| Power | "estimated power of 0.29" | 0.22–0.50 across the registered discordance grid | E02 `results.json` `power.power_primary` |
| Vowel share | "40–50%" (prose) | 41–49%, derived from error-kind counts | E11 `results.json` `eval.error_kinds` |
| Wording | "left acceptability unchanged"; "review separated models"; "no (pipeline) bug" | "equal observed acceptability; no difference was detected, and the intervals permit gains or losses"; "the exploratory development review suggested a preference for the expanded real-data model" (post hoc, one judge, disagreement-enriched sample reported separately); "the audit found no pipeline defect explaining the collapse" | — |
| Traceability claim | "Every number … is generated from committed result files" | Values are `json`, `derived` or labelled `transcribed`; settings and design constants are literal | `tools/extract.py` docstring, README, appendix C |

**Checks run by `tools/extract.py`** (each one fails the build):
- candidate-anchored primary + flagged = combined (82 + 12 = 94), and population sizes 165 + 26 = 191;
- the sensitivity range is the minimum and maximum over all three seeds;
- overlap counts are paired with their denominators: 110 + 118 = 228 = the historical eval item count, and 122
  of 245 for the full partition;
- every `\val{key}` in `main.tex` exists in the extract;
- derived percentages match their counts under half-up rounding: the four test shares, 28/72 = 39% and
  8/37 = 22%;
- also: the cross-table totals (28 + 41 + 3 = 72; 28 + 8 + 1 = 37; 72 + 37 = 109), the coverage provenance
  (50 new + 43 reused = 93; 40 + 3 = 43), the arm R composition (4,764 − 1,292 − 1,217 = 2,255), the
  small-model dev scores being multiples of 1/86, and the earlier test, group and coverage relationships.
