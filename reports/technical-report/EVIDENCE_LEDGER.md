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
- **Not read:** no protected final reference spelling, reserve target or judgment file was opened. The numbers use
  the committed aggregate files only. For the worked examples (§6), individual rows were looked up in the local
  training file of the released adapter (`data/e08/training/arm_R.tsv`), the mining record
  (`data/mined_fineweb.tsv`) and the CMU Pronouncing Dictionary (`data/src/cmudict/cmudict.dict`); nothing else
  was read from them.

## 1. Headline claims

| Claim in the report | Value | Source file (at the revision) | Field | Metric, denominator | Limitations |
|---|---|---|---|---|---|
| Released adapter acceptable, test sample 1 | 77/120 = 64.2% (`t1.e11.*`) | `experiments/E12-longer-schedule/reports/test_scores.json` | `analysis.counts.e11`, `analysis.confirmed_acceptable_share.e11` | Confirmed acceptable (uncertain = not confirmed); 120 families, one word each | One project-affiliated judge; 40/40/40 design; not a population estimate |
| 6,000-update model, same sample | 77/120 = 64.2%; Δ 0.0 [−8.3, +8.3]; 14 vs 14 discordant; p = 1.00 (`t1.*`) | same | `analysis.counts.e12`, `analysis.primary`, `analysis.paired`, `analysis.exact_test` | Paired family bootstrap, 10,000 resamples, seed 20261027 | One seed; non-significance is not equivalence |
| Released adapter acceptable, test sample 2 | 90/120 = 75.0% (`t2.e11.*`) | `experiments/E13-pronunciation-guided/reports/test_scores.json` | `analysis.counts.e11` | As above, different 120 families | A different sample from test 1: not a later improvement |
| Pronunciation-guided model, same sample | 90/120; Δ 0.0 [−7.5, +7.5]; 11 vs 11 (`t2.*`) | same | `analysis.primary`, `analysis.paired` | As above | Several ingredients changed together; no attribution |
| Groups, test sample 2 | ordinary 34/32, irregular 20/21, names 36/37 (E11/E13) | same | `analysis.secondary_groups.<g>.counts` | 40 words per group | Exploratory; not tested |
| Groups, test sample 1 | ordinary 27/27, irregular 14/19, names 36/31 (E11/E12) | `experiments/E12-longer-schedule/REPORT.md` §5.1 | **table only (no JSON field)**, transcribed | 40 words per group | Transcribed; `extract.py` asserts that they sum to the JSON totals (77, 77) |
| Several acceptable spellings: 12 of 23 words had 2 or 3 accepted candidates (`cov.multi`) | 9 + 3 | `experiments/E11-real-data-byt5/coverage/results.json` | `overall.acceptable_per_word.2`, `.3` (derived sum) | Words of 23 | Names no spellings; individual judgments stay private |
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
| Greedy agreement, historical eval | E07 49.9, E08 52.4, E11 52.3, E12 53.0, E13 51.1 (`h.*.top1`) | E11/E12/E13 `results.json` | `eval.table.<m>.top1_macro` | 228 items, 222 families | Table 2 uses E11 vs E07 only; the former dot chart was removed in the rewrite; exploratory |
| Decoding | canonical − registered beam +5.9 [+1.4, +10.4]; canonical − greedy +0.9 [−1.4, +3.6] (`e09.*`) | `experiments/E09-byt5-decoding/results.json` | `contrasts.e08:selected_minus_beam4_reg.top1`, `contrasts.e08:selected_minus_greedy.top1` | 228 items | "3.60 candidates per item" is from E09 REPORT §3, the *mani* example from §4 |
| Longer schedule, historical | +0.7 [−4.1, +5.4] (`e12.vs_e11.hist`) | `experiments/E12-longer-schedule/results.json` | `eval.contrasts.e12_minus_e11.top1` | 222 families | "65 of 120 spellings changed" is from E12 REPORT §5.2 |
| Release parity (kept here, not in the paper) | 0/276 dev, 0/245 eval greedy, 0 top-4 (`par.*`) | `experiments/E11-real-data-byt5/release/verify.json` | `parity.*` | All dev/eval rows, including non-primary items | — |
| Isolated install (kept here, not in the paper) | 19/19 checks, dev parity 0, load 5.5 s, 14.1 words/s (`b.i.*`) | `.../release/install_check.json` | `checks`, `api.*` | Fresh uv venv, CPU torch | Network blocked by a Python socket guard (unprivileged namespaces refused); one machine |
| Adapter runtime (kept here, not in the paper) | 554 ms greedy, 820 ms top-4, 14.5 words/s, peak 2,019 MB, load 7.1 s (`b.*`) | `.../release/verify.json` | `latency_ms.*`, `parity.dev.greedy_words_per_second`, `peak_rss_mb`, `load_seconds` | i7-8750H, 6 threads | Single machine |
| Small model runtime (kept here, not in the paper) | load 0.15–0.16 s; p50 17–19 ms; batch-16 13 ms/word; peak 65 MiB (`s.t1..t3.*`) | `experiments/E02-release/reports/install_verify.json` at 3 commits | `performance.*` | 103 dev words, fresh process | Range over 3 runs (see §3) |

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
   19.2 / 38.0 ms (`1422153`), and the experiment report gives 18 / 32 ms. The first version gave the 17–19 ms range; the
   rewritten report no longer quotes latency ("tens of milliseconds" only).
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

## 6. Worked examples and figure provenance (rewrite, 2026-11-03)

Every example in the report is a recorded artifact. Role = training, development or test, as recorded.

| Example in the report | Role | Verified against |
|---|---|---|
| Figure 1A source text ``…ត្រូវបានចែករំលែកតាមបណ្តាញសង្គមហ្វេសប៊ុក (Facebook)'' and the pair facebook → ហ្វេសប៊ុក | training | `data/mined_fineweb.tsv` (`source_phrase`, class `khmer_like`, 1,576 occurrences in the FineWeb-2 mining run); the pair is in `data/e08/training/arm_R.tsv` (provenance `control`). The English gloss is the author's translation |
| facebook was dropped by the old filter because its Khmer side is a dictionary word | training | `scripts/translit_mine.py` rule (`khmer_like` if Latin is romanised Khmer *or* Khmer is known); ហ្វេសប៊ុក is a key of the Google lexicon (`khmer_translit.lexicon.keys()`) |
| Figure 1A pairs taxi → តាក់ស៊ី, google → ហ្គូហ្គល | training | `arm_R.tsv`, provenance `control` |
| coffee: កាហ្វេ, កូហ្វី, ខូហ្វី | training | `arm_R.tsv` (provenance `control`; a fourth spelling ខូហ្វ៊ី is `recovered`) |
| hotel → តែល | training (recovered) | `arm_R.tsv`, provenance `recovered` |
| london → ឡុនដុន (intro) | training word; released model's output | `arm_R.tsv` (`recovered`); release smoke output in the adapter card |
| Figure 1B: band, reading `B AE1 N D`, released adapter output បេន, judged acceptable | development: excluded from training, but in the dev set used for checkpoint selection (E11 REPORT §5: "the words are dev items (the selection set)"; §3: "Dev was used for checkpoint selection"). Exploratory, not a test example | output: `experiments/E11-real-data-byt5/predictions/e11_dev_greedy.jsonl` and `results_per_item_dev.jsonl` (`correct: 0`, reference differs); judgment: E11 REPORT §5 ("Differs but acceptable … *band* បេន"); reading: CMUdict |
| band → បាន by earlier adapters, matches the reference, not acceptable | development | E07/E08 dev predictions (`correct: 1`); E11 REPORT §5 ("Matches but not acceptable … *band* បាន") |
| phi: released ភី matches the reference, not acceptable; earlier ហ្វី acceptable | development | `results_per_item_dev.jsonl` (E11 `correct: 1`, E07 `correct: 0`); E11 REPORT §5; reading `F AY1` from CMUdict |
| cameroon → ខេមឺរ៉ុន, differs, acceptable | development | E11 dev predictions; E11 REPORT §5 |
| mani: greedy ម៉ានី, original beam មានី | development (40-item disagreement inspection) | E09 REPORT §4 |
| `fire` → `fire\|F AY1 ER0` input format | format example | E13 REPORT "What was changed" |
| ហ្វេន for 204 of 228 words | historical evaluation | E07 REPORT §1 (`e06.top`) |
| share / shares / sharing / shared as one family | rule example | `src/khmer_translit/reference_data.py` `stems` and `group_families` docstring at the revision (suffix stripping, joined transitively) |
| *colonel* (irregular spelling) and *sok sabay* (romanised Khmer) | illustrations only | not data claims |
| Figure 2 (coverage bars) | development diagnostic | `cov.greedy`, `cov.beam1`, `cov.any4`, `cov.none` |

Multiple accepted spellings: the committed records count words with two or three accepted candidates (`cov.multi`)
but do not name the spellings, and the judgment files were not opened, so the report gives the count only.

## 7. What the practitioner rewrite moved out of the paper (2026-11-03)

The rewrite changed presentation only; no value was recomputed or changed. Moved out of the main text and kept
here or in the repository:
- release checks (export parity, isolated install, hashes of model files and checkpoints), latency and memory
  figures (§1 rows marked "kept here"; §4);
- hashes, revisions, update numbers and checkpoint identifiers (§1, §4, `data/results_extract.json`);
- the experiment-by-experiment inventory with its intervals (the experiment folders and `docs/DECISIONS.md`);
- per-seed scores, power estimates and discordant-pair counts other than the test-1 split (§1);
- the former Figure 1 (reference agreement and acceptability dot charts) and Table 1 (release comparison).

Kept in the paper: the corrected D203 values (candidate-anchored 82/165 and 94/191, sensitivity 33.0–44.7%,
overlap 110/228 and 122/245) in Appendix A and §4.3; the chat-transcribed vs click distinction in §3 and Appendix A;
"no clear difference / not equivalence" wording in Table 2 and §4; "no pipeline defect explaining the collapse" in
§2.1. Paired differences and intervals appear once, in Table 2, with four further comparisons in Appendix A.

**Editorial pass (2026-11-04).** Figure 1B's note previously said *band*'s family was kept out of training and
checkpoint selection; the second half was wrong and now reads "excluded from training; included in the development
set used to select the checkpoint". The detailed comparison table (all paired differences and intervals, the former
Table 2) moved unchanged to Appendix A as Table 3; the main text now has an outcomes table (Table 2: question,
observed outcome, main limitation) whose only numbers are `t1.e11.acceptable`, `t1.changed` and
`t2.e11.acceptable`. The abstract and introduction now say "we found no suitable curated training dataset or
established benchmark" instead of stating that none exists.

## 8. Architecture figure (Figure 1, added 2026-11-04)

Drawn from the implementation, not from a generic Transformer or PEFT diagram. Figures are now numbered
1 (architectures), 2 (data and review pipeline) and 3 (candidate coverage).

| Element in Figure 1 | Verified against |
|---|---|
| Small model input = task marker + one token per letter, no end symbol | `src/khmer_translit/phonetic_source.py` `encode_source` (letters mode: `<task:en>` then the letters; doctest `['<task:en>', 'o', 'n', 'e']`); `translit.Transliterator.encode_ids` adds nothing else |
| Small model: encoder and decoder, 3 layers each, width 192, feed-forward 512, 4 heads; learned embeddings and positions; decoder cross-attends to the encoder states | `models/e02/letters_s0/model.npz`: `enc.0-2`, `dec.0-2`, `src_emb (157, 192)`, `tgt_emb (81, 192)`, `enc.0.ff1 (512, 192)`, meta `heads 4`, `layers 3`; `translit.py` `decode_step` (self, cross, ff) |
| Small model output = one Khmer Unicode character per step; start symbol first, stops at the end symbol | `translit.py` (`PAD, BOS, EOS = 0, 1, 2`; beam search starts from `[BOS]`, stops at `EOS`); target vocabulary = 78 Khmer characters + 3 specials; all 7 characters of ខូហ្វ៊ី are in it |
| Small model fully trained | trained from scratch (E02); no frozen part |
| ByT5 input = UTF-8 bytes, token id = byte + 3, then an end symbol | `src/khmer_translit/byt5.py` `encode` (`OFFSET = 3`, `EOS` appended; doctest `encode("ab") == [100, 101, 1]`); `coffee` bytes 63 6F 66 66 65 65 computed with Python |
| ByT5 sizes: 12 encoder layers, 4 decoder layers, width 1472, gated feed-forward | base `config.json` at the pinned revision (`num_layers 12`, `num_decoder_layers 4`, `d_model 1472`, `feed_forward_proj gated-gelu`) |
| ByT5 generation starts from the start symbol and stops at the end symbol | `config.json` `decoder_start_token_id 0`, `eos_token_id 1`; `byt5.decode` stops at EOS |
| LoRA placement: q, k, v, o of every `T5Attention` (encoder self-attention, decoder self- and cross-attention) and wi_0, wi_1, wo of every gated feed-forward block; everything else frozen | `byt5.py` `TARGETS` and the loop over `T5Attention` / `T5DenseGatedActDense` after `requires_grad_(False)`; released `adapter_config.json` targets `["attn", "ffn"]`; 128 wrapped layers × (A, B) = 256 tensors = `x.tensors` |
| Output bytes E1 9E 81 = ខ (U+1781); ខូហ្វី is 18 bytes | computed with Python from the displayed text |
| coffee → ខូហ្វ៊ី (small model) | recorded first candidate in `experiments/E02-release/reports/install_verify.json` (`cli_single_json`) at the archived revision; also the model card |
| coffee → ខូហ្វី (adapter) | recorded greedy output in `experiments/E11-real-data-byt5/release/verify.json` `smoke.coffee`; also the model card |

The two outputs differ; the figure shows each as recorded and makes no quality comparison. *coffee* is a training
word for both models, so the example illustrates the mechanism, not evaluation. No model was run for the figure.

**Figure 1 redesign (2026-11-04).** Two rows per panel (word → tokens → embeddings → encoder; decoder → generated
units → Khmer text), one encoder-states arrow between rows, and a separate dashed feedback path. In panel B the
attention and feed-forward parts are drawn frozen with a separate trained block labelled "LoRA adapter" beside each,
matching `byt5.py` (the adapter adds `B·A·x` to a frozen linear layer). The panel A output shows all 7 characters
of ខូហ្វ៊ី; panel B shows the first three output bytes (E1 9E 81 = ខ) as a labelled excerpt with "ខ: 3 bytes;
complete output: 18 bytes". Layer counts and the byte-to-token offset (+3) moved to the caption. The figure is a
full-width float on its own page (page 2); the PDF stays at 9 pages.
