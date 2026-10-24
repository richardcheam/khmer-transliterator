# E00 — History before the research protocol (D133–D164, 2026-10-12 → 2026-10-15)

These experiments were run before the protocol in `docs/RESEARCH_PROTOCOL.md` existed. Their rules were written in
the decision log before results (archived: `git -C ../pocket-tts-khmer show archive/checkpoints-p0-p6:state/DECISIONS.md`),
but they were mostly single-seed and were not scored on a frozen benchmark, so their numbers are not directly
comparable with E01 and later. Their scripts are archived in the speech repo's tag
(`archive/scripts/transliterator/`), and the weights of the unserved arms are on the HDD
(`/mnt/hdd/data/khmer-tts-archive/data/translit/`).

| Arm | Recipe | Key result | Fate | Report |
|---|---|---|---|---|
| A `models/model` | Chhunneng only, letters | Romanised 69.2%; owner A/B vs rules 8 / 2 / 2 (D134) | default weights | `reports/translit_eval.json` |
| B `models/model_b` | A + mined pairs | Romanised 70.0%; names test2 48.5%; owner A/B vs A 9 / 2 / 1 (D138) | **served** (names, romanised) | `reports/translit_ab_*.json`, `translit_diag_*.json`, `translit_listen*.json` |
| C | B + list + phones | fair-37: 10 vs B 7 (D146) | not served | `reports/translit_c_eval.json` |
| D | C + LLM drafts ×1 | tie with C (D147) | not served | `reports/translit_d_eval.json` |
| E `models/model_e` | C + drafts as loans ×4 | Better on English (10 / 4), worse on names; owner A/B 14 / 8 / 8 (D150) | **served** (English words) | `reports/translit_e_eval.json`, `translit_ab_e.json`, `translit_ab_cat*.json`, `translit_ab_route.json`, `translit_ab_split.json` |
| T | E + group tags (single model) | Romanised 68.4% < 68.9% bar; names 1 / 2 (D149) | failed | `reports/translit_t_eval.json` |
| E2a / E2c | English-only fine-tune of E | dev-41 21 vs E 25 (D151) | failed | — |
| reranker | beam 10 + character LM + web counts | no gain over served pick (D157) | not used | — |
| ranker | logistic regression on 125 owner picks | dev-41 20 < 25 (D159) | failed | `reports/translit_ranker_fit.json`, `translit_pick.json` |
| s0 / s1 | E recipe + picks; s1 + 3,976 Gemini pairs | pick test 49 vs 45 (p 0.83) (D164); s0/s1 used max_in 32 by mistake (D171) | failed | `reports/translit_synth_*.json` |

Lessons carried into the protocol: single runs move dev-41 by about ±5 words (use paired seeds); the mined pairs
leaked 99 of 150 test words (filter test sets first); more LLM-written data stopped helping.
