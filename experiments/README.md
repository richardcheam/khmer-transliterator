# Experiments

One folder per experiment: `PLAN.md` (committed before training), `config.yaml`, `RESULTS.md`, `results.json`.
How to run one: `../docs/RESEARCH_PROTOCOL.md`. New experiment: copy `TEMPLATE/`.

| Id | Question | Outcome | Decision |
|---|---|---|---|
| [E00-history](E00-history/) | Arms A–T, E2, reranking, ranker, s0/s1 (before this protocol) | B and E served; every later English-spelling route failed its rule | D133–D164 |
| [E01-unified](E01-unified/) | Does a Khmer-pronunciation side task improve one clean model? | +4.67 pts on English, all seeds positive, no harm elsewhere; **missed the 5-pt rule**. Clean models trail served B/E on real romanised typing (33–35% vs 70%) | D173, D174 |

| [E02-english-phonetic](E02-english-phonetic/) | English words → Khmer phonetic spelling: do English phone inputs, or Khmer P2G transfer, help a clean model? | phones +1.65 pts [−1.65, +4.94] over 3 paired seeds; benchmark unqualified → inconclusive; letters baseline kept (research preview) | D178 |
| [E02-diagnostics-2026-10-27](E02-diagnostics-2026-10-27/) | Diagnostics (no training): do E02's training targets follow the supplied phones, and do the phone models use them? | 0% of targets readable by exact lookup (by construction); 30% consonant-conflict flags on a validated spelling reading; phone sensitivity shallow and seed-dependent | D179 |
| [E03-assessable-phones](E03-assessable-phones/) | Repaired `khmer_like` selection; letters vs phones on dictionary-assessable, consonant-screened pairs | phones +0.60 pts [−2.40, +3.75]; donor phones change 34% of outputs but steer consonants +1.4% [−0.8, +3.5]; default unchanged | D180 |
| [E04-pronunciation-pathways](E04-pronunciation-pathways/) | Letters (G) vs phones only (P) vs both (GP), corrected schedule; conditional three-view (not triggered) | GP − G +2.20 pts [−1.48, +6.01] (3 seeds); P − G −8.0 (1 seed); G ∪ P top-4 +8–12 pts; default unchanged | D181 |
| [E05-candidate-diversity](E05-candidate-diversity/) | Inference only: G+P candidate pooling vs G8 and G+G at matched beam-8 (exploratory, historical data) | eval G+P − G8 +1.6 [−1.1, +4.5], − G+G +0.8 [−2.3, +3.9]: no clear advantage; review queue prepared | D182 |
| [E06-reference-model](E06-reference-model/) | ByT5-small + LoRA (letters only) as a reference/teacher; Gemini route unavailable; teacher gate; constrained-discovery audit | ByT5 ignores its input (eval top-1 0% vs G 40–43%); gate unresolved (no judgments); distillation skipped; discovery 3/200 plausible | D183 |
| [E07-byt5-diagnosis](E07-byt5-diagnosis/) | Why E06's ByT5 collapsed; 32-pair memorisation gate; one corrected LoRA adaptation (attn+FFN, 3,000 updates) | No pipeline bug (adaptation limitation); gate passed; eval top-1 47.2% vs G mean 41.3% (+5.8 [+0.05, +11.6]), vs default +1.0 [−6.1, +8.0]; ~35× slower | D184 |
| [E08-data-expansion](E08-data-expansion/) | Data for ByT5-small: recovery of E03-excluded pairs, source audit (ALT used; ECCC/Khmerologie blocked), 200-word Google AI Mode pilot, phase-B plan (C vs R vs R+S) | +1,292 recovered, +1,217 ALT-attested pairs; 193 synthetic words; one R+S ByT5 run: eval +0.3 [−5.7, +6.5] vs E07 → small or uncertain change | D185, D186 |
| [E09-byt5-decoding](E09-byt5-decoding/) | Inference-only audit of ByT5 decoding: why greedy beat beam-4 on E07/E08 | Configuration defect (`early_stopping=True` stops byte beams on junk); canonical beam ≈ greedy; vs registered beam-4 E08 +5.9 [+1.4, +10.4]; E08 − E07 matched +2.6 / +3.9 (CIs include 0) | D187 |
| [E10-release-cycle](E10-release-cycle/) | Bounded improvement (recipe A: word-level sampling + LR 2e-3) and a local ByT5 release candidate | Gate failed (dev +1.2 < +3.0; eval +0.6 [−4.5, +5.9]); E08 kept and packaged as `khmer-translit-byt5` 0.2.0 (local research preview; review prepared, not collected) | D188 |

**Later candidate:** E01's aux recipe plus clean real romanised-typing data, scored on benchmark v1.
