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

**Later candidate:** E01's aux recipe plus clean real romanised-typing data, scored on benchmark v1.
