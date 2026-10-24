# Experiments

One folder per experiment: `PLAN.md` (committed before training), `config.yaml`, `RESULTS.md`, `results.json`.
How to run one: `../docs/RESEARCH_PROTOCOL.md`. New experiment: copy `TEMPLATE/`.

| Id | Question | Outcome | Decision |
|---|---|---|---|
| [E00-history](E00-history/) | Arms A–T, E2, reranking, ranker, s0/s1 (before this protocol) | B and E served; every later English-spelling route failed its rule | D133–D164 |
| [E01-unified](E01-unified/) | Does a Khmer-pronunciation side task improve one clean model? | +4.67 pts on English, all seeds positive, no harm elsewhere; **missed the 5-pt rule**. Clean models trail served B/E on real romanised typing (33–35% vs 70%) | D173, D174 |

**Next candidate (E02):** E01's aux recipe plus clean real romanised-typing data, scored on benchmark v1.
