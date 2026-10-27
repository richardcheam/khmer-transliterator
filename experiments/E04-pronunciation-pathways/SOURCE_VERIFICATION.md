# E04 source verification (2026-10-27)

Each source was checked against its abstract page (fetched this session). No full PDF was re-read, so the claims
below are limited to what the abstracts state. None of these works studies English → Khmer; they motivate the
design, they do not predict its result.

| Source | Checked | What the abstract supports | What it does not establish | Use in E04 |
|---|---|---|---|---|
| Oh, Choi & Isahara, "A Comparison of Different Machine Transliteration Models", JAIR 27 (2006), arXiv:1110.1391 | arxiv.org/abs/1110.1391 | Grapheme-, phoneme-, hybrid- and correspondence-based models compared in one framework on the same data; hybrid and correspondence-based models the most effective; the models "can be used in a complementary manner" | Khmer; which language pairs (not on the abstract page); a specific complementary-error analysis | Motivation for G vs P vs GP and the complementarity counts |
| Wang, Tran & Feiszli, "What Makes Training Multi-Modal Classification Networks Hard?", CVPR 2020 | the CVF page returned HTTP 403; verified on arXiv:1905.12681 | The best uni-modal network can beat the multi-modal one; causes named: overfitting from added capacity, and modalities generalising at different rates; remedy Gradient-Blending | Anything about derived "modalities" (our phones are computed from the letters); transliteration | Analogy for "GP may not keep P's signal"; no Gradient-Blending arm |
| Neverova, Wolf, Taylor & Nebout, "ModDrop: adaptive multi-modal gesture recognition", TPAMI 2016 (doi 10.1109/TPAMI.2015.2461544) | the DOI resolves to IEEE Xplore, whose page content was empty; verified on arXiv:1501.00102 | Random dropping of separate channels ("ModDrop") during gradual fusion; robustness to missing signals; predictions from any subset of modalities | Text transduction; that sampling one view per example helps | Motivation for the conditional three-view recipe, which is this project's hypothesis |
| Kunchukuttan, Khapra, Singh & Bhattacharyya, "Leveraging Orthographic Similarity for Multilingual Neural Transliteration", TACL 6 (2018) | aclanthology.org/Q18-1022 | Multilingual shared encoder-decoder transliteration between orthographically similar languages; "further improvements can be achieved by using phonetic feature input" | Which features exactly (the abstract does not say articulatory); Khmer | Background; articulatory features deferred |
| Rios, Amrhein, Aepli & Sennrich, "On Biasing Transformer Attention Towards Monotonicity", NAACL 2021 | aclanthology.org/2021.naacl-main.354 | A monotonicity loss tested on G2P, inflection, transliteration and normalisation; results mixed; general monotonicity does not help multi-head Transformer attention; isolated gains from biasing a subset of heads | Transliteration-specific numbers (not in the abstract) | Reason for no monotonic-attention arm |

**This project's hypotheses, not sourced findings:**
- that phones alone carry a usable English → Khmer signal here;
- that GP may fail to keep it;
- that the three-view recipe would recover it.
