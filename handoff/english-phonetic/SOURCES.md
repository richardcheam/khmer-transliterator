# Consulted sources and verification ledger

This is a source map, not a claim that every proposed component is established to improve English→Khmer. Consult the primary documents and actual local versions before implementation. Record inaccessible resources, changed claims and permission uncertainty in `SOURCE_VERIFICATION.md`. Do not present a search snippet as a completed dataset/code audit.

The preparation inspected repository commit `b3f2096c94c13d8e87c5a771fbc79af623cc1c89` and the linked primary research. Research support and project design decisions are separated below. These notes paraphrase sources; no paper is bundled.

## Repository evidence

**R1 — Benchmark v1 manifest**

https://github.com/richardcheam/khmer-transliterator/blob/b3f2096c94c13d8e87c5a771fbc79af623cc1c89/benchmarks/v1.json

Pins five panels: romanised 2,226; Cambodian-name words 223; English owner picks 150; development diagnostic 41; list holdout 125. States picks mostly come from E's top five. It freezes D173/D174 panels and excludes the sealed reserve. It establishes panel definitions, not independent phonetic label validity. Local TSVs were unavailable during preparation.

**R2 — Benchmark implementation**

https://github.com/richardcheam/khmer-transliterator/blob/b3f2096c94c13d8e87c5a771fbc79af623cc1c89/src/khmer_translit/benchmark.py

Provides hash/count checks, multiple accepted forms, raw/reranked scoring and paired word bootstrap. Display scores are rounded while exact counts are also recorded. Existing `outcomes()` uses literal string equality; the new scorer's Unicode policy must be explicit and does not change v1 retroactively. The bootstrap is conditional on the fixed trained seeds and does not establish linguistic reference validity.

**R3 — Data card**

https://github.com/richardcheam/khmer-transliterator/blob/b3f2096c94c13d8e87c5a771fbc79af623cc1c89/datasets/README.md

Documents source origins, noisy mined alignments, owner-preference data, LLM drafts, historical leakage findings and the undocumented origin of Chhunneng pairs. Verify actual source notices locally; do not infer blanket redistribution permission from this table alone.

**R4 — Research protocol**

https://github.com/richardcheam/khmer-transliterator/blob/b3f2096c94c13d8e87c5a771fbc79af623cc1c89/docs/RESEARCH_PROTOCOL.md

Requires registration, provenance, held-out filtering, paired seeds and honest failed gates. It says adaptively inspected material becomes development data. Its historical TTS listening/owner-serving procedure belongs to the old workflow; this new English-only standalone experiment excludes TTS by explicit owner instruction.

**R5 — E01 actual implementation**

https://github.com/richardcheam/khmer-transliterator/blob/b3f2096c94c13d8e87c5a771fbc79af623cc1c89/scripts/translit_unified.py

Actual encoding, counts-based gate and training/export code. At the inspected version both saved/exported source metadata use `phones=False`. Compare this with R6 rather than assuming the prose is authoritative. E01's auxiliary target is the Google lexicon pronunciation associated with Latin inputs; do not describe it as audio supervision.

**R6 — Architecture/history and existing English phone input**

https://github.com/richardcheam/khmer-transliterator/blob/b3f2096c94c13d8e87c5a771fbc79af623cc1c89/docs/TRANSLITERATOR.md

https://github.com/richardcheam/khmer-transliterator/blob/b3f2096c94c13d8e87c5a771fbc79af623cc1c89/src/khmer_translit/en_g2p.py

Describes the small Transformer, the existing CMUdict phone-aware model E, routing, reranking and history. The sentence calling E01 phone-aware conflicts with R5 and requires code/data tracing. Existing model E is evidence of an implemented pathway, not an eligible clean baseline by itself.

**R7 — E01 results**

https://github.com/richardcheam/khmer-transliterator/blob/b3f2096c94c13d8e87c5a771fbc79af623cc1c89/experiments/E01-unified/RESULTS.md

Three paired seeds: +4.67 pp mean reranked pick agreement; registered 5 pp adoption gate missed; conditional word-bootstrap interval [+2.0,+7.6]; raw gains differ from reranked gains; approximately 7% additional compute. The reserve remained sealed. These are historical reference-agreement results, not an independent verification of our revised phonetic target.

**R8 — Historical owner-pick record**

https://github.com/richardcheam/khmer-transliterator/blob/b3f2096c94c13d8e87c5a771fbc79af623cc1c89/experiments/E00-history/reports/translit_pick.json

Records 300 choices, ranks and selected Khmer outputs. Useful evidence of annotation procedure and candidate anchoring; not a comprehensive enumeration of all acceptable forms and not a fresh independent final set. Do not redistribute personal notes unnecessarily.

## Primary research and resource sources

**P1 — He & Cohen (AACL 2020), English-to-Chinese Transliteration with a Phonetic Auxiliary Task**

Paper: https://aclanthology.org/2020.aacl-main.40/

PDF: https://aclanthology.org/2020.aacl-main.40.pdf

Supports: pronunciation-related auxiliary supervision can help transliteration; multiple spellings complicate exact-match evaluation. It studies English–Chinese, not Khmer. It does not demonstrate our phone-input/P2G-transfer recipe. Read its actual target task before claiming our E01 replicates it exactly.

**P2 — Xue et al. (TACL 2022), ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models**

https://aclanthology.org/2022.tacl-1.17/

Supports: byte-level pretrained models can perform strongly on spelling/pronunciation-sensitive tasks, with computational trade-offs. Does not establish that a large ByT5 deployment fits this laptop or that pretraining guarantees Khmer loan spelling. Comparison option only; do not add a large model download/training arm automatically.

**P3 — Zhu, Zhang & Jurgens (Interspeech 2022), ByT5 model for massively multilingual grapheme-to-phoneme conversion**

Paper: https://www.isca-archive.org/interspeech_2022/zhu22_interspeech.html

Authors' code: https://github.com/lingjzhu/CharsiuG2P

Supports: multilingual G2P transfer and available pronunciation models. It is a possible English G2P provider, subject to local version, model size, data/model notices and budget. It does not supply English→Khmer spellings or independent gold evaluation. Verify actual language coverage rather than assuming Khmer support from “100 languages.”

**P4 — Shurtz, Richardson & Richardson (LREC 2026), MekongPhon**

Paper: https://aclanthology.org/2026.lrec-1.129/

PDF: https://aclanthology.org/2026.lrec-1.129.pdf

Authors' repository: https://github.com/byu-matrix-lab/MekongPhon

Authors' G2P model card: https://huggingface.co/byumatrixlab/khmer-ipa

Supports: a Khmer pronunciation lexicon and automatically expanded orthographic–IPA corpus. The seed contains 8,026 Khmer word–IPA pairs; 1.3M refers to sentence pairs produced with lexical expansion, not independent new loanword labels. The model direction is Khmer spelling→IPA. Reversing the data for P2G is our proposal, and inversion is ambiguous. The model card declares Apache-2.0; this does not establish the precise license of every underlying dictionary/corpus item. The repository itself could not be fetched during preparation; verify files and rights before use. Seed dictionary age and native-word coverage may limit modern English loans.

**P5 — Ngo et al. (Interspeech 2015), Phonology-augmented statistical transliteration for low-resource languages**

Paper: https://www.isca-archive.org/interspeech_2015/ngo15_interspeech.html

PDF: https://www.isca-archive.org/interspeech_2015/ngo15_interspeech.pdf

Supports: incorporating target syllable/phonological structure can help low-resource transliteration in English→Vietnamese experiments. Do not transfer Vietnamese constraints or headline gains to Khmer. Inspiration for later sound-adaptation research; not an additional required arm.

**P6 — Mortensen et al. (COLING 2016), PanPhon**

https://aclanthology.org/C16-1328/

Supports: representing IPA segments using articulatory features, with evaluated benefits in the paper's tasks. Feature-aware English→Khmer adaptation is a hypothesis, not a demonstrated result. Not required for this first bounded experiment.

**P7 — Dror et al. (ACL 2018), statistical significance; Zhu et al. (AACL 2020), NLPStatTest**

https://aclanthology.org/P18-1128/

https://aclanthology.org/2020.aacl-demo.7/

Support: selecting paired statistical tests appropriately, reporting effects and assessing power. They do not validate the references or remove uncertainty from using only three training seeds. The handoff's 5 pp threshold, 80% power target and bootstrap settings are project protocol decisions, not numerical prescriptions quoted from these sources.

**P8 — CMU Pronouncing Dictionary, maintained resource**

https://github.com/cmusphinx/cmudict

https://github.com/cmusphinx/cmudict/blob/master/README

Supports: English pronunciation lookup, including alternatives. It represents US English, can contain errors and omissions, and has its own redistribution notice. It supplies English phones, not adapted Khmer phones or target spellings. Pin the actual local snapshot and notice.

**P9 — Shurtz et al. (MRL 2025), When Scripts Diverge**

https://aclanthology.org/2025.mrl-main.22/

Related evidence for combining orthographic and phonetic representations in a multilingual translation setting including Khmer. It is translation research, not direct evidence of performance for this word-transcription product. Do not conflate language relatedness or translation gains with our transliteration problem.

## Claims that are proposals, not sourced findings

- The three-arm comparison and phone-to-Khmer decoder transfer are this project's proposed experiment.
- Seven supervised runs, P2G replay schedule, CPU/download/time caps and runtime targets are implementation defaults chosen here.
- One phonetic policy rather than translation/conventional-name modes is the owner's clarified product requirement.
- Independent qualified references may be unavailable; this package does not certify they exist.
- No breakthrough, specific gain, package quality or release eligibility has been demonstrated by writing this handoff.

## Verification output schema

For every source used, record `source_id`, `url`, `checked_at`, `local_path_or_snapshot`, `hash`, `supports`, `does_not_support`, `permission_evidence`, `used_for`, `availability`, `unresolved`. Preserve this ledger with the final report. If a source is not used, say why rather than implying the entire bibliography was implemented.
