# E06 source and permission verification (2026-10-27)

## Reference-model routes
| Route | Checked | Finding | Status |
|---|---|---|---|
| A: Gemini (instruction model) | Environment variables (no `GEMINI_*`/`GOOGLE_*` key); the project `.env` (none); installed packages (no `google-genai` / `google-generativeai`); `handoff/english-phonetic/QA.md` ("This project makes no paid API calls and does not require Gemini availability") | No explicitly authorised free programmatic interface exists. The owner's earlier Gemini use was manual chat (`data/synth/GEMINI_PROMPT.md`, `paste_*.txt`; past LLM-drafting rounds, historical material, not distillation). Model identifier, current terms and request limits were therefore not needed and not asserted | **unavailable**; no request sent |
| B: `google/byt5-small` | Hub API (`/api/models/google/byt5-small`, revision `68377bdc18a2ffec8a0533fef03b1c513a4dd49d`, last modified 2023-01-24); model card; `config.json` | Apache-2.0. T5ForConditionalGeneration, 12 encoder / 4 decoder layers, d_model 1472, d_ff 3584, 6 heads, byte vocabulary 384 (bytes + 3). Card: "only pre-trained on mC4 excluding any supervised training … this model has to be fine-tuned before it is useable on a downstream task"; `km` in the language list. Files: `pytorch_model.bin` 1,198,627,927 B (LFS sha256 `5c5aaf56…931a`, verified locally), TF and Flax copies of the same size (not downloaded) | **feasible** |

## Downloads (budget 2 GiB)
| What | Size | Why |
|---|---|---|
| ByT5-small `config.json`, `generation_config.json`, `pytorch_model.bin` (pinned revision) | 1.12 GiB | The one open model of this phase |
| transformers 5.19.0, huggingface-hub 2.3.0, tokenizers 0.23.3, safetensors 0.8.0, regex and 13 small dependencies (`uv pip install`, not in `uv.lock`; a later `uv sync` removes them) | 62 MiB uv-cache growth | Loading and adapting T5 with PyTorch |

## Literature (abstract pages fetched this session; one full text read where noted)
| Source | Checked | What it supports | What it does not establish |
|---|---|---|---|
| Kirov, Johny et al., "Context-aware Transliteration of Romanized South Asian Languages", Computational Linguistics 50(2), 2024 | aclanthology.org/2024.cl-2.2 (abstract) and the full PDF (2024.cl-2.2.pdf, text-searched) | ByT5 fine-tuned for single-word transliteration (ByT5-*base*, 50,000 steps, batch 64, a 24-task mixture); "fine-tuning of the pretrained model was quite sensitive to outliers/noise in the training set", so they pruned ~10% of the data; with pruning ByT5 had the lowest CER in 10 of 12 Dakshina languages, without it in 4; ByT5 had the highest variance | Anything about Khmer or English → Khmer; that ByT5-*small* with LoRA and 1,000 updates is adequate |
| Hugging Face model card `google/byt5-small` | huggingface.co/google/byt5-small (pinned revision) | Requires downstream fine-tuning; byte-level input (ids = bytes + 3) | Any transliteration result |
| Kim & Rush, "Sequence-Level Knowledge Distillation", EMNLP 2016 | aclanthology.org/D16-1139 (title, authors); abstract via arXiv:1606.07947 (the Anthology page carried no abstract text) | Sequence-level KD for NMT; student 10× faster with little loss; better than a baseline trained without KD (+4.2/+1.7 BLEU greedy/beam) | Transfer to word-level transliteration; that a teacher's errors are harmless |
| Dankers & Raunak, "Memorization Inheritance in Sequence-Level Knowledge Distillation for NMT", ACL 2025 (short) | aclanthology.org/2025.acl-short.61 (abstract) | Students "inherit both their teachers' superior performance and their fault modes"; more memorisation and hallucination than baselines; caution and monitoring | Results for transliteration |
| Upadhyay, Kodner & Roth, "Bootstrapping Transliteration with Constrained Discovery for Low-Resource Languages", EMNLP 2018 | aclanthology.org/D18-1046 (abstract) | Discovery (picking a plausible transliteration from a given list) is easier than generation; bootstrapping with constrained discovery from ~500 examples; nine languages; foreign script → English | Khmer, English → native-script direction, or gold labels |
| Madhani et al., "Aksharantar", Findings of EMNLP 2023 | aclanthology.org/2023.findings-emnlp.4 (abstract) | 26M pairs for 21 Indic languages mined from monolingual and parallel corpora plus human annotation; a 103k-pair test set with native/foreign/frequent/rare analysis; IndicXlit | A Khmer model or data; IndicXlit is not a drop-in Khmer system |

## Data licences used in E06
- **Data:**
  - E03 partitions: from mined kmwiki (CC BY-SA 4.0 / GFDL) and FineWeb-2 (ODC-By 1.0, Common Crawl terms);
  - SCOWL word lists (permissive notice);
  - CMUdict (BSD-style);
  - see `data/SOURCES.md`.
- **Teacher outputs:** ByT5 outputs are research outputs of an Apache-2.0 model. They would be compatible with
  student training if a gate ever passed.
