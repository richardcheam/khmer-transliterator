# References consulted (2026-10-31) and how they apply

| Reference | What it says (as checked) | How it applies here |
|---|---|---|
| Grundkiewicz & Heafield 2018, "Neural Machine Translation Techniques for Named Entity Transliteration", NEWS 2018, pp. 89–94 ([W18-2413](https://aclanthology.org/W18-2413/)) | NMT-style encoder-decoders with dropout, ensembling, back-translation and **rescoring with right-to-left models**; first in several NEWS 2018 tracks | Motivates the candidate-coverage check: rescoring can only help if acceptable spellings are already in the n-best list. **No ranker is built or claimed**; the check measures presence only |
| Chen et al. 2018, "NEWS 2018 Whitepaper" ([W18-2408](https://aclanthology.org/W18-2408/)) | Evaluation over reference lists that may hold several accepted transliterations: ACC ("matches one of the references"), mean F-score, MRR, MAP_ref | The human tests judge each spelling, so several can be acceptable. Against the single mined reference only ACC and CER are reported. Coverage counts a word when any of its candidates is confirmed acceptable |
| Xue et al. 2022, "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models", TACL 10:291–306 ([2022.tacl-1.17](https://aclanthology.org/2022.tacl-1.17/)) | Byte-level T5 is more robust to noise and better on spelling- and pronunciation-sensitive tasks | The reason for this base. It isn't evidence of quality on Khmer |
| Hugging Face Hub, "Model Cards" ([docs](https://huggingface.co/docs/hub/model-cards)) | `base_model` with `base_model_relation: adapter`; `library_name` should be set explicitly for non-Transformers models; a custom licence uses `license: other` + `license_name` + `license_link` or a `LICENSE` file | Card metadata: `base_model: google/byt5-small`, `base_model_relation: adapter`, `library_name: khmer-translit`, `inference: false`; licence left out until decided |
| PEFT, "PEFT checkpoint format" ([docs](https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint)) | PEFT checkpoints are `adapter_model.safetensors` with keys like `base_model.model.….lora_A.weight` and an `adapter_config.json` with `peft_type`, `target_modules`, and so on; loaded with `PeftModel.from_pretrained` | **This adapter isn't in PEFT format** (keys `….A`/`….B`, a project config), and PEFT isn't installed or tested. The card tells users to load it with `khmer_translit.byt5` and doesn't promise PEFT or AutoModel compatibility. A conversion would be a separate, verified step |
| `google/byt5-small` model card (Hub API, revision `68377bdc…`) | `license: apache-2.0`; pre-trained on mC4 only; "has to be fine-tuned before it is useable" | Base licence and pin |
| FineWeb-2 dataset card (revision `af9c1333`) | ODC-By 1.0; "also subject to CommonCrawl's Terms of Use" | `NOTICES.md` §2 |
| Common Crawl Terms of Use (updated 2024-03-07) | Crawled content "may be subject to separate terms of use"; respect third-party rights; indemnity for AI/ML use | `NOTICES.md` §2 |
| Wikimedia Terms of Use §7 | Text under CC BY-SA 4.0 and GFDL; attribution by link/URL; modified text under CC BY-SA 4.0 or later | `NOTICES.md` §1 |
| NICT ALT page | ALT Parallel Corpus under CC BY 4.0, cite Riza et al. 2016; the separate Khmer ALT annotation (`km-nova`) is CC BY-NC-SA 4.0 | `DATA_PROVENANCE.md` (only the parallel corpus was used) |

The project's own earlier evidence also applies:
- E12 found a longer schedule gave no gain (0.0 [−8.3, +8.3]).
- E13 found pronunciation input gave no gain (0.0 [−7.5, +7.5]).
- E02–E05 found phone input gave no detectable benefit for the small models.
