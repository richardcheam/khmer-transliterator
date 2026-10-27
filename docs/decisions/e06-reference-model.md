# E06 decisions log (reference model)

Results summary: `docs/DECISIONS.md` (D183).

| # | Date | Decision | Why |
|---|---|---|---|
| 1 | 2026-10-27 | Id E06, folder `experiments/E06-reference-model/` | Next unused id |
| 2 | 2026-10-27 | Route A (Gemini) unavailable; no request sent; prompt frozen for the record only | No explicitly authorised free programmatic interface (no key, no client); chat scraping, account setup and paid calls are not allowed |
| 3 | 2026-10-27 | Route B ByT5-small at revision 68377bdc; only config, generation config and `pytorch_model.bin` downloaded (1.12 GiB, sha256 verified) | No safetensors copy exists; TF/Flax duplicates not needed |
| 4 | 2026-10-27 | transformers 5.19.0 installed with `uv pip` (62 MiB), not added to `pyproject.toml`/`uv.lock` | The offline cache lacked a compatible `regex`; universal lock resolution needs registry metadata; the research dependency must not reach serving |
| 5 | 2026-10-27 | LoRA implemented in `scripts/e06_byt5.py` (no `peft`) | One small class instead of another dependency |
| 6 | 2026-10-27 | ByT5 interface letters-only (no pronunciation) | Isolates pretraining/capacity from extra input; E03–E05 found phones add little; works for dictionary misses |
| 7 | 2026-10-27 | Fresh = stems disjoint from all historical Latin tokens, including docs/reports (inspected examples); benchmark panels and the reserve read for Latin inputs only; E02 final references never opened | The instruction's overlap audit, within the evaluation guard |
| 8 | 2026-10-27 | Pool: 8 per category in sha256 order; SCOWL capitalised entries as "names" (some are derived adjectives, e.g. *stalinist*); dictionary misses are mostly inflections | Reproducible selection fixed before outputs; caveats reported rather than hand-picked away |
| 9 | 2026-10-27 | Gate baseline = product default E02 letters_s0; bootstrap over pool words (one family each) | The practical baseline; pool words were selected one per family |
| 10 | 2026-10-27 | Discovery words frozen before candidate generation; candidates ≥ 3 characters only | Prevent selection on outcomes; one- and two-character Khmer strings match nearly every document |
| 11 | 2026-10-27 | Commits omit Claude attribution lines | Owner's standing rule |
