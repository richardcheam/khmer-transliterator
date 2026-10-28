# E07 decisions log (ByT5 diagnosis)

Results summary: `docs/DECISIONS.md` (D184).

| # | Date (UTC time) | Decision | Why |
|---|---|---|---|
| 1 | 2026-10-27 | Id E07, folder `experiments/E07-byt5-diagnosis/` | Next unused id |
| 2 | 2026-10-27 | Research dependencies pinned in `research/requirements-byt5.txt` + a full freeze `research/requirements-byt5.lock.txt`; not in `pyproject.toml`/`uv.lock` | A separate research specification. Changing the lock would need registry resolution and could reach serving; `uv sync` stays as it was |
| 3 | 2026-10-27 | PEFT not installed; the repository's `LoRALinear` is kept and checked against PEFT's documented defaults | Avoid a new dependency. Audit check 8 shows that the zero-initialised adapter gives exactly the base logits |
| 4 | 2026-10-27 | E06 code and outputs unchanged; E07 reuses E06's byte encoding and adds its own generation (121 new tokens, `no_eos`) | Preserve E06's reproducibility; the latent EOS-room defect is fixed in E07 only, with a regression test |
| 5 | 2026-10-27 | Tiny configuration: attention+FFN LoRA r16 α32, no dropout, lr 1e-3 constant, batch 16; stop at 32/32 | Memorisation needs capacity and no regularisation. It tests the pipeline, not a new learning rate |
| 6 | 2026-10-27 | Tiny sample kept as selected, including noisy targets (e.g. *bracket* → ប្រកួត) | The selection rule was fixed before any model output; it is a software diagnostic, not a quality sample |
| 7 | 2026-10-27 20:30 | Full recipe registered after the gate passed on attempt 1 (attempt 2 unused) | PLAN §4 |
| 8 | 2026-10-27 20:45 | Post-registration analysis amendment, before any E07 prediction or eval scoring: add the stratum "eval word in the product default's E02 training data" | 110 of 228 primary eval words occur in `data/e02/train.tsv`, so the default's eval score is partly a training-set score. This adds a reporting stratum only; the recipe, selection and primary metrics are unchanged |
| 9 | 2026-10-27 | Commits omit Claude attribution lines | Owner's standing rule |
| 10 | 2026-10-28 | Wall-time cap (12 h from 20:17 UTC) exceeded: the session was idle 01:11–08:18 UTC after training finished; prediction, analysis and cost (about 10 min of computation) were run afterwards | The phase instructions say to complete the analysis when a cap is reached; no training after the cap; reported in REPORT §9 |
