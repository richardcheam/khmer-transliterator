# Research protocol

How every experiment in this repository is run, so that results are trustworthy, comparable and reproducible.
It writes down practice the project already followed by hand (D133–D174) and adds the tooling that enforces it.

## The lifecycle of an experiment

```
 1 PLAN        experiments/Exx-name/PLAN.md + config.yaml   committed BEFORE any training
 2 RUN         training script --config ...                  each run: config, provenance, metrics → run folder + W&B
 3 EVALUATE    frozen benchmark (khmer_translit.benchmark)   results.json, written by the experiment's score command
 4 RESULTS     RESULTS.md                                    applies the plan's rule exactly as written
 5 DECIDE      docs/DECISIONS.md entry; owner A/B if served  only the owner switches what is served
```

1. **Plan** (copy `experiments/TEMPLATE/`). Write down, before any training:
   - the question and the hypothesis;
   - the data, and which sets are held out;
   - the arms and what differs between them (one change at a time against a control);
   - seeds (paired: arms share data order and initialisation per seed);
   - the primary measure, the guard measures, and the **adoption rule** as numbers;
   - the limitations already known.

   Commit `PLAN.md` and `config.yaml` together. The commit date is the registration: the git history then shows
   the rule existed before the results.
2. **Run.** Training reads only `config.yaml`; a key missing from it, or unknown to the script, is an error, and a
   setting the code cannot honour is refused (`check_implemented`). Each run folder gets:
   - `config.json`: the resolved config, including any `--set` overrides;
   - `provenance.json`: commit, uncommitted-change flag and diff hash, versions, input hashes;
   - `metrics.jsonl`: one record per epoch, the source of truth;
   - `summary.json`: final values.

   Records are mirrored to W&B (project `khmer-transliterator`, group = experiment id). Smoke runs with `--set`
   must use `--out` (never a registered folder) and are not results.
3. **Evaluate** on a frozen benchmark version only (`benchmarks/v1.json`). Never on the reserve. A new or changed
   test set is a new benchmark version, never an edit.
4. **Results.** Apply the adoption rule exactly. Report every seed, the paired differences and the bootstrap CI;
   report failures as plainly as successes. A missed rule is a result, not a reason to tune and retry.
5. **Decide.** Record the outcome in `docs/DECISIONS.md`. A model is served only after the owner's blinded A/B
   (speech repo: `scripts/translit_listen_clean.py`, review app). Then write its model card.

## Rules
- **No tuning on test data.** Panels, the reserve and test2 names are never used for selection. Anything inspected
  adaptively becomes development data, and the next benchmark version must replace it.
- **Leakage:** every test set (including word variants) is filtered out of every training file before training;
  the filter's counts go into the run's data manifest.
- **Noise:** one retrain moves dev-41 by about ±5 words. Claims need paired seeds (≥ 3) and an interval, not one run.
- **Reproducibility:** results come from committed code (`git_dirty: false` in provenance) and pinned data
  (`datasets/manifest.json`; `scripts/data_manifest.py check`). Every `results.json` can be regenerated from the
  exported models by the experiment's score command.
- **No recipe change after seeing results.** If a bug is found, document it, fix it, rerun the affected runs, and
  say so in RESULTS.md.
- **Compute and money:** laptop CPU. Any paid run needs the owner's explicit approval and a cost cap.

## Where things are
| What | Where |
|---|---|
| Experiments (plan, config, results) | `experiments/` (index: `experiments/README.md`) |
| Benchmark definitions (tracked) / panel files (local) | `benchmarks/v1.json` / `data/benchmark/v1/` |
| Data card, data manifest, data-build summaries | `datasets/` |
| Model cards (served models) | `model_cards/` |
| Run folders (weights, records) | `models/<experiment runs_dir>/<run>/` (not in git) |
| Tooling | `src/khmer_translit/experiment.py`, `benchmark.py`; `scripts/evaluate.py`, `data_manifest.py` |
| W&B | entity `richardcheam912-ensiie`, project `khmer-transliterator` (E01's training runs are in the older project `khmer-pocket-tts`, group `translit_unified_d173`) |
