# Exx — <title>

**Status:** planned (registered <date>, commit <hash> — fill in after committing this file, before training).
Config: `config.yaml`. Code: `scripts/<script>.py`. Results: `RESULTS.md`, `results.json`.

## Question
<One sentence. What will we know after this experiment that we do not know now?>

## Hypothesis
<What you expect and why, including what would show it wrong.>

## Data
- Training sources and sizes (files in `datasets/README.md`; hashes recorded by the run).
- Held out: benchmark <version> panels, the reserve, validation sets. Leak filter: <rule>.

## Arms
- **control:** <the baseline, usually the previous best recipe>
- **treatment:** <exactly one change>

## Training
<Everything in config.yaml, in words. Seeds: 0, 1, 2, paired.> Checkpoint selection: <validation only>.

## Primary measure and adoption rule (fixed now)
- **Primary:** <panel, metric>.
- **Adopt / advance only if:** <numbers: mean paired gain ≥ …, positive in ≥ … seeds, guards ≤ … regression>.
- Diagnostics (not part of the rule): <…>.

## Execution plan
<Pilot seed, technical checks, then the remaining seeds. What counts as a bug and what happens then.>

## Limitations stated up front
- <…>
