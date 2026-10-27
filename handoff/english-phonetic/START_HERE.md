# Claude handoff: English pronunciation to Khmer script

Prepared 27 October 2026. Repository: https://github.com/richardcheam/khmer-transliterator

## One instruction to paste into Claude

> Work in the `richardcheam/khmer-transliterator` repository. Read `START_HERE.md`, `DESIGN.md`, `AUTONOMY.md`, `IMPLEMENTATION_PLAN.md`, all `tasks/` files, `SOURCES.md`, `QA.md`, and the templates in this handoff. Execute the complete approved project in order, through data and benchmark qualification, implementation, bounded experiments, evaluation, packaging and final verification. This is an execution instruction, not a request for another plan or approval round. Make ordinary engineering decisions yourself, record them, fix routine failures, and advance automatically after each completed task. Do not stop to ask whether to continue, whether to run the next seed, or whether to finish packaging. Read current repository instructions and verify the evidence pointers against actual code and local data. Preserve E01, benchmark v1, old models and the sealed reserve. Use the handoff's explicit scientific fallbacks when reliable labels or resources are unavailable; never invent verification or weaken the task definition to force a success. Complete every independent deliverable even if an experiment or quality claim is blocked. Stay within the compute and action boundaries in `AUTONOMY.md`. Keep durable progress and resume commands so context compaction or process interruption does not restart completed work. End with the final report and working product artifacts, not a menu of proposed next steps.

## Placement and read order

Extract this folder inside the repository, for example as `handoff/english-phonetic/`. The paths named by implementation tasks are relative to the repository root, not this handoff folder. Do not overwrite an existing handoff folder; choose a new dated path.

1. `DESIGN.md`: the agreed product and research scope.
2. `AUTONOMY.md`: authority, budgets, continuation and stopping rules.
3. `IMPLEMENTATION_PLAN.md`: task order, interfaces and integration checks.
4. `tasks/01-audit.md` through `tasks/06-release.md`: execution detail.
5. `SOURCES.md`: primary papers and repository evidence, with limitations.
6. `QA.md`: answers to expected research and implementation questions.
7. `templates/`: records to instantiate, not existing experimental results.

## What has and has not been done

This folder is an instruction package. It does not contain newly trained weights, a newly validated benchmark, or a claim that the proposed methods improve Khmer transliteration. The source audit used public repository commit `b3f2096c94c13d8e87c5a771fbc79af623cc1c89`. The actual benchmark TSVs and model files were not available in that public checkout; Claude must inspect them locally.

The approved target is English words/names pronounced as English, written phonetically in Khmer script. Translation and arbitrary romanised Khmer chat are outside this experiment. A readable approximation is expected where Khmer cannot represent an English sound exactly. A conventional Khmer equivalent is not automatically a correct answer.

## Completion has two independent dimensions

- Product: installable package, usable CLI/API, a runnable checkpoint where data/resources permit, deterministic export, tested examples, documentation and a reproducible artifact bundle.
- Evidence: `supported_improvement`, `meaningful_gain_not_supported`, or `inconclusive`, with benchmark qualification and uncertainty. A working product does not imply a validated quality improvement.

If qualified independent references cannot be established without new human work, finish the runnable research-preview product and the audit. Mark the independent quality claim `inconclusive`. Do not fabricate a gold benchmark, silently count unreviewed LLM outputs as gold, or ask the owner to type hundreds of labels.
