# Autonomous execution contract

## Continue without asking

The owner approved the project design and requested end-to-end execution. Run the task files in order. Do not ask for routine approval between audit, implementation, smoke tests, training, seeds, export, evaluation, fixing, packaging or documentation. Treat a successful task as authorization to begin the next authorized task. Do not respond only with a plan or implementation scaffolding.

Resolve routine ambiguity with a sensible conservative choice, record its reason in `docs/decisions/english-phonetic.md`, and proceed. If agent workflow skills demand another design/continuation question, the owner's explicit instruction to execute this approved scope takes precedence over routine workflow questions.

Use the current repository and its actual local paths. Preserve uncommitted work. Use an isolated branch/worktree when appropriate and safe; do not stash, reset or overwrite someone else's changes. Commit task milestones locally. A push to a dedicated feature branch is allowed if the repository's configured remote and existing credentials work; never force-push, merge into main, create a release tag, or publish to PyPI automatically. No messages to third parties. No TTS serving changes.

## Concrete bounds

| Resource | Bound |
| --- | --- |
| Runtime | Existing laptop CPU; default at most 6 training threads; cap to available cores |
| Paid APIs/cloud/GPU | No paid requests, cloud provisioning or GPU training |
| Full supervised jobs | At most 7 total attempts, including failed full training attempts |
| P2G pretraining | At most 3 jobs; each at most 10 epochs and 2 elapsed hours, whichever comes first |
| Aggregate training time | At most 96 hours summed job elapsed time, including P2G and failed runs; this is not summed CPU core time |
| Smoke jobs | At most 10 jobs, each at most 10 minutes; use a small development-only subset |
| Resource downloads | Reuse local data first; at most 2 GiB additional downloads total; no model over 500 MiB, no automatic download of a large pretrained language model |
| Parallel jobs | One training process at a time by default on the constrained laptop |
| Model/recipe variants | Exactly the registered 3 arms; no schedule sweep, new teacher data round or post-test tuning |

These are implementation defaults chosen for the handoff, not measurements of the user's machine. Enforce them in the runner. If capacity is lower, reduce concurrency, streaming buffers and preprocessing batch sizes, not registered scientific settings silently. Record any pre-training feasibility amendment before full training.

## Missing resources or labels

Try the documented local source and an eligible free source within budget. If a resource is unavailable or its rights are unclear, omit it, record the limitation and continue independent work. Do not pay for access or substitute synthetic material while calling it verified. If no valid P2G resource is available, mark that arm unavailable and execute the remaining arms; do not fabricate a replacement task.

If a frozen independent benchmark cannot be qualified, still build the source pipeline, model training, local preview checkpoint where possible, tests and artifact bundle. Training/evaluation may proceed explicitly as exploratory on suitable development data; final performance status stays `inconclusive`. Never use the E01 reserve to rescue a weak evaluation. Never recruit the owner for hundreds of new annotations.

If no eligible supervised training pairs remain after audit, deliver the complete runnable training/inference tooling with honest `weights_unavailable` status rather than packaging an undocumented historical model or random weights. This is a real data blocker, not successful completion of a trained product; state it clearly in the final report.

## Long jobs, resumption and progress

Implement a durable runner with explicit phase state, atomic status writes, exit codes, checkpoints, input hashes and a one-command resume. Use the session's supported background-job monitoring; do not finish the assistant turn merely because a long run was launched. Give concise progress updates while working. Do not repeatedly consume model turns polling at a high frequency. A runner must not execute new experiments beyond the fixed queue.

Persist `state/english-phonetic/status.json` and `state/english-phonetic/PROGRESS.md` after each milestone and before context compaction. Record current branch, HEAD, dirty status, phase, exact job, PID/job ID, log path, budget spent, outputs, next command and unresolved blockers. Resume validates hashes, detects a live job and avoids duplicate launches. Never kill an unrelated process. Distinguish interrupted/resumable jobs from corrupt jobs.

Context compaction is not a new approval boundary. Resume the existing plan. An external session termination cannot be guaranteed away: leave state and an exact resume command, and never claim execution continued after the environment stopped it.

## Stop only when necessary

For scientific or resource unavailability, take the documented fallback and finish other tasks. Ask the owner only if an indispensable action would require new credentials, paid spending, destructive operations or a changed goal and no authorized alternative exists. Bundle the specific blocker and completed independent work in one concise message. Do not turn ordinary negative experimental results into a request for another variant.

## Definition of final completion

Complete the checklist in `tasks/06-release.md`, verify the installed wheel and actual artifacts, record any unmet dimension honestly, and provide one self-contained final report. Include the product location, install/example/resume commands, model and evidence status, results and confidence limitations, compute spent, tests and any real blocker. Do not end with “would you like me to continue?”
