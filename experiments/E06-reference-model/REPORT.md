# E06 — Reference model: report

**Status:** done 2026-10-27. Registered at `6a3e382`, after the feasibility check and before adaptation and
generation. Previous results, checkpoints, benchmarks, the reserve, the product default (E02 `letters_s0`) and
serving are unchanged.

## In plain words
We tried to get a "stronger teacher" for English → Khmer spelling.
- **Gemini:** there is no authorised free way to call it from here, so it was not used.
- **ByT5-small:** an open model pretrained on web text in about 100 languages, Khmer included. We adapted it
  with one registered, light recipe (LoRA, 1,000 updates, 37 minutes on the laptop).

The adapted ByT5 did **not** learn the task. It writes almost the same Khmer word whatever English word it is
given:

| English | Letters model (E02 default) | E04 letters G_s0 | ByT5 + LoRA |
|---|---|---|---|
| elect | អេឡិច | អេឡិកត | ហ្វេន |
| holst | ហូលស្ត៍ | ហូលស្ត | ហ្វេន |
| berm | ប៊ឺម | ប៊ែម | ហ្វេន |

So no teacher passed, distillation was skipped, and the read-only search for existing English–Khmer pairs in
local text found almost nothing new. Reliable supervision and evaluation are still the blocker.

## Results by question
| Question | Result |
|---|---|
| Operational feasibility | **Yes for ByT5-small on this laptop:** 1.12 GiB download, 3.5–3.9 GB peak RSS, 1.9 s per update, 0.61 h for 1,000 updates, about 0.13 s per word at beam 4. **Route A (Gemini): unavailable.** |
| Reference agreement (level 2, historical E03 eval, unreviewed) | ByT5 0.0% top-1 and 0.0% top-4, vs G 40.0–42.7% top-1 and 64.6–66.8% top-4. ByT5 − mean G: −41.3 points [−47.1, −35.7]. Dev (selection set): 0.4%. |
| Independently supported phonetic quality | **None:** no independent judgments exist. The blinded queue is prepared, not collected. |
| Teacher advantage | **`teacher_advantage_unresolved`** (no judgments). On every proxy, ByT5 is far below the baseline. |
| Student-transfer evidence | **None:** distillation skipped, because the gate did not pass. 0 of 6 student runs used. |
| Deployment readiness | **Not applicable.** Nothing changes in the product or serving. |

## What ran, and the budget
| Item | Used | Bound |
|---|---|---|
| Downloads | 1.12 GiB model + 62 MiB packages ≈ 1.18 GiB | 2 GiB |
| ByT5 adaptation | 1 recipe, 1 run, 1,000 updates, 0.61 h | 1 recipe, ≤ 1,000 updates / 4 h |
| Aggregate training | 0.62 h (feasibility 5 updates + adaptation) | 12 h |
| Inference and analysis | predictions 118 s; baselines/compare/queue/gate/discovery 153 s; conditioning check about 2 min | — |
| Instruction-model requests | 0 (route unavailable) | 1,100 |
| Student runs | 0 | 6 |

Everything ran on CPU, one model job at a time, with 6 threads.

## Route audit (`SOURCE_VERIFICATION.md`)
- **Route A (Gemini): unavailable.**
  - There is no API key and no Gemini client. Scraping the chat interface, creating an account or paying is not
    allowed.
  - The earlier Gemini use was the owner's manual LLM-drafting rounds (`data/synth/`). Those are historical
    material and are not relabelled as distillation.
  - The frozen prompt is saved in `prompts/route_a_prompt.txt` but was never sent.
- **Route B (ByT5-small): feasible.**
  - Pinned revision `68377bdc…`, Apache-2.0, sha256 verified.
  - The card states the model must be fine-tuned first. The unadapted zero-shot outputs were malformed UTF-8
    (`reports/feasibility.json`), which is not an assessment of transfer.

## Adaptation (`models/e06/byt5_lora/`, `reports/feasibility.json`, `reports/conditioning_check.json`)
- **Recipe, as registered:**
  - LoRA r 16 / alpha 32 on q, k, v, o of all 20 attention blocks;
  - 2,375,680 trainable of 302,013,440 parameters;
  - AdamW lr 1e-3, effective batch 16, 50 warm-up updates, then linear decay;
  - 1,000 updates (8 passes over the 2,255 rows); 0 oversized rows rejected.
- **Curve** (`metrics.jsonl`; W&B run `byt5_lora`):
  - train-val loss per target byte: 1.009 (update 100), 0.861 (300), 0.846 (500), 0.797 (800), 0.789 (1,000);
  - greedy dev exact match: 0 at every checkpoint. The selection rule therefore took update 1,000 (lowest loss).
- **Does it use the input?** Mostly not. Post-hoc check, teacher-forced loss per target byte:

  | Rows | Correct input | Shuffled input |
  |---|---|---|
  | train-val | 0.789 | 0.851 |
  | rows it trained on | 0.743 | 0.783 |

  The decoder learned a Khmer-word prior: 53 of 56 pool top-1s are Khmer lexicon words, mostly ហ្វេន. The input
  barely changes the output.
- **Implementation checks:**
  - loss fell from 7.8 (unadapted) to 0.79;
  - beam scores equal teacher-forced sums (max gap 3e-6);
  - no malformed or empty outputs after adaptation;
  - batch padding and labels are covered by tests.

  Not done: an overfit check on a tiny subset, which would rule out a subtle bug more directly. A tiny-subset
  overfit was not part of this phase's registered work.
- **Interpretation:** this is an **inadequate adaptation, not evidence that pretrained transfer is ineffective.**
  Kirov et al. (2024) fine-tuned ByT5-*base* in full for 50,000 steps at batch 64, and found fine-tuning sensitive
  to noisy data. Our run was attention-only LoRA on the small model for 1,000 steps at batch 16, on 2,255 mined
  pairs. Per the instruction, no larger model was downloaded and no second recipe was tried.

## Fresh pilot pool and overlap audit (`pool.jsonl`, `reports/overlap_audit.json`)
- **Pool:** 56 words in sha256 order, selected before any output.
  - Primary (48): ordinary, name, difficult, short, long, dictionary miss; 8 each.
  - Ambiguous (8): reported separately.
- **Freshness:** no word shares a lexical-family stem with any of 168,338 historical Latin tokens. Those tokens
  cover:
  - E02–E05 data, mined pairs, the LLM-drafting rounds, owner picks, names and romanised sets;
  - the speech repo's loanword list;
  - every report, doc and handoff file;
  - the Latin inputs only of the benchmark panels and the reserve.

  E02's final references were never opened. 142 candidates were skipped as historical families during
  selection.
- **Caveats:**
  - SCOWL's capitalised entries include derived words (*stalinist*, *anzac*) besides names;
  - the "dictionary misses" are mostly inflections absent from CMUdict (*frothiest*, *alighted*);
  - *massenet* is a French name with an anglicised CMUdict reading.
- **References:** none exist for these words at evidence level 1 or 2, by construction. They have only level-3
  diagnostics and the review queue.

## Level-3 diagnostics (automated proxies, not correctness; `results.json`)
| Pool, 56 words | ByT5 | G_s0 | G_s1 | G_s2 | Default E02 |
|---|---|---|---|---|---|
| Orthographic rules OK (top-1) | 56 | 56 | 56 | 56 | 56 |
| Consonant screen level 0, 48 words with phones | 0 | 25 | 22 | 23 | 37 |
| Consonant screen 3+ (gross mismatch) | 27 | 3 | 3 | 3 | 0 |
| Top-1 is a Khmer lexicon word | 53 | 8 | 11 | 12 | 6 |
| Malformed / empty / refused | 0 | 0 | 0 | 0 | 0 |
| Top-1 equal to ByT5's | — | 0 | 0 | 0 | 0 |

- **Orthography rules:** they accept every system's output, so they do not separate the systems here.
- **Consonant screen:** it is the frozen E02-diagnostics proxy. It agrees with the reference finding: ByT5's
  consonants rarely match the English ones.
- **Lexicon membership:** for ByT5 this signals the input-independent prior, not established loanwords.
- **Same picture on the historical eval (245 items):** screen level 0 for 2 ByT5 top-1s against 206–218 for the G
  seeds.

## Review queue (prepared, not collected)
- **Files:**
  - `data/e06/review/review_queue.tsv` and `review_queue.json`, with a separate `origin_key.json`;
  - git-ignored, local only, not published, nobody contacted;
  - hashes in `reports/review_queue_manifest.json`.
- **Content:** 56 words, 643 candidates. Deduplicated ByT5, G_s0 and default-E02 top-4 candidates in random order,
  with the origin hidden.
  - The policy text and the default pronunciation are given per word; the ambiguous words list their other
    readings.
  - Several candidates may be acceptable; "uncertain" is allowed.
- **Judgments:** none exist.

## Teacher-quality gate (`reports/gate.json`)
- **Procedure:** frozen in the registration before any judgment. Paired ByT5 − default-E02 top-1 acceptance on
  ≥ 40 primary words, a 10,000-resample bootstrap, and a positive lower bound with the advantage spread over
  ≥ 3 categories.
- **Result:** `teacher_advantage_unresolved`, because no independent judgments exist.
- **Proxies:** they would not have helped ByT5. Unreviewed outputs and automated proxies are not allowed to
  unlock distillation anyway.

## Distillation (Task 8): skipped
No teacher passed the gate. No synthetic targets were generated and no student was trained.

## Constrained-discovery feasibility (Task 9; `reports/discovery/audit.json`, `inspection.json`)
- **Words:** 200 fresh words, frozen before any candidate was generated.
- **Candidates:** ByT5 and G_s0 top-4 (≥ 3 characters), used only as search constraints.
- **Corpora:** 32,587 kmwiki pages and 1,586,460 FineWeb-2 documents, scanned read-only in 138 s.

| Outcome | Words |
|---|---|
| English word never appears | 65 |
| English word appears, no candidate in those documents | 112 |
| Candidate in the same document | 21 |
| Candidate within 60 characters of the English word | 2 |

- **My inspection** of the 23 matches (not independent review):
  - 3 are plausible existing targets:
    - *lyle* → លីល, in "ថេត និង លីល (Tate & Lyle)";
    - *lindy* → លីនឌី, in "(Lindy Elkins-Tanton)";
    - *holistic* → ហូលីស្ទីក, same document only.
  - 20 are spurious: mostly ByT5's prior strings such as ហ្វេន, or common Khmer words such as ដែល "which".
- **Yield:** about 1.5% of sampled words. The median word appears in only 2 documents.
- **Reading:**
  - The plausible hits are multi-word bracketed names, which the existing miner probably skips because it
    expects one-word brackets.
  - Existing-target discovery in these local corpora is **not a credible source of additional supervision at
    useful scale**.
  - Nothing found was kept as a label, and no target was invented.

## Conclusions and limitations
- **One recipe only:** a pretrained byte model adapted this lightly gives no useful English → Khmer knowledge
  here. It ignores the input. This is an adaptation failure under one registered recipe. It does not establish
  that a properly fine-tuned ByT5 (base size, full fine-tuning, many more steps, cleaned data) could not help.
  That would need GPU-scale compute, outside this phase's bounds.
- **No teacher and no student:** no teacher passed and no student was trained. The current letters model remains
  the practical baseline, and it beats ByT5 on every proxy.
- **Instruction models:** they remain untested, because no authorised free route exists. The owner's favourable
  Gemini impressions are still unverified.
- **References:** all reference agreement is against mined, unreviewed spellings on a historical set. The pool
  has no references, so its diagnostics are proxies.
- **Next step:**
  - unchanged from E03–E05: independent review of a reference set is the prerequisite, and the E05/E06 queues are
    ready for it;
  - a possible cheap extension of the existing miner: multi-word bracketed names (as in the *lyle* and *lindy*
    hits). That would be a separately registered mining task, not done here.

## Reproduction
```bash
cd ~/projects/khmer-transliterator && export CUDA_VISIBLE_DEVICES=
uv pip install --python .venv/bin/python transformers==5.19.0 safetensors==0.8.0        # research-only extras
R=https://huggingface.co/google/byt5-small/resolve/68377bdc18a2ffec8a0533fef03b1c513a4dd49d
mkdir -p models/e06/byt5-small && for f in config.json generation_config.json pytorch_model.bin; do
  curl -fL -o models/e06/byt5-small/$f $R/$f; done                                       # sha256 checked on load
.venv/bin/python scripts/e06.py pool               # pool + overlap audit (no outputs)
.venv/bin/python scripts/e06_byt5.py smoke         # feasibility
.venv/bin/python scripts/e06.py discovery_words    # freeze the 200 discovery words
.venv/bin/python scripts/e06.py register           # registration.json (committed at 6a3e382)
.venv/bin/python scripts/e06_byt5.py train         # resumes from models/e06/byt5_lora/last.pt
.venv/bin/python scripts/e06_byt5.py predict
.venv/bin/python scripts/e06.py run                # baselines → compare → queue → gate → discovery (skips done steps)
.venv/bin/python -m unittest tests.test_e06
```

## Artifacts
- **Plan and registration:**
  - `PLAN.md`, `config.yaml`, `registration.json` (code, data, checkpoint, pool, prompt, recipe and gate hashes);
  - `SOURCE_VERIFICATION.md` (routes, downloads, literature, licences);
  - `prompts/route_a_prompt.txt`.
- **Reports:**
  - `reports/feasibility.json`, `reports/overlap_audit.json`, `reports/conditioning_check.json`;
  - `reports/gate.json`, `reports/review_queue_manifest.json`;
  - `reports/discovery/{words,audit,inspection}.json`.
- **Data, predictions and results:**
  - `pool.jsonl`;
  - `predictions/byt5_{dev,eval,pool,discovery}.jsonl` and `predictions/{G_s0,G_s1,G_s2}_{pool,discovery}.jsonl`;
  - `predictions/default_e02_pool.jsonl`;
  - `results.json`.
- **Models** (local, git-ignored):
  - `models/e06/byt5-small/` (base);
  - `models/e06/byt5_lora/`: `best.pt` sha256 `609dd835…`, `metrics.jsonl`, `done.json`, `last.pt`,
    `provenance.json`.
- **Review files** (local, git-ignored): `data/e06/review/`.
- **Code:** `scripts/e06.py`, `scripts/e06_byt5.py`, `tests/test_e06.py`. Decisions: `docs/decisions/e06-reference-model.md`.
- **Progress:** `state/e06/status.json`, `PROGRESS.md`.
