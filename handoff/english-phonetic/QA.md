# Q&A for the owner and Claude

## What exactly are we building?

A written-English pronunciation rendered in readable Khmer script. The input is an English word/name, not arbitrary romanised Khmer. The output preserves sound rather than translating meaning. This is the owner's clarified policy in `DESIGN.md`, not a claim drawn from a paper.

## Why not automatically use the familiar Khmer word for coffee?

Because a familiar local equivalent can follow a different pronunciation. It is accepted only if compatible with the intended phonetic policy. We are avoiding inconsistent targets, not forbidding established spellings. Phonetic approximation is necessarily language-specific; P5 motivates modeling target phonology without supplying Khmer rules.

## Does this require the owner to type hundreds of examples?

No. Reuse documented eligible resources and existing independent references. If independent quality evidence is unavailable, finish the runnable research preview and report the gap. Neither an LLM nor this instruction package can make unavailable gold labels magically reliable. R3/R8 explain limitations of the existing owner-preference data.

## Is English pronunciation input genuinely new here?

It already exists in historical model E. E01 documentation calls E01 phone-aware, but the inspected E01 code sets `phones=False`. Claude must trace actual tensors/checkpoints and correct stale descriptions before framing the new ablation. The experiment is a clean controlled comparison, not an unsupported novelty claim. See R5/R6.

## How does this differ from E01's side decoder?

E01 predicts a pronunciation target from a Latin input as an auxiliary task; its code does not indicate audio training. The new phone arms receive English source pronunciation at inference, while C additionally learns to write Khmer from Khmer phones. These are distinct mechanisms. P1 supports auxiliary phonetic supervision in another language pair; it does not prove C will work.

## Why retain English letters if phones are available?

Letters retain lexical identity and conventions; pronunciation can be wrong, ambiguous or missing. Combining both lets the model use them together. This is an explicit design hypothesis to test, not a guarantee. P9 is related orthographic/phonetic-input research in translation; its task is different.

## Can Khmer phones→spelling be learned without English–Khmer pairs?

That pretraining task can use a Khmer pronunciation lexicon by reversing its pairs. English-to-Khmer sound adaptation still needs bilingual supervision or other knowledge. P4 supplies potential Khmer phonetic material; it does not remove this bridge or prove inverse spelling is unique.

## Can we reverse an existing G2P model?

No. Reverse the eligible data and explicitly train P2G. Homophones create multiple possible spellings. An accurate Khmer→IPA model is not an IPA→Khmer decoder. See P4's actual model direction.

## Is C a pure test of pretraining?

No: C is a registered transfer recipe with pretraining plus fixed P2G replay. B→C estimates that recipe's combined effect. We do not isolate pretraining and replay independently in the seven-run budget, and must say so. A later factorial ablation would be a different project.

Also, the seven-run design confirms only the selected treatment against letters. B versus C has one pilot seed, so even a confirmed C-versus-letters gain does not establish an incremental C advantage over B. State that comparison as exploratory.

## Are reference pronunciations secretly supplied to the model during testing?

Not in the primary end-to-end track. It uses the deployed default dictionary/G2P pipeline and a fixed English-reading policy. Bad provider phones remain errors. A separate supplied-pronunciation track can isolate behavior when the intended reading is given explicitly, but must not replace or be pooled with primary results. Two readings of the same spelling have distinct IDs and share one lexical split group.

## Why not train a much bigger model or start with ByT5?

First test whether supplying sound information and transferring Khmer spelling knowledge help the small model. P2/P3 make pretrained byte models credible alternatives, but their results do not establish a good CPU size/latency trade-off here. No extra large-model arm is authorized in this budget.

## Is benchmark v1 reliable?

It is versioned and useful for historical comparison, but its mixed tasks, preferred candidate-derived targets and previous use limit the new phonetic claim. Hash checks prove files did not change, not that all linguistic answers are right. See R1/R2/R3/R8. New qualification and untouched final evaluation are needed.

## Why use multi-reference exact match if the goal is sound?

It is transparent and useful when references are sufficiently qualified and accepted alternatives are covered. It otherwise measures agreement with an incomplete spelling list. Automated phonetic diagnostics add information but can also be wrong. P1 addresses multiple valid transliterations in another script; Khmer needs its own appropriate reference policy.

## Could Gemini be the evaluator?

An optional secondary check, not the sole gold-label authority or both teacher and final judge. The user's favorable experience is motivation to consider it, not a measured validation study. This project makes no paid API calls and does not require Gemini availability. Proxy-only evidence remains explicitly unqualified.

## Why not penalize every difference between English and Khmer phones?

English and Khmer have different sound systems and borrowing adaptations. Raw equality can penalize a legitimate approximation. Use phones as input, inverse-lexicon training and independently accepted targets; validate any later cross-language phonetic penalty before using it as a training or adoption gate. P5/P6 motivate structured representations, not universal sound equivalence.

## What does a failed 5-point gate mean?

It means the adoption requirement was not met. It does not prove no improvement. E01's estimated +4.67 and interval [+2.0,+7.6] illustrate the distinction. See R7. A narrow interval excluding +5 supports a conditional statement about that practical threshold; an interval spanning zero and +5 is inconclusive. P7 supports effect-size/power thinking.

## Do three seeds prove the result is real?

They provide a useful small check of training variability, not certainty. Group bootstrap measures sensitivity to the evaluated lexical sample conditional on those trained models. Report seed results separately and avoid pretending hundreds of bootstrap samples create hundreds of independent training runs. See R2/P7.

## What if nobody can independently validate the new benchmark?

Then the honest conclusion is an unqualified/inconclusive quality result, even if reference agreement rises. Claude completes the code, preview model if feasible, reports and reproducible package. A future independent expert review can increase evidence quality; routine work must not stall while requesting more owner labor.

Qualification requires independent acceptance evidence for each primary item under the phonetic policy. Unreviewed attestation and proxy judgments stay exploratory; the agent cannot make them gold merely by announcing it audited them.

## What happens if no model improves?

Keep the eligible clean letters baseline as default, preserve candidate results and publish the negative/inconclusive finding in the local report. No unbounded tuning, threshold changes or quiet fallback to historical B/E. The product and the hypothesis have separate completion statuses.

## What happens if the agent runs out of context or the machine stops?

Durable state, checkpoint/RNG restoration and an exact resume command continue the fixed queue without repeating completed work. No instruction can guarantee computation after an external runtime terminates; the agent must never claim otherwise. See `AUTONOMY.md`.

## Are all these source claims already verified locally?

No. Public code/docs and primary papers were inspected to prepare the handoff. Local datasets, panel contents, permissions and model exports require Claude's audit. In particular, the MekongPhon code repository was not fetched during preparation. Claude records each source's verification status instead of treating this bibliography as a certificate.
