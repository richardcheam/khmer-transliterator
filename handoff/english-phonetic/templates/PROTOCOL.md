# Registered English phonetic experiment protocol

Complete and commit before full training; replace instructions below with actual resolved values, never results guessed in advance.

## Question and scope

Does English phone input help a small clean English→Khmer phonetic model, and does Khmer P2G pretraining plus replay improve the phone-input recipe? No translation, chat romanisation or audio/TTS.

## Fixed inputs

Repository/commit, code diff status, supervised source files/hashes/permissions/counts/tiers, P2G files/hashes/splits, common vocabularies, normalization/grouping, source encoder hash, pronunciation provider/alphabet/accent/resource hashes, training/dev/final manifests and qualification status. Explicitly settle the E01 source-encoding documentation conflict.

## Arms and training

letters / phones / phones_p2g, matching initialization tensors/batch IDs, exact hyperparameters, source lengths, 10% missing-phone training dropout in both phone arms, P2G pretrain cap and replay every 8 supervised updates, separate LR/step clocks, checkpoint metric and tie-break. Full attempts <=7; pretrain jobs <=3; aggregate time <=96 hours. All other bounds from AUTONOMY apply.

## Selection and confirmation

Seed-0 pilots; choose by English validation primary score, ties prefer letters then phones then phones_p2g. If a treatment wins, compare that fixed treatment with letters at paired seeds 0,1,2. No final-test selection. A letters win has no claimed confirmatory treatment improvement. B versus C remains a one-seed exploratory comparison; a three-seed C-versus-letters gain does not establish C's incremental advantage over B.

## Measurement

Qualified primary multi-reference exact with per-item independent acceptance evidence, raw beam4 top1, exact lexical-group weighting formula; item accuracy; reference-only status if labels are unqualified. Primary uses deployed default-provider phones, never injected reference phones. Fix the intended reading/default disambiguation contract before predictions; count actual provider errors. Supplied-pronunciation evaluation is a separate conditional track. Diagnostic top-k/CER/phonetic proxies/fixed reranker, not extra hidden primary outcomes. Fixed power simulation inputs and achievable sample size; 10,000 paired group bootstrap resamples/fixed seed and conditional interpretation; every training seed shown separately.

## Decision

Practical threshold +5 pp. Adoption: qualified references, mean >=5, positive in >=2/3 seeds, conditional lower 95% bound >0, controls and product guards pass. Positive small effects, exclusion of a useful gain and equivalence are separate fields. Apply the explicit Task 2 outcome definitions. No post-test threshold tuning, reference expansion to favor outputs or extra variants.

## Product fallback

Treatment passes → promote eligible treatment. Otherwise → eligible letters baseline. Unqualified benchmark → research-preview/inconclusive, not validated promotion. No eligible supervised data → tooling-only with weights unavailable. Neither unknown-source historical models nor random weights count as an eligible trained default.
