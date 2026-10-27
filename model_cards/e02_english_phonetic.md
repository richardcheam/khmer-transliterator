# Model card — E02 English phonetic transliterator (`models/e02/letters_s0`, research preview)

| | |
|---|---|
| **What it does** | one English word or name → its English pronunciation written in Khmer script (*coffee* → ខូហ្វ៊ី); no translation |
| **Default weights** | `models/e02/letters_s0/model.npz`, sha256 `8901050b1093cf90f60523c7941d6c12dbd8571d71698e37abcf874ba437904b`; 2,609,169 parameters |
| **Research weights** | `phones_s0..s2` (letters + CMUdict phones), `phones_p2g_s0` (+ Khmer P2G pretraining/replay), `letters_s1..s2`; same format |
| **Experiment** | `experiments/E02-english-phonetic/` (PLAN.md registered before training; RESULTS.md) |
| **Evidence status** | **inconclusive**: the benchmark is unqualified; scores are agreement with the owner's list |
| **Served** | no. The speech repo still serves B/E; serving is the owner's decision |
| **Redistribution** | `restricted_artifacts_local_only` (CC BY-SA mined training pairs; code licence not chosen) |

## Model
Transformer encoder–decoder over characters/tokens: d 192, 3 + 3 layers, 4 heads, feed-forward 512, GELU, learned
positions; NumPy inference with beam 4 (`khmer_translit.english.Transliterator`, built on `translit.Transliterator`).
- **Input** (`phonetic_source.encode_source`, encoder `en-phonetic-src-v1`, 157 tokens): `<task:en>` + the letters
  (default model). Phone models add `<sep>` + the first CMUdict reading with stress (ARPAbet), or `<no_pron>` if the
  word is not in CMUdict (no G2P fallback).
- **Output:** Khmer characters (78), NFC.
- Normalisation: NFKC, lower case, accents removed, apostrophes/hyphens dropped, `a–z` only, ≤ 32 letters; anything
  else is refused with a status (`empty`, `unsupported_multiword`, `unsupported_characters`, `too_long`).
- `log_score` is the beam log-probability, not a probability of being correct.

## Training
`scripts/english_phonetic_run.py` (durable queue) → `english_phonetic_train.py train/export/verify`. CPU FP32, AdamW
lr 1e-3, wd 0.01, 500 warm-up steps then cosine, label smoothing 0.1, batch 128, clip 1.0, 40 epochs; checkpoint by
dev macro reference agreement (86 items). Default model: seed 0, best epoch 27, 3,080 steps, 0.25 h on the laptop.
- **Data:** 10,031 English–Khmer pairs mined from brackets in Khmer Wikipedia and FineWeb-2 (9,735 train, 296
  validation), kept only if the word is in CMUdict, does not look like romanised Khmer, passes the D135 noise filter
  and is unrelated to any benchmark word. No Chhunneng data, no LLM output, no owner list entry.
- `phones_p2g`: 10-epoch pretraining on 58,716 Google Khmer lexicon pairs (Khmer phones → spelling), then one
  replay batch every 8 supervised updates.

## Results (`en-phonetic-v1`, final primary track, 165 items / 162 families; raw beam-4 top-1)
| Model | Primary macro | Fixed re-ranker (diagnostic) | Top-4 | Proxy track (400) |
|---|---|---|---|---|
| letters (default), seeds 0/1/2 | 16.7 / 15.1 / 16.0 | 18.5 / 18.2 / 19.1 | 40.4 / 39.2 / 43.5 | 30.1 / 25.6 / 27.1 |
| phones, seeds 0/1/2 | 17.3 / 16.7 / 18.8 | 22.8 / 21.9 / 23.8 | 42.9 / 42.9 / 42.3 | 28.9 / 30.0 / 28.1 |
| phones_p2g, seed 0 | 16.7 | 21.9 | 46.0 | 30.7 |

Phones vs letters, paired over 3 seeds: +1.65 points, 95% interval [−1.65, +4.94] → below the registered +5-point
adoption threshold; not adopted. Runtime on the laptop (fresh process): letters cold load 0.09 s, p95 32 ms/word,
65 MiB; phones 1.47 s, 36 ms, 137 MiB (CMUdict).

## Why the plain letters model is the default
The registered rule (PLAN.md) promotes a treatment only on a qualified benchmark with a ≥ 5-point gain; neither
holds. The phone models are kept as research artifacts: they agree slightly more often with the list and need
CMUdict at run time.

## Limits
- Exact match against one accepted spelling per word: correct-sounding variants count as wrong, so ~16% is a floor
  on usefulness, not an accuracy. Top-4 (40%) shows the right list spelling is often among the candidates.
- The training pairs are web-mined: some are conventional Khmer spellings or translations, not sound renderings;
  some names follow their own language's reading.
- US English pronunciations (CMUdict); a wrong or missing dictionary reading becomes a wrong input for phone
  models.
- The phone models lean mostly on the letters: for *record* the verb reading (R IH0 K AO1 R D) changes only the
  second candidate; *live* comes out as ឡៃវ៍ ("laiv") although its only CMUdict reading is L IH1 V.
- Owner's list references, 94 of 191 final items anchored on historical model E's candidates.

## Sources and hypotheses
CMUdict (BSD-style), Google Khmer lexicon (CC BY 4.0, Copyright 2018 Google Inc.), Khmer Wikipedia (CC BY-SA 4.0),
FineWeb-2 (ODC-By 1.0). Phonetic auxiliary information helping transliteration is motivated by prior work in other
language pairs (He & Cohen 2020; Ngo et al. 2015; see `experiments/E02-english-phonetic/SOURCE_VERIFICATION.md`);
that it helps this Khmer model is this project's hypothesis, tested here with an inconclusive result.
