# Summary: English-to-Khmer phonetic transliteration

*Richard Cheam (richardcheam912@gmail.com). A summary of the technical report `main.pdf` (31 October 2026).*

## Task
Given one English word or name, the system writes in Khmer script how it sounds when read in English: *coffee* →
ខូហ្វី, *london* → ឡុនដុន.

It is not translation, not conversion of romanised Khmer (*sok sabay*), and not reproduction of established loan
spellings (the usual Khmer *coffee*, កាហ្វេ, follows another reading). Each word needs a declared pronunciation:
the first General American reading in the CMU Pronouncing Dictionary. Khmer often allows several acceptable
spellings of one sound.

## Released systems
Both are research previews with one command-line and Python interface. Both weights are licensed CC BY-NC-SA 4.0.

1. **Small character-level Transformer:** 2.6M parameters, trained from scratch on 9,735 mined pairs. NumPy only;
   about 17–19 ms per word and 65 MiB.
   - Its file on Hugging Face is byte-identical to the recorded checkpoint.
2. **LoRA adapter for ByT5-small:** 6.26M parameters on a frozen 300M base, trained on 4,764 pairs. Needs PyTorch
   and the pinned base; about 0.55 s per word on a laptop CPU, about 2 GB.
   - It is stored in the project's own format, so it loads through the `khmer_translit` package, not PEFT or
     `AutoModel`.
   - Its tensors are bit-identical to the training checkpoint.

## Data
Pairs were mined from Khmer web text (FineWeb-2, Khmer Wikipedia) where a Khmer spelling is followed by the English
word in brackets. Nobody reviewed the training spellings.

- **A selection error:** a rule meant to drop romanised-Khmer pairs also dropped every pair whose Khmer side was a
  dictionary word. Fixing it recovered 11,377 candidate rows.
- **Screening without vowels:** a consonant screen checks Khmer consonants against the English pronunciation. Vowels
  were never checked, and vowel differences were the largest class of disagreement.
- **Adapter data:** it adds 1,292 pairs recovered from known screen weaknesses and 1,217 spellings attested in the
  ALT Parallel Corpus.
- **No LLM labels:** the adapter contains none. A historical experiment used 384 such rows; they are not in the
  release.

## Evaluation
**Reference agreement was weak.** Early experiments counted exact matches with one reference spelling. In an
exploratory, post hoc development review (one judge, answers given in chat and transcribed under three confirmed
rules), 28 of 72 spellings that missed the reference were acceptable, and 8 of 37 that matched it were not.
Agreement errs in both directions. The same review suggested a preference for the expanded real-data model; its
representative (40 words) and disagreement-enriched (20 words) samples are reported separately.

**Final evaluation.** One judge affiliated with the project (the author) marked each spelling acceptable, not
acceptable or uncertain, without knowing its model, by clicking on a local review page. Inputs, checkpoints and
predictions were frozen first.

Each of two tests drew 120 fresh word families (40 everyday words, 40 irregular spellings, 40 names) and compared
one candidate model with the released adapter on that sample.

## Findings
- **Released adapter:** acceptable for 77 of 120 words (64.2%) on the first sample and 90 of 120 (75.0%) on the
  second.
  - These are two samples of the same model, not an improvement.
  - Irregular spellings were weakest: 14 and 20 of 40.
- **Longer training:** changed 65 of 120 spellings. The models had equal observed acceptability, 77 vs 77,
  difference 0.0 [−8.3, +8.3] points: no difference was detected, and the interval permits gains or losses.
- **Pronunciation-guided training:** giving the CMUdict reading plus some accepted difficult spellings also gave
  equal observed acceptability, 90 vs 90, 0.0 [−7.5, +7.5]; the interval again permits gains or losses. Several
  ingredients changed together, so no single one can be credited or blamed.
- **English pronunciation input:** no detectable change in agreement in four experiments. In the small model's
  registered comparison it gave +1.6 [−1.6, +4.9] over three seeds.
  - Another word's phones change 33.0–44.7% of the small model's outputs (by seed), mostly with vowel or
    diacritic edits rather than the consonants the phones dictate.
  - That shows weak use of pronunciation here, not that it cannot help.
- **Decoding:** an apparent beam-search deficit was a configuration error. With `early_stopping=True`, junk byte
  hypotheses ended the search. Corrected beam search is about as good as greedy decoding.
- **Candidate ranking:** on 23 hard irregular development words, the single answer was acceptable for 8, beam
  top-1 for 10, and at least one of four beam candidates for 15; 8 words had none.
  - A perfect selector could add 7 words over greedy.
  - This is a small upper bound, not an achieved result.

## Limits
- **Data and references:** the data are mined and unreviewed, and references give one spelling per word.
- **Repeated use:** much reference evidence comes from an evaluation set examined many times.
- **Human tests:** one non-independent judge; samples of 120 (each rate about ±7.7–8.6 points).
- **Replication:** one seed per adapter.
- **Scope:** single words only.
- **Permissions:** rights in web pages stay with their authors. Whether Wikipedia's share-alike reaches trained
  weights is unsettled.
- **No head-to-head comparison:** the two released models were trained on different data and never compared on a
  common clean test set. 110 of the 228 primary historical evaluation words occur in the small model's training
  data.

## Bottom line
- **Established:** a documented pipeline, two released models, and a blinded acceptability protocol; an
  exploratory review suggested a preference between models that reference agreement did not distinguish.
- **Negative results:** pronunciation input, longer training and pronunciation-guided training gave no detectable
  gain within these data and models.
- **Remaining errors:** they concentrate in irregular spellings. Acceptable alternatives are often generated but
  not chosen, while 8 of 23 hard words (35%) have none.
