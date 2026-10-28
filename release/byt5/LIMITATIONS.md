# Intended use and limitations (khmer-translit-byt5 0.2.0, research preview)

## Intended use
- **Input:** one English word or name at a time.
- **Output:** a Khmer-script spelling that approximates its English pronunciation (General American, CMUdict-like
  reading). Typical use is drafting Khmer phonetic spellings for loanwords and names, for a person to check, e.g. in
  text-to-speech front-ends or glossaries.
- **Use offline on a CPU, in batches of independent words.**

## Not intended for
- **Translation:** it writes sound, not meaning.
- **Other inputs:** sentences, phrases, romanised Khmer chat ("sok sabay"), Khmer input or other source languages.
- **Unsupervised publication:** spellings should not be published without a Khmer-literate check.
- **Names read in their original language:** it follows an English reading.

## Known limitations
- **Unverified phonetic quality.** No independent human review exists yet.
  - Every number in this release is agreement with web-mined, automatically screened references. Those references
    were never reviewed, and the historical evaluation set has been examined many times.
- **Typical errors** (automatic categories against mined references):
  - vowel choices (the largest group);
  - ៉ / ៊ diacritics;
  - dropped or added final consonants;
  - occasionally a missing syllable.
- **Training data has known quality limits:**
  - mined pairs passed only a consonant screen, so vowel errors pass;
  - 384 training rows are unverified Google AI Mode output.
- **Ambiguous words** (*record*, *live*) get one reading, chosen implicitly by the model. There is no
  pronunciation input.
- **Long inputs:** words longer than 32 letters are refused, never truncated.
- **Scores** are summed log-probabilities. They rank candidates and are not confidence.
- **Cost:** about 0.5 s per word on a laptop CPU for one greedy answer, and about 2 GB of RAM (see
  `MODEL_CARD.md`). The base model is about 1.2 GB.
- **One training run per recipe:** seed-to-seed variation of the released checkpoint is not measured, unless the
  model card says otherwise.
