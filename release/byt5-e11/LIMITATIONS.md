# Intended use and limitations (khmer-translit-byt5-e11 0.2.0)

## Intended use
- **Input:** one English word or name at a time.
- **Output:** a Khmer-script spelling approximating its English (General American, CMUdict-like) pronunciation.
- **Typical use:** drafting spellings of loanwords and names for a person to check, for example in text-to-speech
  front-ends or glossaries.
- **Running:** offline, on a CPU, in batches of independent words.

## Not intended for
- **Translation:** it writes sound, not meaning.
- **Other inputs:** sentences, phrases, romanised Khmer chat, Khmer input or other source languages.
- **Unchecked publication:** spellings shouldn't be published without a Khmer-literate check.
- **Original-language names:** names are read the English way, not in their original language.

## Known limitations
- **Quality evidence is narrow.**
  - One project-affiliated judge.
  - Two 120-family test samples (64.2% and 75.0% confirmed acceptable), which aren't a population benchmark.
  - One training run (seed 0); seed-to-seed variation isn't measured.
- **Irregular English spellings are weak.** Only 14/40 and 20/40 were acceptable in the two tests.
  - Silent letters and letter-by-letter readings occur, for example *fire* read as f-i-r-e.
  - Past-tense *-ed* is sometimes sounded as a syllable.
- **Vowel choices** are the most frequent disagreement with mined references. ៉ / ៊ diacritics and dropped or
  added final consonants also occur.
- **Training labels were never reviewed.** The mined pairs passed only a consonant screen, so some vowel errors
  are in the training data.
- **Ambiguous words** (*record*, *live*) get one reading, chosen implicitly by the model. It has no pronunciation
  input (adding one in E13 gave no detectable gain).
- **Limits:**
  - words longer than 32 bytes are refused, never truncated;
  - multi-word inputs are refused, not split.
- **Scores** rank candidates. They are not confidence.
- **Unassessed spellings:** a spelling missing from an accepted list was never judged and stays unassessed. It is
  not automatically wrong.
- **Cost:** see `EVALUATION.md` §6 for time and memory. The base model is about 1.2 GB on disk.
