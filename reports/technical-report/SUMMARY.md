# Summary: English-to-Khmer phonetic transliteration

*Richard Cheam (richardcheam912@gmail.com). A practitioner's summary of the technical report `main.pdf`
(November 2026).*

## The problem
Khmer text often contains English words and names. A Khmer text-to-speech system reads only Khmer script, so each
English word must first be rewritten in Khmer letters so that a Khmer reader says something close to the English
pronunciation: *london* → ឡុនដុន.

This is not translation, and it is not turning romanised Khmer (*sok sabay*) back into Khmer script. It also
differs from established loan spellings: Khmer writers usually spell *coffee* កាហ្វេ, but a spelling closer to the
English sound is ខូហ្វី. Several spellings can be acceptable for one word, because Khmer offers more than one way
to approximate many English sounds.

The hard part is supervision: no curated dictionary to learn from and no test set listing every acceptable spelling.

## What we built
- **A small character-level Transformer** (2.6M parameters), trained from scratch on about 9,700 web pairs. It
  needs only NumPy and answers in tens of milliseconds.
- **An adapted ByT5-small**, a 300M-parameter pretrained model that reads and writes raw bytes. We adapted it with
  LoRA, which trains about 6M extra parameters while the pretrained weights stay frozen. It was trained on about
  4,800 pairs.

Training pairs come from web text where a Khmer spelling is followed by the English word in brackets, for example
ហ្វេសប៊ុក (Facebook). They are mined automatically and never reviewed by a person. Evaluation words are kept out of
training by word family (*share*, *sharing*, *shared*), using a simple suffix-stripping rule.

## The central lesson: how you measure decides what you learn
Scoring a spelling by exact match with one reference is misleading here. For *band*, the released model wrote
បេន, which a Khmer reader accepts, but the mined reference was បាន, so exact matching scored it wrong. Older
models wrote បាន, which matched the reference and scored as right, yet the reader rejected it. In a review of 109
spellings, 28 of the 72 that missed the reference were acceptable, and 8 of the 37 that matched it were not.

We therefore switched to blinded human review. The reviewer sees the English word and its dictionary
pronunciation, then judges each candidate spelling as acceptable, not acceptable or uncertain, without knowing
which model produced it. An exploratory development review guided decisions; two prospective tests of 120 fresh
words each, frozen before training, gave the main results. One reviewer, the author, made all judgments, so the
results reflect one reader and one policy.

## What changed quality, and what did not
- **More real data looked encouraging.** Recovering wrongly filtered pairs and adding corpus-attested spellings
  roughly doubled the data. Exact matching barely moved, but the development review favoured the larger-data model.
  This is development evidence, not a confirmed result.
- **Training twice as long did not help.** It changed 65 of 120 test spellings but left the acceptance rate the
  same (77 of 120 for both models). The sample cannot rule out modest differences.
- **Adding English pronunciation as input gave no clear benefit.** The models largely ignored it. For example,
  removing it from the pronunciation-guided input (`fire|F AY1 ER0`) changed only about 7% of outputs. This does not
  show that pronunciation cannot help.
- **A decoding flag caused an apparent beam-search deficit.** With early stopping on, byte-level beam search
  returned truncated spellings. Turning it off made beam search as good as greedy decoding.
- **Acceptable spellings are often generated but not chosen.** On 23 hard words, the first answer was acceptable
  for 8, but one of four candidates was acceptable for 15; 8 had none. That points to candidate selection as an
  opportunity, not an achieved result.

## Lessons for similar projects
Inspect examples of what filters remove, check generation settings before blaming the model, do not equate
one-reference agreement with acceptability, and do not expect more training to repair noisy supervision.

## Where things stand
On two separate, deliberately hard 120-word samples, the released adapter's spelling was acceptable for 64.2% and
75.0% of words, and for 35–50% of irregularly spelled words. These two samples contain different words, so the
rates are not a trend. Irregular spellings, unreviewed training data and a single reviewer are the main
limitations. The two released models have never been compared on a common, untouched test.

Code: https://github.com/richardcheam/khmer-transliterator. Models: https://huggingface.co/ric912/en-km-phonetic-transliterator
and https://huggingface.co/ric912/en-km-phonetic-byt5-lora (CC BY-NC-SA 4.0).
