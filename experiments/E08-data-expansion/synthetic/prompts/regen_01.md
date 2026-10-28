# Khmer phonetic spelling of English words (replacement batch 01)

## Task
Write each English word below in Khmer script so that a Khmer reader would pronounce it like the given
**English** pronunciation. This is phonetic spelling (transliteration), not translation.

## Rules
1. Spell the sound, not the meaning. Never give a Khmer translation or a native Khmer word that means the same
   thing.
2. The pronunciation column (US English; ARPAbet with stress digits, and IPA) is the intended reading. Follow it,
   even for names: do not switch to the name's pronunciation in another language.
3. If an established Khmer loanword spelling exists and matches this pronunciation, you may use it.
4. Give one preferred spelling. Optionally give up to two alternative Khmer spellings **of the same
   pronunciation** (not a different pronunciation, not a translation).
5. If you are not sure, write `yes` in the `uncertain` column. You may then leave the spellings empty or give your
   best attempt.
6. Keep every id exactly as given: one row per id, in any order, and no extra rows.
7. Spelling columns contain Khmer script only: no Latin letters, digits, brackets, quotation marks or
   explanations. Put a short comment (optional, a few words) in `note` only.
8. Do not invent sources or confidence scores.

## Answer format
Return only this Markdown table, with no text before or after it:

| id | english | preferred_khmer | alternative_1 | alternative_2 | uncertain | note |
|---|---|---|---|---|---|---|

## Words (8)

| id | english | pronunciation (ARPAbet) | pronunciation (IPA) |
|---|---|---|---|
| e08s020 | told | T OW1 L D | /tˈoʊld/ |
| e08s028 | moment | M OW1 M AH0 N T | /mˈoʊmənt/ |
| e08s035 | overcome | OW1 V ER0 K AH2 M | /ˈoʊvɝkˌʌm/ |
| e08s057 | showmanship | SH OW1 M AH0 N SH IH2 P | /ʃˈoʊmənʃˌɪp/ |
| e08s075 | catacomb | K AE1 T AH0 K OW2 M | /kˈætəkˌoʊm/ |
| e08s117 | vocalist | V OW1 K AH0 L IH0 S T | /vˈoʊkəlɪst/ |
| e08s146 | noblest | N OW1 B L IH0 S T | /nˈoʊblɪst/ |
| e08s175 | shawnee | SH AO1 N IY0 | /ʃˈɔni/ |

## Note for this replacement batch
Use Khmer Unicode letters only (block U+1780–U+17FF). Do not use Lao, Thai or Tibetan letters that look similar:
for example write the Khmer vowel ោ, not the Lao letter ໂ.
