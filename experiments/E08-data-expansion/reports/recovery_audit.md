# E08 recovery of E03-excluded mined pairs (2026-10-28)

Code: `scripts/e08.py recover`. Counts: `recovery_summary.json`. Manifests (local): `data/e08/recovery/`.

## What was reconsidered
- **Reconsidered:** E03's quarantine (920 unresolved conflicts, 98 "established misalignments", 4 unknown) and its
  14 rows without a lexicon reading.
- **Also reconsidered:** E03's `noise_filter` rows (3,870 lower-frequency forms of a word). Of these, 2,090 could
  be screened (CMUdict word, Khmer word) and were run through **the same frozen screen** for the first time.
- **Not reconsidered:**
  - pairs without an English reading (2,368): no pronunciation can be specified;
  - romanised-Khmer inputs (still excluded);
  - held-out families and held-out targets (still excluded).

## Recovery rules
Each rule corresponds to a screen weakness documented before this phase (E02 diagnostics §2 type 5; E03 REPORT,
"uncertainty that remains"). A pair is recovered only if **every** unexplained consonant operation is explained by
one of these rules:

| Rule | Pairs | Evidence |
|---|---|---|
| `screen_compatible_alternative` (a noise-filtered form that the screen finds compatible) | 943 | The frozen screen, unchanged |
| `coda_h`: English s/z/sh written ស and read as Khmer coda h | 243 | Khmer final ស is pronounced h; E03 counted 207 conflicts with this as their only problem |
| `y_unwritten`: the English y glide is not written | 49 | E02 diagnostics |
| `final_ic`: final k written ច after ិ | 40 | The Google lexicon reads អាឡិច "ʔ aa . l ə c" |
| `y_glide`: an inserted y between vowels where the spelling has i/y + vowel | 22 | E02 diagnostics |
| `doubled_letter`: an extra consonant matching a doubled English letter | 19 | E02/E03 (អានណា *anna*) |
| compound flag with a doubled letter or glide (screen compatible) | 1 | E03 post-hoc: 87 of 98 flags were doubled letters or glides |

**Never recovered:**
- pairs that fit Google's Khmer reading better than the English one (a possible reading switch): left uncertain;
- possible translations (rule similarity < 0.15);
- level-3+ conflicts with no explained operation;
- anything relying on a model's preference.

## Result
- **1,309 recovered pairs** (1,112 words, 1,081 families). 813 of the words are not in the control data, and 396
  pairs are alternatives for control words.
- 109 errors or translations, and 1,693 uncertain.
- **Excluded after recovery:**
  - 368 pairs in protected families;
  - 156 whose target is a dev/eval or held-out target (mirroring E03's rule; cross-split target reuse is recorded,
    not treated as proof of leakage);
  - 17 more at arm build, whose family is in E03's train-val diagnostic set.

## Agent inspection (not independent review)
20 pairs per category, in sha256 order (`audit_samples` in the JSON):
- **Recovered: about 16/20 plausible.** Examples: *katy* ខេធី, *jerry* ជេរី, *claire* ក្លៀរ, *wes* វ៉េស, *trish*
  ទ្រីស, *afghanistan* អាហ្វហ្កានិស្ថាន.
  - Questionable: *desire* ដេសៀ, *erin* អ៊ីរិន, *hyun* ហុន, *tho* ធូ. The screen ignores vowels, so vowel errors pass.
- **Errors/translations: 20/20 not eligible.**
  - Translations: *fats* ខ្លាញ់, *wolff* លោក, *cartoon* ចាស់.
  - Wrong entities: *office* → Australia, *silicon* → ស៊ីលីស្យូម "silicium".
  - Truncations: *easter* ស្ទើរ, *foster* ស្ទ័រ.
- **Uncertain: mixed.**
  - Several are plausible but unproven: *plus* ផ្លឹស, *websites* វេបសាយ, *sunni* ស៊ុននី.
  - Others are reading switches or wrong: *sergey* សឺហ្គេ, *melon* ម៉ឺឡុង, *small* សៀមរាប "Siem Reap".
  - Uncertain is the right label: they are neither accepted nor declared wrong.

**Caveat:** about 20% of the recovered pairs may have vowel problems that the consonant screen cannot see.
