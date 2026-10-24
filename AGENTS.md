# Working rules for AI agents (Claude Code, Codex)

**Project:** the Latin → Khmer transliterator of the Khmer TTS. Start with `README.md`, then
`docs/TRANSLITERATOR.md` (data, models, results, open problems); recent decisions in `docs/DECISIONS.md`.
The speech system itself is the sibling repository `../pocket-tts-khmer`.

## Owner's rules (always)
- **No spending without explicit approval:** no GPU rental, paid API or paid job without the owner's yes and a cost
  cap. Training runs on the laptop CPU.
- **No publishing without explicit approval:** weights, data or the repository.
- **The owner decides what is served,** by a blinded listening or spelling A/B. An agent never fills in answers.
- **Keep the package's public functions stable** (`guess`, `guess_many`, `caps_reads_as_word`, `configure`,
  `word_tag`, `pick`, `load`, `latin_g2p.transliterate`, `lexicon.keys`, `lexicon.segment`): the speech repo calls
  them. After changing them, run the speech repo's tests too.

## How to work
- Experiments: write the adoption rule down **before** seeing results; never tune on test sets or the reserve
  (`data/e2/reserve.json`); compare with paired seeds (one retrain moves dev-41 by about ±5 words); filter every
  test set out of every training file.
- Keep the repository small: when an experiment is decided, archive or delete its one-off scripts and models.
- Record consequential choices and results in `docs/DECISIONS.md` (numbers continue the speech repo's log).
- Code style: `docs/FUNCTION_DOCUMENTATION.md` (Google-style docstrings with arguments, returns, errors and a real
  example; type hints).
- Before committing: `.venv/bin/python -m unittest discover -s tests`.
- Explain to the owner what each step does in plain words, with a small runnable example where it helps.
