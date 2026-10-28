# Google AI Mode handoff (E08 synthetic pilot, 200 words)

The only manual step: copy ten prompts into Google AI Mode and save the ten answers. No Khmer typing, no choosing
between spellings.

## For each batch `NN` = 01 … 10
1. Open `experiments/E08-data-expansion/synthetic/prompts/batch_NN.md` and copy **all** of it into Google AI Mode
   (one new conversation per batch is best).
2. Copy the **complete** answer, table and any text around it, into a new file:
   `data/e08/ai_mode_responses/batch_NN.md`
   (for example `data/e08/ai_mode_responses/batch_01.md`). The folder is local and git-ignored. Do not edit the
   answer.
3. When the files are saved (all ten, or only some), run:

   ```bash
   CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e08.py import --responses data/e08/ai_mode_responses
   ```

## What happens next (automatic)
- **Import:** every row is matched to the frozen manifest by its id (`e08s001` … `e08s200`).
  - Raw answers are copied to `data/e08/synthetic/raw/`.
  - Accepted rows go to `data/e08/synthetic/accepted.jsonl`, and problem rows to `quarantine.jsonl`.
  - Counts go to `../reports/import_summary.json`.
- **Rows needing another try:** missing ids, a Khmer cell with Latin or an explanation in it, an English mismatch,
  and similar problems.
  - These are listed in `../reports/needs_regeneration.md`.
  - A replacement prompt is written as `prompts/regen_01.md` (and `regen_02.md` if needed).
  - Paste it the same way and save the answer as `data/e08/ai_mode_responses/regen_01.md`. Then run the same
    import command.
  - Accepted rows are never asked again, and a later file cannot overwrite them.
- **Uncertain rows:** rows the answer marks `uncertain` are kept in quarantine and not asked again.
- **State:** `scripts/e08.py status` prints the state: `awaiting_ai_mode_responses`, then
  `awaiting_regeneration_responses`, then `imported_ready_to_freeze`.
- **Freeze:** `scripts/e08.py freeze` builds the training arm with synthetic rows and freezes every arm's hash.
  This does **not** train anything.

## What the import does and does not establish
- **It checks structure only:** ids, columns, Khmer script, length and orthography rules.
- **It does not check spelling quality:** a valid import says nothing about whether the spelling is phonetically
  good.
- **Synthetic rows are training data only** (provenance `google_ai_mode_synthetic`). They are never evaluation
  references and never "reviewed".

## Files
- `input_manifest.jsonl`: the 200 frozen inputs. Each row has an id, the English word, the category, the CMUdict
  reading (ARPAbet and IPA), the selection rule and the batch. The sha256 is in
  `../reports/input_manifest_frozen.json`.
- `prompts/batch_01.md` … `batch_10.md`: 20 words each; hashes in `../reports/prompts_frozen.json`.
