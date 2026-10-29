# Changelog

## 0.2.0 (unreleased; local research preview, 2026-10-28)
- **Checkpoint:** E08 `rs_s0`, kept by the E10 gate. The exploratory candidate A_s0 (word-level sampling, LR
  2e-3) reached dev 55.4% vs 54.2%, +1.2 points against the required +3.0, so no replication was run. On eval it
  was +0.6 [−4.5, +5.9], with CER not better.
- **New backend:** `khmer_translit.byt5`, ByT5-small + LoRA, English word → Khmer phonetic spelling.
  - Optional extra: `pip install 'khmer-translit[byt5]'`.
  - The historical E02 NumPy model stays the default.
- **CLI:** `khmer-translit predict|inspect --backend byt5 --model ADAPTER_DIR [--base BASE_DIR] [--offline]
  [--threads N]`, for one word, a file or stdin, with text, JSON or JSONL output and `--top-k` up to 8.
  - JSON/JSONL rows gain a `backend` field.
- **Decoding:** greedy for single answers; canonical beam search (`early_stopping="never"`) for candidate lists.
  - E09 found that `early_stopping=True` truncated byte-level beams (D187).
- **Inputs:** refused inputs (empty, several words, unsupported characters, more than 32 letters) return a status
  instead of a guess. No truncation, no fallback to another model.
- **Research code:** `e07.gen` gains `early_stopping` / `length_penalty` arguments, and `e07.fit` gains an `order`
  hook. Both defaults are unchanged, so E07/E08 still reproduce.
- **Unchanged:** the public speech-front-end functions (`guess`, `guess_many`, …).

## 0.1.0 (local, 2026-10-27)
- E02 English → Khmer phonetic model (NumPy inference) and the `khmer-translit` CLI.
