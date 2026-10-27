# E07 pipeline audit (2026-10-27)

Evidence: `pipeline_audit.json` (step `audit`), `e06_reproduction.json` (step `reproduce`) and `environment.json`.

**Setup:**
- Transformers 5.19.0, PyTorch 2.14.0 (CPU, 6 threads), base `google/byt5-small` at `68377bdc…`
  (`pytorch_model.bin` sha256 `5c5aaf56…931a`), E06 adapter `best.pt` sha256 `609dd835…`.
- Special ids are read from the local `config.json`: pad 0, eos 1, decoder start 0, vocab 384.

## Reproduction of E06's behaviour
- **Sample:** E06's selected adapter on a fixed 16-row sample (8 optimisation and 8 train-val rows, sha256 order).
- **Decoding:** greedy, eval mode.
- **Result:** 0/16 exact, 8 distinct outputs; អារ៉ានី appears 5 times and អាម៉ាន 4 times.
- On the 93 train-val rows: 13 distinct outputs, with អាលី 19 times.

Raw input ids, masks, decoded inputs, labels, decoded targets, generated ids and beam-4 outputs are saved per row.

## Boundary checks (all pass)
| # | Boundary | Evidence |
|---|---|---|
| 1 | English → ids | Unmasked id rows are pairwise distinct and decode back to the lower-cased words (+ EOS) |
| 2 | Khmer → ids | Label rows decode back to the intended targets, in batch order |
| 3 | Padding and masks | Mask sums equal source length + EOS; no pad id (0) appears at a loss position (labels are padded with −100) |
| 4 | Label padding ignored | Model loss = manual mean of token NLL over non-ignored positions (difference < 1e-5); loss positions = target bytes + EOS |
| 5 | Decoder inputs, shift, EOS | `_shift_right(labels)` = [0] + labels[:-1], with −100 → 0; the last label of every row is EOS; `decoder_start_token_id` = 0 from the config |
| 6 | Adapter insertion → optimiser | E06 recipe: 160 tensors (encoder self 1,425,408, decoder self 475,136, decoder cross 475,136 parameters). E07 tiny recipe: 256 tensors (adds encoder FFN 2,912,256 and decoder FFN 970,752). The optimiser's parameters are exactly the trainable set |
| 7 | Gradients → updates | Step 0: every B has a nonzero finite gradient, and A has zero gradient because B = 0 at initialisation (expected). Step 1: all A and B nonzero. After 2 updates every adapter tensor changed and the sampled base weights did not |
| 8 | Save → reload → generation | A freshly built model with the saved adapter, loaded strictly (no missing or unknown adapter keys), gives identical generated ids. A zero-initialised adapter gives the base logits (atol 1e-4) |
| 9 | Byte limits | The longest training target is 51 bytes, against a registered limit of 120 bytes. No target was truncated |
| 10 | Scores | Greedy and beam-4 scores equal teacher-forced sums of log-probs including EOS: maximum gap 1.2e-5 and 6.5e-6 |

**Also checked:**
- Transformers v5 sets `tie_word_embeddings = True` on every T5 config object, and rescales decoder outputs by
  d_model^-0.5 only when `scale_decoder_outputs`.
- For this checkpoint `scale_decoder_outputs` is `False`, and `lm_head` is not tied to the input embedding. The
  loader keeps both matrices and warns. So ByT5's untied head is used correctly.

## Verified defects
1. **Latent: no room for EOS at the length limit.**
   - E06 accepted targets of up to 120 bytes but generated at most 120 new tokens, so a 120-byte target could
     never end with EOS. Its decoder also did not flag a missing EOS.
   - **Not active in E06:** the longest target is 51 bytes, and every E06 output ended with EOS.
   - **E07 fix:** 121 new tokens and a `no_eos` status. Regression test: `test_generation_room_for_eos`.
2. **E07-only, found while building the tiny recipe:** T5's gated FFN reads `wo.weight.dtype`, which E06's
   wrapper did not expose.
   - It never affected E06, which adapted attention only.
   - **E07 fix:** a `weight` property returning the frozen base weight; the forward pass is unchanged.
   - Covered by `test_adapters_inserted_and_updated`.

No defect was found that explains E06's collapse.

## How the E06 loss fell while outputs stayed nearly constant
Each Khmer character is 3 UTF-8 bytes:
- **Lead byte:** almost always 0xE1.
- **Block byte:** 0x9E or 0x9F, nearly predictable.
- **Final byte:** chooses the character.

On train-val these are 31%, 33% and 30% of loss positions, plus EOS at 6%. The per-byte mean therefore averages two
almost-free bytes with one informative byte.

E06 adapter, train-val (93 rows), mean NLL in nats:

| Input | All bytes | Lead | Block | Final | EOS |
|---|---|---|---|---|---|
| Correct English word | 0.787 | 0.078 | 0.365 | **1.992** | 0.851 |
| Shuffled (another row's word) | 0.848 | 0.175 | 0.375 | **2.030** | 1.128 |
| Empty (EOS only) | 0.934 | 0.614 | 0.374 | **2.036** | 0.227 |
| Base model, no adapter (correct input) | 7.836 | 11.15 | 5.74 | 5.74 | 12.55 |

**What the table shows:**
- The fall from 7.8 to 0.79 is mostly the adapter learning Khmer UTF-8 structure (lead and block bytes) and a
  Khmer letter model.
- The informative final byte costs 1.99 nats with the correct word and 2.03 with someone else's word. The English
  input improves the character choice by about 0.04 nats.
- What the input does carry is length and when to stop (lead and EOS rows): a word-length signal, not letter
  identity.
- **Exposure bias:** under teacher forcing, the gold prefix keeps this loss low. Free-running greedy decoding has
  no gold prefix, so it follows the strongest letter-model path from a frequent onset. That gives a few frequent
  outputs.
- **Frequency bias:** the onsets match the most common training prefixes (ហ្ 136 targets, ម៉ 126, អា 113;
  common targets include វ៉ាល់, អេលី, ហ្វេន).

## Hypotheses
| Hypothesis | Evidence | Smallest discriminating check | Status |
|---|---|---|---|
| Inputs, labels or masks misaligned | Checks 1–5 | Decode the batch back; manual loss | Rejected |
| Adapters missing, untrained or not used at inference | Checks 6–8; E06 outputs are Khmer whereas the base model's are malformed | Gradient and update census; strict reload | Rejected |
| Train/inference mismatch (different base, dropout, adapter) | One `build` path; check 8; eval mode for generation | Reload equality | Rejected |
| Silent truncation in generation | Check 9 | Longest target vs `max_new_tokens` | Latent defect only (fixed), not the cause |
| v5 head scaling or tying error | Config inspection | `scale_decoder_outputs`, pointer equality | Rejected |
| Per-byte loss dilution + weak conditioning (the loss can fall without input use) | Decomposition table | Final-byte loss, correct vs shuffled | Confirmed, as a description of what E06 learned |
| Inadequate adaptation (capacity, regularisation, decaying LR, 1,000 updates) | Tiny gate passed (`tiny_overfit.json`): 32/32 exact after 100 updates; 32 distinct outputs; a shuffled input yields the other word's target 32/32; final-byte loss 0.0001 correct vs 5.67 shuffled; reload identical | 32-pair memorisation from the base model (PLAN §3) | **Supported:** the pipeline can learn an input-dependent mapping, so E06's collapse is an adaptation limitation. Which of E06's settings caused it (attention-only targets, dropout, the 1,000-update budget, the decaying LR) is **unresolved**: the tiny recipe differs in several of them at once and in its data size |
