# Publication checklist: E02 letters model (before any Hugging Face upload)

## Done locally (2026-10-29)
| Item | Status | Evidence |
|---|---|---|
| **Checkpoint identity:** `model.npz` sha256 `8901050b…904b`, lineage to `best.pt`, config and data | Done | `PROVENANCE_AUDIT.md` §1, `evidence/provenance_facts.json` |
| **Training data traced:** FineWeb-2 + Khmer Wikipedia only; no Chhunneng, LLM or owner labels; random initialisation | Done | `PROVENANCE_AUDIT.md` §2 |
| **Source hashes:** Wikipedia dump against Wikimedia's sha1, FineWeb-2 shard against the Hub LFS hash | Done | `PROVENANCE_AUDIT.md` §3 |
| **Permissions table,** source by source, with primary links | Done | `NOTICES.md` |
| **Package:** wheel + sdist built from a clean `git archive` of a commit | Done | `evidence/build.json` |
| **Isolated install,** offline inference, parity, edge cases, latency and memory | Done (17/17 checks) | `evidence/install_verify.json` |
| **Hugging Face folder:** card, `model.npz`, notices; no unsupported licence identifier | Prepared, **not uploaded** | `huggingface/` |
| **Bundle** with per-file and archive checksums | Done | `SHA256SUMS`, `.tar.gz.sha256` |

## Uploaded (private) — 2026-10-29
- https://huggingface.co/ric912/en-km-phonetic-transliterator, **private**, first commit `b7f31150`.
- **Files:** `README.md`, `model.npz` (`8901050b…`), `LICENSE` (CC BY-NC-SA 4.0), wheel (`1bc8242a…`).
- **Checked on the Hub:**
  - the metadata parses (licence `cc-by-nc-sa-4.0`, languages en/km, FineWeb-2 dataset link);
  - a fresh `hf download` plus a clean install runs the quick start, and the hashes match.
- **Tag:** `v0.2.0` → Hub commit `145adf99` (the current card without hashes). Later versions of the same model
  family go into this repo as new tags; another architecture goes into its own repo, linked by `new_version:`.
- **Next:** the owner reviews the page and makes it public (`hf repos settings ric912/en-km-phonetic-transliterator
  --public`, or the repository settings page).

## Remaining before upload (owner)
1. ~~Choose a code licence~~: **done**, PolyForm Noncommercial 1.0.0 (`LICENSE`, `pyproject.toml`). Rebuild the
   wheel and bundle so they carry it.
2. ~~Choose a weights licence~~: **done**, CC BY-NC-SA 4.0 on the current weights (`license: cc-by-nc-sa-4.0` in
   the card, `LICENSE` in the Hugging Face folder). The ShareAlike risk is accepted (D191).
3. **Decide on the open data questions** in `NOTICES.md`:
   - Wikipedia CC BY-SA and the weights;
   - rights in the FineWeb-2 underlying pages;
   - the indirect Chhunneng influence through mining;
   - the LLM-suggested selection items.
4. **Approve publication explicitly** (repository rule: no publishing without the owner's yes), including:
   - repository name: **`en-km-phonetic-transliterator`** (chosen 2026-10-29), i.e. `ric912/en-km-phonetic-transliterator`.
     Visibility: start private, check the card renders, then make it public;
   - whether `evidence/` (hashes, counts, model outputs on test words) is uploaded.
5. **Optional but recommended:** an independent phonetic review of a sample, so the card can report more than
   reference agreement.
6. **After upload:** check that the card renders, the licence shows, and the `datasets:` link resolves.
   - The Hub has no NumPy loader and no inference widget for this format. `library_name: numpy` labels it; users
     load it with the package as shown in the card.

## Not to upload
Training pairs, mined files, raw web text, references, review queues, the reserve, CMUdict, torch checkpoints, W&B
folders, ByT5 files.
