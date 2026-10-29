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

## Remaining before upload (owner)
1. **Choose a code licence** and add `LICENSE` and the `pyproject.toml` `license` field. Rebuild the wheel.
2. **Choose a weights licence,** or decide to retrain without the Wikipedia rows (`LICENSING_DECISION.md`). Then add
   the `license` field (and `LICENSE` file, if custom) to `huggingface/README.md`.
3. **Decide on the open data questions** in `NOTICES.md`:
   - Wikipedia CC BY-SA and the weights;
   - rights in the FineWeb-2 underlying pages;
   - the indirect Chhunneng influence through mining;
   - the LLM-suggested selection items.
4. **Approve publication explicitly** (repository rule: no publishing without the owner's yes), including:
   - the repository name and visibility;
   - whether `evidence/` (hashes, counts, model outputs on test words) is uploaded.
5. **Optional but recommended:** an independent phonetic review of a sample, so the card can report more than
   reference agreement.
6. **After upload:** check that the card renders, the licence shows, and the `datasets:` link resolves.
   - The Hub has no NumPy loader and no inference widget for this format. `library_name: numpy` labels it; users
     load it with the package as shown in the card.

## Not to upload
Training pairs, mined files, raw web text, references, review queues, the reserve, CMUdict, torch checkpoints, W&B
folders, ByT5 files.
