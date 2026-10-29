# Licensing decision needed (owner): E02 letters-only release

**Status (2026-10-29):**
- **Code: decided.** The owner chose **PolyForm Noncommercial 1.0.0**: research, study and other non-commercial use;
  commercial use needs the owner's permission.
  - Applied as `LICENSE` (the official 1.0.0 text plus a `Required Notice:` line).
  - `pyproject.toml` has `license = "PolyForm-Noncommercial-1.0.0"`; the wheel's metadata carries
    `License-Expression` and the licence file.
- **Weights: decided.** The owner chose **CC BY-NC-SA 4.0** for the current `letters_s0` weights (`8901050b…`),
  with no retrain.
  - The owner accepts the risk that Wikipedia's ShareAlike, if it reaches the weights, would call for CC BY-SA.
  - The Hugging Face card carries `license: cc-by-nc-sa-4.0`; `LICENSE-weights.md` holds the notice.

## Two separate choices
### 1. Code (`khmer_translit` package)
- This is project-authored code, and nothing upstream constrains its licence:
  - the wheel contains only project code;
  - NumPy is a separately installed dependency;
  - no third-party code is vendored.
- **Common options:**
  - permissive (MIT, Apache-2.0: Apache adds a patent grant and NOTICE handling);
  - copyleft (GPL-3.0, MPL-2.0: file-level).
- Whatever is chosen goes into `LICENSE` and `pyproject.toml` (`license = "…"`), and applies to the sdist and wheel.

### 2. Model weights (`model.npz`)
The choice is constrained by the training data, which the code licence is not:
- **Khmer Wikipedia (CC BY-SA 4.0):** if the weights were Adapted Material, ShareAlike would require a BY-SA (or
  BY-SA-compatible) licence for them. Whether they are is unresolved (`NOTICES.md`). Options:
  - (a) license the weights CC BY-SA 4.0, which is conservative and satisfies SA if it applies;
  - (b) choose another licence and accept the open SA question;
  - (c) retrain without the 693 Wikipedia-derived rows: a new experiment, not done here.
- **FineWeb-2 (ODC-By 1.0 + Common Crawl terms):** the notice in `NOTICES.md` is required for public use of a
  Produced Work. Rights in the underlying pages are not granted by ODC-By. A licence chosen for the weights cannot
  grant more than the owner holds.
- **Hugging Face metadata after a decision:**
  - a listed licence → its identifier (e.g. `cc-by-sa-4.0`, `apache-2.0`);
  - a custom licence → `license: other`, a `license_name`, and a `LICENSE` file in the repo, per the Hub's
    licence documentation.

## Questions for the owner
1. Which licence for the code?
2. ~~Which licence for the weights?~~ CC BY-NC-SA 4.0, current weights (2026-10-29).
3. Is the indirect use of a Chhunneng-trained model during mining (`PROVENANCE_AUDIT.md` §3) acceptable for
   release?
4. Was the LLM that suggested the 17 owner-verified selection spellings used under terms that allow this? Which
   service was it?
