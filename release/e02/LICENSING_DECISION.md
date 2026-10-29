# Licensing decision needed (owner): E02 letters-only release

**Status: no licence has been chosen**, for the code or for the model. Checked 2026-10-29:
- no `LICENSE` file;
- no `license` field in `pyproject.toml`;
- no licence decision in `docs/DECISIONS.md` (D178–D188 all record it as open).

This document does not choose a licence and applies none. The Hugging Face card therefore carries **no `license`
field**: an identifier must not be invented for a licence that does not exist.

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
2. Which licence for the weights, given the open CC BY-SA question? Or retrain without Wikipedia rows first?
3. Is the indirect use of a Chhunneng-trained model during mining (`PROVENANCE_AUDIT.md` §3) acceptable for
   release?
4. Was the LLM that suggested the 17 owner-verified selection spellings used under terms that allow this? Which
   service was it?
