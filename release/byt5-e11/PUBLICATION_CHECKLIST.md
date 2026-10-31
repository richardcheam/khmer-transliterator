# Publication checklist (E11 ByT5 adapter): executed; public since 2026-10-31 (D200)

Nothing below has been run, and no repository has been created. Each step needs the owner's explicit go-ahead.
The existing public model `ric912/en-km-phonetic-transliterator` is a different model and must not be touched.

## Before upload (owner)
1. ~~**Adapter licence**~~ **done (D198):** CC BY-NC-SA 4.0, `license: cc-by-nc-sa-4.0` in the card,
   `huggingface/LICENSE`, bundle rebuilt.
2. ~~**Data questions**~~ **accepted as for E02** (owner chose the E02 terms, D198; `NOTICES.md` §1–3).
3. ~~**Repository**~~ **done (owner, D199):** `ric912/en-km-phonetic-byt5-lora`, private first.
4. ~~**Wheel**~~ **done (owner, D199):** not uploaded here; the card links the identical wheel already public in
   `ric912/en-km-phonetic-transliterator` at tag `v0.2.0` (sha256 `1bc8242a…`, checked byte-identical).

## Upload (after approval; private first)
```bash
cd dist/khmer-translit-byt5-e11-0.2.0
sha256sum -c SHA256SUMS                                     # bundle intact
hf repos create ric912/en-km-phonetic-byt5-lora --private
hf upload ric912/en-km-phonetic-byt5-lora huggingface . \
    --commit-message "E11 ByT5-small LoRA adapter (khmer-translit-byt5-e11 0.2.0)"
```

## After upload (private)
1. **Check the page:** the card renders, the metadata parses (`base_model: google/byt5-small`, relation
   `adapter`), and no inference widget appears (`inference: false`).
2. **Check a fresh download** in a new CPU environment:
   - run `hf download ric912/en-km-phonetic-byt5-lora --local-dir a`;
   - check the adapter sha256 against the card;
   - run the README quick start offline with the base at revision `68377bdc…`.
3. **Make it public:** only on the owner's instruction, after reviewing the private page. Tag it `v0.2.0-e11` if
   the owner wants a tag.
4. **Record it:** add a decision record (D-number) and update `CLAUDE.md`.

Not part of this release: PyPI, serving changes, merging `english-phonetic` into `main`.
