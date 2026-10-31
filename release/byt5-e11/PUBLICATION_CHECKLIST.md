# Publication checklist (E11 ByT5 adapter): prepared, not executed

Nothing below has been run, and no repository has been created. Each step needs the owner's explicit go-ahead.
The existing public model `ric912/en-km-phonetic-transliterator` is a different model and must not be touched.

## Before upload (owner)
1. **Adapter licence:** choose one (`LICENSING_DECISION.md`, decision 1). Then:
   - add `license: <id>` to the YAML header of `huggingface/README.md`;
   - add the licence notice as `huggingface/LICENSE`;
   - rebuild the bundle (`scripts/e11_release.py bundle`) so the checksums are rewritten.
2. **Data questions:** accept or resolve them for this adapter (`NOTICES.md` §1–3).
3. **Repository:** confirm the name, `ric912/en-km-phonetic-byt5-lora`, and that it starts private (as D192).
4. **Wheel:** decide whether the wheel goes into the repository (`package/`) or is linked from the card.

## Upload (after approval; private first)
```bash
cd dist/khmer-translit-byt5-e11-0.2.0
sha256sum -c SHA256SUMS                                     # bundle intact
hf repos create ric912/en-km-phonetic-byt5-lora --private
hf upload ric912/en-km-phonetic-byt5-lora huggingface . \
    --commit-message "E11 ByT5-small LoRA adapter (khmer-translit-byt5-e11 0.2.0)"
# optional, only if decision 4 says so:
# hf upload ric912/en-km-phonetic-byt5-lora wheel/khmer_translit-0.2.0-py3-none-any.whl package/khmer_translit-0.2.0-py3-none-any.whl
```

## After upload (private)
1. **Check the page:** the card renders, the metadata parses (`base_model: google/byt5-small`, relation
   `adapter`), and no inference widget appears (`inference: false`).
2. **Check a fresh download** in a new CPU environment:
   - run `hf download ric912/en-km-phonetic-byt5-lora --local-dir a`;
   - check the hashes against `MANIFEST.json`;
   - run the README quick start offline with the base at revision `68377bdc…`.
3. **Make it public:** only on the owner's instruction, after reviewing the private page. Tag it `v0.2.0-e11` if
   the owner wants a tag.
4. **Record it:** add a decision record (D-number) and update `CLAUDE.md`.

Not part of this release: PyPI, serving changes, merging `english-phonetic` into `main`.
