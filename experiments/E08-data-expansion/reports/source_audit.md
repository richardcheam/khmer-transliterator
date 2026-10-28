# E08 public-source audit (2026-10-28)

**Budget:** three substantive audits, at most 100 inspected examples each. 20.4 MiB downloaded (cap 250 MiB). No
crawl, no forms, no contact with authors. Downloads are in `data/e08/sources/` (local). Pilot outputs are in
`source_pilots.json` and `alt_attested_summary.json`.

| Source | What it contains | Fits the task? | Already local? | Pilot yield | Reuse conditions | Verdict |
|---|---|---|---|---|---|---|
| **ALT Parallel Corpus** (NICT; `ALT-Parallel-Corpus-20191206.zip`, 20.1 MiB; only `data_en.txt` and `data_khm.txt` used) | 20,106 English Wikinews sentences with NICT's Khmer translations | Partly: names and loanwords inside translated sentences are attested spellings; ordinary translation pairs are **not** used | No | 100 sentences (sha256 order): 249 capitalised CMUdict tokens, 48 verbatim matches of a letters-model guess in the Khmer sentence, 42 of them consonant-compatible. Agent inspection: about 4 spurious (*the* → ថា, *flo* → ផ្លូ inside ផ្លូវ) | README: "released the ALT Parallel Text as a Creative Commons Attribution 4.0 International (CC BY 4.0) License"; the English is from Wikinews (CC BY 2.5); cite Riza et al. 2016 | **Useful.** Full extraction with the pilot's fixes: 1,222 eligible pairs (1,217 after the train-val family rule), 288 uncertain, 5 errors. Agent inspection of 40: about 34 plausible, about 6 questionable, mostly truncated endings (*victoria* → វីកតូរី, *felt* → ហ្វេល) |
| **WAT km-en page** (ALT + ECCC) | The same ALT text, plus ECCC court records | ECCC is a translation corpus | — | Not downloaded | ECCC "Please fill and sign the ECCC-corpus-licence … Send the scan of the form"; the WAT page states no ALT licence (the CC BY 4.0 statement is on the NICT ALT page) | ECCC **permission-blocked** (needs a signed form, not submitted); ALT taken from NICT's own release instead |
| **Khmerologie** "Les mots anglais utilisés en khmer" v3 (Pascal Médeville, 2016; PDF, 258 KB) | About 280 Khmer lines with French glosses; English given in parentheses on 95 lines; includes compounds and expressions | Partly: loan spellings, but mixed with compounds and translations (ម្សៅការី "curry powder") | No | Text extraction garbles the Khmer (legacy font mapping, e.g. កាម្សរ៉ា for កាមេរ៉ា), so the entries cannot be read without manual re-keying | None: no licence, permission or reuse statement on the download page or the file page | **Lead only, permission-blocked.** Nothing used; not part of any training file |
| **google/language-resources `km`** (lexicon) | Khmer pronunciation lexicon: 62,150 Khmer-script and 7,206 Latin-script entries with Khmer phonemic readings | **Auxiliary only:** readings, not English → Khmer spelling pairs | **Yes:** `data/lexicon/km_lexicon.tsv`, already the consonant screen's reading source | 100 Latin entries: 43 in CMUdict; 65 have a Khmer entry with the identical reading (50 unique). Many Latin entries are romanised Khmer (*kmau*, *bamnak*), and matches often follow French or Khmer conventions (*napoleon* → ណាប៉ូលេអុង, *total* → តូតាល់) | CC BY 4.0 (file header) | **Unsuitable as a pair source.** Reading-matched pairs would encode Khmer and French conventions against the English-reading policy, and they share the screen's own evidence (circular). It stays auxiliary evidence |

## Published findings and their limits
- **Kirov et al., CL 50(2), 2024:** ByT5 transliteration fine-tuning was "quite sensitive to outliers/noise in the
  training set", and they pruned about 10% of the training data.
  - **Used:** every added provenance passes the same rule-based screen. Uncertain and error rows are quarantined,
    not trained on, and each provenance is a separate arm, so noise can be attributed.
  - **Not used:** the 10% figure. Their data and tasks (Dakshina, romanised South Asian languages) differ, so no
    pruning percentage is assumed here.
- **Xue et al., TACL 2022:** byte models suit spelling- and pronunciation-sensitive tasks. That is why the E07
  ByT5-small model is the one these data are for. The paper does not say how much data such a model needs.
