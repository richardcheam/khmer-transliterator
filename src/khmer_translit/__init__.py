"""English word -> Khmer phonetic spelling ("coffee" -> ខូហ្វ៊ី), with two models:

- `khmer_translit.english.Transliterator`: the small standalone model (NumPy only);
- `khmer_translit.byt5.ByT5Transliterator`: ByT5-small with a LoRA adapter (needs the `byt5` extra).

The command line is `khmer-translit` (`khmer_translit.cli`).
"""
__version__ = "0.2.1"
