#!/usr/bin/env bash
# Build main.pdf and the portable source archive.
#   tools/build.sh            regenerate values from git (needs the repository), compile, archive
#   tools/build.sh --no-extract   compile from the shipped generated/ files (for the source archive)
# Needs XeLaTeX (or LuaLaTeX) with fontspec, pgfplots, natbib, and BibTeX, plus the Noto Serif Khmer font.
# Tectonic (`tectonic main.tex`) also works: it runs BibTeX itself.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" != "--no-extract" ]]; then
  python3 tools/extract.py
fi

if command -v latexmk >/dev/null && command -v xelatex >/dev/null; then
  latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
elif command -v xelatex >/dev/null; then
  xelatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  xelatex -interaction=nonstopmode -halt-on-error main.tex
  xelatex -interaction=nonstopmode -halt-on-error main.tex
elif command -v lualatex >/dev/null; then
  lualatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  lualatex -interaction=nonstopmode -halt-on-error main.tex
  lualatex -interaction=nonstopmode -halt-on-error main.tex
elif command -v tectonic >/dev/null; then
  tectonic main.tex
else
  echo "no TeX engine found (xelatex, lualatex or tectonic)" >&2
  exit 1
fi

# portable sources: everything needed to compile without the repository
tar czf technical-report-src.tar.gz --transform 's,^,technical-report/,' \
  main.tex references.bib generated data/results_extract.json tools/extract.py tools/build.sh \
  README.md SUMMARY.md EVIDENCE_LEDGER.md
echo "built main.pdf and technical-report-src.tar.gz"
