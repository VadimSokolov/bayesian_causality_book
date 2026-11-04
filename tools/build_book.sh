#!/bin/sh
# Build the book PDF and the KDP cover from this package. Run from the package root:
#   sh tools/build_book.sh
# Needs pdflatex, makeindex and pdfinfo. The log of the last pass is book_tex/main.log.
set -e
cd book_tex
rm -f figs; ln -s ../figs figs
pass() {
  if ! pdflatex -interaction=nonstopmode -halt-on-error main.tex >pdflatex.out 2>&1; then
    echo "pdflatex stopped on an error:"; grep -m1 -A6 '^!' pdflatex.out; exit 1
  fi
}
pass
pass
python3 ../tools/idx_dedupe.py main.idx
makeindex -q main.idx
pass
P=$(pdfinfo main.pdf | awk '/^Pages/{print $2}')
echo "book_tex/main.pdf: $P pages"
echo "  undefined references:    $(grep -c 'Reference .* undefined' main.log || true)"
echo "  undefined citations:     $(grep -c 'Citation .* undefined' main.log || true)"
echo "  multiply defined labels: $(grep -c 'multiply defined' main.log || true)"
echo "  overfull hboxes:         $(grep -c 'Overfull .hbox' main.log || true)"
echo "  makeindex warnings:      $(grep -c '## Warning' main.ilg || true)"
# Non-fatal report: do the number fragments still match results/*.json? A missing JSON (NO SOURCE) or a failure here never stops the build.
NM=$(cd .. && python3 tools/make_numbers.py --check 2>/dev/null || true)
echo "  number fragments:        $(echo "$NM" | grep -c '^OK' || true) match, $(echo "$NM" | grep -c '^DIFFERS' || true) differ, $(echo "$NM" | grep -c '^NO SOURCE' || true) without saved JSON (tools/make_numbers.py --check)"
echo "$NM" | grep -E '^(DIFFERS|NO SOURCE)' || true
cp ../tools/cover.py cover/ && cd cover && python3 cover.py "$P" && xelatex -interaction=nonstopmode cover.tex >/dev/null && xelatex -interaction=nonstopmode cover.tex >/dev/null
echo "book_tex/cover/cover.pdf written"
