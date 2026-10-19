# Overleaf / LaTeX package

Upload `paper-i2cit2026.zip` via Overleaf's **New Project -> Upload Project**.
Set the main document to `main.tex` and the compiler to pdfLaTeX.
`IEEEtran.cls` is provided by Overleaf, so it is not bundled.

Contents:
  main.tex            manuscript (IEEEtran, conference, 6-page limit)
  refs.bib            bibliography; every entry verified against the
                      publisher or arXiv listing
  tables/*.tex        \input-ed tables
  figures/*.pdf       figures
  make_tables.py      regenerates tables/ from ../results/*.json
  make_figures.py     regenerates figures/ from ../results/*.json

No number in the manuscript is typed by hand: tables and figures are
generated from the run artefacts, so re-running an experiment and re-running
the two scripts updates the paper. `facts.tex` carries the few figures quoted
in the prose as LaTeX macros for the same reason.

Local build: `latexmk -pdf main.tex`
