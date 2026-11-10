# docs/

The project report: IEEE-format LaTeX source, currently a skeleton with no
content.

```
docs/
├── main.tex          # document class, packages, title/author, section includes
├── references.bib     # bibliography (biblatex/biber) — exported from Zotero, not hand-edited
├── sections/           # one file per report section
│   ├── 01-introduction.tex … 05-conclusion.tex
│   └── appendices/     # incl. 0a-ai-usage.tex, the LLM-usage disclosure
└── figures/            # generated PDFs/tables, referenced by \includegraphics — gitignored
```

Build the PDF with:

```bash
make report     # cd docs && latexmk -pdf ... main.tex
make clean-report   # remove LaTeX build artifacts (not main.pdf itself)
```

`references.bib` is exported from Zotero (Better BibTeX) — don't hand-edit it.
Figures under `figures/` are never committed individually — they should be
regenerated from `scripts/` once an experiment pipeline exists.
