# TRB 2027 manuscript package

The submission manuscript and compiled proof are at the root of this folder:

- `trb_template.tex`
- `trb_template.bib`
- `trb_template.pdf`
- `trbunofficial.cls`

The proof is 17 pages. It uses standard `chicago.bst` for alphabetical author--date references; the supplied `trb.bst` remains bundled as the original template artifact.

## Overleaf

Upload this folder while preserving `figures/` and `tables/`, then select `trb_template.tex` as the main document. The source uses relative paths, standard packages, BibTeX, and no shell escape or machine-specific path.

## Rebuild analyses and figures

From the repository root:

```bash
.venv/bin/python trb/scripts/generate_manuscript_assets.py
```

This regenerates `generated_results.tex`, all empirical tables, the two machine-readable figure datasets under `generated/`, and six figures in PDF and PNG form. It reads retained survey and frozen experiment artifacts; it does not retrain or resample a generator. Statistical bootstrap seed is 20260730.

## Compile

With a current TeX distribution:

```bash
latexmk -pdf trb_template.tex
```

The final local proof was built with Tectonic 0.17.0. See `manuscript_audit.md` for the exact command and production checks.

## Evidence and author checks

- `evidence_manifest.md` traces major claims to files, code, denominators, and reproduced calculations.
- `manuscript_audit.md` records the page count, central and contrary results, three review passes, warnings, file inventory, and all unresolved items.

Before submission, replace every visible `[VERIFY ...]` or `[AUTHOR APPROVAL REQUIRED]` marker. In particular, confirm all job titles, corresponding-author status, CRediT roles, acknowledgments, conflicts, funding, data rights, and the generative-AI disclosure. Also verify the retained household-final weight against the missing source dictionaries and confirm eligibility relative to any prior WCTR dissemination.
