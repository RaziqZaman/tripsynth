# TRB 2027 manuscript package

This folder contains a compiled, 16-page TRB review manuscript built from the frozen `wctr-aadt` run at commit `95dab02c`.

## Overleaf upload

1. Upload `paper/trb_template.tex` and `paper/trb_template.bib` to the Overleaf project root.
2. Upload the PDF files from `figures/` into an Overleaf folder named `figures/`.
3. Set `trb_template.tex` as the main document. Overleaf will run Biber for the Chicago author-date bibliography.
4. The local proof is `paper/trb_template.pdf`.

The source also resolves figures when the repository structure (`paper/` beside `figures/`) is preserved.

## Rebuild figures

From the repository root:

```bash
.venv/bin/python trb/generate_figures.py
```

The script reads the preserved Parquet/CSV artifacts; it does not retrain or resample a generator. Figure 5 recomputes the audited date-pooled 56-screenline by 24-hour comparison and paired screenline-block intervals.

## Author checks before submission

- Verify every job title, author order, author contribution, funding statement, and conflict-of-interest declaration.
- Verify the use of the retained household-final weight at trip-row level against both source-survey data dictionaries.
- Confirm rights and access language for the transformed survey data.
- Disclose generative-AI use in the TRB submission form; the manuscript already contains a disclosure.
- Confirm that any WCTR paper or submission based on this branch does not make the TRB paper ineligible for Presentation + Publication. Select the appropriate TRB submission track if material overlaps.

The manuscript deliberately omits unsupported unique-OD counts, the unarchived hyperparameter-sweep claim, the incorrect GEH implementation, and the non-comparable exact-date hourly leaderboard.
