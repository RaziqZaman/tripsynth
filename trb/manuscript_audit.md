# Manuscript Audit

Audit date: 2026-07-30  
Final frozen repository basis: branch `wctr-aadt`, commit `3363e21ed781af9c5d659c3fed6d08ea661170e2`

## Final manuscript

- **Title:** *Generative Trip Synthesis Halves Sparse Date--Hour Error but Not Pooled Volume Error: An External Count Audit of a Contrastive Variational Autoencoder*
- **Primary source:** `trb/trb_template.tex`
- **Bibliography:** `trb/trb_template.bib`
- **Compiled proof:** `trb/trb_template.pdf`
- **Final page count:** 17 pages, including title, structured abstract, all figures/tables, declarations, and references.
- **Abstract:** page 2 only; 238 source words including the five headings; headings occur in the exact required order; no citations, figures, tables, equations, or undefined acronyms.
- **References:** 49 cited and 49 listed; zero missing/uncited keys; alphabetical Chicago author--date output.
- **Figures/tables:** 6 figures and 10 tables.

## Compilation and production status

Successful local command, run from `trb/`:

```bash
tectonic --keep-logs --keep-intermediates trb_template.tex
```

Tectonic 0.17.0 completed BibTeX and all required reruns and wrote `trb_template.pdf`. The proof is US Letter (612 x 792 points), one column, 10-point Times-compatible text, 1-inch margins, centered bottom page numbers, and pagewise-restarting line numbers. Every final page was visually inspected. No page is clipped, overflowing, blank, or broken. Figures 5 and 6 were regenerated at 9-point source label size so their labels remain at least 8 points after final scaling.

There are no overfull boxes, undefined citations, undefined references, or missing manuscript glyphs. Two harmless underfull boxes remain in compact abstract/title material and have no visible defect. Tectonic emits an environment-level UTF-8 warning in bundled `lineno.sty` and informational font-request messages; these do not originate in manuscript text. BibTeX warns that the online-first SyncMove paper lacks assigned volume/pages and that the arXiv CPC preprint has no pages. Those fields were unavailable and were not fabricated. Float-only pages 4 and 10 contain no prose lines for `lineno` to number; numbered prose restarts at 1 on all pages containing prose.

The package uses relative paths and standard Overleaf packages. It selects standard `chicago.bst` to meet the explicit Chicago requirement; the supplied `trb.bst` is retained unchanged as a bundled template artifact.

## Verified principal results

### Principal exact-date hourly result

On 521,424 common boundary--date--hour cells spanning 56 boundaries, 37 stations, 427 dates, and all 24 hours, every method was evaluated with the same 423-date factor.

- Direct weighted expansion RMSE: **13,553.1735 vehicles**
- Contrastive-VAE RMSE: **6,919.8886 vehicles**
- Relative reduction: **48.9427%**
- Paired station-cluster 95% bootstrap CI: **42.0811%--56.6092%**
- Clusters/replicates: **37 stations / 10,000 replicates**

The noncontrastive VAE has RMSE 6,909.7027. Contrastive minus noncontrastive RMSE is +10.1859 vehicles (95% CI -86.7646 to 79.0992), so the large exact-date gain is attributed to generative smoothing, not to the contrastive term.

### Required contrary hourly evidence

After pooling dates to 1,344 common boundary-hour cells, RMSE is 6,083.944 for direct expansion, 6,085.613 for bootstrap, 5,916.489 for Bayesian network, 5,929.232 for noncontrastive VAE, and 6,192.574 for contrastive VAE. Contrastive minus direct is +108.630 vehicles (station-cluster 95% CI -29.741 to 257.277). The contrastive term does improve normalized profile shape: profile TV is 0.196686 versus 0.249352 for its ablation, a 21.1214% reduction (95% CI 16.6138%--25.7223%); Bayesian network remains best at 0.140795.

### Main annual result

Across 3,107 common boundaries, Bayesian network has the lowest aggregate RMSE (**55,771.985 vehicles**) and contrastive VAE the highest (**57,842.981 vehicles**). Unique five-way local wins, excluding one five-way zero tie, are direct 342, bootstrap 359, Bayesian network 772, noncontrastive VAE 808, and contrastive VAE 825. Contrastive wins are low-volume concentrated (426 in quartile 1 and 14 in quartile 4) and contain only 7.46% of observed volume. The manuscript therefore reports wins as a descriptive rank and not evidence of aggregate superiority.

## Methods and uncertainty

Compared alternatives:

1. deterministic direct expansion of retained household-final weights;
2. one-million-row weighted bootstrap;
3. one-million-row weighted Chow--Liu Bayesian network;
4. one-million-row noncontrastive VAE;
5. one-million-row otherwise matched contrastive VAE.

Statistical procedures:

- paired cluster bootstrap over 37 hourly stations, retaining all linked boundaries/dates/hours, 10,000 replicates, seed 20260730;
- paired station-cluster contrasts for exact-date RMSE reduction/difference, pooled RMSE difference, and profile-TV reduction;
- two-sided paired Wilcoxon signed-rank comparisons across 26 categorical fields, 18 numeric fields, and nine stored cross-tables; treated as descriptive because endpoints are dependent and not multiplicity-adjusted;
- annual win counts exclude ties and are accompanied by aggregate error, volume strata, and paired error distributions rather than significance claims.

## Three explicit review passes

### Pass 1: Scientific reviewer

The archived 50.8985% leaderboard was found not to be like-for-like because methods used different calendar cells and temporal multipliers. The manuscript was corrected to lead with a reconstructed deterministic direct benchmark on identical keys and scaling. A 37-station clustered interval, a noncontrastive ablation contrast, pooled-date sensitivity, annual aggregate errors, low-volume win concentration, scaling/path caveats, and the absence of formal privacy were added. Conclusions now distinguish sparse-cell smoothing, hourly shape, total volume, and network calibration.

### Pass 2: TRB reviewer

The transportation contribution is stated in the title, structured abstract, last Introduction paragraph, contribution table, dominant hourly figure, Discussion, and Conclusions. The literature review narrows rather than overstates novelty and acknowledges the closest count-validation precedent. The abstract, title page, margins, page cap, practical applications, embedded figures/tables, and back matter were checked against the supplied 2027 brief.

### Pass 3: Copy editor and production reviewer

Terminology, acronym definitions, author--date citations, alphabetical bibliography, labels/cross-references, units, precision, table widths, float placement, grayscale/color accessibility, and final-size figure fonts were checked. All 17 PDF pages and all six source figures were visually inspected. The final build has no overfull box or unresolved reference/citation warning.

## Submission blockers and author verification

The following visible placeholders or decisions must be resolved before submission:

1. job title for each of the five authors;
2. confirmation that Raziq Zaman is the corresponding author;
3. author order, affiliations, email addresses, and any optional ORCiD values;
4. confirmed acknowledgments, or an explicit statement that there are none;
5. CRediT roles for every author;
6. conflict-of-interest declaration;
7. funding statement and grant numbers, or an explicit no-specific-funding statement;
8. author approval of the generative-AI disclosure;
9. provider rights/access language for transformed travel-survey data;
10. source-dictionary confirmation of trip/mode definitions and correct operational use of the retained household-final weight;
11. confirmation of the complete human model/hyperparameter selection history;
12. confirmation that prior WCTR-related dissemination does not conflict with the intended TRB presentation/publication track;
13. submission-date verification when the final portal copy is compiled.

The absence of source dictionaries, some preprocessing programs, generated one-million-row CSVs, pinned dependencies/hardware, and a household-level split audit remains a scientific limitation rather than a fillable manuscript placeholder. The straight-line screenline proxy is not road assignment. No formal privacy guarantee is claimed.

## AI-use disclosure requiring author approval

> OpenAI Codex was used for repository and code inspection, targeted literature discovery and metadata-verification support, statistical reanalysis, reproducible visualization and table code, and manuscript drafting, editing, and reference formatting. The authors must independently review and verify the manuscript, analyses, numerical results, figures, declarations, and references before submission.

This text is present in the manuscript and visibly marked `[AUTHOR APPROVAL REQUIRED.]`.

## Files created or modified

Primary package:

- `trb/trb_template.tex`
- `trb/trb_template.bib`
- `trb/trb_template.pdf`
- `trb/trbunofficial.cls`
- `trb/trb.bst` (unchanged copy of the supplied file)
- `trb/generated_results.tex`
- `trb/README.md`
- `trb/evidence_manifest.md`
- `trb/manuscript_audit.md`

Reproducible analysis:

- `trb/scripts/generate_manuscript_assets.py`
- `trb/generated/exact_date_common_support_predictions.csv`
- `trb/generated/annual_screenline_error_differences.csv`

Figures (vector proof and raster inspection copy):

- `trb/figures/fig_01_workflow.pdf` and `.png`
- `trb/figures/fig_02_architecture.pdf` and `.png`
- `trb/figures/fig_03_internal_tradeoff.pdf` and `.png`
- `trb/figures/fig_04_screenline_method.pdf` and `.png`
- `trb/figures/fig_05_hourly_granularity.pdf` and `.png`
- `trb/figures/fig_06_annual_validation.pdf` and `.png`

Tables and machine-readable summaries:

- `trb/tables/literature_deep.tex`
- `trb/tables/literature_trip.tex`
- `trb/tables/literature_external.tex`
- `trb/tables/contributions.tex`
- `trb/tables/survey_data_summary.tex` and `.csv`
- `trb/tables/traffic_count_summary.tex` and `.csv`
- `trb/tables/internal_results.tex`
- `trb/tables/external_results.tex`
- `trb/tables/annual_results.tex`
- `trb/tables/internal_validation_results.csv`
- `trb/tables/internal_paired_tests.csv`
- `trb/tables/exact_date_common_support_results.csv`
- `trb/tables/exact_date_clustered_contrasts.csv`
- `trb/tables/hourly_validation_results.csv`
- `trb/tables/paired_hourly_effects.csv`
- `trb/tables/annual_validation_results.csv`

Compiler byproducts retained from the verified build: `trb/trb_template.aux`, `trb/trb_template.bbl`, `trb/trb_template.blg`, `trb/trb_template.log`, and `trb/trb_template.out`. They are not research inputs.
