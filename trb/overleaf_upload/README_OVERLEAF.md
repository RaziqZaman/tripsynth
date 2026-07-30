# Uploading to Overleaf

1. Create a new blank Overleaf project.
2. Upload the contents of this folder, preserving the `figures/` and `tables/`
   subfolders. Alternatively, upload the ZIP directly using Overleaf's
   **New Project → Upload Project** option.
3. Set `trb_template.tex` as the main document if Overleaf does not select it
   automatically.
4. Compile with pdfLaTeX and BibTeX. No shell escape or external data files are
   required.

The package includes `chicago.bst`, so bibliography compilation does not rely
on a machine-specific cache.

Before submission, replace every visible `[VERIFY ...]` and
`[AUTHOR APPROVAL REQUIRED]` marker in `trb_template.tex`.
