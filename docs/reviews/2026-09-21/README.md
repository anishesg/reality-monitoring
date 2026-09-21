# Reality-monitoring paper audit and ICLR blueprint

Prepared 21 September 2026. Read `review.pdf` for the full report.

- `review.tex`: self-contained LaTeX source with clickable primary-source and commit-pinned repository citations.
- `audit.py`: CPU-only reproducibility checks; requires Python and NumPy.
- `audit.json`: all extracted summary contrasts, recomputed v17 rates, question-bootstrap intervals, exclusion bounds, and generator checks.
- `audit_tables.csv`: v17 recomputed results in tabular form.

Repository examined: https://github.com/anishesg/reality-monitoring
Commit: `d2e42b3f64483a48ee2b87c371f03957a0ddcce2`

Build the PDF:

```bash
pdflatex -interaction=nonstopmode -halt-on-error review.tex
pdflatex -interaction=nonstopmode -halt-on-error review.tex
```

Reproduce the audit against a checkout of that commit:

```bash
python3 audit.py /path/to/reality-monitoring
```

No new LLM inference or training was run. Recomputed rates use the original outcome labels and are not independent semantic regrades. The v16 comparisons come from committed summaries; corresponding raw trial files were unavailable in this snapshot. The confidence-conditioned SFT results are author-reported rather than independently reproduced. Proposed experiments and target abstract placeholders are explicitly marked in the report.
