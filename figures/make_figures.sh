#!/bin/bash
# Render the paper's figures from the CSVs in figures/data/.
#
#   bash figures/make_figures.sh            # writes PDFs to figures/output/
#
# Needs R with ggplot2, dplyr, tidyr, readr and cowplot.
set -euo pipefail
cd "$(dirname "$0")/.."
D=figures/data
OUT=${1:-figures/output}
mkdir -p "$OUT"
Rscript figures/R/figure1_context.R    "$D" "$OUT" "$D/fig1_settings.csv"
Rscript figures/R/paper_figures.R      "$D" "$OUT"
Rscript figures/R/figure4_jobtype.R    "$D" "$OUT"
Rscript figures/R/figure_calibration.R "$D" "$OUT"
ls "$OUT"/*.pdf
