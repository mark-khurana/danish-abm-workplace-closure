# Workplace closure and who it protects: an agent-based model of Denmark

Code and figure data for the paper

> **Indirect protection and unequal reach of workplace closure across income groups in Denmark**

The model represents every resident of Denmark as an individual, connected through
five layers of recorded contacts (household, family, neighbourhood, school and
workplace) plus an age-structured community channel. It compares the
winter 2020–21 reduction in workplace contact with a counterfactual in which workplace
contact continued, and asks how the infections it prevented were shared across
workplace groups, income quartiles and occupations.

This repository lets you

1. **run the model** end to end on a synthetic population, in about two minutes on a
   laptop, and
2. **redraw every figure** in the paper from the aggregate results.

No individual-level data are included (see [Data](#data)).

---

## Quick start

```bash
# 1. install (Python 3.10 or later)
pip install -r requirements.txt
pip install --no-deps agent-torch==0.6.0

# 2. run the model on a synthetic population of 20,000 people
python examples/run_demo.py

# 3. redraw the paper's figures (needs R, see below)
bash figures/make_figures.sh
```

Step 2 builds a synthetic population, simulates the three scenarios for four parameter
draws and prints infections prevented by workplace group and income quartile. The
population is invented, so the numbers are illustrative only: they show what the code
does, not what happened in Denmark. Use `--runs` for more draws and `--n` for a larger
population.

Step 3 writes five PDFs to `figures/output/`, identical to the figures in the paper.

---

## What the model does

For each parameter draw the same population is simulated under three scenarios, with
the same random seed:

| Scenario | Workplace contact | Displaced contact into homes and neighbourhoods |
|---|---|---|
| A | reduced, as observed | yes |
| B | not reduced | no |
| C | reduced, as observed | no |

**B − C** is the protection given by the reduction in workplace contact, and
**A − C** is the extra infections from contact that moved elsewhere. Both are
computed for every group of people, within each draw, and summarised as medians with
95% uncertainty intervals across draws.

The reduction in workplace contact is taken from the Google Community Mobility Reports
and applied to every colleague link, either equally or weighted by the teleworkability
of each occupational group (set `ABM_TELEWORK=weighted`).

## Repository layout

```
danish_abm/            the model: network, transmission, disease progression,
                       testing, vaccination and interventions (built on AgentTorch)
  data/                public inputs: OxCGRT, Google mobility, Prem et al. contact
                       matrices, national SSI daily series (see data/SOURCES.md)
decomposition/         runs scenarios A/B/C and summarises them by group
config/                model parameters and input paths used by the demo
params/                fitted parameter values for the study period
synthetic/             generator for the synthetic population
examples/run_demo.py   the end-to-end example
figures/
  data/                aggregate results behind each figure
  R/                   plotting scripts
  make_figures.sh
tests/                 a quick end-to-end check
```

## Figures

| Figure | Script | Data |
|---|---|---|
| 1 Model structure, study period, prevented transmission by setting | `figure1_context.R` | `observed_series.csv`, `npi_schedule.csv`, `fig1_settings.csv` |
| 2 Protection by workplace size | `paper_figures.R` | `fig2_protection.csv`, `fig2_leakage.csv`, `headline.csv` |
| 3 Income gradients | `paper_figures.R` | `fig3_income.csv` |
| 4 Occupation and teleworkability | `figure4_jobtype.R` | `fig4_jobtype.csv`, `fig4_presence.csv` |
| Supplementary calibration figure | `figure_calibration.R` | `calibration_fit.csv`, `calibration_runs.txt` |

R packages: `ggplot2`, `dplyr`, `tidyr`, `readr`, `cowplot`.

## Data

The study used individual-level data from the Danish national registers and
Statens Serum Institut. **These cannot be shared under Danish law and are not in this
repository.** Access can be requested from the Danish Health Data Authority
(<https://english.sundhedsdatastyrelsen.dk/health-data-and-registers/research-services/apply-for-data>)
and, for the microbiology data, from Statens Serum Institut, by researchers at
authorised institutions.

What is included:

- **Aggregate results** (`figures/data/`): medians and uncertainty intervals by
  workplace group, income quartile and occupational group, each covering hundreds of
  thousands of people, and the national daily series of cases, hospital admissions and
  deaths published by Statens Serum Institut.
- **Fitted parameters** (`params/wave1_calibrated.json`): the weekly transmission scale,
  severity multipliers and setting weights used for the study period.
- **Public inputs** (`danish_abm/data/`), listed with their sources in
  `danish_abm/data/SOURCES.md`.
- **A synthetic population generator** (`synthetic/make_population.py`), with simple
  invented distributions. It is not derived from register data.

With register access, the model runs on the real population by pointing
`config/data_config.yaml` at a folder with the same layout as the synthetic one
(`agents.parquet`, one edge list per contact layer in `networks/`, and the test and
vaccination records in `events/`).

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## Licence

MIT, see `LICENSE`.
