# Reflexive Selection Bias in Deployed Multimodal Fusion

Code and data for reproducing the experiments in:

> **Reflexive Selection Bias in Deployed Multimodal Fusion: Causal Correction for Policy-Induced Feedback Loops**
> Mehmet E. Bakir, New Mexico State University

## Overview

This repository contains the simulation experiments that validate the theoretical results in the paper.  The single script `simulation_experiments.py` runs all eleven experiments and generates the corresponding figures.

## Requirements

- Python 3.9+
- NumPy >= 1.24
- Matplotlib >= 3.6

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run all experiments:

```bash
python simulation_experiments.py
```

Generated figures are saved to `figures/` and numerical results to `results/`.

## Data

- **Synthetic data** -- generated programmatically by the script.
- **UCI Cleveland Heart Disease** -- included as `heart_test.csv` (test split, 61 samples).  The full dataset is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).
- **SUPPORT Study** -- not redistributed here.  Download `support2.csv` from the [Vanderbilt Biostatistics repository](https://hbiostat.org/data/repo/support2csv.zip) and place it in the working directory.

## Experiments

| # | Experiment | Figure(s) |
|---|-----------|-----------|
| 1 | Bias vs. Selectivity | fig1 |
| 2 | Decision Boundary Visualization | fig2 |
| 3 | Acquisition Efficiency | fig3 |
| 4 | IPW Convergence | fig4 |
| 5 | Reflexive MNAR Feedback Loop | fig5, fig7 |
| 6 | Heart Disease Semi-Synthetic | fig7_heart |
| 7 | Baseline Comparison | fig8 |
| 8 | SUPPORT ICU Real-World | fig10 |
| 9 | Multi-Feature Scaling | fig11 |
| 10 | MLP Architecture | fig6 |

## License

MIT
