# Reflexive Selection Bias in Deployed Multimodal Fusion

## Overview

This repository contains simulation code and data for reproducing the experiments how multimodal fusion systems can create feedback loops where the model's own predictions influence future data acquisition, and we propose causal corrections for the resulting bias.

The single script `simulation_experiments.py` runs all eleven experiments and generates the corresponding figures.

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

| # | Experiment | Function | Figure(s) |
|---|-----------|----------|-----------|
| 1 | Bias vs. Policy Selectivity | `run_exp1` | fig1_bias_vs_selectivity |
| 2 | Decision Boundary Analysis | `run_exp2` | fig2_decision_boundaries |
| 3 | Acquisition Budget Analysis | `run_exp3` | fig3_acquisition_efficiency |
| 4 | Finite-Sample Convergence | `run_exp4` | fig4_ipw_convergence |
| 5 | Reflexive MNAR with Noisy Scan Policy | `run_exp5` | fig5_reflexive_mnar |
| 6 | Architecture Generalisation (MLP) | `run_exp6_mlp` | fig6_mlp_arch |
| 7 | Feedback Loop Dynamics | `run_exp_feedback_loop` | fig7_feedback_loop |
| 8 | Semi-Synthetic Heart Disease | `run_exp7_semisynthetic` | fig7_heart_semisynthetic |
| 9 | Baseline Comparison | `run_exp8_baseline_and_calib` | fig8_baseline_comparison |
| 10 | SUPPORT Study, Real Clinical Data | `run_exp_support` | fig10_support_results |
| 11 | Multi-Feature Scalability | `run_exp11_multifeature` | fig11_multifeature_scaling |

## License

MIT
