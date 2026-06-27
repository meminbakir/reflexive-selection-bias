# Reflexive Selection Bias in Deployed Multimodal Fusion

Code for the experiments in the paper. A deployed fusion model often decides when to acquire an expensive modality based on a cheaper one, then retrains on the data it collected. That feedback loop biases the model. We study the bias and correct it with inverse-propensity weighting (CC-IPW) and a gradient-split variant (DIME).

## Requirements

Python 3.9+ with NumPy, SciPy, and Matplotlib.

```bash
pip install -r requirements.txt
```

## Running

`simulation_experiments.py` runs the main experiments:

```bash
python simulation_experiments.py
```

The `exp_*.py` files are separate experiments. Run any of them directly:

```bash
python exp_reflexive_sweep.py
```

Figures go to `figures/`, numerical results to `results/`. Both are created on first run and are not committed.

## Data

- Synthetic data is generated in code.
- UCI Cleveland Heart Disease: `heart_test.csv` (test split). Full dataset at the [UCI repository](https://archive.ics.uci.edu/dataset/45/heart+disease).
- SUPPORT Study: download `support2.csv` from the [Vanderbilt Biostatistics repository](https://hbiostat.org/data/repo/support2csv.zip) and place it in this folder.

## Experiments

In `simulation_experiments.py`:

| # | Experiment | Function |
|---|---|---|
| 1 | Bias vs. policy selectivity | `run_exp1` |
| 2 | Decision boundary analysis | `run_exp2` |
| 3 | Acquisition budget | `run_exp3` |
| 4 | Finite-sample convergence | `run_exp4` |
| 5 | Reflexive MNAR with a noisy scan policy | `run_exp5` |
| 6 | Architecture generalisation (MLP) | `run_exp6_mlp` |
| 7 | Feedback-loop dynamics | `run_exp_feedback_loop` |
| 8 | Semi-synthetic heart disease | `run_exp7_semisynthetic` |
| 9 | Baseline comparison | `run_exp8_baseline_and_calib` |
| 10 | SUPPORT study (real clinical data) | `run_exp_support` |
| 11 | Multi-feature scalability | `run_exp11_multifeature` |

Standalone scripts:

| Script | Experiment |
|---|---|
| `exp_bias_recursion.py` | Bias recursion under deploy-retrain |
| `exp_reflexive_sweep.py` | Reflexive MNAR sweep |
| `exp_deep_fusion.py` | Deep, higher-dimensional fusion |
| `exp_misspecification.py` | Propensity misspecification |
| `exp_clip_sweep.py` | Clip-threshold sensitivity |
| `exp_dependent_acquisition.py` | Dependent acquisition |

## License

MIT
