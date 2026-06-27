"""
Reflexive-MNAR coverage sweep.

We extend the reflexive noisy-scan DGP (run_exp5 reports a single
(selectivity, scan_noise) operating point) into a 2D sweep over

    * scan noise sigma_s  (small => strongly reflexive / TRUE MNAR;
                           large => the scan is uninformative about x2, so the
                           policy depends on x1 + noise only => effectively MAR)
    * policy selectivity  lambda

For each setting, averaged over >=20 trials, we report the EXPENSIVE-COEFFICIENT
BIAS (the fitted x2 coefficient minus the Oracle x2 coefficient) and the test
ACCURACY for five estimators:

    Oracle  : logistic fit on full (x1, x2) population               (reference)
    ERM     : logistic fit on ALL rows, x2 zero-imputed where missing (biased)
    CC-ERM  : logistic fit on complete cases only, re-included, NO reweighting
    CC-IPW  : complete cases, IPW with ESTIMATED propensity on [x1, s2, |x1+s2|]
    DIME    : gradient-split per-feature IPW (uplift coefficient on M*x2)

HYPOTHESIS UNDER TEST: across the MNAR->MAR sweep, ERM and CC-ERM stay biased
while CC-IPW and DIME track the Oracle.

Everything is logged to stdout; a figure (PDF+PNG) and a JSON of all numbers,
settings and the seed are written to ./figures/ and ./results/.

Run:
    conda run -n veri_bilimi python exp_reflexive_sweep.py
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import simulation_experiments as S

# ---------------------------------------------------------------------------
# Output locations (local repo paths)
# ---------------------------------------------------------------------------
FIG_DIR = HERE / "figures"
RESULTS_DIR = HERE / "results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
FIG_PDF = str(FIG_DIR / "reflexive_sweep.pdf")
FIG_PNG = str(FIG_DIR / "reflexive_sweep.png")
JSON_OUT = str(RESULTS_DIR / "reflexive_sweep.json")

# ---------------------------------------------------------------------------
# Pre-specified settings (FIXED; not tuned to a favorable outcome)
# ---------------------------------------------------------------------------
SEED = 20240613               # fixed master seed
N_TRAIN = 3000                # matches run_exp5 default
N_TEST = 3000                 # matches run_exp5 default
N_BIG = 50000                 # population size for the Oracle fit
N_TRIALS = 30                 # >= 20 for stable mean +/- std

# scan-noise sweep: small sigma_s => strongly reflexive MNAR;
# large sigma_s => scan ~ pure noise => policy depends on x1 only => MAR.
SCAN_NOISE_GRID = [0.25, 0.5, 1.0, 1.5, 2.5, 4.0, 8.0]
# policy selectivity sweep (held as a small grid; the headline figure fixes the
# middle selectivity, but we report all selectivities in the JSON).
SELECTIVITY_GRID = [1.0, 2.0, 3.0]
SELECTIVITY_MAIN = 2.0        # selectivity used for the headline figure
PI_MIN = 0.10                 # positivity floor (matches run_exp5)
CLIP = (0.05, 0.95)           # propensity clipping for IPW (matches run_exp5)

METHODS = ["Oracle", "ERM", "CC-ERM", "CC-IPW", "DIME"]


# ---------------------------------------------------------------------------
# Rank-based AUC helper (no sklearn dependency; ties handled by averaging).
# ---------------------------------------------------------------------------
def rank_auc(scores, labels):
    """Area under the ROC curve via the Mann-Whitney U statistic.

    AUC = P(score_pos > score_neg) with ties counted as 0.5.  Implemented with
    average ranks so it is exact and O(n log n).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # 1-based average rank for the tie block
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ---------------------------------------------------------------------------
# One reflexive-MNAR trial for a given (selectivity, scan_noise).
# Returns per-method (x2_coef, accuracy, auc) plus the acquisition rate.
# DGP is exactly run_exp5's reflexive noisy-scan policy.
# ---------------------------------------------------------------------------
def run_one_trial(selectivity, scan_noise, trial, X_full_te, y_te,
                  x1_te, x2_te, w_oracle):
    rng = np.random.RandomState(trial * 31 + 5)
    x1_tr, x2_tr, y_tr = S.generate_data(N_TRAIN, seed=trial * 7 + 1)

    # Noisy cheap scan s2 = x2 + N(0, scan_noise^2).
    s2_tr = x2_tr + rng.randn(N_TRAIN) * scan_noise

    # TRUE MNAR policy: acquire when the scan is uncertain (|x1 + s2| small).
    pi_true = np.maximum(S.sigmoid(-selectivity * np.abs(x1_tr + s2_tr)), PI_MIN)
    acquired = rng.binomial(1, pi_true).astype(bool)
    acq_rate = float(acquired.mean())

    out = {}

    # --- Oracle x2 coefficient (population reference; same across trials) ---
    out["Oracle"] = {
        "x2coef": float(w_oracle[1]),
        "acc": float(S.logistic_acc(X_full_te, y_te, w_oracle)),
        "auc": rank_auc(S.sigmoid(X_full_te @ w_oracle), y_te),
    }

    # --- ERM (zero-imputed over ALL rows) ---
    x2_imp = x2_tr.copy()
    x2_imp[~acquired] = 0.0
    X_biased = S.build_feature_matrix(x1_tr, x2_imp)
    w_erm = S.logistic_fit(X_biased, y_tr)
    out["ERM"] = {
        "x2coef": float(w_erm[1]),
        "acc": float(S.logistic_acc(X_full_te, y_te, w_erm)),
        "auc": rank_auc(S.sigmoid(X_full_te @ w_erm), y_te),
    }

    # Complete-case design matrix used by CC-ERM and CC-IPW.
    n_cc = int(acquired.sum())
    if n_cc >= 20:
        X_cc = S.build_feature_matrix(x1_tr[acquired], x2_tr[acquired])
        y_cc = y_tr[acquired]

        # --- CC-ERM (complete cases re-included, no reweighting) ---
        w_cc_erm = S.logistic_fit(X_cc, y_cc)
        out["CC-ERM"] = {
            "x2coef": float(w_cc_erm[1]),
            "acc": float(S.logistic_acc(X_full_te, y_te, w_cc_erm)),
            "auc": rank_auc(S.sigmoid(X_full_te @ w_cc_erm), y_te),
        }

        # --- CC-IPW with ESTIMATED propensity on [x1, s2, |x1+s2|, 1] ---
        X_prop = np.column_stack([x1_tr, s2_tr, np.abs(x1_tr + s2_tr),
                                  np.ones(N_TRAIN)])
        w_prop = S.logistic_fit(X_prop, acquired.astype(int))
        pi_hat = np.clip(S.sigmoid(X_prop @ w_prop), CLIP[0], CLIP[1])
        w_ipw = S.logistic_fit(X_cc, y_cc, weights=1.0 / pi_hat[acquired])
        out["CC-IPW"] = {
            "x2coef": float(w_ipw[1]),
            "acc": float(S.logistic_acc(X_full_te, y_te, w_ipw)),
            "auc": rank_auc(S.sigmoid(X_full_te @ w_ipw), y_te),
        }
    else:
        for m in ("CC-ERM", "CC-IPW"):
            out[m] = {"x2coef": np.nan, "acc": np.nan, "auc": np.nan}

    # --- DIME (gradient-split per-feature IPW) ---
    # Single expensive feature (J=1).  x1 is the 1-D cheap feature.
    x1_col = x1_tr.reshape(-1, 1)
    dime_p = S.dime_train(x1_col, [x2_tr], [acquired], y_tr.astype(float))
    # Effective x2 coefficient at test time (M=1): the uplift beta_0 lives at
    # index d1 + J + 0 in the DIME parameter vector.
    d1, J = dime_p["d1"], dime_p["J"]
    dime_x2coef = float(dime_p["w"][d1 + J + 0])
    # Evaluate with x2 fully observed at test time (M=1 everywhere).
    m_te_full = [np.ones(N_TEST, dtype=bool)]
    prob_dime = S.dime_predict(x1_te.reshape(-1, 1), [x2_te], m_te_full, dime_p)
    out["DIME"] = {
        "x2coef": dime_x2coef,
        "acc": float(((prob_dime > 0.5) == y_te).mean()),
        "auc": rank_auc(prob_dime, y_te),
    }

    return out, acq_rate


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    print("=" * 72)
    print(" Reflexive-MNAR coverage sweep")
    print("=" * 72)
    print(f"  seed={SEED}  N_train={N_TRAIN}  N_test={N_TEST}  trials={N_TRIALS}")
    print(f"  scan_noise grid = {SCAN_NOISE_GRID}")
    print(f"  selectivity grid = {SELECTIVITY_GRID}  (main figure sel={SELECTIVITY_MAIN})")
    print(f"  pi_min={PI_MIN}  clip={CLIP}")

    # Fixed test set and Oracle (population fit), identical across the whole sweep.
    x1_te, x2_te, y_te = S.generate_data(N_TEST, seed=777)
    X_full_te = S.build_feature_matrix(x1_te, x2_te)
    x1_big, x2_big, y_big = S.generate_data(N_BIG, seed=0)
    w_oracle = S.logistic_fit(S.build_feature_matrix(x1_big, x2_big), y_big)
    oracle_acc = float(S.logistic_acc(X_full_te, y_te, w_oracle))
    oracle_x2coef = float(w_oracle[1])
    oracle_auc = rank_auc(S.sigmoid(X_full_te @ w_oracle), y_te)
    print(f"\n  Oracle: x2coef={oracle_x2coef:.4f}  acc={oracle_acc*100:.2f}%  "
          f"AUC={oracle_auc:.4f}")

    # results[sel][sigma][method] -> dict of arrays
    results = {}
    for sel in SELECTIVITY_GRID:
        results[sel] = {}
        print("\n" + "-" * 72)
        print(f" selectivity lambda = {sel}")
        print("-" * 72)
        for sigma in SCAN_NOISE_GRID:
            per_method = {m: {"x2coef": [], "acc": [], "auc": []} for m in METHODS}
            acq_rates = []
            for trial in range(N_TRIALS):
                out, acq = run_one_trial(sel, sigma, trial, X_full_te, y_te,
                                         x1_te, x2_te, w_oracle)
                acq_rates.append(acq)
                for m in METHODS:
                    per_method[m]["x2coef"].append(out[m]["x2coef"])
                    per_method[m]["acc"].append(out[m]["acc"])
                    per_method[m]["auc"].append(out[m]["auc"])

            # Reduce to mean/std and bias (coef - oracle_coef).
            summary = {"acq_rate": float(np.mean(acq_rates))}
            for m in METHODS:
                coefs = np.array(per_method[m]["x2coef"], dtype=float)
                accs = np.array(per_method[m]["acc"], dtype=float)
                aucs = np.array(per_method[m]["auc"], dtype=float)
                bias = coefs - oracle_x2coef
                summary[m] = {
                    "x2coef_mean": float(np.nanmean(coefs)),
                    "x2coef_std": float(np.nanstd(coefs)),
                    "bias_mean": float(np.nanmean(bias)),
                    "bias_std": float(np.nanstd(bias)),
                    "abs_bias_mean": float(np.nanmean(np.abs(bias))),
                    "acc_mean": float(np.nanmean(accs)),
                    "acc_std": float(np.nanstd(accs)),
                    "auc_mean": float(np.nanmean(aucs)),
                    "auc_std": float(np.nanstd(aucs)),
                }
            results[sel][sigma] = summary

            print(f"\n  sigma_s={sigma:<5}  acq_rate={summary['acq_rate']*100:5.1f}%")
            print(f"    {'method':8s} {'x2coef':>16s} {'bias(coef)':>16s} "
                  f"{'acc(%)':>14s} {'AUC':>14s}")
            for m in METHODS:
                s = summary[m]
                print(f"    {m:8s} "
                      f"{s['x2coef_mean']:7.3f}+/-{s['x2coef_std']:5.3f} "
                      f"{s['bias_mean']:7.3f}+/-{s['bias_std']:5.3f} "
                      f"{s['acc_mean']*100:6.2f}+/-{s['acc_std']*100:4.2f} "
                      f"{s['auc_mean']:6.3f}+/-{s['auc_std']:4.3f}")

    # -------------------------------------------------------------------
    # Outcome computation (on the MAIN selectivity).
    # Hypothesis: ERM & CC-ERM stay biased; CC-IPW & DIME track the Oracle.
    # -------------------------------------------------------------------
    main_res = results[SELECTIVITY_MAIN]
    abs_bias = {m: np.mean([main_res[sig][m]["abs_bias_mean"]
                            for sig in SCAN_NOISE_GRID]) for m in METHODS}
    print("\n" + "=" * 72)
    print(f" Mean |coef bias| across the scan-noise sweep (lambda={SELECTIVITY_MAIN}):")
    for m in METHODS:
        print(f"    {m:8s}: {abs_bias[m]:.4f}")

    # A method "tracks the oracle" if its mean abs bias is much smaller than
    # ERM's; "stays biased" if comparable to ERM. Use ERM as the bias yardstick.
    erm_bias = abs_bias["ERM"]
    corrected = {m: abs_bias[m] for m in ("CC-IPW", "DIME")}
    biased = {m: abs_bias[m] for m in ("ERM", "CC-ERM")}
    # Strong support: both corrected methods <50% of ERM bias AND both biased
    # methods >50% of ERM bias.
    corrected_ok = all(v < 0.5 * erm_bias for v in corrected.values())
    biased_stays = all(v > 0.5 * erm_bias for v in biased.values())
    supports = bool(corrected_ok and biased_stays)
    print(f"\n  ERM bias yardstick = {erm_bias:.4f}")
    print(f"  CC-IPW < 0.5*ERM ? {abs_bias['CC-IPW'] < 0.5*erm_bias}  "
          f"DIME < 0.5*ERM ? {abs_bias['DIME'] < 0.5*erm_bias}")
    print(f"  CC-ERM stays biased (>0.5*ERM) ? {abs_bias['CC-ERM'] > 0.5*erm_bias}")
    print(f"\n  >>> supportsClaim = {supports}")

    # -------------------------------------------------------------------
    # Figure: bias and accuracy vs scan noise (main selectivity).
    # -------------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 12,
        "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 2.2,
    })
    style = {
        "Oracle": ("black", "--", "o"),
        "ERM": ("red", "-", "o"),
        "CC-ERM": ("darkorange", "-", "^"),
        "CC-IPW": ("royalblue", "-", "s"),
        "DIME": ("green", "-", "D"),
    }
    sig = np.array(SCAN_NOISE_GRID, dtype=float)
    fig, (axb, axa) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) expensive-coefficient bias vs scan noise
    axb.axhline(0.0, color="black", ls="--", lw=1.5, label="Oracle (no bias)")
    for m in ("ERM", "CC-ERM", "CC-IPW", "DIME"):
        col, ls, mk = style[m]
        means = np.array([main_res[s][m]["bias_mean"] for s in SCAN_NOISE_GRID])
        stds = np.array([main_res[s][m]["bias_std"] for s in SCAN_NOISE_GRID])
        axb.plot(sig, means, color=col, ls=ls, marker=mk, label=m, markersize=7)
        axb.fill_between(sig, means - stds, means + stds, color=col, alpha=0.12)
    axb.set_xscale("log")
    axb.set_xticks(sig)
    axb.set_xticklabels([str(s) for s in SCAN_NOISE_GRID])
    axb.set_xlabel(r"Scan noise $\sigma_s$  (small=MNAR  $\rightarrow$  large=MAR)")
    axb.set_ylabel(r"Expensive-coef bias ($\hat\beta_{x_2}-\beta^{\mathrm{oracle}}_{x_2}$)")
    axb.set_title(f"(a) Coefficient bias vs scan noise ($\\lambda$={SELECTIVITY_MAIN})")
    axb.legend(fontsize=9)

    # (b) accuracy vs scan noise
    for m in METHODS:
        col, ls, mk = style[m]
        means = np.array([main_res[s][m]["acc_mean"] for s in SCAN_NOISE_GRID]) * 100
        stds = np.array([main_res[s][m]["acc_std"] for s in SCAN_NOISE_GRID]) * 100
        axa.plot(sig, means, color=col, ls=ls, marker=mk, label=m, markersize=7)
        axa.fill_between(sig, means - stds, means + stds, color=col, alpha=0.12)
    axa.set_xscale("log")
    axa.set_xticks(sig)
    axa.set_xticklabels([str(s) for s in SCAN_NOISE_GRID])
    axa.set_xlabel(r"Scan noise $\sigma_s$  (small=MNAR  $\rightarrow$  large=MAR)")
    axa.set_ylabel("Test accuracy (%)")
    axa.set_title(f"(b) Test accuracy vs scan noise ($\\lambda$={SELECTIVITY_MAIN})")
    axa.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        "Reflexive MNAR-to-MAR coverage sweep",
        fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_PDF, bbox_inches="tight")
    fig.savefig(FIG_PNG, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\n  Saved figure: {FIG_PDF}")
    print(f"                {FIG_PNG}")

    # -------------------------------------------------------------------
    # JSON dump of everything.
    # -------------------------------------------------------------------
    payload = {
        "id": "reflexive_sweep",
        "description": "Reflexive-MNAR coverage sweep over scan noise sigma_s "
                       "and policy selectivity; expensive-coefficient bias and "
                       "test accuracy for Oracle/ERM/CC-ERM/CC-IPW/DIME.",
        "settings": {
            "seed": SEED,
            "N_train": N_TRAIN, "N_test": N_TEST, "N_big": N_BIG,
            "n_trials": N_TRIALS,
            "scan_noise_grid": SCAN_NOISE_GRID,
            "selectivity_grid": SELECTIVITY_GRID,
            "selectivity_main": SELECTIVITY_MAIN,
            "pi_min": PI_MIN, "clip": list(CLIP),
            "methods": METHODS,
            "dgp": "run_exp5 reflexive noisy-scan: s2=x2+N(0,sigma_s^2); "
                   "pi=max(sigmoid(-lambda*|x1+s2|), pi_min)",
            "metric_bias": "fitted x2 coefficient minus Oracle x2 coefficient",
            "auc": "rank-based (Mann-Whitney) AUC helper defined in script",
        },
        "oracle": {"x2coef": oracle_x2coef, "acc": oracle_acc, "auc": oracle_auc},
        "results": {
            str(sel): {
                str(sig): results[sel][sig] for sig in SCAN_NOISE_GRID
            } for sel in SELECTIVITY_GRID
        },
        "mean_abs_coef_bias_main_selectivity": abs_bias,
        "erm_bias_yardstick": float(erm_bias),
        "supportsClaim": supports,
        "figure_pdf": FIG_PDF, "figure_png": FIG_PNG, "json": JSON_OUT,
    }
    with open(JSON_OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved JSON:   {JSON_OUT}")
    print("\nDone.")
    return supports


if __name__ == "__main__":
    main()
