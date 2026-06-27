"""
Deep, higher-dimensional fusion under selective acquisition
===========================================================

A controlled study of whether inverse-propensity weighting (IPW) still helps a
deep fusion model when the expensive modality is higher-dimensional and carries
genuinely nonlinear signal.

DESIGN
------
Two modalities are fused by a single deep model:

  * Cheap modality   x1 in R^{d1} (d1 = 5)  -- ALWAYS observed. Carries a
    weak/moderate linear signal about the label.
  * Expensive modality x2 in R^{d2} (d2 = 12) -- observed only when a
    selective policy fires. Carries GENUINELY NONLINEAR signal about the label
    (pairwise products + squared terms), scaled by a signal-strength parameter
    s that is swept. A linear model in x2 cannot represent this functional, so
    a deep fusion model is warranted.

Label
  logit(y) = w1 . x1 + s * g(x2) + noise ;  y = 1[logit > 0],
  where g(x2) is a fixed nonlinear functional (cross terms + quadratics),
  z-standardised on a held-out population so that s directly controls the x2
  signal magnitude.

Acquisition policy (x1-gated, inducing selection on the x2 part)
  pi(x1) = max( sigmoid(-lambda * |score(x1)|), pi_min ),
  where score(x1) is a fixed linear readout of the cheap modality. x2 is
  acquired mainly when x1 is uninformative about y and rarely when x1 already
  decides y. pi_min enforces positivity (overlap) so IPW is well-defined.
  Because x2 is signal-bearing and acquisition depends on x1 (correlated with
  y), the acquired subsample is a biased view of the x1 -> x2 -> y relationship.

Estimators (all share the SAME deep MLP and identical hyperparameters)
  ERM     : all N rows, x2 zero-imputed where not acquired, uniform weights.
  CC-ERM  : acquired rows only (genuine full features), uniform weights
            (complete-case, no propensity weighting) -- ablation.
  IPW     : acquired rows only, weights = 1/pi_hat with pi_hat estimated from
            the cheap modality x1 -- the correction.
  Oracle  : all N rows with x2 fully observed, uniform weights -- the
            unattainable reference.

Two evaluations (both reported)
  full       : test x2 always available (intrinsic fusion quality).
  deployment : test x2 zero-imputed wherever the same policy did not acquire it
               (a realistic deployed selective-acquisition pipeline).

HYPOTHESIS under test
  Does the correction make the learned deep model closer to the oracle, and on
  which evaluation does it show? We explicitly distinguish "the correction
  improves the model" (full-feature evaluation) from "the correction fills a
  missing modality at test" (it cannot: a test point whose x2 was never
  acquired cannot benefit from any training-time fix).

Everything is averaged over N_TRIALS trials with a fixed master seed and
pre-specified settings. Numbers are printed to stdout, a matplotlib figure is
saved (PDF+PNG), and the numeric results are dumped to JSON.

Run:  conda run -n veri_bilimi python exp_deep_fusion.py
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the existing simulator's model-fitting primitives.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import simulation_experiments as S


# Output locations (local to this script).
EXP_ID = "deep_fusion"
FIG_DIR = HERE / "figures"
RESULTS_DIR = HERE / "results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ----------------------- Pre-specified settings -----------------------
SEED          = 20240617
D1            = 5          # cheap modality dim (always observed)
D2            = 12         # expensive modality dim (selectively acquired)
N_TRAIN       = 4000
N_TEST        = 8000
N_ORACLE      = 4000       # oracle uses same training budget but full x2
N_TRIALS      = 30
SIGNAL_GRID   = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]   # x2 signal strength sweep
LAMBDA        = 2.5        # policy selectivity
PI_MIN        = 0.08       # positivity floor for IPW overlap
NOISE_SD      = 0.6        # label logit noise

# Deep model hyperparameters -- identical for every estimator.
HIDDEN        = 32
LR            = 0.05
N_ITER        = 2500
L2            = 1e-4

# Fixed ground-truth generative parameters (frozen across all trials/signals).
_gen_rng = np.random.RandomState(SEED)
W1_TRUE  = _gen_rng.randn(D1) * 0.9          # cheap linear signal
# Nonlinear x2 functional: K random cross-term pairs + quadratic weights.
K_CROSS  = 8
CROSS_I  = _gen_rng.randint(0, D2, size=K_CROSS)
CROSS_J  = _gen_rng.randint(0, D2, size=K_CROSS)
CROSS_W  = _gen_rng.randn(K_CROSS)
QUAD_W   = _gen_rng.randn(D2) * 0.5
# Fixed "uncertainty readout" the policy uses on x1 (a cheap-model proxy).
S_READOUT = _gen_rng.randn(D1)


def g_x2(x2):
    """Genuinely nonlinear functional of x2 (cross terms + quadratics)."""
    cross = np.zeros(len(x2))
    for k in range(K_CROSS):
        cross += CROSS_W[k] * x2[:, CROSS_I[k]] * x2[:, CROSS_J[k]]
    quad = (x2 ** 2) @ QUAD_W
    return cross + quad


# Pre-compute population standardisation for g(x2) so the signal strength is
# comparable across the sweep (frozen, not re-estimated per trial).
_pop = np.random.RandomState(SEED + 1).randn(200000, D2)
_g_pop = g_x2(_pop)
G_MEAN, G_STD = _g_pop.mean(), _g_pop.std()


def generate(N, signal, seed):
    rng = np.random.RandomState(seed)
    x1 = rng.randn(N, D1)
    x2 = rng.randn(N, D2)
    g  = (g_x2(x2) - G_MEAN) / G_STD                 # z-scored nonlinear signal
    logit = x1 @ W1_TRUE + signal * g + NOISE_SD * rng.randn(N)
    y = (logit > 0).astype(int)
    return x1, x2, y


def policy_prob(x1):
    """pi(acquire | x1): high when x1 is uninformative (|score| ~ 0)."""
    score = x1 @ S_READOUT
    raw = 1.0 / (1.0 + np.exp(LAMBDA * np.abs(score)))
    return np.maximum(raw, PI_MIN)


def apply_policy(x1, rng):
    pi = policy_prob(x1)
    acquired = rng.binomial(1, pi).astype(bool)
    return acquired, pi


def build_X(x1, x2, mask=None):
    """Fusion feature matrix [x1 | x2 | bias]. If mask given, zero-impute x2
    where mask is False (deployment-style missingness)."""
    x2u = x2.copy()
    if mask is not None:
        x2u[~mask] = 0.0
    return np.column_stack([x1, x2u, np.ones(len(x1))])


def fit_mlp(X, y, weights, seed):
    return S.mlp_fit(X, y, weights=weights, hidden=HIDDEN, lr=LR,
                     n_iter=N_ITER, l2=L2, seed=seed)


def est_propensity(x1, acquired):
    """Estimate pi_hat from x1 via logistic_fit (cheap-feature propensity)."""
    Xp = np.column_stack([x1, np.ones(len(x1))])
    wp = S.logistic_fit(Xp, acquired.astype(float), lr=0.1, n_iter=1500)
    pi_hat = S.sigmoid(Xp @ wp)
    return np.clip(pi_hat, PI_MIN, 1.0 - 1e-3)


def main():
    print("=" * 78)
    print("Deep higher-dimensional fusion under selective acquisition")
    print(f"d1={D1}  d2={D2}  N_train={N_TRAIN}  N_test={N_TEST}  "
          f"trials={N_TRIALS}  hidden={HIDDEN}")
    print(f"policy lambda={LAMBDA}  pi_min={PI_MIN}  signal grid={SIGNAL_GRID}")
    print("=" * 78)

    # accumulators: per signal -> per method -> list over trials
    methods = ['ERM_zero', 'CC_ERM', 'IPW', 'Oracle']
    res = {s: {m: {'dep': [], 'full': []} for m in methods} for s in SIGNAL_GRID}
    acq_rate = {s: [] for s in SIGNAL_GRID}
    # Bayes ceilings (best achievable) for context.
    bayes = {s: {'dep': [], 'full': []} for s in SIGNAL_GRID}

    for signal in SIGNAL_GRID:
        for t in range(N_TRIALS):
            tseed = SEED + 1000 * int(signal * 10) + t
            x1_tr, x2_tr, y_tr = generate(N_TRAIN, signal, seed=tseed)
            rng = np.random.RandomState(tseed + 7)
            acquired, pi = apply_policy(x1_tr, rng)
            acq_rate[signal].append(acquired.mean())

            pi_hat = est_propensity(x1_tr, acquired)

            # ---- Fixed test set (same policy applied for deployment eval) ----
            x1_te, x2_te, y_te = generate(N_TEST, signal, seed=tseed + 50000)
            rng_te = np.random.RandomState(tseed + 99)
            acq_te, _ = apply_policy(x1_te, rng_te)

            X_te_dep  = build_X(x1_te, x2_te, mask=acq_te)   # missing -> zeros
            X_te_full = build_X(x1_te, x2_te, mask=None)     # all observed

            # ---------------- ERM (zero-imputed, all rows) ----------------
            X_erm = build_X(x1_tr, x2_tr, mask=acquired)
            We = fit_mlp(X_erm, y_tr.astype(float), np.ones(N_TRAIN), seed=t)
            res[signal]['ERM_zero']['dep'].append(
                S.mlp_acc(X_te_dep, y_te, *We) * 100)
            res[signal]['ERM_zero']['full'].append(
                S.mlp_acc(X_te_full, y_te, *We) * 100)

            # ---------------- CC-ERM (acquired only, unweighted) ----------
            idx = acquired
            X_cc = build_X(x1_tr[idx], x2_tr[idx], mask=None)
            Wc = fit_mlp(X_cc, y_tr[idx].astype(float),
                         np.ones(idx.sum()), seed=t)
            res[signal]['CC_ERM']['dep'].append(
                S.mlp_acc(X_te_dep, y_te, *Wc) * 100)
            res[signal]['CC_ERM']['full'].append(
                S.mlp_acc(X_te_full, y_te, *Wc) * 100)

            # ---------------- IPW (acquired only, 1/pi_hat) ---------------
            w_ipw = 1.0 / pi_hat[idx]
            Wi = fit_mlp(X_cc, y_tr[idx].astype(float), w_ipw, seed=t)
            res[signal]['IPW']['dep'].append(
                S.mlp_acc(X_te_dep, y_te, *Wi) * 100)
            res[signal]['IPW']['full'].append(
                S.mlp_acc(X_te_full, y_te, *Wi) * 100)

            # ---------------- Oracle (all rows, full x2) ------------------
            x1_o, x2_o, y_o = generate(N_ORACLE, signal, seed=tseed + 250000)
            X_o = build_X(x1_o, x2_o, mask=None)
            Wo = fit_mlp(X_o, y_o.astype(float), np.ones(N_ORACLE), seed=t)
            res[signal]['Oracle']['dep'].append(
                S.mlp_acc(X_te_dep, y_te, *Wo) * 100)
            res[signal]['Oracle']['full'].append(
                S.mlp_acc(X_te_full, y_te, *Wo) * 100)

            # ---------------- Bayes ceilings (no noise tiebreak) ----------
            # full: use true logit sign; dep: zero-impute x2 in true model
            g_te = (g_x2(x2_te) - G_MEAN) / G_STD
            true_full = (x1_te @ W1_TRUE + signal * g_te) > 0
            bayes[signal]['full'].append((true_full == y_te).mean() * 100)
            # deployment bayes: where x2 missing, the truth integrates x2 out;
            # best constant for missing x2 is sign of x1 part only.
            g_dep = g_te.copy()
            g_dep[~acq_te] = 0.0   # E[g] ~ 0 after z-score
            true_dep = (x1_te @ W1_TRUE + signal * g_dep) > 0
            bayes[signal]['dep'].append((true_dep == y_te).mean() * 100)

    # ----------------------- summarise + print -----------------------
    def ms(lst):
        a = np.array(lst)
        return float(a.mean()), float(a.std())

    out = {'settings': {
        'seed': SEED, 'd1': D1, 'd2': D2, 'N_train': N_TRAIN, 'N_test': N_TEST,
        'n_trials': N_TRIALS, 'signal_grid': SIGNAL_GRID, 'lambda': LAMBDA,
        'pi_min': PI_MIN, 'noise_sd': NOISE_SD, 'hidden': HIDDEN,
        'lr': LR, 'n_iter': N_ITER, 'l2': L2,
        'metric': 'accuracy_x100'}, 'results': {}}

    for signal in SIGNAL_GRID:
        am, asd = ms(acq_rate[signal])
        bf_m, bf_s = ms(bayes[signal]['full'])
        bd_m, bd_s = ms(bayes[signal]['dep'])
        print("\n" + "-" * 78)
        print(f"signal = {signal}   acquisition rate = {am:.3f} +/- {asd:.3f}")
        print(f"  Bayes ceiling   full = {bf_m:5.2f}   deployment = {bd_m:5.2f}")
        print(f"  {'method':<10} {'DEPLOY (x2 zero-imp)':>24}   {'FULL (x2 avail)':>22}")
        out['results'][str(signal)] = {
            'acq_rate': [am, asd],
            'bayes_full': [bf_m, bf_s], 'bayes_dep': [bd_m, bd_s], 'methods': {}}
        for m in methods:
            dm, ds = ms(res[signal][m]['dep'])
            fm, fs = ms(res[signal][m]['full'])
            print(f"  {m:<10} {dm:8.2f} +/- {ds:4.2f}        {fm:8.2f} +/- {fs:4.2f}")
            out['results'][str(signal)]['methods'][m] = {
                'dep_mean': dm, 'dep_std': ds, 'full_mean': fm, 'full_std': fs}

    # ----------------------- gap-to-oracle analysis -----------------------
    print("\n" + "=" * 78)
    print("GAP TO ORACLE  (oracle_acc - method_acc; smaller = closer to oracle)")
    print("Does the correction CLOSE the gap as the x2 signal grows?")
    print("=" * 78)
    out['gap_to_oracle'] = {}
    for evalkind in ['full', 'dep']:
        print(f"\n--- {evalkind.upper()} evaluation: oracle - method (pp) ---")
        print(f"  {'signal':>6} {'ERM_zero':>10} {'CC_ERM':>10} {'IPW':>10}")
        out['gap_to_oracle'][evalkind] = {}
        for signal in SIGNAL_GRID:
            om = np.mean(res[signal]['Oracle'][evalkind])
            row = {}
            line = f"  {signal:>6}"
            for m in ['ERM_zero', 'CC_ERM', 'IPW']:
                gap = om - np.mean(res[signal][m][evalkind])
                row[m] = float(gap)
                line += f" {gap:10.2f}"
            print(line)
            out['gap_to_oracle'][evalkind][str(signal)] = row

    # IPW-vs-ERM improvement as a function of the x2 signal strength.
    print("\n" + "=" * 78)
    print("CORRECTION EFFECT = IPW - ERM_zero  (pp; >0 means correction helps)")
    print("=" * 78)
    out['ipw_minus_erm'] = {}
    for evalkind in ['full', 'dep']:
        print(f"\n--- {evalkind.upper()} eval: (IPW - ERM_zero) and (IPW - CC_ERM) ---")
        out['ipw_minus_erm'][evalkind] = {}
        for signal in SIGNAL_GRID:
            ipw = np.array(res[signal]['IPW'][evalkind])
            erm = np.array(res[signal]['ERM_zero'][evalkind])
            cc  = np.array(res[signal]['CC_ERM'][evalkind])
            d_erm_m, d_erm_s = ms((ipw - erm).tolist())
            d_cc_m,  d_cc_s  = ms((ipw - cc).tolist())
            print(f"  signal={signal:>4}: IPW-ERM = {d_erm_m:6.2f} +/- {d_erm_s:4.2f}"
                  f"   IPW-CC = {d_cc_m:6.2f} +/- {d_cc_s:4.2f}")
            out['ipw_minus_erm'][evalkind][str(signal)] = {
                'IPW_minus_ERM': [d_erm_m, d_erm_s],
                'IPW_minus_CC':  [d_cc_m, d_cc_s]}

    # ----------------------- figure -----------------------
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 2.2,
    })
    xs = np.array(SIGNAL_GRID, dtype=float)

    def arr(signal_key, m, evalkind, stat):
        return np.array([out['results'][str(s)]['methods'][m][f'{evalkind}_{stat}']
                         for s in SIGNAL_GRID])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=False)

    panels = [
        ('full', "(a) Full-feature evaluation\n(expensive modality present at test)"),
        ('dep',  "(b) Deployment evaluation\n(expensive modality never acquired at test)"),
    ]
    for ax, (evalkind, title) in zip(axes, panels):
        orc_m, orc_s = arr(None, 'Oracle', evalkind, 'mean'), arr(None, 'Oracle', evalkind, 'std')
        ipw_m, ipw_s = arr(None, 'IPW', evalkind, 'mean'), arr(None, 'IPW', evalkind, 'std')
        erm_m, erm_s = arr(None, 'ERM_zero', evalkind, 'mean'), arr(None, 'ERM_zero', evalkind, 'std')
        cc_m         = arr(None, 'CC_ERM', evalkind, 'mean')

        ax.fill_between(xs, orc_m - orc_s, orc_m + orc_s, color="black", alpha=0.10)
        ax.plot(xs, orc_m, "k--o", label="Oracle (full data)", markersize=7)
        ax.fill_between(xs, ipw_m - ipw_s, ipw_m + ipw_s, color="tab:blue", alpha=0.15)
        ax.plot(xs, ipw_m, "b-s", label="IPW-ERM (1/$\\hat\\pi$)", markersize=7)
        ax.fill_between(xs, erm_m - erm_s, erm_m + erm_s, color="tab:red", alpha=0.15)
        ax.plot(xs, erm_m, "r-^", label="Zero-imputed ERM", markersize=7)
        # CC-ERM ablation (full-feature panel, where the paper discusses it).
        if evalkind == 'full':
            ax.plot(xs, cc_m, color="tab:green", marker="d", linestyle=":",
                    label="CC-ERM (complete-case)", markersize=7)
        ax.set_xlabel("Expensive-modality ($x_2$) signal strength $s$")
        ax.set_ylabel("Accuracy ($\\times 100$)")
        ax.set_title(title)
        ax.legend(loc="upper right" if evalkind == 'full' else "lower left")

    fig.suptitle("Deep higher-dimensional fusion (MLP, hidden $32$, $d_2{=}12$): "
                 "full-feature vs. deployment performance",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_pdf = FIG_DIR / f"{EXP_ID}.pdf"
    out_png = FIG_DIR / f"{EXP_ID}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {out_pdf}")
    print(f"Saved figure: {out_png}")

    # ----------------------- JSON dump -----------------------
    out_json = RESULTS_DIR / f"{EXP_ID}.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved JSON: {out_json}")


if __name__ == '__main__':
    np.random.seed(SEED)
    main()
