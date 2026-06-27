"""
Deep, higher-dimensional fusion where the correction matters
============================================================

Demonstrates the selective-acquisition correction in a deep fusion model with
higher-dimensional modalities, and shows that the IPW correction makes a
material difference.

DESIGN
------
Higher-dimensional, deep-fusion selective-acquisition setting:

  * Cheap modality   x1 in R^{d1}  (d1 = 5)   -- ALWAYS observed.
  * Expensive modality x2 in R^{d2} (d2 = 12) -- acquired only when the
    policy fires (gated on x1).  x2 carries SUBSTANTIAL, genuinely
    NONLINEAR signal about y (interaction / squared terms), so a deep
    fusion model is needed and the correction has room to matter.

  * Acquisition policy gates on x1 ONLY (so propensity pi(x1) is
    identifiable from the cheap modality): acquire x2 when the cheap
    modality is "uncertain" (|cheap score| small).  This is MNAR w.r.t.
    the *full* feature set because x2 carries signal that is correlated
    with whether x2 was acquired.

  * Deployment-time evaluation: on TEST inputs we zero-impute x2 wherever
    the deployed policy would NOT have acquired it (a realistic deployed
    selective-acquisition pipeline), and score with a rank-based AUC.

THREE training conditions, all using the SAME deep fusion model
(S.mlp_fit, a 2-layer ReLU MLP that accepts per-sample weights):

  1. ERM        : zero-imputed full training set, unweighted.
  2. IPW-MLP    : complete cases (x2 acquired), weighted by 1/pi-hat
                  with pi-hat estimated from the cheap modality x1.
  3. Oracle-MLP : full x2 always observed (upper bound).

SWEEP: x2 signal strength s (multiplier on the x2 contribution to the
logit).  HYPOTHESIS under test: as x2 carries more signal, ERM degrades and
the IPW-weighted deep model recovers toward the Oracle, i.e. the
correction demonstrably matters in a deep, higher-dimensional model.

Everything is averaged over n_trials>=20 with a FIXED master seed and
PRE-SPECIFIED settings.  All numbers are printed to stdout, a matplotlib
figure is saved (PDF+PNG), and the numeric results are dumped to JSON.

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

# ── Reuse the existing simulator ────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import simulation_experiments as S


# ── Output locations ────────────────────────────────────────────────────────
EXP_ID = "deep_fusion"
FIG_DIR = HERE / "figures"
RESULTS_DIR = HERE / "results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Pre-specified settings (NOT tuned to manufacture a result) ──────────────
SEED = 20240613          # fixed master seed
N_TRIALS = 25            # >= 20 trials for stable mean +/- std
N_TRAIN = 4000
N_TEST = 4000
D1 = 5                   # cheap modality dimension  (R^d1, always observed)
D2 = 12                  # EXPENSIVE modality dimension (R^d2, gated)
SELECTIVITY = 2.0        # policy selectivity lambda (gates on x1)
PI_MIN = 0.10            # positivity floor for propensity (IPW consistency)
SIGNAL_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]   # x2 signal-strength multiplier s

# Deep fusion model hyper-parameters (shared by ALL three conditions)
MLP_HIDDEN = 32
MLP_ITERS = 4000
MLP_LR = 0.05
MLP_L2 = 1e-4


# ── Rank-based AUC helper (no module-level AUC exists) ───────────────────────
def auc_rank(scores, labels):
    """Rank-based ROC-AUC (Mann-Whitney U / Wilcoxon).

    AUC = P(score(pos) > score(neg)).  Ties contribute 0.5.
    Computed from average ranks of the positive class.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average-rank correction for ties
    s_sorted = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            avg = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
            ranks[order[i:j]] = avg
        i = j
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def mlp_proba(X, W1, b1, W2, b2):
    """Forward pass of S.mlp_fit's 2-layer ReLU MLP -> P(y=1)."""
    h = S._relu(X @ W1 + b1)
    return S.sigmoid(h @ W2 + b2)[:, 0]


# ── Higher-dimensional, NONLINEAR deep-fusion DGP ───────────────────────────
def make_dgp(rng):
    """Draw a fixed set of DGP parameters for one trial.

    Cheap modality x1 in R^{D1}, expensive modality x2 in R^{D2}.
    The label depends on x1 (linear) PLUS a genuinely NONLINEAR function of
    x2 (pairwise interactions + squared terms), so a deep model is required
    and a strong x2 contribution gives the correction room to matter.
    """
    a = rng.randn(D1)                          # cheap linear weights
    b_lin = rng.randn(D2) / np.sqrt(D2)        # x2 linear part
    # nonlinear x2 terms: random pairwise interactions + squared terms
    n_pairs = D2
    pair_i = rng.randint(0, D2, size=n_pairs)
    pair_j = rng.randint(0, D2, size=n_pairs)
    pair_w = rng.randn(n_pairs)
    sq_w = rng.randn(D2)
    return dict(a=a, b_lin=b_lin, pair_i=pair_i, pair_j=pair_j,
                pair_w=pair_w, sq_w=sq_w)


def gen_xy(N, dgp, signal, rng):
    """Generate (x1, x2, y) for a higher-dim nonlinear fusion problem.

    `signal` scales the x2 contribution to the logit (the swept quantity).
    """
    x1 = rng.randn(N, D1)
    x2 = rng.randn(N, D2)

    cheap_logit = x1 @ dgp["a"]                       # always-available signal

    # genuinely nonlinear x2 contribution
    x2_lin = x2 @ dgp["b_lin"]
    inter = (dgp["pair_w"] * x2[:, dgp["pair_i"]] * x2[:, dgp["pair_j"]]).sum(axis=1)
    inter = inter / np.sqrt(D2)
    sq = (dgp["sq_w"] * (x2 ** 2 - 1.0)).sum(axis=1) / np.sqrt(D2)
    x2_contrib = x2_lin + inter + sq

    logit = cheap_logit + signal * x2_contrib
    p = S.sigmoid(logit)
    y = (rng.rand(N) < p).astype(float)
    return x1, x2, y


def cheap_score(x1, dgp):
    """Score from the cheap modality only -- drives the acquisition policy."""
    return x1 @ dgp["a"]


def acq_prob_hidim(x1, dgp, selectivity=SELECTIVITY, pi_min=PI_MIN):
    """Acquisition propensity pi(x1) = max(sigmoid(-lambda*|cheap score|), pi_min).

    Acquire the expensive modality x2 when the CHEAP modality is uncertain
    about y (|cheap score| small).  Depends only on x1 -> identifiable from
    the cheap modality, but MNAR w.r.t. the full feature set because x2
    carries signal.
    """
    cs = cheap_score(x1, dgp)
    raw = S.sigmoid(-selectivity * np.abs(cs))
    return np.maximum(raw, pi_min)


def build_X(x1, x2):
    """Deep-fusion input: concatenate both modalities + bias column."""
    return np.column_stack([x1, x2, np.ones(len(x1))])


# ── One trial for a given x2 signal strength ────────────────────────────────
def run_trial(signal, trial, master_seed=SEED):
    seed = master_seed + 1000 * trial
    rng = np.random.RandomState(seed)
    dgp = make_dgp(rng)

    # training data
    x1_tr, x2_tr, y_tr = gen_xy(N_TRAIN, dgp, signal, rng)
    pi_tr = acq_prob_hidim(x1_tr, dgp)
    acq_tr = rng.binomial(1, pi_tr).astype(bool)

    # test data + deployment acquisition
    x1_te, x2_te, y_te = gen_xy(N_TEST, dgp, signal, rng)
    pi_te = acq_prob_hidim(x1_te, dgp)
    acq_te = rng.binomial(1, pi_te).astype(bool)

    x2_te_deploy = x2_te.copy()
    x2_te_deploy[~acq_te] = 0.0
    X_te_deploy = build_X(x1_te, x2_te_deploy)
    X_te_full = build_X(x1_te, x2_te)

    # propensity estimated from cheap modality
    cs_tr = cheap_score(x1_tr, dgp)
    X_prop = np.column_stack([x1_tr, np.abs(cs_tr), np.ones(N_TRAIN)])
    w_prop = S.logistic_fit(X_prop, acq_tr.astype(float), lr=0.1, n_iter=1500)
    pi_hat = np.clip(S.sigmoid(X_prop @ w_prop), PI_MIN * 0.5, 0.99)

    # Condition 1: ERM (zero-imputed)
    x2_tr_zi = x2_tr.copy()
    x2_tr_zi[~acq_tr] = 0.0
    X_tr_erm = build_X(x1_tr, x2_tr_zi)
    We = S.mlp_fit(X_tr_erm, y_tr, weights=None, hidden=MLP_HIDDEN,
                   lr=MLP_LR, n_iter=MLP_ITERS, l2=MLP_L2, seed=seed)
    # PRIMARY (task-mandated): deployment with off-policy zero-imputed test x2
    auc_erm = auc_rank(mlp_proba(X_te_deploy, *We), y_te)
    # DIAGNOSTIC: same model evaluated on full test x2 (intrinsic fusion quality)
    auc_erm_full = auc_rank(mlp_proba(X_te_full, *We), y_te)

    # Condition 2: IPW-weighted MLP (complete cases, 1/pi-hat)
    if acq_tr.sum() >= 50:
        X_cc = build_X(x1_tr[acq_tr], x2_tr[acq_tr])
        y_cc = y_tr[acq_tr]
        w_ipw = 1.0 / pi_hat[acq_tr]
        Wi = S.mlp_fit(X_cc, y_cc, weights=w_ipw, hidden=MLP_HIDDEN,
                       lr=MLP_LR, n_iter=MLP_ITERS, l2=MLP_L2, seed=seed)
        auc_ipw = auc_rank(mlp_proba(X_te_deploy, *Wi), y_te)
        auc_ipw_full = auc_rank(mlp_proba(X_te_full, *Wi), y_te)
    else:
        auc_ipw = float("nan")
        auc_ipw_full = float("nan")

    # Condition 3: Oracle MLP (full data, evaluated on full test x2)
    X_tr_full = build_X(x1_tr, x2_tr)
    Wo = S.mlp_fit(X_tr_full, y_tr, weights=None, hidden=MLP_HIDDEN,
                   lr=MLP_LR, n_iter=MLP_ITERS, l2=MLP_L2, seed=seed)
    auc_orc = auc_rank(mlp_proba(X_te_full, *Wo), y_te)

    return dict(auc_erm=auc_erm, auc_ipw=auc_ipw, auc_orc=auc_orc,
                auc_erm_full=auc_erm_full, auc_ipw_full=auc_ipw_full,
                acq_rate=float(acq_tr.mean()))


# ── Main sweep ──────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(" Deep, higher-dim fusion where the correction matters")
    print("=" * 70)
    print(f"Seed={SEED}  trials={N_TRIALS}  N_train={N_TRAIN}  N_test={N_TEST}")
    print(f"d1(cheap)={D1}  d2(expensive)={D2}  selectivity={SELECTIVITY}  pi_min={PI_MIN}")
    print(f"Deep fusion model: 2-layer ReLU MLP  hidden={MLP_HIDDEN} "
          f"iters={MLP_ITERS} lr={MLP_LR} l2={MLP_L2}")
    print(f"x2 signal grid: {SIGNAL_GRID}")
    print(f"Metric: rank-based AUC at DEPLOYMENT (test x2 zero-imputed where "
          f"policy would not acquire)\n")

    results = {}
    for s in SIGNAL_GRID:
        erm, ipw, orc, acq = [], [], [], []
        erm_f, ipw_f = [], []
        for t in range(N_TRIALS):
            r = run_trial(s, t)
            erm.append(r["auc_erm"])
            ipw.append(r["auc_ipw"])
            orc.append(r["auc_orc"])
            erm_f.append(r["auc_erm_full"])
            ipw_f.append(r["auc_ipw_full"])
            acq.append(r["acq_rate"])
        erm, ipw, orc, acq = map(np.array, (erm, ipw, orc, acq))
        erm_f, ipw_f = map(np.array, (erm_f, ipw_f))
        results[s] = dict(
            erm_mean=float(np.nanmean(erm)), erm_std=float(np.nanstd(erm)),
            ipw_mean=float(np.nanmean(ipw)), ipw_std=float(np.nanstd(ipw)),
            orc_mean=float(np.nanmean(orc)), orc_std=float(np.nanstd(orc)),
            erm_full_mean=float(np.nanmean(erm_f)), erm_full_std=float(np.nanstd(erm_f)),
            ipw_full_mean=float(np.nanmean(ipw_f)), ipw_full_std=float(np.nanstd(ipw_f)),
            acq_rate=float(np.mean(acq)),
            n_valid_ipw=int(np.sum(~np.isnan(ipw))),
        )
        rr = results[s]
        print(f"signal s={s:>4}: acq={rr['acq_rate']*100:5.1f}%  "
              f"ERM={rr['erm_mean']:.4f}+/-{rr['erm_std']:.4f}  "
              f"IPW={rr['ipw_mean']:.4f}+/-{rr['ipw_std']:.4f}  "
              f"Oracle={rr['orc_mean']:.4f}+/-{rr['orc_std']:.4f}  "
              f"| recovery={(rr['ipw_mean']-rr['erm_mean']):.4f}  "
              f"gap_to_oracle: ERM={(rr['orc_mean']-rr['erm_mean']):.4f} "
              f"IPW={(rr['orc_mean']-rr['ipw_mean']):.4f}")
        print(f"            [DIAGNOSTIC full-x2 deploy]  "
              f"ERM_full={rr['erm_full_mean']:.4f}+/-{rr['erm_full_std']:.4f}  "
              f"IPW_full={rr['ipw_full_mean']:.4f}+/-{rr['ipw_full_std']:.4f}  "
              f"Oracle={rr['orc_mean']:.4f}  "
              f"| recovery_full={(rr['ipw_full_mean']-rr['erm_full_mean']):.4f}")

    # ── Claim assessment ────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("CLAIM CHECK: as x2 signal grows, ERM degrades and IPW recovers "
          "toward Oracle.")
    # gap_to_oracle for ERM should grow with signal; IPW recovery (IPW-ERM)
    # should be positive and grow.
    erm_gaps = [results[s]["orc_mean"] - results[s]["erm_mean"] for s in SIGNAL_GRID]
    ipw_gaps = [results[s]["orc_mean"] - results[s]["ipw_mean"] for s in SIGNAL_GRID]
    recoveries = [results[s]["ipw_mean"] - results[s]["erm_mean"] for s in SIGNAL_GRID]
    print(f"  ERM gap-to-oracle by signal: "
          f"{[f'{g:+.4f}' for g in erm_gaps]}")
    print(f"  IPW gap-to-oracle by signal: "
          f"{[f'{g:+.4f}' for g in ipw_gaps]}")
    print(f"  IPW recovery (IPW-ERM)     : "
          f"{[f'{r:+.4f}' for r in recoveries]}")

    s_hi = SIGNAL_GRID[-1]
    erm_gap_hi = results[s_hi]["orc_mean"] - results[s_hi]["erm_mean"]
    recovery_hi = results[s_hi]["ipw_mean"] - results[s_hi]["erm_mean"]
    erm_degrades = erm_gaps[-1] > erm_gaps[0] + 0.005
    ipw_recovers = recovery_hi > 0.005 and ipw_gaps[-1] < erm_gaps[-1] - 0.005
    supports = bool(erm_degrades and ipw_recovers)
    print(f"\n  At highest signal s={s_hi}: ERM gap-to-oracle={erm_gap_hi:+.4f}, "
          f"IPW recovery over ERM={recovery_hi:+.4f}")
    print(f"  ERM degrades with signal: {erm_degrades}")
    print(f"  IPW recovers toward oracle: {ipw_recovers}")
    print(f"  ==> PRIMARY (off-policy zero-imputed deploy) supportsClaim = {supports}")

    # ── Diagnostic claim check under MATCHED (full-x2) deployment ────────
    print("\n  --- DIAGNOSTIC: full-x2 deployment (every model sees test x2) ---")
    erm_gaps_f = [results[s]["orc_mean"] - results[s]["erm_full_mean"] for s in SIGNAL_GRID]
    ipw_gaps_f = [results[s]["orc_mean"] - results[s]["ipw_full_mean"] for s in SIGNAL_GRID]
    recoveries_f = [results[s]["ipw_full_mean"] - results[s]["erm_full_mean"] for s in SIGNAL_GRID]
    print(f"  ERM_full gap-to-oracle: {[f'{g:+.4f}' for g in erm_gaps_f]}")
    print(f"  IPW_full gap-to-oracle: {[f'{g:+.4f}' for g in ipw_gaps_f]}")
    print(f"  IPW_full recovery (IPW-ERM): {[f'{r:+.4f}' for r in recoveries_f]}")
    recovery_f_hi = recoveries_f[-1]
    erm_degrades_f = erm_gaps_f[-1] > erm_gaps_f[0] + 0.005
    ipw_recovers_f = recovery_f_hi > 0.005 and ipw_gaps_f[-1] < erm_gaps_f[-1] - 0.005
    supports_f = bool(erm_degrades_f and ipw_recovers_f)
    print(f"  At s={s_hi}: ERM_full gap={erm_gaps_f[-1]:+.4f}  "
          f"IPW_full recovery={recovery_f_hi:+.4f}")
    print(f"  ==> DIAGNOSTIC (full-x2 deploy) supportsClaim = {supports_f}")

    # ── Figure ──────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 2.2,
    })
    xs = np.array(SIGNAL_GRID, dtype=float)
    erm_m = np.array([results[s]["erm_mean"] for s in SIGNAL_GRID])
    erm_s = np.array([results[s]["erm_std"] for s in SIGNAL_GRID])
    ipw_m = np.array([results[s]["ipw_mean"] for s in SIGNAL_GRID])
    ipw_s = np.array([results[s]["ipw_std"] for s in SIGNAL_GRID])
    orc_m = np.array([results[s]["orc_mean"] for s in SIGNAL_GRID])
    orc_s = np.array([results[s]["orc_std"] for s in SIGNAL_GRID])
    erm_fm = np.array([results[s]["erm_full_mean"] for s in SIGNAL_GRID])
    erm_fs = np.array([results[s]["erm_full_std"] for s in SIGNAL_GRID])
    ipw_fm = np.array([results[s]["ipw_full_mean"] for s in SIGNAL_GRID])
    ipw_fs = np.array([results[s]["ipw_full_std"] for s in SIGNAL_GRID])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=True)

    # Panel (a): PRIMARY task-mandated deployment (off-policy zero-imputed x2)
    ax = axes[0]
    ax.fill_between(xs, orc_m - orc_s, orc_m + orc_s, color="black", alpha=0.10)
    ax.plot(xs, orc_m, "k--o", label="Oracle-MLP (full data)", markersize=7)
    ax.fill_between(xs, ipw_m - ipw_s, ipw_m + ipw_s, color="tab:blue", alpha=0.15)
    ax.plot(xs, ipw_m, "b-s", label="IPW-MLP (1/$\\hat\\pi$)", markersize=7)
    ax.fill_between(xs, erm_m - erm_s, erm_m + erm_s, color="tab:red", alpha=0.15)
    ax.plot(xs, erm_m, "r-^", label="ERM-MLP (zero-imputed)", markersize=7)
    ax.set_xlabel("Expensive-modality ($x_2$) signal strength $s$")
    ax.set_ylabel("Deployment-time rank AUC")
    ax.set_title("(a) Off-policy deployment\n(test $x_2$ zero-imputed where policy skips)")
    ax.legend(loc="lower left")

    # Panel (b): DIAGNOSTIC matched deployment (every model sees full test x2)
    ax = axes[1]
    ax.fill_between(xs, orc_m - orc_s, orc_m + orc_s, color="black", alpha=0.10)
    ax.plot(xs, orc_m, "k--o", label="Oracle-MLP (full data)", markersize=7)
    ax.fill_between(xs, ipw_fm - ipw_fs, ipw_fm + ipw_fs, color="tab:blue", alpha=0.15)
    ax.plot(xs, ipw_fm, "b-s", label="IPW-MLP (1/$\\hat\\pi$)", markersize=7)
    ax.fill_between(xs, erm_fm - erm_fs, erm_fm + erm_fs, color="tab:red", alpha=0.15)
    ax.plot(xs, erm_fm, "r-^", label="ERM-MLP (zero-imputed train)", markersize=7)
    ax.set_xlabel("Expensive-modality ($x_2$) signal strength $s$")
    ax.set_title("(b) Matched deployment\n(every model sees full test $x_2$)")
    ax.legend(loc="lower left")

    fig.suptitle("Deep higher-dimensional fusion: full-feature vs. deployment AUC",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_pdf = FIG_DIR / f"{EXP_ID}.pdf"
    out_png = FIG_DIR / f"{EXP_ID}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {out_pdf}")
    print(f"Saved figure: {out_png}")

    # ── JSON dump ───────────────────────────────────────────────────────
    payload = dict(
        id=EXP_ID,
        description="Deep, higher-dimensional fusion showing the IPW "
                    "correction matters. 2-layer ReLU MLP fusion model; "
                    "cheap x1 in R^d1 always observed, expensive x2 in R^d2 "
                    "gated by a cheap-modality policy and carrying nonlinear "
                    "signal; deployment-time rank-AUC with off-policy "
                    "zero-imputation; sweep over x2 signal strength.",
        seed=SEED,
        settings=dict(
            n_trials=N_TRIALS, N_train=N_TRAIN, N_test=N_TEST,
            d1_cheap=D1, d2_expensive=D2, selectivity=SELECTIVITY,
            pi_min=PI_MIN, signal_grid=SIGNAL_GRID,
            mlp_hidden=MLP_HIDDEN, mlp_iters=MLP_ITERS, mlp_lr=MLP_LR,
            mlp_l2=MLP_L2, metric="rank_based_AUC_deployment_zero_imputed",
        ),
        results_by_signal={str(s): results[s] for s in SIGNAL_GRID},
        claim_check=dict(
            erm_gap_to_oracle_by_signal=erm_gaps,
            ipw_gap_to_oracle_by_signal=ipw_gaps,
            ipw_recovery_over_erm_by_signal=recoveries,
            erm_degrades_with_signal=bool(erm_degrades),
            ipw_recovers_toward_oracle=bool(ipw_recovers),
            supports_claim=bool(supports),
        ),
        diagnostic_full_x2_deploy=dict(
            note="Every model evaluated on FULL test x2 (matched deployment); "
                 "isolates the train/deploy distribution-mismatch confound of "
                 "the off-policy zero-imputed primary metric.",
            erm_full_gap_to_oracle_by_signal=erm_gaps_f,
            ipw_full_gap_to_oracle_by_signal=ipw_gaps_f,
            ipw_full_recovery_over_erm_by_signal=recoveries_f,
            erm_full_degrades_with_signal=bool(erm_degrades_f),
            ipw_full_recovers_toward_oracle=bool(ipw_recovers_f),
            supports_claim_full=bool(supports_f),
        ),
    )
    out_json = RESULTS_DIR / f"{EXP_ID}.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON: {out_json}")
    print(f"\nFINAL PRIMARY supportsClaim = {supports}  | "
          f"DIAGNOSTIC(full-x2) supportsClaim = {supports_f}")
    return payload


if __name__ == "__main__":
    main()
