"""
Propensity-model misspecification ablation.

How sensitive is the IPW/DIME correction to MISSPECIFICATION of the propensity
model?

DESIGN
------
We use the *reflexive MNAR* DGP from run_exp5 (simulation_experiments.py).
A cheap noisy "scan"  s2 = x2 + noise  drives a Value-of-Information style
acquisition policy:

      pi_true(x1, s2) = max( sigmoid( -lambda * |x1 + s2| ),  pi_min ).

The true propensity is genuinely NONLINEAR: it needs the |x1 + s2| basis
(an x1<->s2 interaction through the absolute value). This is exactly the
setting where propensity misspecification can bite.

We estimate the propensity two ways:
  (a) CORRECT  : logistic on [x1, s2, |x1 + s2|, 1]   (includes the nonlinear basis)
  (b) MISSPEC. : logistic on [x1, s2, 1]              (omits |x1 + s2|, x1-only-ish linear)

and run five estimators:
  1. ERM                      (zero-imputed, biased baseline)
  2. CC-IPW (correct  pi)
  3. CC-IPW (misspecified pi)
  4. DIME   (correct  pi)
  5. DIME   (misspecified pi)

For each we report:
  - expensive-coefficient (x2) bias relative to the ORACLE coefficient
  - test accuracy
  - test AUC (rank-based helper defined here; there is NO module-level AUC)

HYPOTHESIS UNDER TEST
---------------------
Under misspecification the correction degrades GRACEFULLY (reduces but may
not fully remove the bias) rather than failing catastrophically.  We report
how much is lost.

Run:  conda run -n veri_bilimi python exp_misspecification.py
"""

import os
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import simulation_experiments as S

# ─── Pre-specified settings (FIXED; not tuned to the result) ────────────────
SEED          = 20240613
N_TRAIN       = 3000
N_TEST        = 5000
SELECTIVITY   = 2.0      # same as run_exp5 default
SCAN_NOISE    = 1.5      # same as run_exp5 default
PI_MIN        = 0.10     # positivity floor (matches run_exp5)
N_TRIALS      = 40       # >= 20 for stable mean +/- std
EPS_CLIP      = 0.05     # propensity clip for IPW weights
N_BIG_ORACLE  = 50000

FIG_DIR = HERE / 'figures'
RESULTS_DIR = HERE / 'results'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
EXP_ID = 'misspecification'

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.0,
})


# ─── Rank-based AUC helper (no module-level AUC exists) ──────────────────────
def auc_score(y_true, scores):
    """Rank-based ROC-AUC (Mann-Whitney U). Returns 0.5 if one class absent."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def logit_scores(X, w):
    return S.sigmoid(X @ w)


# ─── Propensity estimation ───────────────────────────────────────────────────
def fit_propensity_correct(x1, s2, acquired):
    """CORRECT specification: includes the nonlinear |x1+s2| basis."""
    X_prop = np.column_stack([x1, s2, np.abs(x1 + s2), np.ones(len(x1))])
    w_prop = S.logistic_fit(X_prop, acquired.astype(int))
    pi_hat = S.sigmoid(X_prop @ w_prop)
    return pi_hat


def fit_propensity_misspec(x1, s2, acquired):
    """MISSPECIFIED: omits the nonlinear |x1+s2| basis (linear-only)."""
    X_prop = np.column_stack([x1, s2, np.ones(len(x1))])
    w_prop = S.logistic_fit(X_prop, acquired.astype(int))
    pi_hat = S.sigmoid(X_prop @ w_prop)
    return pi_hat


# ─── DIME with an explicitly supplied propensity ─────────────────────────────
def dime_train_with_pi(x1, x2, m, y, pi_supplied, lr=0.05, n_iter=2000,
                       eps=EPS_CLIP):
    """DIME (single expensive feature) using a SUPPLIED propensity vector.

    Mirrors S.dime_train's gradient-split structure exactly, but lets us
    inject either the correctly-specified or the misspecified propensity so
    the misspecification ablation is faithful.

    Model:  logit P(Y=1) = w1*x1 + gamma*M + beta*(M*x2) + b
      base params (w1, gamma, b): standard ERM gradient over all N
      uplift param (beta): IPW-weighted gradient over the acquired subset.
    """
    N = len(x1)
    x1 = x1.reshape(-1)
    M = m.astype(float)
    Mx2 = M * x2
    X = np.column_stack([x1, M, Mx2, np.ones(N)])   # [x1 | M | M*x2 | bias]
    d_total = 4
    # layout: 0=w1, 1=gamma, 2=beta, 3=b
    pi_j = np.clip(pi_supplied, eps, 1.0 - eps)

    obs = m.astype(bool)
    ipw_w = np.zeros(N)
    if obs.sum() >= 20:
        ipw_w[obs] = 1.0 / pi_j[obs]
        ipw_w[obs] /= ipw_w[obs].mean()   # Hajek normalise

    w = np.zeros(d_total)
    l2 = 1e-4
    for _ in range(n_iter):
        p = S.sigmoid(X @ w)
        err = p - y
        grad = (X.T @ err) / N + l2 * w
        n_obs = obs.sum()
        if n_obs >= 20:
            col = 2  # beta
            grad[col] = (np.sum(err[obs] * ipw_w[obs] * X[obs, col]) / n_obs
                         + l2 * w[col])
        w -= lr * grad
    return {'w': w}


def dime_predict_single(x1, x2, m, params):
    N = len(x1)
    x1 = x1.reshape(-1)
    M = m.astype(float)
    Mx2 = M * x2
    X = np.column_stack([x1, M, Mx2, np.ones(N)])
    return S.sigmoid(X @ params['w'])


def dime_x2_coef(params):
    """Effective expensive-feature coefficient (beta on M*x2)."""
    return params['w'][2]


# ─── Main experiment ────────────────────────────────────────────────────────
def run():
    print("=" * 72)
    print(" Propensity Misspecification Ablation (reflexive MNAR)")
    print("=" * 72)
    print(f" seed={SEED}  N_train={N_TRAIN}  N_test={N_TEST}  n_trials={N_TRIALS}")
    print(f" selectivity(lambda)={SELECTIVITY}  scan_noise(sigma)={SCAN_NOISE}  pi_min={PI_MIN}")
    print(f" Correct propensity basis : [x1, s2, |x1+s2|, 1]")
    print(f" Misspec propensity basis : [x1, s2, 1]  (omits |x1+s2|)")
    print("-" * 72)

    rng_master = np.random.RandomState(SEED)

    # ----- Fixed test set -----
    x1_te, x2_te, y_te = S.generate_data(N_TEST, seed=777)
    X_full_te = S.build_feature_matrix(x1_te, x2_te)

    # ----- Oracle (population) -----
    x1_big, x2_big, y_big = S.generate_data(N_BIG_ORACLE, seed=0)
    w_oracle = S.logistic_fit(S.build_feature_matrix(x1_big, x2_big), y_big)
    oracle_x2_coef = w_oracle[1]
    acc_oracle = S.logistic_acc(X_full_te, y_te, w_oracle)
    auc_oracle = auc_score(y_te, logit_scores(X_full_te, w_oracle))
    print(f" Oracle: x2_coef={oracle_x2_coef:.4f}  acc={acc_oracle*100:.2f}%  auc={auc_oracle:.4f}")
    print("-" * 72)

    # estimators
    est_names = ['ERM', 'CC-IPW(correct)', 'CC-IPW(misspec)',
                 'DIME(correct)', 'DIME(misspec)']

    # collectors: bias (coef - oracle_coef), accuracy, auc
    coef = {k: [] for k in est_names}
    acc = {k: [] for k in est_names}
    auc = {k: [] for k in est_names}
    # diagnostics: how well does each propensity model fit acquisition
    prop_correct_auc = []
    prop_misspec_auc = []
    acq_rates = []

    # seeds for trials, derived from the master rng (fixed, reproducible)
    trial_data_seeds = rng_master.randint(0, 10_000_000, size=N_TRIALS)
    trial_policy_seeds = rng_master.randint(0, 10_000_000, size=N_TRIALS)

    for t in range(N_TRIALS):
        rng = np.random.RandomState(int(trial_policy_seeds[t]))
        x1_tr, x2_tr, y_tr = S.generate_data(N_TRAIN, seed=int(trial_data_seeds[t]))

        # Noisy cheap scan -> reflexive MNAR policy (true propensity nonlinear)
        s2_tr = x2_tr + rng.randn(N_TRAIN) * SCAN_NOISE
        pi_true = np.maximum(S.sigmoid(-SELECTIVITY * np.abs(x1_tr + s2_tr)), PI_MIN)
        acquired = rng.binomial(1, pi_true).astype(bool)
        acq_rates.append(acquired.mean())

        # ---- Estimate propensities (correct vs misspecified) ----
        pi_correct = fit_propensity_correct(x1_tr, s2_tr, acquired)
        pi_misspec = fit_propensity_misspec(x1_tr, s2_tr, acquired)
        prop_correct_auc.append(auc_score(acquired.astype(int), pi_correct))
        prop_misspec_auc.append(auc_score(acquired.astype(int), pi_misspec))

        # ===== 1. ERM (zero-imputed) =====
        x2_imp = x2_tr.copy()
        x2_imp[~acquired] = 0.0
        X_biased = S.build_feature_matrix(x1_tr, x2_imp)
        w_erm = S.logistic_fit(X_biased, y_tr)
        coef['ERM'].append(w_erm[1])
        acc['ERM'].append(S.logistic_acc(X_full_te, y_te, w_erm))
        auc['ERM'].append(auc_score(y_te, logit_scores(X_full_te, w_erm)))

        # Complete-case design matrix shared by CC-IPW variants
        X_cc = S.build_feature_matrix(x1_tr[acquired], x2_tr[acquired])
        y_cc = y_tr[acquired]

        # ===== 2. CC-IPW (correct pi) =====
        pic = np.clip(pi_correct[acquired], EPS_CLIP, 1.0 - EPS_CLIP)
        w_ipw_c = S.logistic_fit(X_cc, y_cc, weights=1.0 / pic)
        coef['CC-IPW(correct)'].append(w_ipw_c[1])
        acc['CC-IPW(correct)'].append(S.logistic_acc(X_full_te, y_te, w_ipw_c))
        auc['CC-IPW(correct)'].append(auc_score(y_te, logit_scores(X_full_te, w_ipw_c)))

        # ===== 3. CC-IPW (misspecified pi) =====
        pim = np.clip(pi_misspec[acquired], EPS_CLIP, 1.0 - EPS_CLIP)
        w_ipw_m = S.logistic_fit(X_cc, y_cc, weights=1.0 / pim)
        coef['CC-IPW(misspec)'].append(w_ipw_m[1])
        acc['CC-IPW(misspec)'].append(S.logistic_acc(X_full_te, y_te, w_ipw_m))
        auc['CC-IPW(misspec)'].append(auc_score(y_te, logit_scores(X_full_te, w_ipw_m)))

        # ===== 4. DIME (correct pi) =====
        d_c = dime_train_with_pi(x1_tr, x2_tr, acquired, y_tr, pi_correct)
        coef['DIME(correct)'].append(dime_x2_coef(d_c))
        m_te = np.ones(N_TEST, dtype=bool)   # full features at test
        prob_dc = dime_predict_single(x1_te, x2_te, m_te, d_c)
        acc['DIME(correct)'].append(((prob_dc > 0.5) == y_te).mean())
        auc['DIME(correct)'].append(auc_score(y_te, prob_dc))

        # ===== 5. DIME (misspecified pi) =====
        d_m = dime_train_with_pi(x1_tr, x2_tr, acquired, y_tr, pi_misspec)
        coef['DIME(misspec)'].append(dime_x2_coef(d_m))
        prob_dm = dime_predict_single(x1_te, x2_te, m_te, d_m)
        acc['DIME(misspec)'].append(((prob_dm > 0.5) == y_te).mean())
        auc['DIME(misspec)'].append(auc_score(y_te, prob_dm))

    # ---- Aggregate ----
    def msd(arr):
        a = np.asarray(arr, dtype=float)
        return float(a.mean()), float(a.std())

    print(f" Mean acquisition rate: {np.mean(acq_rates)*100:.1f}%")
    print(f" Propensity-model fit AUC (acquired vs not):")
    pc_m, pc_s = msd(prop_correct_auc)
    pm_m, pm_s = msd(prop_misspec_auc)
    print(f"   correct  : {pc_m:.4f} +/- {pc_s:.4f}")
    print(f"   misspec  : {pm_m:.4f} +/- {pm_s:.4f}")
    print("-" * 72)
    print(f" Oracle x2 coefficient (target) = {oracle_x2_coef:.4f}")
    print(f"{'Estimator':<18} {'x2_coef':>16} {'|bias| vs oracle':>18} {'acc(%)':>14} {'AUC':>14}")

    results = {}
    bias_means, bias_stds = {}, {}
    for k in est_names:
        cm, cs = msd(coef[k])
        bias_arr = np.abs(np.asarray(coef[k]) - oracle_x2_coef)
        bm, bs = msd(bias_arr)
        am, as_ = msd(acc[k])
        um, us = msd(auc[k])
        bias_means[k] = bm
        bias_stds[k] = bs
        results[k] = {
            'x2_coef_mean': cm, 'x2_coef_std': cs,
            'x2_coef_abs_bias_mean': bm, 'x2_coef_abs_bias_std': bs,
            'x2_coef_signed_bias_mean': float(np.mean(np.asarray(coef[k]) - oracle_x2_coef)),
            'acc_mean': am, 'acc_std': as_,
            'auc_mean': um, 'auc_std': us,
        }
        print(f"{k:<18} {cm:>9.3f}+/-{cs:<5.3f} {bm:>11.3f}+/-{bs:<5.3f} "
              f"{am*100:>8.2f}+/-{as_*100:<4.2f} {um:>8.4f}+/-{us:<6.4f}")

    print("-" * 72)
    # Honest interpretation: how much bias is removed vs ERM, and how much
    # CC-IPW/DIME LOSE under misspecification.
    erm_bias = bias_means['ERM']
    ipw_c_bias = bias_means['CC-IPW(correct)']
    ipw_m_bias = bias_means['CC-IPW(misspec)']
    dime_c_bias = bias_means['DIME(correct)']
    dime_m_bias = bias_means['DIME(misspec)']

    def pct_removed(corrected):
        if erm_bias <= 1e-9:
            return float('nan')
        return 100.0 * (erm_bias - corrected) / erm_bias

    print(" Bias removed relative to ERM (higher = better):")
    print(f"   CC-IPW(correct): {pct_removed(ipw_c_bias):.1f}%  "
          f"CC-IPW(misspec): {pct_removed(ipw_m_bias):.1f}%")
    print(f"   DIME(correct)  : {pct_removed(dime_c_bias):.1f}%  "
          f"DIME(misspec)  : {pct_removed(dime_m_bias):.1f}%")
    print(" Bias LOST to misspecification (correct -> misspec):")
    ipw_lost = ipw_m_bias - ipw_c_bias
    dime_lost = dime_m_bias - dime_c_bias
    print(f"   CC-IPW: +{ipw_lost:.4f} abs-bias  ({(ipw_m_bias/max(ipw_c_bias,1e-9)):.2f}x)")
    print(f"   DIME  : +{dime_lost:.4f} abs-bias  ({(dime_m_bias/max(dime_c_bias,1e-9)):.2f}x)")

    # Graceful-degradation test: misspecified correction still beats ERM,
    # and does NOT blow past ERM bias (no catastrophic failure).
    ipw_graceful = (ipw_m_bias < erm_bias)
    dime_graceful = (dime_m_bias < erm_bias)
    # "catastrophic" = misspecified bias worse than the uncorrected ERM bias
    ipw_catastrophic = (ipw_m_bias > erm_bias * 1.05)
    dime_catastrophic = (dime_m_bias > erm_bias * 1.05)
    print("-" * 72)
    print(" Graceful degradation (misspec still reduces bias vs ERM)?")
    print(f"   CC-IPW: {'YES' if ipw_graceful else 'NO'}   "
          f"catastrophic(worse than ERM)?: {'YES' if ipw_catastrophic else 'no'}")
    print(f"   DIME  : {'YES' if dime_graceful else 'NO'}   "
          f"catastrophic(worse than ERM)?: {'YES' if dime_catastrophic else 'no'}")

    claim_supported = bool(
        (ipw_m_bias < erm_bias) and (dime_m_bias < erm_bias)
        and (not ipw_catastrophic) and (not dime_catastrophic)
    )
    print("-" * 72)
    print(f" CLAIM (graceful degradation, not catastrophic failure) SUPPORTED: "
          f"{claim_supported}")
    print("=" * 72)

    # ---- Figure: bias (left) and AUC (right) across the five estimators ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ['#888888', '#1f77b4', '#7fbfff', '#2ca02c', '#98df8a']
    xpos = np.arange(len(est_names))

    bias_m = [bias_means[k] for k in est_names]
    bias_s = [bias_stds[k] for k in est_names]
    bars = axL.bar(xpos, bias_m, yerr=bias_s, capsize=5, color=colors,
                   alpha=0.9, edgecolor='black', linewidth=0.6)
    axL.axhline(bias_means['ERM'], color='red', ls='--', lw=1.5, alpha=0.7,
                label=f'ERM bias ({bias_means["ERM"]:.2f})')
    axL.set_ylabel('|x₂-coefficient bias| vs oracle')
    axL.set_title('(a) Expensive-coefficient bias\nunder propensity misspecification')
    axL.set_xticks(xpos)
    axL.set_xticklabels(est_names, rotation=20, ha='right')
    axL.legend(fontsize=9)
    for b, m, sdv in zip(bars, bias_m, bias_s):
        axL.text(b.get_x() + b.get_width() / 2, m + sdv + 0.01,
                 f'{m:.2f}', ha='center', va='bottom', fontsize=9)

    auc_m = [results[k]['auc_mean'] for k in est_names]
    auc_s = [results[k]['auc_std'] for k in est_names]
    bars2 = axR.bar(xpos, auc_m, yerr=auc_s, capsize=5, color=colors,
                    alpha=0.9, edgecolor='black', linewidth=0.6)
    axR.axhline(auc_oracle, color='black', ls='--', lw=1.5, alpha=0.7,
                label=f'Oracle AUC ({auc_oracle:.3f})')
    axR.set_ylabel('Test AUC')
    axR.set_title('(b) Test AUC\nunder propensity misspecification')
    axR.set_xticks(xpos)
    axR.set_xticklabels(est_names, rotation=20, ha='right')
    lo = min(min(auc_m) - max(auc_s) - 0.01, auc_oracle - 0.01)
    axR.set_ylim([max(0.5, lo), 1.0015])
    axR.legend(fontsize=9, loc='lower right')
    for b, m, sdv in zip(bars2, auc_m, auc_s):
        # place label just below the cap when very close to the 1.0 ceiling
        if m + sdv > 0.997:
            axR.text(b.get_x() + b.get_width() / 2, m - sdv - 0.001,
                     f'{m:.3f}', ha='center', va='top', fontsize=9)
        else:
            axR.text(b.get_x() + b.get_width() / 2, m + sdv + 0.002,
                     f'{m:.3f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle(
        f'Propensity-model misspecification ablation '
        f'({N_TRIALS} trials, N={N_TRAIN})',
        fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out_pdf = FIG_DIR / f'{EXP_ID}.pdf'
    out_png = FIG_DIR / f'{EXP_ID}.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f" Saved figure: {out_pdf}")
    print(f"               {out_png}")

    # ---- JSON ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        'id': EXP_ID,
        'description': 'Propensity-model misspecification ablation on reflexive '
                       'MNAR DGP. Correct propensity includes nonlinear |x1+s2| '
                       'basis; misspecified omits it. Five estimators: ERM, '
                       'CC-IPW(correct), CC-IPW(misspec), DIME(correct), DIME(misspec).',
        'settings': {
            'seed': SEED,
            'N_train': N_TRAIN,
            'N_test': N_TEST,
            'selectivity_lambda': SELECTIVITY,
            'scan_noise_sigma': SCAN_NOISE,
            'pi_min': PI_MIN,
            'n_trials': N_TRIALS,
            'eps_clip': EPS_CLIP,
            'N_big_oracle': N_BIG_ORACLE,
            'correct_propensity_basis': '[x1, s2, |x1+s2|, 1]',
            'misspec_propensity_basis': '[x1, s2, 1]',
            'metric_note': 'expensive-coef bias = |estimated x2/uplift coef - oracle x2 coef|',
            'dime_coef_caveat': ('DIME beta is the uplift on M*x2 and is NOT '
                                 'structurally identical to the oracle marginal '
                                 'x2 coefficient (DIME also carries x2 signal '
                                 'through the missingness-indicator gamma). The '
                                 'coefficient-bias column understates DIME; its '
                                 'acc/AUC are the fair model-agnostic metrics.'),
        },
        'oracle': {
            'x2_coef': float(oracle_x2_coef),
            'acc': float(acc_oracle),
            'auc': float(auc_oracle),
        },
        'mean_acquisition_rate': float(np.mean(acq_rates)),
        'propensity_fit_auc': {
            'correct_mean': pc_m, 'correct_std': pc_s,
            'misspec_mean': pm_m, 'misspec_std': pm_s,
        },
        'estimators': results,
        'bias_removed_vs_erm_pct': {
            'CC-IPW(correct)': pct_removed(ipw_c_bias),
            'CC-IPW(misspec)': pct_removed(ipw_m_bias),
            'DIME(correct)': pct_removed(dime_c_bias),
            'DIME(misspec)': pct_removed(dime_m_bias),
        },
        'bias_lost_to_misspecification': {
            'CC-IPW_abs': float(ipw_lost),
            'CC-IPW_ratio': float(ipw_m_bias / max(ipw_c_bias, 1e-9)),
            'DIME_abs': float(dime_lost),
            'DIME_ratio': float(dime_m_bias / max(dime_c_bias, 1e-9)),
        },
        'graceful_degradation': {
            'CC-IPW_still_beats_ERM': bool(ipw_graceful),
            'CC-IPW_catastrophic': bool(ipw_catastrophic),
            'DIME_still_beats_ERM': bool(dime_graceful),
            'DIME_catastrophic': bool(dime_catastrophic),
        },
        'claim_supported': claim_supported,
        'figure_pdf': str(out_pdf),
        'figure_png': str(out_png),
    }
    out_json = RESULTS_DIR / f'{EXP_ID}.json'
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f" Saved JSON  : {out_json}")
    return payload


if __name__ == '__main__':
    run()
