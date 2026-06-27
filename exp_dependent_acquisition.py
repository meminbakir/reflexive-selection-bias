"""
Dependent (correlated) acquisition for DIME.

DIME assumes per-feature acquisition independence and additive log-odds.  This
experiment tests whether DIME still works when the expensive modalities are
acquired with CORRELATED indicators (violating the per-feature
acquisition-independence assumption).

We use an ADDITIVE-LOGIT DGP with KNOWN true coefficients, so coefficient
recovery is well-defined and we can test (a) that DIME recovers beta_true
under independent acquisition, and (b) how far it degrades as acquisition
becomes correlated.

    x1 ~ N(0,1)              cheap, always observed
    x2_j ~ N(0,1), j=1..J    expensive, acquired selectively
    logit P(y=1) = w1*x1 + sum_j beta_true_j * x2_j        (additive in log-odds)
    pi_j(x1)  = max(sigmoid(-lambda*|x1|), pi_min)         MAR given x1
    correlated acquisition via a Gaussian copula with correlation rho
        (marginals pi_j held EXACT for every rho)

Methods (all logistic, same expensive-uplift parameterisation where relevant):
    Oracle : full data        -> beta_oracle ~ beta_true (sanity check)
    ERM    : zero-imputed      -> beta biased toward 0
    DIME   : per-feature IPW   -> the method under test
Metrics, swept over rho: mean_j |beta_j_hat - beta_true_j|  and test rank-AUC.

A second arm holds acquisition independent and grows a cross-feature outcome
interaction kappa to probe the genuine additive-log-odds limit of DIME.

Run:  conda run -n veri_bilimi python exp_dependent_acquisition.py
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simulation_experiments as S
from scipy.stats import norm, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED       = 20260618
J          = 4
W1_TRUE    = 1.0
BETA_TRUE  = np.array([1.4, -1.0, 1.8, -0.6])
LAMBDA     = 1.0
PI_MIN     = 0.20
N_TRAIN    = 4000
N_TEST     = 4000
N_TRIALS   = 15
RHOS       = [0.0, 0.3, 0.6, 0.9]
FIG_DIR    = os.path.join(HERE, "figures")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_JSON   = os.path.join(RESULTS_DIR, "dependent_acquisition.json")


def rank_auc(y, scores):
    y = np.asarray(y, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def fit_logistic(X, y, w=None, n_iter=1500, lr=0.4, l2=1e-4):
    """Plain weighted logistic with explicit trailing intercept. Returns beta (last=intercept)."""
    Xb = np.column_stack([X, np.ones(len(X))])
    beta = np.zeros(Xb.shape[1])
    wv = np.ones(len(y)) if w is None else (w / np.mean(w))
    for _ in range(n_iter):
        p = S.sigmoid(Xb @ beta)
        g = Xb.T @ (wv * (p - y)) / len(y) + l2 * beta
        beta -= lr * g
    return beta


def make_data(rng, N, rho, kappa=0.0):
    x1 = rng.standard_normal(N)
    x2 = rng.standard_normal((N, J))
    logit = W1_TRUE * x1 + x2 @ BETA_TRUE
    if kappa != 0.0:
        # additivity violation: a cross-feature outcome interaction (the genuine
        # limit of DIME's additive-log-odds assumption, independent of acquisition).
        logit = logit + kappa * x2[:, 0] * x2[:, 1]
    y = (rng.uniform(size=N) < S.sigmoid(logit)).astype(float)
    pi = np.maximum(S.sigmoid(-LAMBDA * np.abs(x1)), PI_MIN)        # (N,) marginal per feature
    # Gaussian copula: shared latent z, per-feature noise; threshold so P(M_j=1)=pi exactly
    z = rng.standard_normal(N)
    thr = norm.ppf(1.0 - pi)                                        # (N,)
    M = np.zeros((N, J))
    for j in range(J):
        e = rng.standard_normal(N)
        u = np.sqrt(rho) * z + np.sqrt(1.0 - rho) * e               # ~N(0,1), corr=rho across j
        M[:, j] = (u > thr).astype(float)
    return x1, x2, y, M, pi


def empirical_corr(M):
    if M.shape[1] < 2:
        return 0.0
    c = np.corrcoef(M, rowvar=False)
    iu = np.triu_indices(M.shape[1], k=1)
    return float(np.nanmean(c[iu]))


def beta_from_dime(x1, x2, M, y):
    d1 = 1
    x2_cols = [x2[:, j] for j in range(J)]
    m_cols = [M[:, j].astype(bool) for j in range(J)]
    dp = S.dime_train(x1.reshape(-1, 1), x2_cols, m_cols, y, lr=0.15, n_iter=2500)
    w = dp["w"]                       # layout [w1(d1) | gamma(J) | beta(J) | b]
    beta = w[d1 + J: d1 + 2 * J]
    return dp, beta


def deploy_features_oracle(x1, x2):
    return np.column_stack([x1, x2])               # full data


def deploy_features_zeroimp(x1, x2, M):
    return np.column_stack([x1, x2 * M])           # zero-imputed expensive block


def main():
    rng = np.random.default_rng(SEED)
    results = {}
    emp_corr = {}
    for rho in RHOS:
        bias_oracle, bias_erm, bias_dime = [], [], []
        auc_oracle, auc_erm, auc_dime = [], [], []
        corrs = []
        for t in range(N_TRIALS):
            x1, x2, y, M, pi = make_data(rng, N_TRAIN, rho)
            x1te, x2te, yte, Mte, _ = make_data(rng, N_TEST, rho)
            corrs.append(empirical_corr(M))

            # Oracle (full data)
            w_or = fit_logistic(deploy_features_oracle(x1, x2), y)
            beta_or = w_or[1:1 + J]
            bias_oracle.append(np.mean(np.abs(beta_or - BETA_TRUE)))
            s_or = np.column_stack([x1te, x2te, np.ones(N_TEST)]) @ w_or
            auc_oracle.append(rank_auc(yte, s_or))

            # ERM (zero-imputed)
            w_erm = fit_logistic(deploy_features_zeroimp(x1, x2, M), y)
            beta_erm = w_erm[1:1 + J]
            bias_erm.append(np.mean(np.abs(beta_erm - BETA_TRUE)))
            # deploy with full features available at test (measure the learned relationship)
            s_erm = np.column_stack([x1te, x2te, np.ones(N_TEST)]) @ w_erm
            auc_erm.append(rank_auc(yte, s_erm))

            # DIME
            dp, beta_dime = beta_from_dime(x1, x2, M, y)
            bias_dime.append(np.mean(np.abs(beta_dime - BETA_TRUE)))
            # DIME deploy score with full expensive features (M=1 for all at eval)
            x2_cols_te = [x2te[:, j] for j in range(J)]
            m_cols_te = [np.ones(N_TEST, bool) for _ in range(J)]
            s_dime = S.dime_predict(x1te.reshape(-1, 1), x2_cols_te, m_cols_te, dp)
            auc_dime.append(rank_auc(yte, s_dime))

        def ms(a):
            a = np.asarray(a, float)
            return float(np.mean(a)), float(np.std(a))
        rec = {}
        rec["emp_acq_corr_mean"], rec["emp_acq_corr_std"] = ms(corrs)
        rec["bias_oracle_mean"], rec["bias_oracle_std"] = ms(bias_oracle)
        rec["bias_erm_mean"], rec["bias_erm_std"] = ms(bias_erm)
        rec["bias_dime_mean"], rec["bias_dime_std"] = ms(bias_dime)
        rec["auc_oracle_mean"], rec["auc_oracle_std"] = ms(auc_oracle)
        rec["auc_erm_mean"], rec["auc_erm_std"] = ms(auc_erm)
        rec["auc_dime_mean"], rec["auc_dime_std"] = ms(auc_dime)
        results[f"{rho}"] = rec
        print(f"rho={rho} (emp corr {rec['emp_acq_corr_mean']:.2f}): "
              f"|beta-beta*|  ERM {rec['bias_erm_mean']:.3f}  DIME {rec['bias_dime_mean']:.3f}  "
              f"Oracle {rec['bias_oracle_mean']:.3f}  | AUC ERM {rec['auc_erm_mean']:.3f} "
              f"DIME {rec['auc_dime_mean']:.3f} Oracle {rec['auc_oracle_mean']:.3f}")

    # ---- additivity-violation arm: independent acquisition, growing outcome interaction ----
    # The genuine limit of DIME is a cross-feature outcome interaction (not correlated
    # acquisition). We compare DIME / ERM to an INTERACTION-AWARE oracle (the correctly
    # specified full-data model) so the gap isolates the additive-log-odds limitation.
    KAPPAS = [0.0, 0.25, 0.5, 1.0]
    kappa_results = {}
    for kappa in KAPPAS:
        gap_dime, gap_erm = [], []
        for t in range(N_TRIALS):
            x1, x2, y, M, pi = make_data(rng, N_TRAIN, 0.0, kappa=kappa)
            x1te, x2te, yte, Mte, _ = make_data(rng, N_TEST, 0.0, kappa=kappa)
            int_tr = (x2[:, 0] * x2[:, 1]).reshape(-1, 1)
            int_te = (x2te[:, 0] * x2te[:, 1]).reshape(-1, 1)
            w_oi = fit_logistic(np.column_stack([x1, x2, int_tr]), y)          # correct model
            auc_oi = rank_auc(yte, np.column_stack([x1te, x2te, int_te, np.ones(N_TEST)]) @ w_oi)
            w_er = fit_logistic(deploy_features_zeroimp(x1, x2, M), y)          # additive ERM
            auc_er = rank_auc(yte, np.column_stack([x1te, x2te, np.ones(N_TEST)]) @ w_er)
            dp, _ = beta_from_dime(x1, x2, M, y)                                # additive DIME
            s_di = S.dime_predict(x1te.reshape(-1, 1), [x2te[:, j] for j in range(J)],
                                  [np.ones(N_TEST, bool) for _ in range(J)], dp)
            auc_di = rank_auc(yte, s_di)
            gap_dime.append(auc_oi - auc_di)
            gap_erm.append(auc_oi - auc_er)
        kappa_results[f"{kappa}"] = {
            "auc_gap_dime_to_oracle_mean": float(np.mean(gap_dime)),
            "auc_gap_dime_to_oracle_std": float(np.std(gap_dime)),
            "auc_gap_erm_to_oracle_mean": float(np.mean(gap_erm)),
        }
        print(f"kappa={kappa}: AUC gap to interaction-aware oracle  "
              f"DIME {np.mean(gap_dime):.3f}  ERM {np.mean(gap_erm):.3f}")

    # verdict
    b_indep = results[f"{RHOS[0]}"]
    dime_recovers_indep = b_indep["bias_dime_mean"] < 0.5 * b_indep["bias_erm_mean"]
    dime_better_than_erm = all(results[f"{r}"]["bias_dime_mean"] <
                               results[f"{r}"]["bias_erm_mean"] for r in RHOS)
    degr = (results[f"{RHOS[-1]}"]["bias_dime_mean"] - results[f"{RHOS[0]}"]["bias_dime_mean"])
    out = {
        "description": "Dependent/correlated acquisition for DIME; additive-logit DGP, known beta_true",
        "settings": {"J": J, "beta_true": BETA_TRUE.tolist(), "w1_true": W1_TRUE,
                     "lambda": LAMBDA, "pi_min": PI_MIN, "n_train": N_TRAIN,
                     "n_test": N_TEST, "n_trials": N_TRIALS, "rhos": RHOS, "seed": SEED},
        "results_by_rho": results,
        "results_by_kappa": kappa_results,
        "verdict": {
            "dime_recovers_beta_under_independent_acq": bool(dime_recovers_indep),
            "dime_beats_erm_on_coef_at_all_rho": bool(dime_better_than_erm),
            "dime_bias_increase_indep_to_max_rho": float(degr),
        },
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # figure
    rs = [float(r) for r in RHOS]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for key, lab, c in [("bias_erm", "ERM (zero-imputed)", "tab:red"),
                        ("bias_dime", "DIME", "tab:blue"),
                        ("bias_oracle", "Oracle", "tab:green")]:
        m = [results[f"{r}"][f"{key}_mean"] for r in RHOS]
        s = [results[f"{r}"][f"{key}_std"] for r in RHOS]
        ax[0].errorbar(rs, m, yerr=s, marker="o", label=lab, color=c, capsize=2)
    ax[0].set_xlabel("acquisition correlation $\\rho$")
    ax[0].set_ylabel(r"mean $|\hat\beta_j-\beta_j^\star|$")
    ax[0].set_title("Coefficient recovery"); ax[0].legend(fontsize=8)
    for key, lab, c in [("auc_erm", "ERM", "tab:red"), ("auc_dime", "DIME", "tab:blue"),
                        ("auc_oracle", "Oracle", "tab:green")]:
        m = [results[f"{r}"][f"{key}_mean"] for r in RHOS]
        ax[1].plot(rs, m, marker="o", label=lab, color=c)
    ax[1].set_xlabel("acquisition correlation $\\rho$")
    ax[1].set_ylabel("test AUC"); ax[1].set_title("Predictive AUC"); ax[1].legend(fontsize=8)
    ks = [float(k) for k in KAPPAS]
    ax[2].plot(ks, [kappa_results[f"{k}"]["auc_gap_erm_to_oracle_mean"] for k in KAPPAS],
               marker="s", label="ERM", color="tab:red")
    ax[2].plot(ks, [kappa_results[f"{k}"]["auc_gap_dime_to_oracle_mean"] for k in KAPPAS],
               marker="o", label="DIME", color="tab:blue")
    ax[2].set_xlabel(r"outcome interaction $\kappa$")
    ax[2].set_ylabel("AUC gap to correct oracle")
    ax[2].set_title("Additivity violation (the real limit)"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "dependent_acquisition.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "dependent_acquisition.png"), dpi=200, bbox_inches="tight")
    print("\nverdict:", json.dumps(out["verdict"], indent=2))
    print("saved:", OUT_JSON)


if __name__ == "__main__":
    main()
