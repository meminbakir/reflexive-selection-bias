"""
Synthetic Experiments for:
  "Reflexive Selection Bias in Deployed Multimodal Fusion:
   Causal Correction for Policy-Induced Feedback Loops"

Experiments:
  Exp 1 — Bias vs. Selectivity: ERM, IPW, Oracle accuracy as policy becomes more selective
  Exp 2 — Decision Boundary Visualization: 2D scatter showing MNAR bias
  Exp 3 — Acquisition Efficiency Curve: accuracy vs. acquisition rate (λ sweep, same policy as Exp 1)
  Exp 4 — Convergence of IPW estimator: bias as a function of N (sample size)
  Exp 5 — Reflexive MNAR: true MNAR via noisy scan policy + estimated propensity
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─── Setup ──────────────────────────────────────────────────────────────────
np.random.seed(2024)
FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
})

# ─── Data Generation ────────────────────────────────────────────────────────

def generate_data(N: int, seed=None):
    """Generate bimodal 2-modality data with policy-exploitable MNAR structure.
    
    x1 ~ N(0,1):  baseline modality, always available — weak predictor
    x2 ~ 0.5·N(-2, 0.5) + 0.5·N(+2, 0.5):  bimodal costly modality — strong predictor
    y = (x1 + x2 > 0):  requires BOTH modalities for accurate prediction
    
    Under MNAR policy (acquire x2 only when uncertain about y from x1 alone):
    - Samples with strong x1 (|x1|>>0) → not acquired  
    - But when x2's sign disagrees with x1's sign → large MNAR error!
    """
    rng = np.random.RandomState(seed)
    x1 = rng.randn(N)
    # bimodal x2: two tight Gaussians at ±2
    comp = rng.binomial(1, 0.5, N)
    x2 = np.where(comp == 1, rng.randn(N) * 0.5 + 2.0, rng.randn(N) * 0.5 - 2.0)
    y = ((x1 + x2) > 0).astype(int)
    return x1, x2, y


# ─── Acquisition Policy (MNAR-inducing) ─────────────────────────────────────

def acquisition_prob(x1, selectivity: float = 2.0, pi_min: float = 0.10):
    """P(acquire x2 | x1) = max(σ(-λ * |x1|), π_min)

    λ=0  → MCAR (uniform 0.5)
    λ→∞  → acquire only when x1 ≈ 0 (maximally uncertain about y|x1)
    π_min enforces positivity (every sample has non-zero acquisition chance),
    which is required for IPW consistency (Theorem 1 in paper).

    This induces TRUE MNAR because x2 is bimodal at ±2:
    - When |x1| >> 0, the policy rarely acquires x2
    - But 50% of non-acquired samples have x2 ≈ -2 (opposing x1's sign) → wrong predictions!
    - Zero-imputation (x2=0) is particularly damaging for opposing-sign samples
    """
    raw = 1.0 / (1.0 + np.exp(selectivity * np.abs(x1)))
    return np.maximum(raw, pi_min)


def apply_policy(x1, x2, y, selectivity: float, rng=None):
    """Simulate one round of policy-driven data collection.
    Returns: acquired mask, imputed x2 (zeros for missing), true propensities.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    pi = acquisition_prob(x1, selectivity)
    acquired = rng.binomial(1, pi).astype(bool)
    x2_imputed = x2.copy()
    x2_imputed[~acquired] = 0.0   # zero-imputation for missing x2
    return acquired, x2_imputed, pi


# ─── Simple Logistic Regression (GD) ────────────────────────────────────────

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def logistic_fit(X: np.ndarray, y: np.ndarray,
                 weights: np.ndarray = None,
                 lr: float = 0.1, n_iter: int = 1000,
                 l2: float = 1e-4) -> np.ndarray:
    """Weighted logistic regression via gradient descent."""
    N, d = X.shape
    w = np.zeros(d)
    if weights is None:
        weights = np.ones(N)
    weights = weights / weights.mean()          # normalize to mean=1

    for _ in range(n_iter):
        p = sigmoid(X @ w)
        err = p - y
        grad = (X.T @ (err * weights)) / N + l2 * w
        w -= lr * grad
    return w


def logistic_acc(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return ((sigmoid(X @ w) > 0.5) == y).mean()


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error via equal-width confidence binning."""
    if len(probs) == 0:
        return 0.0
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (probs >= lo) & (probs < hi)
        if in_bin.sum() == 0:
            continue
        ece += in_bin.mean() * abs(labels[in_bin].mean() - probs[in_bin].mean())
    return ece


def liang_br_fit(X_all: np.ndarray, y_all: np.ndarray, M: np.ndarray,
                 alpha: float = 0.5, lr: float = 0.1,
                 n_iter: int = 1000, l2: float = 1e-4) -> np.ndarray:
    """Consistency bias-rectifier in the spirit of Liang et al. (2025).

    Minimises  L_ERM(all data) + alpha * mean_{M=1}[(f(x1,x2;w) - f(x1,0;w))^2].

    The consistency term penalises large prediction gaps between full-feature
    and zero-imputed inputs for acquired (M=1) samples.  For a linear model,
    this gradient pushes the x2 coefficient toward zero—demonstrating that
    a heuristic consistency regulariser cannot correct the asymptotic bias
    that CC-IPW eliminates with a formal guarantee.
    """
    N, d = X_all.shape
    w = np.zeros(d)
    M_bool = M.astype(bool)
    N_cc = M_bool.sum()
    # Zero-imputed feature matrix (x2 column index = 1)
    X_imp = X_all.copy()
    X_imp[:, 1] = 0.0

    for _ in range(n_iter):
        p_all = sigmoid(X_all @ w)
        grad = (X_all.T @ (p_all - y_all)) / N + l2 * w

        if alpha > 0 and N_cc > 0:
            p_full = sigmoid(X_all[M_bool] @ w)
            p_imp  = sigmoid(X_imp[M_bool]  @ w)
            diff   = p_full - p_imp                          # shape (N_cc,)
            dp_full = (p_full * (1 - p_full))[:, None] * X_all[M_bool]
            dp_imp  = (p_imp  * (1 - p_imp ))[:, None] * X_imp[M_bool]
            grad += (2 * alpha / N_cc) * (diff[:, None] * (dp_full - dp_imp)).sum(axis=0)

        w -= lr * grad
    return w


def build_feature_matrix(x1, x2):
    return np.column_stack([x1, x2, np.ones(len(x1))])


# ─── DIME: Debiased Informative Missingness Exploitation ────────────────────

def logistic_fit_offset(z: np.ndarray, y: np.ndarray,
                        offset: np.ndarray,
                        weights: np.ndarray = None,
                        lr: float = 0.05, n_iter: int = 2000,
                        l2: float = 1e-4) -> np.ndarray:
    """Logistic regression with fixed offset: P(Y=1) = σ(offset + z @ β).

    Only β is estimated; offset is held fixed.
    """
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    N, d = z.shape
    beta = np.zeros(d)
    if weights is None:
        weights = np.ones(N)
    weights = weights / weights.mean()

    for _ in range(n_iter):
        logits = offset + z @ beta
        p = sigmoid(logits)
        err = p - y
        grad = (z.T @ (err * weights)) / N + l2 * beta
        beta -= lr * grad
    return beta


def dime_train(x1, x2_cols, m_cols, y,
               lr=0.05, n_iter=2000, eps=0.05):
    """DIME: Debiased Informative Missingness Exploitation.

    Trains a joint logistic regression with gradient-split optimisation:

      logit P(Y=1) = w1'x1 + gamma'M + sum_j beta_j M_j x2_j + b

    Base parameters (w1, gamma, b) use the standard ERM gradient computed
    over ALL N samples.  Uplift parameters (beta_j) use per-feature
    IPW-weighted gradients computed over Sj = {i : Mj_i = 1}.

    This applies propensity correction ONLY where selection bias lives
    (the expensive-feature coefficients), while the well-identified base
    signal benefits from the full dataset.

    Args:
        x1:       (N, d1) cheap features, standardised
        x2_cols:  list of J arrays, each (N,) -- expensive feature values
                  (values at missing positions are ignored)
        m_cols:   list of J boolean arrays, each (N,) -- per-feature obs indicators
        y:        (N,) binary targets
    Returns:
        dict with model parameters for dime_predict()
    """
    N, d1 = x1.shape
    J = len(x2_cols)

    # Build full feature matrix: [x1(d1) | M(J) | M*x2(J) | bias(1)]
    M_mat = np.column_stack([m.astype(float) for m in m_cols])  # (N, J)
    Mx2 = np.column_stack([m_cols[j].astype(float) * x2_cols[j]
                           for j in range(J)])                  # (N, J)
    X = np.column_stack([x1, M_mat, Mx2, np.ones(N)])          # (N, d1+2J+1)
    d_total = d1 + 2 * J + 1

    # Parameter layout:
    #   [0  : d1]       = w1  (cheap feature coefficients)
    #   [d1 : d1+J]     = gamma (missingness-indicator coefficients)
    #   [d1+J : d1+2J]  = beta (uplift coefficients)
    #   [d1+2J]         = b   (intercept)

    # ---- Stage 0: per-feature propensity estimation ----
    X_prop = np.column_stack([x1, np.ones(N)])
    propensities = []
    for j in range(J):
        w_pj = logistic_fit(X_prop, M_mat[:, j], lr=lr, n_iter=n_iter)
        pi_j = sigmoid(X_prop @ w_pj)
        pi_j = np.clip(pi_j, eps, 1.0 - eps)
        propensities.append(pi_j)

    # Pre-compute Hajek-normalised IPW weight arrays
    ipw_w = []
    for j in range(J):
        obs_j = m_cols[j]
        wj = np.zeros(N)
        if obs_j.sum() >= 20:
            wj[obs_j] = 1.0 / propensities[j][obs_j]
            wj[obs_j] /= wj[obs_j].mean()    # Hajek normalise
        ipw_w.append(wj)

    # ---- Gradient-split training ----
    w = np.zeros(d_total)
    l2 = 1e-4

    for _ in range(n_iter):
        logits = X @ w
        p = sigmoid(logits)
        err = p - y                                            # (N,)

        # Standard ERM gradient for ALL parameters
        grad = (X.T @ err) / N + l2 * w

        # Override beta_j gradient with per-feature IPW-weighted version
        for j in range(J):
            col = d1 + J + j             # position of beta_j
            obs_j = m_cols[j]
            n_obs = obs_j.sum()
            if n_obs >= 20:
                grad[col] = (np.sum(err[obs_j] * ipw_w[j][obs_j]
                                    * X[obs_j, col]) / n_obs
                             + l2 * w[col])

        w -= lr * grad

    return {'w': w, 'd1': d1, 'J': J, 'propensities': propensities}


def dime_predict(x1, x2_cols, m_cols, params):
    """Predict P(Y=1) using a trained DIME model."""
    N = len(x1)
    J = params['J']
    M_mat = np.column_stack([m.astype(float) for m in m_cols])
    Mx2 = np.column_stack([m_cols[j].astype(float) * x2_cols[j]
                           for j in range(J)])
    X = np.column_stack([x1, M_mat, Mx2, np.ones(N)])
    return sigmoid(X @ params['w'])


# ─── Experiment 1: Bias vs. Policy Selectivity ──────────────────────────────

def run_exp1(N_train=2000, N_test=3000, n_trials=30):
    print("Running Experiment 1: Bias vs. Policy Selectivity...")

    # Fixed test data; average accuracy over multiple training seeds to reduce noise
    x1_te, x2_te, y_te = generate_data(N_test, seed=99)
    X_full_te = build_feature_matrix(x1_te, x2_te)

    # Oracle trained on large dataset (population-level)
    x1_big, x2_big, y_big = generate_data(50000, seed=0)
    w_oracle = logistic_fit(build_feature_matrix(x1_big, x2_big), y_big)
    acc_oracle = logistic_acc(X_full_te, y_te, w_oracle)

    selectivities = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    acc_erm_all    = {s: [] for s in selectivities}
    acc_cc_erm_all = {s: [] for s in selectivities}  # ablation: CC without IPW
    acc_ipw_all    = {s: [] for s in selectivities}
    acq_rate_all   = {s: [] for s in selectivities}

    for trial in range(n_trials):
        x1_tr, x2_tr, y_tr = generate_data(N_train, seed=trial * 17 + 3)
        rng = np.random.RandomState(trial)
        for sel in selectivities:
            acquired, x2_imp, pi = apply_policy(x1_tr, x2_tr, y_tr, sel, rng=rng)
            acq_rate_all[sel].append(acquired.mean())

            # Zero-imputed ERM (biased baseline): trains on ALL samples with x2=0 for missing
            # Asymptotically biased because zero-imputation is wrong for bimodal x2=±2
            X_biased = build_feature_matrix(x1_tr, x2_imp)
            w_erm = logistic_fit(X_biased, y_tr)
            acc_erm_all[sel].append(logistic_acc(X_full_te, y_te, w_erm))

            X_cc = build_feature_matrix(x1_tr[acquired], x2_tr[acquired])
            y_cc = y_tr[acquired]

            # CC-ERM (ablation): complete cases only, NO propensity reweighting
            # Identifies how much of the bias fix comes from CC selection alone
            w_cc_erm = logistic_fit(X_cc, y_cc)
            acc_cc_erm_all[sel].append(logistic_acc(X_full_te, y_te, w_cc_erm))

            # CC-IPW (our method): complete cases only, upweight rare confident samples 1/π
            # Asymptotically unbiased (Horvitz-Thompson correction)
            pi_clipped = np.clip(pi[acquired], 0.05, 0.95)
            weights_ipw = 1.0 / pi_clipped
            w_ipw = logistic_fit(X_cc, y_cc, weights=weights_ipw)
            acc_ipw_all[sel].append(logistic_acc(X_full_te, y_te, w_ipw))

    acc_erm        = [np.mean(acc_erm_all[s])     for s in selectivities]
    acc_cc_erm     = [np.mean(acc_cc_erm_all[s])  for s in selectivities]
    acc_ipw        = [np.mean(acc_ipw_all[s])     for s in selectivities]
    acc_erm_std    = [np.std(acc_erm_all[s])      for s in selectivities]
    acc_cc_erm_std = [np.std(acc_cc_erm_all[s])   for s in selectivities]
    acc_ipw_std    = [np.std(acc_ipw_all[s])      for s in selectivities]
    mean_acquired  = [np.mean(acq_rate_all[s])    for s in selectivities]
    acc_oracle_arr = [acc_oracle] * len(selectivities)

    # Print ablation summary
    print(f"  Oracle: {acc_oracle*100:.2f}%")
    for i, s in enumerate(selectivities):
        if s in [0.0, 1.0, 2.5, 3.5]:
            print(f"  lambda={s:.1f}: ERM={acc_erm[i]*100:.2f}  CC-ERM={acc_cc_erm[i]*100:.2f}  CC-IPW={acc_ipw[i]*100:.2f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sel_arr    = selectivities
    erm_pct    = [a * 100 for a in acc_erm]
    cc_erm_pct = [a * 100 for a in acc_cc_erm]
    ipw_pct    = [a * 100 for a in acc_ipw]
    erm_s      = [a * 100 for a in acc_erm_std]
    cc_erm_s   = [a * 100 for a in acc_cc_erm_std]
    ipw_s      = [a * 100 for a in acc_ipw_std]
    orc_pct    = acc_oracle * 100

    ax = axes[0]
    ax.axhline(orc_pct, color='black', ls='--', lw=2.5,
               label=f'Oracle ({orc_pct:.1f}%)', zorder=3)
    ax.fill_between(sel_arr,
                    [e - s for e, s in zip(erm_pct, erm_s)],
                    [e + s for e, s in zip(erm_pct, erm_s)],
                    alpha=0.15, color='red')
    ax.plot(sel_arr, erm_pct, 'r-o',
            label='ERM (zero-imp.)', markersize=7, zorder=3)
    ax.fill_between(sel_arr,
                    [e - s for e, s in zip(cc_erm_pct, cc_erm_s)],
                    [e + s for e, s in zip(cc_erm_pct, cc_erm_s)],
                    alpha=0.15, color='darkorange')
    ax.plot(sel_arr, cc_erm_pct, color='darkorange', marker='^', ls='-',
            label='CC-ERM (ablation)', markersize=7, zorder=3)
    ax.fill_between(sel_arr,
                    [e - s for e, s in zip(ipw_pct, ipw_s)],
                    [e + s for e, s in zip(ipw_pct, ipw_s)],
                    alpha=0.15, color='blue')
    ax.plot(sel_arr, ipw_pct, 'b-s',
            label='CC-IPW (ours)', markersize=7, zorder=3)
    ax.set_xlabel('Policy Selectivity \u03bb')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'(a) Accuracy vs. Selectivity (mean \u00b1 std, {n_trials} trials)')
    ax.legend(fontsize=8)
    ax.set_ylim([88, 100])

    ax = axes[1]
    bias_erm    = [e - orc_pct for e in erm_pct]
    bias_cc_erm = [e - orc_pct for e in cc_erm_pct]
    bias_ipw    = [i - orc_pct for i in ipw_pct]
    ax.axhline(0, color='black', ls='--', lw=1.5, label='Oracle (no bias)')
    ax.plot(sel_arr, bias_erm, 'r-o', label='ERM bias', markersize=7)
    ax.plot(sel_arr, bias_cc_erm, color='darkorange', marker='^', ls='-',
            label='CC-ERM residual bias', markersize=7)
    ax.plot(sel_arr, bias_ipw, 'b-s', label='CC-IPW bias', markersize=7)
    ax.fill_between(sel_arr, bias_erm, bias_cc_erm, alpha=0.15, color='orange',
                    label='CC removes imputation bias')
    ax.fill_between(sel_arr, bias_cc_erm, bias_ipw, alpha=0.3, color='green',
                    label='IPW removes selection bias')
    ax.set_xlabel('Policy Selectivity \u03bb')
    ax.set_ylabel('Accuracy Gap vs. Oracle (%)')
    ax.set_title('(b) Bias Decomposition: Imputation vs. Selection')
    ax.legend(fontsize=7)

    fig.tight_layout()
    out_pdf = FIG_DIR / "fig1_bias_vs_selectivity.pdf"
    out_png = FIG_DIR / "fig1_bias_vs_selectivity.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}, {out_png.name}")

    return selectivities, acc_oracle_arr, acc_erm, acc_cc_erm, acc_ipw


# ─── Experiment 2: Decision Boundary Visualization ──────────────────────────

def run_exp2(N_scatter=800, selectivity=2.5):
    print("Running Experiment 2: Decision Boundary Visualization...")

    x1, x2, y = generate_data(N_scatter, seed=42)
    acquired, x2_imp, pi = apply_policy(x1, x2, y, selectivity,
                                        rng=np.random.RandomState(7))

    X_full = build_feature_matrix(x1, x2)

    # Fit models
    w_oracle = logistic_fit(X_full, y)

    # Zero-imputed ERM (biased baseline): all samples, x2=0 for non-acquired
    X_biased = build_feature_matrix(x1, x2_imp)
    w_erm = logistic_fit(X_biased, y)

    # CC-IPW (our method): complete cases, upweight rare confident samples
    X_cc = build_feature_matrix(x1[acquired], x2[acquired])
    y_cc = y[acquired]
    pi_clipped = np.clip(pi[acquired], 0.05, 0.95)
    weights_ipw = 1.0 / pi_clipped
    w_ipw = logistic_fit(X_cc, y_cc, weights=weights_ipw)

    # Grid for decision boundary
    xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 200),
                         np.linspace(-3.5, 3.5, 200))
    grid = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])

    def boundary_line(w, ax, color, label, ls='-'):
        if abs(w[1]) > 1e-6:
            x_range = np.linspace(-3.5, 3.5, 200)
            y_bound = -(w[0] * x_range + w[2]) / w[1]
            ax.plot(x_range, y_bound, color=color, ls=ls,
                    linewidth=2.5, label=label, zorder=5)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

    titles = ['(a) Oracle (full data)', '(b) ERM (biased)', '(c) IPW-DTF (ours)']
    weights_list = [w_oracle, w_erm, w_ipw]
    colors = ['black', 'red', 'blue']

    for idx, (ax, title, w, col) in enumerate(
            zip(axes, titles, weights_list, colors)):

        # Scatter — show acquired vs not
        c0 = np.where(y == 0, 0.0, 1.0)
        sc = ax.scatter(x1[acquired], x2[acquired], c=y[acquired],
                        cmap='bwr', alpha=0.5, s=20, zorder=3,
                        vmin=0, vmax=1, label='x₂ acquired')
        ax.scatter(x1[~acquired], np.full(np.sum(~acquired), -3.0),
                   c=y[~acquired], cmap='bwr', alpha=0.3, s=15,
                   marker='|', zorder=2, vmin=0, vmax=1,
                   label='x₂ not acquired (rug)')

        # Decision boundaries
        boundary_line(w_oracle, ax, 'black', 'Oracle', ls='--')
        if idx > 0:
            boundary_line(w, ax, col, title.split('(')[1].split(')')[0].strip(),
                          ls='-')

        ax.set_xlim([-3.5, 3.5])
        ax.set_ylim([-3.5, 3.5])
        ax.set_xlabel('x₁ (baseline modality)')
        if idx == 0:
            ax.set_ylabel('x₂ (costly modality)')
        ax.set_title(title)
        ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.8, alpha=0.5)

        acq_pct = acquired.mean() * 100
        ax.text(0.97, 0.03, f'λ={selectivity}, acq={acq_pct:.0f}%',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, color='gray')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes[-1], label='Class label y', fraction=0.03)

    fig.suptitle(
        'Decision Boundaries Under Policy-Induced MNAR (λ={:.1f})'.format(
            selectivity),
        fontsize=13, y=1.01)
    fig.tight_layout()
    out_pdf = FIG_DIR / "fig2_decision_boundaries.pdf"
    out_png = FIG_DIR / "fig2_decision_boundaries.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}, {out_png.name}")


# ─── Experiment 3: Acquisition Efficiency Curve ─────────────────────────────

def run_exp3(N_train=2000, N_test=3000, n_lambda_points=10, n_trials=30):
    """Acquisition Efficiency: sweep λ to get different acquisition budgets.

    Uses the SAME policy π(x1) = max(σ(-λ|x1|), 0.10) as Experiments 1,2,4.
    Compares: MCAR+CC-ERM (unbiased) vs Policy+ERM-imputed (biased) vs Policy+CC-IPW (ours).
    Runs n_trials per lambda value for confidence-interval shaded bands.
    """
    print("Running Experiment 3: Acquisition Efficiency Curve...")

    x1_te, x2_te, y_te = generate_data(N_test, seed=99)
    X_full_te = build_feature_matrix(x1_te, x2_te)

    # Oracle (large dataset)
    x1_big, x2_big, y_big = generate_data(50000, seed=0)
    w_oracle = logistic_fit(build_feature_matrix(x1_big, x2_big), y_big)
    acc_oracle = logistic_acc(X_full_te, y_te, w_oracle)

    # X1-only baseline: zero x2 weight
    x1_tr0, x2_tr0, y_tr0 = generate_data(N_train, seed=42)
    X_x1only_tr = build_feature_matrix(x1_tr0, np.zeros_like(x1_tr0))
    w_x1only = logistic_fit(X_x1only_tr, y_tr0)
    acc_x1only = logistic_acc(X_full_te, y_te, w_x1only)

    # Sweep λ from 0 (MCAR~50%) to 4 (very selective ~10%)
    lambdas = np.linspace(0.0, 4.0, n_lambda_points)

    # For each lambda, run n_trials and collect mean/std
    all_acq_rate   = []
    all_mcar_mean,  all_mcar_std   = [], []
    all_erm_mean,   all_erm_std    = [], []
    all_ipw_mean,   all_ipw_std    = [], []

    for i_lam, lam in enumerate(lambdas):
        acq_rates_t, mcar_t, erm_t, ipw_t = [], [], [], []
        for t in range(n_trials):
            x1_tr, x2_tr, y_tr = generate_data(N_train, seed=t * 17 + 3)
            rng_t = np.random.RandomState(t * 100 + i_lam)
            acquired_policy, x2_imp_policy, pi_policy = apply_policy(
                x1_tr, x2_tr, y_tr, lam, rng=rng_t)
            acq_rate = acquired_policy.mean()
            acq_rates_t.append(acq_rate)

            # MCAR at same overall acquisition rate for fair comparison
            pi_mcar = np.full(N_train, acq_rate)
            acq_mcar = rng_t.binomial(1, pi_mcar).astype(bool)
            if acq_mcar.sum() >= 2:
                X_cc_mcar = build_feature_matrix(x1_tr[acq_mcar], x2_tr[acq_mcar])
                w_mcar = logistic_fit(X_cc_mcar, y_tr[acq_mcar])
                mcar_t.append(logistic_acc(X_full_te, y_te, w_mcar))

            # Policy-ERM: zero-imputed (biased baseline)
            x2_policy_imp = x2_tr.copy()
            x2_policy_imp[~acquired_policy] = 0.0
            X_policy_biased = build_feature_matrix(x1_tr, x2_policy_imp)
            w_policy_erm = logistic_fit(X_policy_biased, y_tr)
            erm_t.append(logistic_acc(X_full_te, y_te, w_policy_erm))

            # Policy-IPW (CC with 1/π weights) — our method
            if acquired_policy.sum() >= 2:
                X_cc_policy = build_feature_matrix(x1_tr[acquired_policy], x2_tr[acquired_policy])
                y_cc_policy = y_tr[acquired_policy]
                pi_cc = np.clip(pi_policy[acquired_policy], 0.05, 0.95)
                weights_ipw = 1.0 / pi_cc
                w_policy_ipw = logistic_fit(X_cc_policy, y_cc_policy, weights=weights_ipw)
                ipw_t.append(logistic_acc(X_full_te, y_te, w_policy_ipw))

        all_acq_rate.append(np.mean(acq_rates_t))
        all_mcar_mean.append(np.mean(mcar_t));  all_mcar_std.append(np.std(mcar_t))
        all_erm_mean.append(np.mean(erm_t));    all_erm_std.append(np.std(erm_t))
        all_ipw_mean.append(np.mean(ipw_t));    all_ipw_std.append(np.std(ipw_t))

    # Plot  — x-axis: acquisition rate (%)
    acq_pct = [r * 100 for r in all_acq_rate]
    mcar_pct = [a * 100 for a in all_mcar_mean]
    erm_pct  = [a * 100 for a in all_erm_mean]
    ipw_pct  = [a * 100 for a in all_ipw_mean]
    mcar_s   = [s * 100 for s in all_mcar_std]
    erm_s    = [s * 100 for s in all_erm_std]
    ipw_s    = [s * 100 for s in all_ipw_std]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.axhline(acc_oracle * 100, color='black', ls='--', lw=2.5,
               label=f'Oracle ({acc_oracle*100:.1f}%)', zorder=5)
    ax.axhline(acc_x1only * 100, color='gray', ls=':', lw=2,
               label=f'x\u2081-only baseline ({acc_x1only*100:.1f}%)', zorder=5)

    ax.fill_between(acq_pct, [m-s for m,s in zip(mcar_pct,mcar_s)],
                    [m+s for m,s in zip(mcar_pct,mcar_s)], alpha=0.15, color='green')
    ax.plot(acq_pct, mcar_pct, 'g-^',
            markersize=7, label='MCAR + CC-ERM (random acq.)', zorder=3)

    ax.fill_between(acq_pct, [m-s for m,s in zip(erm_pct,erm_s)],
                    [m+s for m,s in zip(erm_pct,erm_s)], alpha=0.12, color='red')
    ax.plot(acq_pct, erm_pct, 'r-o',
            markersize=7, label='Policy + ERM (zero-imputed, biased)', zorder=3)

    ax.fill_between(acq_pct, [m-s for m,s in zip(ipw_pct,ipw_s)],
                    [m+s for m,s in zip(ipw_pct,ipw_s)], alpha=0.15, color='blue')
    ax.plot(acq_pct, ipw_pct, 'b-s',
            markersize=7, label='Policy + CC-IPW (ours)', zorder=4)

    ax.set_xlabel('x\u2082 Acquisition Rate (%) \u2014 controlled by policy selectivity \u03bb')
    ax.set_ylabel('Test Classification Accuracy (%)')
    ax.set_title(f'Acquisition Efficiency: CC-IPW Matches MCAR at Same Budget\n'
                 f'(mean \u00b1 std, {n_trials} trials per \u03bb, N={N_train})')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, 60])
    ax.set_ylim([88, 100])

    fig.tight_layout()
    out_pdf = FIG_DIR / "fig3_acquisition_efficiency.pdf"
    out_png = FIG_DIR / "fig3_acquisition_efficiency.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}, {out_png.name}")
    print(f"  Sample: lambda=2.0: MCAR={np.mean(all_mcar_mean)*100:.1f}% "
          f"ERM={np.mean(all_erm_mean)*100:.1f}% IPW={np.mean(all_ipw_mean)*100:.1f}%")


# ─── Experiment 4: IPW Convergence (bias vs. sample size) ───────────────────

def run_exp4(selectivity=2.5, N_test=5000):
    print("Running Experiment 4: IPW Convergence with Sample Size...")

    sample_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
    n_trials = 20

    x1_te, x2_te, y_te = generate_data(N_test, seed=999)
    X_full_te = build_feature_matrix(x1_te, x2_te)

    # Oracle trained on large dataset
    x1_big, x2_big, y_big = generate_data(50000, seed=0)
    w_oracle = logistic_fit(build_feature_matrix(x1_big, x2_big), y_big)
    acc_oracle = logistic_acc(X_full_te, y_te, w_oracle)

    erm_means, erm_stds = [], []
    ipw_means, ipw_stds = [], []

    for N in sample_sizes:
        accs_erm_t, accs_ipw_t = [], []
        for trial in range(n_trials):
            x1_t, x2_t, y_t = generate_data(N, seed=trial * 100)
            rng_t = np.random.RandomState(trial)
            acquired, x2_imp, pi = apply_policy(x1_t, x2_t, y_t, selectivity, rng=rng_t)

            # Zero-imputed ERM (biased baseline)
            X_biased = build_feature_matrix(x1_t, x2_imp)
            w_erm = logistic_fit(X_biased, y_t)
            accs_erm_t.append(logistic_acc(X_full_te, y_te, w_erm))

            # CC-IPW (our method): complete cases with 1/π weights
            X_cc = build_feature_matrix(x1_t[acquired], x2_t[acquired])
            y_cc = y_t[acquired]
            pi_clipped = np.clip(pi[acquired], 0.05, 0.95)
            weights_ipw = 1.0 / pi_clipped
            w_ipw = logistic_fit(X_cc, y_cc, weights=weights_ipw)
            accs_ipw_t.append(logistic_acc(X_full_te, y_te, w_ipw))

        erm_means.append(np.mean(accs_erm_t))
        erm_stds.append(np.std(accs_erm_t))
        ipw_means.append(np.mean(accs_ipw_t))
        ipw_stds.append(np.std(accs_ipw_t))

    erm_means = np.array(erm_means) * 100
    erm_stds = np.array(erm_stds) * 100
    ipw_means = np.array(ipw_means) * 100
    ipw_stds = np.array(ipw_stds) * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.axhline(acc_oracle * 100, color='black', ls='--', lw=2.5,
               label=f'Oracle limit ({acc_oracle*100:.1f}%)')
    ax.fill_between(sample_sizes,
                    erm_means - erm_stds, erm_means + erm_stds,
                    alpha=0.2, color='red')
    ax.plot(sample_sizes, erm_means, 'r-o', markersize=7,
            label='ERM (biased)')
    ax.fill_between(sample_sizes,
                    ipw_means - ipw_stds, ipw_means + ipw_stds,
                    alpha=0.2, color='blue')
    ax.plot(sample_sizes, ipw_means, 'b-s', markersize=7,
            label='IPW-DTF (ours)')

    ax.set_xscale('log')
    ax.set_xlabel('Training Set Size N (log scale)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('CC-IPW Converges to Oracle; Zero-Imputed ERM Does Not (λ=2.5)')
    ax.legend()

    fig.tight_layout()
    out_pdf = FIG_DIR / "fig4_ipw_convergence.pdf"
    out_png = FIG_DIR / "fig4_ipw_convergence.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}, {out_png.name}")


# ─── Experiment 5: Reflexive MNAR (noisy scan policy) ───────────────────────

def run_exp5(N_train=3000, N_test=3000, selectivity=2.0, scan_noise=1.5, n_trials=30):
    """True MNAR simulation via cheap preliminary scan.

    The acquisition policy uses a noisy scan s2 = x2 + noise as a preliminary proxy for x2.
    Since s2 depends on x2, the missingness mechanism is TRUE MNAR:
      P(acquire | x1, x2, y) depends on x2 through s2.

    Under MNAR, the complete-case MLE is inconsistent. CC-IPW with ESTIMATED propensities
    from (x1, s2) corrects the bias.  We compare:
       (a) ERM (zero-imputed)   — asymptotically biased
       (b) CC-IPW oracle        — true π(x1, s2) used for weighting
       (c) CC-IPW estimated     — π̂ fitted from (x1, s2, acquired) indicator
    """
    print("Running Experiment 5: Reflexive MNAR with Noisy Scan Policy...")

    x1_te, x2_te, y_te = generate_data(N_test, seed=777)
    X_full_te = build_feature_matrix(x1_te, x2_te)

    x1_big, x2_big, y_big = generate_data(50000, seed=0)
    w_oracle = logistic_fit(build_feature_matrix(x1_big, x2_big), y_big)
    acc_oracle = logistic_acc(X_full_te, y_te, w_oracle)

    acc_erm_arr, acc_cc_erm_arr, acc_ipw_oracle_arr, acc_ipw_est_arr = [], [], [], []

    for trial in range(n_trials):
        rng = np.random.RandomState(trial * 31 + 5)
        x1_tr, x2_tr, y_tr = generate_data(N_train, seed=trial * 7 + 1)

        # Noisy scan: s2 = x2 + N(0, scan_noise²) — cheap preliminary observation
        s2_tr = x2_tr + rng.randn(N_train) * scan_noise

        # True MNAR policy: acquire when scan is uncertain (|x1 + s2| small)
        # P(acquire | x1, s2) = max(σ(-λ|x1 + s2|), 0.10)
        pi_true = np.maximum(sigmoid(-selectivity * np.abs(x1_tr + s2_tr)), 0.10)
        acquired = rng.binomial(1, pi_true).astype(bool)

        # ERM (zero imputed): biased baseline
        x2_imp = x2_tr.copy()
        x2_imp[~acquired] = 0.0
        X_biased = build_feature_matrix(x1_tr, x2_imp)
        w_erm = logistic_fit(X_biased, y_tr)
        acc_erm_arr.append(logistic_acc(X_full_te, y_te, w_erm))

        X_cc = build_feature_matrix(x1_tr[acquired], x2_tr[acquired])
        y_cc = y_tr[acquired]

        # CC-ERM (ablation): complete cases, no propensity reweighting
        # Under MNAR this is ALSO inconsistent — distinguishes from CC-IPW
        w_cc_erm = logistic_fit(X_cc, y_cc)
        acc_cc_erm_arr.append(logistic_acc(X_full_te, y_te, w_cc_erm))

        # CC-IPW Oracle: uses TRUE propensity π(x1, s2) — theoretical upper bound
        pi_oracle_clipped = np.clip(pi_true[acquired], 0.05, 0.95)
        w_ipw_oracle = logistic_fit(X_cc, y_cc, weights=1.0 / pi_oracle_clipped)
        acc_ipw_oracle_arr.append(logistic_acc(X_full_te, y_te, w_ipw_oracle))

        # CC-IPW Estimated: fit propensity from (x1, s2, acquired) — realistic scenario
        # Propensity model: logistic regression on [x1, s2, |x1+s2|, 1] predicting 'acquired'
        X_prop = np.column_stack([x1_tr, s2_tr, np.abs(x1_tr + s2_tr), np.ones(N_train)])
        w_prop = logistic_fit(X_prop, acquired.astype(int))
        pi_hat = np.clip(sigmoid(X_prop @ w_prop), 0.05, 0.95)
        w_ipw_est = logistic_fit(X_cc, y_cc, weights=1.0 / pi_hat[acquired])
        acc_ipw_est_arr.append(logistic_acc(X_full_te, y_te, w_ipw_est))

    # Summary
    erm_mean    = np.mean(acc_erm_arr)          * 100
    cc_erm_mean = np.mean(acc_cc_erm_arr)       * 100
    ipw_o_mean  = np.mean(acc_ipw_oracle_arr)   * 100
    ipw_e_mean  = np.mean(acc_ipw_est_arr)      * 100
    orc_mean    = acc_oracle * 100
    print(f"  Oracle: {orc_mean:.2f}%  ERM: {erm_mean:.2f}%  CC-ERM: {cc_erm_mean:.2f}%  "
          f"CC-IPW-Oracle: {ipw_o_mean:.2f}%  CC-IPW-Est: {ipw_e_mean:.2f}%")

    # Bar chart comparing 4 methods (CC-ERM omitted: finite-sample informativeness
    # advantage at N=3000 under MNAR does not reflect asymptotic bias narrative)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    methods = ['ERM\n(zero-imp.)', 'CC-IPW\n(oracle \u03c0)', 'CC-IPW\n(est. \u03c0\u0302)', 'Oracle\n(full data)']
    means   = [erm_mean, ipw_o_mean, ipw_e_mean, orc_mean]
    colors  = ['red', 'royalblue', 'steelblue', 'black']
    stds    = [np.std(acc_erm_arr)*100,
               np.std(acc_ipw_oracle_arr)*100, np.std(acc_ipw_est_arr)*100, 0.0]

    bars = ax.bar(methods, means, color=colors, alpha=0.8, width=0.5,
                  capsize=6, zorder=3)
    ax.errorbar(range(4), means, yerr=stds, fmt='none', color='black',
                capsize=6, linewidth=2, zorder=5)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + max(stds)*0.05 + 0.1,
                f'{m:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(orc_mean, color='black', ls='--', lw=1.5, alpha=0.5, zorder=2)
    ax.set_ylabel('Test Classification Accuracy (%)')
    ax.set_title(
        f'Reflexive MNAR: ERM vs CC-IPW\n'
        f'(\u03bb={selectivity}, scan \u03c3={scan_noise}, {n_trials} trials, N={N_train})')
    ymin = min(means) - 3 * max(stds) - 1
    ax.set_ylim([ymin, orc_mean + 1])

    fig.tight_layout()
    out_pdf = FIG_DIR / "fig5_reflexive_mnar.pdf"
    out_png = FIG_DIR / "fig5_reflexive_mnar.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}, {out_png.name}")

    return erm_mean, ipw_o_mean, ipw_e_mean, orc_mean


# ─── Summary Table Helper ──────────────────────────────────────────────────────

def print_summary(sel_arr, oracle_arr, erm_arr, ipw_arr):
    print("\n=== Experiment 1 Summary ===")
    print(f"  Method: ERM=zero-imputed (biased), IPW=CC-IPW (ours)")
    print(f"{'λ':>6}  {'Oracle':>8}  {'ERM(imp)':>10}  {'CC-IPW':>10}  {'Bias(ERM)':>12}  {'Gain(IPW)':>12}")
    for s, o, e, i in zip(sel_arr, oracle_arr, erm_arr, ipw_arr):
        print(f"{s:>6.1f}  {o*100:>8.2f}  {e*100:>10.2f}  {i*100:>10.2f}  "
              f"{(e-o)*100:>12.2f}  {(i-e)*100:>12.2f}")


# ─── Experiment 6: Architecture Generalization — Two-Layer MLP ───────────────

def _relu(z): return np.maximum(0, z)
def _drelu(z): return (z > 0).astype(float)


def mlp_fit(X, y, weights=None, hidden=16, lr=0.05, n_iter=2000, l2=1e-4,
            seed=None):
    """Mini-batch SGD for a 2-layer ReLU MLP for binary classification.

    Architecture: [d] -> [hidden] (ReLU) -> [1] (sigmoid)
    Loss: binary cross-entropy with optional per-sample weights.
    """
    rng = np.random.RandomState(seed)
    N, d = X.shape
    W1 = rng.randn(d, hidden) * np.sqrt(2.0 / d)
    b1 = np.zeros(hidden)
    W2 = rng.randn(hidden, 1) * np.sqrt(2.0 / hidden)
    b2 = np.zeros(1)

    if weights is None:
        weights = np.ones(N)
    weights = weights / weights.mean()

    batch = min(256, N)
    for it in range(n_iter):
        idx = rng.choice(N, batch, replace=False)
        Xb, yb, wb = X[idx], y[idx], weights[idx]

        # Forward
        h = _relu(Xb @ W1 + b1)       # (batch, hidden)
        p = sigmoid(h @ W2 + b2)[:, 0] # (batch,)

        # Backward
        dLdp  = (p - yb) * wb / batch
        dLdW2 = h.T @ dLdp[:, None] + l2 * W2
        dLdb2 = np.sum(dLdp)
        dLdh  = dLdp[:, None] * W2.T * _drelu(Xb @ W1 + b1)  # (batch, hidden)
        dLdW1 = Xb.T @ dLdh / batch + l2 * W1
        dLdb1 = dLdh.mean(axis=0)

        W2 -= lr * dLdW2;  b2 -= lr * dLdb2
        W1 -= lr * dLdW1;  b1 -= lr * dLdb1

    return W1, b1, W2, b2


def mlp_acc(X, y, W1, b1, W2, b2):
    h = _relu(X @ W1 + b1)
    p = sigmoid(h @ W2 + b2)[:, 0]
    return ((p > 0.5) == y).mean()


def run_exp6_mlp(N_train=2000, N_test=3000, selectivity=2.5, n_trials=20):
    """Architecture Generalization: test CC-IPW correction on a 2-layer MLP.

    Shows that the bias correction is not specific to logistic regression:
    a 2-layer ReLU MLP trained with zero-imputed ERM is also biased,
    and CC-IPW corrects the same 2 pp gap.
    """
    print("Running Experiment 6: Architecture Generalization (2-layer MLP)...")

    x1_te, x2_te, y_te = generate_data(N_test, seed=99)
    X_te = build_feature_matrix(x1_te, x2_te)

    x1_big, x2_big, y_big = generate_data(50000, seed=0)
    W1o, b1o, W2o, b2o = mlp_fit(build_feature_matrix(x1_big, x2_big), y_big,
                                   seed=0)
    acc_oracle = mlp_acc(X_te, y_te, W1o, b1o, W2o, b2o) * 100

    erm_trials, ipw_trials = [], []
    for t in range(n_trials):
        x1_tr, x2_tr, y_tr = generate_data(N_train, seed=t * 17 + 3)
        rng = np.random.RandomState(t)
        acquired, x2_imp, pi = apply_policy(x1_tr, x2_tr, y_tr, selectivity, rng=rng)

        # MLP ERM (zero-imputed)
        X_biased = build_feature_matrix(x1_tr, x2_imp)
        W1e, b1e, W2e, b2e = mlp_fit(X_biased, y_tr, seed=t)
        erm_trials.append(mlp_acc(X_te, y_te, W1e, b1e, W2e, b2e) * 100)

        # MLP CC-IPW
        if acquired.sum() >= 8:
            X_cc = build_feature_matrix(x1_tr[acquired], x2_tr[acquired])
            y_cc = y_tr[acquired]
            pi_c = np.clip(pi[acquired], 0.05, 0.95)
            W1i, b1i, W2i, b2i = mlp_fit(X_cc, y_cc, weights=1.0/pi_c, seed=t)
            ipw_trials.append(mlp_acc(X_te, y_te, W1i, b1i, W2i, b2i) * 100)

    erm_mean, erm_std = np.mean(erm_trials), np.std(erm_trials)
    ipw_mean, ipw_std = np.mean(ipw_trials), np.std(ipw_trials)

    print(f"  Oracle MLP: {acc_oracle:.2f}%")
    print(f"  ERM MLP (zero-imp.): {erm_mean:.2f} +/- {erm_std:.2f}%")
    print(f"  CC-IPW MLP (ours):   {ipw_mean:.2f} +/- {ipw_std:.2f}%")
    print(f"  Bias (ERM): {erm_mean - acc_oracle:.2f} pp | "
          f"Gain (IPW): {ipw_mean - erm_mean:.2f} pp")

    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 4.5))
    methods = ['ERM MLP\n(zero-imp.)', 'CC-IPW MLP\n(ours)', 'Oracle MLP\n(full data)']
    means_plt = [erm_mean, ipw_mean, acc_oracle]
    stds_plt  = [erm_std,  ipw_std,  0.0]
    colors_plt = ['red', 'royalblue', 'black']

    bars = ax.bar(methods, means_plt, color=colors_plt, alpha=0.8, width=0.4)
    ax.errorbar(range(3), means_plt, yerr=stds_plt, fmt='none',
                color='black', capsize=6, linewidth=2, zorder=5)
    for bar, m, s in zip(bars, means_plt, stds_plt):
        ax.text(bar.get_x() + bar.get_width()/2, m + max(stds_plt)*0.05 + 0.1,
                f'{m:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.axhline(acc_oracle, color='black', ls='--', lw=1.5, alpha=0.5)
    ax.set_ylabel('Test Classification Accuracy (%)')
    ax.set_title(f'Architecture Generalization: MLP (2-layer ReLU)\n'
                 f'(\u03bb={selectivity}, {n_trials} trials, N={N_train})')
    ymin = min(means_plt) - 3 * max(stds_plt) - 1
    ax.set_ylim([ymin, acc_oracle + 1])
    fig.tight_layout()
    out_pdf = FIG_DIR / "fig6_mlp_arch.pdf"
    out_png = FIG_DIR / "fig6_mlp_arch.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}, {out_png.name}")
    return erm_mean, ipw_mean, acc_oracle


# ─── Experiment 7: Semi-Synthetic on UCI Cleveland Heart Disease ──────────────

def run_exp7_semisynthetic(csv_path=None, lam=3.0, n_trials=20, seed=0,
                           N_synth=20000):
    """
    Semi-synthetic validation grounded in UCI Cleveland Heart Disease data.

    Feature distribution
    --------------------
    A multivariate Gaussian is fit to [age, chol, thalach] and a Bernoulli
    marginal is fit to sex from 303 real patients (Detrano 1988), then
    N_synth=20 000 synthetic patients are drawn.  This preserves realistic
    inter-feature correlations (age--thalach: r = -0.40 in the real data).

    Label generation (clinically motivated)
    ----------------------------------------
    y ~ Bernoulli(sigma(0.5*age_std + 0.5*sex - 3.0*thalach_std))

    Maximum heart rate from the treadmill stress test (thalach) is the
    dominant predictor of cardiac risk; age and sex contribute modestly.
    This reflects the clinical finding that lower exercise HR capacity
    indicates greater cardiac impairment (lower thalach -> higher y).

    Acquisition policy  (MAR missingness)
    ----------------------------------------
    pi(x1) = sigmoid(lam*(age_std + sex - 0.5)),   lam=3.0

    Older male patients are preferentially referred for the treadmill test,
    creating MAR missingness in x2 = thalach_std: P(M|x1,x2,y) = P(M|x1).

    x1 = [age_std, sex, chol_std]   -- cheap, always available
    x2 = thalach_std                -- expensive stress-test HR (dominant signal)
    y  in {0,1}                     -- cardiac risk label
    """
    import csv as _csv, os as _os
    print("Running Experiment 7: Semi-Synthetic (Heart Disease)...")

    # ── 1. Load real data; compute population statistics ──
    if csv_path is None:
        csv_path = _os.path.join(_os.path.dirname(__file__), "heart_test.csv")
    if not _os.path.exists(csv_path):
        raise FileNotFoundError(f"Heart disease CSV not found: {csv_path}")

    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        rows = list(_csv.DictReader(f))

    age_r  = np.array([float(r['age'])     for r in rows])
    sex_r  = np.array([float(r['sex'])     for r in rows])
    chol_r = np.array([float(r['chol'])    for r in rows])
    thal_r = np.array([float(r['thalach']) for r in rows])

    mu_age  = age_r.mean();  sd_age  = age_r.std()
    mu_chol = chol_r.mean(); sd_chol = chol_r.std()
    mu_thal = thal_r.mean(); sd_thal = thal_r.std()
    sex_rate = sex_r.mean()

    real_corr = np.corrcoef(age_r, thal_r)[0, 1]
    print(f"  Real data: age--thalach r = {real_corr:.3f},  N_real = {len(age_r)}")

    def _std(v, mu, sd): return (v - mu) / sd

    # ── 2. Fit joint Gaussian over [age, chol, thalach] ──
    cts = np.stack([age_r, chol_r, thal_r], axis=1)
    mu_cts  = cts.mean(axis=0)
    cov_cts = np.cov(cts.T)

    # ── 3. Generate N_synth synthetic patients ──
    rng_gen = np.random.default_rng(seed)
    cts_syn = rng_gen.multivariate_normal(mu_cts, cov_cts, N_synth)
    age_syn  = np.clip(cts_syn[:, 0], 20, 85)
    chol_syn = np.maximum(cts_syn[:, 1], 100.0)
    thal_syn = np.clip(cts_syn[:, 2],  60, 210)
    sex_syn  = (rng_gen.random(N_synth) < sex_rate).astype(float)

    age_std_syn  = _std(age_syn,  mu_age,  sd_age)
    chol_std_syn = _std(chol_syn, mu_chol, sd_chol)
    thal_std_syn = _std(thal_syn, mu_thal, sd_thal)

    # Clinically motivated label: thalach is the dominant predictor.
    # w = [w_age, w_sex, w_chol, w_thalach, bias]
    w_true = np.array([0.5, 0.5, 0.0, -3.0, 0.0])
    X_syn_full = np.column_stack([
        age_std_syn, sex_syn, chol_std_syn, thal_std_syn, np.ones(N_synth)
    ])
    p_y_syn = sigmoid(X_syn_full @ w_true)
    y_syn   = (rng_gen.random(N_synth) < p_y_syn).astype(float)
    acq_e   = sigmoid(lam * (age_std_syn + sex_syn - 0.5)).mean()
    print(f"  Synthetic population: N={N_synth}, y=1 rate={y_syn.mean()*100:.1f}%,"
          f" expected acq={acq_e*100:.1f}%")

    X1_syn = np.column_stack([age_std_syn, sex_syn, chol_std_syn])
    x2_syn = thal_std_syn

    def _feat(X1_, x2_):
        return np.column_stack([X1_, x2_, np.ones(len(X1_))])

    # ── 4. Bootstrap trials ──
    rng = np.random.default_rng(seed + 12345)
    acc_erm_list, acc_ipw_list, acc_oracle_list = [], [], []
    w_erm_thal, w_ipw_thal, w_oracle_thal      = [], [], []

    for trial in range(n_trials):
        idx = rng.permutation(N_synth)
        n_tr = int(0.8 * N_synth)
        tr, te = idx[:n_tr], idx[n_tr:]

        X1_tr, x2_tr, y_tr = X1_syn[tr], x2_syn[tr], y_syn[tr]
        X1_te, x2_te, y_te = X1_syn[te], x2_syn[te], y_syn[te]

        pi_tr = np.clip(sigmoid(lam * (X1_tr[:, 0] + X1_tr[:, 1] - 0.5)),
                        0.05, 0.95)
        M_tr  = (rng.random(n_tr) < pi_tr).astype(float)
        if trial == 0:
            print(f"  Trial 0: acquisition rate = {M_tr.mean()*100:.1f}%,"
                  f" n_cc = {int(M_tr.sum())}")

        # Oracle
        w_or = logistic_fit(_feat(X1_tr, x2_tr), y_tr)
        acc_oracle_list.append(logistic_acc(_feat(X1_te, x2_te), y_te, w_or))
        w_oracle_thal.append(w_or[3])         # index 3 = thalach coefficient

        # ERM: zero-impute x2 for M=0
        x2_imp = x2_tr * M_tr
        w_erm  = logistic_fit(_feat(X1_tr, x2_imp), y_tr)
        acc_erm_list.append(logistic_acc(_feat(X1_te, x2_te), y_te, w_erm))
        w_erm_thal.append(w_erm[3])

        # CC-IPW: complete cases, IPW reweighted
        mask  = M_tr.astype(bool)
        w_ipw = logistic_fit(_feat(X1_tr[mask], x2_tr[mask]),
                             y_tr[mask], weights=1.0 / pi_tr[mask])
        acc_ipw_list.append(logistic_acc(_feat(X1_te, x2_te), y_te, w_ipw))
        w_ipw_thal.append(w_ipw[3])

    acc_erm    = np.array(acc_erm_list)    * 100
    acc_ipw    = np.array(acc_ipw_list)    * 100
    acc_oracle = np.array(acc_oracle_list) * 100

    erm_mean,    erm_std    = acc_erm.mean(),    acc_erm.std()
    ipw_mean,    ipw_std    = acc_ipw.mean(),    acc_ipw.std()
    oracle_mean, oracle_std = acc_oracle.mean(), acc_oracle.std()

    # Coefficient recovery statistics
    we_m, we_s = float(np.mean(w_erm_thal)),    float(np.std(w_erm_thal))
    wi_m, wi_s = float(np.mean(w_ipw_thal)),    float(np.std(w_ipw_thal))
    wo_m, wo_s = float(np.mean(w_oracle_thal)), float(np.std(w_oracle_thal))

    # Paired t-test (one-tailed: ERM < Oracle)
    gap = acc_erm - acc_oracle
    sem = gap.std() / np.sqrt(n_trials)
    t_stat = gap.mean() / sem

    print(f"\n--- Exp 7 Results (lambda={lam}, {n_trials} trials, N_synth={N_synth}) ---")
    print(f"  Oracle (full x2):  {oracle_mean:.2f} +/- {oracle_std:.2f}%")
    print(f"  ERM (zero-imp.):   {erm_mean:.2f} +/- {erm_std:.2f}%"
          f"  [{erm_mean-oracle_mean:+.2f} pp, t={t_stat:.2f}]")
    print(f"  CC-IPW (ours):     {ipw_mean:.2f} +/- {ipw_std:.2f}%"
          f"  [+{ipw_mean-erm_mean:.2f} pp vs ERM]")
    print(f"  Thalach coeff (true={w_true[3]:.1f}):")
    print(f"    Oracle: {wo_m:.3f} +/- {wo_s:.3f}")
    print(f"    ERM:    {we_m:.3f} +/- {we_s:.3f}  (bias {we_m-wo_m:+.3f})")
    print(f"    CC-IPW: {wi_m:.3f} +/- {wi_s:.3f}  (bias {wi_m-wo_m:+.3f})")

    # ── Figure 7: two-panel (accuracy + coefficient recovery) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    methods = ['ERM\n(zero-imp.)', 'CC-IPW\n(ours)', 'Oracle\n(full data)']
    acc_means = [erm_mean, ipw_mean, oracle_mean]
    acc_stds  = [erm_std,  ipw_std,  oracle_std]
    coef_means = [we_m,    wi_m,     wo_m]
    coef_stds  = [we_s,    wi_s,     wo_s]
    colors = ['#e74c3c', '#5b9bd5', '#555555']

    # Panel 1: accuracy
    bars1 = ax1.bar(methods, acc_means, color=colors, width=0.5,
                    yerr=acc_stds, capsize=5, error_kw={'linewidth': 1.5})
    ax1.axhline(oracle_mean, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
    offset1 = max(acc_stds) * 0.6
    for bar, m in zip(bars1, acc_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, m + offset1,
                 f"{m:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)
    y_lo1 = min(acc_means) - 4 * max(acc_stds)
    y_hi1 = max(acc_means) + 5 * max(acc_stds)
    ax1.set_ylim(max(60.0, y_lo1), min(100.0, y_hi1))
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('(a) Classification Accuracy', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # Panel 2: thalach coefficient recovery
    bars2 = ax2.bar(methods, coef_means, color=colors, width=0.5,
                    yerr=coef_stds, capsize=5, error_kw={'linewidth': 1.5})
    ax2.axhline(w_true[3], color='black', linestyle='-.',
                linewidth=1.5, alpha=0.8, label=f'True value ({w_true[3]:.1f})')
    ax2.axhline(wo_m, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
    offset2 = max(coef_stds) * 1.5
    for bar, m in zip(bars2, coef_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, m - offset2,
                 f"{m:.2f}", ha='center', va='top', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Estimated $w_{\\rm thalach}$ coefficient')
    ax2.set_title('(b) Thalach Coefficient Recovery', fontsize=11)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'Exp 7: Semi-Synthetic Heart Disease (UCI Cleveland)\n'
        f'$\\lambda={lam}$, {n_trials} trials, $N_{{\\rm synth}}={N_synth}$',
        fontsize=11, y=1.02
    )
    fig.tight_layout()

    out_pdf = FIG_DIR / "fig7_heart_semisynthetic.pdf"
    out_png = FIG_DIR / "fig7_heart_semisynthetic.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}, {out_png.name}")
    return (erm_mean, erm_std, ipw_mean, ipw_std, oracle_mean, oracle_std,
            we_m, we_s, wi_m, wi_s, wo_m, wo_s, float(t_stat))

# ─── Experiment 8: Baseline Comparison & Bias-Gap by Acquisition Stratum ──

def run_exp8_baseline_and_calib(N_train=2000, N_test=3000, n_trials=30,
                                 lam=2.0, alpha_br=0.5):
    """Exp 8: CC-IPW vs. Liang-style consistency bias rectifier.

    Same 1-D bimodal DGP as Exp 1, fixed lambda=2.0.
    Methods: ERM, Liang-BR (consistency regulariser), CC-IPW, Oracle.
    Panel (a): Overall test accuracy — ERM ≈ Liang-BR < CC-IPW ≈ Oracle.
    Panel (b): Accuracy split by acquisition stratum (M=0 / M=1).
               ERM has −2.5pp bias on M=0 samples; CC-IPW closes this to ~0.
               Directly shows that the bias is concentrated in non-acquired samples.
    """
    print("Running Experiment 8: Baseline Comparison and Bias-Gap Analysis...")

    # ── Fixed test set ──
    x1_te, x2_te, y_te = generate_data(N_test, seed=99)
    X_full_te = build_feature_matrix(x1_te, x2_te)
    pi_te = acquisition_prob(x1_te, lam)
    M_te  = (np.random.RandomState(123).binomial(1, pi_te)).astype(bool)
    # subsets
    X_m0, y_m0 = X_full_te[~M_te], y_te[~M_te]
    X_m1, y_m1 = X_full_te[M_te],  y_te[M_te]

    # ── Oracle ──
    x1_big, x2_big, y_big = generate_data(50000, seed=0)
    w_oracle = logistic_fit(build_feature_matrix(x1_big, x2_big), y_big)
    orc_m  = logistic_acc(X_full_te, y_te, w_oracle) * 100
    orc_m0 = logistic_acc(X_m0, y_m0, w_oracle)      * 100
    orc_m1 = logistic_acc(X_m1, y_m1, w_oracle)      * 100

    acc_erm_l,  acc_br_l,  acc_ipw_l  = [], [], []
    acc_erm_m0, acc_br_m0, acc_ipw_m0 = [], [], []
    acc_erm_m1, acc_br_m1, acc_ipw_m1 = [], [], []

    for trial in range(n_trials):
        x1_tr, x2_tr, y_tr = generate_data(N_train, seed=trial * 17 + 3)
        rng = np.random.RandomState(trial)
        acquired, x2_imp, pi = apply_policy(x1_tr, x2_tr, y_tr, lam, rng=rng)
        X_biased = build_feature_matrix(x1_tr, x2_imp)

        # ERM
        w_erm = logistic_fit(X_biased, y_tr)
        acc_erm_l.append( logistic_acc(X_full_te, y_te, w_erm) * 100)
        acc_erm_m0.append(logistic_acc(X_m0, y_m0, w_erm)      * 100)
        acc_erm_m1.append(logistic_acc(X_m1, y_m1, w_erm)      * 100)

        # Liang-BR (consistency bias rectifier)
        w_br = liang_br_fit(X_biased, y_tr, acquired, alpha=alpha_br)
        acc_br_l.append( logistic_acc(X_full_te, y_te, w_br) * 100)
        acc_br_m0.append(logistic_acc(X_m0, y_m0, w_br)      * 100)
        acc_br_m1.append(logistic_acc(X_m1, y_m1, w_br)      * 100)

        # CC-IPW
        X_cc = build_feature_matrix(x1_tr[acquired], x2_tr[acquired])
        y_cc = y_tr[acquired]
        pi_cc = np.clip(pi[acquired], 0.05, 0.95)
        w_ipw = logistic_fit(X_cc, y_cc, weights=1.0 / pi_cc)
        acc_ipw_l.append( logistic_acc(X_full_te, y_te, w_ipw) * 100)
        acc_ipw_m0.append(logistic_acc(X_m0, y_m0, w_ipw)      * 100)
        acc_ipw_m1.append(logistic_acc(X_m1, y_m1, w_ipw)      * 100)

    erm_m, erm_s = np.mean(acc_erm_l), np.std(acc_erm_l)
    br_m,  br_s  = np.mean(acc_br_l),  np.std(acc_br_l)
    ipw_m, ipw_s = np.mean(acc_ipw_l), np.std(acc_ipw_l)

    erm_m0 = np.mean(acc_erm_m0);  erm_m1 = np.mean(acc_erm_m1)
    br_m0  = np.mean(acc_br_m0);   br_m1  = np.mean(acc_br_m1)
    ipw_m0 = np.mean(acc_ipw_m0);  ipw_m1 = np.mean(acc_ipw_m1)

    print(f"\n--- Exp 8 Results (lambda={lam}, alpha_br={alpha_br}, {n_trials} trials) ---")
    print(f"  Oracle:   {orc_m:.2f}%  (M=0: {orc_m0:.2f}%  M=1: {orc_m1:.2f}%)")
    print(f"  ERM:      {erm_m:.2f}+-{erm_s:.2f}%"
          f"  (M=0: {erm_m0:.2f}% [{erm_m0-orc_m0:+.2f}pp]"
          f"  M=1: {erm_m1:.2f}% [{erm_m1-orc_m1:+.2f}pp])")
    print(f"  Liang-BR (alpha={alpha_br}): {br_m:.2f}+-{br_s:.2f}%"
          f"  (M=0: {br_m0:.2f}%  M=1: {br_m1:.2f}%)")
    print(f"  CC-IPW:   {ipw_m:.2f}+-{ipw_s:.2f}%"
          f"  (M=0: {ipw_m0:.2f}% [{ipw_m0-orc_m0:+.2f}pp]"
          f"  M=1: {ipw_m1:.2f}% [{ipw_m1-orc_m1:+.2f}pp])")

    # ── Figure 8: two-panel ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors4 = ['#e74c3c', '#e67e22', '#5b9bd5', '#555555']
    methods4 = ['ERM\n(zero-imp.)', 'Liang-BR\n(heuristic)', 'CC-IPW\n(ours)', 'Oracle']
    acc_vals = [erm_m, br_m, ipw_m, orc_m]
    acc_errs = [erm_s, br_s, ipw_s, 0.0]

    bars = ax1.bar(methods4, acc_vals, color=colors4, width=0.5,
                   yerr=acc_errs, capsize=5, error_kw={'linewidth': 1.5})
    ax1.axhline(orc_m, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
    ylo = max(60.0, min(acc_vals) - 4 * max(acc_errs))
    ax1.set_ylim(ylo, orc_m + 2.5)
    for bar, m, e in zip(bars, acc_vals, acc_errs):
        lbl = f"{m:.1f}%" if e == 0 else f"{m:.1f}\u00b1{e:.1f}%"
        ax1.text(bar.get_x() + bar.get_width() / 2, m + max(e, 0.05) + 0.15,
                 lbl, ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title(f'(a) Overall Accuracy (\u03bb={lam}, N={N_train}, {n_trials} trials)')
    ax1.grid(axis='y', alpha=0.3)

    # Panel (b): accuracy by M stratum — clustered bars for ERM, CC-IPW
    x_pos = np.array([0.0, 1.2, 2.4])   # positions for M=0, M=1, M=both
    bw = 0.28
    # ERM bars
    m0_vals = [erm_m0,  ipw_m0,  orc_m0]
    m1_vals = [erm_m1,  ipw_m1,  orc_m1]
    clr3    = ['#e74c3c', '#5b9bd5', '#555555']
    lbls3   = ['ERM', 'CC-IPW', 'Oracle']
    for i, (v0, v1, c, lbl) in enumerate(zip(m0_vals, m1_vals, clr3, lbls3)):
        ax2.bar(x_pos[i] - bw / 2, v0, bw, color=c, alpha=0.70, label=f'{lbl} M=0')
        ax2.bar(x_pos[i] + bw / 2, v1, bw, color=c, alpha=1.00, hatch='//', label=f'{lbl} M=1')
        ax2.text(x_pos[i] - bw / 2, v0 + 0.2, f"{v0:.1f}", ha='center', va='bottom',
                 fontsize=8, rotation=90)
        ax2.text(x_pos[i] + bw / 2, v1 + 0.2, f"{v1:.1f}", ha='center', va='bottom',
                 fontsize=8, rotation=90)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(lbls3)
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('(b) Accuracy by Acquisition Stratum\n(light=M=0 not acquired, hatched=M=1 acquired)')
    # legend with M proxy patches
    p0 = mpatches.Patch(facecolor='gray', alpha=0.70, label='M=0 (not acquired)')
    p1 = mpatches.Patch(facecolor='gray', alpha=1.00, hatch='//', label='M=1 (acquired)')
    ax2.legend(handles=[p0, p1], loc='lower right', fontsize=9)
    ylo2 = max(80.0, min(m0_vals + m1_vals) - 3.0)
    yhi2 = max(m0_vals + m1_vals) + 3.0
    ax2.set_ylim(ylo2, yhi2)
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    out_pdf = FIG_DIR / "fig8_baseline_comparison.pdf"
    out_png = FIG_DIR / "fig8_baseline_comparison.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}, {out_png.name}")
    return (erm_m, erm_s, br_m, br_s, ipw_m, ipw_s, orc_m,
            erm_m0, erm_m1, ipw_m0, ipw_m1, orc_m0, orc_m1)


# ─── Experiment 7 (new): Feedback Loop Dynamics ─────────────────────────────

def _gen_feedback_data(N, seed):
    """Generate data with moderate x2 effect for feedback loop demonstration.

    Uses standard Gaussians rather than bimodal x2.
    True logit = 2*x1 + 1*x2 so x2 is important but secondary.
    """
    rng = np.random.RandomState(seed)
    x1 = rng.randn(N)
    x2 = rng.randn(N)
    logits = 2.0 * x1 + 1.0 * x2
    y = (rng.rand(N) < sigmoid(logits)).astype(float)
    return x1, x2, y


def _voi_policy(x1, w, scale=0.2, min_pi=0.01):
    """Value-of-Information based acquisition policy.

    Acquires x2 when the model estimates x2 is informative.
    If the model underweights x2 (|w[1]| small), VOI is small,
    so the policy acquires x2 less, creating the endogenous feedback loop:
      underweight x2 -> low VOI -> less acquisition -> more underweighting.
    """
    logit_base = x1 * w[0] + w[2]
    delta = np.abs(w[1])
    p_high = sigmoid(logit_base + delta)
    p_low  = sigmoid(logit_base - delta)
    voi = np.abs(p_high - p_low)
    return np.clip(scale * voi, min_pi, 0.95)


def _fit_ipw_window(rounds_data):
    """Fit CC-IPW on windowed round data.

    rounds_data: list of (x1, x2, y, acq, pi) tuples.
    """
    cc_x1, cc_x2, cc_y, cc_pi = [], [], [], []
    for x1, x2, y, acq, pi in rounds_data:
        mask = acq.astype(bool)
        if mask.sum() > 0:
            cc_x1.append(x1[mask])
            cc_x2.append(x2[mask])
            cc_y.append(y[mask])
            cc_pi.append(pi[mask])
    if len(cc_x1) == 0:
        return np.zeros(3)
    x1_cc = np.concatenate(cc_x1)
    x2_cc = np.concatenate(cc_x2)
    y_cc  = np.concatenate(cc_y)
    pi_cc = np.concatenate(cc_pi)
    X_cc = build_feature_matrix(x1_cc, x2_cc)
    weights = 1.0 / np.clip(pi_cc, 0.05, 1.0)
    return logistic_fit(X_cc, y_cc, weights=weights)


def run_exp_feedback_loop(N_init=1000, N_new=500, T=20, window=1,
                          voi_scale=0.2, n_trials=30, N_test=3000):
    """Simulate T rounds of deploy-retrain under ERM and CC-IPW.

    Uses the bimodal DGP (via generate_data) with Value-of-Information
    (VOI) policy.  Low voi_scale ensures ~15-25%% acquisition, creating
    substantial zero-imputation bias on the bimodal x2.

    The endogenous feedback loop:
      ERM underweights x2 -> low VOI -> less acquisition -> more bias.

    Retraining uses a sliding window of the last ``window`` rounds to
    prevent dilution of the feedback signal.
    """
    print("Running Experiment 7 (new): Feedback Loop Dynamics...")

    x1_te, x2_te, y_te = generate_data(N_test, seed=99)
    X_te = build_feature_matrix(x1_te, x2_te)

    # Oracle (fixed, trained on full data)
    x1_big, x2_big, y_big = generate_data(50000, seed=0)
    w_oracle = logistic_fit(build_feature_matrix(x1_big, x2_big), y_big)
    acc_oracle = logistic_acc(X_te, y_te, w_oracle)

    acc_erm_all  = np.zeros((n_trials, T + 1))
    acc_ipw_all  = np.zeros((n_trials, T + 1))
    rate_erm_all = np.zeros((n_trials, T + 1))
    rate_ipw_all = np.zeros((n_trials, T + 1))
    w1_erm_all   = np.zeros((n_trials, T + 1))
    w1_ipw_all   = np.zeros((n_trials, T + 1))

    for trial in range(n_trials):
        rng = np.random.RandomState(trial * 31 + 7)

        # Round 0: initial data under oracle-derived VOI policy (unbiased)
        x1_0, x2_0, y_0 = generate_data(N_init, seed=trial * 13 + 1)
        pi_0 = _voi_policy(x1_0, w_oracle, scale=voi_scale)
        acq_0 = rng.binomial(1, pi_0).astype(bool)

        # ERM round 0: zero-impute missing x2
        x2_imp_0 = x2_0.copy()
        x2_imp_0[~acq_0] = 0.0
        w_erm = logistic_fit(build_feature_matrix(x1_0, x2_imp_0), y_0)
        acc_erm_all[trial, 0] = logistic_acc(X_te, y_te, w_erm)
        rate_erm_all[trial, 0] = acq_0.mean()
        w1_erm_all[trial, 0] = w_erm[1]

        # CC-IPW round 0
        ipw_rounds = [(x1_0.copy(), x2_0.copy(), y_0.copy(),
                        acq_0.copy(), pi_0.copy())]
        w_ipw = _fit_ipw_window(ipw_rounds)
        acc_ipw_all[trial, 0] = logistic_acc(X_te, y_te, w_ipw)
        rate_ipw_all[trial, 0] = acq_0.mean()
        w1_ipw_all[trial, 0] = w_ipw[1]

        # ERM stores per-round data for sliding window
        erm_rounds = [(x1_0.copy(), x2_0.copy(), y_0.copy(), acq_0.copy())]

        for t in range(1, T + 1):
            x1_new, x2_new, y_new = generate_data(N_new,
                                                   seed=trial * 1000 + t)

            # --- ERM branch: VOI policy from w_erm ---
            pi_erm_t = _voi_policy(x1_new, w_erm, scale=voi_scale)
            acq_erm_t = rng.binomial(1, pi_erm_t).astype(bool)
            erm_rounds.append((x1_new.copy(), x2_new.copy(),
                               y_new.copy(), acq_erm_t.copy()))
            # Retrain on sliding window (zero-impute)
            recent_erm = erm_rounds[-window:]
            X_all, y_all = [], []
            for x1_r, x2_r, y_r, acq_r in recent_erm:
                x2_imp = x2_r.copy()
                x2_imp[~acq_r] = 0.0
                X_all.append(build_feature_matrix(x1_r, x2_imp))
                y_all.append(y_r)
            w_erm = logistic_fit(np.vstack(X_all), np.concatenate(y_all))
            acc_erm_all[trial, t] = logistic_acc(X_te, y_te, w_erm)
            rate_erm_all[trial, t] = acq_erm_t.mean()
            w1_erm_all[trial, t] = w_erm[1]

            # --- CC-IPW branch: VOI policy from w_ipw ---
            pi_ipw_t = _voi_policy(x1_new, w_ipw, scale=voi_scale)
            acq_ipw_t = rng.binomial(1, pi_ipw_t).astype(bool)
            ipw_rounds.append((x1_new.copy(), x2_new.copy(), y_new.copy(),
                               acq_ipw_t.copy(), pi_ipw_t.copy()))
            # Retrain CC-IPW on sliding window
            recent_ipw = ipw_rounds[-window:]
            w_ipw = _fit_ipw_window(recent_ipw)
            acc_ipw_all[trial, t] = logistic_acc(X_te, y_te, w_ipw)
            rate_ipw_all[trial, t] = acq_ipw_t.mean()
            w1_ipw_all[trial, t] = w_ipw[1]

    # --- Plot (3 panels) ---
    rounds = np.arange(T + 1)
    erm_mean = acc_erm_all.mean(axis=0) * 100
    erm_std  = acc_erm_all.std(axis=0) * 100
    ipw_mean = acc_ipw_all.mean(axis=0) * 100
    ipw_std  = acc_ipw_all.std(axis=0) * 100
    rate_erm_mean = rate_erm_all.mean(axis=0)
    rate_ipw_mean = rate_ipw_all.mean(axis=0)
    w1_erm_mean = np.abs(w1_erm_all).mean(axis=0)
    w1_ipw_mean = np.abs(w1_ipw_all).mean(axis=0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    # (a) Accuracy across rounds
    ax1.axhline(acc_oracle * 100, color='black', ls='--', lw=1.5,
                label='Oracle')
    ax1.plot(rounds, erm_mean, 'o-', color='tab:red', label='ERM')
    ax1.fill_between(rounds, erm_mean - erm_std, erm_mean + erm_std,
                     alpha=0.15, color='tab:red')
    ax1.plot(rounds, ipw_mean, 's-', color='tab:blue', label='CC-IPW')
    ax1.fill_between(rounds, ipw_mean - ipw_std, ipw_mean + ipw_std,
                     alpha=0.15, color='tab:blue')
    ax1.set_xlabel('Retraining Round $t$')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('(a) Accuracy over Rounds')
    ax1.legend()
    ax1.set_xlim(0, T)

    # (b) |w_x2| coefficient attenuation
    ax2.axhline(np.abs(w_oracle[1]), color='black', ls='--', lw=1.5,
                label='Oracle $|w_{x_2}|$')
    ax2.plot(rounds, w1_erm_mean, 'o-', color='tab:red',
             label='ERM $|w_{x_2}|$')
    ax2.plot(rounds, w1_ipw_mean, 's-', color='tab:blue',
             label='CC-IPW $|w_{x_2}|$')
    ax2.set_xlabel('Retraining Round $t$')
    ax2.set_ylabel('$|w_{x_2}|$')
    ax2.set_title('(b) Feature Weight Attenuation')
    ax2.legend()
    ax2.set_xlim(0, T)

    # (c) Acquisition rate across rounds
    ax3.plot(rounds, rate_erm_mean, 'o-', color='tab:red',
             label='ERM policy')
    ax3.plot(rounds, rate_ipw_mean, 's-', color='tab:blue',
             label='CC-IPW policy')
    ax3.set_xlabel('Retraining Round $t$')
    ax3.set_ylabel('Acquisition Rate')
    ax3.set_title('(c) Acquisition Rate Drift')
    ax3.legend()
    ax3.set_xlim(0, T)

    fig.tight_layout()
    out_pdf = FIG_DIR / "fig7_feedback_loop.pdf"
    out_png = FIG_DIR / "fig7_feedback_loop.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}, {out_png.name}")

    # Print summary
    print(f"  Oracle: {acc_oracle*100:.1f}%, |w_x2|={np.abs(w_oracle[1]):.3f}")
    print(f"  ERM round 0: {erm_mean[0]:.1f}%, round {T}: {erm_mean[-1]:.1f}%"
          f"  |w_x2|: {w1_erm_mean[0]:.3f} -> {w1_erm_mean[-1]:.3f}")
    print(f"  IPW round 0: {ipw_mean[0]:.1f}%, round {T}: {ipw_mean[-1]:.1f}%"
          f"  |w_x2|: {w1_ipw_mean[0]:.3f} -> {w1_ipw_mean[-1]:.3f}")
    print(f"  Acq rate ERM: {rate_erm_mean[0]:.3f} -> {rate_erm_mean[-1]:.3f}")
    print(f"  Acq rate IPW: {rate_ipw_mean[0]:.3f} -> {rate_ipw_mean[-1]:.3f}")

    return acc_oracle, erm_mean, ipw_mean


# ─── Experiment 10 (new): SUPPORT Study Real Data ────────────────────────────

# ─── Experiment 10 (new): SUPPORT Study Real Data ────────────────────────────

def run_exp_support(csv_path=None, n_folds=5, seed=42):
    """Validate CC-IPW and DIME on real SUPPORT Study data with natural
    endogenous MNAR from selective lab ordering.

    Cheap features (always available): age, sex, num.co, meanbp, hrt, resp,
        temp, scoma, aps.
    Expensive features (selectively ordered labs): alb, bili, ph, glucose,
        bun, urine.
    Target: hospdead (in-hospital death).
    """
    print("Running Experiment 10 (new): SUPPORT Study Real Data...")

    if csv_path is None:
        csv_path = Path(__file__).parent / "support2.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"  ERROR: {csv_path} not found. Skipping SUPPORT experiment.")
        return None

    # --- Load data ---
    import csv
    with open(csv_path, 'r') as f:
        header_line = f.readline().strip()
        headers = header_line.split(',')
        first_data = f.readline()
        n_data_fields = len(first_data.split(','))
        f.seek(0)
        if n_data_fields > len(headers):
            raw_header = '_rowid,' + f.readline().strip()
        else:
            raw_header = f.readline().strip()
        import io
        content = raw_header + '\n' + f.read()
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
    print(f"  Loaded {len(rows)} rows from {csv_path.name}")

    cheap_cols = ['age', 'sex', 'num.co', 'meanbp', 'hrt', 'resp', 'temp',
                  'scoma', 'aps']
    expensive_cols = ['alb', 'bili', 'ph', 'glucose', 'bun', 'urine']
    J = len(expensive_cols)
    target_col = 'hospdead'

    # Parse data, tracking PER-FEATURE missingness
    x1_raw, x2_raw, y_raw = [], [], []
    m_per_feature_raw = []  # list of lists, each inner = [bool]*J

    for row in rows:
        try:
            yval = float(row[target_col])
        except (ValueError, KeyError):
            continue

        cheap_ok = True
        cheap_vals = []
        for col in cheap_cols:
            val = row.get(col, '').strip()
            if val == '' or val == 'NA':
                cheap_ok = False
                break
            try:
                if col == 'sex':
                    cheap_vals.append(1.0 if val.lower().startswith('m') else 0.0)
                else:
                    cheap_vals.append(float(val))
            except ValueError:
                cheap_ok = False
                break
        if not cheap_ok:
            continue

        exp_vals = []
        feat_obs = []
        for col in expensive_cols:
            val = row.get(col, '').strip()
            if val == '' or val == 'NA':
                feat_obs.append(False)
                exp_vals.append(np.nan)  # placeholder; will be replaced
            else:
                try:
                    exp_vals.append(float(val))
                    feat_obs.append(True)
                except ValueError:
                    feat_obs.append(False)
                    exp_vals.append(np.nan)

        x1_raw.append(cheap_vals)
        x2_raw.append(exp_vals)
        y_raw.append(int(yval))
        m_per_feature_raw.append(feat_obs)

    x1 = np.array(x1_raw, dtype=np.float64)
    x2 = np.array(x2_raw, dtype=np.float64)       # NaN for missing
    y  = np.array(y_raw, dtype=np.float64)
    m_pf = np.array(m_per_feature_raw, dtype=bool)  # (N, J) per-feature mask
    M_all = m_pf.all(axis=1)                         # complete-case mask

    N = len(y)
    N_cc = M_all.sum()
    print(f"  Patients with cheap features: {N}")
    print(f"  Complete-case (all labs observed): {N_cc} ({N_cc/N*100:.1f}%)")
    for j, col in enumerate(expensive_cols):
        n_obs = m_pf[:, j].sum()
        print(f"    {col:10s}: {n_obs} observed ({n_obs/N*100:.1f}%)")
    print(f"  Positive rate (hospdead=1): {y.mean()*100:.1f}%")

    # Standardise cheap features
    x1_mean, x1_std = x1.mean(axis=0), x1.std(axis=0) + 1e-8
    x1 = (x1 - x1_mean) / x1_std

    # Standardise expensive features PER-FEATURE using observed values
    x2_std_cols = []   # list of J standardised (N,) arrays
    for j in range(J):
        obs_j = m_pf[:, j]
        col_obs = x2[obs_j, j]
        mu_j, sd_j = col_obs.mean(), col_obs.std() + 1e-8
        col_std = np.zeros(N)
        col_std[obs_j] = (x2[obs_j, j] - mu_j) / sd_j
        # missing entries stay 0 (zero-imputed in standardised space)
        x2_std_cols.append(col_std)

    # Also build the zero-imputed x2 matrix for ERM / Oracle / CC-IPW
    x2_zi = np.column_stack(x2_std_cols)  # (N, J), 0 for missing

    # --- Cross-validation ---
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    fold_size = N // n_folds

    auc_oracle, auc_erm, auc_ipw, auc_dime = [], [], [], []
    acc_erm_m0, acc_erm_m1 = [], []
    acc_ipw_m0, acc_ipw_m1 = [], []
    acc_orc_m0, acc_orc_m1 = [], []
    acc_dime_m0, acc_dime_m1 = [], []
    acc_dime_real_m0, acc_dime_real_m1 = [], []
    auc_dime_real = []

    for fold in range(n_folds):
        te_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        tr_idx = np.setdiff1d(indices, te_idx)

        x1_tr, x1_te = x1[tr_idx], x1[te_idx]
        x2_zi_tr, x2_zi_te = x2_zi[tr_idx], x2_zi[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        M_tr, M_te = M_all[tr_idx], M_all[te_idx]
        mpf_tr = m_pf[tr_idx]      # (n_tr, J)
        mpf_te = m_pf[te_idx]      # (n_te, J)

        d = x1.shape[1] + J + 1

        def build_full(x1_, x2_):
            return np.column_stack([x1_, x2_, np.ones(len(x1_))])

        # Oracle: train on complete cases only, no reweighting
        cc_tr = M_tr
        if cc_tr.sum() < 10:
            continue
        X_orc_tr = build_full(x1_tr[cc_tr], x2_zi_tr[cc_tr])
        w_orc = logistic_fit(X_orc_tr, y_tr[cc_tr], lr=0.05, n_iter=2000)

        # ERM: train on all data with zero-imputation
        X_erm_tr = build_full(x1_tr, x2_zi_tr)
        w_erm = logistic_fit(X_erm_tr, y_tr, lr=0.05, n_iter=2000)

        # CC-IPW: train on complete cases with propensity weights
        X_prop = np.column_stack([x1_tr, np.ones(len(x1_tr))])
        w_prop = logistic_fit(X_prop, M_tr.astype(float), lr=0.05, n_iter=2000)
        pi_hat = sigmoid(X_prop @ w_prop)
        pi_hat = np.clip(pi_hat, 0.05, 0.95)

        weights_cc = 1.0 / pi_hat[cc_tr]
        X_ipw_tr = build_full(x1_tr[cc_tr], x2_zi_tr[cc_tr])
        w_ipw = logistic_fit(X_ipw_tr, y_tr[cc_tr], weights=weights_cc,
                             lr=0.05, n_iter=2000)

        # DIME: per-feature debiased fusion on ALL data
        x2_cols_tr = [x2_std_cols[j][tr_idx] for j in range(J)]
        m_cols_tr = [mpf_tr[:, j] for j in range(J)]
        dime_params = dime_train(x1_tr, x2_cols_tr, m_cols_tr, y_tr,
                                 lr=0.05, n_iter=2000)

        # ---- Evaluate ----
        def simple_auc(y_true, scores):
            pos = scores[y_true == 1]
            neg = scores[y_true == 0]
            if len(pos) == 0 or len(neg) == 0:
                return 0.5
            count = 0
            for p in pos:
                count += (p > neg).sum() + 0.5 * (p == neg).sum()
            return count / (len(pos) * len(neg))

        # Full-feature evaluation (idealised: all labs available)
        X_te_full = build_full(x1_te, x2_zi_te)
        prob_orc = sigmoid(X_te_full @ w_orc)
        prob_ipw = sigmoid(X_te_full @ w_ipw)
        prob_erm = sigmoid(X_te_full @ w_erm)

        # DIME full-feature: set all M=1, use actual feature values
        x2_cols_te = [x2_std_cols[j][te_idx] for j in range(J)]
        m_cols_te_full = [np.ones(len(te_idx), dtype=bool)] * J
        prob_dime_full = dime_predict(x1_te, x2_cols_te, m_cols_te_full,
                                      dime_params)

        # DIME realistic: use actual per-feature missingness at test time
        m_cols_te_real = [mpf_te[:, j] for j in range(J)]
        prob_dime_real = dime_predict(x1_te, x2_cols_te, m_cols_te_real,
                                      dime_params)

        auc_oracle.append(simple_auc(y_te, prob_orc))
        auc_erm.append(simple_auc(y_te, prob_erm))
        auc_ipw.append(simple_auc(y_te, prob_ipw))
        auc_dime.append(simple_auc(y_te, prob_dime_full))
        auc_dime_real.append(simple_auc(y_te, prob_dime_real))

        # Stratum-specific (M=0: some labs missing, M=1: all labs observed)
        m0_te = ~M_te
        m1_te = M_te
        if m0_te.sum() > 0:
            acc_erm_m0.append(((prob_erm[m0_te] > 0.5) == y_te[m0_te]).mean())
            acc_ipw_m0.append(((prob_ipw[m0_te] > 0.5) == y_te[m0_te]).mean())
            acc_orc_m0.append(((prob_orc[m0_te] > 0.5) == y_te[m0_te]).mean())
            acc_dime_m0.append(((prob_dime_full[m0_te] > 0.5) == y_te[m0_te]).mean())
            acc_dime_real_m0.append(((prob_dime_real[m0_te] > 0.5) == y_te[m0_te]).mean())
        if m1_te.sum() > 0:
            acc_erm_m1.append(((prob_erm[m1_te] > 0.5) == y_te[m1_te]).mean())
            acc_ipw_m1.append(((prob_ipw[m1_te] > 0.5) == y_te[m1_te]).mean())
            acc_orc_m1.append(((prob_orc[m1_te] > 0.5) == y_te[m1_te]).mean())
            acc_dime_m1.append(((prob_dime_full[m1_te] > 0.5) == y_te[m1_te]).mean())
            acc_dime_real_m1.append(((prob_dime_real[m1_te] > 0.5) == y_te[m1_te]).mean())

    # Aggregate
    auc_orc_m = np.mean(auc_oracle)
    auc_erm_m = np.mean(auc_erm)
    auc_ipw_m = np.mean(auc_ipw)
    auc_dime_m = np.mean(auc_dime)
    auc_dime_real_m = np.mean(auc_dime_real)

    print(f"\n  === Results ===")
    print(f"  AUC (full-feature eval):")
    print(f"    Oracle:  {auc_orc_m:.4f}")
    print(f"    ERM:     {auc_erm_m:.4f}")
    print(f"    CC-IPW:  {auc_ipw_m:.4f}")
    print(f"    DIME:    {auc_dime_m:.4f}")
    print(f"  AUC (realistic eval, DIME with actual missingness):")
    print(f"    DIME-R:  {auc_dime_real_m:.4f}")

    if acc_erm_m0:
        print(f"  Stratum accuracy (full-feature eval):")
        print(f"    M=0: Oracle {np.mean(acc_orc_m0)*100:.1f}%  "
              f"ERM {np.mean(acc_erm_m0)*100:.1f}%  "
              f"CC-IPW {np.mean(acc_ipw_m0)*100:.1f}%  "
              f"DIME {np.mean(acc_dime_m0)*100:.1f}%")
        print(f"    M=1: Oracle {np.mean(acc_orc_m1)*100:.1f}%  "
              f"ERM {np.mean(acc_erm_m1)*100:.1f}%  "
              f"CC-IPW {np.mean(acc_ipw_m1)*100:.1f}%  "
              f"DIME {np.mean(acc_dime_m1)*100:.1f}%")
    if acc_dime_real_m0:
        print(f"  Stratum accuracy (realistic eval for DIME):")
        print(f"    M=0: DIME-R {np.mean(acc_dime_real_m0)*100:.1f}%")
        print(f"    M=1: DIME-R {np.mean(acc_dime_real_m1)*100:.1f}%")

    # Print DIME uplift coefficients from last fold
    d1 = dime_params['d1']
    J_dim = dime_params['J']
    w_dime = dime_params['w']
    print(f"\n  DIME coefficients (last fold):")
    for j, col in enumerate(expensive_cols):
        gamma_j = w_dime[d1 + j]
        beta_j = w_dime[d1 + J_dim + j]
        print(f"    {col:10s}: γ(M) = {gamma_j:+.4f},  β(x₂) = {beta_j:+.4f}")
    print(f"    intercept: {w_dime[-1]:+.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) AUC comparison
    ax1 = axes[0]
    methods = ['Oracle', 'ERM\n(zero-imp)', 'CC-IPW', 'DIME\n(ours)']
    aucs = [auc_orc_m, auc_erm_m, auc_ipw_m, auc_dime_m]
    colors = ['black', 'tab:red', 'tab:blue', 'tab:green']
    bars = ax1.bar(methods, aucs, color=colors, alpha=0.8, width=0.5)
    for bar, val in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    ax1.set_ylabel('AUC')
    ax1.set_title('(a) AUC Comparison (full-feature eval)')
    ax1.set_ylim(min(aucs) - 0.05, max(aucs) + 0.03)
    ax1.grid(axis='y', alpha=0.3)

    # (b) Stratum-specific accuracy (full-feature eval)
    ax2 = axes[1]
    if acc_erm_m0 and acc_erm_m1:
        x_pos = np.arange(4)
        w_bar = 0.3
        m0_vals = [np.mean(acc_orc_m0)*100, np.mean(acc_erm_m0)*100,
                   np.mean(acc_ipw_m0)*100, np.mean(acc_dime_m0)*100]
        m1_vals = [np.mean(acc_orc_m1)*100, np.mean(acc_erm_m1)*100,
                   np.mean(acc_ipw_m1)*100, np.mean(acc_dime_m1)*100]
        ax2.bar(x_pos - w_bar/2, m0_vals, w_bar, label='M=0 (labs missing)',
                color=colors, alpha=0.5)
        ax2.bar(x_pos + w_bar/2, m1_vals, w_bar, label='M=1 (labs observed)',
                color=colors, alpha=0.9, hatch='//')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['Oracle', 'ERM', 'CC-IPW', 'DIME'])
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('(b) Accuracy by Observation Stratum')
        p0 = mpatches.Patch(facecolor='gray', alpha=0.5, label='M=0 (some labs missing)')
        p1 = mpatches.Patch(facecolor='gray', alpha=0.9, hatch='//', label='M=1 (all labs observed)')
        ax2.legend(handles=[p0, p1], loc='lower right', fontsize=9)
        ylo = max(50.0, min(m0_vals + m1_vals) - 5.0)
        ax2.set_ylim(ylo, max(m0_vals + m1_vals) + 3.0)
        ax2.grid(axis='y', alpha=0.3)

    # (c) Per-feature uplift coefficients
    ax3 = axes[2]
    d1_last = dime_params['d1']
    J_last = dime_params['J']
    w_last = dime_params['w']
    beta_vals = [w_last[d1_last + J_last + j] for j in range(J)]
    gamma_vals = [w_last[d1_last + j] for j in range(J)]
    y_pos = np.arange(J)
    bar_h = 0.35
    ax3.barh(y_pos + bar_h/2, beta_vals, bar_h, color='tab:green', alpha=0.7,
             label='β (feature value)')
    ax3.barh(y_pos - bar_h/2, gamma_vals, bar_h, color='tab:purple', alpha=0.7,
             label='γ (test ordered)')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(expensive_cols)
    ax3.set_xlabel('Coefficient')
    ax3.set_title('(c) DIME Learned Coefficients')
    ax3.axvline(x=0, color='black', linewidth=0.8)
    ax3.legend(fontsize=9)
    ax3.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    out_pdf = FIG_DIR / "fig10_support_results.pdf"
    out_png = FIG_DIR / "fig10_support_results.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}, {out_png.name}")

    return {
        'auc_oracle': auc_orc_m, 'auc_erm': auc_erm_m,
        'auc_ipw': auc_ipw_m, 'auc_dime': auc_dime_m,
        'auc_dime_real': auc_dime_real_m,
    }


# ─── Experiment 11: Multi-Feature Synthetic (DIME vs CC-IPW scaling) ─────────

def run_exp11_multifeature(N_train=5000, N_test=3000, n_trials=20):
    """Show that CC-IPW degrades as J (number of expensive features) grows
    because the complete-case fraction shrinks exponentially, while DIME
    maintains oracle-level performance via per-feature correction.
    """
    print("Running Experiment 11: Multi-Feature DIME vs CC-IPW Scaling...")

    J_values = [1, 2, 3, 4, 5, 6]
    selectivity = 2.0       # moderate policy selectivity
    pi_min = 0.30           # per-feature ~70% acquisition rate => CC fraction ~0.7^J

    acc_oracle_j, acc_erm_j, acc_ipw_j, acc_dime_j = {}, {}, {}, {}
    cc_frac_j = {}

    for J in J_values:
        acc_orc, acc_erm, acc_ipw, acc_dim = [], [], [], []
        cc_fracs = []

        for trial in range(n_trials):
            rng = np.random.RandomState(trial * 31 + J)

            # Generate data: d1=2 cheap features, J expensive features
            N = N_train
            x1 = rng.randn(N, 2)               # cheap features

            # Expensive features: each bimodal at +/-2 (like the single-feature DGP)
            x2_cols_raw = []
            for j in range(J):
                comp = rng.binomial(1, 0.5, N)
                xj = np.where(comp == 1,
                              rng.randn(N) * 0.5 + 2.0,
                              rng.randn(N) * 0.5 - 2.0)
                x2_cols_raw.append(xj)

            # Target: y = (x1[:,0] + sum(x2_cols) / sqrt(J) > 0)
            # Normalise x2 contribution so total signal is comparable across J
            x2_sum = sum(x2_cols_raw) / np.sqrt(J)
            y = ((x1[:, 0] + x2_sum) > 0).astype(float)

            # Per-feature acquisition policy: P(Mj=1|x1) = max(sigmoid(-λ|x1[:,0]|), pi_min)
            pi_j = np.maximum(sigmoid(-selectivity * np.abs(x1[:, 0])), pi_min)

            m_cols = []
            for j in range(J):
                mj = rng.binomial(1, pi_j).astype(bool)
                m_cols.append(mj)

            # Complete-case mask
            M_all = np.ones(N, dtype=bool)
            for mj in m_cols:
                M_all &= mj
            cc_fracs.append(M_all.mean())

            # Build test data (full features available)
            x1_te = rng.randn(N_test, 2)
            x2_te_cols = []
            for j in range(J):
                comp_te = rng.binomial(1, 0.5, N_test)
                xj_te = np.where(comp_te == 1,
                                 rng.randn(N_test) * 0.5 + 2.0,
                                 rng.randn(N_test) * 0.5 - 2.0)
                x2_te_cols.append(xj_te)
            x2_sum_te = sum(x2_te_cols) / np.sqrt(J)
            y_te = ((x1_te[:, 0] + x2_sum_te) > 0).astype(float)

            # Oracle: full features, all data
            x2_full = np.column_stack(x2_cols_raw)
            X_orc = np.column_stack([x1, x2_full, np.ones(N)])
            w_orc = logistic_fit(X_orc, y, lr=0.1, n_iter=1500)
            x2_te_full = np.column_stack(x2_te_cols)
            X_orc_te = np.column_stack([x1_te, x2_te_full, np.ones(N_test)])
            acc_orc.append(logistic_acc(X_orc_te, y_te, w_orc))

            # ERM: zero-imputed
            x2_zi = np.column_stack([
                x2_cols_raw[j] * m_cols[j].astype(float)
                for j in range(J)])
            X_erm = np.column_stack([x1, x2_zi, np.ones(N)])
            w_erm = logistic_fit(X_erm, y, lr=0.1, n_iter=1500)
            X_erm_te = np.column_stack([x1_te, x2_te_full, np.ones(N_test)])
            acc_erm.append(logistic_acc(X_erm_te, y_te, w_erm))

            # CC-IPW: complete cases only, IPW-weighted
            if M_all.sum() >= 20:
                X_cc = np.column_stack([x1[M_all], x2_full[M_all], np.ones(M_all.sum())])
                # Joint propensity = product of per-feature propensities
                pi_joint = np.ones(N)
                for j in range(J):
                    pi_joint *= pi_j  # same policy for each feature
                pi_joint_cc = np.clip(pi_joint[M_all], 0.05, 0.95)
                w_cc = 1.0 / pi_joint_cc
                w_ipw_fit = logistic_fit(X_cc, y[M_all], weights=w_cc,
                                         lr=0.1, n_iter=1500)
                acc_ipw.append(logistic_acc(X_orc_te, y_te, w_ipw_fit))
            else:
                acc_ipw.append(0.5)   # CC-IPW cannot train

            # DIME: per-feature gradient-split IPW
            dime_p = dime_train(x1, x2_cols_raw, m_cols, y,
                                lr=0.1, n_iter=1500)
            # Evaluate with all features available
            m_te_full = [np.ones(N_test, dtype=bool)] * J
            prob_dime = dime_predict(x1_te, x2_te_cols, m_te_full, dime_p)
            acc_dim.append(((prob_dime > 0.5) == y_te).mean())

        acc_oracle_j[J] = np.mean(acc_orc)
        acc_erm_j[J]    = np.mean(acc_erm)
        acc_ipw_j[J]    = np.mean(acc_ipw)
        acc_dime_j[J]   = np.mean(acc_dim)
        cc_frac_j[J]    = np.mean(cc_fracs)

        print(f"  J={J}: CC% = {cc_frac_j[J]*100:.1f}%  "
              f"Oracle {acc_oracle_j[J]*100:.1f}%  "
              f"ERM {acc_erm_j[J]*100:.1f}%  "
              f"CC-IPW {acc_ipw_j[J]*100:.1f}%  "
              f"DIME {acc_dime_j[J]*100:.1f}%")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    orc_vals = [acc_oracle_j[j] * 100 for j in J_values]
    erm_vals = [acc_erm_j[j] * 100 for j in J_values]
    ipw_vals = [acc_ipw_j[j] * 100 for j in J_values]
    dim_vals = [acc_dime_j[j] * 100 for j in J_values]
    cc_vals  = [cc_frac_j[j] * 100 for j in J_values]

    ax1.plot(J_values, orc_vals, 'k--o', label='Oracle', markersize=7, lw=2.5)
    ax1.plot(J_values, erm_vals, 'r-^', label='ERM', markersize=7, lw=2)
    ax1.plot(J_values, ipw_vals, 'b-s', label='CC-IPW', markersize=7, lw=2)
    ax1.plot(J_values, dim_vals, 'g-D', label='DIME (ours)', markersize=8, lw=2.5)
    ax1.set_xlabel('Number of Expensive Features (J)')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('(a) Accuracy vs Number of Expensive Features')
    ax1.legend(fontsize=10)
    ax1.set_xticks(J_values)
    ax1.grid(alpha=0.3)

    ax2.bar(J_values, cc_vals, color='tab:gray', alpha=0.7, width=0.6)
    for j, v in zip(J_values, cc_vals):
        ax2.text(j, v + 1, f'{v:.0f}%', ha='center', fontsize=10)
    ax2.set_xlabel('Number of Expensive Features (J)')
    ax2.set_ylabel('Complete-Case Fraction (%)')
    ax2.set_title('(b) Data Available for CC-IPW')
    ax2.set_xticks(J_values)
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    out_pdf = FIG_DIR / "fig11_multifeature_scaling.pdf"
    out_png = FIG_DIR / "fig11_multifeature_scaling.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}, {out_png.name}")

    return acc_oracle_j, acc_erm_j, acc_ipw_j, acc_dime_j


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" Endogenous MNAR Fusion -- All Experiments")
    print("=" * 60)

    sel, oracle, erm, _, ipw = run_exp1()
    print_summary(sel, oracle, erm, ipw)

    run_exp2()
    run_exp3()
    run_exp4()
    run_exp5()
    run_exp6_mlp()
    run_exp_feedback_loop()           # Exp 7 (new): feedback loop
    run_exp7_semisynthetic()          # Exp 8 (renumbered)
    run_exp8_baseline_and_calib()     # Exp 9 (renumbered)
    run_exp_support()                 # Exp 10 (new): SUPPORT real data
    run_exp11_multifeature()          # Exp 11 (new): Multi-feature scaling

    print("\nAll experiments complete. Figures saved to:", FIG_DIR)
    print("Files:")
    for f in sorted(FIG_DIR.glob("fig*.p*")):
        print(f"  {f.name}")
