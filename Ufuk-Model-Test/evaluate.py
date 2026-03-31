"""
evaluate.py
-----------
Loads saved models and generates all evaluation outputs.
Run this after train.py has completed — no retraining happens here.

HOW TO RUN
----------
    python evaluate.py

Requires (all produced by train.py)
-------------------------------------
    preprocessor.pkl
    test_data.pkl
    cv_results.pkl
    linear_regression.pkl
    random_forest.pkl
    xgboost.pkl

Outputs saved to Ufuk-Model-Test/
-----------------------------------
    cv_summary.png
    residual_diagnostics.png
    shap_bar_<model>.png
    shap_beeswarm_<model>.png
    metrics_summary.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model_utils import (
    OUTPUT_DIR, SPLIT_RANDOM_STATE, N_FOLDS,
    PALETTE, compute_metrics, style_ax,
)


# =============================================================================
# LOAD HELPERS
# =============================================================================

def _load(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {path}\n"
            f"  -> Run train.py first to generate this file."
        )
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_all():
    """Loads preprocessor, test data, CV results, and all fitted models."""
    print("Loading saved files...")

    pre       = _load('preprocessor.pkl')
    test_data = _load('test_data.pkl')
    cv_results = _load('cv_results.pkl')

    col_names = pre['col_names']
    X_test_t  = test_data['X_test_t']
    y_test    = test_data['y_test']

    model_files = {
        'Linear Regression': 'linear_regression.pkl',
        'Random Forest':     'random_forest.pkl',
        'XGBoost':           'xgboost.pkl',
    }
    fitted_models = {}
    for name, fname in model_files.items():
        fitted_models[name] = _load(fname)
        print(f"  Loaded: {fname}")

    print(f"  Test rows  : {len(y_test):,}")
    print(f"  Features   : {col_names}")

    return fitted_models, X_test_t, y_test, col_names, cv_results


# =============================================================================
# 1. TEST-SET METRICS
# =============================================================================

def evaluate_all(fitted_models, X_test_t, y_test):
    """Runs predictions and prints MAE/RMSE/R2 for every model."""
    print("\n" + "=" * 60)
    print("  TEST-SET METRICS")
    print("=" * 60)

    predictions  = {}
    summary_rows = []

    for name, model in fitted_models.items():
        y_pred = model.predict(X_test_t)
        predictions[name] = y_pred
        metrics = compute_metrics(y_test, y_pred, label=name)
        metrics['Model'] = name
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)[['Model', 'MAE', 'RMSE', 'R2']]
    csv_path   = os.path.join(OUTPUT_DIR, 'metrics_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Saved -> {csv_path}")

    return predictions, summary_df


# =============================================================================
# 2. CV SUMMARY PLOT
# =============================================================================

def plot_cv_summary(cv_results):
    """Bar chart of mean CV MAE and R2 with +/- 1 std error bars."""
    print("\n" + "=" * 60)
    print("  CV SUMMARY PLOT")
    print("=" * 60)

    names     = list(cv_results.keys())
    mae_means = [cv_results[n]['MAE'].mean() for n in names]
    mae_stds  = [cv_results[n]['MAE'].std()  for n in names]
    r2_means  = [cv_results[n]['R2'].mean()  for n in names]
    r2_stds   = [cv_results[n]['R2'].std()   for n in names]
    colors    = [PALETTE[n] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(names, mae_means, yerr=mae_stds, color=colors,
            capsize=6, edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('Mean Absolute Error ($)', fontsize=10)
    ax1.set_title(f'CV MAE  (mean +/- std,  {N_FOLDS} folds)', fontsize=11)
    style_ax(ax1)

    ax2.bar(names, r2_means, yerr=r2_stds, color=colors,
            capsize=6, edgecolor='white', linewidth=0.5)
    ax2.set_ylabel('R2', fontsize=10)
    ax2.set_title(f'CV R2  (mean +/- std,  {N_FOLDS} folds)', fontsize=11)
    ax2.set_ylim(0, 1)
    style_ax(ax2)

    fig.suptitle('Cross-Validation Summary', fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'cv_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {path}")


# =============================================================================
# 3. SHAP ANALYSIS
# =============================================================================

def plot_shap_all(fitted_models, X_test_t, col_names, sample_n=3000):
    """
    SHAP bar + beeswarm for every model.

    Explainer choice
    ----------------
    LinearExplainer  -> Linear Regression  (exact, uses feature covariance)
    TreeExplainer    -> Random Forest and XGBoost  (exact, uses tree structure)

    sample_n rows from the test set keep runtime manageable.
    3000 rows gives a reliable picture of global feature importance.
    """
    print("\n" + "=" * 60)
    print("  SHAP ANALYSIS")
    print("=" * 60)

    rng = np.random.default_rng(SPLIT_RANDOM_STATE)
    idx = rng.choice(len(X_test_t),
                     size=min(sample_n, len(X_test_t)),
                     replace=False)
    X_sample = X_test_t[idx]

    for name, model in fitted_models.items():
        print(f"\n  Computing SHAP for {name}...")

        if name == 'Linear Regression':
            explainer   = shap.LinearExplainer(model, X_test_t)
            shap_values = explainer(X_sample)
        else:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer(X_sample, check_additivity=False)

        shap_values.feature_names = col_names

        # Bar plot — mean |SHAP| per feature
        fig_bar, ax = plt.subplots(figsize=(8, 5))
        shap.plots.bar(shap_values, max_display=len(col_names), ax=ax, show=False)
        ax.set_title(f'SHAP Feature Importance — {name}', fontsize=13, pad=10)
        ax.set_xlabel('Mean |SHAP value|  (avg impact on fare in $)', fontsize=9)
        style_ax(ax)
        fig_bar.tight_layout()
        bar_path = os.path.join(OUTPUT_DIR,
                                f'shap_bar_{name.lower().replace(" ", "_")}.png')
        fig_bar.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.close(fig_bar)
        print(f"    Saved -> {bar_path}")

        # Beeswarm — direction and spread of each feature's effect
        fig_bee = plt.figure(figsize=(9, 6))
        shap.plots.beeswarm(shap_values, max_display=len(col_names), show=False)
        plt.title(f'SHAP Beeswarm — {name}', fontsize=13, pad=10)
        fig_bee.tight_layout()
        bee_path = os.path.join(OUTPUT_DIR,
                                f'shap_beeswarm_{name.lower().replace(" ", "_")}.png')
        fig_bee.savefig(bee_path, dpi=150, bbox_inches='tight')
        plt.close(fig_bee)
        print(f"    Saved -> {bee_path}")


# =============================================================================
# 4. RESIDUAL DIAGNOSTICS
# =============================================================================

def plot_residuals(y_test, predictions, sample_n=5000):
    """
    Three-row diagnostic figure — one column per model.

    Row 1  Predicted vs Actual     — should hug the diagonal y = x
    Row 2  Residuals vs Predicted  — should be randomly scattered around zero
    Row 3  Residual distribution   — should be symmetric around zero
    """
    print("\n" + "=" * 60)
    print("  RESIDUAL DIAGNOSTICS")
    print("=" * 60)

    rng   = np.random.default_rng(SPLIT_RANDOM_STATE)
    idx   = rng.choice(len(y_test), size=min(sample_n, len(y_test)), replace=False)
    y_s   = y_test[idx]

    names    = list(predictions.keys())
    n_models = len(names)

    fig = plt.figure(figsize=(6 * n_models, 14))
    gs  = gridspec.GridSpec(3, n_models, figure=fig, hspace=0.45, wspace=0.35)

    for col, name in enumerate(names):
        y_pred = predictions[name][idx]
        resid  = y_s - y_pred
        color  = PALETTE[name]

        # Row 1 — Predicted vs Actual
        ax1 = fig.add_subplot(gs[0, col])
        ax1.scatter(y_s, y_pred, alpha=0.15, s=6, color=color, rasterized=True)
        lim = (min(y_s.min(), y_pred.min()) - 1,
               max(y_s.max(), y_pred.max()) + 1)
        ax1.plot(lim, lim, 'k--', lw=1, label='Perfect fit')
        ax1.set_xlim(lim); ax1.set_ylim(lim)
        ax1.set_xlabel('Actual fare ($)', fontsize=9)
        ax1.set_ylabel('Predicted fare ($)', fontsize=9)
        ax1.set_title(f'{name}\nPredicted vs Actual', fontsize=10, fontweight='bold')
        ax1.legend(fontsize=8)
        style_ax(ax1)

        # Row 2 — Residuals vs Predicted
        ax2 = fig.add_subplot(gs[1, col])
        ax2.scatter(y_pred, resid, alpha=0.15, s=6, color=color, rasterized=True)
        ax2.axhline(0, color='black', lw=1, ls='--')
        ax2.set_xlabel('Predicted fare ($)', fontsize=9)
        ax2.set_ylabel('Residual  (Actual - Predicted)', fontsize=9)
        ax2.set_title('Residuals vs Predicted', fontsize=10)
        style_ax(ax2)

        # Row 3 — Residual distribution
        ax3 = fig.add_subplot(gs[2, col])
        ax3.hist(resid, bins=80, color=color, edgecolor='white', linewidth=0.3)
        ax3.axvline(0,            color='black', lw=1.2, ls='--', label='Zero')
        ax3.axvline(resid.mean(), color='red',   lw=1.2, ls='-',
                    label=f'Mean = {resid.mean():.2f}')
        ax3.set_xlabel('Residual ($)', fontsize=9)
        ax3.set_ylabel('Count', fontsize=9)
        ax3.set_title('Residual Distribution', fontsize=10)
        ax3.legend(fontsize=8)
        style_ax(ax3)

    fig.suptitle('Residual Diagnostics — All Models', fontsize=14,
                 fontweight='bold', y=1.01)
    path = os.path.join(OUTPUT_DIR, 'residual_diagnostics.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    fitted_models, X_test_t, y_test, col_names, cv_results = load_all()

    predictions, summary_df = evaluate_all(fitted_models, X_test_t, y_test)

    plot_cv_summary(cv_results)

    plot_shap_all(fitted_models, X_test_t, col_names)

    plot_residuals(y_test, predictions)

    print("\n" + "=" * 60)
    print("  FINAL COMPARISON TABLE")
    print("=" * 60)
    print(summary_df.to_string(index=False, float_format='{:.4f}'.format))

    print("\nDone. All outputs saved to:")
    print(f"  {OUTPUT_DIR}")