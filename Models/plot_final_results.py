# =============================================================================
# plot_final_results.py
# Diagnostic plots for the final tuned models (RF and XGBoost).
# =============================================================================

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import main as run_preprocessing
from parameter_tuning import deep_tune_model
from models import get_rf_model, get_xgb_model
from metrics_evaluating import calculate_metrics
import config

warnings.filterwarnings("ignore")

# Optional: SHAP check
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── Output folder ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("figures_final")
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_HEXBIN   = 50_000
SAMPLE_RESIDUAL = 50_000
SAMPLE_SHAP     = 2_000 # SHAP is very slow; 2k is enough for a beeswarm

# ── Plot style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 140, "axes.titleweight": "bold"})

# ── Helpers ────────────────────────────────────────────────────────────────────

def save_close(fig, filename: str):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")

def sample_xy(X, y, n, seed=42):
    """Safely samples X and y together to avoid duplicate index issues."""
    if len(X) <= n:
        return X.copy(), y.copy()
    # We use the same random_state on both to keep them aligned
    X_s = X.sample(n=n, random_state=seed)
    y_s = y.sample(n=n, random_state=seed)
    return X_s, y_s

def get_feature_names():
    """Matches the order produced by model_utiles.build_pipeline()."""
    names = []
    names += config.LOG_FEATURES
    for f in config.CYCLIC_FEATURES:
        names += [f"{f}_sin", f"{f}_cos"]
    names += config.PASSTHROUGH_FEATURES
    return names

# ── Plotting functions (Comparison, Hexbin, Residuals) ─────────────────────────

def plot_model_comparison(results):
    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R²"]):
        sns.barplot(data=df, x="Model", y=metric, ax=ax, palette="viridis")
        ax.set_title(metric)
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    save_close(fig, "01_model_comparison.png")

def plot_actual_vs_pred_hexbin(y_true, y_pred, model_name: str, filename: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(y_true, y_pred, gridsize=60, mincnt=1, cmap="YlGnBu")
    fig.colorbar(hb, ax=ax, label="Density (Count)")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", alpha=0.5)
    ax.set_title(f"{model_name}: Actual vs Predicted")
    ax.set_xlabel("Actual Fare ($)"); ax.set_ylabel("Predicted Fare ($)")
    save_close(fig, filename)

def plot_residual_diagnostics(y_true, y_pred, model_name: str, filename: str):
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(residuals, bins=80, ax=axes[0], kde=True)
    axes[0].set_title("Residual Distribution")
    axes[1].hexbin(y_pred, residuals, gridsize=50, mincnt=1, cmap="magma")
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title("Residuals vs Predicted")
    save_close(fig, filename)

# ── Extended Accuracy Analysis (Your new sub-questions) ───────────────────────

def plot_most_and_least_accurate(X_test, y_test, y_pred, model_name: str, top_n=15):
    df = X_test.copy()
    df["y_true"], df["y_pred"] = y_test.values, y_pred
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])
    df["residual"] = df["y_true"] - df["y_pred"]
    
    prefix = model_name.lower().replace(" ", "_")
    worst = df.nlargest(top_n, "abs_error").sort_values("abs_error")
    
    plt.figure(figsize=(10, 7))
    colors = ["tomato" if r > 0 else "steelblue" for r in worst["residual"]]
    labels = [f"Dist:{d:.1f} | Hr:{h} | ID:{z}" for d, h, z in zip(worst["trip_distance"], worst["hour"], worst["PULocationID"])]
    plt.barh(labels, worst["abs_error"], color=colors)
    plt.title(f"{model_name}: Top {top_n} Worst Predictions\n(Red=Underpriced, Blue=Overpriced)")
    plt.xlabel("Absolute Error ($)")
    plt.savefig(OUTPUT_DIR / f"{prefix}_worst_trips.png", bbox_inches="tight")
    plt.close()

def plot_error_heatmap(X_test, y_test, y_pred, model_name: str):
    df = X_test.copy()
    df["MAE"] = np.abs(y_test.values - y_pred)
    
    # Combined heatmap: Distance Bins vs Hour
    bins = [0, 1, 2, 5, 10, 20, 50]
    labels = ["0-1", "1-2", "2-5", "5-10", "10-20", "20+"]
    df["dist_bin"] = pd.cut(df["trip_distance"], bins=bins, labels=labels)
    
    heat = df.pivot_table(index="dist_bin", columns="hour", values="MAE", aggfunc="mean")
    plt.figure(figsize=(12, 6))
    sns.heatmap(heat, cmap="YlOrRd", annot=False)
    plt.title(f"{model_name}: MAE by Distance & Hour")
    plt.savefig(OUTPUT_DIR / f"{model_name.lower()}_heatmap.png", bbox_inches="tight")
    plt.close()

def plot_shap_analysis(pipeline, X_test, prefix):
    if not SHAP_AVAILABLE: return
    print(f"  Computing SHAP for {prefix}...")
    X_s, _ = sample_xy(X_test, pd.Series(dtype=float), SAMPLE_SHAP)
    
    preprocessor = pipeline[:-1]
    model = pipeline.named_steps["model"]
    X_trans = preprocessor.transform(X_s)
    names = get_feature_names()
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_trans, feature_names=names, show=False)
    plt.savefig(OUTPUT_DIR / f"{prefix}_shap.png", bbox_inches="tight")
    plt.close()

# ── Main Execution ─────────────────────────────────────────────────────────────

def main():
    print("=== STEP 1: Loading Data ===")
    X_train, X_test, y_train, y_test, kfold, *rest = run_preprocessing()

    # GRIDS MUST MATCH final_result.py exactly to load from disk
    deep_rf_grid = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    }
    deep_xgb_grid = {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0],
    }

    print("\n=== STEP 2: Loading Models (Joblib) ===")
    # These calls will automatically detect the .joblib files and NOT retrain
    rf_pipe = deep_tune_model("RandomForest", get_rf_model(), deep_rf_grid, X_train, y_train, kfold)
    xgb_pipe = deep_tune_model("XGBoost", get_xgb_model(), deep_xgb_grid, X_train, y_train, kfold)

    all_metrics = []
    for name, pipe, prefix in [("Random Forest", rf_pipe, "rf"), ("XGBoost", xgb_pipe, "xgb")]:
        print(f"\n=== STEP 3: Diagnostics for {name} ===")
        y_pred = pipe.predict(X_test)
        all_metrics.append(calculate_metrics(y_test, y_pred, model_name=name))
        
        # Plots
        X_h, y_h = sample_xy(X_test, y_test, SAMPLE_HEXBIN)
        plot_actual_vs_pred_hexbin(y_h, pipe.predict(X_h), name, f"{prefix}_actual_vs_pred.png")
        plot_residual_diagnostics(y_h, pipe.predict(X_h), name, f"{prefix}_residuals.png")
        plot_most_and_least_accurate(X_test, y_test, y_pred, name)
        plot_error_heatmap(X_test, y_test, y_pred, name)
        
        if prefix == "xgb": # SHAP only for XGBoost to save time
            plot_shap_analysis(pipe, X_test, prefix)

    plot_model_comparison(all_metrics)
    print(f"\nDone! All plots saved to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()