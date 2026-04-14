# =============================================================================
# plot_final_results.py
#
# PURPOSE:
# This script is dedicated to evaluating the final tuned models visually. 
# While raw metrics (like MAE or R²) are good for a quick summary, they hide 
# the underlying behavior of the model. 
#
# This script generates:
# 1. Model Comparisons: Bar charts comparing overall metrics.
# 2. Hexbin Plots: To visualize density and check for general correlation.
# 3. Residual Diagnostics: To see if the model consistently overprices or 
#    underprices specific types of trips (e.g., via the new fare-bin boxplot).
# 4. Error Heatmaps: To identify specific combinations (like long distances 
#    during rush hour) where the model struggles.
# 5. SHAP Values: To interpret which features drive the XGBoost predictions.
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

# Optional: SHAP check for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── Output folder setup ────────────────────────────────────────────────────────
OUTPUT_DIR = Path("figures_final")
OUTPUT_DIR.mkdir(exist_ok=True)

# Sampling limits to keep plotting fast and readable
SAMPLE_HEXBIN   = 50_000
SAMPLE_RESIDUAL = 50_000
SAMPLE_SHAP     = 2_000 # SHAP is computationally expensive

# ── Plot style configuration ───────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 140, "axes.titleweight": "bold"})

# ── Helper Functions ───────────────────────────────────────────────────────────

def save_close(fig, filename: str):
    """Saves the figure to the output directory and closes it to free memory."""
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")

def sample_xy(X, y, n, seed=42):
    """Safely samples X and y together to avoid duplicate index issues."""
    if len(X) <= n:
        return X.copy(), y.copy()
    X_s = X.sample(n=n, random_state=seed)
    y_s = y.sample(n=n, random_state=seed)
    return X_s, y_s

def get_feature_names():
    """Retrieves the ordered list of features produced by the ColumnTransformer pipeline."""
    names = []
    names += config.LOG_FEATURES
    for f in config.CYCLIC_FEATURES:
        names += [f"{f}_sin", f"{f}_cos"]
    names += config.PASSTHROUGH_FEATURES
    return names

# ── Standard Plotting Functions ────────────────────────────────────────────────

def plot_model_comparison(results):
    """Plots a bar chart comparing MAE, RMSE, and R² across different models."""
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
    """Creates a hexbin plot to show the density of predictions vs actual values."""
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(y_true, y_pred, gridsize=60, mincnt=1, cmap="YlGnBu")
    fig.colorbar(hb, ax=ax, label="Density (Count)")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", alpha=0.5)
    ax.set_title(f"{model_name}: Actual vs Predicted")
    ax.set_xlabel("Actual Fare ($)")
    ax.set_ylabel("Predicted Fare ($)")
    save_close(fig, filename)

def plot_residual_diagnostics(y_true, y_pred, model_name: str, filename: str):
    """Plots the overall distribution of residuals and a scatter of residuals vs predicted."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.histplot(residuals, bins=80, ax=axes[0], kde=True)
    axes[0].set_title("Residual Distribution")
    
    axes[1].hexbin(y_pred, residuals, gridsize=50, mincnt=1, cmap="magma")
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title("Residuals vs Predicted")
    
    save_close(fig, filename)

def plot_residual_by_fare_bin(y_true, y_pred, model_name: str, filename: str):
    """
    Plots a boxplot of residuals grouped by fare bins (L/M/H/P) to diagnose 
    if the model systematically struggles with cheap or expensive trips.
    """
    residuals = np.array(y_true) - np.array(y_pred)
    df = pd.DataFrame({'y_true': y_true, 'residual': residuals})
    
    # Bins matching the preprocessing balancing logic
    bins = [0, 15, 25, 40, 51]
    labels = ['L ($0-15)', 'M ($15-25)', 'H ($25-40)', 'P ($40-51)']
    df['fare_bin'] = pd.cut(df['y_true'], bins=bins, labels=labels)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='fare_bin', y='residual', palette='coolwarm', showfliers=False, ax=ax)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label="Zero Error")
    
    ax.set_title(f"{model_name}: Residuals per Fare Bin\n(Below 0 = Overpricing | Above 0 = Underpricing)")
    ax.set_xlabel("Actual Fare Category")
    ax.set_ylabel("Residual (Actual - Predicted) [$]")
    ax.legend()
    save_close(fig, filename)

# ── Advanced Analytical Plots ──────────────────────────────────────────────────

def plot_most_and_least_accurate(X_test, y_test, y_pred, model_name: str, top_n=15):
    """Identifies and plots the top N trips with the highest absolute error."""
    df = X_test.copy()
    df["y_true"], df["y_pred"] = y_test.values, y_pred
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])
    df["residual"] = df["y_true"] - df["y_pred"]
    
    prefix = model_name.lower().replace(" ", "_")
    worst = df.nlargest(top_n, "abs_error").sort_values("abs_error")
    
    plt.figure(figsize=(10, 7))
    colors = ["tomato" if r > 0 else "steelblue" for r in worst["residual"]]
    # Note: Replaced raw PULocationID with PU_fare_avg to reflect target encoding changes
    labels = [f"Dist:{d:.1f} | Hr:{h} | PU_Avg:${z:.1f}" for d, h, z in zip(worst["trip_distance"], worst["hour"], worst["PU_fare_avg"])]
    plt.barh(labels, worst["abs_error"], color=colors)
    plt.title(f"{model_name}: Top {top_n} Worst Predictions\n(Red=Underpriced, Blue=Overpriced)")
    plt.xlabel("Absolute Error ($)")
    plt.savefig(OUTPUT_DIR / f"{prefix}_worst_trips.png", bbox_inches="tight")
    plt.close()

def plot_error_heatmap(X_test, y_test, y_pred, model_name: str):
    """Creates a heatmap showing average error grouped by distance bin and hour of day."""
    df = X_test.copy()
    df["MAE"] = np.abs(y_test.values - y_pred)
    
    # Define distance buckets
    bins = [0, 1, 2, 5, 10, 20, 50]
    labels = ["0-1", "1-2", "2-5", "5-10", "10-20", "20+"]
    df["dist_bin"] = pd.cut(df["trip_distance"], bins=bins, labels=labels)
    
    heat = df.pivot_table(index="dist_bin", columns="hour", values="MAE", aggfunc="mean")
    plt.figure(figsize=(12, 6))
    sns.heatmap(heat, cmap="YlOrRd", annot=False)
    plt.title(f"{model_name}: MAE by Distance & Hour")
    prefix = model_name.lower().replace(" ", "_")
    plt.savefig(OUTPUT_DIR / f"{prefix}_heatmap.png", bbox_inches="tight")
    plt.close()

def plot_shap_analysis(pipeline, X_test, prefix):
    """Calculates and plots SHAP values to show global feature importance and direction."""
    if not SHAP_AVAILABLE: 
        print("  [SHAP skipped] Install 'shap' library to view feature importance.")
        return
        
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

    # Hyperparameter grids must match final_result.py exactly to load correctly from disk
    deep_rf_grid = config.DEEP_RF_GRID
    deep_xgb_grid = config.DEEP_XGB_GRID

    print("\n=== STEP 2: Loading Models (Joblib) ===")
    rf_pipe = deep_tune_model("RandomForest", get_rf_model(), deep_rf_grid, X_train, y_train, kfold)
    xgb_pipe = deep_tune_model("XGBoost", get_xgb_model(), deep_xgb_grid, X_train, y_train, kfold)

    all_metrics = []
    
    # Evaluate and plot for both models
    for name, pipe, prefix in [("Random Forest", rf_pipe, "rf"), ("XGBoost", xgb_pipe, "xgb")]:
        print(f"\n=== STEP 3: Diagnostics for {name} ===")
        y_pred = pipe.predict(X_test)
        
        # Calculate text metrics
        all_metrics.append(calculate_metrics(y_test, y_pred, model_name=name))
        
        # Generate diagnostic plots
        X_h, y_h = sample_xy(X_test, y_test, SAMPLE_HEXBIN)
        plot_actual_vs_pred_hexbin(y_h, pipe.predict(X_h), name, f"{prefix}_actual_vs_pred.png")
        plot_residual_diagnostics(y_h, pipe.predict(X_h), name, f"{prefix}_residuals.png")
        plot_residual_by_fare_bin(y_test, y_pred, name, f"{prefix}_residual_bins.png")
        
        # Generate advanced analytical plots
        plot_most_and_least_accurate(X_test, y_test, y_pred, name)
        plot_error_heatmap(X_test, y_test, y_pred, name)
        
        # SHAP analysis is generally only run on XGBoost to save execution time
        if prefix == "xgb": 
            plot_shap_analysis(pipe, X_test, prefix)

    # Final comparison bar chart
    plot_model_comparison(all_metrics)
    print(f"\nDone! All plots saved to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()