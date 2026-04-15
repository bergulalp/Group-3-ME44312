# =============================================================================
# test_seasonality.py
# -----------------------------------------------------------------------------
# Evaluates how well models trained on January 2025 data generalize to
# unseen July 2025 data. This script tests for concept drift / seasonality
# effects by applying January-derived feature mappings to July trips and
# comparing RF and XGBoost performance side by side.
# =============================================================================

import os, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

import config
from preprocessing import load_and_clean
from metrics_evaluating import calculate_metrics

warnings.filterwarnings("ignore")


def apply_january_mappings(df_jul, df_jan):
    """
    Applies all feature engineering to July data using statistics derived
    exclusively from January training data. This prevents data leakage and
    ensures a fair generalization test.
    """

    # Target Encoding: historical average fare per pickup/dropoff zone
    pu_map = df_jan.groupby('PULocationID')['fare_amount'].mean().to_dict()
    do_map = df_jan.groupby('DOLocationID')['fare_amount'].mean().to_dict()
    global_mean = df_jan['fare_amount'].mean()
    df_jul['PU_fare_avg'] = df_jul['PULocationID'].map(pu_map).fillna(global_mean)
    df_jul['DO_fare_avg'] = df_jul['DOLocationID'].map(do_map).fillna(global_mean)

    # Spatial Clustering: group zones by avg fare and distance
    zone_stats = df_jan.groupby('PULocationID').agg(
        {'fare_amount': 'mean', 'trip_distance': 'mean'}
    ).reset_index()
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    zone_stats['pickup_cluster'] = kmeans.fit_predict(
        zone_stats[['fare_amount', 'trip_distance']]
    )
    cluster_map = zone_stats.set_index('PULocationID')['pickup_cluster'].to_dict()

    # Unseen zones in July get cluster -1 as fallback
    df_jul['pickup_cluster'] = df_jul['PULocationID'].map(cluster_map).fillna(-1)

    # Route Features: median speed and popularity per route/hour
    duration_h = (
        df_jan['tpep_dropoff_datetime'] - df_jan['tpep_pickup_datetime']
    ).dt.total_seconds() / 3600
    df_jan['speed_mph'] = (
        df_jan['trip_distance'] / duration_h.replace(0, np.nan)
    ).clip(upper=100)

    route_stats = df_jan.groupby(['PULocationID', 'DOLocationID', 'hour']).agg(
        route_avg_speed=('speed_mph', 'median'),
        route_popularity=('speed_mph', 'count')
    ).reset_index()

    # Fallback: if full route is unknown, use zone+hour median; then global median
    backup_speed = df_jan.groupby(
        ['PULocationID', 'hour']
    )['speed_mph'].median().to_dict()
    global_speed = df_jan['speed_mph'].median()

    df_jul = df_jul.merge(
        route_stats, on=['PULocationID', 'DOLocationID', 'hour'], how='left'
    )
    mask = df_jul['route_avg_speed'].isna()
    df_jul.loc[mask, 'route_avg_speed'] = (
        df_jul.loc[mask]
        .set_index(['PULocationID', 'hour'])
        .index.map(backup_speed)
    )
    df_jul['route_avg_speed'] = df_jul['route_avg_speed'].fillna(global_speed)
    df_jul['route_popularity'] = df_jul['route_popularity'].fillna(0)

    return df_jul


def plot_generalization(y_true, pred_dict, output_dir="figures_seasonality"):
    """
    Generates two diagnostic plots for each model:
      1. Actual vs Predicted scatter — shows overall prediction quality
      2. Residual vs Actual — reveals where the largest errors occur
         (systematic over/underestimation at high fares is a red flag)

    Args:
        y_true (Series): True fare amounts from July test set.
        pred_dict (dict): {"Model Name": y_pred array} for each model.
        output_dir (str): Folder to save figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    colors = {"Random Forest": "#2196F3", "XGBoost": "#FF5722"}

    # Plot 1: Actual vs Predicted, one panel per model side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Generalization to July 2025: Actual vs Predicted Fare",
        fontsize=15, fontweight='bold'
    )

    for ax, (name, y_pred) in zip(axes, pred_dict.items()):
        ax.hexbin(y_true, y_pred, gridsize=60, mincnt=1, cmap="YlGnBu")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=1.5, label="Perfect prediction")
        ax.set_title(name, fontsize=13, color=colors[name], fontweight='bold')
        ax.set_xlabel("Actual Fare ($)")
        ax.set_ylabel("Predicted Fare ($)")
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=150)
    plt.close()
    print("[INFO] Saved: actual_vs_predicted.png")

    # Plot 2: Residuals vs Actual, where are the largest errors made?
    # A random scatter around 0 means good generalization.
    # An upward/downward trend means the model systematically under/overpredicts
    # for certain fare ranges, which is common for expensive trips.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Residuals vs Actual Fare: Where Are the Largest Errors?",
        fontsize=15, fontweight='bold'
    )

    for ax, (name, y_pred) in zip(axes, pred_dict.items()):
        residuals = np.array(y_pred) - np.array(y_true)
        ax.scatter(y_true, residuals, alpha=0.15, s=8, color=colors[name])
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
        # Smoothed trend line to highlight systematic bias
        smoothed = pd.Series(residuals).rolling(window=500, min_periods=1).mean()
        ax.plot(
            np.sort(y_true), smoothed.values,
            color='black', linewidth=1.2, label="Trend (rolling mean)"
        )
        ax.set_title(name, fontsize=13, color=colors[name], fontweight='bold')
        ax.set_xlabel("Actual Fare ($)")
        ax.set_ylabel("Residual (Predicted minus Actual) ($)")
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "residuals_vs_actual.png"), dpi=150)
    plt.close()
    print("[INFO] Saved: residuals_vs_actual.png")


def plot_error_heatmap(df_jul, pred_dict, output_dir="figures_seasonality"):
    """
    Plots a heatmap of mean absolute error (MAE) per hour of day and day of
    week for each model. Reveals whether errors cluster around rush hours or
    specific weekdays.

    Rows = hour of day (0-23), Columns = day of week (Mon-Sun)

    Args:
        df_jul (DataFrame): July data including hour and day of week columns.
        pred_dict (dict): {"Model Name": y_pred array} for each model.
        output_dir (str): Folder to save figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        "Mean Absolute Error by Hour and Day of Week (July 2025)\n"
        "Darker cells indicate larger average prediction error",
        fontsize=14, fontweight='bold'
    )

    for ax, (name, y_pred) in zip(axes, pred_dict.items()):
        # Compute absolute error per trip
        errors = pd.DataFrame({
            'abs_error':  np.abs(np.array(y_pred) - np.array(df_jul[config.TARGET])),
            'hour':       df_jul['hour'].values,
            'day_of_week': df_jul['day of week'].values
        })

        # Pivot to hour x day matrix of mean absolute error
        pivot = errors.pivot_table(
            values='abs_error',
            index='hour',
            columns='day_of_week',
            aggfunc='mean'
        )
        pivot.columns = [day_labels[d] for d in pivot.columns]

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            cbar_kws={'label': 'Mean Absolute Error ($)'}
        )
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Hour of Day")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "error_heatmap_hour_day.png"), dpi=150)
    plt.close()
    print("[INFO] Saved: error_heatmap_hour_day.png")


def print_comparison_table(january_metrics, july_metrics):
    """
    Prints a side-by-side comparison of January test metrics vs July
    generalization metrics. A large drop in R2 or rise in MAE indicates
    concept drift due to seasonal differences.

    Args:
        january_metrics (list): List of metric dicts saved by final_result.py.
        july_metrics (list): List of metric dicts computed on July data.
    """
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON: January (test set) vs July (generalization)")
    print("=" * 65)

    jan_df = pd.DataFrame(january_metrics).set_index('Model')
    jul_df = pd.DataFrame(july_metrics).set_index('Model')

    for model in jul_df.index:
        print(f"\n  {model}")
        print(f"  {'Metric':<20} {'January':>12} {'July':>12} {'Delta':>10}")
        print(f"  {'-' * 54}")
        for metric in ['MAE', 'RMSE', 'R2', 'Accuracy (%)']:
            # Use R2 as key since calculate_metrics stores it as R2
            metric_key = 'R²' if 'R' in metric else metric
            jan_val = jan_df.loc[model, metric_key] if model in jan_df.index else float('nan')
            jul_val = jul_df.loc[model, metric_key]
            delta = jul_val - jan_val
            print(f"  {metric:<20} {jan_val:>12.4f} {jul_val:>12.4f} {delta:>+10.4f}")


def load_and_filter(path):
    """Loads, cleans, and removes outliers from a raw parquet file."""
    df = load_and_clean(path, config.ZONE_URL)
    clf = IsolationForest(contamination=0.01, random_state=42).fit(
        df[['fare_amount', 'trip_distance']]
    )
    return df[clf.predict(df[['fare_amount', 'trip_distance']]) == 1]


def main():
    print("=" * 60)
    print("  SEASONALITY GENERALIZATION TEST")
    print("  Train: January 2025  ->  Test: July 2025")
    print("=" * 60)

    # Load data
    print("\n[STEP 1] Loading January and July data...")
    df_jan = load_and_filter(config.FILE_PATH)

    july_path = config.FILE_PATH.replace("2025-01", "2025-07")
    assert "2025-07" in july_path, "Path replacement failed — check FILE_PATH in config"
    df_jul = load_and_filter(july_path)

    # Apply January-based feature engineering to July
    print("[STEP 2] Applying January feature mappings to July data...")
    df_jul = apply_january_mappings(df_jul, df_jan)

    X_jul = df_jul[config.MODEL_FEATURES]
    y_jul = df_jul[config.TARGET]

    # Load trained models
    print("[STEP 3] Loading trained models from disk...")
    model_dir = os.path.join(config.PROJECT_ROOT, "Models")
    rf_pipeline  = joblib.load(os.path.join(model_dir, "deep_tuned_randomforest.joblib"))
    xgb_pipeline = joblib.load(os.path.join(model_dir, "deep_tuned_xgboost.joblib"))

    # Predict and evaluate both models
    print("[STEP 4] Evaluating models on July data...\n")
    pred_dict = {
        "Random Forest": rf_pipeline.predict(X_jul),
        "XGBoost":       xgb_pipeline.predict(X_jul),
    }

    july_metrics = []
    for name, y_pred in pred_dict.items():
        metrics = calculate_metrics(y_jul, y_pred, model_name=name)
        july_metrics.append(metrics)
        print(f"--- {name} ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print()

    # Load January results and print comparison table if available
    jan_metrics_path = os.path.join(model_dir, "january_test_metrics.joblib")
    if os.path.exists(jan_metrics_path):
        january_metrics = joblib.load(jan_metrics_path)
        print_comparison_table(january_metrics, july_metrics)
    else:
        print("[INFO] January metrics not found. Run final_result.py first for a comparison.")
        print("\n=== July metrics only ===")
        print(pd.DataFrame(july_metrics).to_string(index=False))

    # Diagnostic plots
    print("\n[STEP 5] Generating diagnostic plots...")
    plot_generalization(y_jul, pred_dict)
    plot_error_heatmap(df_jul, pred_dict)

    print("\n[DONE] All figures saved to figures_seasonality/")


if __name__ == "__main__":
    main()