# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 01:11:06 2026

@author: Ber
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

from preprocessing import main as run_preprocessing
from model_utiles import build_pipeline, get_feature_names

# Optional packages
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


# =========================================================
# CONFIG
# =========================================================
OUTPUT_DIR = Path("figures_advanced2")
OUTPUT_DIR.mkdir(exist_ok=True)

# If you have Lynn's model_results_initial.csv, this script will use it
RESULTS_CSV = Path("model_results_initial.csv")

# Optional Manhattan taxi zone shapefile for choropleth residual map
# Change only if needed
SHAPEFILE_PATH = Path("taxi_zones") / "taxi_zones.shp"

RANDOM_STATE = 42

# Sampling sizes for heavier diagnostics
SAMPLE_HEXBIN = 100_000
SAMPLE_RESIDUAL = 100_000
SAMPLE_PERMUTATION = 30_000
SAMPLE_SHAP = 5_000
SAMPLE_PDP = 20_000

# Base XGBoost params
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Pretty plotting style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.family": "DejaVu Sans",
})


# =========================================================
# HELPERS
# =========================================================
def save_close(fig, filename: str):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def sample_xy(X, y, n, seed=RANDOM_STATE):
    if len(X) <= n:
        return X.copy(), y.copy()
    idx = X.sample(n=n, random_state=seed).index
    return X.loc[idx].copy(), y.loc[idx].copy()


def get_pipeline_preprocessor(fitted_pipeline):
    # Everything before the final model step
    return fitted_pipeline[:-1]


def compute_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def try_load_results_csv():
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    return None


# =========================================================
# MODEL COMPARISON VISUALS
# =========================================================
def plot_model_comparison(results_df: pd.DataFrame):
    # 1) Error metrics together
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, results_df["Test_MAE"], width, label="MAE")
    bars2 = ax.bar(x + width/2, results_df["Test_RMSE"], width, label="RMSE")

    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"])
    ax.set_ylabel("Error")
    ax.set_title("Model comparison: test errors")
    ax.legend()

    for bars in [bars1, bars2]:
        for b in bars:
            ax.text(
                b.get_x() + b.get_width()/2,
                b.get_height() + 0.02,
                f"{b.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=11
            )

    save_close(fig, "01_model_comparison_errors.png")

    # 2) R²
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(results_df["Model"], results_df["Test_R2"])
    ax.set_ylabel("Test R²")
    ax.set_title("Model comparison: test R²")

    for b in bars:
        ax.text(
            b.get_x() + b.get_width()/2,
            b.get_height() + 0.005,
            f"{b.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=11
        )

    save_close(fig, "02_model_comparison_r2.png")

    # 3) Relative MAE improvement vs LR
    lr_mae = results_df.loc[results_df["Model"] == "Linear Regression", "Test_MAE"].values[0]
    tmp = results_df.copy()
    tmp["MAE_improvement_vs_LR_pct"] = (lr_mae - tmp["Test_MAE"]) / lr_mae * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(tmp["Model"], tmp["MAE_improvement_vs_LR_pct"])
    ax.set_ylabel("MAE improvement vs LR (%)")
    ax.set_title("Relative improvement over linear regression")

    for b in bars:
        ax.text(
            b.get_x() + b.get_width()/2,
            b.get_height() + 0.25,
            f"{b.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=11
        )

    save_close(fig, "03_mae_improvement_vs_lr.png")


# =========================================================
# LINEAR / TREE IMPORTANCE VISUALS FROM EXISTING CSVs
# =========================================================
def plot_existing_importances():
    lr_path = Path("linear_regression_coefficients.csv")
    rf_path = Path("random_forest_feature_importance.csv")
    xgb_path = Path("xgboost_feature_importance.csv")

    if lr_path.exists():
        lr_coef = pd.read_csv(lr_path).copy().head(12).sort_values("Coefficient")
        colors = ["tab:red" if c < 0 else "tab:blue" for c in lr_coef["Coefficient"]]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(lr_coef["Feature"], lr_coef["Coefficient"], color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title("Linear regression coefficients")
        ax.set_xlabel("Coefficient")

        for b in bars:
            val = b.get_width()
            ax.text(
                val + (0.03 if val >= 0 else -0.03),
                b.get_y() + b.get_height()/2,
                f"{val:.3f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=10
            )

        save_close(fig, "04_linear_regression_coefficients.png")

    def tree_plot(csv_path, full_name, sec_name, full_file, sec_file):
        if not csv_path.exists():
            return
        df = pd.read_csv(csv_path).copy()

        # Full
        top_full = df.head(12).sort_values("Importance")
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(top_full["Feature"], top_full["Importance"])
        ax.set_title(full_name)
        ax.set_xlabel("Importance")
        for b in bars:
            val = b.get_width()
            ax.text(val + 0.0015, b.get_y() + b.get_height()/2, f"{val:.3f}", va="center", fontsize=10)
        save_close(fig, full_file)

        # Secondary (excluding trip_distance)
        top_sec = df[df["Feature"] != "trip_distance"].head(10).sort_values("Importance")
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(top_sec["Feature"], top_sec["Importance"])
        ax.set_title(sec_name)
        ax.set_xlabel("Importance")
        for b in bars:
            val = b.get_width()
            ax.text(val + 0.0006, b.get_y() + b.get_height()/2, f"{val:.3f}", va="center", fontsize=10)
        save_close(fig, sec_file)

    tree_plot(
        rf_path,
        "Random Forest: feature importance",
        "Random Forest: secondary feature importance",
        "05_random_forest_importance_full.png",
        "06_random_forest_importance_secondary.png"
    )

    tree_plot(
        xgb_path,
        "XGBoost: feature importance",
        "XGBoost: secondary feature importance",
        "07_xgboost_importance_full.png",
        "08_xgboost_importance_secondary.png"
    )


# =========================================================
# ADVANCED XGBOOST VISUALS
# =========================================================
def plot_actual_vs_pred_hexbin(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    hb = ax.hexbin(y_true, y_pred, gridsize=60, mincnt=1)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Count")

    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1)

    ax.set_xlabel("Actual fare")
    ax.set_ylabel("Predicted fare")
    ax.set_title("XGBoost: actual vs predicted")

    save_close(fig, "09_xgb_actual_vs_pred_hexbin.png")


def plot_residual_diagnostics(y_true, y_pred):
    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].hist(residuals, bins=80)
    axes[0].set_xlabel("Residual (actual - predicted)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Residual distribution")

    hb = axes[1].hexbin(y_pred, residuals, gridsize=60, mincnt=1)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted fare")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs predicted")
    cbar = fig.colorbar(hb, ax=axes[1])
    cbar.set_label("Count")

    save_close(fig, "10_xgb_residual_diagnostics.png")


def plot_error_by_distance_bin(X_test, y_true, y_pred):
    df = X_test.copy()
    df["y_true"] = np.array(y_true)
    df["y_pred"] = np.array(y_pred)
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])

    bins = [0, 1, 2, 5, 10, 20, 50, 100]
    labels = ["0-1", "1-2", "2-5", "5-10", "10-20", "20-50", "50+"]
    df["distance_bin"] = pd.cut(
        df["trip_distance"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    summary = df.groupby("distance_bin", observed=False)["abs_error"].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Mean MAE by bin
    bars = axes[0].bar(summary["distance_bin"].astype(str), summary["abs_error"])
    axes[0].set_xlabel("Trip distance bin (miles)")
    axes[0].set_ylabel("Mean absolute error")
    axes[0].set_title("Error by distance bin")
    axes[0].tick_params(axis="x", rotation=30)

    for b in bars:
        axes[0].text(
            b.get_x() + b.get_width()/2,
            b.get_height() + 0.02,
            f"{b.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    # Distribution by bin
    sns.boxplot(data=df, x="distance_bin", y="abs_error", ax=axes[1], showfliers=False)
    axes[1].set_xlabel("Trip distance bin (miles)")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_title("Absolute error distribution by distance bin")
    axes[1].tick_params(axis="x", rotation=30)

    save_close(fig, "11_xgb_error_by_distance_bin.png")


def plot_error_heatmap_hour_weekday(X_test, y_true, y_pred):
    df = X_test.copy()
    df["y_true"] = np.array(y_true)
    df["y_pred"] = np.array(y_pred)
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])

    heat = df.pivot_table(
        index="day of week",
        columns="hour",
        values="abs_error",
        aggfunc="mean"
    ).sort_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(heat, ax=ax, cmap="viridis", cbar_kws={"label": "Mean absolute error"})
    ax.set_title("XGBoost: mean absolute error by hour × day of week")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week (0=Mon)")

    save_close(fig, "12_xgb_error_heatmap_hour_weekday.png")


def plot_zone_residual_analysis_thresholded(X_test, y_true, y_pred, min_trips=500):
    df = X_test.copy()
    df["y_true"] = np.array(y_true)
    df["y_pred"] = np.array(y_pred)
    df["residual"] = df["y_true"] - df["y_pred"]
    df["abs_error"] = np.abs(df["residual"])

    zone_summary = (
        df.groupby("PULocationID")
        .agg(
            mean_residual=("residual", "mean"),
            mae=("abs_error", "mean"),
            n_trips=("residual", "size")
        )
        .reset_index()
    )

    # Apply trip-count threshold
    zone_summary_thr = zone_summary[zone_summary["n_trips"] >= min_trips].copy()

    # Save table
    zone_summary_thr.sort_values("mean_residual", ascending=False).to_csv(
        OUTPUT_DIR / "zone_residual_summary_thresholded.csv",
        index=False
    )

    top_under = zone_summary_thr.sort_values("mean_residual", ascending=False).head(10)
    top_over = zone_summary_thr.sort_values("mean_residual", ascending=True).head(10)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].barh(top_under["PULocationID"].astype(str), top_under["mean_residual"])
    axes[0].set_title(f"Top underpredicted pickup zones\n(min {min_trips} trips)")
    axes[0].set_xlabel("Mean residual (actual - predicted)")
    axes[0].invert_yaxis()

    axes[1].barh(top_over["PULocationID"].astype(str), top_over["mean_residual"])
    axes[1].set_title(f"Top overpredicted pickup zones\n(min {min_trips} trips)")
    axes[1].set_xlabel("Mean residual (actual - predicted)")
    axes[1].invert_yaxis()

    save_close(fig, "zone_residual_bars_thresholded.png")

    # Also plot MAE by zone for the largest-error zones
    top_mae = zone_summary_thr.sort_values("mae", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_mae["PULocationID"].astype(str), top_mae["mae"])
    ax.set_title(f"Pickup zones with highest MAE\n(min {min_trips} trips)")
    ax.set_xlabel("Mean absolute error")
    ax.invert_yaxis()

    for b in bars:
        val = b.get_width()
        ax.text(val + 0.02, b.get_y() + b.get_height()/2, f"{val:.2f}", va="center")

    save_close(fig, "zone_mae_thresholded.png")

    # Optional Manhattan choropleth if shapefile exists
    if GEOPANDAS_AVAILABLE and SHAPEFILE_PATH.exists():
        try:
            zones = gpd.read_file(SHAPEFILE_PATH)

            # Common TLC shapefile column name is LocationID
            if "LocationID" in zones.columns:
                merge_key = "LocationID"
            elif "locationid" in zones.columns:
                merge_key = "locationid"
            else:
                merge_key = None

            if merge_key is not None:
                gdf = zones.merge(zone_summary, left_on=merge_key, right_on="PULocationID", how="inner")

                # Manhattan-only if borough column exists
                borough_cols = [c for c in gdf.columns if c.lower() == "borough"]
                if borough_cols:
                    gdf = gdf[gdf[borough_cols[0]].str.lower() == "manhattan"].copy()

                fig, axes = plt.subplots(1, 2, figsize=(16, 8))

                gdf.plot(column="mean_residual", legend=True, ax=axes[0])
                axes[0].set_title("Pickup-zone mean residual")
                axes[0].axis("off")

                gdf.plot(column="mae", legend=True, ax=axes[1])
                axes[1].set_title("Pickup-zone MAE")
                axes[1].axis("off")

                save_close(fig, "14_xgb_zone_residual_choropleth.png")

        except Exception as e:
            print(f"[WARN] Could not generate zone choropleth: {e}")


def plot_permutation_importance(fitted_pipeline, X_test, y_test):
    Xp, yp = sample_xy(X_test, y_test, SAMPLE_PERMUTATION)

    perm = permutation_importance(
        fitted_pipeline,
        Xp,
        yp,
        scoring="neg_mean_absolute_error",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    imp_df = pd.DataFrame({
        "Feature": Xp.columns,
        "MeanImportance": perm.importances_mean,
        "StdImportance": perm.importances_std
    }).sort_values("MeanImportance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        imp_df["Feature"],
        imp_df["MeanImportance"],
        xerr=imp_df["StdImportance"],
        capsize=3
    )
    ax.set_title("Permutation importance on test set")
    ax.set_xlabel("Decrease in score after shuffling feature")

    save_close(fig, "15_xgb_permutation_importance.png")
    imp_df.to_csv(OUTPUT_DIR / "15_xgb_permutation_importance.csv", index=False)


def plot_shap_visuals(fitted_pipeline, X_test):
    if not SHAP_AVAILABLE:
        print("[INFO] shap not installed. Skipping SHAP visuals.")
        return

    Xs, _ = sample_xy(X_test, pd.Series(index=X_test.index, dtype=float), SAMPLE_SHAP)

    preprocessor = get_pipeline_preprocessor(fitted_pipeline)
    model = fitted_pipeline.named_steps["model"]

    X_trans = preprocessor.transform(Xs)
    feature_names = get_feature_names()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    # Beeswarm
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_trans,
        feature_names=feature_names,
        show=False
    )
    plt.title("SHAP summary (beeswarm)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "16_xgb_shap_beeswarm.png", bbox_inches="tight")
    plt.close()

    # Dependence on transformed trip_distance if present
    if "trip_distance" in feature_names:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            "trip_distance",
            shap_values,
            X_trans,
            feature_names=feature_names,
            show=False
        )
        plt.title("SHAP dependence: trip_distance")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "17_xgb_shap_dependence_trip_distance.png", bbox_inches="tight")
        plt.close()

    # Optional temporal SHAP dependence
    if "hour_cos" in feature_names:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            "hour_cos",
            shap_values,
            X_trans,
            feature_names=feature_names,
            show=False
        )
        plt.title("SHAP dependence: hour_cos")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "18_xgb_shap_dependence_hour_cos.png", bbox_inches="tight")
        plt.close()

def plot_error_by_fare_bin(y_true, y_pred):
    df = pd.DataFrame({
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred)
    })
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])

    bins = [0, 10, 20, 30, 40, 60, 100]
    labels = ["0-10", "10-20", "20-30", "30-40", "40-60", "60+"]
    df["fare_bin"] = pd.cut(df["y_true"], bins=bins, labels=labels, include_lowest=True)

    summary = (
        df.groupby("fare_bin", observed=False)
        .agg(
            mae=("abs_error", "mean"),
            median_ae=("abs_error", "median"),
            n=("abs_error", "size")
        )
        .reset_index()
    )

    summary.to_csv(OUTPUT_DIR / "mae_by_fare_bin.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    bars = axes[0].bar(summary["fare_bin"].astype(str), summary["mae"])
    axes[0].set_title("MAE by actual fare bin")
    axes[0].set_xlabel("Actual fare bin")
    axes[0].set_ylabel("Mean absolute error")

    for b in bars:
        val = b.get_height()
        axes[0].text(
            b.get_x() + b.get_width()/2,
            val + 0.03,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    sns.boxplot(data=df, x="fare_bin", y="abs_error", ax=axes[1], showfliers=False)
    axes[1].set_title("Absolute error distribution by fare bin")
    axes[1].set_xlabel("Actual fare bin")
    axes[1].set_ylabel("Absolute error")

    save_close(fig, "mae_by_fare_bin.png")

def plot_partial_dependence_and_ice(fitted_pipeline, X_test):
    Xp, _ = sample_xy(X_test, pd.Series(index=X_test.index, dtype=float), SAMPLE_PDP)

    # Use original feature names because the estimator is the full pipeline
    pd_features = ["trip_distance", "hour", "is weekend"]
    pd_features = [f for f in pd_features if f in Xp.columns]

    if len(pd_features) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 4.5 * len(pd_features)))
    if len(pd_features) == 1:
        ax = [ax]

    PartialDependenceDisplay.from_estimator(
        fitted_pipeline,
        Xp,
        features=pd_features,
        kind="average",
        ax=ax
    )
    fig.suptitle("Partial dependence plots", y=1.02)
    save_close(fig, "19_xgb_partial_dependence.png")

    # ICE on trip_distance and hour if available
    ice_features = [f for f in ["trip_distance", "hour"] if f in Xp.columns]
    if len(ice_features) > 0:
        fig, ax = plt.subplots(figsize=(14, 4.5 * len(ice_features)))
        if len(ice_features) == 1:
            ax = [ax]

        PartialDependenceDisplay.from_estimator(
            fitted_pipeline,
            Xp,
            features=ice_features,
            kind="both",
            subsample=200,
            random_state=RANDOM_STATE,
            ax=ax
        )
        fig.suptitle("ICE + partial dependence", y=1.02)
        save_close(fig, "20_xgb_ice_and_pdp.png")
        
def plot_spatial_zone_maps(
    X_test,
    y_true,
    y_pred,
    shapefile_path=SHAPEFILE_PATH,
    min_trips=500
):
    """
    Spatial diagnostics at pickup-zone level.

    Creates:
    - average actual fare by pickup zone
    - pickup volume by pickup zone
    - mean residual by pickup zone
    - MAE by pickup zone
    - filtered maps using a minimum trip threshold
    - support vs error scatter
    - CSV summary
    """

    if not GEOPANDAS_AVAILABLE:
        print("[INFO] geopandas not installed. Skipping spatial zone maps.")
        return

    if not Path(shapefile_path).exists():
        print(f"[INFO] Shapefile not found at {shapefile_path}. Skipping spatial zone maps.")
        return

    # -------------------------
    # Build zone-level summary
    # -------------------------
    df = X_test.copy()
    df["y_true"] = np.array(y_true)
    df["y_pred"] = np.array(y_pred)
    df["residual"] = df["y_true"] - df["y_pred"]
    df["abs_error"] = np.abs(df["residual"])

    zone_summary = (
        df.groupby("PULocationID")
        .agg(
            avg_actual_fare=("y_true", "mean"),
            avg_pred_fare=("y_pred", "mean"),
            mean_residual=("residual", "mean"),
            mae=("abs_error", "mean"),
            median_ae=("abs_error", "median"),
            n_trips=("residual", "size")
        )
        .reset_index()
    )

    zone_summary["abs_mean_residual"] = zone_summary["mean_residual"].abs()
    zone_summary["support_group"] = pd.cut(
        zone_summary["n_trips"],
        bins=[0, 200, 500, 1000, 5000, np.inf],
        labels=["0-200", "200-500", "500-1000", "1000-5000", "5000+"],
        include_lowest=True
    )

    zone_summary.to_csv(OUTPUT_DIR / "zone_summary_spatial_diagnostics.csv", index=False)

    # -------------------------
    # Load shapefile
    # -------------------------
    gdf = gpd.read_file(shapefile_path)

    # Determine merge key
    if "LocationID" in gdf.columns:
        merge_key = "LocationID"
    elif "locationid" in gdf.columns:
        merge_key = "locationid"
    else:
        raise ValueError("Could not find LocationID column in shapefile.")

    gdf = gdf.merge(zone_summary, left_on=merge_key, right_on="PULocationID", how="left")

    # Manhattan only if possible
    borough_cols = [c for c in gdf.columns if c.lower() == "borough"]
    if borough_cols:
        borough_col = borough_cols[0]
        gdf_manhattan = gdf[gdf[borough_col].str.lower() == "manhattan"].copy()
    else:
        gdf_manhattan = gdf.copy()

    # -------------------------
    # 2x2 main choropleth panel
    # -------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Average actual fare
    gdf_manhattan.plot(
        column="avg_actual_fare",
        ax=axes[0, 0],
        legend=True,
        cmap="YlOrRd",
        missing_kwds={"color": "lightgrey"}
    )
    axes[0, 0].set_title("Average actual fare by pickup zone")
    axes[0, 0].axis("off")

    # Pickup volume
    gdf_manhattan.plot(
        column="n_trips",
        ax=axes[0, 1],
        legend=True,
        cmap="Blues",
        missing_kwds={"color": "lightgrey"}
    )
    axes[0, 1].set_title("Pickup volume by zone")
    axes[0, 1].axis("off")

    # Mean residual (diverging)
    # positive = underprediction, negative = overprediction
    vmax = np.nanmax(np.abs(gdf_manhattan["mean_residual"]))
    gdf_manhattan.plot(
        column="mean_residual",
        ax=axes[1, 0],
        legend=True,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        missing_kwds={"color": "lightgrey"}
    )
    axes[1, 0].set_title("Mean residual by pickup zone\n(actual - predicted)")
    axes[1, 0].axis("off")

    # MAE
    gdf_manhattan.plot(
        column="mae",
        ax=axes[1, 1],
        legend=True,
        cmap="viridis",
        missing_kwds={"color": "lightgrey"}
    )
    axes[1, 1].set_title("MAE by pickup zone")
    axes[1, 1].axis("off")

    save_close(fig, "21_spatial_2x2_zone_panel.png")

    # -------------------------
    # Volume-thresholded maps
    # -------------------------
    gdf_thr = gdf_manhattan[gdf_manhattan["n_trips"] >= min_trips].copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    vmax_thr = np.nanmax(np.abs(gdf_thr["mean_residual"])) if len(gdf_thr) else 1.0

    gdf_thr.plot(
        column="mean_residual",
        ax=axes[0],
        legend=True,
        cmap="coolwarm",
        vmin=-vmax_thr,
        vmax=vmax_thr,
        missing_kwds={"color": "lightgrey"}
    )
    axes[0].set_title(f"Mean residual by pickup zone\n(min {min_trips} trips)")
    axes[0].axis("off")

    gdf_thr.plot(
        column="mae",
        ax=axes[1],
        legend=True,
        cmap="magma",
        missing_kwds={"color": "lightgrey"}
    )
    axes[1].set_title(f"MAE by pickup zone\n(min {min_trips} trips)")
    axes[1].axis("off")

    save_close(fig, "22_spatial_thresholded_residual_mae_maps.png")

    # -------------------------
    # Support vs error scatter
    # -------------------------
    zone_plot = zone_summary.copy()
    zone_plot = zone_plot[zone_plot["n_trips"] > 0].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].scatter(zone_plot["n_trips"], zone_plot["mae"], alpha=0.7)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Zone support (number of trips, log scale)")
    axes[0].set_ylabel("Zone MAE")
    axes[0].set_title("Zone support vs MAE")

    axes[1].scatter(zone_plot["n_trips"], zone_plot["abs_mean_residual"], alpha=0.7)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Zone support (number of trips, log scale)")
    axes[1].set_ylabel("|Mean residual|")
    axes[1].set_title("Zone support vs absolute mean residual")

    save_close(fig, "23_zone_support_vs_error.png")

    # -------------------------
    # Top zones table exports
    # -------------------------
    zone_summary.sort_values("mean_residual", ascending=False).head(15).to_csv(
        OUTPUT_DIR / "top_underpredicted_zones.csv", index=False
    )
    zone_summary.sort_values("mean_residual", ascending=True).head(15).to_csv(
        OUTPUT_DIR / "top_overpredicted_zones.csv", index=False
    )
    zone_summary.sort_values("mae", ascending=False).head(15).to_csv(
        OUTPUT_DIR / "top_mae_zones.csv", index=False
    )

    print("[INFO] Spatial zone diagnostics saved.")


# =========================================================
# MAIN
# =========================================================
def main():
    print("=== RUNNING PREPROCESSING ===")
    result = run_preprocessing()

    (
        X_train, X_test, y_train, y_test, kfold,
        df_step1, df_manhattan, df_work, df_clean,
        zone_map_manhattan
    ) = result

    print(f"Train size: {len(X_train):,}")
    print(f"Test size : {len(X_test):,}")

    # Existing comparison / importance visuals if files exist
    results_df = try_load_results_csv()
    if results_df is not None:
        print("=== PLOTTING EXISTING MODEL COMPARISON FIGURES ===")
        plot_model_comparison(results_df)
    plot_existing_importances()

    # Fit XGBoost pipeline
    print("=== FITTING BASE XGBOOST MODEL ===")
    xgb = XGBRegressor(**XGB_PARAMS)
    xgb_pipeline = build_pipeline(xgb)
    xgb_pipeline.fit(X_train, y_train)
    y_pred = xgb_pipeline.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    print("XGBoost metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Samples for expensive plots
    X_hex, y_hex = sample_xy(X_test, y_test, SAMPLE_HEXBIN)
    y_hex_pred = xgb_pipeline.predict(X_hex)

    X_res, y_res = sample_xy(X_test, y_test, SAMPLE_RESIDUAL)
    y_res_pred = xgb_pipeline.predict(X_res)

    # 1. Actual vs predicted
    plot_actual_vs_pred_hexbin(np.array(y_hex), np.array(y_hex_pred))

    # 2. Residual diagnostics
    plot_residual_diagnostics(np.array(y_res), np.array(y_res_pred))

    # 3. Error by distance bin
    plot_error_by_distance_bin(X_test, y_test, y_pred)

    # 4. Error heatmap hour x weekday
    plot_error_heatmap_hour_weekday(X_test, y_test, y_pred)
    
    plot_spatial_zone_maps(
    X_test,
    y_test,
    y_pred,
    shapefile_path=SHAPEFILE_PATH,
    min_trips=500
    )

    # 5. Zone residual analysis + optional map
    plot_zone_residual_analysis_thresholded(X_test, y_test, y_pred, min_trips=500)

    # 6. Permutation importance
    plot_permutation_importance(xgb_pipeline, X_test, y_test)

    # 7. SHAP
    plot_shap_visuals(xgb_pipeline, X_test)

    # 8. PDP + ICE
    plot_partial_dependence_and_ice(xgb_pipeline, X_test)
    
    plot_error_by_fare_bin(y_test, y_pred)

    print(f"\nSaved advanced figures to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()