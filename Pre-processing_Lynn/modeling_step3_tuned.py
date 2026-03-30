# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:22:53 2026

@author: Ber
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import main as run_preprocessing
from model_utiles import build_pipeline, get_feature_names

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: xgboost is not installed. XGBoost will be skipped.")


# =========================
# Helper functions
# =========================

def evaluate_model(name, model, X_train, X_test, y_train, y_test, kfold):
    pipeline = build_pipeline(model)

    cv_scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=kfold,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=-1,
        return_train_score=False,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    results = {
        "Model": name,
        "CV_MAE_mean": -cv_scores["test_mae"].mean(),
        "CV_MAE_std": cv_scores["test_mae"].std(),
        "CV_RMSE_mean": -cv_scores["test_rmse"].mean(),
        "CV_RMSE_std": cv_scores["test_rmse"].std(),
        "CV_R2_mean": cv_scores["test_r2"].mean(),
        "CV_R2_std": cv_scores["test_r2"].std(),
        "Test_MAE": mean_absolute_error(y_test, y_pred),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Test_R2": r2_score(y_test, y_pred),
    }

    return results, pipeline, y_pred


def extract_linear_coefficients(fitted_pipeline):
    feature_names = get_feature_names()
    model = fitted_pipeline.named_steps["model"]

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", key=np.abs, ascending=False)

    return coef_df


def extract_tree_importance(fitted_pipeline):
    feature_names = get_feature_names()
    model = fitted_pipeline.named_steps["model"]

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return imp_df


def save_metric_barplots(results_df):
    metrics = [("Test_MAE", "Model Comparison - Test MAE"),
               ("Test_RMSE", "Model Comparison - Test RMSE"),
               ("Test_R2", "Model Comparison - Test R2")]

    for metric, title in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(results_df["Model"], results_df[metric])
        plt.title(title)
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(f"{metric}_barplot.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_actual_vs_pred(y_test, y_pred, model_name):
    sample_idx = np.random.RandomState(42).choice(len(y_test), size=min(5000, len(y_test)), replace=False)
    y_true_sample = np.array(y_test)[sample_idx]
    y_pred_sample = np.array(y_pred)[sample_idx]

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true_sample, y_pred_sample, alpha=0.25)
    min_val = min(y_true_sample.min(), y_pred_sample.min())
    max_val = max(y_true_sample.max(), y_pred_sample.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual fare")
    plt.ylabel("Predicted fare")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_actual_vs_pred.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_residual_hist(y_test, y_pred, model_name):
    residuals = np.array(y_test) - np.array(y_pred)

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=80)
    plt.xlabel("Residual (actual - predicted)")
    plt.ylabel("Frequency")
    plt.title(f"Residual Distribution - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_residual_hist.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_feature_importance_plot(df_imp, model_name, top_n=15):
    plot_df = df_imp.head(top_n).iloc[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["Feature"], plot_df.iloc[:, 1])
    plt.xlabel(plot_df.columns[1])
    plt.title(f"Top {top_n} Features - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_error_by_distance_bin(X_test, y_test, y_pred, model_name):
    df_err = X_test.copy()
    df_err["y_true"] = np.array(y_test)
    df_err["y_pred"] = np.array(y_pred)
    df_err["abs_error"] = np.abs(df_err["y_true"] - df_err["y_pred"])

    bins = [0, 1, 2, 5, 10, 20, 50, 100]
    df_err["distance_bin"] = pd.cut(df_err["trip_distance"], bins=bins)

    err_summary = df_err.groupby("distance_bin", observed=False)["abs_error"].mean().reset_index()

    plt.figure(figsize=(9, 5))
    plt.bar(err_summary["distance_bin"].astype(str), err_summary["abs_error"])
    plt.xticks(rotation=45)
    plt.ylabel("Mean Absolute Error")
    plt.xlabel("Trip Distance Bin")
    plt.title(f"Error by Distance Bin - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_error_by_distance_bin.png", dpi=300, bbox_inches="tight")
    plt.close()

    err_summary.to_csv(f"{model_name.lower().replace(' ', '_')}_error_by_distance_bin.csv", index=False)


# =========================
# Optional tuning
# =========================

def tune_random_forest(X_train, y_train, kfold):
    base_pipeline = build_pipeline(
        RandomForestRegressor(random_state=42, n_jobs=-1)
    )

    param_dist = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_dist,
        n_iter=10,
        scoring="neg_mean_absolute_error",
        cv=kfold,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    print("\nBest Random Forest params:", search.best_params_)
    print("Best Random Forest CV MAE:", -search.best_score_)

    return search.best_estimator_


def tune_xgboost(X_train, y_train, kfold):
    base_pipeline = build_pipeline(
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
    )

    param_dist = {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0],
    }

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_dist,
        n_iter=10,
        scoring="neg_mean_absolute_error",
        cv=kfold,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    print("\nBest XGBoost params:", search.best_params_)
    print("Best XGBoost CV MAE:", -search.best_score_)

    return search.best_estimator_


# =========================
# Main
# =========================

def main():
    print("\n=== RUNNING LYNN PREPROCESSING ===")
    result = run_preprocessing()

    (
        X_train, X_test, y_train, y_test, kfold,
        df_step1, df_manhattan, df_work, df_clean,
        zone_map_manhattan
    ) = result

    print("\n=== DATA SUMMARY ===")
    print(f"Train size: {len(X_train):,}")
    print(f"Test size : {len(X_test):,}")
    print(f"Features  : {list(X_train.columns)}")
    print("Target    : fare_amount")

    # =========================
    # Base models
    # =========================
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )

    results = []
    fitted_models = {}
    predictions = {}

    print("\n=== TRAINING BASE MODELS ===")
    for name, model in models.items():
        print(f"\n--- {name} ---")
        res, fitted_pipeline, y_pred = evaluate_model(
            name, model, X_train, X_test, y_train, y_test, kfold
        )
        results.append(res)
        fitted_models[name] = fitted_pipeline
        predictions[name] = y_pred

        print(f"CV MAE   : {res['CV_MAE_mean']:.4f}")
        print(f"CV RMSE  : {res['CV_RMSE_mean']:.4f}")
        print(f"CV R2    : {res['CV_R2_mean']:.4f}")
        print(f"Test MAE : {res['Test_MAE']:.4f}")
        print(f"Test RMSE: {res['Test_RMSE']:.4f}")
        print(f"Test R2  : {res['Test_R2']:.4f}")

    results_df = pd.DataFrame(results).sort_values("Test_MAE")
    print("\n=== BASE MODEL RESULTS ===")
    print(results_df)
    results_df.to_csv("model_results_initial.csv", index=False)

    # =========================
    # Save visuals for base models
    # =========================
    save_metric_barplots(results_df)

    for model_name, y_pred in predictions.items():
        save_actual_vs_pred(y_test, y_pred, model_name)
        save_residual_hist(y_test, y_pred, model_name)
        save_error_by_distance_bin(X_test, y_test, y_pred, model_name)

    # Coefficients / feature importance
    if "Linear Regression" in fitted_models:
        coef_df = extract_linear_coefficients(fitted_models["Linear Regression"])
        coef_df.to_csv("linear_regression_coefficients.csv", index=False)

    if "Random Forest" in fitted_models:
        rf_imp = extract_tree_importance(fitted_models["Random Forest"])
        rf_imp.to_csv("random_forest_feature_importance.csv", index=False)
        save_feature_importance_plot(rf_imp, "Random Forest")

    if "XGBoost" in fitted_models:
        xgb_imp = extract_tree_importance(fitted_models["XGBoost"])
        xgb_imp.to_csv("xgboost_feature_importance.csv", index=False)
        save_feature_importance_plot(xgb_imp, "XGBoost")

    # =========================
    # Optional tuning
    # =========================
    print("\n=== OPTIONAL LIGHT TUNING ===")

    # Random Forest tuning
    print("\nTuning Random Forest...")
    best_rf_pipeline = tune_random_forest(X_train, y_train, kfold)
    rf_pred = best_rf_pipeline.predict(X_test)

    rf_tuned_results = {
        "Model": "Random Forest Tuned",
        "Test_MAE": mean_absolute_error(y_test, rf_pred),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, rf_pred)),
        "Test_R2": r2_score(y_test, rf_pred),
    }
    print(rf_tuned_results)

    rf_tuned_imp = extract_tree_importance(best_rf_pipeline)
    rf_tuned_imp.to_csv("random_forest_tuned_feature_importance.csv", index=False)
    save_feature_importance_plot(rf_tuned_imp, "Random Forest Tuned")
    save_actual_vs_pred(y_test, rf_pred, "Random Forest Tuned")
    save_residual_hist(y_test, rf_pred, "Random Forest Tuned")
    save_error_by_distance_bin(X_test, y_test, rf_pred, "Random Forest Tuned")

    # XGBoost tuning
    if XGB_AVAILABLE:
        print("\nTuning XGBoost...")
        best_xgb_pipeline = tune_xgboost(X_train, y_train, kfold)
        xgb_pred = best_xgb_pipeline.predict(X_test)

        xgb_tuned_results = {
            "Model": "XGBoost Tuned",
            "Test_MAE": mean_absolute_error(y_test, xgb_pred),
            "Test_RMSE": np.sqrt(mean_squared_error(y_test, xgb_pred)),
            "Test_R2": r2_score(y_test, xgb_pred),
        }
        print(xgb_tuned_results)

        xgb_tuned_imp = extract_tree_importance(best_xgb_pipeline)
        xgb_tuned_imp.to_csv("xgboost_tuned_feature_importance.csv", index=False)
        save_feature_importance_plot(xgb_tuned_imp, "XGBoost Tuned")
        save_actual_vs_pred(y_test, xgb_pred, "XGBoost Tuned")
        save_residual_hist(y_test, xgb_pred, "XGBoost Tuned")
        save_error_by_distance_bin(X_test, y_test, xgb_pred, "XGBoost Tuned")

    print("\nDone. Saved results, plots, and importance tables.")


if __name__ == "__main__":
    main()