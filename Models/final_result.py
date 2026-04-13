# =============================================================================
# final_result.py
# Orchestrates the full pipeline: preprocessing → fast comparison →
# deep tuning (RF + XGBoost) → final evaluation on unseen test data.
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from preprocessing import main as run_preprocessing
from models import get_baseline_lg, get_rf_model, get_xgb_model
from metrics_evaluating import fast_model_comparison, calculate_metrics
from parameter_tuning import deep_tune_model

warnings.filterwarnings("ignore")

def plot_results(y_test, y_pred, model_name):
    """Plots actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3, color="blue")
    
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Perfect Prediction")
    
    plt.title(f"Final Model ({model_name}): Actual vs Predicted", fontsize=14)
    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    print("=== STEP 1: Preprocessing ===")
    X_train, X_test, y_train, y_test, kfold, *rest = run_preprocessing()

    # --- SUBSET FOR RANDOM FOREST TUNING ---
    subset_size = 150000
    if len(X_train) > subset_size:
        print(f"[INFO] Training set is large ({len(X_train)} rows). Subsetting {subset_size} for RF tuning...")
        # We sample both X and y with the same random_state to keep them synced
        # without using .loc, which avoids the duplicate index explosion.
        X_train_rf = X_train.sample(n=subset_size, random_state=42)
        y_train_rf = y_train.sample(n=subset_size, random_state=42)
    else:
        X_train_rf, y_train_rf = X_train, y_train

    # Define grids for the fast comparison
    models_to_compare = {
        "Linear Baseline": {
            "pipe": get_baseline_lg(),
            "params": {}
        },
        "Random Forest": {
            "pipe": get_rf_model(),
            "params": {
                "model__max_depth": [10, 15],
                "model__min_samples_leaf": [5, 10]
            }
        },
        "XGBoost": {
            "pipe": get_xgb_model(),
            "params": {
                "model__max_depth": [6, 8],
                "model__learning_rate": [0.05, 0.1]
            }
        }
    }

    # === STEP 2: Fast Evaluation ===
    fast_model_comparison(X_train, y_train, kfold, models_to_compare)

    # === STEP 3: Deep Parameter Tuning — RF and XGBoost ===
    
    # Expanded grids based on model_results_initial.py (Lynn) — broader search space
    deep_rf_grid = {
        "model__n_estimators":      [200, 300, 500],
        "model__max_depth":         [10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf":  [1, 2, 4],
        "model__max_features":      ["sqrt", "log2"],
    }

    deep_xgb_grid = {
        "model__n_estimators":    [200, 400, 600],
        "model__max_depth":       [4, 6, 8],
        "model__learning_rate":   [0.03, 0.05, 0.1],
        "model__subsample":       [0.7, 0.8, 1.0],
        "model__colsample_bytree":[0.7, 0.8, 1.0],
        "model__reg_lambda":      [0.5, 1.0, 2.0],
    }

    # RF uses the SUBSET (X_train_rf)
    rf_pipeline = deep_tune_model(
        model_name="RandomForest",
        pipe=get_rf_model(),
        param_grid=deep_rf_grid,
        X_train=X_train_rf, y_train=y_train_rf, kfold=kfold
    )

    # XGBoost uses the FULL training set (X_train)
    xgb_pipeline = deep_tune_model(
        model_name="XGBoost",
        pipe=get_xgb_model(),
        param_grid=deep_xgb_grid,
        X_train=X_train, y_train=y_train, kfold=kfold
    )

    # === STEP 4: Final Evaluation — compare both tuned models on unseen test data ===
    print("\n=== FINAL EVALUATION ON UNSEEN TEST DATA ===")

    for model_name, pipeline in [("Deep Tuned RF", rf_pipeline), ("Deep Tuned XGBoost", xgb_pipeline)]:
        y_pred = pipeline.predict(X_test)
        final_metrics = calculate_metrics(y_test, y_pred, model_name=model_name)
        print(f"\n--- {model_name} ---")
        for k, v in final_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        plot_results(y_test, y_pred, model_name=model_name)

if __name__ == "__main__":
    main()