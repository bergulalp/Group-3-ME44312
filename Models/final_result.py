# =============================================================================
# final_result.py
#
# PURPOSE:
# This is the master orchestration script. It ties together the entire machine 
# learning pipeline from data loading to final evaluation.
#
# WORKFLOW:
# 1. Preprocessing: Loads, cleans, balances, and splits the data into train/test.
# 2. Fast Evaluation: Runs a quick baseline check on a small subset (20k rows).
#    This proves the pipelines function correctly and establishes a performance floor.
# 3. Deep Tuning: Executes the heavy hyperparameter optimization.
#    - Random Forest is tuned on a subset (150k rows) because bagging ensembles 
#      scale poorly (in terms of memory and time) on massive tabular datasets.
#    - XGBoost is tuned on the full dataset because histogram-based gradient 
#      boosting is highly optimized and requires maximum data to perform best.
# 4. Final Evaluation: Tests the fully tuned models against the completely 
#    unseen test set to calculate the final, true generalization metrics.
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os

from preprocessing import main as run_preprocessing
from models import get_baseline_lg, get_rf_model, get_xgb_model
from metrics_evaluating import fast_model_comparison, calculate_metrics
from parameter_tuning import deep_tune_model
import config

warnings.filterwarnings("ignore")

def plot_results(y_test, y_pred, model_name):
    """
    Plots a scatter chart comparing actual fare values against the model's predictions.
    Includes a red dashed line representing perfect predictions (y = x) for visual reference.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3, color="blue")
    
    # Dynamically scale the perfect prediction line based on the data's range
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
    """
    Main execution function. Runs preprocessing, fast baseline comparisons, 
    deep model tuning, and the final metric evaluation.
    """
    # Print which features we are going to use
    print(f"\n[CONFIG MODE] Currently executing with FEATURE_MODE = '{config.FEATURE_MODE}'")
    if config.FEATURE_MODE != 'FULL':
        print(f"  -> Using a reduced feature set ({len(config.MODEL_FEATURES)} features).")

    print("=== STEP 1: Preprocessing ===")
    
    # Stability check to make sure, balanced dataset is deleted.
    cache_path = os.path.join(config.PROCESSED_DATA_DIR, "balanced_data.joblib")
    if os.path.exists(cache_path):
        print(f"[Warning] Delete '{cache_path}'")

    # Extract the preprocessed features, targets, and cross-validation strategy
    X_train, X_test, y_train, y_test, kfold = run_preprocessing()

    # --- SUBSET FOR RANDOM FOREST TUNING ---
    subset_size = 150000
    if len(X_train) > subset_size:
        print(f"[INFO] Training set is large ({len(X_train)} rows). Subsetting {subset_size} for RF tuning...")
        X_train_rf = X_train.sample(n=subset_size, random_state=42)
        y_train_rf = y_train.sample(n=subset_size, random_state=42)
    else:
        X_train_rf, y_train_rf = X_train, y_train

    # Define simple, shallow grids strictly for the fast initial comparison
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
    rf_pipeline = deep_tune_model(
        model_name="RandomForest",
        pipe=get_rf_model(),
        param_grid=config.DEEP_RF_GRID,
        X_train=X_train_rf, 
        y_train=y_train_rf, 
        kfold=kfold
    )

    xgb_pipeline = deep_tune_model(
        model_name="XGBoost",
        pipe=get_xgb_model(),
        param_grid=config.DEEP_XGB_GRID,
        X_train=X_train, 
        y_train=y_train, 
        kfold=kfold
    )

    # === STEP 4: Final Evaluation ===
    print("\n=== FINAL EVALUATION ON UNSEEN TEST DATA ===")
    
    all_january_metrics = []  
    
    for model_name, pipeline, save_name in [
        ("Deep Tuned RF",      rf_pipeline,  "Random Forest"),
        ("Deep Tuned XGBoost", xgb_pipeline, "XGBoost")
    ]:
        y_pred = pipeline.predict(X_test)
        
        final_metrics = calculate_metrics(y_test, y_pred, model_name=save_name)
        all_january_metrics.append(final_metrics)
        
        print(f"\n--- {model_name} ---")
        for k, v in final_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            
        plot_results(y_test, y_pred, model_name=model_name)

    results_path = os.path.join(config.PROJECT_ROOT, "Models", "january_test_metrics.joblib")
    joblib.dump(all_january_metrics, results_path)
    print(f"\n[INFO] January test metrics saved to: {results_path}")

    
if __name__ == "__main__":
    main()