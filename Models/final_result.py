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
    print("=== STEP 1: Preprocessing ===")
    # Extract the preprocessed features, targets, and cross-validation strategy
    X_train, X_test, y_train, y_test, kfold = run_preprocessing()

    # --- SUBSET FOR RANDOM FOREST TUNING ---
    # Random Forests train extremely slowly on hundreds of thousands of rows. 
    # We subset the training data here specifically for RF to save compute time.
    subset_size = 150000
    if len(X_train) > subset_size:
        print(f"[INFO] Training set is large ({len(X_train)} rows). Subsetting {subset_size} for RF tuning...")
        # We sample both X and y using the same random_state to ensure rows stay aligned.
        # Using .sample() avoids pandas duplicate index explosion issues.
        X_train_rf = X_train.sample(n=subset_size, random_state=42)
        y_train_rf = y_train.sample(n=subset_size, random_state=42)
    else:
        X_train_rf, y_train_rf = X_train, y_train

    # Define simple, shallow grids strictly for the fast initial comparison
    models_to_compare = {
        "Linear Baseline": {
            "pipe": get_baseline_lg(),
            "params": {} # Linear Regression doesn't need hyperparameter tuning here
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
    # Evaluate the shallow models on a 20k row subset to establish a baseline
    fast_model_comparison(X_train, y_train, kfold, models_to_compare)

    # === STEP 3: Deep Parameter Tuning — RF and XGBoost ===
    # We import the deep grids directly from config.py to ensure consistency 
    # between this script and plot_final_results.py.

    # Execute tuning for Random Forest using the smaller subset
    rf_pipeline = deep_tune_model(
        model_name="RandomForest",
        pipe=get_rf_model(),
        param_grid=config.DEEP_RF_GRID,
        X_train=X_train_rf, 
        y_train=y_train_rf, 
        kfold=kfold
    )

    # Execute tuning for XGBoost using the complete training dataset
    xgb_pipeline = deep_tune_model(
        model_name="XGBoost",
        pipe=get_xgb_model(),
        param_grid=config.DEEP_XGB_GRID,
        X_train=X_train, 
        y_train=y_train, 
        kfold=kfold
    )

    # === STEP 4: Final Evaluation ===
    # Compare both fully optimized models strictly against the unseen test dataset.
    print("\n=== FINAL EVALUATION ON UNSEEN TEST DATA ===")

    for model_name, pipeline in [("Deep Tuned RF", rf_pipeline), ("Deep Tuned XGBoost", xgb_pipeline)]:
        # Generate predictions on the hold-out test set
        y_pred = pipeline.predict(X_test)
        
        # Calculate and print formatting metrics
        final_metrics = calculate_metrics(y_test, y_pred, model_name=model_name)
        print(f"\n--- {model_name} ---")
        for k, v in final_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            
        # Display the actual vs predicted scatter plot
        plot_results(y_test, y_pred, model_name=model_name)

if __name__ == "__main__":
    main()