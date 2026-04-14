# =============================================================================
# metrics_evaluating.py
#
# PURPOSE:
# This module centralizes all model evaluation and scoring logic to ensure 
# consistent comparison across all models (Linear, Random Forest, XGBoost).
#
# WHY THESE SPECIFIC METRICS?
# - MAPE (Mean Absolute Percentage Error): Taxi fares vary wildly (e.g., a $10 
#   trip vs a $100 trip). An absolute error of $5 is unacceptable on a short trip 
#   but highly accurate on a long trip. MAPE normalizes these errors into 
#   percentages, making it the fairest primary metric for fare prediction.
# - MAE (Mean Absolute Error): Provides a highly interpretable real-world 
#   business metric (e.g., "The model is off by $2.50 on average").
# - MedAE (Median Absolute Error): Extremely robust to outliers. If the MedAE 
#   is significantly lower than the MAE, it indicates that a small handful of 
#   extreme prediction errors are skewing the average.
# - RMSE (Root Mean Squared Error): Squares the errors before averaging, which 
#   heavily penalizes massive mistakes. Useful for ensuring the model doesn't 
#   make extreme pricing errors that would severely anger a customer or driver.
# - R² (R-Squared): Indicates the proportion of variance explained by the model, 
#   showing how much better our predictions are compared to simply guessing 
#   the average fare every time.
# =============================================================================

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score,
    median_absolute_error
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculates standard regression metrics: MAE, MedAE, RMSE, MAPE-based accuracy, and R².
    Clips MAPE at 1.0 (100% error) so extreme outlier predictions do not result 
    in a negative accuracy percentage.
    """
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Clip MAPE to a maximum of 1.0 (100% error) so accuracy bottoms out at 0%
    accuracy = (1 - min(mape, 1.0)) * 100
    
    return {
        "Model":         model_name,
        "Accuracy (%)":  accuracy,
        "MAE":           mean_absolute_error(y_true, y_pred),
        "MedAE":         median_absolute_error(y_true, y_pred),
        "RMSE":          np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²":            r2_score(y_true, y_pred),
    }

def fast_model_comparison(X_train, y_train, kfold, models_dict):
    """
    Runs a quick evaluation on a 20,000 row subset to establish baseline performance.
    Uses cross-validation to measure true generalization rather than training set memorization.
    """
    print("\n=== FAST COMPARISON (Subset: 20k rows) ===")
    
    # Randomly sample 20k rows. Using .sample() avoids bias from the fare-bin 
    # ordering introduced by the balancing function in preprocessing.py.
    X_sub = X_train.sample(n=20000, random_state=42)
    y_sub = y_train.loc[X_sub.index]
    
    results = []
    
    for name, config in models_dict.items():
        print(f"  > Quickly evaluating {name}...")
        pipe = config['pipe']
        param_grid = config.get('params', {})
        
        t0 = time.time()
        
        if not param_grid:
            # Baseline (Linear Regression): Score directly via cross-validation
            cv_scores = cross_val_score(
                pipe, X_sub, y_sub, 
                cv=kfold, 
                scoring="neg_mean_absolute_percentage_error", 
                n_jobs=-1
            )
            pipe.fit(X_sub, y_sub)  # Refit to leave pipeline usable
        else:
            # Complex Models (RF, XGBoost): Shallow random search with built-in CV
            search = RandomizedSearchCV(
                pipe, param_grid, n_iter=3, 
                scoring="neg_mean_absolute_percentage_error", 
                cv=kfold, n_jobs=-1, random_state=42
            )
            search.fit(X_sub, y_sub)
            pipe = search.best_estimator_
            cv_scores = np.array([search.best_score_])

        t_elapsed = time.time() - t0

        # Convert negative MAPE back to a positive decimal, then calculate accuracy
        mean_mape = -cv_scores.mean()
        
        results.append({
            "Model":              name,
            "CV Accuracy (%)":    (1 - mean_mape) * 100,
            "CV MAPE":            mean_mape,
            "Fast Tune Time (s)": round(t_elapsed, 2)
        })
        
    print("\n--- FAST COMPARISON RESULTS ---")
    print(pd.DataFrame(results).to_string(index=False))