# =============================================================================
# metrics_evaluating.py
# Model evaluation helpers: metric calculation and fast model comparison.
#
# Key design decision:
#   - fast_model_comparison uses cross_val_score instead of predicting on the
#     training subset itself, which would only measure overfitting, not generalisation.
# =============================================================================

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score  # added r2_score
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """Calculates standard regression metrics: MAE, RMSE, MAPE-based accuracy, and R²."""
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {
        "Model":         model_name,
        "Accuracy (%)":  (1 - mape) * 100,
        "MAE":           mean_absolute_error(y_true, y_pred),
        "RMSE":          np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²":            r2_score(y_true, y_pred),   # added: proportion of variance explained
    }

def fast_model_comparison(X_train, y_train, kfold, models_dict):
    """Runs a fast random grid search on a 20k subset to compare baseline vs complex models."""
    print("\n=== FAST COMPARISON (Subset: 20k rows) ===")
    
    # Subset to keep it fast
    X_sub = X_train.iloc[:20000]
    y_sub = y_train.iloc[:20000]
    
    results = []
    
    for name, config in models_dict.items():
        print(f"  > Quickly evaluating {name}...")
        pipe = config['pipe']
        param_grid = config.get('params', {})
        
        t0 = time.time()
        
        if not param_grid:
            # Baseline: no hyperparameters to tune, score directly with cross-validation
            cv_scores = cross_val_score(pipe, X_sub, y_sub, cv=kfold, scoring="neg_mean_absolute_percentage_error", n_jobs=-1)
            pipe.fit(X_sub, y_sub)  # Refit on full subset so we can report train metrics consistently
        else:
            # Tune hyperparameters; cross-validation is already inside RandomizedSearchCV
            search = RandomizedSearchCV(
                pipe, param_grid, n_iter=3, 
                scoring="neg_mean_absolute_percentage_error", 
                cv=kfold, n_jobs=-1, random_state=42
            )
            search.fit(X_sub, y_sub)
            pipe = search.best_estimator_
            # best_score_ is the mean CV score across folds (negative MAPE)
            cv_scores = np.array([search.best_score_])

        t_elapsed = time.time() - t0

        # Report cross-validated accuracy: this reflects generalisation, not memorisation
        mean_mape = -cv_scores.mean()
        results.append({
            "Model":           name,
            "CV Accuracy (%)": (1 - mean_mape) * 100,
            "CV MAPE":         mean_mape,
            "Fast Tune Time (s)": round(t_elapsed, 2)
        })
        
    print("\n--- FAST COMPARISON RESULTS ---")
    print(pd.DataFrame(results).to_string(index=False))