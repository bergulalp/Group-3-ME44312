# =============================================================================
# parameter_tuning.py
#
# PURPOSE:
# This module handles the computationally expensive task of hyperparameter tuning.
# To optimize training time and model performance, we use two distinct strategies:
# 
# 1. Random Forest: Uses HalvingRandomSearchCV. This is an efficient successive 
#    halving approach that trains many candidates on a small subset of data, 
#    and only trains the best performing candidates on larger subsets.
# 
# 2. XGBoost: Uses a pragmatic Two-Step Early Stopping approach. 
#    - Step A: Tune core tree parameters (depth, learning rate) using standard CV.
#    - Step B: Isolate the best parameters, set 'n_estimators' artificially high, 
#              and use a hold-out validation set to trigger Early Stopping. This 
#              finds the absolute optimal number of trees without overfitting.
#
# Artifacts (best pipelines and parameter JSONs) are saved to disk so the tuning 
# only needs to execute once.
# =============================================================================

import os
import json
import time
import joblib
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# Explicitly enable the experimental Halving search feature in scikit-learn
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingRandomSearchCV

import config

def deep_tune_model(model_name, pipe, param_grid, X_train, y_train, kfold):
    """
    Executes in-depth parameter tuning based on the model type.
    Saves the best pipeline and a readable JSON of the best parameters locally.
    
    Args:
        model_name (str): "RandomForest" or "XGBoost".
        pipe (Pipeline): The scikit-learn pipeline containing the preprocessor and model.
        param_grid (dict): The hyperparameter search space.
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        kfold (CV Splitter): Cross-validation strategy.
        
    Returns:
        Pipeline: The fully tuned and fitted model pipeline.
    """
    model_dir  = os.path.join(config.PROJECT_ROOT, "Models")
    save_path  = os.path.join(model_dir, f"deep_tuned_{model_name.lower()}.joblib")
    params_path = os.path.join(model_dir, f"deep_tuned_{model_name.lower()}_params.json")

    # Skip tuning if we already did it, load pipeline and print saved params
    if os.path.exists(save_path):
        print(f"\n[INFO] Loaded pre-tuned {model_name} from disk ({save_path}).")
        if os.path.exists(params_path):
            with open(params_path) as f:
                print(f"[INFO] Best parameters: {json.load(f)}")
        return joblib.load(save_path)
        
    print(f"\n=== DEEP PARAMETER TUNING: {model_name} ===")
    print("This will take a while, but only runs once...")
    
    t0 = time.time()
    
    # -------------------------------------------------------------------------
    # STRATEGY 1: RANDOM FOREST (Standard RandomizedSearchCV)
    # -------------------------------------------------------------------------
    if model_name == "RandomForest":
        print("[INFO] Using HalvingRandomSearchCV for Random Forest (Speed optimization)...")
        from sklearn.experimental import enable_halving_search_cv
        from sklearn.model_selection import HalvingRandomSearchCV

        search = HalvingRandomSearchCV(
            pipe, param_grid,
            factor=3,           
            resource='n_samples', 
            max_resources=len(X_train),
            scoring="neg_mean_absolute_error",
            cv=kfold,               
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        best_params = search.best_params_

    # -------------------------------------------------------------------------
    # STRATEGY 2: XGBOOST (Two-Step Pragmatic Early Stopping)
    # -------------------------------------------------------------------------
    elif model_name == "XGBoost":
        print("[INFO] Step A: Tuning core parameters using RandomizedSearchCV...")
        search = RandomizedSearchCV(
            pipe, param_grid, 
            n_iter=10,  
            scoring="neg_mean_absolute_error", 
            cv=kfold, 
            n_jobs=-1, 
            verbose=1,
            random_state=42
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        best_params = search.best_params_
        
        print("\n[INFO] Step B: Applying Early Stopping to find optimal n_estimators...")
        # 1. Extract the model step from the pipeline and set a high ceiling for trees
        xgb_model = best_pipeline.named_steps['model']
        xgb_model.set_params(n_estimators=2000, early_stopping_rounds=50)
        
        # 2. Create a temporary validation set specifically for early stopping
        X_sub_train, X_val, y_sub_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42
        )
        
        # 3. Manually transform the data through the preprocessor so XGBoost can read it
        preprocessor = best_pipeline.named_steps['pre']
        
        # Fit only on sub_train, transform both
        preprocessor.fit(X_sub_train)
        X_sub_train_trans = preprocessor.transform(X_sub_train)
        X_val_trans = preprocessor.transform(X_val)
        
        # 4. Fit with the eval_set to trigger early stopping
        xgb_model.fit(
            X_sub_train_trans, y_sub_train, 
            eval_set=[(X_val_trans, y_val)], 
            verbose=False
        )
        
        # 5. Retrieve the optimal number of trees discovered before the model overfit
        optimal_trees = xgb_model.best_iteration
        print(f"[INFO] Optimal n_estimators found: {optimal_trees} trees.")
        
        # 6. Rebuild the pipeline with the exact optimal parameters and fit on FULL training data
        best_params['model__n_estimators'] = optimal_trees
        best_pipeline.named_steps['model'].set_params(
            n_estimators=optimal_trees, 
            early_stopping_rounds=None # Remove early stopping for the final full fit
        )
        
        print("[INFO] Finalizing pipeline fit on the complete training dataset...")
        best_pipeline.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # FALLBACK
    # -------------------------------------------------------------------------
    else:
        raise ValueError("Model name must be either 'RandomForest' or 'XGBoost'")
        
    print(f"\nDeep tuning finished in {time.time() - t0:.2f} seconds.")
    print("Final Optimal Parameters:", best_params)
    
    # Ensure directory exists before saving
    os.makedirs(model_dir, exist_ok=True)

    # Save the full, ready-to-use pipeline (preprocessor + tuned model)
    joblib.dump(best_pipeline, save_path)
    print(f"Pipeline safely saved to: {save_path}")

    # Save best parameters as readable JSON for easy inspection
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Best parameters documented at: {params_path}")
    
    return best_pipeline