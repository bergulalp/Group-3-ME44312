# =============================================================================
# parameter_tuning.py
# In-depth hyperparameter tuning for a single model on the full training set.
#
# Saves both the best pipeline AND the best parameters to disk so tuning only
# needs to run once. On subsequent runs the saved files are loaded directly.
# =============================================================================

import os
import json
import time
import joblib
from sklearn.model_selection import RandomizedSearchCV
import config

def deep_tune_model(model_name, pipe, param_grid, X_train, y_train, kfold):
    """
    In-depth parameter tuning. Takes more time but runs only once.
    Saves the best pipeline (preprocessor + model) and best parameters locally.
    """
    model_dir  = os.path.join(config.PROJECT_ROOT, "Models")
    save_path  = os.path.join(model_dir, f"deep_tuned_{model_name.lower()}.joblib")
    params_path = os.path.join(model_dir, f"deep_tuned_{model_name.lower()}_params.json")  # added: save best params separately

    # Skip tuning if we already did it — load pipeline and print saved params
    if os.path.exists(save_path):
        print(f"\n[INFO] Loaded pre-tuned {model_name} from disk ({save_path}).")
        if os.path.exists(params_path):
            with open(params_path) as f:
                print(f"[INFO] Best parameters: {json.load(f)}")
        return joblib.load(save_path)
        
    print(f"\n=== DEEP PARAMETER TUNING: {model_name} ===")
    print("Running on full dataset. This will take a while, but only runs once...")
    
    t0 = time.time()
    
    search = RandomizedSearchCV(
        pipe, param_grid, 
        n_iter=10,  
        scoring="neg_mean_absolute_percentage_error", 
        cv=kfold, 
        n_jobs=-2, 
        verbose=3,
        random_state=42
    )
    
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    
    print(f"Deep tuning finished in {time.time() - t0:.2f} seconds.")
    print("Best parameters found:", search.best_params_)
    
    os.makedirs(model_dir, exist_ok=True)

    # Save the full pipeline (preprocessor + tuned model)
    joblib.dump(best_pipeline, save_path)
    print(f"Pipeline saved to {save_path}")

    # Save best parameters as readable JSON so you can inspect them without loading the model
    with open(params_path, "w") as f:
        json.dump(search.best_params_, f, indent=2)
    print(f"Best parameters saved to {params_path}")
    
    return best_pipeline