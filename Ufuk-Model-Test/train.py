"""
train.py
--------
Runs preprocessing, trains all three models, runs cross-validation,
and saves everything to disk. Only needs to run ONCE.

HOW TO RUN
----------
    python train.py

What gets saved to Ufuk-Model-Test/
-------------------------------------
    preprocessor.pkl   — fitted ColumnTransformer + TargetEncoder + col_names
    test_data.pkl      — X_test, y_test (for evaluate.py)
    cv_results.pkl     — per-fold CV metrics for all models
    linear_regression.pkl
    random_forest.pkl
    xgboost.pkl

After this finishes, run evaluate.py to get all plots and metrics.
"""

import sys
import os
import pickle
import numpy as np

# model_utils handles path setup and imports from Pre-processing_Lynn
from model_utils import (
    THIS_DIR, LYNN_DIR, OUTPUT_DIR,
    TREE_SAMPLE, SPLIT_RANDOM_STATE,
    build_preprocessor, apply_preprocessor,
    get_models, compute_metrics, run_cv,
)

import preprocessing as pp   # imported via sys.path set in model_utils


# =============================================================================
# SAVE / LOAD HELPERS
# =============================================================================

def _save(obj, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  Saved -> {path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # STEP 1 — PREPROCESSING
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  STEP 1 — PREPROCESSING")
    print("=" * 60)

    result = pp.main()
    if result is None:
        print(f"\nPreprocessing failed. Check FILE_PATH in:")
        print(f"  {os.path.join(LYNN_DIR, 'config.py')}")
        sys.exit(1)

    X_train, X_test, y_train, y_test, kfold, *_ = result

    # ------------------------------------------------------------------
    # STEP 2 — FIT PREPROCESSOR
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 2 — FITTING FEATURE PREPROCESSOR")
    print("=" * 60)

    ct, te, col_names, X_train_t = build_preprocessor(X_train, y_train)
    X_test_t = apply_preprocessor(ct, te, X_test)

    print(f"  Transformed shape : {X_train_t.shape}")
    print(f"  Feature names     : {col_names}")

    # Save preprocessor and test data so evaluate.py never needs to refit
    _save({'ct': ct, 'te': te, 'col_names': col_names}, 'preprocessor.pkl')
    _save({'X_test_t': X_test_t, 'y_test': np.array(y_test)}, 'test_data.pkl')

    # ------------------------------------------------------------------
    # STEP 3 — TRAIN MODELS + CROSS-VALIDATION
    # ------------------------------------------------------------------
    y_train_arr = np.array(y_train)
    y_test_arr  = np.array(y_test)

    # Fixed subsample for tree models — drawn once, reused for CV and fit
    rng      = np.random.default_rng(SPLIT_RANDOM_STATE)
    tree_idx = rng.choice(len(X_train_t),
                          size=min(TREE_SAMPLE, len(X_train_t)),
                          replace=False)

    models     = get_models()
    cv_results = {}

    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"  MODEL: {name}")
        print("=" * 60)

        if name == 'Linear Regression':
            X_fit = X_train_t
            y_fit = y_train_arr
            print(f"  Training rows : {len(X_fit):,}  (full dataset)")
        else:
            X_fit = X_train_t[tree_idx]
            y_fit = y_train_arr[tree_idx]
            print(f"  Training rows : {len(X_fit):,}  (subsampled from {len(X_train_t):,})")

        # Cross-validation
        fold_df = run_cv(model, X_fit, y_fit, kfold, name)
        cv_results[name] = fold_df

        # Fit on (sub)sample
        print(f"\n  Fitting...")
        model.fit(X_fit, y_fit)

        # Quick sanity check on test set
        y_pred = model.predict(X_test_t)
        print(f"  Test-set metrics:")
        compute_metrics(y_test_arr, y_pred, label=name)

        # Save fitted model
        filename = name.lower().replace(' ', '_') + '.pkl'
        _save(model, filename)

    # Save CV results
    _save(cv_results, 'cv_results.pkl')

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print("  All files saved to:")
    print(f"  {OUTPUT_DIR}")
    print("\n  Run evaluate.py to generate all plots and the final comparison table.")