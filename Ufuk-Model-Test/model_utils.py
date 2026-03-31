"""
model_utils.py
--------------
Shared utilities for train.py and evaluate.py.
Contains: path setup, feature transformation, model definitions, metrics.

Do not run this file directly.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from category_encoders import TargetEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# PATH SETUP
# =============================================================================

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
LYNN_DIR   = os.path.normpath(os.path.join(THIS_DIR, '..', 'Pre-processing_Lynn'))
OUTPUT_DIR = THIS_DIR

if LYNN_DIR not in sys.path:
    sys.path.insert(0, LYNN_DIR)

from config import (
    MODEL_FEATURES, TARGET,
    LOG_FEATURES, CYCLIC_FEATURES, PASSTHROUGH_FEATURES,
    N_FOLDS, SPLIT_RANDOM_STATE,
)

# Re-export so train.py and evaluate.py can import everything
# from model_utils without needing their own path setup for config.
__all__ = [
    'THIS_DIR', 'LYNN_DIR', 'OUTPUT_DIR',
    'TREE_SAMPLE', 'SPLIT_RANDOM_STATE', 'N_FOLDS',
    'PALETTE',
    'build_preprocessor', 'apply_preprocessor',
    'get_models', 'compute_metrics', 'run_cv', 'style_ax',
]

# Rows used for tree model training/CV (LR always uses full data)
TREE_SAMPLE = 200_000

# =============================================================================
# COLOUR PALETTE
# =============================================================================

PALETTE = {
    'Linear Regression': '#4C72B0',
    'Random Forest':     '#DD8452',
    'XGBoost':           '#55A868',
}

# =============================================================================
# 1. FEATURE TRANSFORMATION
# =============================================================================

def _cyclic_transform(values, period):
    radians = 2 * np.pi * values / period
    return np.column_stack([np.sin(radians), np.cos(radians)])


def build_preprocessor(X_train, y_train):
    """
    Fits a feature preprocessor on the training set only. Never call on test data.

    Transformations
    ---------------
    log1p + StandardScaler  ->  trip_distance, passenger_count
    Sin/cos cyclic encoding ->  hour, day of week
    Target encoding         ->  PULocationID, DOLocationID
    Passthrough             ->  is weekend

    Returns: ct, te, col_names, X_train_transformed
    """
    zone_cols   = ['PULocationID', 'DOLocationID']
    passthrough = [c for c in PASSTHROUGH_FEATURES if c not in zone_cols]

    log_pipeline = Pipeline([
        ('log',   FunctionTransformer(np.log1p, validate=True)),
        ('scale', StandardScaler()),
    ])
    ct = ColumnTransformer(
        transformers=[
            ('log_scale', log_pipeline,  LOG_FEATURES),
            ('passthru',  'passthrough', passthrough),
        ],
        remainder='drop',
    )
    ct.fit(X_train, y_train)
    X_base = ct.transform(X_train)

    te = TargetEncoder(cols=zone_cols, smoothing=10)
    te.fit(X_train[zone_cols], y_train)
    X_zones = te.transform(X_train[zone_cols]).values

    cyclic_parts, cyclic_names = [], []
    for col, period in CYCLIC_FEATURES.items():
        cyclic_parts.append(_cyclic_transform(X_train[col].values, period))
        cyclic_names += [f'{col}_sin', f'{col}_cos']
    X_cyclic = np.hstack(cyclic_parts)

    X_out     = np.hstack([X_base, X_zones, X_cyclic])
    col_names = (
        [f'log_{c}' for c in LOG_FEATURES] +
        passthrough +
        [f'te_{c}' for c in zone_cols] +
        cyclic_names
    )
    return ct, te, col_names, X_out


def apply_preprocessor(ct, te, X):
    """Applies a pre-fitted preprocessor to new data (test set)."""
    zone_cols = ['PULocationID', 'DOLocationID']
    X_base    = ct.transform(X)
    X_zones   = te.transform(X[zone_cols]).values

    cyclic_parts = []
    for col, period in CYCLIC_FEATURES.items():
        cyclic_parts.append(_cyclic_transform(X[col].values, period))
    X_cyclic = np.hstack(cyclic_parts)

    return np.hstack([X_base, X_zones, X_cyclic])


# =============================================================================
# 2. MODEL DEFINITIONS
# =============================================================================

def get_models():
    """Returns a fresh dict of {name: unfitted estimator}."""
    return {
        'Linear Regression': LinearRegression(),

        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_features='sqrt',
            min_samples_leaf=10,
            max_samples=0.5,
            random_state=SPLIT_RANDOM_STATE,
            n_jobs=-1,
        ),

        'XGBoost': XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method='hist',
            random_state=SPLIT_RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
    }


# =============================================================================
# 3. METRICS
# =============================================================================

def compute_metrics(y_true, y_pred, label=''):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    if label:
        print(f"  {label:<22}  MAE={mae:.3f}   RMSE={rmse:.3f}   R2={r2:.4f}")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


# =============================================================================
# 4. CROSS-VALIDATION
# =============================================================================

def run_cv(model, X_t, y, kfold, model_name):
    """Runs KFold CV and prints per-fold results. Returns a fold metrics DataFrame."""
    scoring = {
        'mae':  'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2':   'r2',
    }
    cv_res = cross_validate(
        model, X_t, y,
        cv=kfold, scoring=scoring,
        n_jobs=-1, return_train_score=False,
    )
    fold_df = pd.DataFrame({
        'Fold': range(1, N_FOLDS + 1),
        'MAE':  -cv_res['test_mae'],
        'RMSE': -cv_res['test_rmse'],
        'R2':    cv_res['test_r2'],
    })
    print(f"\n  --- {model_name} ({N_FOLDS}-Fold CV) ---")
    print(fold_df.to_string(index=False, float_format='{:.4f}'.format))
    print(f"  Mean  MAE={fold_df['MAE'].mean():.3f}  "
          f"RMSE={fold_df['RMSE'].mean():.3f}  "
          f"R2={fold_df['R2'].mean():.4f}")
    return fold_df


# =============================================================================
# 5. PLOT STYLE HELPER
# =============================================================================

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', color='#dddddd', linewidth=0.6, linestyle='--')