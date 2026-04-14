# =============================================================================
# model_utiles.py
#
# PURPOSE:
# This module defines the custom feature transformation pipeline used across 
# all machine learning models in this project. By centralizing the preprocessor,
# we guarantee that training data, validation data, and unseen test data all 
# undergo the exact same mathematical transformations without data leakage.
#
# KEY TRANSFORMATIONS:
# 1. Log-Scaling: Applies a logarithmic transformation to highly skewed 
#    continuous variables (like 'trip_distance') to normalize their distribution, 
#    followed by standard scaling.
# 2. Cyclic Encoding: Time variables (like 'hour') are continuous but circular. 
#    For example, 23:00 and 00:00 are numerically far apart (23 vs 0) but 
#    temporally only 1 hour apart. We convert these into Sine and Cosine 
#    coordinates so the model understands this circular relationship.
# 3. Passthrough: Categorical/binary features and already encoded features 
#    bypass transformation and go straight to the model.
# =============================================================================

import sys, os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

# Ensure the root directory is in the path to import from config.py
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path: 
    sys.path.append(root)
    
from config import LOG_FEATURES, CYCLIC_FEATURES, PASSTHROUGH_FEATURES

def _sin_cos_encode(X, period):
    """
    Mathematically converts a 1D array of cyclic values into 2D (Sine, Cosine) 
    coordinates on a unit circle.
    
    Args:
        X (array-like): The cyclic feature (e.g., hour of day).
        period (int/float): The maximum value of the cycle (e.g., 24 for hours).
    """
    X = np.asarray(X).ravel()
    # Calculate the angle based on the period
    angle = 2 * np.pi * X / period
    # Return two new columns: sin(angle) and cos(angle)
    return np.column_stack([np.sin(angle), np.cos(angle)])

def _make_cyclic_transformer(period):
    """
    Wraps the mathematical _sin_cos_encode function into a scikit-learn 
    compatible FunctionTransformer.
    """
    # Note: Passing kw_args={'period': period} explicitly is required so that 
    # joblib can successfully serialize (save) this pipeline to disk later.
    return FunctionTransformer(func=_sin_cos_encode, kw_args={'period': period}, validate=False)

def build_pipeline(model):
    """
    Constructs the master scikit-learn Pipeline combining all feature 
    transformations and the final predictive model.
    
    Args:
        model (estimator): The scikit-learn or XGBoost model object.
        
    Returns:
        Pipeline: The complete, ready-to-train sequence.
    """
    # 1. Log-Scale Pipeline:
    # np.log1p is used instead of np.log to safely handle values of 0 (log(1+x)).
    log_pipe = Pipeline([
        ('log', FunctionTransformer(np.log1p)), 
        ('scale', StandardScaler())
    ])
    
    # Initialize the list of column transformations
    transformers = [
        ('log_scale', log_pipe, LOG_FEATURES)
    ]
    
    # 2. Cyclic Pipeline:
    # Dynamically create a sine/cosine transformer for every cyclic feature defined in config
    for f, p in CYCLIC_FEATURES.items():
        transformers.append((f'cyc_{f}', _make_cyclic_transformer(p), [f]))
        
    # 3. Passthrough Pipeline:
    # Let binary flags, target-encoded averages, and clusters pass through unmodified
    transformers.append(('pass', 'passthrough', PASSTHROUGH_FEATURES))
    
    # Combine everything into a ColumnTransformer, then append the predictive model
    return Pipeline([
        ('pre', ColumnTransformer(transformers)), 
        ('model', model)
    ])