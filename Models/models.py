# =============================================================================
# models.py
#
# PURPOSE:
# This module serves as a centralized "Model Factory." In machine learning 
# projects, it is vital to separate the model definition from the execution 
# scripts. This ensures consistency across training, tuning, and testing.
#
# DESIGN STRATEGY:
# Each function returns a scikit-learn 'Pipeline' object. These pipelines 
# combine our custom preprocessing steps (defined in model_utiles.py) with 
# a specific estimator. 
#
# MODELS INCLUDED:
# 1. Linear Regression: Acts as the "Baseline" to measure if complex models 
#    actually provide significant value.
# 2. Random Forest: A robust bagging ensemble useful for capturing 
#    non-linear spatial relationships.
# 3. XGBoost: A high-performance gradient boosting framework optimized for 
#    speed and predictive power on tabular data.
# =============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Import the custom pipeline builder that handles scaling, logging, and encoding
from model_utiles import build_pipeline

def get_baseline_lg():
    """
    Initializes a standard Linear Regression model wrapped in the project pipeline.
    This serves as the simplest predictive baseline for comparison.
    """
    return build_pipeline(LinearRegression())

def get_rf_model(**kwargs):
    """
    Initializes a Random Forest Regressor pipeline. 
    
    Args:
        **kwargs: Allows for dynamic overriding of model hyperparameters 
                  (e.g., max_depth, n_estimators) during tuning.
    """
    # Default settings: use all CPU cores (-1) and fixed seed for reproducibility
    params = {'n_jobs': -1, 'random_state': 42}
    params.update(kwargs)
    
    return build_pipeline(RandomForestRegressor(**params))

def get_xgb_model(**kwargs):
    """
    Initializes an XGBoost Regressor pipeline.
    
    Args:
        **kwargs: Parameters passed to the XGBRegressor during the tuning phase.
    """
    # tree_method='hist' is used to significantly speed up training on large 
    # datasets by binning continuous features into histograms.
    params = {
        'n_jobs': -1, 
        'random_state': 42, 
        'tree_method': 'hist',
        'objective': 'reg:squarederror' # Explicitly set for continuous fare prediction
    }
    params.update(kwargs)
    
    return build_pipeline(XGBRegressor(**params))