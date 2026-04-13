from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Import YOUR pipeline builder
from model_utiles import build_pipeline

def get_baseline_lg():
    """Returns a simple, non-optimized Linear Regression baseline inside your pipeline."""
    return build_pipeline(LinearRegression())

def get_rf_model(**kwargs):
    """Returns a Random Forest pipeline. Kwargs allow overriding defaults."""
    params = {'n_jobs': -1, 'random_state': 42}
    params.update(kwargs)
    return build_pipeline(RandomForestRegressor(**params))

def get_xgb_model(**kwargs):
    """Returns an XGBoost pipeline. Kwargs allow overriding defaults."""
    params = {'n_jobs': -1, 'random_state': 42, 'tree_method': 'hist'}
    params.update(kwargs)
    return build_pipeline(XGBRegressor(**params))