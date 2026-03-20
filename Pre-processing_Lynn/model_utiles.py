"""
model_utils.py
--------------
Pipeline construction for the NYC Taxi Fare Prediction project.

Builds a sklearn Pipeline per model that applies the correct feature
transformations before fitting. All three model types (Linear Regression,
Random Forest, XGBoost) share the same interface — call build_pipeline(model)
and pass the result directly to cross_val_score or fit/predict.

Transformation strategy per feature group (defined in config.py):

    LOG_FEATURES      right-skewed continuous features (trip_distance,
                      passenger_count). Transformation: log1p then
                      StandardScaler. Reduces the influence of extreme
                      values, which matters most for Linear Regression.

    CYCLIC_FEATURES   features that wrap around (hour, day of week).
                      Transformation: sin and cos encoding. Ensures that
                      hour 23 and hour 0 are treated as close together,
                      which a plain integer cannot express.

    PASSTHROUGH       binary flags and zone IDs. Passed unchanged.
                      Tree models are scale-invariant so no transformation
                      is needed. Linear Regression receives zone IDs as
                      integers, which is acceptable given the large number
                      of zones — target encoding can be added later if needed.

All configuration values are imported from config.py.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

from config import LOG_FEATURES, CYCLIC_FEATURES, PASSTHROUGH_FEATURES


# =============================================================================
# 1. CUSTOM TRANSFORMERS
# =============================================================================

def _sin_cos_encode(X, period):
    """
    Applies sin and cos encoding to a single-column array.
    Returns a two-column array: [sin(2pi * x / period), cos(2pi * x / period)].

    This maps any cyclic feature onto a unit circle so that the distance
    between hour 23 and hour 0 equals the distance between hour 0 and hour 1.
    """
    X = np.asarray(X).ravel()
    angle = 2 * np.pi * X / period
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _make_cyclic_transformer(period):
    """Returns a FunctionTransformer that applies sin/cos encoding for a given period."""
    return FunctionTransformer(
        func=lambda X: _sin_cos_encode(X, period),
        validate=False,
    )


def _log_then_scale():
    """
    Returns a Pipeline that applies log1p followed by StandardScaler.
    Used for right-skewed continuous features.
    """
    return Pipeline([
        ('log1p', FunctionTransformer(np.log1p, validate=False)),
        ('scaler', StandardScaler()),
    ])


# =============================================================================
# 2. COLUMN TRANSFORMER
# =============================================================================

def build_column_transformer():
    """
    Builds a ColumnTransformer that applies the correct transformation
    to each feature group as defined in config.py.

    Output column order:
    - log-scaled features (one column each)
    - cyclic features (two columns each: sin and cos)
    - passthrough features (unchanged)
    """
    transformers = []

    # Log + scale for skewed continuous features
    transformers.append((
        'log_scale',
        _log_then_scale(),
        LOG_FEATURES,
    ))

    # Sin/cos encoding for each cyclic feature individually
    for feature, period in CYCLIC_FEATURES.items():
        transformers.append((
            f'cyclic_{feature.replace(" ", "_")}',
            _make_cyclic_transformer(period),
            [feature],
        ))

    # Passthrough for binary flags and zone IDs
    transformers.append((
        'passthrough',
        'passthrough',
        PASSTHROUGH_FEATURES,
    ))

    return ColumnTransformer(transformers=transformers, remainder='drop')


# =============================================================================
# 3. PIPELINE BUILDER
# =============================================================================

def build_pipeline(model):
    """
    Wraps a model in a full preprocessing + model Pipeline.

    Usage:
        from sklearn.linear_model import LinearRegression
        pipeline = build_pipeline(LinearRegression())
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

    The ColumnTransformer is fitted only on training data inside each
    cross-validation fold, so there is no leakage from validation data
    into the scaler or encoder.

    Parameters
    ----------
    model : sklearn estimator
        Any unfitted sklearn-compatible model, e.g.:
        LinearRegression(), RandomForestRegressor(), XGBRegressor()

    Returns
    -------
    sklearn Pipeline
    """
    return Pipeline([
        ('preprocessor', build_column_transformer()),
        ('model', model),
    ])


# =============================================================================
# 4. FEATURE NAME HELPER
# =============================================================================

def get_feature_names():
    """
    Returns the list of feature names produced by the ColumnTransformer,
    in the same order as the transformer output columns.

    Useful for interpreting coefficients (Linear Regression) or
    feature importances (Random Forest, XGBoost) after fitting.
    """
    names = []

    # Log-scaled features keep their original name
    names.extend(LOG_FEATURES)

    # Cyclic features produce two columns each
    for feature in CYCLIC_FEATURES:
        names.append(f'{feature}_sin')
        names.append(f'{feature}_cos')

    # Passthrough features keep their original name
    names.extend(PASSTHROUGH_FEATURES)

    return names


# =============================================================================
# ENTRY POINT — quick sanity check
# =============================================================================

if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression

    # Show the transformer structure
    ct = build_column_transformer()
    print("ColumnTransformer steps:")
    for name, transformer, cols in ct.transformers:
        print(f"  {name:30} -> {cols}")

    print(f"\nOutput feature names ({len(get_feature_names())} total):")
    for name in get_feature_names():
        print(f"  - {name}")

    # Build example pipeline
    pipeline = build_pipeline(LinearRegression())
    print(f"\nPipeline steps:")
    for step_name, step in pipeline.steps:
        print(f"  {step_name}: {step}")