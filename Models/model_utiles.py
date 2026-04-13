import sys, os, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path: sys.path.append(root)
from config import LOG_FEATURES, CYCLIC_FEATURES, PASSTHROUGH_FEATURES

def _sin_cos_encode(X, period):
    X = np.asarray(X).ravel()
    angle = 2 * np.pi * X / period
    return np.column_stack([np.sin(angle), np.cos(angle)])

def _make_cyclic_transformer(period):
    # Named arguments allow joblib to save the model
    return FunctionTransformer(func=_sin_cos_encode, kw_args={'period': period}, validate=False)

def build_pipeline(model):
    log_pipe = Pipeline([('log', FunctionTransformer(np.log1p)), ('scale', StandardScaler())])
    transformers = [('log_scale', log_pipe, LOG_FEATURES)]
    for f, p in CYCLIC_FEATURES.items():
        transformers.append((f'cyc_{f}', _make_cyclic_transformer(p), [f]))
    transformers.append(('pass', 'passthrough', PASSTHROUGH_FEATURES))
    
    return Pipeline([('pre', ColumnTransformer(transformers)), ('model', model)])