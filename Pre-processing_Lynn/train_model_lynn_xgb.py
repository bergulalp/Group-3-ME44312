"""
train_model_lynn_xgb.py
----------------------
Use Lynn preprocessing and RandomizedSearchCV with XGBoost + RandomForest for fare prediction.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

from preprocessing import main


def get_train_test(pre_sample=100000, train_sample=None, test_sample=None):
    # load and preprocess with Lydia pipeline
    result = main(sample_size=pre_sample)
    if result is None:
        raise RuntimeError('Preprocessing failed.')

    X_train, X_test, y_train, y_test, *_ = result

    if train_sample is not None and len(X_train) > train_sample:
        from sklearn.utils import resample
        X_train, y_train = resample(X_train, y_train, n_samples=train_sample, random_state=42)
    if test_sample is not None and len(X_test) > test_sample:
        from sklearn.utils import resample
        X_test, y_test = resample(X_test, y_test, n_samples=test_sample, random_state=42)

    return X_train, X_test, y_train, y_test


def run_randomized_search(X_train, y_train):
    models = {
        'rf': Pipeline([
            ('scale', StandardScaler()),
            ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
        ]),
        'xgb': Pipeline([
            ('scale', StandardScaler()),
            ('xgb', XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1, verbosity=0))
        ])
    }

    params = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [10, 15, 20, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__max_features': ['sqrt', 'log2', 0.5],

        'xgb__n_estimators': [100, 200, 300],
        'xgb__max_depth': [4, 6, 8, 10],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__subsample': [0.6, 0.8, 1.0],
        'xgb__colsample_bytree': [0.6, 0.8, 1.0],
        'xgb__reg_alpha': [0, 0.1, 1],
        'xgb__reg_lambda': [1, 2, 3]
    }

    best_models = {}

    for name, pipeline in models.items():
        print(f"Starting RandomizedSearchCV for {name}")
        param_dist = {k: v for k, v in params.items() if k.startswith(name + '__')}
        if not param_dist:
            continue

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=20,
            scoring='neg_mean_absolute_error',
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_
        print(f"Best {name} params: {search.best_params_}")
        print(f"Best {name} MAE (CV): {-search.best_score_:.3f}")

    return best_models


def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        results[name] = { 'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape }
        print(f"=== {name} test ===")
        print(f"MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
    return results


def main_run():
    X_train, X_test, y_train, y_test = get_train_test(pre_sample=500000, train_sample=100000, test_sample=20000)
    best_models = run_randomized_search(X_train, y_train)
    results = evaluate_models(best_models, X_test, y_test)

    # save best overall by MAE
    best_name = min(results.keys(), key=lambda k: results[k]['mae'])
    joblib.dump(best_models[best_name], f"best_model_lynn_{best_name}.pkl")
    print(f"Best model saved: best_model_lynn_{best_name}.pkl")

    return results


if __name__ == '__main__':
    main_run()