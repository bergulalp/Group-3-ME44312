"""
simple_model_training.py
------------------------
Simple script to run Linear Regression and Random Forest on Lynn's preprocessed data.
Uses X_train, X_test, y_train, y_test from preprocessing.py.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import argparse
from preprocessing import main

def run_models(pre_sample=100000, train_sample=None, test_sample=None):
    # Get preprocessed data from Lynn's pipeline
    result = main(sample_size=pre_sample)
    if result is None:
        print("Preprocessing failed.")
        return

    X_train, X_test, y_train, y_test, kfold, *_ = result

    # Sample to reduce size for faster training (optional)
    if train_sample is not None and len(X_train) > train_sample:
        from sklearn.utils import resample
        X_train, y_train = resample(X_train, y_train, n_samples=train_sample, random_state=42)
    if test_sample is not None and len(X_test) > test_sample:
        from sklearn.utils import resample
        X_test, y_test = resample(X_test, y_test, n_samples=test_sample, random_state=42)
    
    # Optimize memory usage
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    }
    
    # Train and evaluate
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)
        
        # Predict on test
        y_pred = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        print(f"MAE: ${mae:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"R²: {r2:.3f}")
        print(f"Accuracy (MAPE): {mape:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple model training on Lynn data')
    parser.add_argument('--pre_sample', type=int, default=100000, help='Number of rows to sample before preprocessing')
    parser.add_argument('--train_sample', type=int, default=None, help='Number of training rows after split (None uses full split)')
    parser.add_argument('--test_sample', type=int, default=None, help='Number of test rows after split (None uses full split)')
    args = parser.parse_args()

    run_models(pre_sample=args.pre_sample, train_sample=args.train_sample, test_sample=args.test_sample)