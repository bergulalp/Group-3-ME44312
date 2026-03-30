import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import IsolationForest
import joblib
import os


def load_and_preprocess_lynn(file_path: str, zone_url: str, sample_size: int = 100000) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Lynn's preprocessing: load, filter, feature engineer, outlier detection.
    Simplified version for training.
    """
    # Load raw data
    df = pd.read_parquet(file_path)
    
    # Sample
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    
    # Basic filtering
    keep_columns = ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 
                    'trip_distance', 'RatecodeID', 'PULocationID', 'DOLocationID', 'fare_amount']
    df = df[keep_columns].dropna()
    
    # Fee columns >=0
    fee_cols = ['fare_amount']  # simplified
    for col in fee_cols:
        df = df[df[col] >= 0]
    
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    df = df[(df['trip_distance'] > 0) & (df['fare_amount'] > 0)]
    df = df[df['RatecodeID'] != 99]
    
    # Time features
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['trip_duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[df['trip_duration_min'] > 0]
    
    # Custom filtering: Manhattan only (simplified)
    try:
        zone_df = pd.read_csv(zone_url)
        zone_map = zone_df.set_index('LocationID')['Borough'].to_dict()
        df['PU_Borough'] = df['PULocationID'].map(zone_map)
        df['DO_Borough'] = df['DOLocationID'].map(zone_map)
        df = df[(df['PU_Borough'] == 'Manhattan') & (df['DO_Borough'] == 'Manhattan')]
        df = df[df['RatecodeID'] == 1]
        df = df[df['trip_distance'] < 100]
    except:
        pass  # skip if zone data not available
    
    # Outlier detection (simplified)
    df['fare_per_mile'] = df['fare_amount'] / df['trip_distance']
    features_outlier = ['fare_amount', 'trip_distance', 'fare_per_mile']
    log_trans = FunctionTransformer(np.log1p)
    scaled = StandardScaler().fit_transform(log_trans.transform(df[features_outlier]))
    clf = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    df['is_outlier'] = clf.fit_predict(scaled) == -1
    df = df[~df['is_outlier']]
    
    # Select features
    model_features = ['trip_distance', 'hour', 'day_of_week', 'is_weekend', 'passenger_count', 'PULocationID', 'DOLocationID']
    target = 'fare_amount'
    
    X = df[model_features]
    y = df[target]
    
    # One-hot for locations
    loc_cols = ['PULocationID', 'DOLocationID']
    for col in loc_cols:
        if col in X.columns:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
    
    return X, y


def train_fare_prediction_models_lynn(X: pd.DataFrame, y: pd.Series, model_dir: str = "models") -> dict:
    """
    Train Linear Regression en Random Forest voor Lynn's preprocessing.
    """
    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"Data splits: Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")
    
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=10, 
                                               min_samples_leaf=5, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)
        val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
        
        results[name] = {
            'model': model,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_mape': val_mape
        }
        
        print(f"{name.upper()} - Val MAE: ${val_mae:.2f}, RMSE: ${val_rmse:.2f}, R²: {val_r2:.3f}, Accuracy: {val_mape:.2f}%")
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['val_r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 Best model: {best_model_name.upper()} (R²: {results[best_model_name]['val_r2']:.3f})")
    
    y_test_pred = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
    
    print(f"Test Performance - MAE: ${test_mae:.2f}, RMSE: ${test_rmse:.2f}, R²: {test_r2:.3f}, Accuracy: {test_mape:.2f}%")
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{best_model_name}_fare_predictor_lynn.pkl")
    joblib.dump(best_model, model_path)
    print(f"Best model saved: {model_path}")
    
    results['best_model'] = best_model_name
    results['test_mae'] = test_mae
    results['test_rmse'] = test_rmse
    results['test_r2'] = test_r2
    results['test_mape'] = test_mape
    results['model_path'] = model_path
    
    return results


if __name__ == "__main__":
    file_path = "data/raw_data/yellow_tripdata_2025-01.parquet"
    zone_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"
    model_dir = "models"
    sample_size = 100000
    
    X, y = load_and_preprocess_lynn(file_path, zone_url, sample_size)
    results = train_fare_prediction_models_lynn(X, y, model_dir)