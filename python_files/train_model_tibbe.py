import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import os


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tibbe's preprocessing: clean and engineer features.
    """
    # Drop missing and logical errors
    df = df.dropna(subset=['fare_amount', 'trip_distance', 'tpep_pickup_datetime'])
    df = df[(df['fare_amount'] >= 2.50) & (df['fare_amount'] <= 200.0)]
    df = df[(df['trip_distance'] > 0) & (df['trip_distance'] <= 50.0)]
    
    # Time Features
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    
    # Cyclic Encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour']/24.0)
    
    return df


def load_training_data(data_path: str, sample_size: int = 100000) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Laad en preprocess data volgens Tibbe's methode.

    Args:
        data_path: Pad naar raw data
        sample_size: Aantal samples om te gebruiken (None voor alles)
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y) voor training
    """
    # Laad raw data
    df = pd.read_parquet(data_path)
    
    # Preprocess
    df = clean_and_engineer(df)
    
    # Sample als nodig
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    # Select features
    features = ['trip_distance', 'pickup_hour', 'day_of_week', 'PULocationID', 'DOLocationID', 'hour_sin', 'hour_cos']
    target = 'fare_amount'
    
    # Zorg dat alle features bestaan
    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    y = df[target]
    
    # One-hot encode location IDs als categorisch
    location_cols = ['PULocationID', 'DOLocationID']
    existing_loc_cols = [c for c in location_cols if c in X.columns]
    if existing_loc_cols:
        X = pd.get_dummies(X, columns=existing_loc_cols, drop_first=True)
    
    return X, y


def train_fare_prediction_models(X: pd.DataFrame, y: pd.Series, model_dir: str = "models_tibbe") -> dict:
    """
    Train zowel Linear Regression als Random Forest modellen voor fare amount prediction.
    Vergelijk prestaties op validation set en sla beste model op.
    
    Args:
        X: Features dataframe
        y: Target series (fare_amount)
        model_dir: Directory waar modellen opgeslagen worden
    
    Returns:
        dict: Performance metrics voor beide modellen
    """
    # Split de data: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"Data splits: Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")
    
    # Definieer modellen
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    
    # Train en evalueer beide modellen
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Voorspel op validation set
        y_val_pred = model.predict(X_val)
        
        # Bereken validation metrics
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)
        val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100  # algehele voorspelling accuracy in %
        
        results[name] = {
            'model': model,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_mape': val_mape
        }
        
        print(f"{name.upper()} - Val MAE: ${val_mae:.2f}, RMSE: ${val_rmse:.2f}, R²: {val_r2:.3f}, Accuracy: {val_mape:.2f}%")
    
    # Kies beste model gebaseerd op validation R²
    best_model_name = max(results.keys(), key=lambda x: results[x]['val_r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 Best model: {best_model_name.upper()} (R²: {results[best_model_name]['val_r2']:.3f})")
    
    # Test beste model op test set
    y_test_pred = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
    
    print(f"Test Performance - MAE: ${test_mae:.2f}, RMSE: ${test_rmse:.2f}, R²: {test_r2:.3f}, Accuracy: {test_mape:.2f}%")
    
    # Sla beste model op
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{best_model_name}_fare_predictor_tibbe.pkl")
    joblib.dump(best_model, model_path)
    print(f"Best model saved: {model_path}")
    
    # Voeg test metrics toe aan results
    results['best_model'] = best_model_name
    results['test_mae'] = test_mae
    results['test_rmse'] = test_rmse
    results['test_r2'] = test_r2
    results['test_mape'] = test_mape
    results['model_path'] = model_path
    
    return results


def predict_fare_amount(model_path: str, features: pd.DataFrame) -> np.ndarray:
    """
    Gebruik een getraind model om fare amounts te voorspellen.

    Args:
        model_path: Pad naar het opgeslagen model
        features: DataFrame met features (zelfde kolommen als training)

    Returns:
        np.ndarray: Voorspelde fare amounts
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model niet gevonden: {model_path}")
    
    model = joblib.load(model_path)
    predictions = model.predict(features)
    return predictions


if __name__ == "__main__":
    # Pad naar raw data
    data_path = "data/raw_data/yellow_tripdata_2025-01.parquet"
    model_dir = "models"
    sample_size = 100000  # Gebruik sample voor snellere training

    # Laad training data
    X, y = load_training_data(data_path, sample_size)

    # Train beide modellen en vergelijk
    results = train_fare_prediction_models(X, y, model_dir)