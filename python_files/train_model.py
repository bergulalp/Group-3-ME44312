import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import os


def load_training_data(features_path: str, target_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Laad features en target voor model training.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y) voor training
    """
    # Laad data
    X = pd.read_parquet(features_path)
    y = pd.read_parquet(target_path)["fare_amount"]

    # Verwijder datetime kolommen die niet geschikt zijn voor training
    datetime_cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
    X = X.drop(columns=[c for c in datetime_cols if c in X.columns])

    # One-hot encode categorische variabelen
    categorical_cols = [
        c for c in ["VendorID", "RatecodeID", "PULocationID", "DOLocationID", "payment_type", "store_and_fwd_flag"]
        if c in X.columns
    ]
    
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Zorg ervoor dat indices overeenkomen
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len]
    y = y.iloc[:min_len]
    
    return X, y


def train_fare_prediction_models(X: pd.DataFrame, y: pd.Series, model_dir: str = "models") -> dict:
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
        val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100  # in percentage
        val_accuracy_2 = np.mean(np.abs(y_val - y_val_pred) <= 2) * 100  # percentage within $2
        
        results[name] = {
            'model': model,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_mape': val_mape,
            'val_accuracy_2': val_accuracy_2
        }
        
        print(f"{name.upper()} - Val MAE: ${val_mae:.2f}, RMSE: ${val_rmse:.2f}, R²: {val_r2:.3f}, MAPE: {val_mape:.2f}%, Accuracy (±$2): {val_accuracy_2:.2f}%")
    
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
    test_accuracy_2 = np.mean(np.abs(y_test - y_test_pred) <= 2) * 100
    
    print(f"Test Performance - MAE: ${test_mae:.2f}, RMSE: ${test_rmse:.2f}, R²: {test_r2:.3f}, MAPE: {test_mape:.2f}%, Accuracy (±$2): {test_accuracy_2:.2f}%")
    
    # Sla beste model op
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{best_model_name}_fare_predictor.pkl")
    joblib.dump(best_model, model_path)
    print(f"Best model saved: {model_path}")
    
    # Voeg test metrics toe aan results
    results['best_model'] = best_model_name
    results['test_mae'] = test_mae
    results['test_rmse'] = test_rmse
    results['test_r2'] = test_r2
    results['test_mape'] = test_mape
    results['test_accuracy_2'] = test_accuracy_2
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
    # Paden naar preprocessed data
    features_path = "data/pre_processed_data/yellow_tripdata_2025-01_features.parquet"
    target_path = "data/pre_processed_data/yellow_tripdata_2025-01_target.parquet"
    model_dir = "models"

    # Laad training data
    X, y = load_training_data(features_path, target_path)

    # Train beide modellen en vergelijk
    results = train_fare_prediction_models(X, y, model_dir)