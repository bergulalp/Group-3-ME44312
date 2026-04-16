# =============================================================================
# config.py
#
# PURPOSE:
# This file serves as the central configuration hub for the entire project.
# It defines file paths, data loading parameters, cross-validation settings,
# and hyperparameter search grids. 
#
# FEATURE SELECTION:
# The FEATURE_MODE variable controls which features are passed to the models.
# - 'FULL': All 13 original features (Baseline).
# - 'XGB_95': Top 10 features needed for XGBoost to reach 95% predictive influence.
# - 'RF_95': Top 3 features needed for Random Forest to reach 95% predictive influence.
# =============================================================================

import os

# --- PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(PROJECT_ROOT, "Data", "yellow_tripdata_2025-01.parquet")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Processed")
BASIC_FIG_DIR = os.path.join(PROJECT_ROOT, "Figures")

# Ensure necessary directories exist
for folder in [PROCESSED_DATA_DIR, BASIC_FIG_DIR]:
    os.makedirs(folder, exist_ok=True)

ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

# --- DATA SETTINGS ---
# The raw columns required from the parquet file before preprocessing
KEEP_COLUMNS = ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 
                'trip_distance', 'RatecodeID', 'PULocationID', 'DOLocationID', 'fare_amount', 
                'extra', 'mta_tax', 'improvement_surcharge', 'congestion_surcharge', 'cbd_congestion_fee']

# Columns representing regulatory fees and surcharges
FEE_COLUMNS = ['extra', 'mta_tax', 'improvement_surcharge', 'congestion_surcharge', 'cbd_congestion_fee']

# --- MODEL VALIDATION SETTINGS ---
TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
N_FOLDS = 5
TARGET = 'fare_amount'

# =============================================================================
# FEATURE SELECTION TOGGLE
# Options: 'FULL', 'XGB_95', 'RF_95'
# =============================================================================
FEATURE_MODE = 'RF_95'

if FEATURE_MODE == 'XGB_95':
    # --- Ultra-Lightweight Set (Top 3 RF Features) ---
    MODEL_FEATURES = [
        'trip_distance', 'route_popularity', 'route_avg_speed'
    ]
    CYCLIC_FEATURES = {} 
    LOG_FEATURES = ['trip_distance']
    PASSTHROUGH_FEATURES = [
        'route_avg_speed', 'route_popularity'
    ]

elif FEATURE_MODE == 'XGB_95':
    # --- Optimized XGBoost Set (10 Features) ---
    # Dropped: 'day of week', 'passenger_count', 'is_rush_hour'
    MODEL_FEATURES = [
        'trip_distance', 'hour', 'PU_fare_avg', 'DO_fare_avg', 
        'route_avg_speed', 'pickup_cluster', 'route_popularity', 
        'is_night_fare', 'is weekend', 'is_holiday'
    ]
    CYCLIC_FEATURES = {'hour': 24} 
    LOG_FEATURES = ['trip_distance']
    PASSTHROUGH_FEATURES = [
        'is weekend', 'PU_fare_avg', 'DO_fare_avg', 'route_avg_speed', 
        'pickup_cluster', 'route_popularity', 'is_night_fare', 'is_holiday'
    ]

else: # 'FULL' mode
    # --- Full Feature Set (All 13 Features) ---
    MODEL_FEATURES = [
        'trip_distance', 'hour', 'day of week', 'is weekend', 'passenger_count', 
        'PU_fare_avg', 'DO_fare_avg', 'route_avg_speed', 'pickup_cluster',
        'route_popularity', 'is_night_fare', 'is_rush_hour', 'is_holiday'
    ]
    CYCLIC_FEATURES = {'hour': 24, 'day of week': 7}
    LOG_FEATURES = ['trip_distance']
    PASSTHROUGH_FEATURES = [
        'passenger_count', 'is weekend', 'PU_fare_avg', 'DO_fare_avg', 'route_avg_speed', 
        'pickup_cluster', 'route_popularity', 'is_night_fare', 'is_rush_hour', 'is_holiday'
    ]

# =============================================================================
# HYPERPARAMETER GRIDS
# =============================================================================

# Grid for HalvingRandomSearchCV (Optimized for Random Forest)
DEEP_RF_GRID = {
    "model__n_estimators":      [200, 300, 500],
    "model__max_depth":         [10, 15, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf":  [1, 2, 4],
    "model__max_features":      [1, 0.8],
}

# Grid for RandomizedSearchCV (Optimized for XGBoost)
DEEP_XGB_GRID = {
    "model__n_estimators":     [200, 400, 600],
    "model__max_depth":        [4, 6, 8],
    "model__learning_rate":    [0.03, 0.05, 0.1],
    "model__subsample":        [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0],
    "model__reg_lambda":       [0.5, 1.0, 2.0],
}