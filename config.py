import os

# --- PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(PROJECT_ROOT, "Data", "yellow_tripdata_2025-01.parquet")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Processed")
BASIC_FIG_DIR = os.path.join(PROJECT_ROOT, "Figures")
for folder in [PROCESSED_DATA_DIR, BASIC_FIG_DIR]:
    os.makedirs(folder, exist_ok=True)
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

# --- DATA SETTINGS ---
KEEP_COLUMNS = ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 
                'trip_distance', 'RatecodeID', 'PULocationID', 'DOLocationID', 'fare_amount', 
                'extra', 'mta_tax', 'improvement_surcharge', 'congestion_surcharge', 'cbd_congestion_fee']
FEE_COLUMNS = ['extra', 'mta_tax', 'improvement_surcharge', 'congestion_surcharge', 'cbd_congestion_fee']

# --- MODEL SETTINGS ---
TEST_SIZE, SPLIT_RANDOM_STATE, N_FOLDS = 0.2, 42, 5
# Master list: Removed raw IDs, added PU_fare_avg and DO_fare_avg for Target Encoding
MODEL_FEATURES = [
    'trip_distance', 'hour', 'day of week', 'is weekend', 'passenger_count', 
    'PU_fare_avg', 'DO_fare_avg', 'route_avg_speed', 'pickup_cluster',
    'route_popularity', 'is_night_fare', 'is_rush_hour', 'is_holiday'
]
TARGET = 'fare_amount'

# --- TRANSFORMATIONS ---
CYCLIC_FEATURES = {'hour': 24, 'day of week': 7}
LOG_FEATURES = ['trip_distance']

# Passthrough: Added the new target-encoded average fare features
PASSTHROUGH_FEATURES = [
    'passenger_count', 'is weekend', 'PU_fare_avg', 'DO_fare_avg', 'route_avg_speed', 
    'pickup_cluster', 'route_popularity', 'is_night_fare', 'is_rush_hour', 'is_holiday'
]

# --- HYPERPARAMETER GRIDS ---
DEEP_RF_GRID = {
    "model__n_estimators":      [200, 300, 500],
    "model__max_depth":         [10, 15, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf":  [1, 2, 4],
    "model__max_features":      [1, 0.8],
}

DEEP_XGB_GRID = {
    "model__n_estimators":     [200, 400, 600],
    "model__max_depth":        [4, 6, 8],
    "model__learning_rate":    [0.03, 0.05, 0.1],
    "model__subsample":        [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0],
    "model__reg_lambda":       [0.5, 1.0, 2.0],
}