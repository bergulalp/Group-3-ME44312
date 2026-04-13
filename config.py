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

# Master list: Added popularity and NYC surcharge rules
MODEL_FEATURES = [
    'trip_distance', 'hour', 'day of week', 'is weekend', 'passenger_count', 
    'PULocationID', 'DOLocationID', 'route_avg_speed', 'pickup_cluster',
    'route_popularity', 'is_night_fare', 'is_rush_hour', 'is_holiday'
]
TARGET = 'fare_amount'

# --- TRANSFORMATIONS ---
CYCLIC_FEATURES = {'hour': 24, 'day of week': 7}
LOG_FEATURES = ['trip_distance', 'passenger_count']

# Passthrough: Added the new numerical/binary features
PASSTHROUGH_FEATURES = [
    'is weekend', 'PULocationID', 'DOLocationID', 'route_avg_speed', 
    'pickup_cluster', 'route_popularity', 'is_night_fare', 'is_rush_hour', 'is_holiday'
]