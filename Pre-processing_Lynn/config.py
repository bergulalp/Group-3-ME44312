"""
config.py
---------
Central configuration for the NYC Taxi Fare Prediction project (Group 3, ME44312).

All hardcoded values from preprocessing.py, eda.py, and model_utils.py are
defined here. To change any parameter — data paths, model features, filter
thresholds, transformation settings, or plot settings — edit this file only.
"""

import os
import numpy as np

# =============================================================================
# PATHS
# =============================================================================

# Absolute path to the folder containing this config file (project root).
# All other paths are derived from this so the project works on any machine.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Raw data file — place the parquet file two levels above this script,
# or update this path to wherever the file is stored on your machine.
FILE_PATH = os.path.normpath(
    os.path.join("/home/pimovergaag/Documents/ME43321/data/raw_data/yellow_tripdata_2025-01.parquet")
)

# NYC Taxi Zone lookup CSV — used to map LocationID to Borough and Zone name.
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"

# Folder containing the NYC Taxi Zone shapefile for the Manhattan map visual.
# Download and unzip from: https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip
SHAPEFILE_DIR = "taxi_zones"

# =============================================================================
# DATA COLUMNS
# =============================================================================

# Columns to keep from the raw parquet file. All others are dropped immediately.
KEEP_COLUMNS = [
    'VendorID',
    'tpep_pickup_datetime',
    'tpep_dropoff_datetime',
    'passenger_count',
    'trip_distance',
    'RatecodeID',
    'PULocationID',
    'DOLocationID',
    'fare_amount',
    'extra',
    'mta_tax',
    'improvement_surcharge',
    'congestion_surcharge',
    'cbd_congestion_fee',
]

# Fee columns that must be non-negative to be valid.
FEE_COLUMNS = [
    'extra',
    'mta_tax',
    'improvement_surcharge',
    'congestion_surcharge',
    'cbd_congestion_fee',
]

# =============================================================================
# FILTERING THRESHOLDS
# =============================================================================

# Passenger count bounds (inclusive).
PASSENGER_MIN = 1
PASSENGER_MAX = 6

# Maximum trip distance in miles. Trips above this are physically impossible
# within Manhattan and are treated as data errors.
MAX_TRIP_DISTANCE_MILES = 100

# RatecodeID to keep. 1 = Standard rate. 99 = Unknown/invalid (excluded).
VALID_RATECODE = 1

# =============================================================================
# OUTLIER DETECTION
# =============================================================================

# Fraction of trips flagged as outliers by Isolation Forest.
# 0.01 = remove the most anomalous 1% of trips.
ISOLATION_FOREST_CONTAMINATION = 0.01

# Random seed for Isolation Forest reproducibility.
ISOLATION_FOREST_RANDOM_STATE = 42

# =============================================================================
# TRAIN / TEST SPLIT & CROSS-VALIDATION
# =============================================================================

# Fraction of data held out as test set.
TEST_SIZE = 0.2

# Random seed for train/test split and KFold shuffle reproducibility.
SPLIT_RANDOM_STATE = 42

# Number of folds for cross-validation.
N_FOLDS = 5

# Hour bins used to create the stratification key.
# Coarse binning (4 periods) keeps the number of strata manageable:
# Night (0-5), Morning (6-11), Afternoon (12-17), Evening (18-23).
# Combined with a coarse zone cluster this ensures train and test have
# representative coverage of all time periods and pickup areas.
HOUR_BINS   = [0, 6, 12, 18, 24]
HOUR_LABELS = ['night', 'morning', 'afternoon', 'evening']

# Number of zone clusters for stratification.
# PULocationID has ~60 unique Manhattan zones — too fine for stratify=.
# We bin them into this many roughly equal-sized groups.
N_ZONE_STRATA = 10

# =============================================================================
# MODEL FEATURES & TARGET
# =============================================================================

# Dispatch-time features only — everything known BEFORE the trip starts.
# trip_duration_min is intentionally excluded: it is only known after the trip
# ends and would cause target leakage if included as a model input.
MODEL_FEATURES = [
    'trip_distance',
    'hour',
    'day of week',
    'is weekend',
    'passenger_count',
    'PULocationID',
    'DOLocationID',
]

# Target variable to predict.
TARGET = 'fare_amount'

# =============================================================================
# FEATURE TRANSFORMATIONS
# =============================================================================
# Used by model_utils.py to build a ColumnTransformer pipeline per model.
# Each group receives a different transformation strategy:
#
# LOG_FEATURES     : right-skewed continuous features.
#                    Transformation: log1p -> StandardScaler.
#                    Why: reduces the effect of extreme values on linear models.
#
# CYCLIC_FEATURES  : features that wrap around (hour 23 is close to hour 0).
#                    Transformation: sin/cos encoding.
#                    Why: a plain integer treats hour 0 and 23 as far apart.
#
# PASSTHROUGH_FEATURES : binary flags and zone IDs passed as-is.
#                    Tree models (RF, XGBoost) do not need scaling.
#                    Linear regression receives these unscaled intentionally —
#                    zone IDs will be target-encoded separately in model_utils.
#
# The cyclic period for each feature is the natural cycle length.
CYCLIC_FEATURES = {
    'hour':       24,   # repeats every 24 hours
    'day of week': 7,   # repeats every 7 days
}

LOG_FEATURES = [
    'trip_distance',
    'passenger_count',
]

PASSTHROUGH_FEATURES = [
    'is weekend',
    'PULocationID',
    'DOLocationID',
]

# =============================================================================
# EDA — FEATURE ANALYSIS
# =============================================================================

# Features to check for quasi-constant behaviour.
# Columns whose most frequent value exceeds QUASI_CONSTANT_THRESHOLD
# are dropped from correlation and PCA analysis.
QUASI_CONSTANT_FEATURES = [
    'passenger_count',
    'trip_distance',
    'fare_amount',
    'extra',
    'mta_tax',
    'improvement_surcharge',
    'congestion_surcharge',
    'cbd_congestion_fee',
    'hour',
    'day of week',
    'trip_duration_min',
]

QUASI_CONSTANT_THRESHOLD = 0.98

# =============================================================================
# EDA — VISUALIZATIONS
# =============================================================================

# Distance bins and labels used in multiple plots.
DIST_BINS   = [0, 1, 2, 5, 10, 20, 50, 100]
DIST_LABELS = ['0-1 mi', '1-2 mi', '2-5 mi', '5-10 mi', '10-20 mi', '20-50 mi', '50+ mi']

# Day-of-week labels for the temporal heatmap (0 = Monday).
DAY_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Number of normal (inlier) trips to sample for the before/after outlier scatter.
# Outliers are sampled at 1% of this value to reflect the true contamination rate.
OUTLIER_SCATTER_SAMPLE_SIZE = 100000

# Color scale bounds for fare-per-mile in the outlier scatter plot.
FARE_PER_MILE_VMIN = 0
FARE_PER_MILE_VMAX = 20