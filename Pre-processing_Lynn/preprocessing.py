"""
preprocessing.py
----------------
Data loading, cleaning, feature engineering, outlier detection,
stratified train/test split, and cross-validation setup for the
NYC Taxi Fare Prediction project.

All configuration values are imported from config.py.
This file contains no hardcoded paths, thresholds, or magic numbers.

Returns X_train, X_test, y_train, y_test, and a KFold object
for use in model training, plus intermediate dataframes for eda.py.
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import warnings
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, KFold

from config import (
    FILE_PATH, ZONE_URL,
    KEEP_COLUMNS, FEE_COLUMNS,
    PASSENGER_MIN, PASSENGER_MAX,
    MAX_TRIP_DISTANCE_MILES, VALID_RATECODE,
    ISOLATION_FOREST_CONTAMINATION, ISOLATION_FOREST_RANDOM_STATE,
    TEST_SIZE, SPLIT_RANDOM_STATE,
    N_FOLDS,
    HOUR_BINS, HOUR_LABELS, N_ZONE_STRATA,
    MODEL_FEATURES, TARGET,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# 1. CLEANING & FEATURE ENGINEERING
# =============================================================================

def standard_filtering(df):
    """
    Basic data integrity cleaning.
    - Keeps only relevant columns and drops nulls.
    - Removes negative fees, zero/negative distances and fares.
    - Restricts passenger count to configured min/max.
    - Drops invalid RatecodeID 99.
    """
    df = df[KEEP_COLUMNS].dropna().copy()

    for col in FEE_COLUMNS:
        df = df[df[col] >= 0]

    df = df[(df['passenger_count'] >= PASSENGER_MIN) & (df['passenger_count'] <= PASSENGER_MAX)]
    df = df[(df['trip_distance'] > 0) & (df['fare_amount'] > 0)]
    df = df[df['RatecodeID'] != 99]

    return df


def extract_time_features(df):
    """
    Extracts temporal features from pickup/dropoff datetimes.
    - hour, day of week, is weekend: capture demand and pricing patterns.
    - trip_duration_min: used for outlier detection only, not model input.
    """
    df = df.copy()
    df['tpep_pickup_datetime']  = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    df['hour']        = df['tpep_pickup_datetime'].dt.hour
    df['day of week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['is weekend']  = df['day of week'].isin([5, 6]).astype(int)

    df['trip_duration_min'] = (
        df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    ).dt.total_seconds() / 60
    df = df[df['trip_duration_min'] > 0]

    return df


def custom_filtering(df):
    """
    Scope filter: Standard Rate Manhattan-to-Manhattan trips only.
    Removes physically impossible distances above the configured maximum.
    """
    df = df[df['RatecodeID'] == VALID_RATECODE].copy()
    df = df[(df['PU_Borough'] == 'Manhattan') & (df['DO_Borough'] == 'Manhattan')].copy()
    df = df[df['trip_distance'] < MAX_TRIP_DISTANCE_MILES]
    return df


def load_and_filter(file_path, zone_lookup_url):
    """
    Loads raw parquet, applies all filters, and adds zone/borough info.

    Returns:
    - df_step1          : all NYC trips after integrity checks (used in EDA scope plots)
    - df_manhattan      : Manhattan-only scoped trips
    - zone_map_manhattan: dict mapping LocationID to Zone name for Manhattan zones
    """
    df_raw   = pq.read_table(file_path).to_pandas()
    df_step1 = standard_filtering(df_raw)
    df_step1 = extract_time_features(df_step1)

    try:
        zone_df       = pd.read_csv(zone_lookup_url)
        zone_map_boro = zone_df.set_index('LocationID')['Borough'].to_dict()
        df_step1['PU_Borough'] = df_step1['PULocationID'].map(zone_map_boro)
        df_step1['DO_Borough'] = df_step1['DOLocationID'].map(zone_map_boro)
        zone_map_manhattan = (
            zone_df[zone_df['Borough'] == 'Manhattan']
            .set_index('LocationID')['Zone'].to_dict()
        )
    except Exception as e:
        print(f"Error loading zones: {e}")
        zone_map_manhattan = {}

    df_manhattan = custom_filtering(df_step1)
    return df_step1, df_manhattan, zone_map_manhattan


# =============================================================================
# 2. OUTLIER DETECTION
# =============================================================================

def run_outlier_detection(df_manhattan):
    """
    Isolation Forest on fare, distance, and fare-per-mile.
    Log transform reduces skewness before scaling.
    Contamination rate and random state are set in config.py.
    Returns full dataframe with 'is outlier' column (True = flagged for removal).
    """
    df_work = df_manhattan.copy()
    df_work['fare per mile'] = df_work['fare_amount'] / df_work['trip_distance']

    feature_cols = ['fare_amount', 'trip_distance', 'fare per mile']
    log_trans    = FunctionTransformer(np.log1p)
    scaled       = StandardScaler().fit_transform(
        log_trans.transform(df_work[feature_cols])
    )

    clf = IsolationForest(
        contamination=ISOLATION_FOREST_CONTAMINATION,
        random_state=ISOLATION_FOREST_RANDOM_STATE,
        n_jobs=-1,
    )
    df_work['is outlier'] = clf.fit_predict(scaled) == -1

    n_out = df_work['is outlier'].sum()
    print(f"\n--- OUTLIER DETECTION ---")
    print(f"  Flagged: {n_out:,} trips ({n_out / len(df_work) * 100:.2f}%) as outliers and removed.")

    return df_work


# =============================================================================
# 3. STRATIFIED TRAIN / TEST SPLIT
# =============================================================================

def _make_strata_key(df):
    """
    Creates a combined stratification key from hour period, zone cluster,
    and weekday/weekend flag.

    - Hour period : ensures all time-of-day patterns are proportionally
                    represented in train and test.
    - Zone cluster: ensures all pickup areas appear in both sets.
    - Day type    : ensures weekday and weekend patterns are both represented.

    Coarse binning keeps the number of unique strata manageable.
    Too many unique strata causes train_test_split to fail on small groups.
    """
    # Bin hours into 4 time periods (night / morning / afternoon / evening)
    hour_period = pd.cut(
        df['hour'], bins=HOUR_BINS, labels=HOUR_LABELS, right=False
    )

    # Bin zone IDs into N_ZONE_STRATA roughly equal clusters using rank-based binning
    zone_cluster = pd.qcut(
        df['PULocationID'], q=N_ZONE_STRATA, labels=False, duplicates='drop'
    )

    # Weekday vs weekend flag
    dag_type = df['is weekend'].map({0: 'weekday', 1: 'weekend'})

    # Combine into a single string key, e.g. "morning_3_weekday"
    strata_key = (
        hour_period.astype(str) + '_' +
        zone_cluster.astype(str) + '_' +
        dag_type.astype(str)
    )

    return strata_key


def make_train_test(df_clean):
    """
    Stratified train/test split on hour period x zone cluster.

    Stratification ensures that the distribution of time periods and pickup
    zones is proportionally identical in train and test — important because
    zone and hour are strong fare predictors and unevenly distributed.

    Also returns a KFold object for use during model cross-validation.
    The KFold is not applied here — splitting happens inside model training
    to avoid any data leakage from validation folds into preprocessing.

    Returns: X_train, X_test, y_train, y_test, kfold
    """
    df_model = df_clean[MODEL_FEATURES + [TARGET]].dropna().copy()

    X = df_model[MODEL_FEATURES]
    y = df_model[TARGET]

    strata = _make_strata_key(df_model)

    # Drop strata groups that are too small for splitting (rare edge cases)
    min_stratum_size = int(1 / TEST_SIZE) + 1
    valid_strata     = strata[strata.map(strata.value_counts()) >= min_stratum_size]
    X = X.loc[valid_strata.index]
    y = y.loc[valid_strata.index]
    strata = strata.loc[valid_strata.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=strata,
    )

    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_RANDOM_STATE)

    print(f"\n--- STRATIFIED TRAIN/TEST SPLIT ---")
    print(f"  Stratified on : hour period x zone cluster x weekday/weekend")
    print(f"  Train         : {len(X_train):,} trips ({(1 - TEST_SIZE) * 100:.0f}%)")
    print(f"  Test          : {len(X_test):,} trips ({TEST_SIZE * 100:.0f}%)")
    print(f"  CV folds      : {N_FOLDS}-fold KFold (shuffle=True)")
    print(f"  Features      : {MODEL_FEATURES}")
    print(f"  Target        : {TARGET}")

    return X_train, X_test, y_train, y_test, kfold


# =============================================================================
# 4. MAIN
# =============================================================================

def main(file_path=FILE_PATH, zone_url=ZONE_URL):
    """
    Full preprocessing pipeline.

    Returns:
    - X_train, X_test, y_train, y_test : ready for model training
    - kfold                             : KFold object for cross-validation
    - df_step1, df_manhattan, df_work   : intermediate data for EDA
    - df_clean                          : final clean dataset for EDA
    - zone_map_manhattan                : zone name lookup for map plots
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    print("Loading and filtering data...")
    df_step1, df_manhattan, zone_map_manhattan = load_and_filter(file_path, zone_url)

    print("Detecting outliers...")
    df_work  = run_outlier_detection(df_manhattan)
    df_clean = df_work[df_work['is outlier'] == False].copy()
    df_clean['fare per mile'] = df_clean['fare_amount'] / df_clean['trip_distance']

    print("Creating stratified train/test split...")
    X_train, X_test, y_train, y_test, kfold = make_train_test(df_clean)

    print("Preprocessing complete. Ready for model training!")

    return (
        X_train, X_test, y_train, y_test, kfold,
        df_step1, df_manhattan, df_work, df_clean,
        zone_map_manhattan,
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print(f"Looking for data at: {FILE_PATH}")
    result = main()
    if result:
        X_train, X_test, y_train, y_test, kfold, *_ = result
        print(f"\nX_train shape : {X_train.shape}")
        print(f"X_test shape  : {X_test.shape}")
        print(f"KFold         : {kfold}")