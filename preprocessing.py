# =============================================================================
# preprocessing.py
# 
# PURPOSE:
# This script is the foundation of the machine learning pipeline. Raw NYC taxi 
# data is massive, messy, and heavily skewed towards short, cheap trips. 
# If we feed this directly into a model, it will perform poorly. 
# 
# Therefore, this script handles:
# 1. Cleaning: Removing GPS errors, negative fares, and invalid passenger counts.
# 2. Feature Engineering: Extracting time, holidays, and spatial clusters.
# 3. Leakage Prevention: Splitting the data *before* calculating historical 
#    averages (Target Encoding) to ensure the model doesn't "cheat" by looking
#    at the test set.
# 4. Balancing: Ensuring the model learns how to price expensive, rare trips 
#    just as well as standard short trips.
# =============================================================================

import os, joblib, warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import holidays
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
from sklearn.cluster import KMeans
import config

warnings.filterwarnings("ignore")

def load_and_clean(file_path, zone_url):
    """
    Loads raw parquet data, removes invalid entries (e.g., negative fares),
    generates time-based features, and filters trips to Manhattan only.
    """
    # Load data and drop rows with missing essential values
    df = pq.read_table(file_path).to_pandas()[config.KEEP_COLUMNS].dropna()
    
    # Ensure no negative regulatory fees are present
    for col in config.FEE_COLUMNS: 
        df = df[df[col] >= 0]
        
    # Filter out illogical trips (ghost cars or extreme distances/fares)
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    df = df[(df['trip_distance'] > 0) & (df['fare_amount'] > 0) & (df['fare_amount'] <= 50)]
    df = df[df['RatecodeID'] == 1] # Standard meter rate only
    
    # --- Core Datetime Extraction ---
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day of week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['is weekend'] = df['day of week'].isin([5, 6]).astype(int)
    
    # --- NYC Regulatory Flags ---
    # Night surcharges apply between 8 PM and 6 AM
    df['is_night_fare'] = ((df['hour'] >= 20) | (df['hour'] < 6)).astype(int)
    # Rush hour surcharges apply 4 PM to 8 PM on weekdays
    df['is_rush_hour'] = ((df['hour'] >= 16) & (df['hour'] < 20) & (df['day of week'] < 5)).astype(int)
    
    # Flag official NY state holidays
    ny_holidays = holidays.US(state='NY')
    df['is_holiday'] = df['tpep_pickup_datetime'].dt.date.apply(lambda x: x in ny_holidays).astype(int)
    
    # --- Borough filtering ---
    # Map raw Location IDs to their respective Borough names
    zone_df = pd.read_csv(zone_url)
    z_map = zone_df.set_index('LocationID')['Borough'].to_dict()
    df['PU_Boro'] = df['PULocationID'].map(z_map)
    df['DO_Boro'] = df['DOLocationID'].map(z_map)
    
    # Restrict scope to trips strictly within Manhattan
    return df[(df['PU_Boro'] == 'Manhattan') & (df['DO_Boro'] == 'Manhattan')]

def balance_data(df, target=30000):
    """
    Balances the dataset by upsampling minority fare classes (expensive trips).
    Adds slight Gaussian noise to trip_distance to prevent exact row duplication.
    """
    rng = np.random.default_rng(42)
    # Categorize fares into Low, Medium, High, and Premium bins
    df['bin'] = pd.cut(df[config.TARGET], bins=[0, 15, 25, 40, 51], labels=['L', 'M', 'H', 'P'])
    
    balanced = []
    for lab in ['L', 'M', 'H', 'P']:
        sub = df[df['bin'] == lab]
        if len(sub) == 0: continue
        
        # Upsample or downsample the bin to hit the target count
        res = resample(sub, replace=(len(sub) < target), n_samples=target, random_state=42)
        
        # If we upsampled (copied rows), add tiny noise to distance to help the model generalize
        if len(sub) < target:
            for c in ['trip_distance']:
                res[c] = (res[c] + rng.normal(0, sub[c].std() * 0.01, len(res))).clip(lower=sub[c].min())
        
        balanced.append(res)
        
    # Combine all balanced bins, shuffle them, and drop the temporary 'bin' column
    return pd.concat(balanced).sample(frac=1, random_state=42).drop(columns=['bin']).reset_index(drop=True)

def main(apply_balancing=True):
    """
    Orchestrates the entire preprocessing pipeline. Splits data, builds features
    based strictly on training data, balances the set, and caches the result.
    """
    path = os.path.join(config.PROCESSED_DATA_DIR, "balanced_data.joblib")
    
    # Load cached data if it exists to save time during repeated runs
    if os.path.exists(path): 
        print(f"[INFO] Loading existing preprocessed data from {path}...")
        return joblib.load(path)
    
    # Load and perform initial cleaning
    df = load_and_clean(config.FILE_PATH, config.ZONE_URL)

    # 1. Outlier removal: Drop top 1% of abnormal fare/distance combinations
    clf = IsolationForest(contamination=0.01, random_state=42).fit(df[['fare_amount', 'trip_distance']])
    df = df[clf.predict(df[['fare_amount', 'trip_distance']]) == 1]

    # 2. Split FIRST: Crucial step to prevent data leakage before Target Encoding
    train_df, test_df = train_test_split(df, test_size=config.TEST_SIZE, random_state=config.SPLIT_RANDOM_STATE)
    
    # === 3. FEATURE ENGINEERING: Target Encoding ===
    print("[INFO] Applying Target Encoding to Location IDs...")
    # Calculate historical averages exclusively on the training set
    pu_map = train_df.groupby('PULocationID')['fare_amount'].mean().to_dict()
    do_map = train_df.groupby('DOLocationID')['fare_amount'].mean().to_dict()
    
    # Apply mappings to the train set
    train_df['PU_fare_avg'] = train_df['PULocationID'].map(pu_map)
    train_df['DO_fare_avg'] = train_df['DOLocationID'].map(do_map)
    
    # Apply mappings to the test set; use global train mean as a fallback for unseen zones
    global_mean = train_df['fare_amount'].mean()
    test_df['PU_fare_avg'] = test_df['PULocationID'].map(pu_map).fillna(global_mean)
    test_df['DO_fare_avg'] = test_df['DOLocationID'].map(do_map).fillna(global_mean)
    
    # === 4. FEATURE ENGINEERING: Spatial Clustering ===
    print("[INFO] Clustering taxi zones...")
    # Group zones with similar average fares and distances
    zone_stats = train_df.groupby('PULocationID').agg({'fare_amount': 'mean', 'trip_distance': 'mean'}).reset_index()
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    zone_stats['pickup_cluster'] = kmeans.fit_predict(zone_stats[['fare_amount', 'trip_distance']])
    cluster_map = zone_stats.set_index('PULocationID')['pickup_cluster'].to_dict()
    
    train_df['pickup_cluster'] = train_df['PULocationID'].map(cluster_map).fillna(-1)
    test_df['pickup_cluster'] = test_df['PULocationID'].map(cluster_map).fillna(-1)

    # === 5. FEATURE ENGINEERING: Route Stats (Speed & Popularity) ===
    print("[INFO] Calculating historical route metrics...")
    # Calculate speed to find slow vs fast routes
    duration_h = (train_df['tpep_dropoff_datetime'] - train_df['tpep_pickup_datetime']).dt.total_seconds() / 3600
    train_df['speed_mph'] = (train_df['trip_distance'] / duration_h.replace(0, np.nan)).clip(upper=100)
    
    # Aggregate historical route data
    route_stats = train_df.groupby(['PULocationID', 'DOLocationID', 'hour']).agg(
        route_avg_speed=('speed_mph', 'median'),
        route_popularity=('speed_mph', 'count')
    ).reset_index()
    
    # Backup mappings for missing combinations
    backup_speed_map = train_df.groupby(['PULocationID', 'hour'])['speed_mph'].median().to_dict()
    global_median = train_df['speed_mph'].median()

    def add_route_features(target_df):
        """Helper to safely merge route statistics onto train/test sets."""
        target_df = target_df.merge(route_stats, on=['PULocationID', 'DOLocationID', 'hour'], how='left')
        
        # Fallbacks for routes the model has never seen before
        mask = target_df['route_avg_speed'].isna()
        target_df.loc[mask, 'route_avg_speed'] = target_df.loc[mask].set_index(['PULocationID', 'hour']).index.map(backup_speed_map)
        target_df['route_avg_speed'] = target_df['route_avg_speed'].fillna(global_median)
        target_df['route_popularity'] = target_df['route_popularity'].fillna(0)
        
        return target_df

    train_df = add_route_features(train_df)
    test_df = add_route_features(test_df)

    # Prepare final feature matrices
    X_train, y_train = train_df[config.MODEL_FEATURES], train_df[config.TARGET]
    X_test, y_test = test_df[config.MODEL_FEATURES], test_df[config.TARGET]

    # Balance the training data to ensure robust model performance across all fare ranges
    if apply_balancing:
        train_combined = pd.concat([X_train, y_train], axis=1)
        train_balanced = balance_data(train_combined)
        X_train, y_train = train_balanced[config.MODEL_FEATURES], train_balanced[config.TARGET]

    # Setup cross-validation strategy for tuning downstream
    kfold = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SPLIT_RANDOM_STATE)
    
    # Package and save the final cleaned and split data
    out = (X_train, X_test, y_train, y_test, kfold)
    joblib.dump(out, path)
    return out