# =============================================================================
# preprocessing.py
# Handles data loading, cleaning, outlier removal, balancing, and splitting.
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
    df = pq.read_table(file_path).to_pandas()[config.KEEP_COLUMNS].dropna()
    for col in config.FEE_COLUMNS: 
        df = df[df[col] >= 0]
        
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    df = df[(df['trip_distance'] > 0) & (df['fare_amount'] > 0) & (df['fare_amount'] <= 50)]
    df = df[df['RatecodeID'] == 1]
    
    # Core Datetime
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day of week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['is weekend'] = df['day of week'].isin([5, 6]).astype(int)
    
    # === NYC Regulatory Flags ===
    df['is_night_fare'] = ((df['hour'] >= 20) | (df['hour'] < 6)).astype(int)
    df['is_rush_hour'] = ((df['hour'] >= 16) & (df['hour'] < 20) & (df['day of week'] < 5)).astype(int)
    
    ny_holidays = holidays.US(state='NY')
    df['is_holiday'] = df['tpep_pickup_datetime'].dt.date.apply(lambda x: x in ny_holidays).astype(int)
    
    # Borough filtering
    zone_df = pd.read_csv(zone_url)
    z_map = zone_df.set_index('LocationID')['Borough'].to_dict()
    df['PU_Boro'] = df['PULocationID'].map(z_map)
    df['DO_Boro'] = df['DOLocationID'].map(z_map)
    
    return df[(df['PU_Boro'] == 'Manhattan') & (df['DO_Boro'] == 'Manhattan')]

def balance_data(df, target=30000):
    rng = np.random.default_rng(42)
    df['bin'] = pd.cut(df[config.TARGET], bins=[0, 15, 25, 40, 51], labels=['L', 'M', 'H', 'P'])
    balanced = []
    for lab in ['L', 'M', 'H', 'P']:
        sub = df[df['bin'] == lab]
        if len(sub) == 0: continue
        res = resample(sub, replace=(len(sub) < target), n_samples=target, random_state=42)
        if len(sub) < target:
            for c in ['trip_distance', 'passenger_count']:
                res[c] = (res[c] + rng.normal(0, sub[c].std() * 0.01, len(res))).clip(lower=sub[c].min())
        balanced.append(res)
    return pd.concat(balanced).sample(frac=1, random_state=42).drop(columns=['bin']).reset_index(drop=True)

def main(apply_balancing=True):
    path = os.path.join(config.PROCESSED_DATA_DIR, "balanced_data.joblib")
    if os.path.exists(path): 
        print(f"[INFO] Loading existing preprocessed data from {path}...")
        return joblib.load(path)
    
    df = load_and_clean(config.FILE_PATH, config.ZONE_URL)

    # 1. Outlier removal
    clf = IsolationForest(contamination=0.01, random_state=42).fit(df[['fare_amount', 'trip_distance']])
    df = df[clf.predict(df[['fare_amount', 'trip_distance']]) == 1]

    # 2. Split FIRST
    train_df, test_df = train_test_split(df, test_size=config.TEST_SIZE, random_state=config.SPLIT_RANDOM_STATE)

    # === 3. FEATURE ENGINEERING: Spatial Clustering ===
    print("[INFO] Clustering taxi zones...")
    zone_stats = train_df.groupby('PULocationID').agg({'fare_amount': 'mean', 'trip_distance': 'mean'}).reset_index()
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    zone_stats['pickup_cluster'] = kmeans.fit_predict(zone_stats[['fare_amount', 'trip_distance']])
    cluster_map = zone_stats.set_index('PULocationID')['pickup_cluster'].to_dict()
    
    train_df['pickup_cluster'] = train_df['PULocationID'].map(cluster_map).fillna(-1)
    test_df['pickup_cluster'] = test_df['PULocationID'].map(cluster_map).fillna(-1)

    # === 4. FEATURE ENGINEERING: Route Stats (Speed & Popularity) ===
    print("[INFO] Calculating historical route metrics...")
    duration_h = (train_df['tpep_dropoff_datetime'] - train_df['tpep_pickup_datetime']).dt.total_seconds() / 3600
    train_df['speed_mph'] = (train_df['trip_distance'] / duration_h.replace(0, np.nan)).clip(upper=100)
    
    route_stats = train_df.groupby(['PULocationID', 'DOLocationID', 'hour']).agg(
        route_avg_speed=('speed_mph', 'median'),
        route_popularity=('speed_mph', 'count')
    ).reset_index()
    
    backup_speed_map = train_df.groupby(['PULocationID', 'hour'])['speed_mph'].median().to_dict()
    global_median = train_df['speed_mph'].median()

    def add_route_features(target_df):
        target_df = target_df.merge(route_stats, on=['PULocationID', 'DOLocationID', 'hour'], how='left')
        # Fill Speed
        mask = target_df['route_avg_speed'].isna()
        target_df.loc[mask, 'route_avg_speed'] = target_df.loc[mask].set_index(['PULocationID', 'hour']).index.map(backup_speed_map)
        target_df['route_avg_speed'] = target_df['route_avg_speed'].fillna(global_median)
        # Fill Popularity
        target_df['route_popularity'] = target_df['route_popularity'].fillna(0)
        return target_df

    train_df = add_route_features(train_df)
    test_df = add_route_features(test_df)

    X_train, y_train = train_df[config.MODEL_FEATURES], train_df[config.TARGET]
    X_test, y_test = test_df[config.MODEL_FEATURES], test_df[config.TARGET]

    if apply_balancing:
        train_combined = pd.concat([X_train, y_train], axis=1)
        train_balanced = balance_data(train_combined)
        X_train, y_train = train_balanced[config.MODEL_FEATURES], train_balanced[config.TARGET]

    kfold = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SPLIT_RANDOM_STATE)
    
    out = (X_train, X_test, y_train, y_test, kfold)
    joblib.dump(out, path)
    return out