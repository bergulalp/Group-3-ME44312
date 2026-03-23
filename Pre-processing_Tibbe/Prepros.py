# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:46:27 2026

@author: Tibbe
"""

# data_processor.py
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import config

def clean_and_engineer(df):
    # 1. Drop missing and logical errors
    df = df.dropna(subset=[config.TARGET, 'trip_distance', 'tpep_pickup_datetime'])
    df = df[(df['fare_amount'] >= config.MIN_FARE) & (df['fare_amount'] <= config.MAX_FARE)]
    df = df[(df['trip_distance'] > 0) & (df['trip_distance'] <= config.MAX_DISTANCE)]
    
    # 2. Time Features
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    
    # 3. Cyclic Encoding (Sophisticated touch!)
    # Tells the model that hour 23 is close to hour 0
    df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour']/24.0)
    
    return df

def get_dbscan_outliers(df, sample_size=5000):
    # DBSCAN is memory intensive, so we use a sample for the analysis
    sample = df.sample(sample_size, random_state=42).copy()
    
    # Analyze relationship between distance and fare
    X = sample[['trip_distance', 'fare_amount']]
    X_scaled = StandardScaler().fit_transform(X)
    
    # eps = max distance between points, min_samples = density
    db = DBSCAN(eps=0.3, min_samples=10).fit(X_scaled)
    sample['is_outlier'] = db.labels_ == -1
    
    return sample