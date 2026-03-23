# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:45:42 2026

@author: Tibbe
"""

# config.py
import os

# Data Path - Update this to your local path
DATA_PATH = "yellow_tripdata_2025-01 (1).parquet"

# Cleaning Thresholds
MIN_FARE = 2.50
MAX_FARE = 200.0
MAX_DISTANCE = 50.0

# Features we are using and why:
# 1. trip_distance: Primary cost driver
# 2. pickup_hour: Captures rush hour surcharges/traffic
# 3. day_of_week: Weekend vs. Weekday patterns
# 4. PULocationID/DOLocationID: Area-specific pricing (e.g., Airport flats)
FEATURES = ['trip_distance', 'pickup_hour', 'day_of_week', 'PULocationID', 'DOLocationID']
TARGET = 'fare_amount'