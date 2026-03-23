# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:47:26 2026

@author: Tibbe
"""

# main.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
from Prepros import clean_and_engineer, get_dbscan_outliers

def run_analysis():
    print("--- Starting Preprocessing ---")
    raw_data = pd.read_parquet(config.DATA_PATH)
    
    df = clean_and_engineer(raw_data)
    print(f"Data cleaned. Remaining rows: {len(df)}")
    
    print("\n--- Running Feature Analysis (DBSCAN) ---")
    outlier_sample = get_dbscan_outliers(df)
    
    # Visualizing the Outliers
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=outlier_sample, x='trip_distance', y='fare_amount', 
                    hue='is_outlier', palette={True: 'red', False: 'dodgerblue'})
    plt.title("DBSCAN Outlier Detection: Fare vs. Distance")
    plt.show()
    
    # Rationale Summary
    print("\n--- Feature Selection Rationale ---")
    print(f"Features: {config.FEATURES}")
    print("- Cyclic Hour: Captured via Sin/Cos to preserve temporal continuity.")
    print("- Location IDs: Included to capture zone-based surcharges.")
    print("- Trip Distance: Kept as the primary linear predictor of fare.")

if __name__ == "__main__":
    run_analysis()