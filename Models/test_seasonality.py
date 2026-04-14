# =============================================================================
# test_seasonality.py
#
# PURPOSE:
# This script tests the temporal generalizability of the models trained on 
# January data by evaluating them on unseen July data. 
#
# METHODOLOGY:
# To prevent data leakage, we must strictly use the mappings (Target Encoding, 
# Spatial Clusters, and Route Speeds) calculated from the January dataset and 
# apply them to the July dataset. We then load the saved models and evaluate 
# their performance to check for seasonal concept drift.
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

import config
from preprocessing import load_and_clean
from metrics_evaluating import calculate_metrics

# Optional: Reuse the plotting function from final_result if available
try:
    from final_result import plot_results
    CAN_PLOT = True
except ImportError:
    CAN_PLOT = False

warnings.filterwarnings("ignore")

def extract_january_mappings():
    """
    Recreates the January preprocessing state to extract the mapping dictionaries.
    This ensures we apply January's knowledge to July's data, mimicking a 
    real-world production environment where future data is unknown.
    """
    print("[INFO] Loading January data to extract historical mappings...")
    df_jan = load_and_clean(config.FILE_PATH, config.ZONE_URL)
    
    # 1. Remove outliers (exactly as done during training)
    clf = IsolationForest(contamination=0.01, random_state=42).fit(df_jan[['fare_amount', 'trip_distance']])
    df_jan = df_jan[clf.predict(df_jan[['fare_amount', 'trip_distance']]) == 1]
    
    # 2. Extract Target Encoding Maps
    pu_map = df_jan.groupby('PULocationID')['fare_amount'].mean().to_dict()
    do_map = df_jan.groupby('DOLocationID')['fare_amount'].mean().to_dict()
    global_mean_fare = df_jan['fare_amount'].mean()
    
    # 3. Extract Spatial Clustering Map
    zone_stats = df_jan.groupby('PULocationID').agg({'fare_amount': 'mean', 'trip_distance': 'mean'}).reset_index()
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    zone_stats['pickup_cluster'] = kmeans.fit_predict(zone_stats[['fare_amount', 'trip_distance']])
    cluster_map = zone_stats.set_index('PULocationID')['pickup_cluster'].to_dict()
    
    # 4. Extract Route Speed Maps
    duration_h = (df_jan['tpep_dropoff_datetime'] - df_jan['tpep_pickup_datetime']).dt.total_seconds() / 3600
    df_jan['speed_mph'] = (df_jan['trip_distance'] / duration_h.replace(0, np.nan)).clip(upper=100)
    
    route_stats = df_jan.groupby(['PULocationID', 'DOLocationID', 'hour']).agg(
        route_avg_speed=('speed_mph', 'median'),
        route_popularity=('speed_mph', 'count')
    ).reset_index()
    
    backup_speed_map = df_jan.groupby(['PULocationID', 'hour'])['speed_mph'].median().to_dict()
    global_median_speed = df_jan['speed_mph'].median()
    
    return pu_map, do_map, global_mean_fare, cluster_map, route_stats, backup_speed_map, global_median_speed

def prepare_july_data(mappings):
    """
    Loads the July dataset and applies the January historical mappings to it.
    """
    pu_map, do_map, global_mean_fare, cluster_map, route_stats, backup_speed_map, global_median_speed = mappings
    
    july_path = config.FILE_PATH.replace("2025-01", "2025-07")
    if not os.path.exists(july_path):
        raise FileNotFoundError(f"Cannot find July data at: {july_path}")
        
    print(f"\n[INFO] Loading and cleaning July data ({july_path})...")
    df_jul = load_and_clean(july_path, config.ZONE_URL)
    
    # Optional: Filter July outliers so we are testing on normal data
    clf = IsolationForest(contamination=0.01, random_state=42).fit(df_jul[['fare_amount', 'trip_distance']])
    df_jul = df_jul[clf.predict(df_jul[['fare_amount', 'trip_distance']]) == 1]
    
    print("[INFO] Applying January Target Encodings to July data...")
    df_jul['PU_fare_avg'] = df_jul['PULocationID'].map(pu_map).fillna(global_mean_fare)
    df_jul['DO_fare_avg'] = df_jul['DOLocationID'].map(do_map).fillna(global_mean_fare)
    
    print("[INFO] Applying January Clusters to July data...")
    df_jul['pickup_cluster'] = df_jul['PULocationID'].map(cluster_map).fillna(-1)
    
    print("[INFO] Applying January Route Speeds to July data...")
    df_jul = df_jul.merge(route_stats, on=['PULocationID', 'DOLocationID', 'hour'], how='left')
    mask = df_jul['route_avg_speed'].isna()
    df_jul.loc[mask, 'route_avg_speed'] = df_jul.loc[mask].set_index(['PULocationID', 'hour']).index.map(backup_speed_map)
    df_jul['route_avg_speed'] = df_jul['route_avg_speed'].fillna(global_median_speed)
    df_jul['route_popularity'] = df_jul['route_popularity'].fillna(0)
    
    X_jul = df_jul[config.MODEL_FEATURES]
    y_jul = df_jul[config.TARGET]
    
    return X_jul, y_jul

def main():
    print("=== SEASONALITY GENERALIZATION TEST ===")
    
    # 1. Get mappings from January and apply to July
    mappings = extract_january_mappings()
    X_jul, y_jul = prepare_july_data(mappings)
    
    # 2. Load the trained models
    model_dir = os.path.join(config.PROJECT_ROOT, "Models")
    rf_path = os.path.join(model_dir, "deep_tuned_randomforest.joblib")
    xgb_path = os.path.join(model_dir, "deep_tuned_xgboost.joblib")
    
    if not os.path.exists(rf_path) or not os.path.exists(xgb_path):
        print("\n[ERROR] Trained models not found. Please run 'final_result.py' first.")
        return
        
    print("\n[INFO] Loading pre-trained January models from disk...")
    rf_pipeline = joblib.load(rf_path)
    xgb_pipeline = joblib.load(xgb_path)
    
    # 3. Predict and Evaluate
    print("\n=== PERFORMANCE ON JULY DATA ===")
    for model_name, pipeline in [("January RF Model", rf_pipeline), ("January XGBoost Model", xgb_pipeline)]:
        print(f"\nEvaluating {model_name}...")
        y_pred = pipeline.predict(X_jul)
        
        metrics = calculate_metrics(y_jul, y_pred, model_name=model_name)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            
        if CAN_PLOT:
            plot_results(y_jul, y_pred, model_name=f"{model_name} on July Data")

if __name__ == "__main__":
    main()