# =============================================================================
# EDA.py (Exploratory Data Analysis)
#
# PURPOSE:
# This script generates the visual and statistical proof needed to justify our 
# preprocessing decisions in the final report. 
#
# ARCHITECTURE NOTE:
# This script loads and processes its own data rather than importing the ML 
# preprocessing pipeline. This ensures we can visualize intermediate states 
# (like tracking outliers before they are removed) without cluttering or 
# breaking the heavily optimized model training scripts.
# =============================================================================

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from config import (
    FILE_PATH, ZONE_URL, KEEP_COLUMNS, FEE_COLUMNS,
    MODEL_FEATURES,
    QUASI_CONSTANT_FEATURES, QUASI_CONSTANT_THRESHOLD,
    DIST_BINS, DIST_LABELS, DAY_LABELS,
    OUTLIER_SCATTER_SAMPLE_SIZE,
    FARE_PER_MILE_VMIN, FARE_PER_MILE_VMAX,
    SHAPEFILE_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# 0. DEDICATED EDA DATA LOADING
# =============================================================================

def load_eda_data():
    """
    Loads and processes data specifically for visualization purposes.
    Unlike the ML pipeline, this keeps track of intermediate states (like 
    before/after outlier removal) so we can plot the differences.
    """
    print("[INFO] Loading raw data for EDA...")
    df = pq.read_table(FILE_PATH).to_pandas()[KEEP_COLUMNS].dropna()
    
    # Basic logical cleaning
    for col in FEE_COLUMNS: 
        df = df[df[col] >= 0]
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    df = df[(df['trip_distance'] > 0) & (df['fare_amount'] > 0) & (df['fare_amount'] <= 50)]
    df = df[df['RatecodeID'] == 1]
    
    # Time features
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day of week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['is weekend'] = df['day of week'].isin([5, 6]).astype(int)
    df['trip_duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Borough mapping
    zone_df = pd.read_csv(ZONE_URL)
    z_map = zone_df.set_index('LocationID')['Borough'].to_dict()
    zone_map_manhattan = zone_df[zone_df['Borough'] == 'Manhattan'].set_index('LocationID')['Zone'].to_dict()
    
    df['PU_Borough'] = df['PULocationID'].map(z_map)
    df['DO_Borough'] = df['DOLocationID'].map(z_map)
    
    df_step1 = df.copy() # All NYC data
    
    # Filter to Manhattan
    print("[INFO] Filtering to Manhattan...")
    df_manhattan = df[(df['PU_Borough'] == 'Manhattan') & (df['DO_Borough'] == 'Manhattan')].copy()
    
    # Outlier Detection (Tagging instead of dropping immediately)
    print("[INFO] Tagging outliers with Isolation Forest...")
    clf = IsolationForest(contamination=0.01, random_state=42).fit(df_manhattan[['fare_amount', 'trip_distance']])
    df_manhattan['is outlier'] = clf.predict(df_manhattan[['fare_amount', 'trip_distance']]) == -1
    
    df_work = df_manhattan.copy() # Contains the 'is outlier' flags for plotting
    df_clean = df_manhattan[df_manhattan['is outlier'] == False].copy()
    
    # Calculate target encoded fare for PCA diagnostics
    pu_map = df_clean.groupby('PULocationID')['fare_amount'].mean().to_dict()
    df_clean['PU_fare_avg'] = df_clean['PULocationID'].map(pu_map)
    
    # Distance bins for EDA
    df_clean['distance bin'] = pd.cut(df_clean['trip_distance'], bins=DIST_BINS, labels=DIST_LABELS)
    df_clean['fare per mile'] = df_clean['fare_amount'] / df_clean['trip_distance']
    
    return df_step1, df_manhattan, df_work, df_clean, zone_map_manhattan

# =============================================================================
# 1. FEATURE ANALYSIS
# =============================================================================

def get_active_features(df):
    """Identifies and returns features that contain meaningful variance."""
    active_features = []
    print(f"\n--- QUASI-CONSTANT ANALYSIS (Threshold: {QUASI_CONSTANT_THRESHOLD * 100}%) ---")
    
    for col in QUASI_CONSTANT_FEATURES:
        if col in df.columns:
            most_freq_perc = df[col].value_counts(normalize=True).iloc[0]
            is_quasi = most_freq_perc >= QUASI_CONSTANT_THRESHOLD
            status   = "DROP (quasi-constant)" if is_quasi else "KEEP"
            
            print(f"  '{col:25}': top value = {most_freq_perc * 100:6.2f}%  ->  {status}")
            if not is_quasi:
                active_features.append(col)
                
    print(f"\n  Active features kept: {active_features}")
    return active_features

def run_diagnostics(df, active_features):
    """Generates a correlation heatmap for the active features."""
    corr_df = df[active_features].copy()
    
    if 'fare_amount' in corr_df.columns:
        corr_df = corr_df.drop(columns=['fare_amount'])

    corr_df.columns = [c.replace('_', ' ') for c in corr_df.columns]

    plt.figure(figsize=(11, 9))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.4)
    plt.title("Correlation Matrix — Features Only (fare_amount excluded as target)")
    plt.tight_layout()
    plt.show()

def run_pca_99(df, active_features):
    """Performs PCA strictly for diagnostic insight (scree plot & biplot)."""
    features    = df[active_features].copy()
    clean_names = [c.replace('_', ' ') for c in features.columns]
    scaled_data = StandardScaler().fit_transform(features)

    pca_full    = PCA(n_components=0.99)
    pca_full.fit(scaled_data)
    n_components = len(pca_full.explained_variance_ratio_)

    # Scree plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, n_components + 1), pca_full.explained_variance_ratio_, alpha=0.7, color='teal', label='Individual variance')
    plt.step(range(1, n_components + 1), np.cumsum(pca_full.explained_variance_ratio_), where='mid', color='red', label='Cumulative variance')
    plt.axhline(0.99, color='gray', linestyle='--', alpha=0.5, label='99% threshold')
    plt.title(f'PCA Scree Plot — {n_components} components explain 99% variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================================================
# 2. VISUALIZATIONS
# =============================================================================

def plot_manhattan_heatmap(df_clean):
    """Creates a geographic choropleth map of Manhattan taxi zones."""
    try:
        import geopandas as gpd
    except ImportError:
        print("  [Map skipped] Install geopandas: pip install geopandas")
        return

    if not os.path.exists(SHAPEFILE_DIR):
        print(f"  [Map skipped] Folder '{SHAPEFILE_DIR}/' not found.")
        return

    shp_files = [f for f in os.listdir(SHAPEFILE_DIR) if f.endswith(".shp")]
    if not shp_files:
        return

    shp_path = os.path.join(SHAPEFILE_DIR, shp_files[0])
    gdf = gpd.read_file(shp_path)

    zone_fare          = df_clean.groupby('PULocationID')['fare_amount'].mean().reset_index()
    zone_fare.columns  = ['LocationID', 'mean_fare']
    zone_count         = df_clean.groupby('PULocationID').size().reset_index(name='trip_count')

    id_col   = 'location_i' if 'location_i' in gdf.columns else 'LocationID'
    boro_col = 'borough'    if 'borough'    in gdf.columns else 'boro_name'
    gdf[id_col] = gdf[id_col].astype(int)

    gdf           = gdf.merge(zone_fare,  left_on=id_col, right_on='LocationID',  how='left')
    gdf           = gdf.merge(zone_count, left_on=id_col, right_on='PULocationID', how='left')
    manhattan_gdf = gdf[gdf[boro_col].str.lower() == 'manhattan'].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    manhattan_gdf.plot(column='mean_fare', ax=axes[0], cmap='YlOrRd',
                       legend=True, missing_kwds={'color': 'lightgrey'},
                       legend_kwds={'label': 'Avg Fare ($)', 'shrink': 0.6})
    axes[0].set_title("Average Fare per Pickup Zone ($)", fontsize=11)
    axes[0].set_axis_off()

    manhattan_gdf.plot(column='trip_count', ax=axes[1], cmap='Blues',
                       legend=True, missing_kwds={'color': 'lightgrey'},
                       legend_kwds={'label': 'Number of Trips', 'shrink': 0.6})
    axes[1].set_title("Pickup Volume per Zone", fontsize=11)
    axes[1].set_axis_off()

    plt.suptitle("Manhattan Taxi Zone Heatmaps", fontsize=14)
    plt.tight_layout()
    plt.show()

def run_visuals(df_step1, df_manhattan, df_work, df_clean, zone_map_manhattan):
    """Executes the suite of EDA visualizations in logical report order."""

    # -------------------------------------------------------------------------
    # 1. LocationID Volume (All NYC)
    # WHY: To visually demonstrate the extreme spatial imbalance in the dataset.
    # WHAT: A bar chart showing trip counts per zone, highlighting the peaks and valleys.
    # -------------------------------------------------------------------------
    for loc_col, label in [('PULocationID', 'Pickup'), ('DOLocationID', 'Dropoff')]:
        loc_counts = df_step1[loc_col].value_counts().sort_index()
        colors = ['green' if v == loc_counts.max() else 'red' if v == loc_counts.min() else 'skyblue' for v in loc_counts]
        plt.figure(figsize=(15, 4))
        plt.bar(loc_counts.index.astype(str), loc_counts.values, color=colors)
        plt.title(f"{label} Volume by LocationID — Full NYC (shape shows zone imbalance)")
        plt.xticks([])
        plt.xlabel("Zone ID (labels omitted — 260+ zones)")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # 2. Borough Skewness (All NYC)
    # WHY: To mathematically justify our decision to filter the dataset strictly to Manhattan.
    # WHAT: A bar chart proving that Manhattan handles the vast majority of taxi traffic.
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    boro_counts = df_step1['PU_Borough'].value_counts()
    sns.barplot(x=boro_counts.index, y=boro_counts.values, palette='viridis')
    plt.title("Borough Skewness — Manhattan Dominates Pickups (All NYC)")
    plt.ylabel("Number of Trips")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 3. Manhattan Zone Demand
    # WHY: To identify the specific neighborhoods driving the most revenue and traffic.
    # WHAT: Horizontal bar charts of the top 20 and bottom 20 most popular pickup zones.
    # -------------------------------------------------------------------------
    pu_counts = df_clean['PULocationID'].value_counts().rename(index=zone_map_manhattan)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    pu_counts.head(20).plot(kind='barh', ax=axes[0], color='darkgreen')
    axes[0].set_title("Top 20 Manhattan Pickup Zones")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Number of Trips")
    
    pu_counts.tail(20).plot(kind='barh', ax=axes[1], color='firebrick')
    axes[1].set_title("Bottom 20 Manhattan Pickup Zones")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Number of Trips")
    plt.suptitle("Manhattan Zone Demand After Filtering", fontsize=13)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 4. Before/After Outlier Scatter
    # WHY: To visually prove that the IsolationForest algorithm successfully caught 
    #      and removed physically impossible trips (e.g., 0 duration but $100 fare).
    # WHAT: Scatter plot comparing Trip Duration vs Fare Amount, with outliers highlighted in red.
    # -------------------------------------------------------------------------
    outliers = df_work[df_work['is outlier'] == True]
    inliers  = df_work[df_work['is outlier'] == False]

    normal_sample_size  = min(OUTLIER_SCATTER_SAMPLE_SIZE, len(inliers))
    outlier_sample_size = min(max(1, int(normal_sample_size * 0.01)), len(outliers))

    inlier_sample  = inliers.sample(normal_sample_size, random_state=42).copy()
    outlier_sample = outliers.sample(outlier_sample_size, random_state=42).copy()
    
    inlier_sample['fare per mile']  = inlier_sample['fare_amount']  / inlier_sample['trip_distance']
    outlier_sample['fare per mile'] = outlier_sample['fare_amount'] / outlier_sample['trip_distance']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sc = axes[0].scatter(
        inlier_sample['trip_duration_min'], inlier_sample['fare_amount'],
        c=inlier_sample['fare per mile'].clip(FARE_PER_MILE_VMIN, FARE_PER_MILE_VMAX),
        cmap='YlGnBu', alpha=0.4, s=3, vmin=FARE_PER_MILE_VMIN, vmax=FARE_PER_MILE_VMAX, zorder=1
    )
    axes[0].scatter(
        outlier_sample['trip_duration_min'], outlier_sample['fare_amount'],
        color='red', alpha=0.7, s=3, marker='o', zorder=2,
        label=f'Outlier (~1% of trips, n={len(outliers):,} total)'
    )
    axes[0].set_xlim(0, 90); axes[0].set_ylim(0, 150)
    axes[0].set_title("Before: Outliers in Red (~1% shown proportionally)")
    axes[0].set_xlabel("Trip Duration (min)"); axes[0].set_ylabel("Fare Amount ($)")
    axes[0].legend(fontsize=9)
    plt.colorbar(sc, ax=axes[0], label='Fare per Mile ($)')

    final_sample = df_clean.sample(normal_sample_size, random_state=42)
    sc2 = axes[1].scatter(
        final_sample['trip_duration_min'], final_sample['fare_amount'],
        c=final_sample['fare per mile'].clip(FARE_PER_MILE_VMIN, FARE_PER_MILE_VMAX),
        cmap='YlGnBu', alpha=0.4, s=3, vmin=FARE_PER_MILE_VMIN, vmax=FARE_PER_MILE_VMAX
    )
    axes[1].set_xlim(0, 90); axes[1].set_ylim(0, 150)
    axes[1].set_title(f"After: {len(df_clean):,} Clean Trips")
    axes[1].set_xlabel("Trip Duration (min)"); axes[1].set_ylabel("Fare Amount ($)")
    plt.colorbar(sc2, ax=axes[1], label='Fare per Mile ($)')
    plt.suptitle("Isolation Forest — Before vs After Outlier Removal", fontsize=13)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 5. Fare Distribution
    # WHY: To understand the shape of our target variable. Because it is highly 
    #      right-skewed, this justifies why we needed data balancing in preprocessing.py.
    # WHAT: A histogram showing the frequency of different fare amounts.
    # -------------------------------------------------------------------------
    fare_max = df_clean['fare_amount'].max()
    fare_cap = min(100, np.ceil(fare_max / 5) * 5)
    cap_note = (f"view capped at ${fare_cap:.0f}" if fare_cap < fare_max else f"showing full range up to ${fare_max:.0f}")
    plt.figure(figsize=(12, 5))
    sns.histplot(df_clean['fare_amount'], bins=80, color='darkgreen', kde=True)
    plt.xlim(0, fare_cap)
    plt.title(f"Fare Amount Distribution — Manhattan (after outlier removal)\nMax fare: ${fare_max:.2f} | {cap_note}")
    plt.xlabel("Fare Amount ($)")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 6. Fare per Mile by Distance Bin
    # WHY: To prove the non-linear relationship between distance and fare. Short 
    #      trips have a high base fare, making them more expensive per mile.
    # WHAT: A bar chart displaying the median cost of 1 mile depending on the total trip length.
    # -------------------------------------------------------------------------
    df_clean_copy = df_clean.copy()
    df_clean_copy['distance bin'] = pd.cut(df_clean_copy['trip_distance'], bins=DIST_BINS, labels=DIST_LABELS)
    bin_stats = df_clean_copy.groupby('distance bin', observed=True)['fare per mile'].median().reset_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=bin_stats, x='distance bin', y='fare per mile', palette='YlOrRd')
    plt.title("Median Fare per Mile by Distance Bin\n(Short trips are disproportionately expensive per mile — base fare effect)")
    plt.ylabel("Median Fare per Mile ($)")
    plt.xlabel("Trip Distance Bin")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 7. Temporal Heatmap
    # WHY: To confirm that engineered time features (rush hour, night fare) contain real signal.
    # WHAT: A matrix showing how the average fare fluctuates depending on the hour and day.
    # -------------------------------------------------------------------------
    pivot = df_clean.groupby(['day of week', 'hour'])['fare_amount'].mean().unstack()
    pivot.index = DAY_LABELS
    plt.figure(figsize=(14, 5))
    sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.3, cbar_kws={'label': 'Avg Fare ($)'})
    plt.title("Average Fare by Hour x Day of Week — Manhattan\n(Darker = higher avg fare)")
    plt.xlabel("Hour of Day"); plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 8. Trip Distance Distribution
    # WHY: To show the dataset composition. If 90% of trips are under 5 miles, 
    #      the model will naturally be biased towards short trips without balancing.
    # WHAT: A countplot of trip frequencies categorized by distance bins.
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    sns.countplot(x=df_clean_copy['distance bin'], palette='viridis')
    plt.title("Trip Distance Distribution — Final Manhattan Dataset")
    plt.xlabel("Distance Bin")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 9. Fee Occurrence
    # WHY: To mathematically justify dropping quasi-constant columns. If a fee applies 
    #      to 99.9% of trips, it provides zero predictive variance to the model.
    # WHAT: A bar chart showing what percentage of trips include a specific NYC taxi fee.
    # -------------------------------------------------------------------------
    fee_occurrence = (df_clean[FEE_COLUMNS] > 0).mean() * 100
    plt.figure(figsize=(10, 5))
    sns.barplot(x=[c.replace('_', ' ') for c in fee_occurrence.index], y=fee_occurrence.values, palette='magma')
    plt.title("Percentage of Trips with Fee > $0\n(Near-100% fees are quasi-constant — dropped from modeling)")
    plt.ylabel("% of Trips"); plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 10. Boxplot Fare per Distance Bin
    # WHY: To show that even when distance is perfectly constant, fare varies greatly 
    #      (due to traffic/duration), highlighting why spatial/time features are vital.
    # WHAT: Boxplots showing the spread (IQR) of fares within strict distance boundaries.
    # -------------------------------------------------------------------------
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=df_clean_copy, x='distance bin', y='fare_amount', palette='YlOrRd', showfliers=False)
    plt.title("Fare Amount Distribution per Distance Bin\n(Box = IQR, line = median | fliers hidden — outliers already removed)")
    plt.xlabel("Trip Distance Bin"); plt.ylabel("Fare Amount ($)")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 11. Trip Volume per Hour
    # WHY: To show behavioral differences between weekday commutes and weekend nightlife, 
    #      proving that the 'is_weekend' feature is a critical predictor for the model.
    # WHAT: A dual line plot comparing total trip volume by hour on weekends vs weekdays.
    # -------------------------------------------------------------------------
    hourly = df_clean.groupby(['hour', 'is weekend']).size().reset_index(name='trip_count')
    hourly['day type'] = hourly['is weekend'].map({0: 'Weekday', 1: 'Weekend'})
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=hourly, x='hour', y='trip_count', hue='day type', marker='o', palette=['steelblue', 'darkorange'])
    plt.title("Trip Volume per Hour — Weekday vs Weekend\n(Shows where the model has most training data)")
    plt.xlabel("Hour of Day"); plt.ylabel("Number of Trips")
    plt.xticks(range(0, 24)); plt.legend(title='')
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 12. Manhattan Choropleth Map
    # WHY: To provide an intuitive, geographic visualization of where the highest 
    #      value and highest volume zones are located in the real world.
    # WHAT: Geographic heatmaps mapped onto the official NYC shapefiles.
    # -------------------------------------------------------------------------
    plot_manhattan_heatmap(df_clean)

# =============================================================================
# 3. STATISTICS
# =============================================================================

def print_statistics(df_step1, df_manhattan, df_clean):
    """Prints key dataset statistics for the final academic report."""
    final_bins = pd.cut(df_clean['trip_distance'], bins=DIST_BINS, labels=DIST_LABELS)

    print("\n" + "=" * 55)
    print("  FINAL STATISTICS FOR REPORT")
    print("=" * 55)
    print(f"  Records after integrity check (all NYC) : {len(df_step1):>10,}")
    print(f"  Records after Manhattan filter          : {len(df_manhattan):>10,}")
    print(f"  Records after outlier removal (final)   : {len(df_clean):>10,}")

    print("\n--- FARE AMOUNT (final dataset) ---")
    print(df_clean['fare_amount'].describe().round(2).to_string())
    print(f"  Median fare: ${df_clean['fare_amount'].median():.2f}")

    print("\n--- TRIP DURATION (final dataset) ---")
    print(df_clean['trip_duration_min'].describe().round(2).to_string())
    print(f"  Median duration: {df_clean['trip_duration_min'].median():.1f} min")

    print("\n--- TRIP DISTANCE BREAKDOWN ---")
    pct = (final_bins.value_counts(normalize=True) * 100).sort_index()
    for label, p in pct.items():
        print(f"  {label:10} : {p:.2f}%")

    print("\n--- BOROUGH REPRESENTATION (pre-filter, all NYC) ---")
    boro_pct = (df_step1['PU_Borough'].value_counts(normalize=True) * 100).map('{:.2f}%'.format)
    print(boro_pct.to_string())

    print("\n--- MODEL FEATURES ---")
    for f in MODEL_FEATURES:
        print(f"  - {f}")

# =============================================================================
# 4. MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Looking for data at: {FILE_PATH}")

    # Use the local EDA loader instead of the ML preprocessing script
    df_step1, df_manhattan, df_work, df_clean, zone_map_manhattan = load_eda_data()

    # Determine which features survived the quasi-constant check
    active_features = get_active_features(df_clean)
    
    # Add our target encoded feature for PCA analysis
    if 'PU_fare_avg' not in active_features:
        active_features.append('PU_fare_avg')

    run_diagnostics(df_clean, active_features)
    run_pca_99(df_clean, active_features)
    
    run_visuals(df_step1, df_manhattan, df_work, df_clean, zone_map_manhattan)
    print_statistics(df_step1, df_manhattan, df_clean)