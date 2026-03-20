"""
eda.py
------
Exploratory Data Analysis for the NYC Taxi Fare Prediction project.

Imports the preprocessing pipeline from preprocessing.py and runs all
visualizations and statistical summaries on the clean data. No data
transformation or model-related logic lives here.

All configuration values (paths, thresholds, plot settings) are
imported from config.py. This file contains no hardcoded values.

Run directly to produce all plots and print statistics:
    python eda.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from preprocessing import main as run_preprocessing
from config import (
    FILE_PATH, ZONE_URL,
    MODEL_FEATURES,
    QUASI_CONSTANT_FEATURES, QUASI_CONSTANT_THRESHOLD,
    DIST_BINS, DIST_LABELS, DAY_LABELS,
    OUTLIER_SCATTER_SAMPLE_SIZE,
    FARE_PER_MILE_VMIN, FARE_PER_MILE_VMAX,
    SHAPEFILE_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# 1. FEATURE ANALYSIS
# =============================================================================

def get_active_features(df):
    """
    Returns features that are not quasi-constant.
    Columns whose most frequent value exceeds QUASI_CONSTANT_THRESHOLD
    carry near-zero variance and are excluded from correlation and PCA analysis.
    """
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
    """
    Correlation heatmap of active features.
    PULocationID is added as a mean-encoded zone feature (average fare per pickup
    zone) to capture spatial signal without high-cardinality dummy columns.
    fare_amount is excluded — it is the prediction target, not a predictor.
    """
    corr_df = df[active_features].copy()

    zone_mean_fare = df.groupby('PULocationID')['fare_amount'].transform('mean')
    corr_df['pickup zone (mean fare)'] = zone_mean_fare

    if 'fare_amount' in corr_df.columns:
        corr_df = corr_df.drop(columns=['fare_amount'])

    corr_df.columns = [c.replace('_', ' ') for c in corr_df.columns]

    plt.figure(figsize=(11, 9))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.4)
    plt.title("Correlation Matrix — Features Only (fare_amount excluded as target)")
    plt.tight_layout()
    plt.show()


def run_pca_99(df, active_features):
    """
    PCA for dimensionality insight. Used as a diagnostic tool only,
    not as input to any model.

    Produces:
    - Scree plot: number of components needed to explain 99% of variance.
    - Loadings table printed to console: per-component feature weights.
    - Biplot: feature directions in PC1/PC2 space. Arrows pointing in the
      same direction indicate correlated features.
    """
    features    = df[active_features].copy()
    clean_names = [c.replace('_', ' ') for c in features.columns]
    scaled_data = StandardScaler().fit_transform(features)

    pca_full    = PCA(n_components=0.99)
    pca_full.fit(scaled_data)
    n_components = len(pca_full.explained_variance_ratio_)

    # Scree plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, n_components + 1), pca_full.explained_variance_ratio_,
            alpha=0.7, color='teal', label='Individual variance')
    plt.step(range(1, n_components + 1), np.cumsum(pca_full.explained_variance_ratio_),
             where='mid', color='red', label='Cumulative variance')
    plt.axhline(0.99, color='gray', linestyle='--', alpha=0.5, label='99% threshold')
    plt.title(f'PCA Scree Plot — {n_components} components explain 99% variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Loadings table
    print("\n--- PCA COMPONENT LOADINGS ---")
    loadings_df = pd.DataFrame(
        pca_full.components_,
        columns=clean_names,
        index=[f'PC{i + 1}' for i in range(n_components)]
    )
    print(f"\n{'':6}", end='')
    for name in clean_names:
        print(f"{name:>22}", end='')
    print()
    for pc in loadings_df.index:
        var_pct = pca_full.explained_variance_ratio_[loadings_df.index.get_loc(pc)] * 100
        print(f"{pc} ({var_pct:4.1f}%)", end='  ')
        for val in loadings_df.loc[pc]:
            bar  = 'X' * int(abs(val) * 10)
            sign = '+' if val >= 0 else '-'
            print(f"{sign}{abs(val):.2f} {bar:10}", end='  ')
        print()

    # Biplot
    pca2        = PCA(n_components=2)
    coords      = pca2.fit_transform(scaled_data)
    sample_idx  = np.random.choice(len(coords), size=min(3000, len(coords)), replace=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[sample_idx, 0], coords[sample_idx, 1],
               alpha=0.15, s=5, color='steelblue')

    loadings = pca2.components_.T
    scale    = 3
    for i, name in enumerate(clean_names):
        ax.annotate('', xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.8))
        ax.text(loadings[i, 0] * scale * 1.12, loadings[i, 1] * scale * 1.12,
                name, fontsize=9, color='darkred')

    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0] * 100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1] * 100:.1f}% variance)')
    ax.set_title('PCA Biplot — Feature Directions in PC1/PC2 Space')
    plt.tight_layout()
    plt.show()


# =============================================================================
# 2. VISUALIZATIONS
# =============================================================================

def plot_manhattan_heatmap(df_clean):
    """
    Choropleth map of Manhattan taxi zones shaded by average fare (left)
    and pickup volume (right). Requires geopandas to be installed.

    The shapefile must be downloaded and unzipped manually into the folder
    specified by SHAPEFILE_DIR in config.py.
    Download from: https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip
    """
    try:
        import geopandas as gpd
    except ImportError:
        print("  [Map skipped] Install geopandas: pip install geopandas")
        return

    if not os.path.exists(SHAPEFILE_DIR):
        print(f"  [Map skipped] Folder '{SHAPEFILE_DIR}/' not found.")
        print(f"  Download and unzip: https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip")
        return

    shp_files = [f for f in os.listdir(SHAPEFILE_DIR) if f.endswith(".shp")]
    if not shp_files:
        print(f"  [Map skipped] No .shp file found in '{SHAPEFILE_DIR}/'.")
        return

    shp_path = os.path.join(SHAPEFILE_DIR, shp_files[0])
    print(f"  Loading shapefile: {shp_path}")
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
    """
    All EDA visualizations in logical report order.

     1. LocationID volume bars — All NYC (shows zone imbalance)
     2. Borough skewness — All NYC (justifies Manhattan scope)
     3. Manhattan zone demand — Top 20 and Bottom 20 named zones
     4. Before/after outlier scatter — duration vs fare, colored by fare/mile
     5. Fare distribution — shape of the target variable
     6. Fare per mile by distance bin — base fare effect
     7. Temporal heatmap — average fare by hour and day of week
     8. Trip distance distribution — dataset composition
     9. Fee occurrence — justifies dropping quasi-constant fees
    10. Boxplot fare per distance bin — spread within each bin
    11. Trip volume per hour — weekday vs weekend demand pattern
    12. Manhattan choropleth map — spatial fare and volume heatmap
    """

    # ---- 1. LocationID volume — All NYC ----
    for loc_col, label in [('PULocationID', 'Pickup'), ('DOLocationID', 'Dropoff')]:
        loc_counts = df_step1[loc_col].value_counts().sort_index()
        colors = ['green' if v == loc_counts.max()
                  else 'red' if v == loc_counts.min()
                  else 'skyblue' for v in loc_counts]
        plt.figure(figsize=(15, 4))
        plt.bar(loc_counts.index.astype(str), loc_counts.values, color=colors)
        plt.title(f"{label} Volume by LocationID — Full NYC (shape shows zone imbalance)")
        plt.xticks([])
        plt.xlabel("Zone ID (labels omitted — 260+ zones)")
        plt.tight_layout()
        plt.show()

    # ---- 2. Borough skewness — All NYC ----
    plt.figure(figsize=(10, 5))
    boro_counts = df_step1['PU_Borough'].value_counts()
    sns.barplot(x=boro_counts.index, y=boro_counts.values, palette='viridis')
    plt.title("Borough Skewness — Manhattan Dominates Pickups (All NYC)")
    plt.ylabel("Number of Trips")
    plt.tight_layout()
    plt.show()

    # ---- 3. Manhattan zone demand ----
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

    # ---- 4. Before/After Outlier Scatter ----
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
        cmap='YlGnBu', alpha=0.4, s=3,
        vmin=FARE_PER_MILE_VMIN, vmax=FARE_PER_MILE_VMAX, zorder=1
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
        cmap='YlGnBu', alpha=0.4, s=3,
        vmin=FARE_PER_MILE_VMIN, vmax=FARE_PER_MILE_VMAX
    )
    axes[1].set_xlim(0, 90); axes[1].set_ylim(0, 150)
    axes[1].set_title(f"After: {len(df_clean):,} Clean Trips")
    axes[1].set_xlabel("Trip Duration (min)"); axes[1].set_ylabel("Fare Amount ($)")
    plt.colorbar(sc2, ax=axes[1], label='Fare per Mile ($)')
    plt.suptitle("Isolation Forest — Before vs After Outlier Removal", fontsize=13)
    plt.tight_layout()
    plt.show()

    # ---- 5. Fare distribution ----
    fare_max = df_clean['fare_amount'].max()
    fare_cap = min(100, np.ceil(fare_max / 5) * 5)
    cap_note = (f"view capped at ${fare_cap:.0f}"
                if fare_cap < fare_max else f"showing full range up to ${fare_max:.0f}")
    plt.figure(figsize=(12, 5))
    sns.histplot(df_clean['fare_amount'], bins=80, color='darkgreen', kde=True)
    plt.xlim(0, fare_cap)
    plt.title(f"Fare Amount Distribution — Manhattan (after outlier removal)\n"
              f"Max fare: ${fare_max:.2f} | {cap_note}")
    plt.xlabel("Fare Amount ($)")
    plt.tight_layout()
    plt.show()

    # ---- 6. Fare per mile by distance bin ----
    df_clean = df_clean.copy()
    df_clean['distance bin'] = pd.cut(df_clean['trip_distance'], bins=DIST_BINS, labels=DIST_LABELS)
    bin_stats = df_clean.groupby('distance bin', observed=True)['fare per mile'].median().reset_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=bin_stats, x='distance bin', y='fare per mile', palette='YlOrRd')
    plt.title("Median Fare per Mile by Distance Bin\n"
              "(Short trips are disproportionately expensive per mile — base fare effect)")
    plt.ylabel("Median Fare per Mile ($)")
    plt.xlabel("Trip Distance Bin")
    plt.tight_layout()
    plt.show()

    # ---- 7. Temporal heatmap ----
    pivot = df_clean.groupby(['day of week', 'hour'])['fare_amount'].mean().unstack()
    pivot.index = DAY_LABELS
    plt.figure(figsize=(14, 5))
    sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.3, cbar_kws={'label': 'Avg Fare ($)'})
    plt.title("Average Fare by Hour x Day of Week — Manhattan\n(Darker = higher avg fare)")
    plt.xlabel("Hour of Day"); plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.show()

    # ---- 8. Trip distance distribution ----
    plt.figure(figsize=(10, 5))
    sns.countplot(x=df_clean['distance bin'], palette='viridis')
    plt.title("Trip Distance Distribution — Final Manhattan Dataset")
    plt.xlabel("Distance Bin")
    plt.tight_layout()
    plt.show()

    # ---- 9. Fee occurrence ----
    from config import FEE_COLUMNS
    fee_occurrence = (df_clean[FEE_COLUMNS] > 0).mean() * 100
    plt.figure(figsize=(10, 5))
    sns.barplot(x=[c.replace('_', ' ') for c in fee_occurrence.index],
                y=fee_occurrence.values, palette='magma')
    plt.title("Percentage of Trips with Fee > $0\n"
              "(Near-100% fees are quasi-constant — dropped from modeling)")
    plt.ylabel("% of Trips"); plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ---- 10. Boxplot fare per distance bin ----
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=df_clean, x='distance bin', y='fare_amount',
                palette='YlOrRd', showfliers=False)
    plt.title("Fare Amount Distribution per Distance Bin\n"
              "(Box = IQR, line = median | fliers hidden — outliers already removed)")
    plt.xlabel("Trip Distance Bin"); plt.ylabel("Fare Amount ($)")
    plt.tight_layout()
    plt.show()

    # ---- 11. Trip volume per hour ----
    hourly = df_clean.groupby(['hour', 'is weekend']).size().reset_index(name='trip_count')
    hourly['day type'] = hourly['is weekend'].map({0: 'Weekday', 1: 'Weekend'})
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=hourly, x='hour', y='trip_count',
                 hue='day type', marker='o', palette=['steelblue', 'darkorange'])
    plt.title("Trip Volume per Hour — Weekday vs Weekend\n"
              "(Shows where the model has most training data)")
    plt.xlabel("Hour of Day"); plt.ylabel("Number of Trips")
    plt.xticks(range(0, 24)); plt.legend(title='')
    plt.tight_layout()
    plt.show()

    # ---- 12. Manhattan choropleth map ----
    plot_manhattan_heatmap(df_clean)


# =============================================================================
# 3. STATISTICS
# =============================================================================

def print_statistics(df_step1, df_manhattan, df_clean):
    """Prints key dataset statistics for the report."""
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

    result = run_preprocessing()
    if result is None:
        exit()

    X_train, X_test, y_train, y_test, kfold, df_step1, df_manhattan, df_work, df_clean, zone_map_manhattan = result

    active_features = get_active_features(df_manhattan)
    run_diagnostics(df_manhattan, active_features)
    run_pca_99(df_manhattan, active_features)

    run_visuals(df_step1, df_manhattan, df_work, df_clean, zone_map_manhattan)

    print_statistics(df_step1, df_manhattan, df_clean)