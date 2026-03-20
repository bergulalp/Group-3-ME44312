# NYC Yellow Taxi Fare Prediction — Group 3 (ME44312)

This project analyzes NYC Yellow Taxi data from January 2025 and builds machine learning models to predict taxi fares using only information available at dispatch time.

## Prerequisites

You must have Anaconda or Mamba installed on your system to manage the environment and dependencies.

## Installation

1. Clone this repository to your local machine:
   ```
   git clone https://github.com/bergulalp/Group-3-ME44312.git
   cd Group-3-ME44312
   ```

2. Create the environment from the environment file:
   ```
   conda env create -f taxi_environment.yml
   ```
   Or using Mamba (recommended for speed):
   ```
   mamba env create -f taxi_environment.yml
   ```

3. Activate the environment:
   ```
   conda activate taxi_environment
   ```

## Development Environment

This project was developed using the Spyder IDE. Any IDE that supports Python (VS Code, PyCharm) works fine as long as you select `taxi_environment` as your interpreter.

## Data Requirements

Place the following file in the project root directory before running any scripts:

- `yellow_tripdata_2025-01.parquet` — available from the [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) page.

Optionally, for the Manhattan choropleth map visual, download and unzip the NYC Taxi Zone shapefile into a folder called `taxi_zones/` in the project root:

- Download from: https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip

## Project Structure

```
Group-3-ME44312/
├── config.py            — all configuration: paths, thresholds, features, plot settings
├── preprocessing.py     — data loading, cleaning, outlier detection, stratified train/test split
├── eda.py               — all exploratory visualizations and statistics
├── model_utils.py       — pipeline builder with correct feature transformations per model
├── taxi_environment.yml — conda environment file
└── taxi_zones/          — (optional) shapefile folder for Manhattan map
```

## How to Run

### Exploratory Data Analysis

Runs all visualizations and prints dataset statistics:
```
python eda.py
```

### Preprocessing only

Loads data, cleans it, detects outliers, and produces a stratified train/test split:
```
python preprocessing.py
```

### Model pipeline check

Verifies the feature transformation pipeline is configured correctly:
```
python model_utils.py
```

## Configuration

All parameters are controlled from `config.py`. No other file contains hardcoded values. To change the data path, model features, outlier detection threshold, train/test split ratio, or any plot setting — edit `config.py` only.

## Preprocessing Pipeline

1. **Standard filtering** — removes nulls, negative fees, invalid passenger counts and rate codes.
2. **Feature engineering** — extracts hour, day of week, weekend flag, and trip duration from datetimes.
3. **Scope filtering** — restricts to Standard Rate (RatecodeID 1) Manhattan-to-Manhattan trips only.
4. **Outlier detection** — Isolation Forest on fare, distance, and fare-per-mile (1% contamination rate).
5. **Stratified split** — 80/20 train/test split stratified on hour period and pickup zone cluster, ensuring representative coverage of all time periods and areas in both sets.

## Feature Transformations (model_utils.py)

Different features require different transformations depending on their distribution and meaning:

| Feature group | Features | Transformation | Reason |
|---|---|---|---|
| Log + scale | trip_distance, passenger_count | log1p then StandardScaler | Right-skewed, reduces influence of extremes |
| Cyclic | hour, day of week | sin/cos encoding | Wrap-around features: hour 23 is close to hour 0 |
| Passthrough | is weekend, PULocationID, DOLocationID | none | Binary or categorical; tree models are scale-invariant |

## Models

Three models are trained and compared:

- **Linear Regression** — interpretable baseline
- **Random Forest** — captures non-linear relationships
- **XGBoost** — gradient boosting for high accuracy on tabular data

All models use the same `build_pipeline(model)` interface from `model_utils.py` and are evaluated using 5-fold cross-validation with MAE and R-squared.

## Figures

A figures folder is included with the output plots from `eda.py` for reference, if you do not want to run the code yourself.