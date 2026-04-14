# NYC Yellow Taxi Analysis

This project processes, analyzes, and models NYC Yellow Taxi data from January 2025 to predict fare amounts based on trip characteristics at dispatch time. 

## Prerequisites

You must have Anaconda installed on your system to manage the environment and dependencies.

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/bergulalp/Group-3-ME44312.git](https://github.com/bergulalp/Group-3-ME44312.git)
   cd Group-3-ME44312
   ```

2. Create the environment from the environment.yml file:

   Using Conda:
   ```bash
   conda env create -f taxi_environment.yml
   ```

   Using Mamba (recommended for speed):
   ```bash
   mamba env create -f taxi_environment.yml
   ```

3. Activate the environment:

   Using Conda:
   ```bash
   conda activate taxi_environment
   ```

   Using Mamba:
   ```bash
   mamba activate taxi_environment
   ```

## Development Environment

This project was developed using the Spyder IDE. However, you can use any IDE of your choice (such as VS Code or PyCharm) as long as you select the `taxi_environment` as your interpreter.

## Data Requirements

The script expects the following file to be present in the project `Data` directory:
- `yellow_tripdata_2025-01.parquet`

*(Note: The NYC taxi zone lookup CSV is fetched automatically via URL during execution).*

## Project Structure

The codebase is highly modular to separate data processing from model evaluation:
- **`config.py`**: The central configuration file. Contains file paths, feature lists, and hyperparameter grids.
- **`EDA.py`**: Exploratory Data Analysis. Generates visual plots and statistical summaries of the dataset (loads its own data safely to prevent ML pipeline leakage).
- **`preprocessing.py`**: Handles data cleaning, target encoding for spatial IDs, clustering, route statistics, and balancing.
- **`model_utiles.py`**: Defines custom mathematical transformations (e.g., cyclic encoding for hours/days and log-scaling for distance).
- **`models.py`**: The model factory that builds scikit-learn pipelines for Linear Regression, Random Forest, and XGBoost.
- **`parameter_tuning.py`**: Handles heavy hyperparameter tuning using HalvingRandomSearchCV and early stopping. Saves the best models to disk.
- **`metrics_evaluating.py`**: Contains standard functions to calculate and format regression metrics (MAE, RMSE, MAPE, R², etc.).

## How to Run

Once your environment is activated and the dataset is placed in the `Data/` folder, you can run the project in three main steps:

### 1. Data Exploration
To view the dataset distributions, outlier diagnostics, and Manhattan heatmaps, run:
```bash
python EDA.py
```

### 2. Train and Evaluate the Models (Main Pipeline)
To execute the full machine learning pipeline (preprocessing, fast baseline comparison, deep hyperparameter tuning, and final test evaluation), run:
```bash
python final_result.py
```
*Note: The first time this runs, deep tuning may take a while. The optimized models will be saved to a `Models/` directory so subsequent runs are much faster.*

### 3. Generate Visual Diagnostics
To generate and save in-depth performance plots for your tuned models (such as residual boxplots, error heatmaps, and SHAP feature importance), run:
```bash
python plot_final_results.py
```
*The resulting high-resolution plots will be saved in the `figures_final/` directory.*
```