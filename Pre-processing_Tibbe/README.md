# NYC Yellow Taxi Fare Estimation

This project provides a modular pipeline for estimating taxi fares based on the NYC TLC January 2025 dataset. The goal is to build a robust preprocessing and feature engineering flow.

## 🛠 Project Structure
- `config.py`: Centralized configuration for all constants, file paths, and filtering thresholds.
- `data_processor.py`: The "Logic Layer." Contains functions for data cleaning, cyclic time encoding, and DBSCAN-based outlier detection.
- `main.py`: The "Execution Layer." Orchestrates the data flow and generates visualizations.

## 📈 Feature Engineering Rationale
- **Cyclic Time (Sin/Cos)**: We transform the pickup hour into sine and cosine components. This allows the model to understand the 24-hour cycle (e.g., that 23:00 and 00:00 are adjacent).
- **Spatial Features**: PULocationID and DOLocationID are retained to capture zone-specific surcharges and traffic patterns.
- **Logarithmic Scaling**: (Optional) Trip distance can be log-transformed to reduce the influence of extreme long-distance outliers.

## 🔍 Outlier Detection
We utilize **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** to identify anomalies. By looking at the relationship between `trip_distance` and `fare_amount`, we can exclude trips that are mathematically "impossible" or extreme statistical noise.

## 🚀 How to Run
1. Ensure the dataset `yellow_tripdata_2025-01 (1).parquet` is downloaded at  `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet`
2. Install dependencies: `pip install pandas scikit-learn matplotlib seaborn pyarrow`
3. Run the main analysis:
   ```bash
   python main.py