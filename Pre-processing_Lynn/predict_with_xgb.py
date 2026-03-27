"""
predict_with_xgb.py
-------------------
Load the trained XGBoost model and make predictions on sample data.
"""
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("best_model_lynn_xgb.pkl")

# Example input features (based on Lynn preprocessing)
# Features: ['trip_distance', 'hour', 'day of week', 'is weekend', 'passenger_count', 'PULocationID', 'DOLocationID']
sample_data = pd.DataFrame({
    'trip_distance': [5.0, 2.5, 10.0],
    'hour': [14, 8, 22],  # afternoon, morning, night
    'day of week': [1, 5, 6],  # Monday, Friday, Saturday
    'is weekend': [0, 0, 1],
    'passenger_count': [1, 2, 4],
    'PULocationID': [100, 200, 300],
    'DOLocationID': [150, 250, 350]
})

# Make predictions
predictions = model.predict(sample_data)

print("Sample predictions:")
for i, pred in enumerate(predictions):
    print(f"Trip {i+1}: Predicted fare = ${pred:.2f}")

# Optional: Show model type
print(f"\nModel type: {type(model.named_steps['xgb'])}")