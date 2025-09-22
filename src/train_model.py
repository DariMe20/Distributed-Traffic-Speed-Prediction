import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# Paths
data_path = '/home/darime/data/METR-LA.h5'
model_path = '/home/darime/models/linear_model.pkl'

# Load data
df = pd.read_hdf(data_path, key='df')
data = df.values

# Prepare training data for ALL sensors
lookback = 5
X, y = [], []

num_sensors = data.shape[1]
for sensor_id in range(num_sensors):
    sensor_data = data[:, sensor_id]
    for i in range(len(sensor_data) - lookback):
        X.append(sensor_data[i:i + lookback])
        y.append(sensor_data[i + lookback])

X = np.array(X)
y = np.array(y)

print(f"[INFO] Training dataset shape: {X.shape}, {y.shape}")

# Train model
model = LinearRegression()
model.fit(X, y)
print("[INFO] Model trained on ALL sensors.")

# Save model
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"[INFO] Model saved at {model_path}")