from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import time
import random
import os

# Config
model_path = '/home/darime/models/keras_model_all_sensors.h5'
scaler_path = '/home/darime/models/scaler_all_sensors.pkl'
h5_path = '/home/darime/data/METR-LA.h5'
lookback = 5
num_workers = 64

# Load model and scaler
model = tf.keras.models.load_model(model_path, compile=False)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Load data
df = pd.read_hdf(h5_path, key='df')
data = df.values
data_scaled = scaler.transform(data)
X_pred = np.expand_dims(data_scaled[-lookback:, :], axis=0)
num_sensors = data.shape[1]


# Simulate distributed nodes – each one makes inference on one section
def predict_on_sensor(sensor_id):
    time.sleep(random.uniform(0.2, 0.6))  # simulare delay
    single_sensor_input = X_pred[0, :, sensor_id].reshape(1, lookback, 1)
    repeated_input = np.repeat(single_sensor_input, num_sensors, axis=2)  # for model input shape
    y_pred_scaled = model.predict(repeated_input)[0]
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(1, -1))[0]
    avg_speed = np.mean(data[:, sensor_id])
    return {
        "sensor": sensor_id,
        "prediction": y_pred[sensor_id],
        "avg_speed": avg_speed,
        "latency": round(random.uniform(0.2, 0.6), 3)
    }


start_time = time.time()
# Run simulation in parallel
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = list(executor.map(predict_on_sensor, range(num_sensors)))

# Save results for analysis
df_res = pd.DataFrame(results)
os.makedirs("/home/darime/outputs", exist_ok=True)
df_res.to_csv("/home/darime/outputs/simulated_predictions.csv", index=False)
print(f"Execution time: {time.time() - start_time:.2f} seconds")