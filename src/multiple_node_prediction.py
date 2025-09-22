# ----------------------------------------------------
# Import required packages
# ----------------------------------------------------
from globus_compute_sdk import Client, Executor
from globus_compute_sdk.serialize import ComputeSerializer, CombinedCode
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import os


# ----------------------------------------------------
# Main prediction function (per endpoint)
# ----------------------------------------------------
def predict_subset(h5_path, model_path, scaler_path, sensor_indices, lookback, output_txt_path):
    logs = []
    try:
        import pandas as pd
        import numpy as np
        import pickle
        import tensorflow as tf
        import os
        # Load data
        logs.append(f"[INFO] Loading traffic data from: {h5_path}")
        df = pd.read_hdf(h5_path, key='df')
        data = df.values

        logs.append(f"[INFO] Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        data_scaled = scaler.transform(data)

        if data_scaled.shape[0] < lookback:
            logs.append("[ERROR] Not enough data for lookback.")
            return logs

        logs.append("[INFO] Preparing input for prediction.")
        X_pred = np.expand_dims(data_scaled[-lookback:, :], axis=0)

        logs.append(f"[INFO] Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)

        y_pred_scaled = model.predict(X_pred)[0]
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(1, -1))[0]

        logs.append("[INFO] Prediction completed.")

        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        with open(output_txt_path, 'a') as f:
            f.write("===== Traffic Prediction Results =====\n")
            for idx in sensor_indices:
                avg_speed = np.mean(data[:, idx])
                pred_speed = y_pred[idx]
                line = (f"🚗 Driving segment from sensor {idx}:\n"
                        f"   Average speed: {avg_speed:.2f} km/h\n"
                        f"   Predicted next: {pred_speed:.2f} km/h\n")
                f.write(line)
                logs.append(line.strip())
            f.write("\n")

        logs.append(f"[INFO] Results written to: {output_txt_path}")
        return logs

    except Exception as e:
        logs.append(f"[ERROR] Exception occurred: {str(e)}")
        return logs


# ----------------------------------------------------
# Setup: Globus Compute
# ----------------------------------------------------
gc = Client()
gc.serializer = ComputeSerializer(strategy_code=CombinedCode())

# Register function (only once)
func_id = gc.register_function(
    function=predict_subset,
    description="Distributed sensor prediction for METR-LA"
)

# ----------------------------------------------------
# Define distributed endpoints
# ----------------------------------------------------
endpoints = {
    'traffic_node_1': '37295d9e-8427-4fca-934e-3082c7361e76',
    'traffic_node_2': '20d6e4b3-32f0-4b99-bd51-8873fe7f0666',
    'traffic_node_3': '42f456f3-58a4-4e30-9c8a-98c9fe3fdacb',
    'traffic_node_4': 'bc688b64-c13e-4a5d-8a0b-7b0c6425e59c'
}

# ----------------------------------------------------
# Load sensor count from adjacency file
# ----------------------------------------------------
with open('/home/darime/data/adj_METR-LA.pkl', 'rb') as f:
    adj_data = pickle.load(f, encoding='latin1')
total_sensors = adj_data[2].shape[0]

# Split sensors across endpoints
sensor_chunks = np.array_split(np.arange(total_sensors), len(endpoints))

# Prepare parameters for each endpoint
input_params_list = []
for i, (name, eid) in enumerate(endpoints.items()):
    sensor_ids = sensor_chunks[i].tolist()
    txt_out = f"/home/darime/outputs/traffic_node_{i + 1}.txt"
    args_tuple = (
        '/home/darime/data/METR-LA.h5',
        '/home/darime/models/keras_model_all_sensors.h5',
        '/home/darime/models/scaler_all_sensors.pkl',
        sensor_ids,
        5,  # lookback
        txt_out
    )
    input_params_list.append((eid, args_tuple))

# ----------------------------------------------------
# Submit tasks to each endpoint
# ----------------------------------------------------
print("🚀 Submitting distributed tasks to traffic nodes...")
futures = []
start_time = time.time()

for endpoint_id, args in input_params_list:
    executor = Executor(endpoint_id=endpoint_id)
    future = executor.submit_to_registered_function(func_id, args)
    futures.append(future)

# ----------------------------------------------------
# Gather results
# ----------------------------------------------------
results = []
for i, future in enumerate(futures):
    try:
        res = future.result()
        if isinstance(res, list):
            results.extend(res)
        else:
            print(f"[WARNING] Node {i + 1} returned unexpected output:\n{res}")
    except Exception as e:
        print(f"[ERROR] Node {i + 1} failed: {str(e)}")

# ----------------------------------------------------
# Save results
# ----------------------------------------------------
output_path = "/home/darime/outputs/multi_node_traffic_prediction.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(output_path, index=False)

print("✅ Distributed prediction complete!")
print(f"⏱️ Total execution time: {time.time() - start_time:.2f} seconds")
print(df.head())
