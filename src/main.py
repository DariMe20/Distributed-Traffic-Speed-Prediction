# ----------------------------------------------------
# Import required packages
# ----------------------------------------------------
from globus_compute_sdk import Client, Executor
from globus_compute_sdk.serialize import ComputeSerializer, CombinedCode
import time
import pickle


# ----------------------------------------------------
# Main function to analyze traffic and predict speed
# ----------------------------------------------------
def traffic_prediction(params):
    import numpy as np
    import pandas as pd
    import pickle
    import tensorflow as tf

    logs = []

    try:
        adj_pickle_path = params['adj_pickle_path']
        h5_path = params['h5_path']
        model_path = params['model_path']
        scaler_path = params['scaler_path']
        output_path = params['output_path']

        lookback = 5

        # Load data
        logs.append(f"[INFO] Loading traffic data from: {h5_path}")
        df = pd.read_hdf(h5_path, key='df')
        data = df.values  # shape: (num_timesteps, num_sensors)

        # Load scaler
        logs.append(f"[INFO] Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        data_scaled = scaler.transform(data)

        if data_scaled.shape[0] < lookback:
            logs.append("[ERROR] Not enough data for lookback.")
            return logs

        # Prepare input sequence
        X_pred = np.expand_dims(data_scaled[-lookback:, :], axis=0)

        # Load model
        logs.append(f"[INFO] Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        logs.append("[INFO] Model loaded successfully.")

        # Predict traffic speed based on input sequence
        y_pred_scaled = model.predict(X_pred)[0]
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(1, -1))[0]

        logs.append("[INFO] Prediction completed.")

        # Write results in txt file
        with open(output_path, 'a') as f:
            f.write("===== Traffic Prediction Results =====\n")
            for idx, speed in enumerate(y_pred):
                avg_speed = np.mean(data[:, idx])
                line = (f"🚗 Driving segment from sensor {idx}:\n"
                        f"   Average speed: {avg_speed:.2f} km/h\n"
                        f"   Predicted next: {speed:.2f} km/h\n")
                logs.append(line.strip())
                f.write(line)
            f.write("\n")

        logs.append(f"[INFO] Results written to: {output_path}")
        return logs

    except Exception as e:
        logs.append(f"[ERROR] Exception occurred: {e}")
        return logs


# ----------------------------------------------------
# Globus Compute setup
# ----------------------------------------------------
gc = Client()
gc.serializer = ComputeSerializer(strategy_code=CombinedCode())

# Define endpoint ID
endpoint_id = 'ac9f12d1-f1f7-44d3-a40f-b0ee9ea6618b'

# Register the function
func_id = gc.register_function(
    function=traffic_prediction,
    description="Analyze traffic and predict next speed for all sensors using METR-LA dataset."
)
print(f"Function registered with ID: {func_id}")

# ----------------------------------------------------
# Define input parameters
# ----------------------------------------------------
# Load the graph to count sensors
with open('/home/darime/data/adj_METR-LA.pkl', 'rb') as f:
    adj_data = pickle.load(f, encoding='latin1')
if isinstance(adj_data, list) and len(adj_data) >= 3:
    total_sensors = adj_data[2].shape[0]
else:
    total_sensors = 0

print(f"[INFO] Total sensors in the dataset: {total_sensors}")

# Input parameters
params = {
    'adj_pickle_path': '/home/darime/data/adj_METR-LA.pkl',
    'h5_path': '/home/darime/data/METR-LA.h5',
    'model_path': '/home/darime/models/keras_model_all_sensors.h5',
    'scaler_path': '/home/darime/models/scaler_all_sensors.pkl',
    'output_path': '/home/darime/outputs/traffic_analysis_with_nn.txt'
}

# ----------------------------------------------------
# Submit the function for remote execution
# ----------------------------------------------------
executor = Executor(endpoint_id=endpoint_id)
future = executor.submit(traffic_prediction, params)

start_time = time.time()
print("⏳ Waiting for task to complete...")

# Poll for task status
while not future.done():
    print("Task pending...")
    time.sleep(1)

# Retrieve and display results
result = future.result()
print("✅ Task completed!")
print(f"Execution time: {time.time() - start_time:.2f} seconds")
print("Result:")
for line in result:
    print(line)
