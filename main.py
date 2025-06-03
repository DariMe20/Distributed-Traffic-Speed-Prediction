# ----------------------------------------------------
# Import required packages
# ----------------------------------------------------
from globus_compute_sdk import Client, Executor
from globus_compute_sdk.serialize import ComputeSerializer, CombinedCode
import time
import numpy as np
import pandas as pd
import pickle
import h5py
import networkx as nx


# ----------------------------------------------------
# Main function to analyze traffic
# ----------------------------------------------------
def traffic_analysis(params):
    logs = []

    import numpy as np
    import pandas as pd
    import pickle
    import h5py
    import networkx as nx

    try:
        sensor_id = params['sensor_id']
        adj_pickle_path = params['adj_pickle_path']
        h5_path = params['h5_path']

        logs.append(f"[DEBUG] Loading adj pickle file: {adj_pickle_path}")
        with open(adj_pickle_path, 'rb') as f:
            adj_data = pickle.load(f, encoding='latin1')

        if isinstance(adj_data, list) and len(adj_data) >= 3:
            adj_matrix = adj_data[2]
            logs.append(f"[DEBUG] Adjacency matrix shape: {np.shape(adj_matrix)}")
        else:
            logs.append("[ERROR] Unexpected structure in the pickle file.")
            return logs

        if isinstance(adj_matrix, list):
            adj_matrix = np.array(adj_matrix)
            logs.append(f"[DEBUG] Converted adjacency_matrix to numpy array: {adj_matrix.shape}")

        if adj_matrix.ndim != 2:
            logs.append("[ERROR] adjacency_matrix is not 2D!")
            return logs

        logs.append(f"[DEBUG] Loading HDF5 file: {h5_path}")
        try:
            df = pd.read_hdf(h5_path, key='df')
            data = df.values
            logs.append(f"[DEBUG] Traffic data shape: {data.shape}")
        except Exception as e:
            logs.append(f"[ERROR] Could not load HDF5 data using pandas: {e}")
            return logs

        if data.ndim != 2:
            logs.append("[ERROR] Data is not 2D!")
            return logs

        speed_data = data[:, sensor_id]
        avg_speed = np.mean(speed_data)
        logs.append(f"[DEBUG] Average speed for sensor {sensor_id}: {avg_speed:.2f} km/h")

        output_path = '/home/darime/outputs/traffic_analysis.txt'
        with open(output_path, 'a') as f:
            f.write(f"Sensor {sensor_id}: Average speed = {avg_speed:.2f} km/h\n")
        logs.append(f"[DEBUG] Result written to {output_path}")

        logs.append(f"Sensor {sensor_id}: Average speed = {avg_speed:.2f} km/h")
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
    function=traffic_analysis,
    description="Calculate average speed for a given sensor ID using METR-LA dataset."
)
print(f"Function registered with ID: {func_id}")

# ----------------------------------------------------
# Define input parameters
# ----------------------------------------------------
sensor_ids = [10, 20, 21, 22, 25, 30]
for sensor in sensor_ids:
    params = {
        'sensor_id': sensor,
        'adj_pickle_path': '/home/darime/data/adj_METR-LA.pkl',
        'h5_path': '/home/darime/data/METR-LA.h5'
    }

    # ----------------------------------------------------
    # Submit the function for remote execution
    # ----------------------------------------------------
    executor = Executor(endpoint_id=endpoint_id)
    future = executor.submit(traffic_analysis, params)

    start_time = time.time()
    print("Waiting for task to complete...")

    # ----------------------------------------------------
    # Poll for task status
    # ----------------------------------------------------
    while not future.done():
        print("Task pending...")
        time.sleep(1)

    # ----------------------------------------------------
    # Retrieve and display results
    # ----------------------------------------------------
    result = future.result()
    print("Task completed!")
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
    print("Result:")
    for line in result:
        print(line)
