# Import necessary libraries
from funcx.sdk.client import FuncXClient
from globus_compute_sdk import Client
import time
import pandas as pd
import pickle
import h5py
import numpy as np
import networkx as nx


# -------------------------------
# Utility function to load the graph from pickle file
# -------------------------------
def load_graph(pickle_path):
    """
    Loads the adjacency matrix from a pickle file
    and builds a directed graph using NetworkX.
    """
    with open(pickle_path, 'rb') as f:
        adj_data = pickle.load(f)
    adjacency_matrix = adj_data['adj_mx']
    sensor_ids = adj_data['sensor_ids']
    id_to_node = adj_data['id_to_node']
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph())
    return G, sensor_ids, id_to_node


# -------------------------------
# Utility function to load traffic data from HDF5 file
# -------------------------------
def load_traffic_data(h5_path):
    """
    Loads traffic data from an HDF5 file.
    """
    with h5py.File(h5_path, 'r') as f:
        data = f['df'][:]
    return data


# -------------------------------
# Main function to analyze traffic
# -------------------------------
def traffic_analysis(params):
    import pandas as pd
    import pickle
    import h5py
    import numpy as np
    import networkx as nx

    sensor_id = params['sensor_id']
    adj_pickle_path = params['adj_pickle_path']
    h5_path = params['h5_path']

    try:
        with open(adj_pickle_path, 'rb') as f:
            adj_data = pickle.load(f, encoding='latin1')

    except Exception as e:
        return f"[ERROR] Could not load pickle: {e}"

    # Extract adjacency_matrix safely
    try:
        if isinstance(adj_data, dict):
            adjacency_matrix = adj_data['adj_mx']
        elif isinstance(adj_data, list) and len(adj_data) >= 3:
            adjacency_matrix = adj_data[2]
        else:
            return "[ERROR] Unsupported adj_data structure."
    except Exception as e:
        return f"[ERROR] Could not extract adjacency_matrix: {e}"

    # Load the adjacency matrix as graph
    try:
        G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph())
    except Exception as e:
        return f"[ERROR] Could not create graph from adjacency_matrix: {e}"

    with h5py.File(h5_path, 'r') as f:
        if 'df' in f:
            dataset_or_group = f['df']
            if isinstance(dataset_or_group, h5py.Dataset):
                data = dataset_or_group[:]
            elif isinstance(dataset_or_group, h5py.Group):
                first_key = list(dataset_or_group.keys())[0]
                data = dataset_or_group[first_key][:]
            else:
                return "[ERROR] 'df' is neither a Dataset nor a Group."
        else:
            return "[ERROR] Key 'df' not found in HDF5 file."

    try:
        if data.ndim == 2:
            speed_data = data[:, sensor_id]
        elif data.ndim == 1:
            speed_data = data
        else:
            return f"[ERROR] Unexpected data shape: {data.shape}"

        # Convert to numeric (works for both str and bytes)
        speed_data = pd.to_numeric(speed_data.astype(str), errors='coerce')

        avg_speed = np.nanmean(speed_data)
    except Exception as e:
        return f"[ERROR] Could not compute average speed: {e}"

    output_path = '/home/darime/outputs/traffic_analysis.txt'
    try:
        with open(output_path, 'a') as f:
            f.write(f"Sensor {sensor_id}: Average speed = {avg_speed:.2f} km/h\n")
    except Exception as e:
        return f"[ERROR] Could not write output: {e}"

    return f"Sensor {sensor_id}: Average speed = {avg_speed:.2f} km/h"


# -------------------------------
# FuncX Client initialization
# -------------------------------
from globus_compute_sdk.serialize import ComputeSerializer, CombinedCode

fxc = FuncXClient()
fxc.serializer = ComputeSerializer(strategy_code=CombinedCode())

# Define the endpoint ID (replace with your own)
endpoint_id = 'ac9f12d1-f1f7-44d3-a40f-b0ee9ea6618b'

# Globus Compute Client initialization
gc = Client()

# -------------------------------
# Register the function on the endpoint
# -------------------------------
func_id = gc.register_function(
    function=traffic_analysis,
    description="Calculate average speed for a given sensor ID using METR-LA dataset."
)
print(f"Function registered with ID: {func_id}")

# -------------------------------
# Define input parameters
# -------------------------------

params = {
    'sensor_id': 11,
    'adj_pickle_path': '/home/darime/data/adj_METR-LA.pkl',
    'h5_path': '/home/darime/data/METR-LA.h5'
}

# -------------------------------
# Submit the function for remote execution
# -------------------------------
task = fxc.run(
    params,
    function_id=func_id,
    endpoint_id=endpoint_id
)

start_time = time.time()
print("Waiting for task to complete...")

# -------------------------------
# Poll for task status
# -------------------------------
while True:
    task_status = fxc.get_task(task)
    if task_status['pending']:
        print("Task pending...")
        time.sleep(1)
    else:
        break

# -------------------------------
# Retrieve and display results
# -------------------------------
result = fxc.get_result(task)
print("Task completed!")
print(f"Execution time: {time.time() - start_time:.2f} seconds")
print("Result:", result)
