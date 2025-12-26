# Distributed Traffic Speed Prediction using Big Data and Neural Networks

Neural network trained on the **METR-LA dataset (207 sensors)** to predict short-term traffic speeds.  
Includes **distributed deployment on Globus Compute (FuncX)** and **local simulation of multi-node execution**.

## 📌 Project Overview
The system implements a complete pipeline:
- **Training Phase (local):** Keras-based neural network trained on traffic sensor data.  
- **Deployment Phase (distributed):** Remote inference on Globus Compute (FuncX).  
- **Simulation Phase:** Local multi-threaded simulation of distributed execution.  

The architecture supports single-node inference, multi-node inference, and local simulation with artificial latency.  

## Technologies
- **Python**, **TensorFlow/Keras** for model training  
- **Globus Compute (FuncX)** for distributed inference  
- **Pandas, NumPy** for preprocessing  
- **Matplotlib, Seaborn** for visualization  
- **ThreadPoolExecutor** for simulated distributed execution  
- Data formats: **HDF5** (time-series) + **Pickle** (metadata adjacency matrix)

## 📊 Results

  **Model Accuracy**
  - Training MSE: **0.1343**
  - Training MAE: **0.1988 km/h**
  - Validation MAE: **0.2618 km/h** (early stopping at epoch 66)  

  **Distributed Execution (Globus Compute)**
  - Single node (207 sensors): **9.80 s**
  - Four nodes (~51-53 sensors/node): **14.51 s**
  - Multi-node slower due to scheduling overhead and cold starts.

  **Simulated Distributed Execution (ThreadPoolExecutor)**
  - 4 workers: 24.06 s  
  - 8 workers: 13.83 s  
  - 16 workers: 7.91 s (scalability sweet spot)  
  - 32 workers: 7.68 s  
  - 64 workers: 7.61 s  

  **Prediction Quality**
  - Predicted speeds clustered between **45–65 km/h**, matching typical urban traffic.  
  - Predicted vs. average speeds aligned along the diagonal => high correlation between outputs and real data.

