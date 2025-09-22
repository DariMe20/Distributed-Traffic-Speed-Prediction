# Distributed Traffic Speed Prediction using Big Data and Neural Networks

This project explores **real-time traffic speed prediction** using Big Data and distributed computing frameworks.

## 📌 Project Overview
Urban traffic congestion is a major challenge. This project trains a **Keras-based neural network** on the **METR-LA dataset (207 sensors)** and deploys it for distributed inference using **Globus Compute (FuncX)**.  

The system supports:
- Local training and validation  
- Single-node inference  
- Multi-node distributed inference  
- Local simulation with ThreadPoolExecutor  

## 🛠 Technologies
- Python, TensorFlow/Keras  
- Globus Compute (FuncX)  
- Pandas, NumPy, Matplotlib, Seaborn  
- HDF5/CSV for large-scale data handling  

## 📊 Results
- Validation MAE: **0.26 km/h**  
- Simulated distributed execution: reduced latency up to 64 threads  
- Real distributed execution: showed scalability limits due to network overhead  

## 📂 Repository
- `src/` → Training & inference scripts  
- `data/` → Links to METR-LA dataset  
- `outputs/` → Results, logs, visualizations  
- `docs/` → Project report (PDF)  

