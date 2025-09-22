import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os

# Path către fișierul cu rezultate
results_path = "/home/darime/outputs/traffic_analysis_with_nn.txt"

# Citire fișier
with open(results_path, 'r') as f:
    lines = f.readlines()

# Extrage date din fișier
sensors, predicted, avg_speeds = [], [], []

for i in range(len(lines)):
    if "Driving segment from sensor" in lines[i]:
        sensor_id = int(re.findall(r"\d+", lines[i])[0])
        avg = float(re.findall(r"[\d.]+", lines[i + 1])[0])
        pred = float(re.findall(r"[\d.]+", lines[i + 2])[0])
        sensors.append(sensor_id)
        avg_speeds.append(avg)
        predicted.append(pred)

# Asigură-te că directorul pentru grafice există
os.makedirs("/home/darime/outputs", exist_ok=True)

# 1. Predicted vs. Average Speed
plt.figure(figsize=(12, 6))
plt.plot(sensors, avg_speeds, label="Average Speed", linestyle="--")
plt.plot(sensors, predicted, label="Predicted Speed", marker="o")
plt.title("Predicted vs. Average Speed per Sensor")
plt.xlabel("Sensor ID")
plt.ylabel("Speed (km/h)")
plt.legend()
plt.grid(True)
plt.savefig("/home/darime/outputs/predicted_vs_average_speed.png")
plt.close()

# 2. Histogram of predicted speeds
plt.figure(figsize=(10, 5))
sns.histplot(predicted, kde=True, bins=10)
plt.title("Distribution of Predicted Speeds")
plt.xlabel("Predicted Speed (km/h)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("/home/darime/outputs/predicted_speed_distribution.png")
plt.close()

# 3. Speed prediction error (abs error)
errors = np.abs(np.array(predicted) - np.array(avg_speeds))
plt.figure(figsize=(12, 6))
plt.bar(sensors, errors)
plt.title("Absolute Prediction Error per Sensor")
plt.xlabel("Sensor ID")
plt.ylabel("Error (km/h)")
plt.grid(True)
plt.savefig("/home/darime/outputs/prediction_error_per_sensor.png")
plt.close()

print("✅ Graficele au fost generate și salvate în /home/darime/outputs/")
