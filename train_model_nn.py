import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Dropout
from sklearn.preprocessing import StandardScaler
import pickle
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Paths
data_path = '/home/darime/data/METR-LA.h5'
model_path = '/home/darime/models/keras_model_all_sensors.h5'
scaler_path = '/home/darime/models/scaler_all_sensors.pkl'
loss_plot_path = '/home/darime/outputs/loss_plot.png'
mae_plot_path = '/home/darime/outputs/mae_plot.png'

# Hyperparameters
lookback = 5
epochs = 100
batch_size = 64

# Load data
print(f"[INFO] Loading data from {data_path}...")
df = pd.read_hdf(data_path, key='df')
data = df.values
print(f"[INFO] Raw data shape: {data.shape}")

# Scale data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print("[INFO] Data scaled.")

os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"[INFO] Scaler saved at {scaler_path}")

# Prepare sequences
X, y = [], []
T, num_sensors = data_scaled.shape
for i in range(T - lookback):
    X.append(data_scaled[i:i + lookback, :])
    y.append(data_scaled[i + lookback, :])
X = np.array(X)
y = np.array(y)
print(f"[INFO] Training data shape: X={X.shape}, y={y.shape}")

# Build model
model = Sequential([
    InputLayer(input_shape=(lookback, num_sensors)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(num_sensors)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)

# Train
print("[INFO] Training model...")
history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[early_stop, reduce_lr],
                    verbose=2)

# Evaluate
final_loss, final_mae = model.evaluate(X, y, verbose=0)
print(f"[INFO] Final Loss (MSE): {final_loss:.4f}")
print(f"[INFO] Final MAE: {final_mae:.4f} km/h")

# Plot and save loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid()
plt.savefig(loss_plot_path)
plt.close()
print(f"[INFO] Loss plot saved at {loss_plot_path}")

plt.figure(figsize=(10, 5))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid()
plt.savefig(mae_plot_path)
plt.close()
print(f"[INFO] MAE plot saved at {mae_plot_path}")

# Save model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)
print(f"[INFO] Model saved at {model_path}")
