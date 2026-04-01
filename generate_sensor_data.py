"""
AutoEDA++ Anomaly Detection — Sample Sensor Dataset Generator
=============================================================
Run this script once to produce sample_sensor.csv and sample_sensor2.csv
for testing the anomaly pipeline.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 500

timestamps = pd.date_range("2024-01-01", periods=N, freq="10min")
temperature = 70 + 5 * np.sin(np.linspace(0, 4 * np.pi, N)) + rng.normal(0, 1, N)
pressure    = 101.3 + rng.normal(0, 0.5, N)
vibration   = rng.exponential(0.5, N)
humidity    = 55 + rng.normal(0, 3, N)
current     = 5.0 + 0.3 * np.sin(np.linspace(0, 6 * np.pi, N)) + rng.normal(0, 0.2, N)

# Inject hidden anomalies at specific indices
anomaly_idx = rng.choice(N, size=40, replace=False)
temperature[anomaly_idx] += rng.choice([-20, 20], size=40)
vibration[anomaly_idx]   += rng.uniform(5, 10, 40)
current[anomaly_idx]     += rng.choice([-3, 3], size=40)

anomaly = np.zeros(N, dtype=int)
anomaly[anomaly_idx] = 1

df1 = pd.DataFrame({
    "timestamp":   timestamps,
    "temperature": np.round(temperature, 2),
    "pressure":    np.round(pressure, 3),
    "vibration":   np.round(vibration, 4),
    "humidity":    np.round(np.clip(humidity, 20, 90), 1),
    "current":     np.round(current, 3),
    "anomaly":     anomaly,
})

# Introduce a few missing values
for col in ["temperature", "pressure", "humidity"]:
    miss_idx = rng.choice(N, size=5, replace=False)
    df1.loc[miss_idx, col] = np.nan

df1.to_csv("sample_sensor.csv", index=False)
print(f"sample_sensor.csv written — {len(df1)} rows, {df1['anomaly'].sum()} anomalies")

# ------ Second sensor file (different machine / site) ------
N2 = 300
timestamps2 = pd.date_range("2024-01-04", periods=N2, freq="10min")
temperature2 = 68 + 4 * np.sin(np.linspace(0, 3 * np.pi, N2)) + rng.normal(0, 1.2, N2)
pressure2    = 100.8 + rng.normal(0, 0.6, N2)
vibration2   = rng.exponential(0.45, N2)
humidity2    = 52 + rng.normal(0, 4, N2)
current2     = 4.8 + 0.25 * np.sin(np.linspace(0, 5 * np.pi, N2)) + rng.normal(0, 0.18, N2)

anomaly_idx2 = rng.choice(N2, size=20, replace=False)
temperature2[anomaly_idx2] += rng.choice([-18, 18], size=20)
vibration2[anomaly_idx2]   += rng.uniform(4, 9, 20)

anomaly2 = np.zeros(N2, dtype=int)
anomaly2[anomaly_idx2] = 1

df2 = pd.DataFrame({
    "timestamp":   timestamps2,
    "temperature": np.round(temperature2, 2),
    "pressure":    np.round(pressure2, 3),
    "vibration":   np.round(vibration2, 4),
    "humidity":    np.round(np.clip(humidity2, 20, 90), 1),
    "current":     np.round(current2, 3),
    "anomaly":     anomaly2,
})

df2.to_csv("sample_sensor2.csv", index=False)
print(f"sample_sensor2.csv written — {len(df2)} rows, {df2['anomaly'].sum()} anomalies")
