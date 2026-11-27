"""
Synthetic Drive Cycle Generator
Generates a baseline, healthy multivariate time-series dataset mimicking ECU telemetry.
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import os

def load_signal_config(config_path="configs/signals.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["channels"]

def generate_drive_cycle(duration_minutes=15, sample_rate_hz=100):
    print(f"Generating {duration_minutes} minutes of healthy ECU telemetry at {sample_rate_hz}Hz...")
    n_samples = duration_minutes * 60 * sample_rate_hz
    time_idx = pd.date_range(start="2026-06-01 10:00:00", periods=n_samples, freq=f"{1000//sample_rate_hz}ms")
    
    # 1. Base Driving States (Idle, Accelerate, Cruise, Decelerate)
    # Generate smooth transitions using a low-pass filtered random walk
    state_noise = np.cumsum(np.random.normal(0, 0.005, n_samples))
    throttle_base = np.clip(np.sin(np.linspace(0, 50, n_samples)) * 30 + 30 + state_noise, 0, 100)
    
    # Add sudden stops/idling periods (throttle to 0)
    idle_mask = np.random.choice([0, 1], size=n_samples, p=[0.1, 0.9])
    throttle_position = pd.Series(throttle_base * idle_mask).rolling(window=100, min_periods=1).mean().values

    # 2. Physics-based Signal Correlation
    # Engine RPM follows throttle with a slight mechanical lag and base idle (800 RPM)
    engine_rpm = 800 + (throttle_position * 60) + np.random.normal(0, 20, n_samples)
    engine_rpm = pd.Series(engine_rpm).rolling(window=50, min_periods=1).mean().values

    # MAP Sensor (Manifold Absolute Pressure) correlates highly with RPM/Throttle
    map_sensor = 30 + (throttle_position * 1.5) + np.random.normal(0, 2, n_samples)

    # Coolant Temp: Slowly rises from 20C to operating temp ~90C, then stabilizes
    coolant_temp = 20 + 70 * (1 - np.exp(-np.linspace(0, 5, n_samples))) + np.random.normal(0, 0.5, n_samples)

    # Battery Voltage: Stable ~13.8V from alternator with tiny PWM noise
    battery_voltage = 13.8 + np.random.normal(0, 0.05, n_samples)

    # Lambda Sensor (O2): Oscillates rapidly around 1.0 during normal closed-loop operation
    lambda_sensor = 1.0 + 0.05 * np.sin(np.linspace(0, 1000, n_samples)) + np.random.normal(0, 0.02, n_samples)

    # 3. Assemble DataFrame
    df = pd.DataFrame({
        "timestamp": time_idx,
        "throttle_position": throttle_position,
        "engine_rpm": engine_rpm,
        "coolant_temp": coolant_temp,
        "battery_voltage": battery_voltage,
        "map_sensor": map_sensor,
        "lambda_sensor": lambda_sensor
    })
    
    # Ensure raw data directory exists
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    out_path = "data/raw/baseline_drive_cycle.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Baseline data generated and saved to {out_path} ({len(df)} rows)")
    
    return df

if __name__ == "__main__":
    # Test the generator
    generate_drive_cycle(duration_minutes=15)