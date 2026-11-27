"""
Automated FMEA Diagnostic Reporter
Detects an anomaly, isolates the highest-error sensor channel, 
and maps it to the Bosch Fault Taxonomy for root-cause analysis.
"""
import pandas as pd
import numpy as np
import joblib
import onnxruntime as ort
import json
from src.features.pipeline import create_sliding_windows
from src.explainability.fault_taxonomy import diagnose

# System Configurations
THRESHOLD = 0.2598
CHANNELS = [
    "throttle_position", "engine_rpm", "coolant_temp", 
    "battery_voltage", "map_sensor", "lambda_sensor"
]

def generate_fmea_report():
    print("🚨 --- Running Automated FMEA Diagnostic ---")
    
    # 1. Load the ONNX model and Scaler
    session = ort.InferenceSession("mlflow/models/lstm_ae.onnx")
    scaler = joblib.load("data/processed/scaler.pkl")
    
    # 2. Simulate a single 500ms window where the Coolant Temp suddenly spikes to 130C
    print("💉 Injecting simulated Coolant Temperature spike...")
    normal_data = pd.read_csv("data/raw/baseline_drive_cycle.csv").drop(columns=['timestamp']).values[:50]
    faulty_window = normal_data.copy()
    faulty_window[:, 2] = 130.0  # Index 2 is coolant_temp (massive overheat)
    
    # 3. Preprocess and Run Inference
    scaled_window = scaler.transform(faulty_window)
    x = np.array(scaled_window, dtype=np.float32)[np.newaxis, ...]
    x_hat = session.run(None, {"input": x})[0]
    
    # 4. Calculate OVERALL Error
    overall_mse = np.mean((x - x_hat) ** 2)
    is_anomaly = overall_mse > THRESHOLD
    
    if not is_anomaly:
        print("✅ System Healthy. No report needed.")
        return
        
    # 5. Root Cause Analysis: Calculate Error PER CHANNEL
    print("\n🔍 Anomaly Detected! Isolating Root Cause...")
    channel_errors = np.mean((x - x_hat) ** 2, axis=(0, 1)) # Average error across the 50 time steps
    
    # Find the channel with the highest reconstruction error
    worst_channel_idx = np.argmax(channel_errors)
    worst_channel_name = CHANNELS[worst_channel_idx]
    
    # 6. Fetch FMEA Taxonomy Diagnosis
    # We will simulate the fault type as 'voltage_spike' for the taxonomy mapping
    diagnosis = diagnose(worst_channel_name, "voltage_spike")
    
    # 7. Print the beautiful terminal report
    print("\n" + "="*50)
    print(" 📑 AUTOMATED QA DIAGNOSTIC REPORT ")
    print("="*50)
    print(f"Timestamp        : 2026-06-21 12:45:00")
    print(f"Overall Severity : CRITICAL (Score: {overall_mse:.4f} / Limit: {THRESHOLD})")
    print("-" * 50)
    print(f"🚩 Root Cause Signal : {worst_channel_name.upper()}")
    print(f"📉 Channel MSE Score : {channel_errors[worst_channel_idx]:.4f}")
    print("-" * 50)
    print(f"🛠️  FMEA Analysis    :")
    print(f"   - Fault Type      : {diagnosis['fault_type'].upper()}")
    print(f"   - Reference Code  : {diagnosis['fmea_reference']}")
    print(f"   - Probable Cause  : {diagnosis['probable_cause']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    generate_fmea_report()