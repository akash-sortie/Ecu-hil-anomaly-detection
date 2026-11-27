"""
Live ECU Rig Simulator
Streams scaled data from the baseline CSV to the FastAPI endpoint in real-time.
"""
import time
import requests
import pandas as pd
import joblib

def start_rig():
    print("🏎️ Starting ECU Live Test Rig...")
    print("🔌 Connecting to AI Inference API...\n")
    
    # 1. Load the raw data
    df = pd.read_csv("data/raw/baseline_drive_cycle.csv")
    feature_cols = [c for c in df.columns if c != 'timestamp']
    raw_data = df[feature_cols].values
    
    # 2. CRITICAL FIX: Scale the data using our saved training math
    print("⚖️ Applying StandardScaler to telemetry...")
    scaler = joblib.load("data/processed/scaler.pkl")
    scaled_data = scaler.transform(raw_data)
    
    # 3. Send sliding windows to the API continuously
    for i in range(0, len(scaled_data) - 50, 10):
        window = scaled_data[i : i + 50].tolist()
        payload = {
            "window": window,
            "signal_ids": feature_cols,
            "timestamp": "live_stream",
            "threshold": 0.2598
        }
        
        try:
            # Hit your local API
            res = requests.post("http://localhost:8000/predict", json=payload).json()
            
            # Format the terminal output for a cool real-time display
            status = "🚨 ANOMALY DETECTED " if res["anomaly"] else "✅ SYSTEM HEALTHY  "
            score = res["score"]
            
            print(f"📡 [DATA SENT] 500ms window | {status} | MSE Score: {score:.5f}")
            time.sleep(0.05)  # Slight delay to simulate a real-time data stream
            
        except requests.exceptions.ConnectionError:
            print("\n❌ Error: AI API is offline. Run 'uvicorn src.inference.api:app --reload' in another terminal first!")
            break

if __name__ == "__main__":
    start_rig()