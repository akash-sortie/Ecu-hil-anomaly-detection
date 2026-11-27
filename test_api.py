"""
API Test Client
Simulates an ECU test rig sending a 500ms window of telemetry to the AI.
"""
import requests
import numpy as np
import json

# 1. Generate a fake 500ms sliding window (50 time steps, 6 sensor channels)
# In production, this data would come directly from the CAN bus decoder
fake_window = np.random.rand(50, 6).tolist()

# 2. Package the JSON payload exactly how the FastAPI endpoint expects it
payload = {
    "window": fake_window,
    "signal_ids": [
        "throttle_position", "engine_rpm", "coolant_temp", 
        "battery_voltage", "map_sensor", "lambda_sensor"
    ],
    "timestamp": "2026-06-21T12:30:00Z",
    "threshold": 0.2598  # Your freshly calculated threshold!
}

print("📡 Sending 500ms telemetry window to Bosch ECU AI...")

# 3. POST the data to the local server
try:
    response = requests.post("http://localhost:8000/predict", json=payload)
    response.raise_for_status()
    
    print("\n📥 AI Response:")
    print(json.dumps(response.json(), indent=2))
    
except requests.exceptions.ConnectionError:
    print("❌ Error: Could not connect. Is the Uvicorn server running in your other terminal?")