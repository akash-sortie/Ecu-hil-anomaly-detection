"""
Live Visual HIL Test Rig
Streams data to the AI API, then pushes the results into InfluxDB for Grafana.
"""
import time
import requests
import pandas as pd
import joblib
from influxdb import InfluxDBClient
import warnings
warnings.filterwarnings('ignore')

def start_visual_rig():
    print("🏎️ Starting Visual HIL Test Rig...")
    
    # 1. Connect to your Dockerized Database
    print("🗄️ Connecting to InfluxDB...")
    client = InfluxDBClient(host='localhost', port=8086)
    try:
        client.create_database('bosch_ecu')
    except:
        pass # Ignore if it already exists
    client.switch_database('bosch_ecu')

    # 2. Load the Engine Telemetry
    df = pd.read_csv("data/raw/baseline_drive_cycle.csv")
    feature_cols = [c for c in df.columns if c != 'timestamp']
    raw_data = df[feature_cols].values
    scaler = joblib.load("data/processed/scaler.pkl")
    scaled_data = scaler.transform(raw_data)
    
    print("📈 Data flowing to InfluxDB. Open your Grafana dashboard!\n")
    
    # 3. Stream Data (Accelerated for visual effect)
    for i in range(0, len(scaled_data) - 50, 5):
        window = scaled_data[i : i + 50].tolist()
        payload = {
            "window": window,
            "signal_ids": feature_cols,
            "timestamp": "live",
            "threshold": 0.2598
        }
        
        try:
            res = requests.post("http://localhost:8000/predict", json=payload).json()
            score = float(res["score"])
            rpm = float(raw_data[i+49][1])
            temp = float(raw_data[i+49][2])
            
            # Package the telemetry for the dashboard
            json_body = [{
                "measurement": "telemetry",
                "fields": {
                    "ai_score": score,
                    "engine_rpm": rpm,
                    "coolant_temp": temp
                }
            }]
            client.write_points(json_body) # Push to Grafana!
            
            status = "🚨 ANOMALY " if res["anomaly"] else "✅ HEALTHY "
            print(f"📡 [DB PUSH] {status} | AI Score: {score:.4f} | RPM: {rpm:.0f}")
            time.sleep(0.1)  # 100ms delay for a smooth visual graph
            
        except requests.exceptions.ConnectionError:
            print("❌ Error: Is your FastAPI server still running?")
            break
        except Exception as e:
            print(f"❌ Database Error: {e}")
            break

if __name__ == "__main__":
    start_visual_rig()