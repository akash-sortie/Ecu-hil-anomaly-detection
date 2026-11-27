"""
Evaluation Pipeline
Injects synthetic faults into the baseline, runs the ONNX model, 
and calculates Precision, Recall, and F1 Score.
"""
import pandas as pd
import numpy as np
import joblib
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix
from src.ingestion.fault_injector import inject_random_faults
from src.features.pipeline import create_sliding_windows

# The exact threshold your model calculated in Step 2c!
THRESHOLD = 0.2598  

def evaluate_model():
    print("🧪 --- Starting AI QA Evaluation ---")
    
    # 1. Load the clean baseline data
    df = pd.read_csv("data/raw/baseline_drive_cycle.csv")
    feature_cols = [c for c in df.columns if c != 'timestamp']
    
    # 2. Inject random, nasty faults into the clean data
    print("💉 Injecting simulated hardware faults (voltage spikes, dropouts, etc.)...")
    df_faulty, labels_df = inject_random_faults(df[feature_cols], fault_rate=0.01)
    
    # 3. Map exactly where the faults are (Ground Truth)
    is_anomaly = np.zeros(len(df_faulty))
    for _, row in labels_df.iterrows():
        is_anomaly[row['start_idx']:row['end_idx']] = 1
        
    # 4. Process data exactly like we do in production
    scaler = joblib.load("data/processed/scaler.pkl")
    data_scaled = scaler.transform(df_faulty.values)
    
    seq_len = 50
    stride = 10
    sequences = create_sliding_windows(data_scaled, seq_len, stride)
    
    # If a fault happens anywhere inside a 500ms window, label that whole window as an Anomaly (1)
    window_labels = []
    for i in range(0, len(is_anomaly) - seq_len + 1, stride):
        window_labels.append(1 if np.sum(is_anomaly[i : i + seq_len]) > 0 else 0)
    window_labels = np.array(window_labels)
    
    # 5. Load the High-Speed ONNX Model
    print("🧠 Loading ONNX Deep Learning Model...")
    session = ort.InferenceSession("mlflow/models/lstm_ae.onnx")
    
    # 6. Run Inference
    print("🔍 Scanning telemetry for anomalies...")
    predictions = []
    for seq in sequences:
        # Reshape to match the expected (batch, seq_len, features) shape
        x = np.array(seq, dtype=np.float32)[np.newaxis, ...] 
        x_hat = session.run(None, {"input": x})[0]
        
        # Calculate Reconstruction Error (MSE)
        mse = np.mean((x - x_hat) ** 2)
        predictions.append(1 if mse > THRESHOLD else 0)
        
    # 7. Print the Report Card!
    print("\n📊 --- Final Evaluation Report ---")
    print(classification_report(window_labels, predictions, target_names=["Healthy (0)", "Anomaly (1)"]))
    
    print("Confusion Matrix:")
    print(confusion_matrix(window_labels, predictions))

if __name__ == "__main__":
    evaluate_model()