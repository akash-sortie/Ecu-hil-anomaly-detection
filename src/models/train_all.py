"""
Master Training Orchestrator
Trains the Isolation Forest baseline, then the LSTM Autoencoder, and logs to MLflow.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 🚨 CRITICAL: torch MUST be imported before numpy and pandas on Windows
import torch

import yaml
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest

# Import your existing LSTM training function
from src.models.deep.lstm_autoencoder import train as train_lstm

# ... rest of your code below (train_iforest, run_lstm, etc.) ...

def train_iforest(config):
    print("\n🌲 --- Training Isolation Forest Baseline ---")
    df = pd.read_csv("data/features/baseline_features.csv")
    X = df.values
    
    mlflow.set_experiment("ECU_HIL_Classical_Baselines")
    with mlflow.start_run(run_name="Isolation_Forest"):
        mlflow.log_params(config)
        
        clf = IsolationForest(
            n_estimators=config["n_estimators"],
            contamination=config["contamination"],
            max_features=config["max_features"],
            random_state=config["random_state"]
        )
        clf.fit(X)
        
        # Log to MLflow and save locally for the API
        mlflow.sklearn.log_model(clf, "isolation_forest")
        Path("mlflow/models").mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, "mlflow/models/isolation_forest.pkl")
        
        print("✅ Isolation Forest trained and saved.")

def run_lstm(config):
    print("\n🧠 --- Training LSTM Autoencoder ---")
    sequences = np.load("data/processed/baseline_sequences.npy")
    
    # 80/20 Train-Validation Split
    split_idx = int(len(sequences) * 0.8)
    train_data = torch.tensor(sequences[:split_idx], dtype=torch.float32)
    val_data = torch.tensor(sequences[split_idx:], dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(train_data), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=config["batch_size"], shuffle=False)
    
    # Safely override config to use CPU if no GPU is found
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config["device"] = device
    print(f"Hardware selected: {device.upper()}")
    
    # Trigger the scaffolded train function from lstm_autoencoder.py
    model, threshold = train_lstm(config, train_loader, val_loader)
    
    print("\n📦 Exporting LSTM to ONNX for high-speed FastAPI inference...")
    dummy_input = torch.randn(1, config["seq_len"], config["n_features"]).to(device)
    model.eval()
    
    onnx_path = Path("mlflow/models/lstm_ae.onnx")
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ LSTM exported successfully to {onnx_path}")

if __name__ == "__main__":
    # Load hyperparameter configurations
    with open("configs/model_config.yaml", "r") as f:
        configs = yaml.safe_load(f)
        
    train_iforest(configs["isolation_forest"])
    run_lstm(configs["lstm_autoencoder"])
    
    print("\n🚀 All models successfully trained and logged!")