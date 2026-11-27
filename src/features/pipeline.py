"""
Feature Engineering Pipeline
Transforms raw continuous telemetry into sliding windows (for LSTM) 
and statistical aggregates (for Isolation Forest).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def create_sliding_windows(data, seq_len, stride):
    """Slices continuous data into overlapping 3D windows."""
    windows = []
    # Slide across the data, grabbing chunks of 'seq_len'
    for i in range(0, len(data) - seq_len + 1, stride):
        windows.append(data[i : i + seq_len])
    return np.array(windows)

def extract_statistical_features(sequences, feature_names):
    """Compresses 3D sequence windows into 2D statistical summaries for classical ML."""
    # Calculate mean and standard deviation across the time steps (axis 1)
    means = np.mean(sequences, axis=1)
    stds = np.std(sequences, axis=1)

    # Naming the new columns (e.g., engine_rpm_mean, engine_rpm_std)
    cols = []
    for f in feature_names:
        cols.extend([f"{f}_mean", f"{f}_std"])

    # Horizontally stack them together
    features = np.hstack([means, stds])
    return pd.DataFrame(features, columns=cols)

def run_pipeline():
    print("⚙️ Starting Feature Engineering Pipeline...")

    # 1. Load raw data
    raw_path = Path("data/raw/baseline_drive_cycle.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"Cannot find {raw_path}. Run data_generator.py first.")
        
    df = pd.read_csv(raw_path)
    feature_cols = [c for c in df.columns if c != 'timestamp']
    data = df[feature_cols].values
    print(f"Loaded raw data: {df.shape[0]} rows, {len(feature_cols)} channels.")

    # 2. Scale the data (Crucial for LSTM and Isolation Forest)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Save the scaler so we can use the exact same math during real-time inference
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, "data/processed/scaler.pkl")
    print("Data normalized and StandardScaler saved.")

    # 3. Create Sequences for LSTM
    seq_len = 50  # 50 steps @ 100Hz = 500ms window
    stride = 10   # Slide by 10 steps (100ms) to create slight overlap
    sequences = create_sliding_windows(data_scaled, seq_len, stride)
    
    # Save LSTM sequences
    np.save("data/processed/baseline_sequences.npy", sequences)
    print(f"Created 3D sequences for LSTM. Shape: {sequences.shape} (Windows, Steps, Channels)")

    # 4. Extract Statistical Features for Isolation Forest
    stat_features_df = extract_statistical_features(sequences, feature_cols)
    
    # Save Classical features
    Path("data/features").mkdir(parents=True, exist_ok=True)
    stat_features_df.to_csv("data/features/baseline_features.csv", index=False)
    print(f"Extracted 2D statistical features for Isolation Forest. Shape: {stat_features_df.shape}")

    print("✅ Feature engineering complete! Ready for model training.")

if __name__ == "__main__":
    run_pipeline()