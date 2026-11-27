"""
FastAPI Inference Endpoint — ECU HIL Anomaly Detection
POST /predict  →  { anomaly: bool, score: float, signals: [...], explanation: {...} }
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import json
from pathlib import Path

app = FastAPI(title="ECU HIL Anomaly Detector", version="1.0.0")

# Load ONNX model at startup
MODEL_PATH = Path("mlflow/models/lstm_ae.onnx")
session = None

@app.on_event("startup")
async def load_model():
    global session
    if MODEL_PATH.exists():
        session = ort.InferenceSession(str(MODEL_PATH))

class SignalWindow(BaseModel):
    window: list[list[float]]   # shape: (seq_len, n_features)
    signal_ids: list[str]
    timestamp: str
    threshold: float = 0.05     # reconstruction error threshold

class AnomalyResult(BaseModel):
    anomaly: bool
    score: float
    severity: str
    signal_ids: list[str]
    timestamp: str

@app.post("/predict", response_model=AnomalyResult)
async def predict(req: SignalWindow):
    if session is None:
        raise HTTPException(503, "Model not loaded")

    x = np.array(req.window, dtype=np.float32)[np.newaxis]  # (1, seq_len, n_features)
    x_hat = session.run(None, {"input": x})[0]
    score = float(np.mean((x - x_hat) ** 2))
    anomaly = score > req.threshold
    severity = "critical" if score > req.threshold * 3 else "warning" if anomaly else "normal"

    return AnomalyResult(
        anomaly=anomaly, score=round(score, 6),
        severity=severity, signal_ids=req.signal_ids,
        timestamp=req.timestamp
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": session is not None}
