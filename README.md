# ECU HIL Signal Anomaly Detection

> Production-grade ML system for detecting anomalies in ECU Hardware-in-the-Loop telemetry.

## Architecture
```
ECU HIL Testbench → Kafka → Feature Pipeline → 3-Layer Detector → FastAPI → Grafana
```

## 3-Layer Detection Stack
1. **Statistical** — Z-score + IQR (fast, interpretable baseline)
2. **Classical ML** — Isolation Forest + LOF (multivariate, no labels needed)
3. **Deep Learning** — LSTM Autoencoder (temporal patterns, reconstruction error)

## Quick Start
```bash
make install        # Install dependencies
make infra-up       # Start Kafka, InfluxDB, Grafana, MLflow
make inject-faults  # Generate labelled training data
make train          # Train LSTM Autoencoder
make serve          # Start inference API on :8000
```

## Key Metrics (Target)
- Detection latency: < 50ms
- F1 score on injected faults: > 92%
- False positive rate: < 3%

## Dataset Sources
- Kaggle CAN Bus Intrusion Detection Dataset
- NASA CMAPSS Turbofan Prognostics
- OTIDS In-Vehicle CAN Dataset
- Synthetic via CARLA Simulator + fault_injector.py

## Interview Talking Points
- SHAP explainability → maps to Bosch FMEA process
- Sub-50ms inference via ONNX runtime optimization
- Evidently AI for model drift monitoring in CI
- 3-layer ensemble reduces false positives vs single model
