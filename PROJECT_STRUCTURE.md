# ECU HIL Anomaly Detection — Production Project Structure

## Directory Tree

```
ecu_hil_anomaly/
│
├── 📁 data/
│   ├── raw/                        # Raw CAN logs, HIL exports (.blf, .csv, .asc)
│   ├── processed/                  # Decoded + resampled signal DataFrames
│   ├── features/                   # Engineered feature sets (parquet)
│   ├── labels/                     # Injected fault labels (CSV)
│   └── external/                   # Downloaded benchmark datasets
│
├── 📁 src/
│   ├── ingestion/
│   │   ├── can_parser.py           # Decode raw CAN frames using cantools + DBC
│   │   ├── kafka_producer.py       # Stream signals via Kafka topic
│   │   ├── signal_resampler.py     # Upsample/downsample to uniform 10ms windows
│   │   └── fault_injector.py       # Inject synthetic faults for labelling
│   │
│   ├── features/
│   │   ├── time_domain.py          # Mean, std, kurtosis, skewness per window
│   │   ├── frequency_domain.py     # FFT magnitude, dominant freq, spectral entropy
│   │   ├── statistical.py          # Rolling Z-score, IQR, Mahalanobis distance
│   │   └── pipeline.py             # tsfresh-based feature extraction pipeline
│   │
│   ├── models/
│   │   ├── statistical/
│   │   │   └── zscore_detector.py  # Rule-based Z-score + IQR baseline
│   │   ├── classical/
│   │   │   ├── isolation_forest.py # Sklearn IsolationForest wrapper
│   │   │   ├── lof_detector.py     # Local Outlier Factor
│   │   │   └── ocsvm.py            # One-Class SVM
│   │   ├── deep/
│   │   │   ├── lstm_autoencoder.py # PyTorch LSTM Autoencoder (main model)
│   │   │   ├── tcn_detector.py     # Temporal Convolutional Network (optional)
│   │   │   └── transformer_ae.py   # Transformer Autoencoder (advanced)
│   │   └── ensemble.py             # Ensemble voting across all 3 layers
│   │
│   ├── inference/
│   │   ├── api.py                  # FastAPI inference endpoint
│   │   ├── sliding_window.py       # Stateful sliding window inference
│   │   ├── kafka_consumer.py       # Real-time Kafka → model → InfluxDB
│   │   └── onnx_exporter.py        # Export trained model to ONNX
│   │
│   ├── explainability/
│   │   ├── shap_explainer.py       # SHAP values per anomaly event
│   │   ├── fault_taxonomy.py       # Pattern → ECU subsystem mapping
│   │   └── report_generator.py     # Auto-generate PDF anomaly reports
│   │
│   └── monitoring/
│       ├── drift_detector.py       # Model input/output drift (Evidently AI)
│       ├── metrics_exporter.py     # Prometheus metrics push
│       └── alert_manager.py        # Threshold-based alert routing
│
├── 📁 notebooks/
│   ├── 01_eda_signal_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training_isolation_forest.ipynb
│   ├── 04_lstm_autoencoder_training.ipynb
│   ├── 05_ensemble_evaluation.ipynb
│   └── 06_shap_explainability.ipynb
│
├── 📁 configs/
│   ├── signals.yaml                # Signal channel definitions, units, thresholds
│   ├── model_config.yaml           # Hyperparameters for all models
│   ├── kafka_config.yaml           # Broker, topic, partition settings
│   └── alert_rules.yaml            # Severity levels, notification channels
│
├── 📁 tests/
│   ├── unit/
│   │   ├── test_can_parser.py
│   │   ├── test_feature_pipeline.py
│   │   └── test_models.py
│   └── integration/
│       ├── test_kafka_pipeline.py
│       └── test_inference_api.py
│
├── 📁 infra/
│   ├── docker-compose.yml          # Kafka, InfluxDB, Grafana, Redis, API
│   ├── Dockerfile.api              # FastAPI inference service
│   ├── Dockerfile.consumer         # Kafka consumer service
│   ├── grafana/
│   │   └── dashboards/
│   │       └── ecu_anomaly.json    # Pre-built Grafana dashboard
│   └── influxdb/
│       └── init.sh                 # InfluxDB bucket + token setup
│
├── 📁 mlflow/
│   ├── experiments/                # MLflow experiment tracking storage
│   └── models/                     # Registered model versions
│
├── requirements.txt
├── pyproject.toml
├── .env.example
├── Makefile                        # make train / make serve / make test
└── README.md
```

---

## Models — Recommended Stack

### Layer 1 — Statistical Baseline
| Model | Library | Purpose |
|---|---|---|
| Z-Score per channel | numpy/scipy | Fast rule-based threshold |
| IQR Fence | pandas | Robust to non-Gaussian signals |
| Mahalanobis Distance | scipy | Multivariate correlation anomalies |

### Layer 2 — Classical ML
| Model | Library | Notes |
|---|---|---|
| Isolation Forest | scikit-learn | Best for high-dim sparse anomalies |
| Local Outlier Factor | scikit-learn | Good for density-based clusters |
| One-Class SVM | scikit-learn | For well-defined normal envelope |
| HBOS | PyOD | Fast histogram-based, lightweight |

### Layer 3 — Deep Learning
| Model | Library | Notes |
|---|---|---|
| LSTM Autoencoder | PyTorch | Primary model — reconstruction error |
| TCN Autoencoder | PyTorch (tcn) | Alternative: no vanishing gradient |
| Transformer AE | PyTorch | Strongest — use if compute allows |

---

## Dataset Sources

### Primary (Use These First)
1. **Kaggle CAN Bus Intrusion Detection Dataset**
   URL: https://www.kaggle.com/datasets/captaindatasets/automotive-can-bus-security
   Format: CSV with CAN ID, DLC, Data bytes | Size: ~500MB

2. **NASA Prognostics Center — CMAPSS Turbofan**
   URL: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
   Format: txt sensor readings | Use: map to ECU sensor degradation patterns

3. **OTIDS: In-Vehicle Intrusion Detection System Dataset**
   URL: https://ocslab.hksecurity.net/Datasets/can-intrusion-dataset
   Format: CAN frames with attack labels | Size: ~1GB

4. **Synthetic Generation with CARLA Simulator**
   Install: pip install carla | Generate vehicle telemetry at 100Hz

5. **Bosch Production Quality Dataset (Kaggle)**
   URL: https://www.kaggle.com/competitions/bosch-production-line-performance
   Format: sensor readings from production line | Good for fault classification

### Fault Injection Script (generate your own labels)
```python
# fault_injector.py — inject 5 fault types into normal signal
FAULT_TYPES = {
    "voltage_spike":  lambda s: s * np.random.uniform(2.5, 4.0),
    "signal_dropout": lambda s: 0.0,
    "stuck_at":       lambda s: s.iloc[0],
    "noise_burst":    lambda s: s + np.random.normal(0, s.std() * 3),
    "timing_jitter":  lambda s: s.shift(np.random.randint(3, 8))
}
```

---

## Key Python Libraries — requirements.txt

```
# Data / CAN
cantools==39.4.5
python-can==4.3.1
pandas==2.2.2
numpy==1.26.4
pyarrow==16.0.0

# Feature Engineering
tsfresh==0.20.2
scipy==1.13.0

# ML Models
scikit-learn==1.5.0
pyod==1.1.3

# Deep Learning
torch==2.3.0
torchvision==0.18.0

# Streaming
kafka-python==2.0.2
confluent-kafka==2.4.0
redis==5.0.4

# Serving
fastapi==0.111.0
uvicorn==0.29.0
onnx==1.16.0
onnxruntime==1.18.0

# Explainability
shap==0.45.1
lime==0.2.0.1

# Monitoring / Tracking
mlflow==2.13.0
evidently==0.4.30
prometheus-client==0.20.0

# Storage
influxdb-client==1.44.0

# Reporting
reportlab==4.2.2
matplotlib==3.9.0
seaborn==0.13.2
```

---

## Prompt Pack for Gemini / Other AI Tools

Copy these when continuing with other AI assistants:

### Context Prompt (paste at start of new chat)
```
I am building a production-grade ML system called "Intelligent Signal Anomaly 
Detection in ECU Hardware-in-the-Loop (HIL) Telemetry" for a Bosch Automotive 
Electronics interview project.

TECH STACK:
- Data: CAN bus signals decoded via cantools + python-can DBC files
- Streaming: Apache Kafka → Redis
- Features: tsfresh + custom FFT/time-domain pipeline
- Models: 3-layer stack (Z-score → Isolation Forest → LSTM Autoencoder)
- Serving: FastAPI + ONNX + Docker
- Explainability: SHAP mapped to ECU fault taxonomy
- Monitoring: MLflow + Evidently AI + Grafana + InfluxDB

PROJECT STRUCTURE:
ecu_hil_anomaly/src/{ingestion, features, models, inference, explainability, monitoring}

CURRENT STATUS: [describe what you've done]
NEXT TASK: [describe what you need help with]
```

### Module-Specific Prompts
```
# For LSTM Autoencoder:
"Build a PyTorch LSTM Autoencoder for multivariate time series anomaly detection.
Input: (batch, seq_len=50, n_features=12). Output: reconstruction of same shape.
Include: encoder LSTM → bottleneck → decoder LSTM → linear projection.
Add threshold computation from validation reconstruction error (95th percentile)."

# For Kafka Pipeline:
"Write a Kafka consumer in Python (confluent-kafka) that reads ECU signal windows
from topic 'ecu-signals', runs ONNX inference, and writes anomaly events with
score + timestamp + signal_id to InfluxDB bucket 'anomalies'."

# For SHAP Explainability:
"Add SHAP TreeExplainer for IsolationForest and KernelExplainer for LSTM output.
Map top-3 contributing features to ECU subsystem using this taxonomy dict: {...}
Output a JSON report per anomaly event."
```

---

## Makefile Commands

```makefile
install:      pip install -r requirements.txt
infra-up:     docker-compose -f infra/docker-compose.yml up -d
train:        python src/models/lstm_autoencoder.py --config configs/model_config.yaml
serve:        uvicorn src.inference.api:app --reload --port 8000
test:         pytest tests/ -v
mlflow-ui:    mlflow ui --port 5000
inject-faults: python src/ingestion/fault_injector.py --mode all
```
