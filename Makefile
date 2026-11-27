.PHONY: install infra-up infra-down train serve test mlflow-ui inject-faults

install:
	pip install -r requirements.txt

infra-up:
	docker-compose -f infra/docker-compose.yml up -d
	@echo "Services: Kafka:9092 | InfluxDB:8086 | Grafana:3000 | MLflow:5000"

infra-down:
	docker-compose -f infra/docker-compose.yml down

train:
	python -m src.models.deep.lstm_autoencoder --config configs/model_config.yaml

serve:
	uvicorn src.inference.api:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --tb=short

mlflow-ui:
	mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db

inject-faults:
	python src/ingestion/fault_injector.py --mode all --output data/labels/injected_faults.csv
