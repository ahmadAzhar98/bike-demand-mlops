# Bike Demand MLOps

Hourly bike-rental demand forecasting per Citi Bike station, built to learn production ML pipelines with a **feature store** and a **model registry**. Everything is open source and runs locally.

| Concern | Tool |
|---|---|
| Feature store | Feast |
| Experiment tracking + model registry | MLflow |
| Model | LightGBM / scikit-learn |
| Serving | FastAPI |
| Monitoring (drift, data quality) | Evidently |

Data: [Citi Bike trip data](https://citibikenyc.com/system-data) and [Open-Meteo](https://open-meteo.com/) historical weather.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
# macOS: LightGBM needs OpenMP
brew install libomp
```

## Get the data

```bash
python -m src.data.download --month 2025-05
python -m src.data.prepare --month 2025-05 --top-n 20
```

## Feature store

```bash
cd feature_repo && feast apply && cd ..        # register entities, feature views, services
python -m src.features.training_data            # point-in-time training frame from the offline store
cd feature_repo && feast materialize-incremental 2025-06-01T04:00:00 && cd ..   # load latest values into the online store
```

| Feature view | Entity | Features |
|---|---|---|
| `station_demand_stats` | station | rentals lag 1h / 24h / 168h, 24h sum, 7-day mean |
| `station_info` | station | latitude, longitude |
| `weather_hourly` | city | temperature, humidity, precipitation, wind, weather code |
| `time_features` (on-demand) | request | hour of day, day of week, weekend flag, cyclical hour |

The `demand_model_v1` feature service bundles these for the model.

## Training and experiment tracking

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 --port 5000            # UI at http://127.0.0.1:5000

python -m src.training.train --model baseline                         # seasonal naive: same hour last week
python -m src.training.train --model lightgbm --objective poisson
python -m src.training.train --model lightgbm --objective regression --num-leaves 63
```

Data is split by time: train May 8–20, validation May 21–24 (early stopping), test May 25–31.
Each run logs parameters, metrics (MAE, RMSE, R²), plots, the feature list, the training dataset and
the model with its signature, tagged with the Feast feature service it was trained on.

## Model registry

Versions of `bike-demand-model` are managed with aliases instead of the deprecated stages:

| Alias | Meaning | Old stage |
|---|---|---|
| `@champion` | serves production traffic | Production |
| `@challenger` | candidate under evaluation | Staging |
| `@previous` | last champion, for one-step rollback | — |
| *(tag `status=archived`)* | retired, no alias | Archived |

```bash
python -m src.training.registry register --best      # best LightGBM run by valid_mae -> @challenger
python -m src.training.registry promote              # challenger -> champion if it beats the champion on valid_mae
python -m src.training.registry promote --version 3 --force
python -m src.training.registry rollback             # swap @champion and @previous
python -m src.training.registry archive --version 3
python -m src.training.registry list
```

Each version is tagged with its `feature_service`, so serving fetches exactly the features the model was trained on:

```python
from src.serving.predictor import Predictor
Predictor("champion").predict(["6140.05"], pd.Timestamp("2025-06-01 04:00", tz="UTC"))
```

## Prediction API

```bash
uvicorn src.serving.app:app --port 8000      # docs at http://127.0.0.1:8000/docs

curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"station_ids": ["6140.05", "5329.03"]}'
```

| Endpoint | Purpose |
|---|---|
| `POST /predict` | next-hour rentals for stations; optional `prediction_time` (UTC) |
| `GET /model` | serving version, alias, feature service, feature list |
| `POST /reload` | re-resolve `@champion` after a promotion or rollback, no restart |
| `GET /health` | liveness |

Every prediction is appended to `logs/predictions.jsonl` with its features and model version, for monitoring.
`MODEL_ALIAS` (default `champion`) and `DEFAULT_PREDICTION_TIME` configure the service.

## Monitoring

A later month is prepared for the same stations, scored by `@champion` and compared with the training data:

```bash
python -m src.data.download --month 2025-01
python -m src.data.prepare --month 2025-01 --stations-from data/processed/stations.parquet \
  --out-dir data/monitoring/2025-01
python -m src.monitoring.monitor --month 2025-01
```

| Check | Compared against | Retrain trigger |
|---|---|---|
| Feature drift (Evidently, per-feature distance tests) | training window | ≥ 50% of features drifted |
| Performance (MAE normalized by mean demand) | held-out test week | ≥ 1.25× the reference |
| Data quality: missing values | — | > 5% in any feature |
| Data quality: values outside the training min/max | training window | > 10% in any feature |

Each run writes an Evidently HTML report to `reports/` and logs metrics, the decision and the report to the
`bike-demand-monitoring` MLflow experiment.

| Month | Drift share | Normalized MAE ratio | Decision |
|---|---|---|---|
| 2025-06 | 29% | 0.90 | retrain: 35% of temperatures above the training range |
| 2025-01 | 65% | 1.31 | retrain: drift, performance and out-of-range features |
