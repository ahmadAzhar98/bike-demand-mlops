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
```
