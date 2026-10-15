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

## Roadmap

- [x] Phase 0 — environment and project layout
- [ ] Phase 1 — data preparation and Feast feature definitions
- [ ] Phase 2 — training with point-in-time features, tracked in MLflow
- [ ] Phase 3 — model registry: versions and aliases (`@champion`, `@challenger`)
- [ ] Phase 4 — FastAPI service: registry model + online features
- [ ] Phase 5 — monitoring: drift and feature quality
- [ ] Phase 6 — production-like stack (Postgres, MinIO, Redis, orchestration) with Docker Compose

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
