"""Prediction API: champion model from the MLflow registry + online features from Feast.

Usage:
    uvicorn src.serving.app:app --port 8000
"""

import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.serving.predictor import Predictor

ROOT = Path(__file__).resolve().parents[2]
PREDICTION_LOG = Path(os.environ.get("PREDICTION_LOG", ROOT / "logs" / "predictions.jsonl"))
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
# The data ends on 2025-06-01, so "now" defaults to the first hour after it.
DEFAULT_PREDICTION_TIME = os.environ.get("DEFAULT_PREDICTION_TIME", "2025-06-01T04:00:00Z")

state: dict[str, Predictor] = {}
log_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["predictor"] = Predictor(MODEL_ALIAS)
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    yield
    state.clear()


app = FastAPI(title="Bike demand API", lifespan=lifespan)


class PredictRequest(BaseModel):
    station_ids: list[str] = Field(min_length=1, max_length=100, examples=[["6140.05", "5329.03"]])
    prediction_time: datetime | None = Field(
        default=None, description="Hour to predict (UTC). Defaults to DEFAULT_PREDICTION_TIME."
    )


class StationPrediction(BaseModel):
    station_id: str
    predicted_rentals: float


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    prediction_time: datetime
    predictions: list[StationPrediction]


def model_info(p: Predictor) -> dict:
    return {
        "model_name": p.version.name,
        "model_version": p.version.version,
        "alias": MODEL_ALIAS,
        "run_name": p.version.tags.get("run_name"),
        "feature_service": p.feature_service.name,
        "features": p.schema.input_names(),
    }


def log_predictions(p: Predictor, when: pd.Timestamp, rows: pd.DataFrame) -> None:
    """Append request features + predictions; monitoring reads this file for drift."""
    logged_at = datetime.now(timezone.utc).isoformat()
    cols = ["station_id", *p.schema.input_names(), "predicted_rentals"]
    with log_lock, PREDICTION_LOG.open("a") as f:
        for record in rows[cols].to_dict(orient="records"):
            f.write(json.dumps({
                "logged_at": logged_at,
                "prediction_time": when.isoformat(),
                "model_version": p.version.version,
                **record,
            }, default=float) + "\n")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in state}


@app.get("/model")
def get_model():
    return model_info(state["predictor"])


@app.post("/reload")
def reload_model():
    """Re-resolve the alias, e.g. after a promotion or rollback in the registry."""
    old = state["predictor"].version.version
    state["predictor"] = Predictor(MODEL_ALIAS)
    return {"previous_version": old, **model_info(state["predictor"])}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    p = state["predictor"]
    when = pd.Timestamp(req.prediction_time or DEFAULT_PREDICTION_TIME)
    when = when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
    station_ids = list(dict.fromkeys(req.station_ids))

    rows = p.predict(station_ids, when)
    # Feast returns nulls for keys it has never seen; static station info is always set otherwise.
    unknown = rows.loc[rows["latitude"].isna(), "station_id"].tolist()
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown station_id(s): {unknown}")

    log_predictions(p, when, rows)
    return PredictResponse(
        model_name=p.version.name,
        model_version=p.version.version,
        prediction_time=when.to_pydatetime(),
        predictions=[
            StationPrediction(station_id=s, predicted_rentals=round(float(v), 2))
            for s, v in zip(rows["station_id"], rows["predicted_rentals"])
        ],
    )
