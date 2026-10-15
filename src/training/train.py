"""Train hourly demand models on Feast features and track them in MLflow.

Usage:
    python -m src.training.train --model baseline
    python -m src.training.train --model lightgbm --objective poisson --num-leaves 31
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.training_data import FEATURE_SERVICE, TARGET, get_store, get_training_frame

EXPERIMENT = "bike-demand"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Time-based split boundaries (UTC, = local midnight). Never shuffle time series:
# a random split would let the model peek at the future.
WARMUP_END = pd.Timestamp("2025-05-08 04:00", tz="UTC")  # first hour with a full week of lags
VALID_START = pd.Timestamp("2025-05-21 04:00", tz="UTC")
TEST_START = pd.Timestamp("2025-05-25 04:00", tz="UTC")

NON_FEATURES = {"station_id", "city", "event_timestamp", "prediction_time", TARGET}


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = df[df["event_timestamp"] >= WARMUP_END]
    return {
        "train": df[df["event_timestamp"] < VALID_START],
        "valid": df[(df["event_timestamp"] >= VALID_START) & (df["event_timestamp"] < TEST_START)],
        "test": df[df["event_timestamp"] >= TEST_START],
    }


def metrics(y_true, y_pred, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_r2": r2_score(y_true, y_pred),
    }


def plot_actual_vs_predicted(test: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    station = test["station_id"].value_counts().index[0]
    mask = (test["station_id"] == station).to_numpy()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test.loc[mask, "event_timestamp"], test.loc[mask, TARGET], label="actual", lw=1.2)
    ax.plot(test.loc[mask, "event_timestamp"], pred[mask], label="predicted", lw=1.2)
    ax.set_title(f"Test week, station {station}")
    ax.set_ylabel("rentals / hour")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importance(model: lgb.LGBMRegressor, features: list[str], path: Path) -> None:
    imp = pd.Series(model.booster_.feature_importance("gain"), index=features).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.plot.barh(ax=ax)
    ax.set_title("Feature importance (gain)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_baseline(parts: dict[str, pd.DataFrame]) -> None:
    """Seasonal naive: predict what happened at the same hour last week."""
    mlflow.log_params({"model_type": "seasonal_naive", "lag": "rentals_lag_168h"})
    for name in ("valid", "test"):
        part = parts[name]
        mlflow.log_metrics(metrics(part[TARGET], part["rentals_lag_168h"], name))


def run_lightgbm(parts: dict[str, pd.DataFrame], features: list[str], args) -> None:
    params = {
        "objective": args.objective,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }
    mlflow.log_params({"model_type": "lightgbm", **params})

    X = {name: part[features] for name, part in parts.items()}
    y = {name: part[TARGET] for name, part in parts.items()}

    model = lgb.LGBMRegressor(**params)
    # Early stopping watches the validation week only; the test week stays untouched.
    model.fit(
        X["train"], y["train"],
        eval_set=[(X["valid"], y["valid"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    mlflow.log_metric("best_iteration", model.best_iteration_)

    preds = {}
    for name in ("train", "valid", "test"):
        # Demand can't be negative; clip the l2 objective's occasional dips below zero.
        preds[name] = np.clip(model.predict(X[name]), 0, None)
        mlflow.log_metrics(metrics(y[name], preds[name], name))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        plot_actual_vs_predicted(parts["test"], preds["test"], tmp / "test_actual_vs_predicted.png")
        plot_importance(model, features, tmp / "feature_importance.png")
        mlflow.log_artifacts(str(tmp), artifact_path="plots")

    mlflow.lightgbm.log_model(
        model,
        name="model",
        signature=infer_signature(X["test"], preds["test"]),
        input_example=X["test"].head(5),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "lightgbm"], default="lightgbm")
    parser.add_argument("--objective", default="poisson", help="lightgbm objective: poisson or regression")
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    store = get_store()
    df = get_training_frame(store)
    parts = split(df)
    features = [c for c in df.columns if c not in NON_FEATURES]

    run_name = args.run_name or (
        "seasonal-naive" if args.model == "baseline" else f"lgbm-{args.objective}-{args.num_leaves}leaves"
    )
    with mlflow.start_run(run_name=run_name):
        # Lineage: which features, from where, over which window.
        mlflow.set_tags({
            "feature_service": FEATURE_SERVICE,
            "feast_project": store.project,
            "target": TARGET,
        })
        mlflow.log_params({
            "train_rows": len(parts["train"]),
            "valid_rows": len(parts["valid"]),
            "test_rows": len(parts["test"]),
            "valid_start": VALID_START.isoformat(),
            "test_start": TEST_START.isoformat(),
        })
        mlflow.log_dict({"features": features}, "features.json")
        mlflow.log_input(
            mlflow.data.from_pandas(parts["train"], name="feast_demand_model_v1_train", targets=TARGET),
            context="training",
        )

        if args.model == "baseline":
            run_baseline(parts)
        else:
            run_lightgbm(parts, features, args)

        summary = {k: round(v, 3) for k, v in mlflow.get_run(mlflow.active_run().info.run_id).data.metrics.items()}
        print(run_name, json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
