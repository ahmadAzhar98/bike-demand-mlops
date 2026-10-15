"""Compare a production batch against the training data and decide whether to retrain.

Reference = the champion's training window (May 2025), current = a later month
prepared into data/monitoring/<month>/ for the same stations.

Usage:
    python -m src.data.prepare --month 2025-01 --stations-from data/processed/stations.parquet \
        --out-dir data/monitoring/2025-01
    python -m src.monitoring.monitor --month 2025-01
"""

import argparse
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset

from src.features.training_data import TARGET, get_training_frame
from src.serving.predictor import Predictor
from src.training.train import split

ROOT = Path(__file__).resolve().parents[2]
MONITORING_DIR = ROOT / "data" / "monitoring"
REPORTS_DIR = ROOT / "reports"
EXPERIMENT = "bike-demand-monitoring"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Retraining triggers.
DRIFT_SHARE_LIMIT = 0.5      # share of features whose distribution drifted
NMAE_RATIO_LIMIT = 1.25      # current normalized MAE vs. reference
MISSING_SHARE_LIMIT = 0.05   # share of missing values in any feature
OUT_OF_RANGE_LIMIT = 0.10    # share of values outside the training min/max in any feature


def current_frame(month: str, predictor: Predictor) -> pd.DataFrame:
    """Rebuild the model's features for a monitoring month the same way Feast does."""
    d = MONITORING_DIR / month
    df = (
        pd.read_parquet(d / "station_hourly_rentals.parquet")
        .merge(pd.read_parquet(d / "station_demand_stats.parquet"), on=["station_id", "event_timestamp"])
        .merge(pd.read_parquet(d / "stations.parquet")[["station_id", "latitude", "longitude"]], on="station_id")
        .merge(pd.read_parquet(d / "weather_hourly.parquet").drop(columns="city"), on="event_timestamp")
    )
    # Reuse the feature store's own on-demand transformation to avoid training/serving skew.
    udf = predictor.store.get_on_demand_feature_view("time_features").feature_transformation.udf
    time_feats = udf(pd.DataFrame({"prediction_time": df["event_timestamp"]}))
    df = pd.concat([df, time_feats], axis=1)
    # Same warm-up rule as training: a full week of lags must exist.
    return df[df["rentals_lag_168h"].notna()].reset_index(drop=True)


def score(df: pd.DataFrame, predictor: Predictor) -> pd.DataFrame:
    df = df.copy()
    df["prediction"] = np.clip(predictor.model.predict(predictor.align_to_signature(df)), 0, None)
    return df


def performance(df: pd.DataFrame) -> dict[str, float]:
    err = df["prediction"] - df[TARGET]
    mae = float(err.abs().mean())
    return {
        "mae": mae,
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),  # > 0 means over-predicting
        # MAE alone looks better whenever demand drops; normalize by the actual level.
        "nmae": mae / float(df[TARGET].mean()),
        "mean_actual": float(df[TARGET].mean()),
        "mean_predicted": float(df["prediction"].mean()),
    }


def quality(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    lo, hi = ref[features].min(), ref[features].max()
    return pd.DataFrame({
        "missing_share": cur[features].isna().mean(),
        "out_of_range_share": ((cur[features] < lo) | (cur[features] > hi)).mean(),
    })


def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str], path: Path):
    definition = DataDefinition(
        numerical_columns=features + [TARGET, "prediction"],
        regression=[Regression(target=TARGET, prediction="prediction")],
    )
    cols = features + [TARGET, "prediction"]
    report = Report([DataDriftPreset(columns=features), DataSummaryPreset(), RegressionPreset()])
    snapshot = report.run(
        current_data=Dataset.from_pandas(cur[cols], data_definition=definition),
        reference_data=Dataset.from_pandas(ref[cols], data_definition=definition),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
    return snapshot


def drift_summary(snapshot) -> dict:
    """Pull the drift share and per-feature drift scores out of the Evidently snapshot.
    Metric names look like ValueDrift(column=x,method=Wasserstein distance (normed),threshold=0.1)."""
    share, scores = None, {}
    for m in snapshot.dict()["metrics"]:
        name, value = m["metric_name"], m["value"]
        if name.startswith("DriftedColumnsCount"):
            share = float(value["share"])
        elif name.startswith("ValueDrift("):
            params = dict(p.split("=", 1) for p in name[len("ValueDrift("):-1].split(","))
            threshold = float(params["threshold"])
            # p-value methods drift when the value is small; distance methods when it is large.
            drifted = value < threshold if "p_value" in params["method"] else value > threshold
            scores[params["column"]] = {"method": params["method"], "score": round(float(value), 4), "drifted": bool(drifted)}
    return {
        "drift_share": share,
        "drifted_features": sorted(c for c, s in scores.items() if s["drifted"]),
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True, help="monitoring month prepared under data/monitoring/")
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()

    predictor = Predictor(args.alias)
    features = predictor.schema.input_names()

    parts = split(get_training_frame(predictor.store))
    # Drift is measured against what the model learned from; performance against held-out
    # data, since in-sample error is optimistic.
    reference = score(parts["train"], predictor)
    holdout = score(parts["test"], predictor)
    current = score(current_frame(args.month, predictor), predictor)

    report_path = REPORTS_DIR / f"monitoring_{args.month}_v{predictor.version.version}.html"
    snapshot = evidently_report(reference, current, features, report_path)
    drift = drift_summary(snapshot)
    ref_perf, cur_perf = performance(holdout), performance(current)
    dq = quality(reference, current, features)

    nmae_ratio = cur_perf["nmae"] / ref_perf["nmae"]
    reasons = []
    if drift["drift_share"] is not None and drift["drift_share"] >= DRIFT_SHARE_LIMIT:
        reasons.append(f"feature drift: {drift['drift_share']:.0%} of features drifted")
    if nmae_ratio >= NMAE_RATIO_LIMIT:
        reasons.append(f"performance: normalized MAE {nmae_ratio:.2f}x the reference")
    if (bad := dq.index[dq["missing_share"] > MISSING_SHARE_LIMIT].tolist()):
        reasons.append(f"data quality: missing values in {bad}")
    if (bad := dq.index[dq["out_of_range_share"] > OUT_OF_RANGE_LIMIT].tolist()):
        reasons.append(f"data quality: values outside training range in {bad}")
    decision = "retrain" if reasons else "ok"

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"{args.month}-v{predictor.version.version}"):
        mlflow.set_tags({
            "decision": decision,
            "model_name": predictor.version.name,
            "model_version": predictor.version.version,
            "feature_service": predictor.feature_service.name,
        })
        mlflow.log_params({"month": args.month, "current_rows": len(current), "reference_rows": len(reference)})
        mlflow.log_metrics({
            **{f"current_{k}": v for k, v in cur_perf.items()},
            **{f"reference_{k}": v for k, v in ref_perf.items()},
            "nmae_ratio": nmae_ratio,
            "drift_share": drift["drift_share"] or 0.0,
            "max_missing_share": float(dq["missing_share"].max()),
            "max_out_of_range_share": float(dq["out_of_range_share"].max()),
        })
        mlflow.log_artifact(str(report_path), artifact_path="evidently")
        mlflow.log_dict({"reasons": reasons, **drift, "quality": dq.round(4).to_dict()}, "monitoring_summary.json")

    print(f"\n== {args.month} vs training data, model v{predictor.version.version} ==")
    print(f"rows: reference {len(reference):,}, current {len(current):,}")
    print(f"drift share: {drift['drift_share']:.0%}  drifted: {drift['drifted_features']}")
    print(pd.DataFrame({"holdout (test week)": ref_perf, "current": cur_perf}).round(3).to_string())
    print(f"normalized MAE ratio: {nmae_ratio:.2f}")
    flagged = dq[(dq["missing_share"] >= 0.001) | (dq["out_of_range_share"] >= 0.001)]
    print("data quality (non-zero only):\n" + (flagged.round(3).to_string() if len(flagged) else "  all clean"))
    print(f"decision: {decision.upper()}" + "".join(f"\n  - {r}" for r in reasons))
    print(f"report: {report_path.relative_to(ROOT)}")
    print(json.dumps({"decision": decision}))


if __name__ == "__main__":
    main()
