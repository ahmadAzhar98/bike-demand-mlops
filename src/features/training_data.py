"""Build point-in-time correct training data from the Feast offline store.

Usage:
    python -m src.features.training_data
"""

from pathlib import Path

import pandas as pd
from feast import FeatureStore

ROOT = Path(__file__).resolve().parents[2]
FEATURE_REPO = ROOT / "feature_repo"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURE_SERVICE = "demand_model_v1"
TARGET = "rentals"


def get_store() -> FeatureStore:
    return FeatureStore(repo_path=str(FEATURE_REPO))


def entity_df() -> pd.DataFrame:
    """One row per (station, hour) we want a prediction for, with its label.
    Feast joins each feature view as of `event_timestamp`."""
    labels = pd.read_parquet(PROCESSED_DIR / "station_hourly_rentals.parquet")
    labels["city"] = "nyc"
    labels["prediction_time"] = labels["event_timestamp"]
    return labels


def get_training_frame(store: FeatureStore | None = None) -> pd.DataFrame:
    store = store or get_store()
    job = store.get_historical_features(
        entity_df=entity_df(),
        features=store.get_feature_service(FEATURE_SERVICE),
    )
    return job.to_df().sort_values(["station_id", "event_timestamp"]).reset_index(drop=True)


def main() -> None:
    df = get_training_frame()
    print(f"training frame: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(df.columns.tolist())
    print(df.head(3).T.to_string())


if __name__ == "__main__":
    main()
