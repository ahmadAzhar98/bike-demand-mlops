"""Load a registry model by alias and score it on online features from Feast."""

import os

import mlflow
import pandas as pd
from mlflow import MlflowClient

from src.features.training_data import get_store
from src.training.registry import MODEL_NAME

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


class Predictor:
    def __init__(self, alias: str = "champion"):
        mlflow.set_tracking_uri(TRACKING_URI)
        self.version = MlflowClient().get_model_version_by_alias(MODEL_NAME, alias)
        self.model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")
        self.schema = self.model.metadata.get_input_schema()
        # The registry tag says which feature set this version was trained on.
        self.store = get_store()
        self.feature_service = self.store.get_feature_service(self.version.tags["feature_service"])

    def align_to_signature(self, features: pd.DataFrame) -> pd.DataFrame:
        """Select and cast columns to the model's training-time schema. The online and
        offline stores can return different dtypes for the same feature."""
        types = dict(zip(self.schema.input_names(), self.schema.numpy_types()))
        return features[list(types)].astype(types)

    def predict(self, station_ids: list[str], when: pd.Timestamp) -> pd.DataFrame:
        online = self.store.get_online_features(
            features=self.feature_service,
            entity_rows=[{"station_id": s, "city": "nyc", "prediction_time": when} for s in station_ids],
        ).to_df()
        online["predicted_rentals"] = self.model.predict(self.align_to_signature(online))
        return online
