from datetime import timedelta
from pathlib import Path

import pandas as pd
from feast import Entity, FeatureService, FeatureView, Field, FileSource, RequestSource, ValueType
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64, String, UnixTimestamp

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# --- Entities: the keys features are looked up by ---------------------------

station = Entity(
    name="station",
    join_keys=["station_id"],
    value_type=ValueType.STRING,
    description="A Citi Bike docking station",
)

city = Entity(
    name="city",
    join_keys=["city"],
    value_type=ValueType.STRING,
    description="Weather is measured per city, shared by all its stations",
)

# --- Sources -----------------------------------------------------------------


def parquet_source(name: str) -> FileSource:
    return FileSource(
        name=name,
        path=str(PROCESSED_DIR / f"{name}.parquet"),
        timestamp_field="event_timestamp",
    )


# --- Feature views -----------------------------------------------------------
# ttl = how old a row may be and still be returned for a given timestamp.

station_demand_stats = FeatureView(
    name="station_demand_stats",
    entities=[station],
    ttl=timedelta(hours=2),
    schema=[
        Field(name="rentals_lag_1h", dtype=Float64),
        Field(name="rentals_lag_24h", dtype=Float64),
        Field(name="rentals_lag_168h", dtype=Float64),
        Field(name="rentals_sum_24h", dtype=Float64),
        Field(name="rentals_mean_7d", dtype=Float64),
    ],
    source=parquet_source("station_demand_stats"),
    tags={"owner": "demand-forecasting"},
)

station_info = FeatureView(
    name="station_info",
    entities=[station],
    ttl=timedelta(days=365),
    schema=[
        Field(name="station_name", dtype=String),
        Field(name="latitude", dtype=Float64),
        Field(name="longitude", dtype=Float64),
    ],
    source=parquet_source("stations"),
)

weather_hourly = FeatureView(
    name="weather_hourly",
    entities=[city],
    ttl=timedelta(hours=2),
    schema=[
        Field(name="temperature_c", dtype=Float64),
        Field(name="humidity_pct", dtype=Int64),
        Field(name="precipitation_mm", dtype=Float64),
        Field(name="wind_speed_kmh", dtype=Float64),
        Field(name="weather_code", dtype=Int64),
    ],
    source=parquet_source("weather_hourly"),
)

# --- On-demand features: computed at request time, nothing stored ------------

prediction_request = RequestSource(
    name="prediction_request",
    schema=[Field(name="prediction_time", dtype=UnixTimestamp)],
)


@on_demand_feature_view(
    sources=[prediction_request],
    schema=[
        Field(name="hour_of_day", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
        Field(name="hour_sin", dtype=Float32),
        Field(name="hour_cos", dtype=Float32),
    ],
    mode="pandas",
)
def time_features(inputs: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    local = pd.to_datetime(inputs["prediction_time"], utc=True).dt.tz_convert("America/New_York")
    out = pd.DataFrame(index=inputs.index)
    out["hour_of_day"] = local.dt.hour.astype("int64")
    out["day_of_week"] = local.dt.dayofweek.astype("int64")
    out["is_weekend"] = (local.dt.dayofweek >= 5).astype("int64")
    # Cyclical encoding so 23:00 and 00:00 are close together.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour_of_day"] / 24).astype("float32")
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_of_day"] / 24).astype("float32")
    return out


# --- Feature service: the exact feature set a model version consumes ---------

demand_model_v1 = FeatureService(
    name="demand_model_v1",
    features=[
        station_demand_stats,
        station_info[["latitude", "longitude"]],
        weather_hourly,
        time_features,
    ],
    description="Features for the hourly station demand model",
)
