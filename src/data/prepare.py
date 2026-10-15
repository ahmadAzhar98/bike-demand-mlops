"""Turn raw trips + weather into hourly tables for the feature store.

Outputs (data/processed/):
    station_hourly_rentals.parquet  station_id, event_timestamp, rentals
    station_demand_stats.parquet    station_id, event_timestamp, rentals_lag_1h, ...
    stations.parquet                station_id, station_name, latitude, longitude, event_timestamp
    weather_hourly.parquet          city, event_timestamp, temperature_c, humidity_pct, ...

All timestamps are hour-start, timezone-aware UTC (what Feast expects).

Usage:
    python -m src.data.prepare --month 2025-05 --top-n 20
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
LOCAL_TZ = "America/New_York"

TRIP_COLUMNS = ["started_at", "start_station_id", "start_station_name", "start_lat", "start_lng"]
WEATHER_RENAME = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "precipitation": "precipitation_mm",
    "wind_speed_10m": "wind_speed_kmh",
    "weather_code": "weather_code",
}


def to_utc(local_times: pd.Series) -> pd.Series:
    """Citi Bike and our weather request are in NYC local time; store UTC."""
    return local_times.dt.tz_localize(
        LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")


def load_trips(yyyymm: str) -> pd.DataFrame:
    files = sorted((RAW_DIR / f"citibike_{yyyymm}").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No trip CSVs for {yyyymm}; run src.data.download first")
    trips = pd.concat(
        (pd.read_csv(f, usecols=TRIP_COLUMNS, dtype={"start_station_id": str}) for f in files),
        ignore_index=True,
    )
    trips = trips.dropna(subset=["start_station_id"])
    trips["started_at"] = pd.to_datetime(trips["started_at"], format="mixed")
    return trips


def build_tables(
    year: int, month: int, top_n: int, station_ids: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    yyyymm = f"{year}{month:02d}"
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end = month_start + pd.offsets.MonthBegin(1)

    trips = load_trips(yyyymm)
    # Files are grouped by trip end, so a few trips started in the previous month.
    trips = trips[(trips["started_at"] >= month_start) & (trips["started_at"] < month_end)]

    if station_ids is None:
        top_ids = trips["start_station_id"].value_counts().head(top_n).index
    else:
        top_ids = pd.Index(station_ids)
    trips = trips[trips["start_station_id"].isin(top_ids)]
    trips["hour"] = trips["started_at"].dt.floor("h")

    # Hours with no rentals are real zeros, not missing data, so fill the full grid.
    hours = pd.date_range(month_start, month_end, freq="h", inclusive="left")
    grid = pd.MultiIndex.from_product([top_ids, hours], names=["station_id", "hour"])
    rentals = (
        trips.groupby(["start_station_id", "hour"]).size()
        .rename_axis(["station_id", "hour"])
        .reindex(grid, fill_value=0)
        .rename("rentals")
        .reset_index()
    )
    rentals["event_timestamp"] = to_utc(rentals.pop("hour"))
    rentals = rentals.dropna(subset=["event_timestamp"])

    stations = (
        trips.groupby("start_station_id")
        .agg(
            station_name=("start_station_name", lambda s: s.mode().iat[0]),
            latitude=("start_lat", "median"),
            longitude=("start_lng", "median"),
        )
        .rename_axis("station_id")
        .reset_index()
    )
    # Station attributes are known from the start of the data window.
    stations["event_timestamp"] = rentals["event_timestamp"].min()

    raw_weather = json.loads((RAW_DIR / f"weather_{yyyymm}.json").read_text())["hourly"]
    weather = pd.DataFrame(raw_weather).rename(columns=WEATHER_RENAME)
    weather["event_timestamp"] = to_utc(pd.to_datetime(weather.pop("time")))
    weather = weather.dropna(subset=["event_timestamp"])
    # A single weather point covers all stations; "city" is the join key.
    weather.insert(0, "city", "nyc")

    return {
        "station_hourly_rentals": rentals,
        "station_demand_stats": demand_stats(rentals),
        "stations": stations,
        "weather_hourly": weather,
    }


def demand_stats(rentals: pd.DataFrame) -> pd.DataFrame:
    """Lagged demand features. The row stamped T only uses rentals from hours before T,
    so it is safe to use when predicting rentals for hour T."""
    df = rentals.sort_values(["station_id", "event_timestamp"]).copy()
    past = df.groupby("station_id")["rentals"].shift(1)
    by_station = past.groupby(df["station_id"])

    df["rentals_lag_1h"] = past
    df["rentals_lag_24h"] = df.groupby("station_id")["rentals"].shift(24)
    df["rentals_lag_168h"] = df.groupby("station_id")["rentals"].shift(168)
    df["rentals_sum_24h"] = by_station.transform(lambda s: s.rolling(24).sum())
    df["rentals_mean_7d"] = by_station.transform(lambda s: s.rolling(168).mean())

    return df.drop(columns="rentals")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True, help="YYYY-MM, e.g. 2025-05")
    parser.add_argument("--top-n", type=int, default=20, help="number of busiest stations to keep")
    parser.add_argument(
        "--stations-from", type=Path,
        help="stations.parquet whose station_ids to keep instead of the top N (e.g. the training set)",
    )
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    year, month = map(int, args.month.split("-"))
    station_ids = (
        pd.read_parquet(args.stations_from)["station_id"].tolist() if args.stations_from else None
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in build_tables(year, month, args.top_n, station_ids).items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"{name}: {len(df):,} rows -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
