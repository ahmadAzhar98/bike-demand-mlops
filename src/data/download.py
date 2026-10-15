"""Download raw Citi Bike trips and hourly NYC weather for one month.

Usage:
    python -m src.data.download --month 2025-05
"""

import argparse
import calendar
import zipfile
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

CITIBIKE_URL = "https://s3.amazonaws.com/tripdata/{yyyymm}-citibike-tripdata.zip"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

# Central Manhattan; one weather point is enough for ~20 stations.
NYC_LAT, NYC_LON = 40.7580, -73.9855
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "weather_code",
]


def download_trips(year: int, month: int) -> Path:
    yyyymm = f"{year}{month:02d}"
    out_dir = RAW_DIR / f"citibike_{yyyymm}"
    zip_path = RAW_DIR / f"{yyyymm}-citibike-tripdata.zip"

    if out_dir.exists() and any(out_dir.glob("*.csv")):
        print(f"[trips] already extracted: {out_dir}")
        return out_dir

    if not zip_path.exists():
        url = CITIBIKE_URL.format(yyyymm=yyyymm)
        print(f"[trips] downloading {url}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r[trips] {done / total:6.1%}", end="", flush=True)
        print()

    print(f"[trips] extracting to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # Archives sometimes nest CSVs in folders or include macOS junk.
            if member.endswith(".csv") and "__MACOSX" not in member:
                target = out_dir / Path(member).name
                with zf.open(member) as src, open(target, "wb") as dst:
                    while chunk := src.read(8 * 1024 * 1024):
                        dst.write(chunk)
    zip_path.unlink()
    return out_dir


def download_weather(year: int, month: int) -> Path:
    last_day = calendar.monthrange(year, month)[1]
    out_path = RAW_DIR / f"weather_{year}{month:02d}.json"
    if out_path.exists():
        print(f"[weather] already downloaded: {out_path}")
        return out_path

    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "start_date": f"{year}-{month:02d}-01",
        "end_date": f"{year}-{month:02d}-{last_day}",
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "America/New_York",
    }
    print("[weather] requesting Open-Meteo archive")
    r = requests.get(WEATHER_URL, params=params, timeout=60)
    r.raise_for_status()
    out_path.write_text(r.text)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True, help="YYYY-MM, e.g. 2025-05")
    args = parser.parse_args()
    year, month = map(int, args.month.split("-"))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(download_trips(year, month))
    print(download_weather(year, month))


if __name__ == "__main__":
    main()
