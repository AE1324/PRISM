"""
Pull Division III reference data from the D3 Dashboard API.

Usage (from the model/ folder):
    python -m prism_data.d3dashboard --years 2023 2024 2025

Reads the API key from model/.env (D3_DASHBOARD_API_KEY=...). Saves to
data/raw/d3dashboard/:
  pitching_<year>.parquet       individual pitching stats (ERA, K%, BB%, WAR, ...)
  pitching_team_<year>.parquet  team pitching stats
  expected_runs_<year>.parquet  D3 run expectancy by base-out state
  guts.parquet                  wOBA weights, league averages, FIP constant
  park_factors.parquet          park factors by team
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from prism_data.paths import RAW_DIR

BASE_URL = "https://d3-dashboard.com/api"
DIVISION = "ncaa_3"
OUT_DIR = RAW_DIR / "d3dashboard"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class D3Dashboard:
    """Small client: API-key auth, polite rate limiting, and retries."""

    def __init__(self, api_key: str | None = None):
        load_dotenv(ENV_FILE)
        self.api_key = api_key or os.getenv("D3_DASHBOARD_API_KEY")
        if not self.api_key:
            raise SystemExit(
                "No API key found. Add a line D3_DASHBOARD_API_KEY=your_key to model/.env "
                "(copy model/.env.example). Keys expire every 90 days."
            )
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = self.api_key

    def get(self, endpoint: str, **params) -> list | dict:
        for attempt in range(4):
            resp = self.session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=60)
            if resp.status_code == 429:
                reset = float(resp.headers.get("X-RateLimit-Reset", time.time() + 10))
                time.sleep(max(1.0, reset - time.time()))
                continue
            if resp.status_code == 400:
                raise SystemExit(f"D3 Dashboard rejected the request to /{endpoint}: {resp.text[:200]}")
            if resp.status_code in (401, 403):
                raise SystemExit("D3 Dashboard rejected the API key. It may have expired; create a new one.")
            if resp.status_code >= 500 and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Gave up on {endpoint} after repeated rate limits or server errors.")


def save(rows: list | dict, name: str) -> None:
    df = pd.json_normalize(rows) if isinstance(rows, list) else pd.json_normalize([rows])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"  {name}: {len(df):,} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    args = parser.parse_args()

    api = D3Dashboard()
    # Every endpoint expects the division as "ncaa_3" (the docs also show 3, which is rejected).
    for year in args.years:
        print(f"Season {year}")
        save(api.get("pitching", years=str(year), division=DIVISION), f"pitching_{year}")
        save(api.get("pitching_team", years=str(year), division=DIVISION), f"pitching_team_{year}")
        save(api.get("expected_runs", year=year, division=DIVISION), f"expected_runs_{year}")
    print("Reference tables")
    save(api.get("guts"), "guts")
    save(api.get("park_factors", division=DIVISION), "park_factors")


if __name__ == "__main__":
    main()
