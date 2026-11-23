"""Geocode flat addresses to lat/lon with OpenStreetMap Nominatim.

Results are cached in geocoded_addresses.csv, so the script can be stopped and resumed. Nominatim's usage policy
allows at most one request per second; this script respects that.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from src.data.data_acquire import load_df  # noqa: E402

CACHE = HERE / "geocoded_addresses.csv"
ENDPOINT = "https://nominatim.openstreetmap.org/search"
# Generic identifier as required by the usage policy; no personal details.
HEADERS = {"User-Agent": "MLProjectBAM-academic-geocoder/1.0"}
# Approximate bounding box of the Madrid municipality, used to bias results.
VIEWBOX = "-3.90,40.65,-3.50,40.30"

PREFIX = re.compile(r"^(Piso|Ático|Estudio|Dúplex|Chalet adosado|Chalet pareado|Chalet|Casa o|Caserón)\s+(en\s+|o\s+)?", re.I)


def clean_number(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip()
    try:
        return str(int(float(s)))
    except ValueError:
        return ""  # "s/n" or a stray street name


def make_query(address: str, number) -> str:
    street = PREFIX.sub("", str(address)).strip()
    num = clean_number(number)
    parts = [f"{street} {num}".strip() if num else street, "Madrid", "Spain"]
    return ", ".join(p for p in parts if p)


def geocode_one(query: str) -> dict:
    params = {"q": query, "format": "jsonv2", "limit": 1, "countrycodes": "es",
              "viewbox": VIEWBOX, "addressdetails": 0}
    for attempt in range(3):
        try:
            r = requests.get(ENDPOINT, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            hits = r.json()
            if hits:
                h = hits[0]
                return {"lat": float(h["lat"]), "lon": float(h["lon"]),
                        "osm_class": h.get("class"), "osm_type": h.get("type"),
                        "display_name": h.get("display_name")}
            return {"lat": None, "lon": None, "osm_class": None, "osm_type": None, "display_name": None}
        except (requests.RequestException, ValueError, KeyError):
            time.sleep(5 * (attempt + 1))
    return {"lat": None, "lon": None, "osm_class": "error", "osm_type": "error", "display_name": None}


def main() -> None:
    df = load_df()
    df["query"] = [make_query(a, n) for a, n in zip(df["Address"], df["Number"])]
    unique = pd.Series(df["query"].unique(), name="query")

    done = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["query", "lat", "lon", "osm_class", "osm_type", "display_name"])
    todo = [q for q in unique if q not in set(done["query"])]
    print(f"{len(unique)} unique queries, {len(todo)} left to geocode")

    rows = []
    for i, q in enumerate(todo, 1):
        rows.append({"query": q, **geocode_one(q)})
        if i % 50 == 0 or i == len(todo):
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(CACHE, index=False)
            print(f"{i}/{len(todo)}", flush=True)
        time.sleep(1.05)
    if rows:
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(CACHE, index=False)
    print("done")


if __name__ == "__main__":
    main()
