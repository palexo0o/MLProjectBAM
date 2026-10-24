"""SQLite persistence. Listings are upserted by Idealista listing id, so repeated
runs refresh existing rows instead of duplicating them."""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import fields
from pathlib import Path
from typing import Iterable

from .models import Listing

LIST_COLUMNS = ("features", "image_urls", "search_snippet")

SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
    listing_id          TEXT PRIMARY KEY,
    url                 TEXT NOT NULL,
    operation           TEXT,
    title               TEXT,
    price_eur           REAL,
    property_type       TEXT,
    rooms               INTEGER,
    bathrooms           INTEGER,
    area_m2             REAL,
    floor               TEXT,
    address             TEXT,
    district            TEXT,
    municipality        TEXT,
    province            TEXT,
    latitude            REAL,
    longitude           REAL,
    description         TEXT,
    features            TEXT,   -- JSON array
    image_urls          TEXT,   -- JSON array
    energy_certificate  TEXT,
    price_per_m2_eur    REAL,
    scraped_at          TEXT NOT NULL,
    search_snippet      TEXT    -- JSON object
);
CREATE INDEX IF NOT EXISTS idx_listings_operation ON listings(operation);
CREATE INDEX IF NOT EXISTS idx_listings_municipality ON listings(municipality);
"""

COLUMNS = tuple(f.name for f in fields(Listing))


class Storage:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def known_ids(self) -> set[str]:
        return {row[0] for row in self.conn.execute("SELECT listing_id FROM listings")}

    def upsert(self, listing: Listing) -> None:
        row = _to_row(listing)
        placeholders = ", ".join("?" for _ in COLUMNS)
        updates = ", ".join(f"{c}=excluded.{c}" for c in COLUMNS if c != "listing_id")
        self.conn.execute(
            f"INSERT INTO listings ({', '.join(COLUMNS)}) VALUES ({placeholders}) "
            f"ON CONFLICT(listing_id) DO UPDATE SET {updates}",
            row,
        )
        self.conn.commit()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]

    def iter_rows(self) -> Iterable[dict]:
        for row in self.conn.execute("SELECT * FROM listings ORDER BY listing_id"):
            yield _from_row(row)


def _to_row(listing: Listing) -> tuple:
    data = listing.to_dict()
    values = []
    for column in COLUMNS:
        value = data[column]
        if column in LIST_COLUMNS:
            value = json.dumps(value, ensure_ascii=False)
        values.append(value)
    return tuple(values)


def _from_row(row: sqlite3.Row) -> dict:
    record = dict(row)
    for column in LIST_COLUMNS:
        record[column] = json.loads(record[column]) if record[column] else None
    return record


def export_csv(rows: Iterable[dict], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for record in rows:
            flat = {
                k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                for k, v in record.items()
            }
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
                writer.writeheader()
            writer.writerow(flat)
            count += 1
    return count


def export_jsonl(rows: Iterable[dict], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
