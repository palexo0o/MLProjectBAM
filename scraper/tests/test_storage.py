import csv
import json
import tempfile
import unittest
from pathlib import Path

from idealista.models import Listing
from idealista.storage import Storage, export_csv, export_jsonl


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.sqlite3"

    def tearDown(self):
        self.tmp.cleanup()

    def test_upsert_updates_instead_of_duplicating(self):
        with Storage(self.db) as storage:
            storage.upsert(Listing(listing_id="1", url="u", price_eur=100.0))
            storage.upsert(Listing(listing_id="1", url="u", price_eur=90.0, features=["Ascensor"]))
            self.assertEqual(storage.count(), 1)
            row = next(storage.iter_rows())
            self.assertEqual(row["price_eur"], 90.0)
            self.assertEqual(row["features"], ["Ascensor"])

    def test_known_ids(self):
        with Storage(self.db) as storage:
            storage.upsert(Listing(listing_id="a", url="u"))
            storage.upsert(Listing(listing_id="b", url="u"))
            self.assertEqual(storage.known_ids(), {"a", "b"})

    def test_list_fields_round_trip_as_json(self):
        with Storage(self.db) as storage:
            storage.upsert(Listing(listing_id="1", url="u", image_urls=["x.jpg"]))
            row = next(storage.iter_rows())
            self.assertEqual(row["image_urls"], ["x.jpg"])
            self.assertEqual(row["search_snippet"], {})

    def test_csv_and_jsonl_export(self):
        with Storage(self.db) as storage:
            storage.upsert(Listing(listing_id="1", url="u", features=["Terraza"], rooms=2))
            rows = list(storage.iter_rows())

        csv_out = Path(self.tmp.name) / "out.csv"
        self.assertEqual(export_csv(iter(rows), csv_out), 1)
        with csv_out.open(encoding="utf-8") as handle:
            record = next(csv.DictReader(handle))
        self.assertEqual(record["listing_id"], "1")
        self.assertEqual(json.loads(record["features"]), ["Terraza"])

        jsonl_out = Path(self.tmp.name) / "out.jsonl"
        self.assertEqual(export_jsonl(iter(rows), jsonl_out), 1)
        self.assertEqual(json.loads(jsonl_out.read_text().strip())["rooms"], 2)


if __name__ == "__main__":
    unittest.main()
