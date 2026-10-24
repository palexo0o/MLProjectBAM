"""Command-line interface.

    python -m idealista scrape --url https://www.idealista.com/venta-viviendas/<zona>/
    python -m idealista scrape --targets targets.txt --max-listings 200
    python -m idealista export --format csv --out data/listings.csv
    python -m idealista stats
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from .config import Settings
from .scraper import run
from .storage import Storage, export_csv, export_jsonl


def _read_targets(args: argparse.Namespace) -> list[str]:
    targets = list(args.url or [])
    if args.targets:
        for line in Path(args.targets).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                targets.append(line)
    return targets


def _cmd_scrape(args: argparse.Namespace) -> int:
    targets = _read_targets(args)
    if not targets:
        print("No search URLs given. Use --url or --targets.", file=sys.stderr)
        return 2

    overrides = {
        "min_delay": args.min_delay,
        "max_delay": args.max_delay,
        "headless": True if args.headless else None,
    }
    if args.db:
        overrides["db_path"] = Path(args.db)
    settings = dataclasses.replace(
        Settings(), **{k: v for k, v in overrides.items() if v is not None}
    )

    summary = run(
        targets,
        settings,
        max_listings=args.max_listings,
        fetch_details=not args.no_details,
        refresh=args.refresh,
    )
    print(
        f"search pages: {summary.search_pages} | saved: {summary.listings_saved} | "
        f"detail pages: {summary.details_fetched} | already known: {summary.skipped_known} | "
        f"blocked: {summary.blocked}"
    )
    if summary.aborted:
        print("Run stopped early after repeated blocks. Data saved so far is kept.")
        return 1
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    settings = Settings()
    db_path = Path(args.db) if args.db else settings.db_path
    with Storage(db_path) as storage:
        rows = storage.iter_rows()
        if args.format == "csv":
            count = export_csv(rows, Path(args.out))
        else:
            count = export_jsonl(rows, Path(args.out))
    print(f"Exported {count} listings to {args.out}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    settings = Settings()
    db_path = Path(args.db) if args.db else settings.db_path
    with Storage(db_path) as storage:
        print(f"listings in {db_path}: {storage.count()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="idealista", description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    scrape = sub.add_parser("scrape", help="scrape search results and detail pages")
    scrape.add_argument("--url", action="append", help="search results URL (repeatable)")
    scrape.add_argument("--targets", help="file with one search URL per line")
    scrape.add_argument("--max-listings", type=int, default=100)
    scrape.add_argument("--no-details", action="store_true", help="search cards only")
    scrape.add_argument("--refresh", action="store_true", help="re-fetch already-stored listings")
    scrape.add_argument("--min-delay", type=float, help="min seconds between requests")
    scrape.add_argument("--max-delay", type=float, help="max seconds between requests")
    scrape.add_argument("--headless", action="store_true", help="run browser without a window")
    scrape.add_argument("--db", help="SQLite database path")
    scrape.set_defaults(func=_cmd_scrape)

    export = sub.add_parser("export", help="export stored listings")
    export.add_argument("--format", choices=["csv", "jsonl"], default="csv")
    export.add_argument("--out", required=True)
    export.add_argument("--db", help="SQLite database path")
    export.set_defaults(func=_cmd_export)

    stats = sub.add_parser("stats", help="show how many listings are stored")
    stats.add_argument("--db", help="SQLite database path")
    stats.set_defaults(func=_cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)
