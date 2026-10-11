# Idealista scraper

Collects property listings (sale and rent) from [idealista.com](https://www.idealista.com)
into a local SQLite database, with CSV/JSONL export.

## Read this first

- **Idealista's terms of use restrict automated access and data extraction.**
  Read them before running this against the live site. Using this tool is your
  responsibility.
- **robots.txt** disallows result pagination (`/pagina/`, `/*/pagina-*.htm`) and
  sort-order URLs. The scraper honours these rules: it reads page 1 of each
  search URL you give it, and never paginates.
- **Personal data.** Listings can include agency and contact details. This
  scraper does not collect phone numbers or contact forms. Personal data is
  subject to GDPR, so keep what you collect to what you need.
- **Bot protection.** The site answers plain HTTP clients with 403, and at the
  time of writing it also refused a Chromium session from the development
  machine (see *Status* below). This scraper uses a real browser, paces
  requests with random delays, reuses a persistent profile and caches HTML. It
  **does not** solve CAPTCHAs, spoof browser fingerprints, or rotate IPs or
  proxies. When a challenge appears, the run stops and keeps the data it has.
  Those techniques are designed to defeat the site's access controls, and I
  have not built them.
- If you need reliable, legitimate access at scale, ask Idealista whether they
  offer a data licence or partner API.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

## Usage

Search URLs come from the site itself. Open a search on idealista.com, filter it
(for example, a district and price range), and copy the URL. Narrower searches
reach more listings, because each search contributes only its first page.

```bash
# One search
python -m idealista scrape --url "https://www.idealista.com/venta-viviendas/madrid/" --max-listings 50

# Many searches, one per line in a file (lines starting with # are ignored)
python -m idealista scrape --targets targets.example.txt --max-listings 500

# Search cards only, no detail pages (much faster, fewer fields)
python -m idealista scrape --targets targets.txt --no-details

# Re-fetch listings already in the database
python -m idealista scrape --targets targets.txt --refresh

# Export
python -m idealista export --format csv --out data/listings.csv
python -m idealista export --format jsonl --out data/listings.jsonl
python -m idealista stats
```

Useful options: `--min-delay` / `--max-delay` (seconds between requests, default
6-12), `--headless`, `--db`, `-v`.

## What is collected

Per listing (where the page provides it): id, URL, sale/rent, title, price,
price per m², rooms, bathrooms, area, floor, address, district, municipality,
coordinates, description, features, image URLs, energy certificate class, and
the search snippet. Stored in `data/idealista.sqlite3`. Listings are upserted by
id, so repeated runs update rows rather than duplicating them.

## Layout

| Path | Purpose |
|---|---|
| `idealista/fetcher.py` | Browser session, pacing, HTML cache, challenge detection |
| `idealista/robots.py` | robots.txt parser with `*` and `$` wildcard support |
| `idealista/parsers.py` | Search-result and detail-page parsing; **all CSS selectors are in `SELECTORS`** |
| `idealista/scraper.py` | Crawl loop: search → cards → detail pages → storage |
| `idealista/storage.py` | SQLite schema, upserts, CSV/JSONL export |
| `idealista/cli.py` | Command-line interface |
| `tests/` | Unit tests (`python -m unittest discover -s tests -t .`) |

## Status and known limits

- **Selectors are unverified against live markup.** The parsers were written
  from knowledge of Idealista's structure and tested against hand-written
  fixtures in `tests/test_parsers.py`. The first live run is what confirms them.
  If a field comes back empty, check the selector in `SELECTORS` against the
  saved HTML in `data/html_cache/`.
- **Live access was blocked during development.** A single search-page request
  returned HTTP 403 in both headless and headed Chromium, from this machine. The
  scraper stopped as designed. Whether it works depends on your network and on
  the site's current policy.
- Pagination is intentionally not implemented (see robots.txt above).
- Parsing for district and municipality is a best-effort split of the address
  line; treat those two fields as approximate.
