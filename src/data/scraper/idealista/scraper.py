"""Crawl orchestration: search pages -> listing cards -> detail pages -> SQLite."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Settings
from .fetcher import BlockedError, Fetcher
from .models import Listing
from .parsers import parse_detail_page, parse_search_results
from .storage import Storage

log = logging.getLogger(__name__)


@dataclass
class RunSummary:
    search_pages: int = 0
    listings_saved: int = 0
    details_fetched: int = 0
    skipped_known: int = 0
    blocked: int = 0
    aborted: bool = False


def run(
    search_urls: list[str],
    settings: Settings,
    *,
    max_listings: int = 100,
    fetch_details: bool = True,
    refresh: bool = False,
) -> RunSummary:
    """Scrape the first results page of each search URL, then detail pages.

    Only page one of each search is read. Pagination URLs (``/pagina-N.htm``)
    are disallowed by Idealista's robots.txt, so reaching more listings means
    passing more specific search URLs (for example, one per district).
    """
    summary = RunSummary()
    consecutive_blocks = 0

    with Fetcher(settings) as fetcher, Storage(settings.db_path) as storage:
        known = set() if refresh else storage.known_ids()

        for search_url in search_urls:
            if summary.listings_saved >= max_listings or summary.aborted:
                break

            try:
                page = fetcher.get(search_url)
                consecutive_blocks = 0
            except BlockedError as exc:
                summary.blocked += 1
                consecutive_blocks += 1
                log.warning("%s", exc)
                if consecutive_blocks >= settings.max_consecutive_blocks:
                    log.error("Stopping: %d consecutive blocks.", consecutive_blocks)
                    summary.aborted = True
                break

            if page.html is None:
                continue
            summary.search_pages += 1
            cards = parse_search_results(page.html, search_url)
            log.info("%d cards on %s", len(cards), search_url)

            for card in cards:
                if summary.listings_saved >= max_listings or summary.aborted:
                    break
                if card.listing_id in known:
                    summary.skipped_known += 1
                    continue

                listing = _enrich(card, fetcher, settings, summary, fetch_details)
                if listing is None:
                    # Blocked during detail fetch; abort if the streak is too long.
                    consecutive_blocks += 1
                    if consecutive_blocks >= settings.max_consecutive_blocks:
                        log.error("Stopping: %d consecutive blocks.", consecutive_blocks)
                        summary.aborted = True
                        break
                    listing = card
                else:
                    consecutive_blocks = 0

                storage.upsert(listing)
                known.add(listing.listing_id)
                summary.listings_saved += 1

    return summary


def _enrich(
    card: Listing,
    fetcher: Fetcher,
    settings: Settings,
    summary: RunSummary,
    fetch_details: bool,
) -> Listing | None:
    """Return the detail-page record, the card itself if details are off or the
    page is missing, or ``None`` if the request was blocked."""
    if not fetch_details:
        return card
    try:
        detail = fetcher.get(card.url)
    except BlockedError as exc:
        summary.blocked += 1
        log.warning("%s", exc)
        return None
    if detail.html is None:
        return card
    summary.details_fetched += 1
    return parse_detail_page(detail.html, card.url, base=card)
