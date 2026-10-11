"""Scraper for Idealista (Spanish property listings)."""

from .models import Listing
from .scraper import RunSummary, run

__all__ = ["Listing", "RunSummary", "run"]
