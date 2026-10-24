"""Data model for a scraped property listing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Listing:
    listing_id: str
    url: str
    operation: str | None = None          # "sale" or "rent", when known
    title: str | None = None
    price_eur: float | None = None
    property_type: str | None = None
    rooms: int | None = None
    bathrooms: int | None = None
    area_m2: float | None = None
    floor: str | None = None
    address: str | None = None
    district: str | None = None
    municipality: str | None = None
    province: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    description: str | None = None
    features: list[str] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)
    energy_certificate: str | None = None
    price_per_m2_eur: float | None = None
    scraped_at: str = field(default_factory=utc_now_iso)
    # The partial record from the search results page, kept so a listing
    # missing from its detail page still produces something useful.
    search_snippet: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
