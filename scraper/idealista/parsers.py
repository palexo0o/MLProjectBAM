"""HTML parsers for Idealista search-result pages and property detail pages.

All CSS selectors live in ``SELECTORS`` so they can be updated in one place when
the site's markup changes. Each field is parsed independently, so a missing
element yields ``None`` for that field rather than failing the whole listing.
"""

from __future__ import annotations

import json
import re
from typing import Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from .models import Listing

BASE_URL = "https://www.idealista.com"

SELECTORS = {
    # Search results page
    "result_card": "article.item",
    "result_link": "a.item-link",
    "result_price": ".item-price",
    "result_details": ".item-detail-char .item-detail",
    "result_description": ".item-description",
    # Detail page
    "detail_title": ".main-info__title-main",
    "detail_location": ".main-info__title-minor",
    "detail_price": ".info-data-price",
    "detail_features": ".info-features span, .details-property_features li, "
    ".details-property-feature-one li, .details-property-feature-two li",
    "detail_description": ".comment, .adCommentsLanguage",
    "detail_images": "#main-multimedia img, .detail-multimedia img, "
    "picture img, img[data-ondemand-img]",
}

LISTING_ID_RE = re.compile(r"/inmueble/(\d+)")
NUMBER_RE = re.compile(r"\d[\d.,]*")
ENERGY_RE = re.compile(
    r"(?:certificado|consumo)\s+energ[ée]tico\D{0,40}?\b([A-G])\b", re.IGNORECASE
)
LATITUDE_RE = re.compile(r'"latitude"\s*:\s*(-?\d+(?:\.\d+)?)')
LONGITUDE_RE = re.compile(r'"longitude"\s*:\s*(-?\d+(?:\.\d+)?)')


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

def parse_eur(text: str | None) -> float | None:
    """'1.250.000 €' -> 1250000.0 (Spanish format: '.' groups thousands)."""
    if not text:
        return None
    match = NUMBER_RE.search(text)
    if not match:
        return None
    number = match.group(0)
    if "," in number:
        number = number.replace(".", "").replace(",", ".")
    else:
        number = number.replace(".", "")
    try:
        return float(number)
    except ValueError:
        return None


def _first_int(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


def _first_area(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"(\d[\d.,]*)\s*m²", text)
    return parse_eur(match.group(1)) if match else None


def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    collapsed = " ".join(text.split())
    return collapsed or None


def listing_id_from_url(url: str) -> str | None:
    match = LISTING_ID_RE.search(url)
    return match.group(1) if match else None


def operation_from_url(url: str) -> str | None:
    if "/venta-" in url:
        return "sale"
    if "/alquiler-" in url:
        return "rent"
    return None


def _apply_feature_text(listing: Listing, text: str) -> None:
    """Assign one feature string (e.g. '3 hab.', '90 m²') to the right field."""
    lowered = text.lower()
    if "m²" in lowered and listing.area_m2 is None:
        listing.area_m2 = _first_area(text)
    elif re.search(r"\bhab", lowered) and listing.rooms is None:
        listing.rooms = _first_int(text)
    elif re.search(r"ba[ñn]o", lowered) and listing.bathrooms is None:
        listing.bathrooms = _first_int(text)
    elif re.search(r"planta|bajo|exterior|interior|entreplanta|sótano", lowered):
        if listing.floor is None:
            listing.floor = text
    else:
        if text not in listing.features:
            listing.features.append(text)


def _json_ld_objects(soup: BeautifulSoup) -> Iterable[dict]:
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict):
                yield item


# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------

def parse_search_results(html: str, page_url: str) -> list[Listing]:
    """Extract every listing card from one search-results page."""
    soup = BeautifulSoup(html, "html.parser")
    listings: list[Listing] = []
    for card in soup.select(SELECTORS["result_card"]):
        link = card.select_one(SELECTORS["result_link"])
        if link is None or not link.get("href"):
            continue
        url = urljoin(BASE_URL, link["href"])
        listing_id = card.get("data-element-id") or listing_id_from_url(url)
        if not listing_id:
            continue

        listing = Listing(
            listing_id=str(listing_id),
            url=url,
            operation=operation_from_url(page_url),
            title=_clean(link.get_text()),
            price_eur=parse_eur(_text(card, SELECTORS["result_price"])),
            description=_text(card, SELECTORS["result_description"]),
        )
        for detail in card.select(SELECTORS["result_details"]):
            _apply_feature_text(listing, _clean(detail.get_text()) or "")

        listing.search_snippet = {"source_page": page_url}
        listing.price_per_m2_eur = _price_per_m2(listing)
        listings.append(listing)
    return listings


# ---------------------------------------------------------------------------
# Detail page
# ---------------------------------------------------------------------------

def parse_detail_page(html: str, url: str, base: Listing | None = None) -> Listing:
    """Extract the full record from a property detail page.

    ``base`` is the listing from the search results page; its fields are kept
    where the detail page does not provide a value.
    """
    soup = BeautifulSoup(html, "html.parser")
    listing_id = listing_id_from_url(url) or (base.listing_id if base else "")
    listing = base or Listing(listing_id=listing_id, url=url)
    listing.url = url
    listing.operation = listing.operation or operation_from_url(url)

    listing.title = _text(soup, SELECTORS["detail_title"]) or listing.title
    location = _text(soup, SELECTORS["detail_location"])
    if location:
        listing.address = location
        _split_location(listing, location)

    price = parse_eur(_text(soup, SELECTORS["detail_price"]))
    if price is not None:
        listing.price_eur = price

    for element in soup.select(SELECTORS["detail_features"]):
        text = _clean(element.get_text())
        if text:
            _apply_feature_text(listing, text)

    listing.description = _text(soup, SELECTORS["detail_description"]) or listing.description

    images = []
    for img in soup.select(SELECTORS["detail_images"]):
        src = img.get("data-ondemand-img") or img.get("src") or img.get("data-src")
        if src and src.startswith("http") and src not in images:
            images.append(src)
    listing.image_urls = images or listing.image_urls

    raw_text = soup.get_text(" ")
    energy = ENERGY_RE.search(raw_text)
    if energy:
        listing.energy_certificate = energy.group(1).upper()

    latitude = LATITUDE_RE.search(html)
    longitude = LONGITUDE_RE.search(html)
    if latitude and longitude:
        listing.latitude = float(latitude.group(1))
        listing.longitude = float(longitude.group(1))

    _fill_from_json_ld(listing, soup)

    listing.price_per_m2_eur = _price_per_m2(listing)
    return listing


def _split_location(listing: Listing, location: str) -> None:
    """Best-effort split of 'Calle X, Barrio, Distrito, Municipio' style strings."""
    parts = [part.strip() for part in location.split(",") if part.strip()]
    if len(parts) >= 2:
        listing.district = parts[-2]
    if parts:
        listing.municipality = parts[-1]


def _fill_from_json_ld(listing: Listing, soup: BeautifulSoup) -> None:
    """Use schema.org JSON-LD, when present, only to fill gaps."""
    for item in _json_ld_objects(soup):
        address = item.get("address")
        if isinstance(address, dict):
            listing.address = listing.address or address.get("streetAddress")
            listing.municipality = listing.municipality or address.get("addressLocality")
            listing.province = listing.province or address.get("addressRegion")
        geo = item.get("geo")
        if isinstance(geo, dict) and listing.latitude is None:
            listing.latitude = _to_float(geo.get("latitude"))
            listing.longitude = _to_float(geo.get("longitude"))
        if listing.description is None and isinstance(item.get("description"), str):
            listing.description = _clean(item["description"])
        if listing.area_m2 is None and isinstance(item.get("floorSize"), dict):
            listing.area_m2 = _to_float(item["floorSize"].get("value"))


def _to_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _price_per_m2(listing: Listing) -> float | None:
    if listing.price_eur and listing.area_m2:
        return round(listing.price_eur / listing.area_m2, 2)
    return None


def _text(scope: Tag | BeautifulSoup, selector: str) -> str | None:
    element = scope.select_one(selector)
    return _clean(element.get_text(" ")) if element else None
