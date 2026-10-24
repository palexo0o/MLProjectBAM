import unittest

from idealista.models import Listing
from idealista.parsers import (
    listing_id_from_url,
    operation_from_url,
    parse_detail_page,
    parse_eur,
    parse_search_results,
)

# Hand-written fixtures that follow the selectors in parsers.SELECTORS. They
# exercise the parsing logic; they are not captures of live Idealista pages.
SEARCH_HTML = """
<html><body>
<article class="item" data-element-id="11111111">
  <a class="item-link" href="/inmueble/11111111/">Piso luminoso en Chamberí</a>
  <span class="item-price">385.000 €</span>
  <div class="item-detail-char">
    <span class="item-detail">3 hab.</span>
    <span class="item-detail">92 m²</span>
    <span class="item-detail">2ª planta exterior</span>
  </div>
  <div class="item-description">Muy bien comunicado.</div>
</article>
<article class="item">
  <a class="item-link" href="/inmueble/22222222/">Ático con terraza</a>
  <span class="item-price">1.250.000 €</span>
  <div class="item-detail-char">
    <span class="item-detail">4 hab.</span>
    <span class="item-detail">150 m²</span>
    <span class="item-detail">2 baños</span>
  </div>
</article>
<article class="item"><span class="item-price">no link here</span></article>
</body></html>
"""

DETAIL_HTML = """
<html><head>
<script type="application/ld+json">
{"@type": "Apartment", "description": "Reformado en 2024.",
 "address": {"streetAddress": "Calle Ejemplo 1", "addressLocality": "Madrid"}}
</script>
</head><body>
<h1 class="main-info__title-main">Piso en calle Ejemplo</h1>
<span class="main-info__title-minor">Calle Ejemplo 1, Chamberí, Madrid</span>
<span class="info-data-price"><span class="txt-bold">412.500 €</span></span>
<div class="info-features"><span>3 hab.</span><span>90 m²</span><span>1 baño</span></div>
<div class="details-property_features"><ul><li>Ascensor</li><li>Calefacción central</li></ul></div>
<div class="comment">Piso exterior con ascensor.</div>
<div id="main-multimedia"><img src="https://img4.idealista.com/a.jpg"/>
<img data-ondemand-img="https://img4.idealista.com/b.jpg"/></div>
<p>Certificado energético: E</p>
<script>var x = {"latitude": 40.4425, "longitude": -3.7001};</script>
</body></html>
"""


class ParseEurTests(unittest.TestCase):
    def test_spanish_thousands_separator(self):
        self.assertEqual(parse_eur("385.000 €"), 385000.0)
        self.assertEqual(parse_eur("1.250.000 €"), 1250000.0)

    def test_decimal_comma(self):
        self.assertEqual(parse_eur("1.250,50 €"), 1250.5)

    def test_missing_or_invalid(self):
        self.assertIsNone(parse_eur(None))
        self.assertIsNone(parse_eur("consultar precio"))


class UrlHelperTests(unittest.TestCase):
    def test_listing_id(self):
        self.assertEqual(listing_id_from_url("https://www.idealista.com/inmueble/12345/"), "12345")
        self.assertIsNone(listing_id_from_url("https://www.idealista.com/venta-viviendas/"))

    def test_operation(self):
        self.assertEqual(operation_from_url("https://x/venta-viviendas/madrid/"), "sale")
        self.assertEqual(operation_from_url("https://x/alquiler-viviendas/madrid/"), "rent")
        self.assertIsNone(operation_from_url("https://x/otro/"))


class SearchResultsTests(unittest.TestCase):
    def setUp(self):
        self.page_url = "https://www.idealista.com/venta-viviendas/madrid/"
        self.listings = parse_search_results(SEARCH_HTML, self.page_url)

    def test_skips_cards_without_link(self):
        self.assertEqual([l.listing_id for l in self.listings], ["11111111", "22222222"])

    def test_fields_from_card(self):
        first = self.listings[0]
        self.assertEqual(first.url, "https://www.idealista.com/inmueble/11111111/")
        self.assertEqual(first.operation, "sale")
        self.assertEqual(first.price_eur, 385000.0)
        self.assertEqual(first.rooms, 3)
        self.assertEqual(first.area_m2, 92.0)
        self.assertEqual(first.floor, "2ª planta exterior")
        self.assertEqual(first.description, "Muy bien comunicado.")

    def test_bathrooms_and_price_per_m2(self):
        second = self.listings[1]
        self.assertEqual(second.bathrooms, 2)
        self.assertEqual(second.price_per_m2_eur, round(1250000 / 150, 2))


class DetailPageTests(unittest.TestCase):
    def setUp(self):
        self.url = "https://www.idealista.com/inmueble/11111111/"
        base = Listing(listing_id="11111111", url=self.url, operation="sale")
        self.listing = parse_detail_page(DETAIL_HTML, self.url, base=base)

    def test_title_location_and_price(self):
        self.assertEqual(self.listing.title, "Piso en calle Ejemplo")
        self.assertEqual(self.listing.address, "Calle Ejemplo 1, Chamberí, Madrid")
        self.assertEqual(self.listing.district, "Chamberí")
        self.assertEqual(self.listing.municipality, "Madrid")
        self.assertEqual(self.listing.price_eur, 412500.0)

    def test_features_are_split_into_fields(self):
        self.assertEqual(self.listing.rooms, 3)
        self.assertEqual(self.listing.area_m2, 90.0)
        self.assertEqual(self.listing.bathrooms, 1)
        self.assertIn("Ascensor", self.listing.features)
        self.assertIn("Calefacción central", self.listing.features)

    def test_images_description_energy_and_coordinates(self):
        self.assertEqual(
            self.listing.image_urls,
            ["https://img4.idealista.com/a.jpg", "https://img4.idealista.com/b.jpg"],
        )
        self.assertEqual(self.listing.description, "Piso exterior con ascensor.")
        self.assertEqual(self.listing.energy_certificate, "E")
        self.assertAlmostEqual(self.listing.latitude, 40.4425)
        self.assertAlmostEqual(self.listing.longitude, -3.7001)

    def test_base_fields_are_kept_when_detail_is_silent(self):
        self.assertEqual(self.listing.operation, "sale")
        self.assertEqual(self.listing.listing_id, "11111111")


if __name__ == "__main__":
    unittest.main()
