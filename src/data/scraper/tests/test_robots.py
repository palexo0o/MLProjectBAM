import unittest

from idealista.robots import RobotsRules

# Lines copied from https://www.idealista.com/robots.txt (the * group).
SAMPLE = """
User-agent: *
Disallow: /ajax/*
Disallow:/inmueble*/foto
Disallow:/*?ordenado-por=precio-desc
Disallow:/pagina/
Disallow: /*/pagina-*.htm
Disallow:/*multimediaNumber=2
User-agent: SomeOtherBot
Disallow: /
"""

BASE = "https://www.idealista.com"


class RobotsTests(unittest.TestCase):
    def setUp(self):
        self.rules = RobotsRules.parse(SAMPLE)

    def test_plain_listing_pages_are_allowed(self):
        self.assertTrue(self.rules.can_fetch(f"{BASE}/venta-viviendas/madrid/"))
        self.assertTrue(self.rules.can_fetch(f"{BASE}/inmueble/12345678/"))

    def test_pagination_is_disallowed(self):
        self.assertFalse(self.rules.can_fetch(f"{BASE}/venta-viviendas/madrid/pagina-2.htm"))
        self.assertFalse(self.rules.can_fetch(f"{BASE}/pagina/2"))

    def test_wildcard_rules_match_mid_path(self):
        self.assertFalse(self.rules.can_fetch(f"{BASE}/inmueble/12345678/foto"))
        self.assertFalse(self.rules.can_fetch(f"{BASE}/ajax/anything"))

    def test_query_string_rules(self):
        self.assertFalse(
            self.rules.can_fetch(f"{BASE}/venta-viviendas/madrid/?ordenado-por=precio-desc")
        )
        self.assertTrue(
            self.rules.can_fetch(f"{BASE}/venta-viviendas/madrid/?ordenado-por=precio-asc")
        )

    def test_other_user_agent_groups_are_ignored(self):
        # The "SomeOtherBot" group disallows everything; it must not apply to us.
        self.assertTrue(self.rules.can_fetch(f"{BASE}/venta-viviendas/madrid/"))

    def test_empty_robots_allows_everything(self):
        self.assertTrue(RobotsRules.parse("").can_fetch(f"{BASE}/pagina/2"))


if __name__ == "__main__":
    unittest.main()
