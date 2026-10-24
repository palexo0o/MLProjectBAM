"""Polite, cache-aware page fetcher built on a real Chromium browser.

Design choices, in order of importance:

* A real browser is used instead of a bare HTTP client. Plain HTTP requests are
  rejected by the site, and a browser also runs the page's own JavaScript.
* Requests are paced with a random delay, so the traffic does not follow a
  fixed machine-like rhythm.
* A persistent profile keeps cookies between runs, like a returning visitor.
* Fetched HTML is cached on disk, so re-parsing never re-hits the site.
* When the site serves a challenge (CAPTCHA, bot wall), the fetcher stops and
  reports it. It does not attempt to solve challenges or rotate identities.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from .config import Settings
from .robots import RobotsRules

log = logging.getLogger(__name__)

BASE_URL = "https://www.idealista.com"
ROBOTS_URL = f"{BASE_URL}/robots.txt"

CHALLENGE_STATUS = {403, 429, 503}
CHALLENGE_MARKERS = (
    "captcha-delivery",
    "geo.captcha",
    "datadome",
    "challenge-platform",
    "cf-challenge",
    "access denied",
    "attention required",
)
COOKIE_ACCEPT_SELECTOR = "#didomi-notice-agree-button"


class BlockedError(RuntimeError):
    """The site served a challenge or refused the request."""


@dataclass
class FetchResult:
    url: str
    html: str | None
    status: int | None
    from_cache: bool = False


class Fetcher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        self.settings.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._robots: RobotsRules | None = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Fetcher":
        self._playwright = sync_playwright().start()
        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.settings.profile_dir),
            headless=self.settings.headless,
            locale="es-ES",
            timezone_id="Europe/Madrid",
            viewport={"width": 1366, "height": 850},
        )
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        self._page.set_default_timeout(self.settings.timeout_ms)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._context is not None:
            self._context.close()
        if self._playwright is not None:
            self._playwright.stop()

    # -- robots.txt --------------------------------------------------------

    def load_robots(self) -> RobotsRules:
        """Fetch robots.txt once per run and reuse it for every URL check."""
        if self._robots is None:
            # The request API returns the raw body. Navigating to robots.txt would
            # wrap it in an HTML page, because Chromium renders text files that way.
            assert self._context is not None, "Fetcher must be used as a context manager"
            response = self._context.request.get(ROBOTS_URL)
            text = response.text() if response.ok else ""
            self._robots = RobotsRules.parse(text)
        return self._robots

    def allowed_by_robots(self, url: str) -> bool:
        return self.load_robots().can_fetch(url)

    # -- fetching ----------------------------------------------------------

    def get(self, url: str) -> FetchResult:
        """Return HTML for ``url``, from cache when fresh, otherwise live.

        Raises ``BlockedError`` when the site serves a challenge. Callers decide
        whether to retry or stop.
        """
        cached = self._read_cache(url)
        if cached is not None:
            return FetchResult(url=url, html=cached, status=200, from_cache=True)

        if not self.allowed_by_robots(url):
            log.info("Skipping %s: disallowed by robots.txt", url)
            return FetchResult(url=url, html=None, status=None)

        self._pace()
        result = self._load_uncached(url)
        if result.status == 404:
            return result
        if result.html is None or self._looks_blocked(result):
            raise BlockedError(f"Challenge or refusal (HTTP {result.status}) at {url}")

        self._write_cache(url, result.html)
        return result

    # -- internals ---------------------------------------------------------

    def _pace(self) -> None:
        delay = random.uniform(self.settings.min_delay, self.settings.max_delay)
        log.debug("Waiting %.1fs before next request", delay)
        time.sleep(delay)

    def _load_uncached(self, url: str) -> FetchResult:
        assert self._page is not None, "Fetcher must be used as a context manager"
        try:
            response = self._page.goto(url, wait_until="domcontentloaded")
        except PlaywrightTimeoutError:
            log.warning("Timed out loading %s", url)
            return FetchResult(url=url, html=None, status=None)

        status = response.status if response else None
        self._dismiss_cookie_banner()
        try:
            self._page.wait_for_load_state("networkidle", timeout=10_000)
        except PlaywrightTimeoutError:
            pass  # Some pages keep long-lived connections open; the DOM is usable.

        return FetchResult(url=url, html=self._page.content(), status=status)

    def _dismiss_cookie_banner(self) -> None:
        assert self._page is not None
        try:
            button = self._page.locator(COOKIE_ACCEPT_SELECTOR)
            if button.count() and button.first.is_visible():
                button.first.click(timeout=3_000)
        except PlaywrightTimeoutError:
            pass

    @staticmethod
    def _looks_blocked(result: FetchResult) -> bool:
        if result.status in CHALLENGE_STATUS:
            return True
        head = (result.html or "")[:20_000].lower()
        return any(marker in head for marker in CHALLENGE_MARKERS)

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()
        return self.settings.cache_dir / f"{digest}.html"

    def _read_cache(self, url: str) -> str | None:
        path = self._cache_path(url)
        if not path.exists():
            return None
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > self.settings.cache_ttl_hours:
            return None
        return path.read_text(encoding="utf-8")

    def _write_cache(self, url: str, html: str) -> None:
        self._cache_path(url).write_text(html, encoding="utf-8")
