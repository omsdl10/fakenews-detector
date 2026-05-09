"""
Article scraper: extract clean article text from a URL.

Primary:   newspaper3k  (handles JS-light sites, paywall detection, NLP)
Fallback:  requests + BeautifulSoup (for simpler pages)

Returns an ArticleScrapeResult with text, title, publish_date, domain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

from backend.core.logging import get_logger

logger = get_logger(__name__)

# Browser-like User-Agent to avoid bot-blocking
_USER_AGENT = (
    "Mozilla/5.0 (compatible; FakeNewsDetector/1.0; "
    "+https://github.com/your-org/fakenews-detector)"
)
_REQUEST_TIMEOUT = 15  # seconds


@dataclass
class ArticleScrapeResult:
    text: str
    title: Optional[str]
    publish_date: Optional[datetime]
    source_url: str
    source_domain: str
    word_count: int


class ArticleScraper:
    """Scrape an article from a URL, returning cleaned body text."""

    def scrape(self, url: str) -> ArticleScrapeResult:
        """
        Main scrape entry point. Tries newspaper3k first, falls back to BS4.

        Raises:
            ValueError: if no meaningful text could be extracted.
        """
        domain = urlparse(url).netloc.replace("www.", "")
        logger.info("Scraping article", url=url, domain=domain)

        # Try newspaper3k
        result = self._scrape_newspaper(url, domain)
        if result and len(result.text.split()) >= 50:
            return result

        # Fallback: BeautifulSoup
        logger.info("Falling back to BeautifulSoup scraper", url=url)
        result = self._scrape_bs4(url, domain)
        if result and len(result.text.split()) >= 50:
            return result

        raise ValueError(
            f"Could not extract sufficient article text from {url}. "
            "The page may require JavaScript or be paywalled."
        )

    def _scrape_newspaper(self, url: str, domain: str) -> Optional[ArticleScrapeResult]:
        try:
            from newspaper import Article as NpArticle, Config

            config = Config()
            config.browser_user_agent = _USER_AGENT
            config.request_timeout = _REQUEST_TIMEOUT
            config.fetch_images = False
            config.memoize_articles = False

            article = NpArticle(url, config=config)
            article.download()
            article.parse()

            text = self._clean(article.text)
            if not text:
                return None

            return ArticleScrapeResult(
                text=text,
                title=article.title or None,
                publish_date=article.publish_date,
                source_url=url,
                source_domain=domain,
                word_count=len(text.split()),
            )
        except Exception as exc:
            logger.warning("newspaper3k scrape failed", url=url, error=str(exc))
            return None

    def _scrape_bs4(self, url: str, domain: str) -> Optional[ArticleScrapeResult]:
        try:
            import requests
            from bs4 import BeautifulSoup

            resp = requests.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=_REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            resp.raise_for_status()

            soup = BeautifulSoup(resp.content, "html.parser")

            # Remove boilerplate elements
            for tag in soup(["script", "style", "nav", "footer", "header",
                              "aside", "form", "iframe", "noscript"]):
                tag.decompose()

            # Try article tag first, then main, then body
            content_el = (
                soup.find("article")
                or soup.find("main")
                or soup.find("div", {"class": re.compile(r"article|content|story|post", re.I)})
                or soup.body
            )

            text = self._clean(content_el.get_text(separator=" ") if content_el else "")
            title_el = soup.find("h1") or soup.find("title")
            title = title_el.get_text().strip() if title_el else None

            if not text:
                return None

            return ArticleScrapeResult(
                text=text,
                title=title,
                publish_date=None,
                source_url=url,
                source_domain=domain,
                word_count=len(text.split()),
            )
        except Exception as exc:
            logger.warning("BeautifulSoup scrape failed", url=url, error=str(exc))
            return None

    @staticmethod
    def _clean(text: str) -> str:
        """Normalise whitespace and strip artefacts."""
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^\x20-\x7E\n]", "", text)
        return text[:8000]  # hard cap to keep inference fast
