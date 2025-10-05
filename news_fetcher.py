#!/usr/bin/env python3
"""
news_fetcher.py
Fetch recent headlines from curated trusted sources using RSS; fallback to lightweight scraping.
Returns list of dicts: {source, title, url, publishedAt, summary}
"""
import os
import logging
import feedparser
import requests
from bs4 import BeautifulSoup

log = logging.getLogger("news_fetcher")
log.setLevel(logging.INFO)

DEFAULT_RSS = [
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/1977021501.cms",
    "https://www.livemint.com/rss/markets",
    "https://www.business-standard.com/rss/markets-xml-feed.rss",
    "https://www.reuters.com/markets/india/rss",
    "https://www.moneycontrol.com/rss/business/markets/feed.xml"
]

DEFAULT_FALLBACK = [
    "https://economictimes.indiatimes.com/markets",
    "https://www.livemint.com/market",
    "https://www.business-standard.com",
    "https://www.reuters.com/finance/markets",
    "https://www.moneycontrol.com"
]

REQUEST_TIMEOUT = 8
MAX_ITEMS_PER_SOURCE = 6
USER_AGENT = "market-notifier-bot/1.0"

def _get_env_list(name, default):
    raw = os.getenv(name, None)
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    return default

RSS_FEEDS = _get_env_list("TRUSTED_RSS", DEFAULT_RSS)
FALLBACK_PAGES = _get_env_list("TRUSTED_FALLBACK", DEFAULT_FALLBACK)


def _parse_feed(url, max_items=MAX_ITEMS_PER_SOURCE):
    try:
        f = feedparser.parse(url)
        items = []
        for entry in f.entries[:max_items]:
            title = entry.get("title") or entry.get("summary") or ""
            link = entry.get("link") or ""
            published = entry.get("published") or entry.get("updated") or ""
            summary = entry.get("summary", "")
            items.append({
                "source": f.feed.get("title", url),
                "title": title.strip(),
                "url": link,
                "publishedAt": published,
                "summary": BeautifulSoup(summary, "lxml").get_text() if summary else ""
            })
        return items
    except Exception:
        log.exception("RSS parse failed for %s", url)
        return []


def _scrape_page(url, max_items=MAX_ITEMS_PER_SOURCE):
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        candidates = []
        for tag in soup.find_all(["h1", "h2", "h3", "a"], limit=60):
            text = tag.get_text().strip()
            href = tag.get("href") or ""
            if not text or len(text) < 30:
                continue
            if href and href.startswith("/"):
                from urllib.parse import urljoin
                href = urljoin(url, href)
            candidates.append({"title": text, "url": href})
            if len(candidates) >= max_items:
                break
        results = []
        for it in candidates[:max_items]:
            results.append({
                "source": url,
                "title": it["title"],
                "url": it["url"],
                "publishedAt": "",
                "summary": ""
            })
        return results
    except Exception:
        log.exception("Scrape failed for %s", url)
        return []


def fetch_headlines(limit=6):
    seen = set()
    results = []
    for feed in RSS_FEEDS:
        items = _parse_feed(feed, max_items=limit)
        for it in items:
            t = it["title"]
            if not t or t in seen:
                continue
            results.append(it)
            seen.add(t)
            if len(results) >= limit:
                return results
    for page in FALLBACK_PAGES:
        items = _scrape_page(page, max_items=limit)
        for it in items:
            t = it["title"]
            if not t or t in seen:
                continue
            results.append(it)
            seen.add(t)
            if len(results) >= limit:
                return results
    return results
