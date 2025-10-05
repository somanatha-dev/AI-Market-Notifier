"""
news_fetcher.py

Provides fetch_news_for_symbol(symbol, api_key=None, max_results=5)

Tries:
 - NewsAPI.org (if api_key given)
 - Fallback: yfinance .news (if available)
 - Final fallback: empty list

Dependencies:
    pip install requests feedparser yfinance
"""

import requests
import logging
from typing import List, Dict, Optional
import yfinance as yf

def fetch_news_for_symbol(symbol: str, api_key: Optional[str] = None, max_results: int = 5) -> List[Dict]:
    """
    Returns list of articles: dicts with keys 'title', 'description', 'url', 'source', 'publishedAt'
    """
    articles = []
    if api_key:
        # NewsAPI.org usage (example). Replace with your preferred news source if needed.
        try:
            q = symbol.split(".")[0]  # crude ticker -> query
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": q,
                "pageSize": max_results,
                "language": "en",
                "sortBy": "publishedAt",
                "apiKey": api_key
            }
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            for it in data.get("articles", [])[:max_results]:
                articles.append({
                    "title": it.get("title"),
                    "description": it.get("description"),
                    "url": it.get("url"),
                    "source": it.get("source", {}).get("name"),
                    "publishedAt": it.get("publishedAt")
                })
            if articles:
                return articles
        except Exception:
            logging.exception("NewsAPI fetch failed, falling back.")

    # Try yfinance news (some tickers include news)
    try:
        t = yf.Ticker(symbol)
        ny = t.news
        for it in (ny or [])[:max_results]:
            articles.append({
                "title": it.get("title"),
                "description": it.get("publisher"),
                "url": it.get("link"),
                "source": it.get("publisher"),
                "publishedAt": it.get("providerPublishTime")
            })
        if articles:
            return articles
    except Exception:
        logging.exception("yfinance news fetch failed.")

    # Fallback: empty
    return []
