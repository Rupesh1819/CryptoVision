"""
News Fetcher Module
Fetches cryptocurrency news from NewsAPI and RSS feeds as fallback.
"""

import os
import json
import time
import pandas as pd
import requests
import feedparser
from datetime import datetime, timedelta

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import NEWSAPI_KEY, NEWSAPI_BASE_URL, DATA_CACHE_DIR


# RSS feed sources for fallback
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptonews.com/news/feed/",
]

CRYPTO_KEYWORDS = [
    "bitcoin", "ethereum", "crypto", "blockchain", "BTC", "ETH",
    "cryptocurrency", "defi", "altcoin", "stablecoin", "solana",
    "cardano", "ripple", "XRP", "trading", "exchange",
]


def fetch_news_api(query: str = "cryptocurrency", days_back: int = 7,
                    page_size: int = 50) -> list:
    """
    Fetch news articles from NewsAPI.

    Returns list of dicts: {title, description, source, url, published_at}
    """
    if not NEWSAPI_KEY:
        return []

    try:
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = f"{NEWSAPI_BASE_URL}/everything"
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "publishedAt",
            "pageSize": min(page_size, 100),
            "language": "en",
            "apiKey": NEWSAPI_KEY,
        }

        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        articles = []
        for article in data.get("articles", []):
            articles.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "source": article.get("source", {}).get("name", "Unknown"),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
            })

        return articles

    except Exception as e:
        print(f"[NewsAPI Error] {e}")
        return []


def fetch_rss_feeds(max_articles: int = 50) -> list:
    """
    Fetch news from cryptocurrency RSS feeds.

    Returns list of dicts: {title, description, source, url, published_at}
    """
    articles = []

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            source = feed.feed.get("title", "Unknown")

            for entry in feed.entries[:20]:
                published = entry.get("published", entry.get("updated", ""))
                description = entry.get("summary", entry.get("description", ""))

                # Clean HTML from description
                if description:
                    import re
                    description = re.sub(r"<[^>]+>", "", description)[:300]

                articles.append({
                    "title": entry.get("title", ""),
                    "description": description,
                    "source": source,
                    "url": entry.get("link", ""),
                    "published_at": published,
                })

        except Exception as e:
            print(f"[RSS Error] {feed_url}: {e}")
            continue

    return articles[:max_articles]


def fetch_crypto_news(query: str = "cryptocurrency", days_back: int = 7,
                       use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch crypto news from all available sources.
    Prioritizes NewsAPI, falls back to RSS feeds, then to sample data.

    Returns DataFrame with news articles.
    """
    cache_file = os.path.join(DATA_CACHE_DIR, "news_cache.csv")

    # Check cache
    if use_cache and os.path.exists(cache_file):
        mod_time = os.path.getmtime(cache_file)
        if (time.time() - mod_time) < 3600:  # 1 hour cache
            return pd.read_csv(cache_file)

    # Try NewsAPI first
    articles = fetch_news_api(query, days_back)

    # Fallback to RSS
    if not articles:
        articles = fetch_rss_feeds()

    # Fallback to sample data
    if not articles:
        articles = _generate_sample_news()

    df = pd.DataFrame(articles)

    if not df.empty and use_cache:
        df.to_csv(cache_file, index=False)

    return df


def _generate_sample_news() -> list:
    """Generate sample crypto news for offline development."""
    import random
    random.seed(42)

    headlines = [
        ("Bitcoin Surges Past $50,000 as Institutional Interest Grows",
         "Major institutional investors are increasing their Bitcoin holdings, driving the price to new highs."),
        ("Ethereum 2.0 Staking Reaches Record Levels",
         "The amount of ETH staked in the Ethereum 2.0 deposit contract has reached all-time highs."),
        ("Federal Reserve Signals Impact on Crypto Markets",
         "The latest Federal Reserve meeting minutes suggest potential implications for cryptocurrency markets."),
        ("Solana Network Experiences Brief Outage",
         "The Solana blockchain experienced a temporary outage, raising concerns about network reliability."),
        ("DeFi Total Value Locked Hits $100 Billion",
         "Decentralized finance protocols have collectively locked over $100 billion in assets."),
        ("Regulatory Clarity Boosts Crypto Adoption in Asia",
         "New cryptocurrency regulations in Singapore and Japan are providing clarity for market participants."),
        ("Bitcoin Mining Difficulty Reaches All-Time High",
         "Bitcoin network hash rate continues to grow as miners expand operations globally."),
        ("Whale Activity Detected: Large BTC Transfer Spotted",
         "On-chain data reveals a significant Bitcoin transfer from an unknown wallet to a major exchange."),
        ("Cryptocurrency Market Cap Exceeds $2 Trillion",
         "The total cryptocurrency market capitalization has surpassed $2 trillion for the first time."),
        ("NFT Market Shows Signs of Recovery After Slump",
         "Non-fungible token trading volumes are picking up after months of declining activity."),
        ("Central Banks Explore Digital Currency Options",
         "Several central banks are accelerating their research into central bank digital currencies."),
        ("Crypto Exchange Reports Record Trading Volume",
         "A major cryptocurrency exchange has reported its highest ever daily trading volume."),
        ("Bitcoin ETF Sees Massive Inflows This Week",
         "Spot Bitcoin ETFs continue to attract billions in new investments from institutional and retail investors."),
        ("Cardano Launches Smart Contract Upgrade",
         "The Cardano blockchain has successfully deployed a major smart contract upgrade."),
        ("Crypto Fear and Greed Index Hits Extreme Greed",
         "The popular market sentiment indicator has reached extreme greed territory, signaling potential market euphoria."),
    ]

    articles = []
    now = datetime.now()
    for i, (title, desc) in enumerate(headlines):
        articles.append({
            "title": title,
            "description": desc,
            "source": random.choice(["CoinDesk", "CoinTelegraph", "CryptoNews", "Bloomberg Crypto"]),
            "url": f"https://example.com/news/{i}",
            "published_at": (now - timedelta(hours=random.randint(1, 168))).isoformat(),
        })

    return articles
