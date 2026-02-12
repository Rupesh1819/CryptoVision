"""
Data Collection Module
Fetches historical and real-time cryptocurrency data from CoinGecko and yfinance.
"""

import os
import time
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import COINGECKO_BASE_URL, COINS, DATA_CACHE_DIR, DEFAULT_DAYS


def fetch_historical_data(coin_id: str = "bitcoin", days: int = DEFAULT_DAYS,
                          use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch historical price data from CoinGecko API.

    Args:
        coin_id: CoinGecko coin identifier (e.g. 'bitcoin', 'ethereum')
        days: Number of days of historical data
        use_cache: Whether to use/save cached data

    Returns:
        DataFrame with columns: date, price, market_cap, total_volume
    """
    cache_file = os.path.join(DATA_CACHE_DIR, f"{coin_id}_{days}d_history.csv")

    # Return cached data if fresh (< 1 hour old)
    if use_cache and os.path.exists(cache_file):
        mod_time = os.path.getmtime(cache_file)
        if (time.time() - mod_time) < 3600:
            df = pd.read_csv(cache_file, parse_dates=["date"])
            return df

    try:
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame({
            "date": [datetime.utcfromtimestamp(p[0] / 1000) for p in data["prices"]],
            "price": [p[1] for p in data["prices"]],
            "market_cap": [p[1] for p in data["market_caps"]],
            "total_volume": [p[1] for p in data["total_volumes"]],
        })

        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
        df["coin_id"] = coin_id
        df["symbol"] = COINS.get(coin_id, {}).get("symbol", coin_id.upper())

        if use_cache:
            df.to_csv(cache_file, index=False)

        return df

    except Exception as e:
        print(f"[CoinGecko Error] {e} — falling back to yfinance")
        return _fetch_yfinance_fallback(coin_id, days, cache_file, use_cache)


def _fetch_yfinance_fallback(coin_id: str, days: int,
                              cache_file: str, use_cache: bool) -> pd.DataFrame:
    """Fallback: fetch OHLCV data from Yahoo Finance."""
    coin_info = COINS.get(coin_id, {})
    ticker = coin_info.get("yfinance", f"{coin_id.upper()}-USD")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return _generate_sample_data(coin_id, days)

        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df = pd.DataFrame({
            "date": data.index,
            "price": data["Close"].values,
            "open": data["Open"].values,
            "high": data["High"].values,
            "low": data["Low"].values,
            "total_volume": data["Volume"].values,
        })

        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["market_cap"] = df["price"] * df["total_volume"]  # approximate
        df["coin_id"] = coin_id
        df["symbol"] = coin_info.get("symbol", coin_id.upper())
        df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

        if use_cache:
            df.to_csv(cache_file, index=False)

        return df

    except Exception as e:
        print(f"[yfinance Error] {e} — using generated sample data")
        return _generate_sample_data(coin_id, days)


def _generate_sample_data(coin_id: str, days: int) -> pd.DataFrame:
    """Generate synthetic sample data for offline development/testing."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=days, freq="D")

    base_prices = {
        "bitcoin": 45000, "ethereum": 3000, "solana": 100,
        "cardano": 0.5, "ripple": 0.6,
    }
    base = base_prices.get(coin_id, 1000)

    # Random walk with drift
    returns = np.random.normal(0.001, 0.03, size=days)
    price = base * np.cumprod(1 + returns)

    volume = np.random.lognormal(mean=20, sigma=1.5, size=days)

    df = pd.DataFrame({
        "date": dates,
        "price": price,
        "market_cap": price * volume,
        "total_volume": volume,
        "coin_id": coin_id,
        "symbol": COINS.get(coin_id, {}).get("symbol", coin_id.upper()),
    })

    return df


def fetch_realtime_prices(coin_ids: list = None) -> pd.DataFrame:
    """
    Fetch current real-time prices for multiple coins.

    Returns:
        DataFrame with columns: coin_id, symbol, price, market_cap,
        volume_24h, change_24h, last_updated
    """
    if coin_ids is None:
        coin_ids = list(COINS.keys())

    try:
        ids_str = ",".join(coin_ids)
        url = f"{COINGECKO_BASE_URL}/simple/price"
        params = {
            "ids": ids_str,
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true",
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        rows = []
        for cid in coin_ids:
            if cid in data:
                d = data[cid]
                rows.append({
                    "coin_id": cid,
                    "symbol": COINS.get(cid, {}).get("symbol", cid.upper()),
                    "price": d.get("usd", 0),
                    "market_cap": d.get("usd_market_cap", 0),
                    "volume_24h": d.get("usd_24h_vol", 0),
                    "change_24h": d.get("usd_24h_change", 0),
                    "last_updated": datetime.utcfromtimestamp(
                        d.get("last_updated_at", time.time())
                    ),
                })

        return pd.DataFrame(rows) if rows else _generate_realtime_fallback(coin_ids)

    except Exception as e:
        print(f"[Real-time Error] {e} — using fallback")
        return _generate_realtime_fallback(coin_ids)


def _generate_realtime_fallback(coin_ids: list) -> pd.DataFrame:
    """Generate mock real-time prices for offline use."""
    np.random.seed(int(time.time()) % 1000)
    base_prices = {
        "bitcoin": 45000, "ethereum": 3000, "solana": 100,
        "cardano": 0.5, "ripple": 0.6,
    }

    rows = []
    for cid in coin_ids:
        base = base_prices.get(cid, 1000)
        price = base * (1 + np.random.normal(0, 0.02))
        rows.append({
            "coin_id": cid,
            "symbol": COINS.get(cid, {}).get("symbol", cid.upper()),
            "price": round(price, 2),
            "market_cap": round(price * np.random.lognormal(20, 1), 0),
            "volume_24h": round(np.random.lognormal(18, 1), 0),
            "change_24h": round(np.random.normal(0, 3), 2),
            "last_updated": datetime.utcnow(),
        })

    return pd.DataFrame(rows)


def fetch_multi_coin_data(coin_ids: list = None, days: int = DEFAULT_DAYS) -> dict:
    """
    Fetch historical data for multiple coins.

    Returns:
        Dict mapping coin_id -> DataFrame
    """
    if coin_ids is None:
        coin_ids = list(COINS.keys())

    result = {}
    for cid in coin_ids:
        result[cid] = fetch_historical_data(cid, days)
        time.sleep(1.5)  # Rate limiting for CoinGecko free tier

    return result
