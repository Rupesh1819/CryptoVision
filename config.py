"""
Global Configuration for Crypto Time Series Analysis Project
"""

# ─── Target Cryptocurrencies ───────────────────────────────────────────
COINS = {
    "bitcoin": {"symbol": "BTC", "yfinance": "BTC-USD", "color": "#F7931A"},
    "ethereum": {"symbol": "ETH", "yfinance": "ETH-USD", "color": "#627EEA"},
    "solana": {"symbol": "SOL", "yfinance": "SOL-USD", "color": "#9945FF"},
    "cardano": {"symbol": "ADA", "yfinance": "ADA-USD", "color": "#0033AD"},
    "ripple": {"symbol": "XRP", "yfinance": "XRP-USD", "color": "#00AAE4"},
}

DEFAULT_COIN = "bitcoin"
DEFAULT_DAYS = 730  # ~2 years of historical data

# ─── API Endpoints ─────────────────────────────────────────────────────
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
NEWSAPI_KEY = ""  # Optional: set your NewsAPI key here for sentiment analysis
NEWSAPI_BASE_URL = "https://newsapi.org/v2"

# ─── Data Paths ────────────────────────────────────────────────────────
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# ─── Model Hyperparameters ─────────────────────────────────────────────
ARIMA_CONFIG = {
    "max_p": 5,
    "max_d": 2,
    "max_q": 5,
    "seasonal": False,
    "forecast_days": 30,
}

SARIMA_CONFIG = {
    "max_p": 3,
    "max_d": 1,
    "max_q": 3,
    "seasonal_period": 7,   # weekly seasonality for crypto
    "forecast_days": 30,
}

PROPHET_CONFIG = {
    "daily_seasonality": True,
    "weekly_seasonality": True,
    "yearly_seasonality": True,
    "changepoint_prior_scale": 0.05,
    "forecast_days": 30,
}

LSTM_CONFIG = {
    "lookback_window": 60,
    "epochs": 50,
    "batch_size": 32,
    "lstm_units_1": 128,
    "lstm_units_2": 64,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "forecast_days": 30,
}

TRAIN_TEST_SPLIT = 0.8

# ─── Dashboard Theme ───────────────────────────────────────────────────
THEME = {
    "bg_primary": "#0E1117",
    "bg_secondary": "#1A1D29",
    "bg_card": "#242736",
    "accent": "#00D4AA",
    "accent_secondary": "#7C3AED",
    "text_primary": "#FFFFFF",
    "text_secondary": "#9CA3AF",
    "success": "#10B981",
    "danger": "#EF4444",
    "warning": "#F59E0B",
    "gradient_start": "#667eea",
    "gradient_end": "#764ba2",
}
