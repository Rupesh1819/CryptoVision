"""
Data Preprocessing Module
Handles missing values, outlier detection, feature engineering, and train/test splitting.
"""

import pandas as pd
import numpy as np
from scipy import stats

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TRAIN_TEST_SPLIT


def handle_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        df: Input DataFrame
        method: 'ffill' (forward fill), 'interpolate', or 'drop'
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if method == "ffill":
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
    elif method == "interpolate":
        df[numeric_cols] = df[numeric_cols].interpolate(method="time")
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
    elif method == "drop":
        df = df.dropna(subset=numeric_cols)

    return df.reset_index(drop=True)


def detect_outliers_iqr(df: pd.DataFrame, column: str = "price",
                         multiplier: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers using the Interquartile Range (IQR) method.

    Returns:
        DataFrame with an 'is_outlier_iqr' boolean column added.
    """
    df = df.copy()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    df["is_outlier_iqr"] = (df[column] < lower) | (df[column] > upper)
    return df


def detect_outliers_zscore(df: pd.DataFrame, column: str = "price",
                            threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect outliers using Z-score method.

    Returns:
        DataFrame with an 'is_outlier_zscore' boolean column added.
    """
    df = df.copy()
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    df["is_outlier_zscore"] = False
    valid_idx = df[column].dropna().index
    df.loc[valid_idx, "is_outlier_zscore"] = z_scores > threshold
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer technical features for time series analysis.

    Adds: returns, log_returns, moving averages (7/30/90),
    rolling volatility, RSI, MACD, Bollinger Bands.
    """
    df = df.copy()

    # ── Returns ──
    df["returns"] = df["price"].pct_change()
    df["log_returns"] = np.log(df["price"] / df["price"].shift(1))

    # ── Moving Averages ──
    for window in [7, 30, 90]:
        df[f"ma_{window}"] = df["price"].rolling(window=window).mean()
        df[f"ema_{window}"] = df["price"].ewm(span=window, adjust=False).mean()

    # ── Rolling Volatility ──
    df["volatility_30d"] = df["returns"].rolling(window=30).std() * np.sqrt(30)

    # ── RSI (Relative Strength Index) ──
    df["rsi_14"] = _compute_rsi(df["price"], period=14)

    # ── MACD ──
    ema12 = df["price"].ewm(span=12, adjust=False).mean()
    ema26 = df["price"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands ──
    df["bb_middle"] = df["price"].rolling(window=20).mean()
    bb_std = df["price"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

    # ── Price Momentum ──
    for lag in [1, 7, 30]:
        df[f"momentum_{lag}d"] = df["price"] / df["price"].shift(lag) - 1

    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def prepare_for_modeling(df: pd.DataFrame, target_col: str = "price") -> dict:
    """
    Prepare data for time series modeling: train/test split by date.

    Returns:
        Dictionary with 'train', 'test', 'split_date', 'full' DataFrames.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    return {
        "train": train,
        "test": test,
        "split_date": df.iloc[split_idx]["date"],
        "full": df,
        "target_col": target_col,
    }


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline:
    1. Handle missing values
    2. Detect outliers
    3. Add technical features
    """
    df = handle_missing_values(df, method="ffill")
    df = detect_outliers_iqr(df, column="price")
    df = detect_outliers_zscore(df, column="price")
    df = add_technical_features(df)
    return df
