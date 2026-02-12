"""
Volatility Analysis Module
Computes and visualizes volatility metrics, Bollinger Bands, ATR, and risk indicators.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import THEME, COINS


def compute_rolling_volatility(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    """Compute rolling volatility for multiple windows."""
    if windows is None:
        windows = [7, 14, 30, 60]

    df = df.copy()
    if "returns" not in df.columns:
        df["returns"] = df["price"].pct_change()

    for w in windows:
        df[f"vol_{w}d"] = df["returns"].rolling(window=w).std() * np.sqrt(365) * 100

    return df


def compute_bollinger_bands(df: pd.DataFrame, window: int = 20,
                             num_std: float = 2.0) -> pd.DataFrame:
    """Compute Bollinger Bands."""
    df = df.copy()
    df["bb_middle"] = df["price"].rolling(window=window).mean()
    rolling_std = df["price"].rolling(window=window).std()
    df["bb_upper"] = df["bb_middle"] + num_std * rolling_std
    df["bb_lower"] = df["bb_middle"] - num_std * rolling_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_pct"] = (df["price"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Average True Range (ATR)."""
    df = df.copy()

    if "high" in df.columns and "low" in df.columns:
        high = df["high"]
        low = df["low"]
    else:
        # Estimate from daily price
        high = df["price"] * (1 + df["price"].pct_change().abs() * 0.5)
        low = df["price"] * (1 - df["price"].pct_change().abs() * 0.5)

    close_prev = df["price"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=period).mean()
    df["atr_pct"] = (df["atr"] / df["price"]) * 100

    return df


def compute_risk_metrics(df: pd.DataFrame, risk_free_rate: float = 0.05) -> dict:
    """
    Compute comprehensive risk metrics.

    Returns dict with: sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
    var_95, cvar_95, avg_daily_vol, annualized_vol
    """
    if "returns" not in df.columns:
        returns = df["price"].pct_change().dropna()
    else:
        returns = df["returns"].dropna()

    daily_rf = risk_free_rate / 365
    excess_returns = returns - daily_rf

    # Sharpe Ratio
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(365) if returns.std() > 0 else 0

    # Sortino Ratio (downside deviation only)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(365) if downside_std > 0 else 0

    # Max Drawdown
    cummax = df["price"].cummax()
    drawdowns = (df["price"] - cummax) / cummax
    max_drawdown = drawdowns.min() * 100

    # Calmar Ratio
    annual_return = (df["price"].iloc[-1] / df["price"].iloc[0]) ** (365 / max(len(df), 1)) - 1
    calmar = abs(annual_return / (max_drawdown / 100)) if max_drawdown != 0 else 0

    # Value at Risk (95%)
    var_95 = np.percentile(returns, 5) * 100

    # Conditional VaR (expected shortfall)
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100

    return {
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": round(max_drawdown, 2),
        "calmar_ratio": round(calmar, 4),
        "var_95": round(var_95, 2),
        "cvar_95": round(cvar_95, 2),
        "avg_daily_volatility": round(returns.std() * 100, 4),
        "annualized_volatility": round(returns.std() * np.sqrt(365) * 100, 2),
        "annual_return_pct": round(annual_return * 100, 2),
    }


def plot_bollinger_bands(df: pd.DataFrame, coin_id: str = "bitcoin") -> go.Figure:
    """Interactive Bollinger Bands chart."""
    symbol = COINS.get(coin_id, {}).get("symbol", coin_id.upper())
    color = COINS.get(coin_id, {}).get("color", THEME["accent"])

    df = compute_bollinger_bands(df)

    fig = go.Figure()

    # Bollinger Bands fill
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["bb_upper"],
        mode="lines", name="Upper Band",
        line=dict(color="rgba(255, 107, 107, 0.3)", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["bb_lower"],
        mode="lines", name="Lower Band",
        line=dict(color="rgba(78, 205, 196, 0.3)", width=1),
        fill="tonexty", fillcolor="rgba(100, 100, 255, 0.08)",
    ))

    # Middle band
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["bb_middle"],
        mode="lines", name="Middle Band (SMA 20)",
        line=dict(color=THEME["warning"], width=1.5, dash="dash"),
    ))

    # Price
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["price"],
        mode="lines", name="Price",
        line=dict(color=color, width=2),
    ))

    fig.update_layout(
        title=f"{symbol} Bollinger Bands",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        paper_bgcolor=THEME["bg_primary"],
        plot_bgcolor=THEME["bg_secondary"],
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def plot_rolling_volatility(df: pd.DataFrame, coin_id: str = "bitcoin") -> go.Figure:
    """Interactive rolling volatility chart."""
    symbol = COINS.get(coin_id, {}).get("symbol", coin_id.upper())

    df = compute_rolling_volatility(df)

    fig = go.Figure()

    colors = [THEME["accent"], "#FF6B6B", "#4ECDC4", "#FFE66D"]
    for i, w in enumerate([7, 14, 30, 60]):
        col = f"vol_{w}d"
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                mode="lines", name=f"{w}-Day Volatility",
                line=dict(color=colors[i % len(colors)], width=1.5),
            ))

    fig.update_layout(
        title=f"{symbol} Rolling Annualized Volatility (%)",
        yaxis_title="Volatility (%)",
        template="plotly_dark",
        paper_bgcolor=THEME["bg_primary"],
        plot_bgcolor=THEME["bg_secondary"],
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def plot_drawdown(df: pd.DataFrame, coin_id: str = "bitcoin") -> go.Figure:
    """Plot drawdown from peak chart."""
    symbol = COINS.get(coin_id, {}).get("symbol", coin_id.upper())

    cummax = df["price"].cummax()
    drawdown = ((df["price"] - cummax) / cummax) * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=drawdown,
        mode="lines", name="Drawdown",
        fill="tozeroy",
        line=dict(color=THEME["danger"], width=1.5),
        fillcolor="rgba(239, 68, 68, 0.2)",
    ))

    fig.update_layout(
        title=f"{symbol} Drawdown from Peak (%)",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        paper_bgcolor=THEME["bg_primary"],
        plot_bgcolor=THEME["bg_secondary"],
        height=350,
    )

    return fig
