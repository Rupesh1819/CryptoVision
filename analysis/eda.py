"""
Exploratory Data Analysis Module
Generates charts and statistics for cryptocurrency price data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import COINS, THEME


def plot_price_trends(df: pd.DataFrame, coin_id: str = "bitcoin") -> go.Figure:
    """Create an interactive price trend chart with volume overlay."""
    symbol = COINS.get(coin_id, {}).get("symbol", coin_id.upper())
    color = COINS.get(coin_id, {}).get("color", THEME["accent"])

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=[f"{symbol} Price Trend", "Trading Volume"],
    )

    # Price line
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["price"],
        mode="lines", name="Price",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=f"rgba({_hex_to_rgb(color)}, 0.1)",
    ), row=1, col=1)

    # Moving averages if available
    for ma_col, ma_name, dash in [
        ("ma_7", "7-Day MA", "dot"),
        ("ma_30", "30-Day MA", "dash"),
        ("ma_90", "90-Day MA", "dashdot"),
    ]:
        if ma_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[ma_col],
                mode="lines", name=ma_name,
                line=dict(width=1.5, dash=dash),
                opacity=0.7,
            ), row=1, col=1)

    # Volume bars
    if "total_volume" in df.columns:
        fig.add_trace(go.Bar(
            x=df["date"], y=df["total_volume"],
            name="Volume", marker_color=THEME["accent"],
            opacity=0.5,
        ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["bg_primary"],
        plot_bgcolor=THEME["bg_secondary"],
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=30, t=60, b=30),
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def plot_seasonal_decomposition(df: pd.DataFrame, period: int = 30) -> go.Figure:
    """Decompose price into trend, seasonal, and residual components."""
    ts = df.set_index("date")["price"].dropna()

    if len(ts) < 2 * period:
        period = max(7, len(ts) // 4)

    decomposition = seasonal_decompose(ts, model="multiplicative", period=period)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=["Original", "Trend", "Seasonal", "Residual"],
        vertical_spacing=0.05,
    )

    components = [
        (ts, "Original", THEME["accent"]),
        (decomposition.trend, "Trend", "#FF6B6B"),
        (decomposition.seasonal, "Seasonal", "#4ECDC4"),
        (decomposition.resid, "Residual", "#FFE66D"),
    ]

    for i, (data, name, color) in enumerate(components, 1):
        fig.add_trace(go.Scatter(
            x=data.index, y=data.values,
            mode="lines", name=name,
            line=dict(color=color, width=1.5),
        ), row=i, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["bg_primary"],
        plot_bgcolor=THEME["bg_secondary"],
        height=800,
        showlegend=False,
        margin=dict(l=50, r=30, t=40, b=30),
    )

    return fig


def plot_returns_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot daily returns distribution histogram with KDE."""
    if "returns" not in df.columns:
        df = df.copy()
        df["returns"] = df["price"].pct_change()

    returns = df["returns"].dropna()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns, nbinsx=80,
        name="Daily Returns",
        marker_color=THEME["accent"],
        opacity=0.7,
    ))

    fig.update_layout(
        title="Daily Returns Distribution",
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        template="plotly_dark",
        paper_bgcolor=THEME["bg_primary"],
        plot_bgcolor=THEME["bg_secondary"],
        height=400,
        annotations=[
            dict(
                text=f"Mean: {returns.mean():.4f} | Std: {returns.std():.4f} | "
                     f"Skew: {returns.skew():.4f} | Kurt: {returns.kurtosis():.4f}",
                xref="paper", yref="paper", x=0.5, y=1.05,
                showarrow=False, font=dict(size=12, color=THEME["text_secondary"]),
            )
        ],
    )

    return fig


def plot_correlation_heatmap(multi_coin_data: dict) -> go.Figure:
    """
    Plot price correlation heatmap across multiple coins.

    Args:
        multi_coin_data: Dict mapping coin_id -> DataFrame with 'date' and 'price' columns.
    """
    # Merge all coin prices on date
    merged = None
    for coin_id, df in multi_coin_data.items():
        symbol = COINS.get(coin_id, {}).get("symbol", coin_id.upper())
        temp = df[["date", "price"]].rename(columns={"price": symbol})
        if merged is None:
            merged = temp
        else:
            merged = merged.merge(temp, on="date", how="outer")

    if merged is None or len(merged.columns) < 2:
        return go.Figure()

    merged = merged.sort_values("date").set_index("date")
    returns = merged.pct_change().dropna()
    corr_matrix = returns.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 3),
        texttemplate="%{text}",
        textfont=dict(size=14),
    ))

    fig.update_layout(
        title="Cryptocurrency Returns Correlation Matrix",
        template="plotly_dark",
        paper_bgcolor=THEME["bg_primary"],
        plot_bgcolor=THEME["bg_secondary"],
        height=500,
        width=600,
    )

    return fig


def plot_candlestick(df: pd.DataFrame, coin_id: str = "bitcoin") -> go.Figure:
    """Create a candlestick chart (requires OHLC columns)."""
    symbol = COINS.get(coin_id, {}).get("symbol", coin_id.upper())

    if all(col in df.columns for col in ["open", "high", "low"]):
        fig = go.Figure(data=[go.Candlestick(
            x=df["date"],
            open=df["open"], high=df["high"],
            low=df["low"], close=df["price"],
            increasing_line_color=THEME["success"],
            decreasing_line_color=THEME["danger"],
        )])
    else:
        # Simulate OHLC from daily price
        fig = go.Figure(data=[go.Candlestick(
            x=df["date"],
            open=df["price"] * (1 + np.random.uniform(-0.02, 0.02, len(df))),
            high=df["price"] * (1 + np.abs(np.random.normal(0, 0.015, len(df)))),
            low=df["price"] * (1 - np.abs(np.random.normal(0, 0.015, len(df)))),
            close=df["price"],
            increasing_line_color=THEME["success"],
            decreasing_line_color=THEME["danger"],
        )])

    fig.update_layout(
        title=f"{symbol} Candlestick Chart",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor=THEME["bg_primary"],
        plot_bgcolor=THEME["bg_secondary"],
        height=500,
    )

    return fig


def compute_summary_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics for a coin's data."""
    returns = df["price"].pct_change().dropna() if "returns" not in df.columns else df["returns"].dropna()

    return {
        "current_price": df["price"].iloc[-1],
        "mean_price": df["price"].mean(),
        "std_price": df["price"].std(),
        "min_price": df["price"].min(),
        "max_price": df["price"].max(),
        "total_return": (df["price"].iloc[-1] / df["price"].iloc[0] - 1) * 100,
        "avg_daily_return": returns.mean() * 100,
        "daily_volatility": returns.std() * 100,
        "annualized_volatility": returns.std() * np.sqrt(365) * 100,
        "sharpe_ratio": (returns.mean() / returns.std()) * np.sqrt(365) if returns.std() > 0 else 0,
        "max_drawdown": _compute_max_drawdown(df["price"]),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
    }


def _compute_max_drawdown(prices: pd.Series) -> float:
    """Compute maximum drawdown percentage."""
    cummax = prices.cummax()
    drawdowns = (prices - cummax) / cummax
    return drawdowns.min() * 100


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to RGB string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"{r}, {g}, {b}"
