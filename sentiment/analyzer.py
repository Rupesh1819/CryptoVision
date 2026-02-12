"""
Sentiment Analysis Module
Uses VADER for sentiment scoring of crypto news headlines.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import THEME


def get_sentiment_scores(texts: list) -> list:
    """
    Compute VADER sentiment scores for a list of texts.

    Returns list of dicts with: compound, positive, negative, neutral, label
    """
    analyzer = SentimentIntensityAnalyzer()
    results = []

    for text in texts:
        if not text or not isinstance(text, str):
            results.append({
                "compound": 0, "positive": 0, "negative": 0,
                "neutral": 1, "label": "Neutral",
            })
            continue

        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        results.append({
            "compound": compound,
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
            "label": label,
        })

    return results


def analyze_news_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sentiment scores to a news DataFrame.

    Expects columns: title, description
    Adds: compound, positive, negative, neutral, label, sentiment_text
    """
    if news_df.empty:
        return news_df

    df = news_df.copy()

    # Combine title and description for better sentiment analysis
    df["sentiment_text"] = df["title"].fillna("") + ". " + df["description"].fillna("")
    df["sentiment_text"] = df["sentiment_text"].str.strip(". ")

    scores = get_sentiment_scores(df["sentiment_text"].tolist())
    scores_df = pd.DataFrame(scores)

    for col in scores_df.columns:
        df[col] = scores_df[col].values

    return df


def compute_daily_sentiment(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment scores by date.

    Returns DataFrame with daily mean compound scores and counts.
    """
    df = sentiment_df.copy()

    if "published_at" in df.columns:
        df["date"] = pd.to_datetime(df["published_at"], errors="coerce").dt.normalize()
    else:
        return pd.DataFrame()

    daily = df.groupby("date").agg(
        mean_compound=("compound", "mean"),
        median_compound=("compound", "median"),
        count=("compound", "size"),
        positive_count=("label", lambda x: (x == "Positive").sum()),
        negative_count=("label", lambda x: (x == "Negative").sum()),
        neutral_count=("label", lambda x: (x == "Neutral").sum()),
    ).reset_index()

    daily["sentiment_ratio"] = daily["positive_count"] / (
        daily["positive_count"] + daily["negative_count"]
    ).replace(0, 1)

    return daily.sort_values("date")


def plot_sentiment_trend(daily_sentiment: pd.DataFrame) -> go.Figure:
    """Plot daily sentiment trend over time."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.35],
        subplot_titles=["Daily Sentiment Score", "Article Count by Sentiment"],
    )

    # Compound score line
    fig.add_trace(go.Scatter(
        x=daily_sentiment["date"],
        y=daily_sentiment["mean_compound"],
        mode="lines+markers",
        name="Mean Compound",
        line=dict(color=THEME["accent"], width=2),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 170, 0.1)",
    ), row=1, col=1)

    # Zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color=THEME["text_secondary"],
                   opacity=0.5, row=1, col=1)

    # Stacked bar for sentiment counts
    fig.add_trace(go.Bar(
        x=daily_sentiment["date"],
        y=daily_sentiment["positive_count"],
        name="Positive",
        marker_color=THEME["success"],
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=daily_sentiment["date"],
        y=daily_sentiment["neutral_count"],
        name="Neutral",
        marker_color=THEME["warning"],
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=daily_sentiment["date"],
        y=daily_sentiment["negative_count"],
        name="Negative",
        marker_color=THEME["danger"],
    ), row=2, col=1)

    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor=THEME["bg_primary"],
        plot_bgcolor=THEME["bg_secondary"],
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    fig.update_yaxes(title_text="Compound Score", row=1, col=1)
    fig.update_yaxes(title_text="Articles", row=2, col=1)

    return fig


def plot_sentiment_vs_price(daily_sentiment: pd.DataFrame,
                             price_df: pd.DataFrame) -> go.Figure:
    """Overlay sentiment with price movements."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.4],
        subplot_titles=["Price", "Sentiment Score"],
    )

    # Price
    fig.add_trace(go.Scatter(
        x=price_df["date"], y=price_df["price"],
        mode="lines", name="Price",
        line=dict(color=THEME["accent"], width=2),
    ), row=1, col=1)

    # Sentiment
    colors = [THEME["success"] if v >= 0 else THEME["danger"]
              for v in daily_sentiment["mean_compound"]]

    fig.add_trace(go.Bar(
        x=daily_sentiment["date"],
        y=daily_sentiment["mean_compound"],
        name="Sentiment",
        marker_color=colors,
        opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["bg_primary"],
        plot_bgcolor=THEME["bg_secondary"],
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Compound Score", row=2, col=1)

    return fig


def get_sentiment_summary(sentiment_df: pd.DataFrame) -> dict:
    """Compute overall sentiment summary statistics."""
    if sentiment_df.empty or "compound" not in sentiment_df.columns:
        return {"overall": "Neutral", "avg_compound": 0, "total_articles": 0}

    compound = sentiment_df["compound"]
    labels = sentiment_df["label"]

    avg = compound.mean()
    overall = "Bullish 🟢" if avg > 0.1 else "Bearish 🔴" if avg < -0.1 else "Neutral 🟡"

    return {
        "overall": overall,
        "avg_compound": round(avg, 4),
        "median_compound": round(compound.median(), 4),
        "total_articles": len(sentiment_df),
        "positive_pct": round((labels == "Positive").mean() * 100, 1),
        "negative_pct": round((labels == "Negative").mean() * 100, 1),
        "neutral_pct": round((labels == "Neutral").mean() * 100, 1),
        "strongest_positive": sentiment_df.loc[compound.idxmax(), "title"]
            if len(sentiment_df) > 0 else "",
        "strongest_negative": sentiment_df.loc[compound.idxmin(), "title"]
            if len(sentiment_df) > 0 else "",
    }
