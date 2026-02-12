"""
Sentiment Page — News feed, sentiment scoring, trend charts,
price-sentiment overlay, and word cloud.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import COINS, THEME
from sentiment.news_fetcher import fetch_crypto_news
from sentiment.analyzer import (
    analyze_news_sentiment, get_sentiment_summary,
    compute_daily_sentiment, plot_sentiment_trend,
)
from dashboard.components import (
    render_metric_card, render_section_header,
    render_sentiment_badge, render_news_card,
)


def render_sentiment_page(selected_coin: str, days: int):
    """Render the Sentiment Analysis page."""

    symbol = COINS.get(selected_coin, {}).get("symbol", selected_coin.upper())
    render_section_header(f"{symbol} Sentiment Intelligence", "💬")

    # ── Fetch and Analyze ──
    with st.spinner(f"Analyzing {symbol} market sentiment..."):
        query = COINS.get(selected_coin, {}).get("name", selected_coin)
        news_df = fetch_crypto_news(query=query, days_back=7)
        sentiment_df = analyze_news_sentiment(news_df)
        summary = get_sentiment_summary(sentiment_df)

    # ── Sentiment Gauge ──
    avg_compound = summary.get("avg_compound", 0)

    if avg_compound > 0.15:
        mood, mood_color, mood_emoji = "BULLISH", THEME["success"], "🟢"
    elif avg_compound < -0.15:
        mood, mood_color, mood_emoji = "BEARISH", THEME["danger"], "🔴"
    else:
        mood, mood_color, mood_emoji = "NEUTRAL", THEME["warning"], "🟡"

    st.markdown(f"""
    <div class="risk-indicator" style="border: 2px solid {mood_color}30;">
        <div style="color: {THEME['text_secondary']}; font-size: 0.85rem; margin-bottom: 0.4rem;">
            {mood_emoji} Market Sentiment for {symbol}
        </div>
        <div class="risk-value" style="color: {mood_color};">{mood}</div>
        <div style="color: {mood_color}; font-size: 1.1rem; font-weight: 600; margin-top: 0.3rem;">
            Average Compound Score: {avg_compound:.3f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Breakdown Cards ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card("Total Articles", str(summary.get("total_articles", 0)))
    with col2:
        pos_pct = summary.get("positive_pct", 0)
        total = summary.get("total_articles", 0)
        pos_count = int(total * pos_pct / 100) if total else 0
        render_metric_card("Positive", f"{pos_count} ({pos_pct:.0f}%)", delta_positive=True)
    with col3:
        neg_pct = summary.get("negative_pct", 0)
        neg_count = int(total * neg_pct / 100) if total else 0
        render_metric_card("Negative", f"{neg_count} ({neg_pct:.0f}%)", delta_positive=False)
    with col4:
        neu_pct = summary.get("neutral_pct", 0)
        neu_count = int(total * neu_pct / 100) if total else 0
        render_metric_card("Neutral", str(neu_count))

    st.markdown("---")

    # ── Tabbed Views ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Sentiment Trend", "🔗 Price vs Sentiment",
        "📰 News Feed", "☁️ Word Cloud",
    ])

    with tab1:
        # Daily sentiment trend
        if not sentiment_df.empty and "published_at" in sentiment_df.columns:
            daily = compute_daily_sentiment(sentiment_df)

            if not daily.empty:
                fig = go.Figure()

                # Compound score line
                fig.add_trace(go.Scatter(
                    x=daily["date"], y=daily["mean_compound"],
                    mode="lines+markers", name="Avg Compound Score",
                    line=dict(color=THEME["accent"], width=3),
                    marker=dict(size=6),
                ))

                # Color bars by sentiment
                colors = [THEME["success"] if v > 0.05 else THEME["danger"] if v < -0.05
                          else THEME["warning"] for v in daily["mean_compound"]]

                fig.add_trace(go.Bar(
                    x=daily["date"], y=daily["mean_compound"],
                    name="Daily Score",
                    marker_color=colors,
                    opacity=0.3,
                ))

                # Zero line
                fig.add_hline(y=0, line_dash="dash",
                              line_color=THEME["text_secondary"], opacity=0.3)

                fig.update_layout(
                    title=dict(text=f"{symbol} — Daily Sentiment Score"),
                    yaxis_title="Compound Score",
                    height=400,
                    template="plotly_dark",
                    paper_bgcolor=THEME["bg_primary"],
                    plot_bgcolor=THEME["bg_secondary"],
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    hovermode="x unified",
                    barmode="overlay",
                )
                fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
                fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
                st.plotly_chart(fig, use_container_width=True)

                # Article count breakdown
                if "count" in daily.columns:
                    render_metric_card(
                        "Avg Daily Articles",
                        f"{daily['count'].mean():.1f}",
                    )
            else:
                st.info("Not enough data for trend analysis.")
        else:
            st.info("No temporal data available for trend chart.")

    with tab2:
        # Price vs Sentiment overlay
        render_section_header("Price-Sentiment Correlation", "🔗")

        try:
            from data.collector import fetch_historical_data

            hist_df = fetch_historical_data(selected_coin, days=30)

            if not hist_df.empty and not sentiment_df.empty and "published_at" in sentiment_df.columns:
                daily = compute_daily_sentiment(sentiment_df)
                daily["date"] = pd.to_datetime(daily["date"])
                hist_df["date"] = pd.to_datetime(hist_df["date"])

                merged = pd.merge(
                    hist_df[["date", "price"]],
                    daily[["date", "mean_compound"]],
                    on="date", how="outer"
                ).sort_values("date")

                merged["price"] = merged["price"].ffill()
                merged["mean_compound"] = merged["mean_compound"].fillna(0)

                fig = make_subplots(specs=[[{"secondary_y": True}]])

                fig.add_trace(go.Scatter(
                    x=merged["date"], y=merged["price"],
                    mode="lines", name="Price",
                    line=dict(color=THEME["accent"], width=2.5),
                ), secondary_y=False)

                fig.add_trace(go.Scatter(
                    x=merged["date"], y=merged["mean_compound"],
                    mode="lines+markers", name="Sentiment Score",
                    line=dict(color=THEME["accent_secondary"], width=2),
                    marker=dict(size=5),
                ), secondary_y=True)

                fig.update_layout(
                    title=dict(text=f"{symbol} — Price vs Sentiment"),
                    height=450,
                    template="plotly_dark",
                    paper_bgcolor=THEME["bg_primary"],
                    plot_bgcolor=THEME["bg_secondary"],
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    hovermode="x unified",
                )
                fig.update_yaxes(title_text="Price (USD)", secondary_y=False,
                                gridcolor="rgba(255,255,255,0.04)")
                fig.update_yaxes(title_text="Sentiment Score", secondary_y=True,
                                gridcolor="rgba(255,255,255,0.04)")
                fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
                st.plotly_chart(fig, use_container_width=True)

                # Correlation stat
                valid = merged.dropna(subset=["price", "mean_compound"])
                if len(valid) > 5:
                    corr = valid["price"].corr(valid["mean_compound"])
                    render_metric_card(
                        "Price-Sentiment Correlation",
                        f"{corr:.3f}",
                        delta="Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak",
                        delta_positive=corr > 0,
                    )
            else:
                st.info("Insufficient data for price-sentiment overlay.")
        except Exception as e:
            st.error(f"Could not compute price-sentiment overlay: {e}")

    with tab3:
        # News Feed
        render_section_header("Latest Crypto News", "📰")

        if not sentiment_df.empty:
            # Sort by date (most recent first)
            feed = sentiment_df.copy()
            if "published_at" in feed.columns:
                feed["published_at"] = pd.to_datetime(feed["published_at"], errors="coerce")
                feed = feed.sort_values("published_at", ascending=False)

            for _, row in feed.head(15).iterrows():
                title = row.get("title", "Untitled")
                source = row.get("source", "Unknown")
                pub = str(row.get("published_at", ""))[:10]
                label = row.get("label", "Neutral")
                compound = row.get("compound", 0)

                render_news_card(title, source, pub, label, compound)
        else:
            st.info("No news articles available.")

    with tab4:
        # Word Cloud
        render_section_header("Trending Topics", "☁️")

        if not sentiment_df.empty and "title" in sentiment_df.columns:
            try:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt

                text = " ".join(sentiment_df["title"].dropna().tolist())

                if text.strip():
                    wc = WordCloud(
                        width=800, height=400,
                        background_color="#0f1117",
                        colormap="cool",
                        max_words=100,
                        max_font_size=150,
                        prefer_horizontal=0.7,
                        collocations=False,
                    ).generate(text)

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    fig.patch.set_facecolor("#0f1117")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No text content available for word cloud.")
            except ImportError:
                st.warning("WordCloud library not installed. Run: `pip install wordcloud`")
        else:
            st.info("No titles available for word cloud generation.")

    st.markdown("---")

    # ── Actionable Sentiment Summary ──
    if avg_compound > 0.15:
        detail = f"Positive coverage dominates ({pos_pct:.0f}% of articles), suggesting bullish sentiment pressure."
    elif avg_compound < -0.15:
        detail = f"Negative coverage prevails ({neg_pct:.0f}% of articles), suggesting bearish sentiment pressure."
    else:
        detail = "Coverage is mixed, indicating market uncertainty. Watch for sentiment shifts as potential leading indicators."

    summary_html = (
        f'<div style="background: {THEME["bg_card"]}; border-radius: 12px;'
        f' padding: 1.2rem 1.5rem; border-left: 3px solid {mood_color};">'
        f'<div style="color: {THEME["text_secondary"]}; font-size: 0.85rem; margin-bottom: 0.3rem;">'
        f'💡 Sentiment Insight</div>'
        f'<div style="color: {THEME["text_primary"]}; font-size: 0.95rem; line-height: 1.5;">'
        f'Market sentiment for <strong>{symbol}</strong> is currently '
        f'<strong style="color: {mood_color};">{mood.lower()}</strong> '
        f'with an average compound score of <strong>{avg_compound:.3f}</strong> '
        f'across {summary.get("total_articles", 0)} articles. {detail}'
        f'</div></div>'
    )
    st.markdown(summary_html, unsafe_allow_html=True)
