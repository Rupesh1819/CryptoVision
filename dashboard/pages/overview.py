"""
Overview Page — Real-time price cards, sparklines, market summary, and quick stats.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import time

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import COINS, THEME
from data.collector import fetch_realtime_prices, fetch_historical_data
from dashboard.components import render_metric_card, render_section_header


def render_overview_page(selected_coin: str, days: int):
    """Render the Overview / Real-time page."""

    render_section_header("Live Market Overview", "🌐")

    # ── Live Prices ──
    with st.spinner("Fetching live prices..."):
        live_df = fetch_realtime_prices()

    if not live_df.empty:
        cols = st.columns(len(live_df))
        for i, (_, row) in enumerate(live_df.iterrows()):
            with cols[i]:
                coin_color = COINS.get(row['coin_id'], {}).get('color', THEME['accent'])
                change = row.get("change_24h", 0)
                is_positive = change >= 0
                price_str = f"${row['price']:,.2f}" if row['price'] >= 1 else f"${row['price']:.4f}"

                st.markdown(f"""
                <div class="metric-card" style="border-top: 3px solid {coin_color};">
                    <div class="label">{row['symbol']}</div>
                    <div class="value">{price_str}</div>
                    <div class="delta {'delta-positive' if is_positive else 'delta-negative'}">
                        {'▲' if is_positive else '▼'} {abs(change):.2f}% (24h)
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Market Summary Stats ──
    render_section_header("Market Summary", "📊")

    if not live_df.empty:
        total_mcap = live_df["market_cap"].sum()
        total_vol = live_df["volume_24h"].sum()
        avg_change = live_df["change_24h"].mean()
        best_coin = live_df.loc[live_df["change_24h"].idxmax()]
        worst_coin = live_df.loc[live_df["change_24h"].idxmin()]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Total Market Cap", f"${total_mcap/1e9:,.1f}B")
        with col2:
            render_metric_card("24h Volume", f"${total_vol/1e9:,.1f}B")
        with col3:
            render_metric_card(
                "Top Performer",
                f"{best_coin['symbol']}",
                delta=f"{best_coin['change_24h']:+.2f}%",
                delta_positive=best_coin['change_24h'] >= 0,
            )
        with col4:
            render_metric_card(
                "Worst Performer",
                f"{worst_coin['symbol']}",
                delta=f"{worst_coin['change_24h']:+.2f}%",
                delta_positive=worst_coin['change_24h'] >= 0,
            )

    st.markdown("---")

    # ── Sparkline Charts ──
    render_section_header("7-Day Price Action", "✨")

    spark_cols = st.columns(min(len(COINS), 5))
    for i, (coin_id, info) in enumerate(COINS.items()):
        with spark_cols[i % len(spark_cols)]:
            try:
                spark_df = fetch_historical_data(coin_id, days=7)
                if not spark_df.empty and len(spark_df) > 1:
                    price_change = ((spark_df["price"].iloc[-1] / spark_df["price"].iloc[0]) - 1) * 100
                    color = THEME["success"] if price_change >= 0 else THEME["danger"]

                    fig = go.Figure(go.Scatter(
                        x=spark_df["date"], y=spark_df["price"],
                        mode="lines",
                        line=dict(color=color, width=2.5),
                        fill="tozeroy",
                        fillcolor=f"rgba({_hex_to_rgb(color)}, 0.12)",
                    ))
                    fig.update_layout(
                        height=130,
                        margin=dict(l=0, r=0, t=28, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        title=dict(
                            text=f"<b>{info['symbol']}</b>  <span style='color:{color}'>{price_change:+.1f}%</span>",
                            font=dict(size=13, color=THEME["text_primary"]),
                        ),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption(f"{info['symbol']}: Loading...")
            except Exception:
                st.caption(f"{info['symbol']}: Unavailable")

    st.markdown("---")

    # ── Selected Coin Deep Dive ──
    symbol = COINS.get(selected_coin, {}).get("symbol", selected_coin.upper())
    coin_color = COINS.get(selected_coin, {}).get("color", THEME["accent"])
    render_section_header(f"{symbol} Quick Analysis", "🔎")

    with st.spinner(f"Loading {symbol} data..."):
        hist_df = fetch_historical_data(selected_coin, days=min(days, 365))

    if not hist_df.empty and len(hist_df) > 7:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Price chart with gradient fill
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_df["date"], y=hist_df["price"],
                mode="lines", name="Price",
                line=dict(color=coin_color, width=2.5),
                fill="tozeroy",
                fillcolor=f"rgba({_hex_to_rgb(coin_color)}, 0.08)",
            ))

            # Add 30-day MA
            if len(hist_df) > 30:
                ma30 = hist_df["price"].rolling(30).mean()
                fig.add_trace(go.Scatter(
                    x=hist_df["date"], y=ma30,
                    mode="lines", name="30-Day MA",
                    line=dict(color=THEME["warning"], width=1.5, dash="dash"),
                    opacity=0.7,
                ))

            fig.update_layout(
                height=350,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=THEME["bg_secondary"],
                margin=dict(l=50, r=20, t=10, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode="x unified",
            )
            fig.update_yaxes(title_text="Price (USD)", gridcolor="rgba(255,255,255,0.04)")
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Quick stats
            current = hist_df["price"].iloc[-1]
            high = hist_df["price"].max()
            low = hist_df["price"].min()
            avg = hist_df["price"].mean()
            total_return = ((current / hist_df["price"].iloc[0]) - 1) * 100
            vol = hist_df["price"].pct_change().std() * (365 ** 0.5) * 100

            render_metric_card("Current", f"${current:,.2f}")
            render_metric_card("Period High", f"${high:,.2f}")
            render_metric_card("Period Low", f"${low:,.2f}")
            render_metric_card("Return",
                                f"{total_return:+.1f}%",
                                delta_positive=total_return >= 0)
            render_metric_card("Annualized Vol", f"{vol:.1f}%")


def _hex_to_rgb(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"{r}, {g}, {b}"
