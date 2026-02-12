"""
Volatility & Risk Page — Bollinger Bands, ATR, risk metrics dashboard,
rolling volatility comparison, and drawdown analysis.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import COINS, THEME
from data.collector import fetch_historical_data
from data.preprocessor import preprocess_pipeline
from analysis.volatility import (
    compute_bollinger_bands, compute_atr, compute_risk_metrics,
    plot_bollinger_bands, plot_drawdown,
)
from dashboard.components import render_metric_card, render_section_header


def render_volatility_page(selected_coin: str, days: int):
    """Render the Volatility & Risk page."""

    symbol = COINS.get(selected_coin, {}).get("symbol", selected_coin.upper())
    coin_color = COINS.get(selected_coin, {}).get("color", THEME["accent"])
    render_section_header(f"{symbol} Volatility & Risk Analysis", "⚡")

    # ── Fetch Data ──
    with st.spinner(f"Analyzing {symbol} risk profile..."):
        raw_df = fetch_historical_data(selected_coin, days)
        df = preprocess_pipeline(raw_df)
        bb_df = compute_bollinger_bands(df)
        risk_metrics = compute_risk_metrics(df)

    # ── Risk Level Indicator ──
    # Determine risk level based on annualized volatility
    ann_vol = risk_metrics.get("annualized_volatility", 0)
    if ann_vol > 80:
        risk_level, risk_label, risk_color = "EXTREME", "Extreme Risk", THEME["danger"]
    elif ann_vol > 60:
        risk_level, risk_label, risk_color = "HIGH", "High Risk", "#FF6B6B"
    elif ann_vol > 40:
        risk_level, risk_label, risk_color = "MODERATE", "Moderate Risk", THEME["warning"]
    else:
        risk_level, risk_label, risk_color = "LOW", "Low Risk", THEME["success"]

    st.markdown(f"""
    <div class="risk-indicator" style="border: 2px solid {risk_color}30;">
        <div style="color: {THEME['text_secondary']}; font-size: 0.85rem; margin-bottom: 0.5rem;">
            📊 Overall Risk Assessment for {symbol}
        </div>
        <div class="risk-value" style="color: {risk_color};">{risk_level}</div>
        <div style="color: {risk_color}; font-size: 1rem; font-weight: 600; margin-top: 0.3rem;">
            {risk_label} — Annualized Volatility: {ann_vol:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Key Risk Metrics ──
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        sharpe = risk_metrics.get("sharpe_ratio", 0)
        render_metric_card(
            "Sharpe Ratio", f"{sharpe:.3f}",
            delta="Good" if sharpe > 1 else "Moderate" if sharpe > 0 else "Poor",
            delta_positive=sharpe > 0,
        )
    with col2:
        sortino = risk_metrics.get("sortino_ratio", 0)
        render_metric_card(
            "Sortino Ratio", f"{sortino:.3f}",
            delta_positive=sortino > 0,
        )
    with col3:
        max_dd = risk_metrics.get("max_drawdown", 0)
        render_metric_card(
            "Max Drawdown", f"{max_dd:.1f}%",
            delta=f"{'Severe' if max_dd < -30 else 'Manageable'}",
            delta_positive=max_dd > -20,
        )
    with col4:
        var_95 = risk_metrics.get("var_95", 0)
        render_metric_card("VaR (95%)", f"{var_95:.2f}%")
    with col5:
        cvar = risk_metrics.get("cvar_95", 0)
        render_metric_card("CVaR (95%)", f"{cvar:.2f}%")

    st.markdown("---")

    # ── Tabbed Charts ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Bollinger Bands", "📊 Rolling Volatility",
        "📉 Drawdown Analysis", "⚡ ATR Analysis",
    ])

    with tab1:
        fig = plot_bollinger_bands(bb_df)
        fig.update_layout(
            height=500,
            title=dict(text=f"{symbol} — Bollinger Bands (20-day, 2σ)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Bollinger Band Width
        if "bb_upper" in bb_df.columns and "bb_lower" in bb_df.columns:
            bb_width = (bb_df["bb_upper"] - bb_df["bb_lower"]) / bb_df["bb_middle"] * 100
            current_width = bb_width.iloc[-1]
            avg_width = bb_width.mean()

            col1, col2 = st.columns(2)
            with col1:
                render_metric_card("Current BB Width", f"{current_width:.2f}%")
            with col2:
                squeeze = "Yes — Breakout Likely" if current_width < avg_width * 0.7 else "No"
                render_metric_card("Bollinger Squeeze?", squeeze)

    with tab2:
        # Rolling Volatility Comparison
        fig = go.Figure()

        for window, color in [(7, THEME["accent"]), (30, THEME["accent_secondary"]), (90, THEME["warning"])]:
            if len(df) > window:
                vol = df["price"].pct_change().rolling(window).std() * np.sqrt(365) * 100
                fig.add_trace(go.Scatter(
                    x=df["date"], y=vol,
                    mode="lines", name=f"{window}-Day Volatility",
                    line=dict(color=color, width=2),
                ))

        fig.update_layout(
            title=dict(text=f"{symbol} — Rolling Annualized Volatility"),
            yaxis_title="Volatility (%)",
            height=450,
            template="plotly_dark",
            paper_bgcolor=THEME["bg_primary"],
            plot_bgcolor=THEME["bg_secondary"],
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = plot_drawdown(df)
        fig.update_layout(height=450, title=dict(text=f"{symbol} — Drawdown from ATH"))
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown stats
        if "price" in df.columns:
            returns = df["price"].pct_change().dropna()
            col1, col2, col3 = st.columns(3)
            with col1:
                render_metric_card("Avg Daily Return", f"{returns.mean() * 100:.3f}%")
            with col2:
                render_metric_card("Worst Daily Loss", f"{returns.min() * 100:.2f}%")
            with col3:
                render_metric_card("Best Daily Gain", f"{returns.max() * 100:.2f}%")

    with tab4:
        atr_df = compute_atr(df)
        if "atr" in atr_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=atr_df["date"], y=atr_df["atr"],
                mode="lines", name="ATR (14-day)",
                line=dict(color=THEME["accent"], width=2),
                fill="tozeroy",
                fillcolor="rgba(0, 212, 170, 0.08)",
            ))
            fig.update_layout(
                title=dict(text=f"{symbol} — Average True Range (14-day)"),
                yaxis_title="ATR (USD)",
                height=400,
                template="plotly_dark",
                paper_bgcolor=THEME["bg_primary"],
                plot_bgcolor=THEME["bg_secondary"],
                hovermode="x unified",
            )
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
            st.plotly_chart(fig, use_container_width=True)

            current_atr = atr_df["atr"].iloc[-1]
            avg_atr = atr_df["atr"].mean()
            render_metric_card(
                "Current ATR vs Average",
                f"${current_atr:,.0f} vs ${avg_atr:,.0f}",
                delta=f"{'Above' if current_atr > avg_atr else 'Below'} average",
                delta_positive=current_atr <= avg_atr,
            )

    st.markdown("---")

    # ── Actionable Risk Summary ──
    st.markdown(f"""
    <div style="background: {THEME['bg_card']}; border-radius: 12px;
                padding: 1.2rem 1.5rem; border-left: 3px solid {risk_color};">
        <div style="color: {THEME['text_secondary']}; font-size: 0.85rem; margin-bottom: 0.3rem;">
            💡 Risk Summary for {symbol}
        </div>
        <div style="color: {THEME['text_primary']}; font-size: 0.95rem; line-height: 1.5;">
            <strong>{symbol}</strong> currently exhibits <strong style="color: {risk_color};">{risk_label.lower()}</strong>
            with {ann_vol:.1f}% annualized volatility.
            The Sharpe ratio of <strong>{sharpe:.2f}</strong> indicates
            {"strong" if sharpe > 1 else "moderate" if sharpe > 0.5 else "weak"} risk-adjusted returns.
            Maximum drawdown of <strong>{max_dd:.1f}%</strong>
            suggests {"significant" if max_dd < -40 else "moderate" if max_dd < -20 else "contained"} downside risk.
            {"Consider position sizing and stop-losses." if ann_vol > 60 else "Risk levels are manageable for diversified portfolios."}
        </div>
    </div>
    """, unsafe_allow_html=True)
