"""
Historical Data Page — Interactive EDA charts, trends, seasonality, correlations.
"""

import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import COINS
from data.collector import fetch_historical_data, fetch_multi_coin_data
from data.preprocessor import preprocess_pipeline, detect_outliers_iqr
from analysis.eda import (
    plot_price_trends, plot_seasonal_decomposition,
    plot_returns_distribution, plot_correlation_heatmap,
    plot_candlestick, compute_summary_stats,
)
from dashboard.components import render_metric_card, render_section_header


def render_historical_page(selected_coin: str, days: int):
    """Render the Historical Data & EDA page."""

    render_section_header("Historical Analysis", "📜")

    # ── Fetch & preprocess data ──
    with st.spinner("Loading historical data..."):
        raw_df = fetch_historical_data(selected_coin, days)
        df = preprocess_pipeline(raw_df)

    symbol = COINS.get(selected_coin, {}).get("symbol", selected_coin.upper())

    # ── Summary Statistics ──
    stats = compute_summary_stats(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Current Price", f"${stats['current_price']:,.2f}")
    with col2:
        render_metric_card(
            "Total Return", f"{stats['total_return']:+.1f}%",
            delta_positive=stats['total_return'] >= 0,
        )
    with col3:
        render_metric_card("Annual Volatility", f"{stats['annualized_volatility']:.1f}%")
    with col4:
        render_metric_card("Sharpe Ratio", f"{stats['sharpe_ratio']:.3f}")

    st.markdown("---")

    # ── Tabs for different charts ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price Trends", "🕯️ Candlestick", "🔄 Seasonality",
        "📊 Returns", "🔗 Correlations"
    ])

    with tab1:
        fig = plot_price_trends(df, selected_coin)
        st.plotly_chart(fig, use_container_width=True)

        # Outlier detection info
        outlier_df = detect_outliers_iqr(df, "price")
        n_outliers = outlier_df["is_outlier_iqr"].sum()
        if n_outliers > 0:
            st.info(f"🔍 Detected **{n_outliers}** price outliers (IQR method) in the selected period.")

    with tab2:
        fig = plot_candlestick(df, selected_coin)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        period = st.slider("Decomposition Period (days)", 7, 90, 30)
        fig = plot_seasonal_decomposition(df, period=period)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = plot_returns_distribution(df)
        st.plotly_chart(fig, use_container_width=True)

        # Additional stats table
        st.markdown("**Distribution Statistics**")
        stats_detail = {
            "Mean Daily Return": f"{stats['avg_daily_return']:.4f}%",
            "Daily Volatility": f"{stats['daily_volatility']:.4f}%",
            "Skewness": f"{stats['skewness']:.4f}",
            "Kurtosis": f"{stats['kurtosis']:.4f}",
            "Max Drawdown": f"{stats['max_drawdown']:.2f}%",
        }
        for k, v in stats_detail.items():
            st.text(f"  {k}: {v}")

    with tab5:
        st.markdown("Loading multi-coin data for correlation analysis...")
        with st.spinner("Fetching data for all coins..."):
            multi_data = fetch_multi_coin_data(days=days)
        fig = plot_correlation_heatmap(multi_data)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Correlation based on daily price returns. Values close to 1 indicate strong positive correlation.")
