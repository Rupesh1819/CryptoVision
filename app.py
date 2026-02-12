"""
CryptoVision — Cryptocurrency Time Series Analysis Dashboard
Main Streamlit Application Entry Point
"""

import streamlit as st
import sys
import os

# ── Path Setup ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import THEME
from dashboard.components import inject_custom_css, sidebar_controls
from dashboard.pages.overview import render_overview_page
from dashboard.pages.historical import render_historical_page
from dashboard.pages.forecast import render_forecast_page
from dashboard.pages.volatility_page import render_volatility_page
from dashboard.pages.sentiment_page import render_sentiment_page


def main():
    # ── Page Config ──
    st.set_page_config(
        page_title="CryptoVision — Time Series Analysis",
        page_icon="🪙",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Inject Custom CSS ──
    inject_custom_css()

    # ── Sidebar Controls ──
    selected_coin, days = sidebar_controls()

    # ── Page Navigation ──
    with st.sidebar:
        st.markdown("---")

        # Navigation with icons
        st.markdown(f"""
        <div style="color: {THEME['text_secondary']}; font-size: 0.75rem;
                    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
            Navigation
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            ["🌐 Overview", "📜 Historical & EDA", "🔮 Forecasting",
             "⚡ Volatility & Risk", "💬 Sentiment"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Model availability status
        st.markdown(f"""
        <div style="color: {THEME['text_secondary']}; font-size: 0.75rem;
                    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
            Model Status
        </div>
        """, unsafe_allow_html=True)

        # Check model availability
        _show_model_status()

        st.markdown("---")

        # Python info
        python_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        st.markdown(f"""
        <div style="color: {THEME['text_secondary']}; font-size: 0.75rem; padding: 0.5rem;">
        <strong>CryptoVision v1.0</strong><br>
        Python {python_ver}<br>
        <em>Real-time analytics, forecasting &amp; sentiment intelligence for crypto markets.</em>
        </div>
        """, unsafe_allow_html=True)

    # ── Header ──
    st.markdown(f"""
    <div style="text-align: center; padding: 0.8rem 0 0.3rem 0;">
        <h1 style="
            background: linear-gradient(135deg, {THEME['accent']} 0%, {THEME['accent_secondary']} 50%, #FF6B6B 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            letter-spacing: -0.5px;
        ">🪙 CryptoVision</h1>
        <p style="color: {THEME['text_secondary']}; font-size: 1.05rem; margin-top: 0;">
            Time Series Analysis · AI Forecasting · Sentiment Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Route Pages ──
    try:
        if page == "🌐 Overview":
            render_overview_page(selected_coin, days)
        elif page == "📜 Historical & EDA":
            render_historical_page(selected_coin, days)
        elif page == "🔮 Forecasting":
            render_forecast_page(selected_coin, days)
        elif page == "⚡ Volatility & Risk":
            render_volatility_page(selected_coin, days)
        elif page == "💬 Sentiment":
            render_sentiment_page(selected_coin, days)
    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")
        with st.expander("Error Details"):
            import traceback
            st.code(traceback.format_exc())

    # ── Footer ──
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 1.2rem; color: {THEME['text_secondary']}; font-size: 0.8rem;">
        <div style="margin-bottom: 0.5rem;">
            <span style="background: linear-gradient(135deg, {THEME['accent']}, {THEME['accent_secondary']});
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                         font-weight: 700; font-size: 1rem;">CryptoVision</span>
        </div>
        Built with Streamlit · Plotly · Statsmodels · VADER NLP<br>
        Data from CoinGecko &amp; Yahoo Finance<br>
        <em style="color: {THEME['warning']};">⚠️ For educational and research purposes only. Not financial advice.</em>
    </div>
    """, unsafe_allow_html=True)


def _show_model_status():
    """Show availability of forecasting models in the sidebar."""
    # ARIMA & SARIMA — always available since statsmodels is a core dependency
    st.markdown(f'<div style="font-size: 0.85rem;">✅ ARIMA</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size: 0.85rem;">✅ SARIMA</div>', unsafe_allow_html=True)

    # Prophet
    try:
        from models.prophet_model import is_available as prophet_ok
        if prophet_ok():
            st.markdown(f'<div style="font-size: 0.85rem;">✅ Prophet</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-size: 0.85rem; color: {THEME["text_secondary"]};">⬜ Prophet <em>(not installed)</em></div>', unsafe_allow_html=True)
    except Exception:
        st.markdown(f'<div style="font-size: 0.85rem; color: {THEME["text_secondary"]};">⬜ Prophet <em>(not installed)</em></div>', unsafe_allow_html=True)

    # LSTM
    try:
        from models.lstm_model import is_available as lstm_ok
        if lstm_ok():
            st.markdown(f'<div style="font-size: 0.85rem;">✅ LSTM</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-size: 0.85rem; color: {THEME["text_secondary"]};">⬜ LSTM <em>(needs TensorFlow)</em></div>', unsafe_allow_html=True)
    except Exception:
        st.markdown(f'<div style="font-size: 0.85rem; color: {THEME["text_secondary"]};">⬜ LSTM <em>(needs TensorFlow)</em></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
