"""
Reusable Streamlit Dashboard Components
Custom-styled metric cards, chart containers, sidebar controls, and premium CSS.
"""

import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import THEME, COINS


def inject_custom_css():
    """Inject premium dark theme CSS into the Streamlit app."""
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <style>
        /* ── Global Theme ── */
        .stApp {{
            background: linear-gradient(180deg, {THEME['bg_primary']} 0%, #0a0d14 100%);
            color: {THEME['text_primary']};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {THEME['bg_secondary']} 0%, #12151f 100%);
            border-right: 1px solid rgba(255,255,255,0.04);
        }}

        section[data-testid="stSidebar"] .stMarkdown h1 {{
            background: linear-gradient(135deg, {THEME['accent']} 0%, {THEME['accent_secondary']} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.6rem;
            font-weight: 800;
        }}

        /* ── Metric Cards ── */
        .metric-card {{
            background: linear-gradient(145deg, {THEME['bg_card']}ee, {THEME['bg_card']}aa);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 0.8rem;
            backdrop-filter: blur(12px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }}
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, {THEME['accent']}00, {THEME['accent']}60, {THEME['accent']}00);
            opacity: 0;
            transition: opacity 0.3s;
        }}
        .metric-card:hover {{
            transform: translateY(-3px);
            border-color: {THEME['accent']}30;
            box-shadow: 0 8px 32px rgba(0, 212, 170, 0.08);
        }}
        .metric-card:hover::before {{
            opacity: 1;
        }}
        .metric-card .label {{
            color: {THEME['text_secondary']};
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }}
        .metric-card .value {{
            color: {THEME['text_primary']};
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1.2;
            letter-spacing: -0.5px;
        }}
        .metric-card .delta {{
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 0.4rem;
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }}
        .delta-positive {{ color: {THEME['success']}; }}
        .delta-negative {{ color: {THEME['danger']}; }}

        /* ── Section Headers ── */
        .section-header {{
            color: {THEME['text_primary']};
            font-size: 1.25rem;
            font-weight: 700;
            margin: 1.8rem 0 1rem 0;
            padding-bottom: 0.6rem;
            border-bottom: 2px solid transparent;
            border-image: linear-gradient(90deg, {THEME['accent']}60, {THEME['accent_secondary']}30, transparent) 1;
            letter-spacing: -0.3px;
        }}

        /* ── Badges ── */
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.2rem 0.7rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-positive {{ background: {THEME['success']}18; color: {THEME['success']}; border: 1px solid {THEME['success']}30; }}
        .badge-negative {{ background: {THEME['danger']}18; color: {THEME['danger']}; border: 1px solid {THEME['danger']}30; }}
        .badge-neutral {{ background: {THEME['warning']}18; color: {THEME['warning']}; border: 1px solid {THEME['warning']}30; }}

        /* ── News Cards ── */
        .news-card {{
            background: {THEME['bg_card']};
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.6rem;
            transition: border-color 0.2s;
        }}
        .news-card:hover {{
            border-color: rgba(255,255,255,0.1);
        }}
        .news-card .news-title {{
            color: {THEME['text_primary']};
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 0.3rem;
            line-height: 1.4;
        }}
        .news-card .news-meta {{
            color: {THEME['text_secondary']};
            font-size: 0.78rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}

        /* ── Risk Indicator ── */
        .risk-indicator {{
            text-align: center;
            padding: 2rem;
            border-radius: 20px;
            background: linear-gradient(145deg, {THEME['bg_card']}ee, {THEME['bg_card']}88);
            border: 1px solid rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
        }}
        .risk-value {{
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: 4px;
        }}

        /* ── Buttons ── */
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {THEME['accent']} 0%, {THEME['accent_secondary']} 100%);
            border: none;
            border-radius: 12px;
            font-weight: 700;
            letter-spacing: 0.5px;
            transition: all 0.3s;
        }}
        .stButton > button[kind="primary"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 212, 170, 0.25);
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.3rem;
            background: {THEME['bg_card']}80;
            border-radius: 12px;
            padding: 0.3rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 10px;
            padding: 0.5rem 1.2rem;
            color: {THEME['text_secondary']};
            font-weight: 500;
        }}
        .stTabs [aria-selected="true"] {{
            background: {THEME['accent']}18;
            color: {THEME['accent']};
        }}

        /* ── Expander ── */
        .streamlit-expanderHeader {{
            background: {THEME['bg_card']};
            border-radius: 10px;
        }}

        /* ── Selectbox / Inputs ── */
        .stSelectbox > div > div {{
            background: {THEME['bg_card']};
            border-color: rgba(255,255,255,0.08);
            border-radius: 10px;
        }}

        /* ── Hide default streamlit elements ── */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        .stDeployButton {{ display: none; }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: {THEME['bg_primary']}; }}
        ::-webkit-scrollbar-thumb {{ background: {THEME['bg_card']}; border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {THEME['accent']}40; }}

        /* ── Dataframe ── */
        .stDataFrame {{
            border-radius: 12px;
            overflow: hidden;
        }}
    </style>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, delta: str = None,
                        delta_positive: bool = True):
    """Render a glassmorphic metric card."""
    delta_html = ""
    if delta is not None:
        cls = "delta-positive" if delta_positive else "delta-negative"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div class="delta {cls}">{arrow} {delta}</div>'

    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, icon: str = "📊"):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{icon} {title}</div>',
                unsafe_allow_html=True)


def render_sentiment_badge(label: str) -> str:
    """Return HTML for a sentiment badge."""
    cls_map = {"Positive": "badge-positive", "Negative": "badge-negative", "Neutral": "badge-neutral"}
    icon_map = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
    cls = cls_map.get(label, "badge-neutral")
    icon = icon_map.get(label, "⚪")
    return f'<span class="badge {cls}">{icon} {label}</span>'


def render_news_card(title: str, source: str, published: str,
                      sentiment_label: str, compound: float):
    """Render a styled news article card."""
    badge = render_sentiment_badge(sentiment_label)

    # Color the compound score
    score_color = THEME['success'] if compound > 0.05 else THEME['danger'] if compound < -0.05 else THEME['warning']

    st.markdown(f"""
    <div class="news-card">
        <div class="news-title">{title}</div>
        <div class="news-meta">
            <span>📰 {source}</span>
            <span>·</span>
            <span>🕐 {published}</span>
            <span>·</span>
            {badge}
            <span>·</span>
            <span style="color: {score_color}; font-weight: 600;">
                Score: {compound:.3f}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def sidebar_controls():
    """Render sidebar with common controls. Returns selected coin_id and date range."""
    with st.sidebar:
        st.markdown("# 🪙 CryptoVision")
        st.markdown("---")

        # Coin selector with emoji indicators
        coin_options = {}
        for coin_id, info in COINS.items():
            coin_options[f"{info['symbol']} — {coin_id.title()}"] = coin_id

        selected_label = st.selectbox(
            "🔍 Select Cryptocurrency",
            options=list(coin_options.keys()),
            index=0,
        )
        selected_coin = coin_options[selected_label]

        st.markdown("")

        # Date range
        days_options = {
            "📅 1 Month": 30,
            "📅 3 Months": 90,
            "📅 6 Months": 180,
            "📅 1 Year": 365,
            "📅 2 Years": 730,
        }
        selected_range = st.selectbox(
            "📆 Historical Range",
            options=list(days_options.keys()),
            index=4,
        )
        days = days_options[selected_range]

        return selected_coin, days
