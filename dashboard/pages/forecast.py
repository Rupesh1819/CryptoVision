"""
Forecast Page — Train and compare ARIMA, SARIMA, Prophet, LSTM models.
Shows availability status and gracefully handles missing dependencies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import COINS, THEME, ARIMA_CONFIG, SARIMA_CONFIG, PROPHET_CONFIG, LSTM_CONFIG
from data.collector import fetch_historical_data
from data.preprocessor import preprocess_pipeline, prepare_for_modeling
from models.arima_model import run_arima_pipeline
from models.sarima_model import run_sarima_pipeline
from models.evaluator import (
    evaluate_model, compare_models,
    plot_model_comparison, plot_predictions_overlay,
)
from dashboard.components import render_metric_card, render_section_header


def _check_model_availability():
    """Check which models are available."""
    available = {"ARIMA": True, "SARIMA": True}

    try:
        from models.prophet_model import is_available
        available["Prophet"] = is_available()
    except Exception:
        available["Prophet"] = False

    try:
        from models.lstm_model import is_available
        available["LSTM"] = is_available()
    except Exception:
        available["LSTM"] = False

    return available


def render_forecast_page(selected_coin: str, days: int):
    """Render the Forecasting page."""

    render_section_header("Price Forecasting", "🔮")
    symbol = COINS.get(selected_coin, {}).get("symbol", selected_coin.upper())

    # ── Fetch and prepare data ──
    with st.spinner("Preparing data..."):
        raw_df = fetch_historical_data(selected_coin, days)
        df = preprocess_pipeline(raw_df)
        splits = prepare_for_modeling(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Data Points", f"{len(df)} days")
    with col2:
        render_metric_card("Training Set", f"{len(splits['train'])} days")
    with col3:
        render_metric_card("Test Set", f"{len(splits['test'])} days")
    with col4:
        render_metric_card("Split Date", splits['split_date'].strftime('%Y-%m-%d'))

    st.markdown("---")

    # ── Check model availability ──
    availability = _check_model_availability()

    # Show availability warning
    unavailable = [m for m, ok in availability.items() if not ok]
    if unavailable:
        st.warning(
            f"⚠️ **{', '.join(unavailable)}** model(s) unavailable on Python {sys.version_info.major}.{sys.version_info.minor}. "
            f"These require Python 3.10-3.12 with `pip install prophet tensorflow`."
        )

    # ── Model Selection ──
    all_models = list(availability.keys())
    available_models = [m for m, ok in availability.items() if ok]

    model_choices = st.multiselect(
        "Select Models to Train & Compare",
        all_models,
        default=available_models[:2],  # default to ARIMA + SARIMA
        help="Select one or more models. Unavailable models will show an install guide.",
    )

    if not model_choices:
        st.info("👆 Please select at least one model above to begin forecasting.")
        return

    # Filter out unavailable models
    valid_choices = []
    for m in model_choices:
        if not availability.get(m, False):
            st.error(f"❌ **{m}** is not available. Install required dependencies first.")
        else:
            valid_choices.append(m)

    if not valid_choices:
        return

    col1, col2 = st.columns(2)
    with col1:
        forecast_days = st.slider("Forecast Horizon (days)", 7, 90,
                                   ARIMA_CONFIG["forecast_days"])
    with col2:
        models_str = ", ".join(valid_choices)
        st.markdown(
            f'<div style="padding: 0.8rem; background: {THEME["bg_card"]};'
            f' border-radius: 12px; margin-top: 0.3rem;">'
            f'<div style="color: {THEME["text_secondary"]}; font-size: 0.85rem;">'
            f'Selected: <strong style="color: {THEME["accent"]};">{models_str}</strong><br>'
            f'Coin: <strong style="color: {THEME["accent"]};">{symbol}</strong> · '
            f'Horizon: <strong style="color: {THEME["accent"]};">{forecast_days}d</strong>'
            f'</div></div>', unsafe_allow_html=True)

    st.markdown("")

    if st.button("🚀 Train & Forecast", type="primary", use_container_width=True):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(valid_choices)

        for idx, model_name in enumerate(valid_choices):
            pct = int(((idx) / total) * 100)
            progress_bar.progress(pct, text=f"Training {model_name}...")

            # ── ARIMA ──
            if model_name == "ARIMA":
                status_text.markdown("⏳ **Training ARIMA model...**")
                try:
                    arima_result = run_arima_pipeline(
                        splits["train"], splits["test"], "price"
                    )
                    results.append(arima_result)
                    st.success(f"✅ **ARIMA** trained — Order: `{arima_result['order']}`, AIC: `{arima_result.get('aic', 'N/A'):.1f}`")
                except Exception as e:
                    st.error(f"❌ ARIMA failed: {e}")

            # ── SARIMA ──
            elif model_name == "SARIMA":
                status_text.markdown("⏳ **Training SARIMA model** (seasonal ARIMA — may take a moment)...")
                try:
                    sarima_result = run_sarima_pipeline(
                        splits["train"], splits["test"], "price"
                    )
                    results.append(sarima_result)
                    st.success(
                        f"✅ **SARIMA** trained — Order: `{sarima_result['order']}`, "
                        f"Seasonal: `{sarima_result['seasonal_order']}`, "
                        f"AIC: `{sarima_result.get('aic', 'N/A'):.1f}`"
                    )
                except Exception as e:
                    st.error(f"❌ SARIMA failed: {e}")

            # ── Prophet ──
            elif model_name == "Prophet":
                status_text.markdown("⏳ **Training Prophet model...**")
                try:
                    from models.prophet_model import run_prophet_pipeline
                    prophet_result = run_prophet_pipeline(
                        splits["train"], splits["test"], "price"
                    )
                    results.append(prophet_result)
                    st.success("✅ **Prophet** trained successfully")
                except ImportError as e:
                    st.error(f"❌ Prophet not available: {e}")
                except Exception as e:
                    st.error(f"❌ Prophet failed: {e}")

            # ── LSTM ──
            elif model_name == "LSTM":
                status_text.markdown("⏳ **Training LSTM model** (this may take 1-2 minutes)...")
                try:
                    from models.lstm_model import run_lstm_pipeline
                    lstm_result = run_lstm_pipeline(
                        splits["train"], splits["test"], "price"
                    )
                    results.append(lstm_result)
                    st.success("✅ **LSTM** trained successfully")
                except ImportError as e:
                    st.error(f"❌ LSTM not available: {e}")
                except Exception as e:
                    st.error(f"❌ LSTM failed: {e}")

        progress_bar.progress(100, text="All models trained!")
        status_text.empty()

        if not results:
            st.error("❌ No models trained successfully. Check the errors above.")
            return

        st.markdown("---")

        # ── Predictions Overlay ──
        render_section_header("Actual vs Predicted", "📉")
        fig = plot_predictions_overlay(results)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Model Comparison Table ──
        render_section_header("Model Comparison", "🏆")
        comparison = compare_models(results)

        # Highlight best model
        best = comparison.iloc[0]
        best_html = (
            f'<div style="background: linear-gradient(135deg, rgba(0,212,170,0.08), rgba(124,58,237,0.08));'
            f' border: 1px solid rgba(0,212,170,0.25); border-radius: 16px;'
            f' padding: 1.2rem; text-align: center; margin-bottom: 1rem;">'
            f'<div style="color: {THEME["text_secondary"]}; font-size: 0.85rem; margin-bottom: 0.3rem;">'
            f'🏆 Best Performing Model</div>'
            f'<div style="color: {THEME["accent"]}; font-size: 1.8rem; font-weight: 800;">'
            f'{best["model_name"]}</div>'
            f'<div style="color: {THEME["text_secondary"]}; font-size: 0.9rem; margin-top: 0.3rem;">'
            f'RMSE: <strong>${best["rmse"]:,.2f}</strong> · '
            f'MAE: <strong>${best["mae"]:,.2f}</strong> · '
            f'MAPE: <strong>{best["mape"]:.2f}%</strong> · '
            f'R²: <strong>{best["r2"]:.4f}</strong></div></div>'
        )
        st.markdown(best_html, unsafe_allow_html=True)

        # Display metrics table
        st.dataframe(
            comparison.style.format({
                "mae": "${:,.2f}",
                "rmse": "${:,.2f}",
                "mape": "{:.2f}%",
                "r2": "{:.4f}",
            }).highlight_min(subset=["mae", "rmse", "mape"], color="#10B98130")
            .highlight_max(subset=["r2"], color="#10B98130"),
            use_container_width=True,
        )

        # ── Comparison Bar Chart ──
        if len(results) > 1:
            fig = plot_model_comparison(comparison)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Future Forecast ──
        render_section_header(f"{forecast_days}-Day Future Forecast", "🔭")

        # Use the best model for future forecast
        best_result = results[0]  # Already sorted by RMSE
        _render_future_forecast(best_result, df, forecast_days, symbol)

        # Store results in session
        st.session_state["forecast_results"] = results
        st.session_state["comparison"] = comparison


def _render_future_forecast(result: dict, df: pd.DataFrame,
                             forecast_days: int, symbol: str):
    """Generate and display future price forecast."""
    model_name = result["model_name"]

    try:
        if model_name == "ARIMA":
            from models.arima_model import forecast_arima
            forecast = forecast_arima(result["trained_model"], steps=forecast_days)
            future_prices = forecast["forecast"].values
            lower = forecast["lower_ci"].values
            upper = forecast["upper_ci"].values

        elif model_name == "SARIMA":
            from models.sarima_model import forecast_sarima
            forecast = forecast_sarima(result["trained_model"], steps=forecast_days)
            future_prices = forecast["forecast"].values
            lower = forecast["lower_ci"].values
            upper = forecast["upper_ci"].values

        elif model_name == "Prophet":
            from models.prophet_model import forecast_prophet
            forecast = forecast_prophet(result["trained_model"], periods=forecast_days)
            future_prices = forecast["yhat"].tail(forecast_days).values
            lower = forecast["yhat_lower"].tail(forecast_days).values
            upper = forecast["yhat_upper"].tail(forecast_days).values

        elif model_name == "LSTM":
            from models.lstm_model import forecast_lstm
            future_prices = forecast_lstm(result["trained_model"], steps=forecast_days)
            # Estimate CI for LSTM
            std = np.std(result["actuals"] - result["predictions"])
            lower = future_prices - 1.96 * std
            upper = future_prices + 1.96 * std

        else:
            st.warning("Future forecast not available for this model.")
            return

        # Create date range for future
        last_date = pd.to_datetime(df["date"].iloc[-1])
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                      periods=forecast_days, freq="D")

        # Plot
        fig = go.Figure()

        # Historical prices (last 90 days)
        recent = df.tail(90)
        fig.add_trace(go.Scatter(
            x=recent["date"], y=recent["price"],
            mode="lines", name="Historical",
            line=dict(color=THEME["accent"], width=2.5),
        ))

        # Connection line (last historical to first forecast)
        fig.add_trace(go.Scatter(
            x=[recent["date"].iloc[-1], future_dates[0]],
            y=[recent["price"].iloc[-1], future_prices[0]],
            mode="lines", name="_connection",
            line=dict(color=THEME["accent_secondary"], width=2, dash="dot"),
            showlegend=False,
        ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_prices,
            mode="lines+markers", name=f"{model_name} Forecast",
            line=dict(color=THEME["accent_secondary"], width=2.5, dash="dash"),
            marker=dict(size=3),
        ))

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself", name="95% Confidence Interval",
            fillcolor="rgba(124, 58, 237, 0.12)",
            line=dict(color="rgba(0,0,0,0)"),
        ))

        # Vertical split line (use add_shape to avoid Plotly annotation_text bug with dates)
        fig.add_shape(
            type="line",
            x0=last_date, x1=last_date,
            y0=0, y1=1, yref="paper",
            line=dict(color=THEME["text_secondary"], width=1, dash="dash"),
            opacity=0.4,
        )
        fig.add_annotation(
            x=last_date, y=1, yref="paper",
            text="Forecast Start",
            showarrow=False,
            font=dict(size=11, color=THEME["text_secondary"]),
            yshift=10,
        )

        fig.update_layout(
            title=dict(
                text=f"{symbol} — {forecast_days}-Day Price Forecast ({model_name})",
                font=dict(size=16),
            ),
            yaxis_title="Price (USD)",
            template="plotly_dark",
            paper_bgcolor=THEME["bg_primary"],
            plot_bgcolor=THEME["bg_secondary"],
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")

        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        current_price = df["price"].iloc[-1]
        pct_change = ((future_prices[-1] / current_price) - 1) * 100

        with col1:
            render_metric_card("Current Price", f"${current_price:,.2f}")
        with col2:
            render_metric_card("Forecast End Price",
                                f"${future_prices[-1]:,.2f}",
                                delta=f"{abs(pct_change):.2f}%",
                                delta_positive=pct_change >= 0)
        with col3:
            render_metric_card("Forecast Range",
                                f"${lower[-1]:,.0f} — ${upper[-1]:,.0f}")
        with col4:
            direction = "📈 Bullish" if pct_change > 2 else "📉 Bearish" if pct_change < -2 else "➡️ Sideways"
            render_metric_card("Signal", direction)

        # Actionable insight
        if pct_change > 0:
            insight_detail = "Consider monitoring momentum indicators for entry points."
        else:
            insight_detail = "Consider risk management strategies and stop-loss levels."

        insight_html = (
            f'<div style="background: {THEME["bg_card"]}; border-radius: 12px;'
            f' padding: 1rem 1.5rem; border-left: 3px solid {THEME["accent"]};">'
            f'<div style="color: {THEME["text_secondary"]}; font-size: 0.85rem; margin-bottom: 0.3rem;">'
            f'💡 Actionable Insight</div>'
            f'<div style="color: {THEME["text_primary"]}; font-size: 0.95rem;">'
            f'Based on the {model_name} model, <strong>{symbol}</strong> is projected to '
            f'{"increase" if pct_change > 0 else "decrease"} by <strong>{abs(pct_change):.2f}%</strong> '
            f'over the next {forecast_days} days (${current_price:,.2f} → ${future_prices[-1]:,.2f}). '
            f'The 95% confidence interval ranges from <strong>${lower[-1]:,.0f}</strong> to '
            f'<strong>${upper[-1]:,.0f}</strong>. {insight_detail}'
            f'</div></div>'
        )
        st.markdown(insight_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Future forecast error: {e}")
        with st.expander("Details"):
            import traceback
            st.code(traceback.format_exc())
