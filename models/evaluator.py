"""
Model Evaluation Module
Computes MAE, RMSE, MAPE, and provides comparison utilities.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import THEME


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    actual, predicted = np.array(actual), np.array(predicted)
    return float(np.mean(np.abs(actual - predicted)))


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    actual, predicted = np.array(actual), np.array(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    actual, predicted = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    # Avoid division by zero
    mask = actual != 0
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def compute_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    actual, predicted = np.array(actual), np.array(predicted)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def evaluate_model(result: dict) -> dict:
    """
    Evaluate a single model's performance.

    Args:
        result: Dict with 'predictions', 'actuals', 'model_name'

    Returns:
        Dict with all metrics
    """
    actual = result["actuals"]
    predicted = result["predictions"]

    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]

    return {
        "model_name": result["model_name"],
        "mae": compute_mae(actual, predicted),
        "rmse": compute_rmse(actual, predicted),
        "mape": compute_mape(actual, predicted),
        "r2": compute_r2(actual, predicted),
        "n_predictions": min_len,
    }


def compare_models(results: list) -> pd.DataFrame:
    """
    Compare multiple models' performance.

    Args:
        results: List of dicts from model pipelines (each with 'predictions', 'actuals', 'model_name')

    Returns:
        DataFrame with metrics for each model, sorted by RMSE
    """
    evaluations = []
    for result in results:
        metrics = evaluate_model(result)
        evaluations.append(metrics)

    df = pd.DataFrame(evaluations)
    df = df.sort_values("rmse").reset_index(drop=True)
    df.index = df.index + 1  # 1-indexed ranking

    return df


def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Create a bar chart comparing model metrics."""
    metrics = ["mae", "rmse", "mape"]
    metric_labels = ["MAE ($)", "RMSE ($)", "MAPE (%)"]
    colors = [THEME["accent"], THEME["accent_secondary"], THEME["warning"]]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=metric_labels,
        horizontal_spacing=0.08,
    )

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors), 1):
        fig.add_trace(go.Bar(
            x=comparison_df["model_name"],
            y=comparison_df[metric],
            name=label,
            marker_color=color,
            text=comparison_df[metric].round(2),
            textposition="auto",
        ), row=1, col=i)

    fig.update_layout(
        title="Model Performance Comparison",
        template="plotly_dark",
        paper_bgcolor=THEME["bg_primary"],
        plot_bgcolor=THEME["bg_secondary"],
        height=400,
        showlegend=False,
    )

    return fig


def plot_predictions_overlay(results: list) -> go.Figure:
    """
    Overlay actual vs predicted for all models on a single chart.
    """
    fig = go.Figure()

    colors = [THEME["accent"], THEME["accent_secondary"], THEME["warning"], THEME["success"]]

    for i, result in enumerate(results):
        dates = result.get("dates")
        actuals = result["actuals"]
        preds = result["predictions"]
        min_len = min(len(actuals), len(preds))
        actuals = actuals[:min_len]
        preds = preds[:min_len]

        x_axis = dates[:min_len] if dates is not None else list(range(min_len))

        # Actual (only once)
        if i == 0:
            fig.add_trace(go.Scatter(
                x=x_axis, y=actuals,
                mode="lines", name="Actual",
                line=dict(color=THEME["text_primary"], width=2),
            ))

        # Model prediction
        fig.add_trace(go.Scatter(
            x=x_axis, y=preds,
            mode="lines", name=f"{result['model_name']} Prediction",
            line=dict(color=colors[i % len(colors)], width=1.5, dash="dot"),
        ))

        # Confidence interval if available
        if "lower_ci" in result and "upper_ci" in result:
            lower = result["lower_ci"][:min_len]
            upper = result["upper_ci"][:min_len]
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_axis, x_axis[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself", name=f"{result['model_name']} CI",
                fillcolor=f"rgba({_get_rgb(colors[i % len(colors)])}, 0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
            ))

    fig.update_layout(
        title="Actual vs Predicted — All Models",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        paper_bgcolor=THEME["bg_primary"],
        plot_bgcolor=THEME["bg_secondary"],
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def _get_rgb(hex_color: str) -> str:
    """Convert hex to RGB string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"{r}, {g}, {b}"
