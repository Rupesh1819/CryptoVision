"""
Facebook Prophet Forecasting Model
Implements Prophet with custom seasonalities and optional regressors.
Gracefully handles missing Prophet installation.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROPHET_CONFIG

# Check Prophet availability
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if Prophet model can run."""
    return PROPHET_AVAILABLE


def train_prophet(df: pd.DataFrame, target_col: str = "price",
                   regressors: list = None) -> dict:
    """
    Train Facebook Prophet model.

    Args:
        df: Training DataFrame with 'date' and target column
        target_col: Column to forecast
        regressors: Optional list of additional regressor column names

    Returns:
        Dict with model, training data, and config
    """
    if not PROPHET_AVAILABLE:
        raise ImportError(
            "Prophet is not installed.\n"
            "Install with: pip install prophet\n"
            "Note: Prophet may require Python 3.10-3.12 and C++ build tools."
        )

    from prophet import Prophet

    # Prophet requires columns named 'ds' and 'y'
    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(df["date"]),
        "y": df[target_col].values,
    })

    model = Prophet(
        daily_seasonality=PROPHET_CONFIG["daily_seasonality"],
        weekly_seasonality=PROPHET_CONFIG["weekly_seasonality"],
        yearly_seasonality=PROPHET_CONFIG["yearly_seasonality"],
        changepoint_prior_scale=PROPHET_CONFIG["changepoint_prior_scale"],
    )

    # Add external regressors
    if regressors:
        for reg in regressors:
            if reg in df.columns:
                prophet_df[reg] = df[reg].values
                model.add_regressor(reg)

    model.fit(prophet_df)

    return {
        "model": model,
        "train_df": prophet_df,
        "regressors": regressors or [],
    }


def forecast_prophet(trained: dict, periods: int = None,
                      future_regressors: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generate forecast from trained Prophet model.

    Returns:
        DataFrame with columns: ds, yhat, yhat_lower, yhat_upper, trend, etc.
    """
    periods = periods or PROPHET_CONFIG["forecast_days"]
    model = trained["model"]

    future = model.make_future_dataframe(periods=periods)

    # Add regressor values for future dates
    if trained["regressors"] and future_regressors is not None:
        for reg in trained["regressors"]:
            if reg in future_regressors.columns:
                future[reg] = future_regressors[reg].values[:len(future)]

    # Fill missing regressor values with last known value
    for reg in trained["regressors"]:
        if reg in future.columns:
            future[reg] = future[reg].ffill().bfill()
        else:
            # Use mean from training data
            if reg in trained["train_df"].columns:
                future[reg] = trained["train_df"][reg].mean()

    forecast = model.predict(future)

    return forecast


def run_prophet_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame,
                          target_col: str = "price",
                          regressors: list = None) -> dict:
    """
    Full Prophet pipeline: train, forecast, and return results with actuals for evaluation.
    """
    if not PROPHET_AVAILABLE:
        raise ImportError(
            "Prophet is not installed.\n"
            "Install with: pip install prophet\n"
            "Note: Prophet may require Python 3.10-3.12 and C++ build tools."
        )

    trained = train_prophet(train_df, target_col, regressors)
    forecast_steps = len(test_df)
    forecast = forecast_prophet(trained, periods=forecast_steps)

    # Extract only the forecast portion (last forecast_steps rows)
    forecast_portion = forecast.tail(forecast_steps).reset_index(drop=True)

    return {
        "model_name": "Prophet",
        "predictions": forecast_portion["yhat"].values,
        "lower_ci": forecast_portion["yhat_lower"].values,
        "upper_ci": forecast_portion["yhat_upper"].values,
        "actuals": test_df[target_col].values,
        "dates": test_df["date"].values if "date" in test_df.columns else None,
        "trend": forecast_portion["trend"].values,
        "full_forecast": forecast,
        "trained_model": trained,
    }
