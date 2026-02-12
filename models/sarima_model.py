"""
SARIMA Forecasting Model
Implements Seasonal ARIMA (SARIMAX) for crypto price forecasting.
Works on Python 3.14 — uses statsmodels only.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SARIMA_CONFIG


def find_best_sarima_order(series: pd.Series) -> dict:
    """
    Find best SARIMA order using grid search with AIC.

    Returns dict with 'order' and 'seasonal_order'.
    """
    best_aic = np.inf
    best_order = (1, 1, 1)
    best_seasonal = (1, 1, 0, SARIMA_CONFIG["seasonal_period"])

    # Grid search over a reduced space for speed
    for p in range(SARIMA_CONFIG["max_p"] + 1):
        for d in range(SARIMA_CONFIG["max_d"] + 1):
            for q in range(SARIMA_CONFIG["max_q"] + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = SARIMAX(
                        series.dropna(),
                        order=(p, d, q),
                        seasonal_order=best_seasonal,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted = model.fit(disp=False, maxiter=100)
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    return {"order": best_order, "seasonal_order": best_seasonal, "aic": best_aic}


def train_sarima(df: pd.DataFrame, target_col: str = "price",
                  order: tuple = None, seasonal_order: tuple = None) -> dict:
    """
    Train SARIMA model.

    Args:
        df: Training DataFrame with target column
        target_col: Column to forecast
        order: (p, d, q) tuple; auto-selects if None
        seasonal_order: (P, D, Q, s) tuple; uses config default if None

    Returns:
        Dict with model, order, seasonal_order, aic, fitted_values, residuals
    """
    series = df[target_col].values

    if order is None or seasonal_order is None:
        best = find_best_sarima_order(pd.Series(series))
        order = order or best["order"]
        seasonal_order = seasonal_order or best["seasonal_order"]

    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted_model = model.fit(disp=False, maxiter=200)

    return {
        "model": fitted_model,
        "order": order,
        "seasonal_order": seasonal_order,
        "aic": fitted_model.aic,
        "bic": fitted_model.bic,
        "fitted_values": fitted_model.fittedvalues,
        "residuals": fitted_model.resid,
    }


def forecast_sarima(trained: dict, steps: int = None,
                     alpha: float = 0.05) -> pd.DataFrame:
    """
    Generate forecast from a trained SARIMA model.

    Returns:
        DataFrame with columns: step, forecast, lower_ci, upper_ci
    """
    steps = steps or SARIMA_CONFIG["forecast_days"]
    model = trained["model"]

    forecast_result = model.get_forecast(steps=steps, alpha=alpha)
    mean_forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=alpha)

    # Handle both numpy arrays and pandas objects
    forecast_vals = mean_forecast.values if hasattr(mean_forecast, 'values') else np.asarray(mean_forecast)

    if hasattr(conf_int, 'iloc'):
        lower_vals = conf_int.iloc[:, 0].values
        upper_vals = conf_int.iloc[:, 1].values
    else:
        conf_arr = np.asarray(conf_int)
        lower_vals = conf_arr[:, 0]
        upper_vals = conf_arr[:, 1]

    return pd.DataFrame({
        "step": range(1, steps + 1),
        "forecast": forecast_vals,
        "lower_ci": lower_vals,
        "upper_ci": upper_vals,
    })


def run_sarima_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         target_col: str = "price") -> dict:
    """
    Full SARIMA pipeline: train, forecast, and return results with actuals.
    """
    trained = train_sarima(train_df, target_col)
    forecast_steps = len(test_df)
    forecast = forecast_sarima(trained, steps=forecast_steps)

    return {
        "model_name": "SARIMA",
        "order": trained["order"],
        "seasonal_order": trained["seasonal_order"],
        "aic": trained["aic"],
        "predictions": forecast["forecast"].values,
        "lower_ci": forecast["lower_ci"].values,
        "upper_ci": forecast["upper_ci"].values,
        "actuals": test_df[target_col].values,
        "dates": test_df["date"].values if "date" in test_df.columns else None,
        "trained_model": trained,
    }
