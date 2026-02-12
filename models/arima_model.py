"""
ARIMA Forecasting Model
Implements Auto-ARIMA with stationarity testing and forecasting.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ARIMA_CONFIG


def test_stationarity(series: pd.Series) -> dict:
    """
    Perform Augmented Dickey-Fuller test for stationarity.

    Returns dict with test_statistic, p_value, is_stationary, critical_values.
    """
    result = adfuller(series.dropna(), autolag="AIC")

    return {
        "test_statistic": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "num_observations": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }


def find_best_order(series: pd.Series, max_p: int = None, max_d: int = None,
                     max_q: int = None) -> tuple:
    """
    Find the best ARIMA (p, d, q) order using AIC minimization.

    Uses pmdarima if available, otherwise a manual grid search.
    """
    max_p = max_p or ARIMA_CONFIG["max_p"]
    max_d = max_d or ARIMA_CONFIG["max_d"]
    max_q = max_q or ARIMA_CONFIG["max_q"]

    try:
        import pmdarima as pm
        model = pm.auto_arima(
            series.dropna(),
            start_p=0, start_q=0,
            max_p=max_p, max_d=max_d, max_q=max_q,
            seasonal=ARIMA_CONFIG["seasonal"],
            stepwise=True, suppress_warnings=True,
            error_action="ignore",
        )
        return model.order
    except ImportError:
        return _manual_grid_search(series, max_p, max_d, max_q)


def _manual_grid_search(series: pd.Series, max_p: int, max_d: int,
                         max_q: int) -> tuple:
    """Fallback: manual grid search for ARIMA order."""
    best_aic = np.inf
    best_order = (1, 1, 1)

    for d in range(max_d + 1):
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(series.dropna(), order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    return best_order


def train_arima(df: pd.DataFrame, target_col: str = "price",
                order: tuple = None) -> dict:
    """
    Train ARIMA model.

    Args:
        df: Training DataFrame with date and target columns
        target_col: Column to forecast
        order: (p, d, q) tuple; if None, auto-selects

    Returns:
        Dict with model, order, aic, fitted_values, residuals
    """
    series = df[target_col].values

    if order is None:
        order = find_best_order(pd.Series(series))

    model = ARIMA(series, order=order)
    fitted_model = model.fit()

    return {
        "model": fitted_model,
        "order": order,
        "aic": fitted_model.aic,
        "bic": fitted_model.bic,
        "fitted_values": fitted_model.fittedvalues,
        "residuals": fitted_model.resid,
    }


def forecast_arima(trained: dict, steps: int = None,
                    alpha: float = 0.05) -> pd.DataFrame:
    """
    Generate forecast from a trained ARIMA model.

    Returns:
        DataFrame with columns: step, forecast, lower_ci, upper_ci
    """
    steps = steps or ARIMA_CONFIG["forecast_days"]
    model = trained["model"]

    forecast_result = model.get_forecast(steps=steps, alpha=alpha)
    mean_forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=alpha)

    result = pd.DataFrame({
        "step": range(1, steps + 1),
        "forecast": mean_forecast,
        "lower_ci": conf_int[:, 0] if hasattr(conf_int, '__len__') else conf_int.iloc[:, 0],
        "upper_ci": conf_int[:, 1] if hasattr(conf_int, '__len__') else conf_int.iloc[:, 1],
    })

    return result


def run_arima_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        target_col: str = "price") -> dict:
    """
    Full ARIMA pipeline: train, forecast, and return results with actuals for evaluation.
    """
    trained = train_arima(train_df, target_col)
    forecast_steps = len(test_df)
    forecast = forecast_arima(trained, steps=forecast_steps)

    return {
        "model_name": "ARIMA",
        "order": trained["order"],
        "aic": trained["aic"],
        "predictions": forecast["forecast"].values,
        "lower_ci": forecast["lower_ci"].values,
        "upper_ci": forecast["upper_ci"].values,
        "actuals": test_df[target_col].values,
        "dates": test_df["date"].values if "date" in test_df.columns else None,
        "trained_model": trained,
    }
