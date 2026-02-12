"""
Test Suite for CryptoVision Pipeline
Tests data collection, preprocessing, model training, and evaluation.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════════════
# Data Collection Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDataCollector:
    def test_fetch_historical_data_returns_dataframe(self):
        """Verify that historical data fetch returns a valid DataFrame."""
        from data.collector import fetch_historical_data
        df = fetch_historical_data("bitcoin", days=30)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "date" in df.columns
        assert "price" in df.columns
        assert "total_volume" in df.columns

    def test_fetch_historical_data_has_correct_types(self):
        """Verify column data types."""
        from data.collector import fetch_historical_data
        df = fetch_historical_data("bitcoin", days=30)

        assert pd.api.types.is_numeric_dtype(df["price"])
        assert df["price"].min() > 0  # Prices should be positive

    def test_fetch_realtime_prices(self):
        """Verify real-time price fetching returns expected structure."""
        from data.collector import fetch_realtime_prices
        df = fetch_realtime_prices(["bitcoin", "ethereum"])

        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 2
        assert "price" in df.columns
        assert "coin_id" in df.columns

    def test_sample_data_generation(self):
        """Verify that sample data can be generated as fallback."""
        from data.collector import _generate_sample_data
        df = _generate_sample_data("bitcoin", 100)

        assert len(df) == 100
        assert "price" in df.columns
        assert df["price"].min() > 0


# ═══════════════════════════════════════════════════════════════════════
# Preprocessing Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPreprocessor:
    def _create_sample_df(self, n=200):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        prices = 40000 + np.cumsum(np.random.normal(0, 500, n))
        prices = np.maximum(prices, 100)  # Ensure positive

        return pd.DataFrame({
            "date": dates,
            "price": prices,
            "total_volume": np.random.lognormal(20, 1, n),
        })

    def test_handle_missing_values(self):
        """Verify missing value handling."""
        from data.preprocessor import handle_missing_values

        df = self._create_sample_df()
        df.loc[5, "price"] = np.nan
        df.loc[10, "price"] = np.nan

        result = handle_missing_values(df, method="ffill")
        assert result["price"].isna().sum() == 0

    def test_outlier_detection_iqr(self):
        """Verify IQR outlier detection adds boolean column."""
        from data.preprocessor import detect_outliers_iqr

        df = self._create_sample_df()
        result = detect_outliers_iqr(df, "price")

        assert "is_outlier_iqr" in result.columns
        assert result["is_outlier_iqr"].dtype == bool

    def test_outlier_detection_zscore(self):
        """Verify Z-score outlier detection adds boolean column."""
        from data.preprocessor import detect_outliers_zscore

        df = self._create_sample_df()
        result = detect_outliers_zscore(df, "price")

        assert "is_outlier_zscore" in result.columns

    def test_add_technical_features(self):
        """Verify technical features are computed correctly."""
        from data.preprocessor import add_technical_features

        df = self._create_sample_df()
        result = add_technical_features(df)

        expected_cols = ["returns", "log_returns", "ma_7", "ma_30", "ma_90",
                         "rsi_14", "macd", "bb_upper", "bb_lower"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_preprocess_pipeline(self):
        """Verify full pipeline runs without errors."""
        from data.preprocessor import preprocess_pipeline

        df = self._create_sample_df()
        result = preprocess_pipeline(df)

        assert len(result) > 0
        assert "returns" in result.columns
        assert "is_outlier_iqr" in result.columns

    def test_prepare_for_modeling(self):
        """Verify train/test split works correctly."""
        from data.preprocessor import prepare_for_modeling

        df = self._create_sample_df()
        splits = prepare_for_modeling(df)

        assert "train" in splits
        assert "test" in splits
        assert len(splits["train"]) + len(splits["test"]) == len(df)
        assert len(splits["train"]) > len(splits["test"])


# ═══════════════════════════════════════════════════════════════════════
# Model Evaluation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEvaluator:
    def test_compute_mae(self):
        """Verify MAE calculation."""
        from models.evaluator import compute_mae

        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190, 310])
        mae = compute_mae(actual, predicted)

        assert abs(mae - 10.0) < 1e-6

    def test_compute_rmse(self):
        """Verify RMSE calculation."""
        from models.evaluator import compute_rmse

        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190, 310])
        rmse = compute_rmse(actual, predicted)

        assert rmse > 0
        assert rmse >= compute_rmse(actual, actual)  # Perfect prediction = 0

    def test_compute_mape(self):
        """Verify MAPE calculation."""
        from models.evaluator import compute_mape

        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190, 310])
        mape = compute_mape(actual, predicted)

        assert 0 < mape < 100  # Should be a reasonable percentage

    def test_compute_r2(self):
        """Verify R² calculation."""
        from models.evaluator import compute_r2

        actual = np.array([100, 200, 300])
        predicted = np.array([100, 200, 300])
        r2 = compute_r2(actual, predicted)

        assert abs(r2 - 1.0) < 1e-6  # Perfect prediction

    def test_compare_models(self):
        """Verify model comparison returns sorted DataFrame."""
        from models.evaluator import compare_models

        results = [
            {"model_name": "A", "predictions": np.array([110, 210]), "actuals": np.array([100, 200])},
            {"model_name": "B", "predictions": np.array([105, 205]), "actuals": np.array([100, 200])},
        ]

        comparison = compare_models(results)
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert comparison.iloc[0]["rmse"] <= comparison.iloc[1]["rmse"]


# ═══════════════════════════════════════════════════════════════════════
# Sentiment Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSentiment:
    def test_vader_scoring(self):
        """Verify VADER returns scores in expected range."""
        from sentiment.analyzer import get_sentiment_scores

        texts = [
            "Bitcoin surges to new all-time high! Great news for investors!",
            "Crypto market crashes, billions lost in sell-off.",
            "Bitcoin trades at $50,000 today.",
        ]

        scores = get_sentiment_scores(texts)
        assert len(scores) == 3

        for s in scores:
            assert -1 <= s["compound"] <= 1
            assert s["label"] in ["Positive", "Negative", "Neutral"]

        # The positive headline should score higher
        assert scores[0]["compound"] > scores[1]["compound"]

    def test_analyze_news_sentiment(self):
        """Verify sentiment analysis on news DataFrame."""
        from sentiment.analyzer import analyze_news_sentiment

        news_df = pd.DataFrame({
            "title": ["Bitcoin surges!", "Market crashes", "Steady trading"],
            "description": ["Great gains", "Major losses", "Sideways movement"],
            "source": ["A", "B", "C"],
            "published_at": ["2024-01-01", "2024-01-02", "2024-01-03"],
        })

        result = analyze_news_sentiment(news_df)
        assert "compound" in result.columns
        assert "label" in result.columns
        assert len(result) == 3

    def test_sentiment_summary(self):
        """Verify sentiment summary computation."""
        from sentiment.analyzer import get_sentiment_summary, analyze_news_sentiment

        news_df = pd.DataFrame({
            "title": ["Great news!", "Bad news", "Neutral"],
            "description": ["Positive", "Negative", "Normal"],
        })
        sentiment_df = analyze_news_sentiment(news_df)
        summary = get_sentiment_summary(sentiment_df)

        assert "overall" in summary
        assert "avg_compound" in summary
        assert summary["total_articles"] == 3


# ═══════════════════════════════════════════════════════════════════════
# Volatility Tests
# ═══════════════════════════════════════════════════════════════════════

class TestVolatility:
    def _create_sample_df(self, n=200):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        prices = 40000 + np.cumsum(np.random.normal(0, 500, n))
        prices = np.maximum(prices, 100)

        return pd.DataFrame({"date": dates, "price": prices})

    def test_compute_risk_metrics(self):
        """Verify risk metrics computation."""
        from analysis.volatility import compute_risk_metrics

        df = self._create_sample_df()
        metrics = compute_risk_metrics(df)

        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "var_95" in metrics
        assert metrics["max_drawdown"] <= 0  # Drawdown is negative

    def test_compute_bollinger_bands(self):
        """Verify Bollinger Bands computation."""
        from analysis.volatility import compute_bollinger_bands

        df = self._create_sample_df()
        result = compute_bollinger_bands(df)

        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_middle" in result.columns

        # Upper band should be above lower band (where both exist)
        valid = result.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()


# ═══════════════════════════════════════════════════════════════════════
# ARIMA Model Tests
# ═══════════════════════════════════════════════════════════════════════

class TestARIMA:
    def _create_sample_df(self, n=200):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        prices = 40000 + np.cumsum(np.random.normal(0, 500, n))
        prices = np.maximum(prices, 100)
        return pd.DataFrame({"date": dates, "price": prices})

    def test_stationarity_test(self):
        """Verify ADF stationarity test."""
        from models.arima_model import test_stationarity

        np.random.seed(42)
        series = pd.Series(np.random.normal(0, 1, 200))
        result = test_stationarity(series)

        assert "p_value" in result
        assert "is_stationary" in result
        assert result["is_stationary"] in (True, False)  # works with numpy.bool_ too

    def test_train_arima(self):
        """Verify ARIMA model training."""
        from models.arima_model import train_arima

        df = self._create_sample_df(100)
        result = train_arima(df, "price", order=(1, 1, 1))

        assert "model" in result
        assert "order" in result
        assert "aic" in result
        assert result["order"] == (1, 1, 1)

    def test_forecast_arima(self):
        """Verify ARIMA forecasting."""
        from models.arima_model import train_arima, forecast_arima

        df = self._create_sample_df(100)
        trained = train_arima(df, "price", order=(1, 1, 1))
        forecast = forecast_arima(trained, steps=10)

        assert len(forecast) == 10
        assert "forecast" in forecast.columns
        assert "lower_ci" in forecast.columns
        assert "upper_ci" in forecast.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
