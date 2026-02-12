# 🪙 CryptoVision

**CryptoVision** is a comprehensive cryptocurrency time series analysis dashboard that leverages advanced AI forecasting and sentiment intelligence to provide actionable insights into the crypto markets.

## 🚀 Features

-   **🌐 Market Overview**: Real-time tracking of top cryptocurrency prices, market caps, and trends.
-   **📜 Historical Analysis & EDA**: Interactive charts for exploring historical price data, volume, and moving averages.
-   **🔮 AI Forecasting**:
    -   **ARIMA / SARIMA**: Classical statistical models for time series forecasting.
    -   **Prophet**: Facebook's robust forecasting procedure for handling trends and seasonality.
    -   **LSTM (Long Short-Term Memory)**: Deep learning model for capturing complex temporal dependencies.
-   **⚡ Volatility & Risk**: Analyze market volatility, value-at-risk (VaR), and other risk metrics.
-   **💬 Sentiment Intelligence**: Real-time sentiment analysis using VADER and NewsAPI to gauge market mood.

## 🛠️ Tech Stack

-   **Frontend**: [Streamlit](https://streamlit.io/)
-   **Visualization**: [Plotly](https://plotly.com/)
-   **Data Analysis**: Pandas, NumPy, SciPy, Statsmodels
-   **Machine Learning**: TensorFlow (Keras), Scikit-learn, Prophet
-   **NLP / Sentiment**: VADER Sentiment, NewsAPI, WordCloud
-   **Data Sources**: Yahoo Finance (`yfinance`), CoinGecko

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/cryptovision.git
    cd cryptovision
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Some libraries like Prophet or TensorFlow may require additional system dependencies.*

## 🚦 Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## 📂 Project Structure

-   `app.py`: Main entry point for the Streamlit application.
-   `dashboard/`: Contains shared components and page rendering logic.
-   `data/`: Data collection and preprocessing modules.
-   `models/`: Implementation of forecasting models (ARIMA, LSTM, Prophet, etc.).
-   `sentiment/`: Sentiment analysis tools and news fetchers.
-   `analysis/`: Statistical analysis and utilities.

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. It does not constitute financial advice. Cryptocurrency markets are highly volatile; trade at your own risk.
