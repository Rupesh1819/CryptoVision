"""
LSTM Forecasting Model (PyTorch)
Implements stacked LSTM for crypto price forecasting.
Uses PyTorch instead of TensorFlow for Python 3.14 compatibility.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LSTM_CONFIG


def is_available() -> bool:
    """Check if PyTorch is available for LSTM."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _get_availability_message() -> str:
    """Return user-friendly install instructions."""
    if is_available():
        return "PyTorch is installed and ready."
    return (
        "LSTM requires PyTorch. Install with:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
    )


# ── PyTorch LSTM Model ──

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


class LSTMModel(nn.Module):
    """Stacked LSTM for time series forecasting."""

    def __init__(self, input_size=1, hidden1=128, hidden2=64, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.fc(out[:, -1, :])  # Last time step
        return out


def _create_sequences(data: np.ndarray, lookback: int):
    """Create sliding window sequences for LSTM."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def train_lstm(df: pd.DataFrame, target_col: str = "price",
               lookback: int = None, epochs: int = None,
               batch_size: int = None) -> dict:
    """
    Train LSTM model using PyTorch.

    Args:
        df: Training DataFrame
        target_col: Column to forecast
        lookback: Number of past days to use as features
        epochs: Training epochs
        batch_size: Mini-batch size

    Returns:
        Dict with model, scaler, lookback, train_loss, last_sequence
    """
    if not is_available():
        raise ImportError(_get_availability_message())

    lookback = lookback or LSTM_CONFIG["lookback_window"]
    epochs = epochs or LSTM_CONFIG["epochs"]
    batch_size = batch_size or LSTM_CONFIG["batch_size"]

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df[[target_col]].values)

    # Create sequences
    X, y = _create_sequences(data_scaled, lookback)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build model
    model = LSTMModel(
        input_size=1,
        hidden1=LSTM_CONFIG["lstm_units_1"],
        hidden2=LSTM_CONFIG["lstm_units_2"],
        dropout=LSTM_CONFIG["dropout"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_CONFIG["learning_rate"])
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    best_loss = np.inf
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Store the last sequence for forecasting
    last_sequence = data_scaled[-lookback:]

    return {
        "model": model,
        "scaler": scaler,
        "lookback": lookback,
        "train_loss": best_loss,
        "last_sequence": last_sequence,
    }


def predict_lstm(trained: dict, X: np.ndarray) -> np.ndarray:
    """Generate predictions from trained LSTM."""
    model = trained["model"]
    scaler = trained["scaler"]

    model.eval()
    with torch.no_grad():
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        X_tensor = torch.FloatTensor(X_reshaped)
        predictions_scaled = model(X_tensor).numpy().flatten()

    # Inverse transform
    predictions = scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).flatten()

    return predictions


def forecast_lstm(trained: dict, steps: int = None) -> np.ndarray:
    """
    Generate multi-step future forecast using recursive prediction.

    Returns array of predicted values.
    """
    steps = steps or LSTM_CONFIG["forecast_days"]
    model = trained["model"]
    scaler = trained["scaler"]
    lookback = trained["lookback"]

    model.eval()
    current_sequence = trained["last_sequence"].copy()

    predictions_scaled = []

    with torch.no_grad():
        for _ in range(steps):
            # Reshape for LSTM input: (1, lookback, 1)
            input_tensor = torch.FloatTensor(
                current_sequence.reshape(1, lookback, 1)
            )
            next_pred = model(input_tensor).item()
            predictions_scaled.append(next_pred)

            # Slide window forward
            current_sequence = np.append(
                current_sequence[1:], [[next_pred]], axis=0
            )

    # Inverse transform
    predictions = scaler.inverse_transform(
        np.array(predictions_scaled).reshape(-1, 1)
    ).flatten()

    return predictions


def run_lstm_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       target_col: str = "price") -> dict:
    """
    Full LSTM pipeline: train, predict on test set, return results.
    """
    if not is_available():
        raise ImportError(_get_availability_message())

    trained = train_lstm(train_df, target_col)

    # Prepare test sequences
    scaler = trained["scaler"]
    lookback = trained["lookback"]

    # Combine train + test for proper sequences
    all_data = pd.concat([train_df, test_df], ignore_index=True)
    all_scaled = scaler.transform(all_data[[target_col]].values)

    # Create test sequences
    train_len = len(train_df)
    X_test, y_test = [], []

    for i in range(train_len, len(all_scaled)):
        X_test.append(all_scaled[i - lookback:i, 0])
        y_test.append(all_scaled[i, 0])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Predict
    predictions = predict_lstm(trained, X_test)
    actuals = test_df[target_col].values

    # Match lengths (sequences may be shorter than test set)
    min_len = min(len(predictions), len(actuals))
    predictions = predictions[:min_len]
    actuals = actuals[:min_len]

    return {
        "model_name": "LSTM",
        "predictions": predictions,
        "actuals": actuals,
        "dates": test_df["date"].values[:min_len] if "date" in test_df.columns else None,
        "trained_model": trained,
        "train_loss": trained["train_loss"],
    }
