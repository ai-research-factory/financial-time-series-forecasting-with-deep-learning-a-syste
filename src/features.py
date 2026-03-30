"""
Feature engineering for BTC-USD price direction prediction.
Computes technical indicators and target variable.
"""
import numpy as np
import pandas as pd


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix and target from OHLCV data.

    Features (all computed using data available at time t):
        - returns: close-to-close log returns
        - rsi_14: RSI with 14-period lookback
        - macd: MACD line (12, 26)
        - macd_signal: MACD signal line (9)
        - macd_hist: MACD histogram

    Target:
        - target: 1 if next day's close > next day's open, else 0

    Args:
        df: OHLCV DataFrame with DatetimeIndex

    Returns:
        DataFrame with features and target, NaN rows dropped
    """
    features = pd.DataFrame(index=df.index)

    # Close-to-close returns (available at end of day t)
    features["returns"] = df["close"].pct_change()

    # RSI(14)
    features["rsi_14"] = _compute_rsi(df["close"], period=14)

    # MACD(12, 26, 9)
    macd_line, macd_signal, macd_hist = _compute_macd(df["close"])
    features["macd"] = macd_line
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd_hist

    # Target: next day's close > next day's open → 1, else 0
    # shift(-1) to get next day's values
    features["target"] = (df["close"].shift(-1) > df["open"].shift(-1)).astype(int)

    # Drop rows with NaN (from indicators warm-up and last row target)
    features = features.dropna()

    return features
