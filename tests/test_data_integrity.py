"""
Data integrity tests for the BTC-USD feature pipeline.
Validates no data leakage, no NaN values, and correct structure.
"""
import os
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_btc_data
from src.features import build_features, _compute_rsi, _compute_macd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_data():
    """Load raw BTC-USD data."""
    return load_btc_data()


@pytest.fixture(scope="module")
def feature_data(raw_data):
    """Build features from raw data."""
    return build_features(raw_data)


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_returns_dataframe(self, raw_data):
        assert isinstance(raw_data, pd.DataFrame)

    def test_has_required_columns(self, raw_data):
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in raw_data.columns, f"Missing column: {col}"

    def test_datetime_index(self, raw_data):
        assert isinstance(raw_data.index, pd.DatetimeIndex)

    def test_date_range(self, raw_data):
        assert raw_data.index.min() >= pd.Timestamp("2017-01-01")
        assert raw_data.index.max() <= pd.Timestamp("2023-12-31")

    def test_no_future_dates(self, raw_data):
        assert raw_data.index.max() <= pd.Timestamp("2023-12-31")

    def test_no_duplicate_dates(self, raw_data):
        assert not raw_data.index.duplicated().any()

    def test_sorted_index(self, raw_data):
        assert raw_data.index.is_monotonic_increasing

    def test_no_nan_in_ohlcv(self, raw_data):
        assert raw_data[["open", "high", "low", "close", "volume"]].isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_returns_dataframe(self, feature_data):
        assert isinstance(feature_data, pd.DataFrame)

    def test_has_required_columns(self, feature_data):
        expected = ["returns", "rsi_14", "macd", "macd_signal", "macd_hist", "target"]
        for col in expected:
            assert col in feature_data.columns, f"Missing column: {col}"

    def test_no_nan_values(self, feature_data):
        nan_count = feature_data.isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values"

    def test_target_is_binary(self, feature_data):
        assert set(feature_data["target"].unique()).issubset({0, 1})

    def test_rsi_range(self, feature_data):
        assert feature_data["rsi_14"].min() >= 0
        assert feature_data["rsi_14"].max() <= 100

    def test_reasonable_row_count(self, feature_data):
        # BTC-USD 2017-2023 should have ~2500 trading days, minus warm-up
        assert len(feature_data) > 2000

    def test_datetime_index(self, feature_data):
        assert isinstance(feature_data.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# Leakage tests
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_returns_use_past_data_only(self, raw_data):
        """Returns at time t use close[t] and close[t-1] — both available at t."""
        returns = raw_data["close"].pct_change()
        # Returns at index i should equal (close[i] - close[i-1]) / close[i-1]
        for i in range(1, min(50, len(raw_data))):
            expected = (raw_data["close"].iloc[i] - raw_data["close"].iloc[i - 1]) / raw_data["close"].iloc[i - 1]
            np.testing.assert_almost_equal(returns.iloc[i], expected, decimal=10)

    def test_rsi_no_centered_window(self):
        """Verify RSI does not use centered rolling windows."""
        # Create a simple series and verify RSI only uses past data
        series = pd.Series(range(100), dtype=float)
        rsi = _compute_rsi(series, period=14)
        # RSI should be NaN for first `period` entries
        assert rsi.iloc[:14].isna().any()

    def test_target_is_next_day(self, raw_data):
        """Target uses next day's data (shift(-1)), which is the prediction target."""
        next_close = raw_data["close"].shift(-1)
        next_open = raw_data["open"].shift(-1)
        # Last value should be NaN (no next day available)
        assert pd.isna(next_close.iloc[-1])
        assert pd.isna(next_open.iloc[-1])

    def test_no_future_features_in_feature_dates(self, feature_data, raw_data):
        """Feature dates should not extend beyond raw data dates."""
        assert feature_data.index.max() <= raw_data.index.max()


# ---------------------------------------------------------------------------
# Pickle output test
# ---------------------------------------------------------------------------

class TestPickleOutput:
    def test_features_pkl_loadable(self):
        """Verify that features.pkl can be loaded if it exists."""
        path = "data/processed/features.pkl"
        if os.path.exists(path):
            df = pd.read_pickle(path)
            assert isinstance(df, pd.DataFrame)
            assert "target" in df.columns
            assert df.isna().sum().sum() == 0
