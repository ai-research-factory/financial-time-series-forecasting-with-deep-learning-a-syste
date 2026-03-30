"""
Data loader for BTC-USD daily OHLCV data.
Fetches from ARF Data API and caches locally.
"""
import os
import pandas as pd

API_BASE = "https://ai.1s.xyz/api/data/ohlcv"
DEFAULT_TICKER = "BTC-USD"
DEFAULT_START = "2017-01-01"
DEFAULT_END = "2023-12-31"


def load_btc_data(
    ticker: str = DEFAULT_TICKER,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_path: str = "data/raw/btc_usd.csv",
) -> pd.DataFrame:
    """
    Load BTC-USD daily OHLCV data from ARF Data API.

    Fetches data via API on first call, then uses local cache.
    Filters to the specified date range.

    Args:
        ticker: Ticker symbol (default: BTC-USD)
        start: Start date string YYYY-MM-DD
        end: End date string YYYY-MM-DD
        cache_path: Path to save/load cached CSV

    Returns:
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume
    """
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["timestamp"], index_col="timestamp")
    else:
        url = f"{API_BASE}?ticker={ticker}&interval=1d&period=10y"
        df = pd.read_csv(url, parse_dates=["timestamp"], index_col="timestamp")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path)

    # Filter to requested date range
    df = df.loc[start:end]

    # Standardize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Sort by date and drop duplicates
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Validate no future dates
    assert df.index.max() <= pd.Timestamp(end), "Data contains dates beyond end date"

    return df
