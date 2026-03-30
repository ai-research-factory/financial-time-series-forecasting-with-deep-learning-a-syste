# Technical Findings — Cycle 2 (Phase 2: Data Pipeline)

## Implementation Summary

### Data Loader (`src/data_loader.py`)
- Fetches BTC-USD daily OHLCV data from the ARF Data API (`period=10y`)
- Filters to 2017-01-01 through 2023-12-31 as specified in the design brief
- Caches raw data locally at `data/raw/btc_usd.csv` for subsequent runs
- Validates no future dates, no duplicates, sorted index

### Feature Engineering (`src/features.py`)
- **Returns**: Close-to-close percentage change (`pct_change()`)
- **RSI(14)**: Exponential weighted moving average method (Wilder's smoothing)
- **MACD(12, 26, 9)**: Standard MACD line, signal line, and histogram
- **Target**: Binary — 1 if next day's close > next day's open, 0 otherwise

### CLI (`main.py`)
- `python3 main.py process-data` fetches data and generates `data/processed/features.pkl`

## Data Statistics

All values sourced from `metrics.json`.

| Metric | Value |
|---|---|
| Raw data rows | 2556 |
| Raw data range | 2017-01-01 to 2023-12-31 |
| Feature matrix rows | 2543 |
| Feature matrix range | 2017-01-14 to 2023-12-31 |
| Feature columns | 6 (returns, rsi_14, macd, macd_signal, macd_hist, target) |
| Target positive (close > open) | 1341 (52.7%) |
| Target negative | 1202 (47.3%) |
| NaN values | 0 |
| Warm-up rows dropped | 13 (from RSI/MACD initialization) |

## Data Integrity Verification

20 tests pass covering:
- Data loader: shape, columns, date range, no duplicates, sorted, no NaN
- Features: shape, columns, no NaN, binary target, RSI in [0, 100]
- Leakage: returns use past data only, no centered windows, target uses shift(-1)
- Output: pickle file loadable with correct structure

## Observations

1. **Target balance**: The target is reasonably balanced (52.7% positive), which is good for classification — no severe class imbalance handling needed.
2. **Warm-up period**: 13 rows are dropped due to RSI(14) and MACD(26) warm-up periods. The RSI EWM requires 14 periods minimum; since MACD's slow EMA (26) warms up faster than RSI with `adjust=False`, the effective warm-up is dominated by RSI.
3. **Data availability**: The ARF Data API provides BTC-USD data from 2016-03-30, which covers the design brief's requested 2017-01-01 start date.

## Next Steps (Phase 3)

- Implement walk-forward evaluation framework using `src/backtest.py`
- Train LSTM model with expanding windows
- Generate out-of-sample predictions
