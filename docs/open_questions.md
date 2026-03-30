# Open Questions

## Data

- The design brief references `yfinance.download('BTC-USD', ...)` but this project uses the ARF Data API as required by CLAUDE.md. The data should be equivalent (both source from Yahoo Finance), but minor differences in OHLCV values are possible due to data provider adjustments.
- BTC-USD trades 365 days/year, unlike traditional equities (252 days). The `periods_per_year` parameter in `compute_metrics()` should use 365 for crypto when computing annualized metrics (Phase 3+).
