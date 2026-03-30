"""
CLI entry point for the Financial Time Series Forecasting project.
"""
import argparse
import os
import pickle

import pandas as pd

from src.data_loader import load_btc_data
from src.features import build_features


def process_data(args: argparse.Namespace) -> None:
    """Fetch BTC-USD data, compute features, and save processed dataset."""
    print("Loading BTC-USD data...")
    df = load_btc_data()
    print(f"  Raw data shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    print("Building features...")
    features = build_features(df)
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Columns: {list(features.columns)}")
    print(f"  Date range: {features.index.min()} to {features.index.max()}")
    print(f"  Target distribution: {features['target'].value_counts().to_dict()}")

    # Validate no NaN
    nan_count = features.isna().sum().sum()
    assert nan_count == 0, f"Feature matrix contains {nan_count} NaN values"

    # Save
    out_path = "data/processed/features.pkl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    features.to_pickle(out_path)
    print(f"Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Financial Time Series Forecasting with Deep Learning"
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("process-data", help="Fetch and preprocess BTC-USD data")

    args = parser.parse_args()

    if args.command == "process-data":
        process_data(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
