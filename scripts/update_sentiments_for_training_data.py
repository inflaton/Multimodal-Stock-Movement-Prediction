#!/usr/bin/env python3
"""
Update Sentiment Scores in Training Data

This script updates the "Weighted Sentiment Score" column in the training data
with values from the filtered sentiment CSV files.

For each stock (AAPL, META, NVDA, SPY, TSLA):
1. Loads training data from dataset/training_data/{STOCK}_data_model_training.csv
2. Loads filtered sentiment from dataset/sentiment/news_sentiment_finbert_tone_weighted_{stock}.csv
3. Updates the Weighted Sentiment Score column
4. Saves back to dataset/training_data/{STOCK}_data_model_training.csv

Usage:
    python scripts/update_sentiments_for_training_data.py
    python scripts/update_sentiments_for_training_data.py --stock NVDA
"""

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

# Configuration
STOCKS = ["AAPL", "META", "NVDA", "SPY", "TSLA"]


def update_stock_sentiment(stock, training_dir, sentiment_dir):
    """
    Update sentiment scores for a single stock.

    Args:
        stock: Stock symbol (e.g., "NVDA")
        training_dir: Directory containing training data
        sentiment_dir: Directory containing sentiment data

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Processing {stock}")
    print(f"{'='*70}")

    # Load training data
    training_file = f"{training_dir}/{stock}_data_model_training.csv"
    print(f"Loading training data from: {training_file}")

    if not os.path.exists(training_file):
        print(f"ERROR: Training file not found: {training_file}")
        return False

    df = pd.read_csv(training_file)

    # Handle different date column names (Date, Date_x, date)
    date_col_training = None
    for col in ["Date", "Date_x", "date"]:
        if col in df.columns:
            date_col_training = col
            break

    if date_col_training is None:
        print(f"ERROR: No date column found in {training_file}")
        print(f"Available columns: {df.columns.tolist()}")
        return False

    # If we have Date_x and Date_y from previous merge, clean up
    if "Date_x" in df.columns:
        if date_col_training != "Date":
            df = df.rename(columns={date_col_training: "Date"})
            date_col_training = "Date"
        # Drop Date_y if it exists
        if "Date_y" in df.columns:
            df = df.drop(columns=["Date_y"])

    df["__DateDT__"] = pd.to_datetime(
        df[date_col_training], format="%d/%m/%Y", errors="coerce"
    )

    print(f"Original dataset shape: {df.shape}")
    print(f"Date range: {df['__DateDT__'].min()} to {df['__DateDT__'].max()}")

    # Store original sentiment stats
    if "Weighted Sentiment Score" in df.columns:
        original_sentiment = df["Weighted Sentiment Score"].copy()
        print(
            f"Original sentiment range: [{original_sentiment.min():.4f}, {original_sentiment.max():.4f}]"
        )
        print(f"Original sentiment mean: {original_sentiment.mean():.4f}")

    # Load sentiment data
    sentiment_file = (
        f"{sentiment_dir}/news_sentiment_finbert_tone_weighted_{stock.lower()}.csv"
    )
    print(f"Loading sentiment from: {sentiment_file}")

    if not os.path.exists(sentiment_file):
        print(f"ERROR: Sentiment file not found: {sentiment_file}")
        return False

    sentiment_df = pd.read_csv(sentiment_file)

    # Check if 'date' column exists, otherwise try 'Date'
    date_col = "date" if "date" in sentiment_df.columns else "Date"
    sentiment_col = (
        "weighted_sentiment"
        if "weighted_sentiment" in sentiment_df.columns
        else "Weighted Sentiment"
    )

    sentiment_df[date_col] = pd.to_datetime(sentiment_df[date_col])
    sentiment_df.rename(columns={sentiment_col: "Updated Sentiment"}, inplace=True)

    print(f"Sentiment data shape: {sentiment_df.shape}")
    print(
        f"Sentiment date range: {sentiment_df[date_col].min()} to {sentiment_df[date_col].max()}"
    )

    # Merge sentiment with training data
    df = df.merge(
        sentiment_df[[date_col, "Updated Sentiment"]],
        left_on="__DateDT__",
        right_on=date_col,
        how="left",
    )

    # Drop the duplicate date column from merge
    if date_col in df.columns and date_col != "__DateDT__":
        df.drop(columns=[date_col], inplace=True)

    # Check for missing values
    missing_count = df["Updated Sentiment"].isna().sum()
    print(f"Missing sentiment values: {missing_count} out of {len(df)}")

    if missing_count > 0:
        if "Weighted Sentiment Score" in df.columns:
            df["Updated Sentiment"] = df["Updated Sentiment"].fillna(
                df["Weighted Sentiment Score"]
            )
            print(
                f"Filled {missing_count} missing values with original sentiment scores"
            )
        else:
            print("WARNING: No original sentiment to fall back on, filling with 0")
            df["Updated Sentiment"] = df["Updated Sentiment"].fillna(0)

    # Update the Weighted Sentiment Score column
    if "Weighted Sentiment Score" in df.columns:
        # Check if values changed
        changes = (df["Weighted Sentiment Score"] != df["Updated Sentiment"]).sum()
        print(f"\nSentiment values changed: {changes} out of {len(df)}")

        df["Weighted Sentiment Score"] = df["Updated Sentiment"]
    else:
        print("WARNING: 'Weighted Sentiment Score' column not found, adding it")
        # Find the position after 'Close' to insert the sentiment column
        if "Close" in df.columns:
            close_idx = df.columns.tolist().index("Close")
            cols = df.columns.tolist()
            cols.insert(close_idx + 1, "Weighted Sentiment Score")
            df["Weighted Sentiment Score"] = df["Updated Sentiment"]
            df = df[cols]
        else:
            df["Weighted Sentiment Score"] = df["Updated Sentiment"]

    # Remove temporary columns
    columns_to_drop = ["Updated Sentiment", "__DateDT__"]

    # Also clean up any Date_x or Date_y columns from merges
    if "Date_x" in df.columns and "Date_y" in df.columns:
        # Rename Date_x back to Date
        df.rename(columns={"Date_x": "Date"}, inplace=True)
        columns_to_drop.append("Date_y")
    elif "Date_x" in df.columns:
        df.rename(columns={"Date_x": "Date"}, inplace=True)
    elif "Date_y" in df.columns:
        columns_to_drop.append("Date_y")

    # Drop all temporary columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)

    # Print updated sentiment stats
    updated_sentiment = df["Weighted Sentiment Score"]
    print(
        f"Updated sentiment range: [{updated_sentiment.min():.4f}, {updated_sentiment.max():.4f}]"
    )
    print(f"Updated sentiment mean: {updated_sentiment.mean():.4f}")

    # Save the updated file
    df.to_csv(training_file, index=False)

    print(f"\n✓ Updated and saved to: {training_file}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Rows: {len(df)}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Update sentiment scores in training data CSV files"
    )
    parser.add_argument(
        "--stock",
        type=str,
        default=None,
        choices=STOCKS,
        help="Update a specific stock (default: all stocks)",
    )
    parser.add_argument(
        "--training-dir",
        type=str,
        default="dataset/training_data",
        help="Directory containing training data (default: dataset/training_data)",
    )
    parser.add_argument(
        "--sentiment-dir",
        type=str,
        default="dataset/sentiment",
        help="Directory containing sentiment data (default: dataset/sentiment)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Update Sentiment Scores in Training Data")
    print("=" * 70)
    print(f"Training data directory: {args.training_dir}")
    print(f"Sentiment data directory: {args.sentiment_dir}")

    # Process stocks
    stocks_to_process = [args.stock] if args.stock else STOCKS

    success_count = 0
    for stock in stocks_to_process:
        if update_stock_sentiment(stock, args.training_dir, args.sentiment_dir):
            success_count += 1

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully updated: {success_count}/{len(stocks_to_process)} stocks")
    print(f"Training data directory: {args.training_dir}")

    print(f"\nUpdated files:")
    for stock in stocks_to_process:
        training_file = f"{args.training_dir}/{stock}_data_model_training.csv"
        if os.path.exists(training_file):
            size_mb = os.path.getsize(training_file) / 1024 / 1024
            print(f"  ✓ {training_file} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {training_file} (not found)")

    print()


if __name__ == "__main__":
    main()
