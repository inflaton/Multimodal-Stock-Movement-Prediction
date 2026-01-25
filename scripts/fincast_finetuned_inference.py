#!/usr/bin/env python3
"""
FinCast Fine-tuned Model Inference

This script runs inference using fine-tuned FinCast models (with LoRA/PEFT).
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from sklearn.metrics import accuracy_score, roc_auc_score
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'FinCast-fts', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'FinCast-fts', 'peft_Fincast'))

from fincast_finetune import load_fincast_model_for_training, wrap_model_with_peft

# Constants
STOCKS = ["AAPL", "META", "NVDA", "SPY", "TSLA"]
TRAIN_YEARS = [2020, 2021, 2022]
TEST_YEAR = 2023
ALL_HORIZONS = list(range(2, 11))
DEFAULT_THRESHOLDS = {h: 0.01 for h in ALL_HORIZONS}
FEE_BPS_ROUND_TRIP = 10  # 10 basis points round-trip transaction cost


def non_overlap_backtest(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    horizon: int,
    fee_bps: int = FEE_BPS_ROUND_TRIP,
) -> dict:
    """
    Simulate trading with non-overlapping positions using long/short strategy.

    Each trade is held for exactly `horizon` days before the next trade begins.
    This ensures h × N ≤ 252 (trading days per year).
    """
    prices = test_df.sort_values("__DateDT__")[["__DateDT__", "Close"]].reset_index(
        drop=True
    )
    fee_mult = 1 - fee_bps / 10000

    long_trades = []
    short_trades = []
    equity = 1.0
    pred_idx = 0
    i = 0

    while i + horizon < len(prices) and pred_idx < len(predictions):
        p_in = prices.iloc[i]["Close"]
        p_out = prices.iloc[i + horizon]["Close"]
        pred = predictions[pred_idx]

        # Long/short strategy: long if pred=1, short if pred=0
        if pred == 1:
            gross_return = p_out / p_in - 1
            net_return = (1 + gross_return) * fee_mult - 1
            long_trades.append(net_return)
            equity *= 1 + net_return
        else:
            gross_return = p_in / p_out - 1
            net_return = (1 + gross_return) * fee_mult - 1
            short_trades.append(net_return)
            equity *= 1 + net_return

        i += horizon  # Non-overlapping: skip h days
        pred_idx += 1

    all_trades = np.array(long_trades + short_trades)

    if len(all_trades) == 0:
        return {
            "Trades": 0,
            "LongTrades": 0,
            "ShortTrades": 0,
            "WinRate": 0.0,
            "Sharpe": 0.0,
            "TotalReturn": 0.0,
        }

    win_rate = (all_trades > 0).mean()
    sharpe = (
        all_trades.mean() / all_trades.std() * np.sqrt(len(all_trades))
        if all_trades.std() > 0
        else 0
    )
    total_return = equity - 1

    return {
        "Trades": len(all_trades),
        "LongTrades": len(long_trades),
        "ShortTrades": len(short_trades),
        "WinRate": win_rate,
        "Sharpe": sharpe,
        "TotalReturn": total_return,
    }


def predict_with_fincast_torch(model, context, horizon, device, freq=0):
    """
    Generate predictions using FinCast PyTorch model directly.

    Parameters:
    -----------
    model : PatchedTimeSeriesDecoder_MOE
        The FinCast model
    context : np.ndarray
        Historical prices [context_length]
    horizon : int
        Prediction horizon
    device : torch.device
        Device to run on
    freq : int
        Frequency indicator (0=daily)

    Returns:
    --------
    float
        Predicted price at the horizon
    """
    model.eval()

    with torch.no_grad():
        # Prepare inputs - shape must be [B, C] where B=1, C=context_length
        context_tensor = torch.from_numpy(context).float().unsqueeze(0)  # [1, C]
        context_tensor = context_tensor.to(device)

        # Create padding tensor - zeros indicate no padding
        # Shape: [B, C] to match context
        padding = torch.zeros(1, len(context), dtype=torch.float32, device=device)

        # Create frequency tensor [B, 1]
        freq_tensor = torch.tensor([[freq]], dtype=torch.long, device=device)

        # Forward pass
        outputs, _ = model(context_tensor, padding, freq_tensor)

        # Extract prediction for the horizon
        # outputs shape: [B, N, H, 1+Q] where N=num_patches, H=horizon, Q=num_quantiles
        # We want the mean prediction (index 0) at the specific horizon
        # Use the last patch (-1) and the specific horizon index (horizon-1)
        prediction = outputs[0, -1, horizon-1, 0].cpu().item()

    return prediction

def load_finetuned_model(model_path, base_model_path, context_length=128, lora_r=8, lora_alpha=16, use_dora=False):
    """
    Load a fine-tuned FinCast model.

    Parameters:
    -----------
    model_path : str
        Path to fine-tuned model weights (.pth file)
    base_model_path : str
        Path to base FinCast model
    context_length : int
        Context length
    lora_r : int
        LoRA rank
    lora_alpha : int
        LoRA alpha
    use_dora : bool
        Whether to use DoRA

    Returns:
    --------
    model : PatchedTimeSeriesDecoder_MOE with PEFT
        Loaded fine-tuned model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base model
    print(f"Loading base model from: {base_model_path}")
    config = SimpleNamespace(
        backend="gpu" if torch.cuda.is_available() else "cpu",
        batch_size=32,
        context_len=context_length,
        horizon_len=max(ALL_HORIZONS),
    )

    model, _ = load_fincast_model_for_training(base_model_path, config)

    # Wrap with PEFT (same configuration as during training)
    print(f"Wrapping with PEFT (r={lora_r}, alpha={lora_alpha}, dora={use_dora})")
    model = wrap_model_with_peft(
        model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        use_dora=use_dora,
    )

    # Load fine-tuned weights
    print(f"Loading fine-tuned weights from: {model_path}")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print("✓ Model loaded successfully")
    return model


def run_inference_for_stock(
    stock,
    finetuned_model_path,
    base_model_path,
    data_dir="./dataset/training_data",
    output_dir="./results/baselines/finetuned_fincast",
    context_length=128,
    lora_r=8,
    lora_alpha=16,
    use_dora=False,
):
    """Run inference for a single stock using fine-tuned model."""

    print(f"\n{'='*80}")
    print(f"INFERENCE: {stock}")
    print(f"{'='*80}")

    # Load model
    model = load_finetuned_model(
        finetuned_model_path,
        base_model_path,
        context_length=context_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        use_dora=use_dora,
    )

    device = next(model.parameters()).device

    # Load data
    df = pd.read_csv(f"{data_dir}/{stock}_data_model_training.csv")
    df["__DateDT__"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["Year"] = df["__DateDT__"].dt.year

    test_df = df[df["Year"] == TEST_YEAR].sort_values("__DateDT__")
    test_prices = test_df["Close"].values

    print(f"Test samples: {len(test_prices)}")

    # Use all horizons with default thresholds
    print(f"Using all horizons with default thresholds")
    horizons = ALL_HORIZONS

    results = []

    for horizon in horizons:
        print(f"\nHorizon {horizon}...")
        threshold = DEFAULT_THRESHOLDS.get(horizon, 0.01)

        # Generate predictions for non-overlapping backtest
        # We need predictions at intervals of `horizon` days
        max_trades = len(test_prices) // horizon
        predictions = []
        actuals = []
        contexts_last = []
        pred_directions = []

        # Generate predictions at non-overlapping intervals
        for trade_idx in range(max_trades):
            i = trade_idx * horizon  # Start index for this trade
            if i + context_length + horizon > len(test_prices):
                break

            context = test_prices[i:i+context_length]
            actual = test_prices[i+context_length+horizon-1]

            # Predict
            pred = predict_with_fincast_torch(
                model, context, horizon, device, freq=0
            )

            predictions.append(pred)
            actuals.append(actual)
            contexts_last.append(context[-1])

            # Compute predicted return and direction
            pred_return = (pred - context[-1]) / (context[-1] + 1e-8)
            actual_return = (actual - context[-1]) / (context[-1] + 1e-8)
            pred_direction = 1 if pred_return > threshold else 0
            pred_directions.append(pred_direction)

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        contexts_last = np.array(contexts_last)
        pred_directions = np.array(pred_directions)

        # Compute returns for accuracy/AUC calculation
        pred_returns = (predictions - contexts_last) / (contexts_last + 1e-8)
        actual_returns = (actuals - contexts_last) / (contexts_last + 1e-8)

        # Direction predictions for metrics
        actual_directions = (actual_returns > threshold).astype(int)

        # Metrics
        accuracy = accuracy_score(actual_directions, pred_directions)
        try:
            auc = roc_auc_score(actual_directions, pred_returns)
        except ValueError:
            auc = 0.5

        # Non-overlapping backtest for trading metrics
        backtest_results = non_overlap_backtest(
            test_df, pred_directions, horizon
        )

        results.append({
            'Stock': stock,
            'Horizon': horizon,
            'Threshold': threshold,
            'Test_Accuracy': accuracy,
            'Test_ROC_AUC': auc,
            'Trades': backtest_results['Trades'],
            'LongTrades': backtest_results['LongTrades'],
            'ShortTrades': backtest_results['ShortTrades'],
            'WinRate': backtest_results['WinRate'],
            'Sharpe': backtest_results['Sharpe'],
            'TotalReturn': backtest_results['TotalReturn'],
        })

        print(f"  Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, "
              f"Trades: {backtest_results['Trades']}, Sharpe: {backtest_results['Sharpe']:.2f}")

    # Save results
    results_df = pd.DataFrame(results)
    output_file = f"{output_dir}/fincast_finetuned_{stock}_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="FinCast fine-tuned model inference")
    parser.add_argument("--stock", type=str, help="Stock symbol (required if not using --all-stocks)")
    parser.add_argument(
        "--all-stocks", action="store_true", help="Run inference for all stocks"
    )
    parser.add_argument(
        "--finetuned-model",
        type=str,
        help="Path to fine-tuned model (.pth file). If not provided, will look in output-dir/{stock}/best_model.pth",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="./FinCast-fts/model_weights/v1.pth",
        help="Path to base FinCast model",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./dataset/training_data",
        help="Directory containing stock data CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/baselines/finetuned_fincast",
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--context-length", type=int, default=128, help="Context length"
    )
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--use-dora", action="store_true", help="Use DoRA instead of LoRA")

    args = parser.parse_args()

    # Validate arguments
    if not args.all_stocks and not args.stock:
        parser.error("Either --stock or --all-stocks must be specified")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Determine stocks to process
    stocks = STOCKS if args.all_stocks else [args.stock]

    # Run inference for each stock
    all_results = []
    for stock in stocks:
        # Determine model path
        if args.finetuned_model:
            model_path = args.finetuned_model
        else:
            # Default: look for best_model.pth in output_dir/{stock}/
            model_path = f"{args.output_dir}/{stock}/best_model.pth"

        if not Path(model_path).exists():
            print(f"\n⚠️  Model not found for {stock}: {model_path}")
            print(f"Skipping {stock}...")
            continue

        results = run_inference_for_stock(
            stock=stock,
            finetuned_model_path=model_path,
            base_model_path=args.base_model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            context_length=args.context_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            use_dora=args.use_dora,
        )
        all_results.append(results)

    # Combine and save all results if processing multiple stocks
    if args.all_stocks and all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_file = f"{args.output_dir}/fincast_finetuned_all_results.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"\n✓ Combined results saved to: {combined_file}")

    print("\n" + "="*80)
    print("All inference complete!")
    print("="*80)


if __name__ == "__main__":
    main()
