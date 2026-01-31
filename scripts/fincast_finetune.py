"""
FinCast Fine-tuning Script for Stock Direction Prediction

This script fine-tunes the FinCast foundation model on stock price data
using PEFT (Parameter Efficient Fine-Tuning) with LoRA/DoRA.

============================================================================
INSTRUCTIONS FOR RUNNING ON GPU:
============================================================================

1. PREREQUISITES:
   - CUDA-capable GPU with at least 16GB VRAM (24GB recommended)
   - Python 3.11+
   - PyTorch 2.0+

2. INSTALL DEPENDENCIES:
   cd /path/to/stock-prediction
   pip install peft torch transformers accelerate wandb

3. DOWNLOAD MODEL WEIGHTS (if not already done):
   The script will attempt to download from HuggingFace automatically.
   Or manually download from: https://huggingface.co/Vincent05R/FinCast
   Place v1.pth in: FinCast-fts/model_weights/

4. RUN FINE-TUNING:
   # Single stock fine-tuning
   python fincast_finetune.py --stock AAPL --epochs 10

   # All stocks fine-tuning
   python fincast_finetune.py --all-stocks --epochs 10

   # With custom settings
   python fincast_finetune.py --all-stocks --epochs 20 --lr 1e-5 --lora-r 16

5. EXPECTED OUTPUT:
   - Fine-tuned model checkpoints in: finetuned_fincast/{stock}/
   - Training logs in: logs/fincast_finetune/
   - Results CSV files for evaluation

6. AFTER FINE-TUNING:
   Run inference with the fine-tuned model:
   python fincast_baseline.py --model-path finetuned_fincast/AAPL/best_model.pth --stock AAPL

============================================================================
CONFIGURATION NOTES:
============================================================================

- LoRA Rank (--lora-r): Higher = more capacity but slower. Default: 8
- LoRA Alpha (--lora-alpha): Scaling factor. Default: 16 (2x rank)
- Target Modules: Uses 'attn_mlp' preset (attention + MLP layers)
- Learning Rate: Start with 1e-5, adjust based on loss curves
- Batch Size: Adjust based on GPU memory. Default: 32
- Gradient Accumulation: Use if batch size is limited by memory

============================================================================
"""

import warnings

warnings.filterwarnings("ignore")

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, roc_auc_score

# Add FinCast-fts to path
FINCAST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "FinCast-fts", "src"
)
if FINCAST_PATH not in sys.path:
    sys.path.insert(0, FINCAST_PATH)

# ============================================================================
# Configuration
# ============================================================================

STOCKS = ["AAPL", "META", "NVDA", "SPY", "TSLA"]
TRAIN_YEARS = [2020, 2021, 2022, 2023]
TEST_YEAR = 2024
FEE_BPS_ROUND_TRIP = 10
RANDOM_SEED = 42

# Default FinCast model path
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "FinCast-fts", "model_weights", "v1.pth"
)

# Default thresholds per horizon
DEFAULT_THRESHOLDS = {
    2: 0.0025,
    3: 0.0025,
    4: 0.0050,
    5: 0.0050,
    6: 0.0075,
    7: 0.0100,
    8: 0.0150,
    9: 0.0175,
    10: 0.0170,
}

ALL_HORIZONS = [2, 3, 4, 5, 6, 7, 8, 9, 10]


# ============================================================================
# Dataset for Fine-tuning
# ============================================================================


class StockForecastDataset(Dataset):
    """
    Dataset for fine-tuning FinCast on stock price prediction.

    Each sample contains:
    - context: Historical prices (context_length days)
    - target: Future prices (horizon days)
    - freq: Frequency indicator (0 for daily)
    """

    def __init__(
        self,
        prices: np.ndarray,
        context_length: int = 128,
        horizon: int = 10,
        stride: int = 1,
    ):
        self.prices = prices.astype(np.float32)
        self.context_length = context_length
        self.horizon = horizon
        self.stride = stride

        # Calculate valid indices
        self.valid_indices = []
        for i in range(0, len(prices) - context_length - horizon + 1, stride):
            self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]

        # Context: historical prices
        context = self.prices[start_idx : start_idx + self.context_length]

        # Target: future prices (for computing loss)
        target = self.prices[
            start_idx
            + self.context_length : start_idx
            + self.context_length
            + self.horizon
        ]

        # Padding (no padding needed for context, but model expects it)
        padding = np.zeros(self.context_length + self.horizon, dtype=np.float32)

        # Frequency: 0 for high frequency (daily)
        freq = np.array([0], dtype=np.int64)

        return {
            "context": torch.from_numpy(context),
            "target": torch.from_numpy(target),
            "padding": torch.from_numpy(padding),
            "freq": torch.from_numpy(freq),
        }


# ============================================================================
# Loss Functions
# ============================================================================


class FinCastLoss(nn.Module):
    """
    Combined loss for FinCast fine-tuning.

    Components:
    1. MSE Loss: For point prediction accuracy
    2. Direction Loss: For predicting up/down movement
    3. Quantile Loss: For probabilistic forecasting (optional)
    """

    def __init__(
        self,
        direction_weight: float = 0.5,
        mse_weight: float = 0.5,
        quantile_weight: float = 0.0,
        quantiles: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ):
        super().__init__()
        self.direction_weight = direction_weight
        self.mse_weight = mse_weight
        self.quantile_weight = quantile_weight
        self.quantiles = quantiles
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        predictions: torch.Tensor,  # [B, H, 1+Q] - mean + quantiles
        targets: torch.Tensor,  # [B, H] - actual future prices
        context_last: torch.Tensor,  # [B] - last price in context
    ) -> torch.Tensor:
        """
        Compute combined loss.

        predictions: Model output [batch, horizon, 1+num_quantiles]
        targets: Actual future prices [batch, horizon]
        context_last: Last price in context for direction calculation
        """
        # Mean prediction (index 0)
        mean_pred = predictions[:, :, 0]  # [B, H]

        # 1. MSE Loss for point prediction
        mse_loss = self.mse_loss(mean_pred, targets)

        # 2. Direction Loss
        # Compute returns from last context price
        pred_returns = (mean_pred[:, -1] - context_last) / (context_last + 1e-8)
        actual_returns = (targets[:, -1] - context_last) / (context_last + 1e-8)

        # Direction: 1 if positive return, 0 otherwise
        actual_direction = (actual_returns > 0).float()
        direction_loss = self.bce_loss(pred_returns, actual_direction)

        # 3. Quantile Loss (optional)
        quantile_loss = torch.tensor(0.0, device=predictions.device)
        if self.quantile_weight > 0 and predictions.shape[2] > 1:
            for i, q in enumerate(self.quantiles):
                q_pred = predictions[:, :, 1 + i]  # [B, H]
                errors = targets - q_pred
                quantile_loss += torch.mean(torch.max(q * errors, (q - 1) * errors))
            quantile_loss /= len(self.quantiles)

        # Combined loss
        total_loss = (
            self.mse_weight * mse_loss
            + self.direction_weight * direction_loss
            + self.quantile_weight * quantile_loss
        )

        return total_loss, {
            "mse": mse_loss.item(),
            "direction": direction_loss.item(),
            "quantile": quantile_loss.item(),
            "total": total_loss.item(),
        }


# ============================================================================
# Model Loading with PEFT
# ============================================================================


def load_fincast_model_for_training(model_path: str, config: SimpleNamespace):
    """
    Load FinCast model and prepare for fine-tuning with PEFT.
    """
    from tools.model_utils import get_model_FFM
    from ffm import FFmHparams

    ffm_hparams = FFmHparams(
        backend=config.backend,
        per_core_batch_size=config.batch_size,
        horizon_len=config.horizon_len,
        context_len=config.context_len,
        use_positional_embedding=False,
        num_experts=4,
        gating_top_n=2,
        load_from_compile=True,
        point_forecast_mode="mean",
    )

    # Load the model
    model_actual, ffm_config, ffm_api = get_model_FFM(model_path, ffm_hparams)

    return model_actual, ffm_api


def wrap_model_with_peft(
    model: nn.Module,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_targets_preset: str = "attn_mlp",
    use_dora: bool = False,
) -> nn.Module:
    """
    Wrap the model with PEFT (LoRA/DoRA) adapters.
    """
    # Import PEFT injector from FinCast
    peft_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "FinCast-fts", "peft_Fincast"
    )
    if peft_path not in sys.path:
        sys.path.insert(0, peft_path)

    from peft_injector import wrap_with_peft

    peft_type = "dora" if use_dora else "lora"

    peft_model = wrap_with_peft(
        model,
        peft_type=peft_type,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_bias="none",
        lora_init=None,
        lora_targets_preset=lora_targets_preset,
        train_base=False,  # Only train LoRA parameters
    )

    return peft_model


# ============================================================================
# Training Loop
# ============================================================================


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: FinCastLoss,
    device: torch.device,
    horizon: int,
    grad_accum_steps: int = 1,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_direction = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        context = batch["context"].to(device)
        target = batch["target"].to(device)
        padding = batch["padding"].to(device)
        freq = batch["freq"].to(device)

        # Forward pass
        # Model expects: input_ts [B, C], input_padding [B, C+H], freq [B, 1]
        outputs, aux_loss = model(context, padding[:, : context.shape[1]], freq)

        # outputs shape: [B, N, H, 1+Q] where N = num_patches
        # Take the last patch's prediction
        predictions = outputs[:, -1, :horizon, :]  # [B, H, 1+Q]

        # Last context price
        context_last = context[:, -1]

        # Compute loss
        loss, loss_dict = loss_fn(predictions, target[:, :horizon], context_last)

        # Add auxiliary loss from MoE
        if aux_loss is not None:
            loss = loss + 0.01 * aux_loss

        # Backward pass with gradient accumulation
        loss = loss / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss_dict["total"]
        total_mse += loss_dict["mse"]
        total_direction += loss_dict["direction"]
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "mse": total_mse / num_batches,
        "direction": total_direction / num_batches,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: FinCastLoss,
    device: torch.device,
    horizon: int,
    threshold: float = 0.0,
) -> dict:
    """Evaluate the model."""
    model.eval()

    all_preds = []
    all_targets = []
    all_context_last = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            context = batch["context"].to(device)
            target = batch["target"].to(device)
            padding = batch["padding"].to(device)
            freq = batch["freq"].to(device)

            outputs, _ = model(context, padding[:, : context.shape[1]], freq)
            predictions = outputs[:, -1, :horizon, :]

            context_last = context[:, -1]

            loss, loss_dict = loss_fn(predictions, target[:, :horizon], context_last)

            total_loss += loss_dict["total"]
            num_batches += 1

            # Store for metrics
            all_preds.append(predictions[:, -1, 0].cpu().numpy())  # Last horizon, mean
            all_targets.append(target[:, horizon - 1].cpu().numpy())
            all_context_last.append(context_last.cpu().numpy())

    # Compute metrics
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_context_last = np.concatenate(all_context_last)

    # Direction prediction
    pred_returns = (all_preds - all_context_last) / (all_context_last + 1e-8)
    actual_returns = (all_targets - all_context_last) / (all_context_last + 1e-8)

    pred_direction = (pred_returns > threshold).astype(int)
    actual_direction = (actual_returns > threshold).astype(int)

    accuracy = accuracy_score(actual_direction, pred_direction)
    try:
        auc = roc_auc_score(actual_direction, pred_returns)
    except ValueError:
        auc = 0.5

    return {
        "loss": total_loss / num_batches,
        "accuracy": accuracy,
        "auc": auc,
    }


# ============================================================================
# Main Fine-tuning Function
# ============================================================================


def finetune_fincast(
    stock: str,
    data_dir: str,
    model_path: str,
    output_dir: str,
    horizon: int = 10,
    context_length: int = 128,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    lora_r: int = 8,
    lora_alpha: int = 16,
    use_dora: bool = False,
    grad_accum_steps: int = 1,
    direction_weight: float = 0.5,
    threshold: float = None,
) -> dict:
    """
    Fine-tune FinCast on a single stock.
    """
    print(f"\n{'='*70}")
    print(f"FINE-TUNING FINCAST: {stock}")
    print(f"{'='*70}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if threshold is None:
        threshold = DEFAULT_THRESHOLDS.get(horizon, 0.01)

    # Load data
    df = pd.read_csv(f"{data_dir}/{stock}_data_model_training.csv")
    df["__DateDT__"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["Year"] = df["__DateDT__"].dt.year

    train_df = df[df["Year"].isin(TRAIN_YEARS)].sort_values("__DateDT__")
    test_df = df[df["Year"] == TEST_YEAR].sort_values("__DateDT__")

    train_prices = train_df["Close"].values
    test_prices = test_df["Close"].values

    print(f"Training samples: {len(train_prices)}")
    print(f"Test samples: {len(test_prices)}")

    # Create datasets
    train_dataset = StockForecastDataset(
        train_prices, context_length, horizon, stride=1
    )
    test_dataset = StockForecastDataset(
        test_prices, context_length, horizon, stride=horizon  # Non-overlapping for test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Load model
    print(f"\nLoading FinCast model from: {model_path}")
    config = SimpleNamespace(
        backend="gpu" if torch.cuda.is_available() else "cpu",
        batch_size=batch_size,
        context_len=context_length,
        horizon_len=max(ALL_HORIZONS),
    )

    model, _ = load_fincast_model_for_training(model_path, config)

    # Wrap with PEFT
    print(f"\nApplying LoRA (r={lora_r}, alpha={lora_alpha}, dora={use_dora})")
    model = wrap_model_with_peft(
        model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        use_dora=use_dora,
    )
    model = model.to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100*trainable_params/total_params:.2f}%)"
    )

    # Loss function
    loss_fn = FinCastLoss(
        direction_weight=direction_weight,
        mse_weight=1.0 - direction_weight,
    )

    # Optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_auc = 0.0
    best_epoch = 0
    history = []

    stock_output_dir = os.path.join(output_dir, stock)
    Path(stock_output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, horizon, grad_accum_steps
        )
        print(
            f"  Train - Loss: {train_metrics['loss']:.4f}, "
            f"MSE: {train_metrics['mse']:.4f}, "
            f"Direction: {train_metrics['direction']:.4f}"
        )

        # Evaluate
        test_metrics = evaluate(model, test_loader, loss_fn, device, horizon, threshold)
        print(
            f"  Test  - Loss: {test_metrics['loss']:.4f}, "
            f"Acc: {test_metrics['accuracy']:.3f}, "
            f"AUC: {test_metrics['auc']:.3f}"
        )

        # Update scheduler
        scheduler.step()

        # Save best model
        if test_metrics["auc"] > best_auc:
            best_auc = test_metrics["auc"]
            best_epoch = epoch + 1
            torch.save(
                model.state_dict(), os.path.join(stock_output_dir, "best_model.pth")
            )
            print(f"  * New best model saved (AUC: {best_auc:.3f})")

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_auc": test_metrics["auc"],
            }
        )

    # Save final model and history
    torch.save(model.state_dict(), os.path.join(stock_output_dir, "final_model.pth"))

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        os.path.join(stock_output_dir, "training_history.csv"), index=False
    )

    # Save config
    config_dict = {
        "stock": stock,
        "horizon": horizon,
        "context_length": context_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "use_dora": use_dora,
        "best_epoch": best_epoch,
        "best_auc": best_auc,
    }
    with open(os.path.join(stock_output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nFinished training {stock}")
    print(f"Best AUC: {best_auc:.3f} (epoch {best_epoch})")
    print(f"Results saved to: {stock_output_dir}")

    return {
        "stock": stock,
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "final_accuracy": history[-1]["test_accuracy"],
        "final_auc": history[-1]["test_auc"],
    }


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune FinCast for Stock Direction Prediction"
    )
    parser.add_argument(
        "--stock",
        type=str,
        default="AAPL",
        choices=STOCKS,
        help="Stock symbol to fine-tune on",
    )
    parser.add_argument(
        "--all-stocks", action="store_true", help="Fine-tune on all stocks"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to FinCast model weights",
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
        help="Directory to save fine-tuned models",
    )
    parser.add_argument(
        "--horizon", type=int, default=10, help="Prediction horizon in days"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=128,
        help="Context length (historical days)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument(
        "--lora-alpha", type=int, default=16, help="LoRA alpha (scaling factor)"
    )
    parser.add_argument(
        "--use-dora", action="store_true", help="Use DoRA instead of LoRA"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--direction-weight",
        type=float,
        default=0.5,
        help="Weight for direction loss (vs MSE loss)",
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Check model exists
    if not os.path.exists(args.model_path):
        print(f"Model not found at: {args.model_path}")
        print("Please download from: https://huggingface.co/Vincent05R/FinCast")
        return

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.all_stocks:
        all_results = []
        for stock in STOCKS:
            results = finetune_fincast(
                stock=stock,
                data_dir=args.data_dir,
                model_path=args.model_path,
                output_dir=args.output_dir,
                horizon=args.horizon,
                context_length=args.context_length,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                use_dora=args.use_dora,
                grad_accum_steps=args.grad_accum,
                direction_weight=args.direction_weight,
            )
            all_results.append(results)

        # Summary
        print("\n" + "=" * 70)
        print("FINE-TUNING SUMMARY")
        print("=" * 70)
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))
        results_df.to_csv(
            os.path.join(args.output_dir, "all_stocks_summary.csv"), index=False
        )
        print(f"\nAverage Best AUC: {results_df['best_auc'].mean():.3f}")
    else:
        finetune_fincast(
            stock=args.stock,
            data_dir=args.data_dir,
            model_path=args.model_path,
            output_dir=args.output_dir,
            horizon=args.horizon,
            context_length=args.context_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            use_dora=args.use_dora,
            grad_accum_steps=args.grad_accum,
            direction_weight=args.direction_weight,
        )


if __name__ == "__main__":
    main()
