#!/bin/bash

# Run foundation model baselines (Chronos-2 and FinCast)
# Usage: ./run_baselines.sh

set -e  # Exit on error

echo "========================================="
echo "Running Foundation Model Baselines"
echo "========================================="
echo ""

# Chronos-2 Zero-shot
echo "-----------------------------------------"
echo "1. Chronos-2 Zero-shot (Price Only)"
echo "-----------------------------------------"
python scripts/chronos_baseline.py --all-stocks
echo "✓ Completed"
echo ""

echo "-----------------------------------------"
echo "2. Chronos-2 Zero-shot (With Sentiment)"
echo "-----------------------------------------"
python scripts/chronos_baseline.py --all-stocks --use-covariates
echo "✓ Completed"
echo ""

# Chronos-2 Fine-tuned
echo "-----------------------------------------"
echo "3. Chronos-2 Fine-tuned (Price Only)"
echo "-----------------------------------------"
python scripts/chronos_finetune.py --all-stocks
echo "✓ Completed"
echo ""

echo "-----------------------------------------"
echo "4. Chronos-2 Fine-tuned (With Sentiment)"
echo "-----------------------------------------"
python scripts/chronos_finetune.py --all-stocks --use-covariates
echo "✓ Completed"
echo ""

# FinCast Zero-shot
echo "-----------------------------------------"
echo "5. FinCast Zero-shot (Price Only)"
echo "-----------------------------------------"
python scripts/fincast_baseline.py --all-stocks
echo "✓ Completed"
echo ""

# FinCast Fine-tuned
echo "-----------------------------------------"
echo "6. FinCast Fine-tuned (With Sentiment)"
echo "-----------------------------------------"
python scripts/fincast_finetune.py --all-stocks --use-covariates
echo "✓ Completed"
echo ""

echo "========================================="
echo "All Baselines Complete!"
echo "========================================="
echo ""
echo "Results saved to the respective output directories."
echo ""
echo "To analyze baseline results, use:"
echo "  jupyter notebook notebooks/08_baseline_results.ipynb"
