#!/bin/bash

# Run foundation model baselines (Chronos-2)
# Usage: ./run_baselines_chronos-2.sh

set -e  # Exit on error

echo "========================================="
echo "Running Chronos-2 Foundation Model Baselines"
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

echo "========================================="
echo "All Chronos-2 Baselines Complete!"
echo "========================================="

