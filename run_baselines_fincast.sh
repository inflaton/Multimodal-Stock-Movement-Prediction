#!/bin/bash

# Run foundation model baselines (FinCast)
# Usage: ./run_baselines_fincast.sh

set -e  # Exit on error

# Force CPU execution if GPU is incompatible
# NVIDIA GB10 (sm_121) is not supported by PyTorch 2.5.0
# Set this to empty string to use GPU if compatible
export CUDA_VISIBLE_DEVICES=""

echo "========================================="
echo "Running Foundation Model Baselines (CPU)"
echo "========================================="
echo ""

# FinCast Zero-shot
echo "-----------------------------------------"
echo "5. FinCast Zero-shot (Price Only)"
echo "-----------------------------------------"
python scripts/fincast_baseline.py --all-stocks
echo "✓ Completed"
echo ""

# FinCast Fine-tuning
echo "-----------------------------------------"
echo "6. FinCast Fine-tuning (Price Only)"
echo "-----------------------------------------"
python scripts/fincast_finetune.py --all-stocks
echo "✓ Completed"
echo ""

# FinCast Fine-tuned Inference
echo "-----------------------------------------"
echo "7. FinCast Fine-tuned Inference"
echo "-----------------------------------------"
python scripts/fincast_finetuned_inference.py --all-stocks
echo "✓ Completed"
echo ""

echo "========================================="
echo "All Baselines Complete!"
echo "========================================="

