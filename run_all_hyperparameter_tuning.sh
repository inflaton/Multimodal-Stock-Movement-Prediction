#!/bin/bash

# Run hyperparameter tuning for all models and all stocks
# Usage: ./run_all_hyperparameter_tuning.sh [--stock STOCK] [--model MODEL]

set -e  # Exit on error

# Parse command line arguments
SINGLE_STOCK=""
SINGLE_MODEL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --stock)
            SINGLE_STOCK="$2"
            shift 2
            ;;
        --model)
            SINGLE_MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--stock STOCK] [--model MODEL]"
            echo "  --stock STOCK  Train only the specified stock (e.g., AAPL)"
            echo "  --model MODEL  Train only the specified model for all stocks"
            echo "                 Available models: xgboost, lightgbm, random_forest,"
            echo "                 logistic_regression, svm, gradient_boosting"
            exit 1
            ;;
    esac
done

# Define stocks to train
if [ -n "$SINGLE_STOCK" ]; then
    STOCKS=("$SINGLE_STOCK")
    echo "Training single stock: $SINGLE_STOCK"
else
    STOCKS=("AAPL" "META" "NVDA" "SPY" "TSLA")
    echo "Training all stocks: ${STOCKS[*]}"
fi

# Define models to train (excluding LSTM which takes much longer)
if [ -n "$SINGLE_MODEL" ]; then
    MODELS=("$SINGLE_MODEL")
    echo "Training single model: $SINGLE_MODEL"
else
    MODELS=(
        "xgboost"
        "lightgbm"
        "random_forest"
        "logistic_regression"
        "svm"
        "gradient_boosting"
    )
fi

echo "========================================="
echo "Starting Hyperparameter Tuning"
echo "========================================="
echo "Stocks: ${STOCKS[*]}"
echo "Models: ${MODELS[*]}"
echo "========================================="

# Train each model for each stock
for STOCK in "${STOCKS[@]}"; do
    echo ""
    echo "========================================="
    echo "Training Stock: $STOCK"
    echo "========================================="

    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "-----------------------------------------"
        echo "Training $MODEL for $STOCK"
        echo "-----------------------------------------"

        python scripts/${MODEL}_hyperparameter_tuning.py \
            --stock "$STOCK" \
            --data-dir dataset/training_data \
            --output-dir results/ours

        if [ $? -eq 0 ]; then
            echo "✓ $MODEL training completed for $STOCK"
        else
            echo "✗ $MODEL training failed for $STOCK"
        fi
    done

    echo ""
    echo "========================================="
    echo "Completed training for $STOCK"
    echo "========================================="
done

echo ""
echo "========================================="
echo "All Training Complete!"
echo "========================================="

