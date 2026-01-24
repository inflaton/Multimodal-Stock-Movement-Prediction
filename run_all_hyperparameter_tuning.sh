#!/bin/bash

# Run hyperparameter tuning for all models and all stocks
# Usage: ./run_all_hyperparameter_tuning.sh [--stock STOCK] [--model MODEL] [--skip-lstm] [--only-lstm]

set -e  # Exit on error

# Parse command line arguments
SINGLE_STOCK=""
SINGLE_MODEL=""
SKIP_LSTM=false
ONLY_LSTM=false
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
        --skip-lstm)
            SKIP_LSTM=true
            shift
            ;;
        --only-lstm)
            ONLY_LSTM=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--stock STOCK] [--model MODEL] [--skip-lstm] [--only-lstm]"
            echo "  --stock STOCK   Train only the specified stock (e.g., AAPL)"
            echo "  --model MODEL   Train only the specified model for all stocks"
            echo "                  Available models: lstm, xgboost, lightgbm, random_forest,"
            echo "                  logistic_regression, svm, gradient_boosting"
            echo "  --skip-lstm     Skip LSTM model (faster training)"
            echo "  --only-lstm     Run only LSTM model (skip other ML models)"
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

# Define models to train
if [ -n "$SINGLE_MODEL" ]; then
    MODELS=("$SINGLE_MODEL")
    echo "Training single model: $SINGLE_MODEL"
elif [ "$ONLY_LSTM" = true ]; then
    MODELS=("lstm")
    echo "Training only LSTM model"
elif [ "$SKIP_LSTM" = true ]; then
    MODELS=(
        "xgboost"
        "lightgbm"
        "random_forest"
        "logistic_regression"
        "gradient_boosting"
        "svm"
    )
    echo "Training all models (excluding LSTM)"
else
    MODELS=(
        "xgboost"
        "lightgbm"
        "random_forest"
        "logistic_regression"
        "gradient_boosting"
        "svm"
        "lstm"
    )
    echo "Training all models (including LSTM)"
fi

echo "========================================="
echo "Starting Hyperparameter Tuning"
echo "========================================="
echo "Stocks: ${STOCKS[*]}"
echo "Models: ${MODELS[*]}"
if [ "$SKIP_LSTM" = true ]; then
    echo "LSTM: Skipped (--skip-lstm)"
elif [ "$ONLY_LSTM" = true ]; then
    echo "Mode: LSTM only (--only-lstm)"
fi
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

