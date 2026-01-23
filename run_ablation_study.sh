#!/bin/bash
#
# Run ablation study for all configurations
#
# Usage:
#   ./run_ablation_study.sh                    # Run all configs with default settings
#   ./run_ablation_study.sh --n-calls 30       # More optimization iterations
#   ./run_ablation_study.sh --config technical_only  # Run specific config only
#   ./run_ablation_study.sh --stock NVDA       # Run for specific stock only
#   ./run_ablation_study.sh --skip-lstm        # Skip LSTM (faster)
#   ./run_ablation_study.sh --only-lstm        # Run only LSTM (skip other ML models)
#   ./run_ablation_study.sh --force-cpu        # Force CPU mode (disable CuDNN GPU)
#
# Ablation configurations:
#   - technical_only: Technical indicators only (no sentiment)
#   - sentiment_only: Sentiment features only (no technical)
#   - equal_weights: Equal weighting (50:50 news/social)
#   - news_only: News sentiment only
#   - social_only: Social media sentiment only
#
# Note: Full Model (Tech + Sent 70:30) results are from main experiments in v1/

set -e  # Exit on error

# Default settings (relative paths from ablation/ directory)
DATA_DIR="../data"
SENTIMENT_DIR="../../dataset/news & social media/finbert sentiment"
OUTPUT_DIR="."
N_CALLS=30
CONFIG=""
STOCK=""
SKIP_LSTM=false
ONLY_LSTM=false
FORCE_CPU=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --sentiment-dir)
            SENTIMENT_DIR="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --n-calls)
            N_CALLS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --stock)
            STOCK="$2"
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
        --force-cpu)
            FORCE_CPU=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================================="
echo "Ablation Study - Stock Prediction"
echo "=============================================================="
echo "Data directory: $DATA_DIR"
echo "Sentiment directory: $SENTIMENT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "N-calls: $N_CALLS"
echo "Skip LSTM: $SKIP_LSTM"
echo "Only LSTM: $ONLY_LSTM"
echo "Force CPU: $FORCE_CPU"
if [ -n "$STOCK" ]; then
    echo "Stock: $STOCK"
else
    echo "Stock: ALL stocks"
fi
if [ -n "$CONFIG" ]; then
    echo "Running config: $CONFIG"
else
    echo "Running: ALL configurations"
fi
echo "=============================================================="
echo ""

# Common arguments
if [ -n "$STOCK" ]; then
    COMMON_ARGS="--stock $STOCK --data-dir $DATA_DIR --sentiment-dir \"$SENTIMENT_DIR\" --output-dir $OUTPUT_DIR --n-calls $N_CALLS"
else
    COMMON_ARGS="--all-stocks --data-dir $DATA_DIR --sentiment-dir \"$SENTIMENT_DIR\" --output-dir $OUTPUT_DIR --n-calls $N_CALLS"
fi

# Add skip-lstm flag if set
if [ "$SKIP_LSTM" = true ]; then
    COMMON_ARGS="$COMMON_ARGS --skip-lstm"
fi

# Add only-lstm flag if set
if [ "$ONLY_LSTM" = true ]; then
    COMMON_ARGS="$COMMON_ARGS --only-lstm"
fi

# Add force-cpu flag if set
if [ "$FORCE_CPU" = true ]; then
    COMMON_ARGS="$COMMON_ARGS --force-cpu"
fi

if [ -n "$CONFIG" ]; then
    # Run specific configuration
    echo ""
    echo "=============================================================="
    echo "Running ablation: $CONFIG"
    echo "=============================================================="
    eval python "$SCRIPT_DIR/ablation_study.py" --config "$CONFIG" $COMMON_ARGS
else
    # Run all configurations
    echo ""
    echo "=============================================================="
    echo "Running ablation: technical_only"
    echo "=============================================================="
    eval python "$SCRIPT_DIR/ablation_study.py" --config technical_only $COMMON_ARGS

    echo ""
    echo "=============================================================="
    echo "Running ablation: sentiment_only"
    echo "=============================================================="
    eval python "$SCRIPT_DIR/ablation_study.py" --config sentiment_only $COMMON_ARGS

    echo ""
    echo "=============================================================="
    echo "Running ablation: equal_weights"
    echo "=============================================================="
    eval python "$SCRIPT_DIR/ablation_study.py" --config equal_weights $COMMON_ARGS

    echo ""
    echo "=============================================================="
    echo "Running ablation: news_only"
    echo "=============================================================="
    eval python "$SCRIPT_DIR/ablation_study.py" --config news_only $COMMON_ARGS

    echo ""
    echo "=============================================================="
    echo "Running ablation: social_only"
    echo "=============================================================="
    eval python "$SCRIPT_DIR/ablation_study.py" --config social_only $COMMON_ARGS
fi

echo ""
echo "=============================================================="
echo "Ablation study complete!"
echo "=============================================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "  (no CSV files yet)"
echo ""
