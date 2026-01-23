# Usage Guide

This document provides detailed instructions for using the codebase to reproduce our results and extend the analysis to new stocks or time periods.

## Table of Contents

1. [Setup](#setup)
2. [Data Preparation](#data-preparation)
3. [Training Models](#training-models)
4. [Evaluation and Analysis](#evaluation-and-analysis)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

## Setup

### Environment Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### GPU Setup (Optional but Recommended)

For faster LSTM training:

```bash
# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import tensorflow as tf; print('GPU devices:', tf.config.list_physical_devices('GPU'))"
```

## Data Preparation

### Using Provided Datasets

The repository includes preprocessed datasets for five stocks (AAPL, META, NVDA, SPY, TSLA) in the `dataset/training_data/` directory. Raw sentiment data is available in `dataset/sentiment/`. These are ready to use for training.

### Data Format

Each CSV file contains the following columns:

- **Date**: Trading date (format: DD/MM/YYYY)
- **Close**: Closing price
- **Weighted Sentiment Score**: Reliability-filtered weighted sentiment (70% news, 30% social media)
- **Combi 1-10**: Technical indicator combinations
- **Combi 1-10 Days**: Days since last signal for each combination

### Generating New Data

If you want to collect data for new stocks or time periods:

1. Collect raw data:
   - Stock prices from Yahoo Finance
   - Financial news from MarketWatch, Google News
   - Social media from Reddit, StockTwits, Twitter/X

2. Process sentiment:
   ```bash
   # See notebook for detailed steps
   jupyter notebook notebooks/02_generate_training_data.ipynb
   ```

3. Merge with technical indicators and save to `dataset/training_data/` directory

## Training Models

### Quick Start: Train a Single Model

Train XGBoost for NVDA stock:

```bash
python scripts/xgboost_hyperparameter_tuning.py --stock NVDA
```

### Train All Models for a Stock

```bash
# Define stock
STOCK="NVDA"

# Train all traditional ML models
python scripts/xgboost_hyperparameter_tuning.py --stock $STOCK
python scripts/lightgbm_hyperparameter_tuning.py --stock $STOCK
python scripts/random_forest_hyperparameter_tuning.py --stock $STOCK
python scripts/logistic_regression_hyperparameter_tuning.py --stock $STOCK
python scripts/svm_hyperparameter_tuning.py --stock $STOCK
python scripts/gradient_boosting_hyperparameter_tuning.py --stock $STOCK

# Train LSTM (requires GPU for reasonable training time)
python scripts/lstm_hyperparameter_tuning.py --stock $STOCK
```

### Train for All Stocks

```bash
# Create a simple script
for STOCK in AAPL META NVDA SPY TSLA; do
    echo "Training XGBoost for $STOCK"
    python scripts/xgboost_hyperparameter_tuning.py --stock $STOCK
done
```

### Hyperparameter Tuning Options

Each training script accepts the following common arguments:

```bash
python scripts/xgboost_hyperparameter_tuning.py \
    --stock NVDA \                    # Stock symbol
    --data-dir data \                 # Data directory
    --output-dir results \            # Output directory
    --n-trials 100 \                  # Number of Optuna trials
    --random-seed 42                  # Random seed for reproducibility
```

### Foundation Model Baselines

#### Chronos-2

**Zero-shot (Price Only):**
```bash
python scripts/chronos_baseline.py --all-stocks
```

**Zero-shot (With Sentiment):**
```bash
python scripts/chronos_baseline.py --all-stocks --use-covariates
```

**Fine-tuned (Price Only):**
```bash
python scripts/chronos_finetune.py --all-stocks
```

**Fine-tuned (With Sentiment):**
```bash
python scripts/chronos_finetune.py --all-stocks --use-covariates
```

**Custom Fine-tuning:**
```bash
python scripts/chronos_finetune.py \
    --stock NVDA \
    --use-covariates \
    --fine-tune-lr 0.0001 \
    --fine-tune-steps 1000 \
    --context-length 60
```

#### FinCast

**Zero-shot:**
```bash
python scripts/fincast_baseline.py --all-stocks
```

**Fine-tuned:**
```bash
python scripts/fincast_finetune.py --all-stocks --use-covariates
```

## Evaluation and Analysis

### Analyzing Results

Results are automatically saved to the `results/` directory with naming convention:
- `{MODEL}_{STOCK}_results.csv`
- `Filtered_{STOCK}_hyperparameter_tuned_results.csv`

### Using Jupyter Notebooks

#### 1. Baseline Results Analysis

```bash
jupyter notebook notebooks/08_baseline_results.ipynb
```

This notebook:
- Compares zero-shot vs fine-tuned performance
- Analyzes Chronos-2 and FinCast results
- Generates comparison tables and visualizations

#### 2. Tuned Models Analysis

```bash
jupyter notebook notebooks/09_tuned_models_results.ipynb
```

This notebook:
- Analyzes all task-specific models
- Compares ROC-AUC vs Sharpe selection criteria
- Generates performance heatmaps
- Creates model comparison tables

#### 3. Ablation Study

```bash
jupyter notebook notebooks/10_ablation_study.ipynb
```

This notebook:
- Analyzes sentiment aggregation strategies
- Compares feature combinations
- Evaluates individual feature contributions
- Tests different weighting schemes

### Generating Paper Tables

Update SOTA comparison table:

```bash
python scripts/update_sota_table.py
```

## Advanced Usage

### Custom Hyperparameter Search Space

Edit the hyperparameter ranges in the training scripts. For example, in `xgboost_hyperparameter_tuning.py`:

```python
def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        # Add or modify parameters here
    }
```

### Custom Trading Strategies

Modify the `non_overlap_backtest()` function to implement different trading strategies:

```python
def non_overlap_backtest(
    test_df,
    predictions,
    horizon,
    fee_bps=10,
    strategy='long_short',  # Options: 'long_short', 'long_only', 'confidence'
    confidence_threshold=0.6
):
    # Implementation
```

### Custom Sentiment Aggregation

Modify sentiment weighting in `generate_training_data.py`:

```python
# Default: 70% news, 30% social media
NEWS_WEIGHT = 0.7
SOCIAL_WEIGHT = 0.3

# Custom weights
NEWS_WEIGHT = 0.8  # Increase news weight
SOCIAL_WEIGHT = 0.2
```

### Adding New Stocks

1. Collect data for the new stock following the data format
2. Save as `dataset/training_data/{STOCK}_data_model_training.csv`
3. Update stock list in training scripts:

```python
STOCKS = ["AAPL", "META", "NVDA", "SPY", "TSLA", "NEW_STOCK"]
```

4. Run training for the new stock:

```bash
python scripts/xgboost_hyperparameter_tuning.py --stock NEW_STOCK
```

### Parallel Training

Train multiple models in parallel using GNU Parallel:

```bash
# Install GNU Parallel
# Ubuntu/Debian: sudo apt-get install parallel
# macOS: brew install parallel

# Train multiple stocks in parallel
parallel python scripts/xgboost_hyperparameter_tuning.py --stock {} ::: AAPL META NVDA SPY TSLA
```

### Batch Processing

Create a shell script for batch processing:

```bash
#!/bin/bash
# run_all.sh

MODELS=("xgboost" "lightgbm" "random_forest")
STOCKS=("AAPL" "META" "NVDA" "SPY" "TSLA")

for MODEL in "${MODELS[@]}"; do
    for STOCK in "${STOCKS[@]}"; do
        echo "Training $MODEL for $STOCK"
        python scripts/${MODEL}_hyperparameter_tuning.py --stock $STOCK
    done
done
```

Run with:
```bash
chmod +x run_all.sh
./run_all.sh
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

If you encounter GPU memory errors during LSTM training:

```python
# In lstm_hyperparameter_tuning.py, reduce batch size
batch_size = trial.suggest_int('batch_size', 16, 32)  # Reduced from 64

# Or reduce sequence length
sequence_length = trial.suggest_int('sequence_length', 10, 30)  # Reduced from 60
```

#### 2. Optuna Database Lock

If Optuna throws database lock errors:

```bash
# Use different storage for each stock
python scripts/xgboost_hyperparameter_tuning.py \
    --stock NVDA \
    --storage sqlite:///optuna_nvda.db
```

#### 3. Date Parsing Errors

If you encounter date parsing errors:

```python
# Check date format in your CSV
df["__DateDT__"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")

# Or let pandas auto-detect
df["__DateDT__"] = pd.to_datetime(df["Date"], errors="coerce")
```

#### 4. Missing Dependencies

If modules are not found:

```bash
# Update pip
pip install --upgrade pip

# Reinstall requirements
pip install -r requirements.txt --upgrade

# Check installation
pip list | grep -i tensorflow
pip list | grep -i torch
```

#### 5. Slow Training

To speed up training:

1. Reduce number of Optuna trials:
   ```bash
   python scripts/xgboost_hyperparameter_tuning.py --stock NVDA --n-trials 50
   ```

2. Use GPU for LSTM:
   ```bash
   # Verify GPU is being used
   nvidia-smi
   ```

3. Reduce data size for quick testing:
   ```python
   # In training script
   train_df = train_df.sample(frac=0.5)  # Use 50% of data
   ```

### Performance Optimization

#### 1. Use Multiple CPUs

For tree-based models:

```python
# In xgboost_hyperparameter_tuning.py
params = {
    'n_jobs': -1,  # Use all CPUs
    'tree_method': 'hist',  # Faster histogram method
}
```

#### 2. Memory-Efficient Loading

For large datasets:

```python
# Load in chunks
chunks = pd.read_csv('dataset/training_data/STOCK_data.csv', chunksize=1000)
df = pd.concat(chunks)
```

#### 3. Caching Results

Save intermediate results:

```python
import joblib

# Save best hyperparameters
joblib.dump(best_params, 'best_hyperparameters_STOCK.pkl')

# Load and reuse
best_params = joblib.load('best_hyperparameters_STOCK.pkl')
```

### Getting Help

1. Check existing issues on GitHub
2. Review the paper for methodology details
3. Contact the authors (details provided upon paper acceptance)

## Next Steps

- Explore different model architectures
- Test alternative sentiment sources
- Extend to intraday prediction
- Implement portfolio optimization
- Add risk management strategies

For more details, refer to the paper and inline code documentation.
