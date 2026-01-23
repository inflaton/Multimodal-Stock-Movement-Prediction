# Dataset

This directory contains all datasets used for training and evaluation.

## Directory Structure

```
dataset/
├── training_data/         # Final preprocessed training datasets
│   ├── AAPL_data_model_training.csv
│   ├── META_data_model_training.csv
│   ├── NVDA_data_model_training.csv
│   ├── SPY_data_model_training.csv
│   └── TSLA_data_model_training.csv
│
└── sentiment/            # Filtered sentiment analysis results
    ├── news_sentiment_finbert_tone_weighted_aapl.csv
    ├── news_sentiment_finbert_tone_weighted_meta.csv
    ├── news_sentiment_finbert_tone_weighted_nvda.csv
    ├── news_sentiment_finbert_tone_weighted_spy.csv
    └── news_sentiment_finbert_tone_weighted_tsla.csv
```

## Dataset Description

### Training Data (`training_data/`)

These are the final preprocessed datasets ready for model training. Each file contains:

**Columns:**
- `Date`: Trading date (format: DD/MM/YYYY)
- `Close`: Closing price
- `Weighted Sentiment Score`: Reliability-filtered weighted sentiment (70% news + 30% social media)
- `Combi 1-10`: Technical indicator combinations (10 signals)
- `Combi 1-10 Days`: Days since last signal change

**Data Split:**
- **Training**: 2020-2023 (~1,006 days)
- **Testing**: 2024 (~251 days)
- **Total**: ~1,267 rows per stock

**Stocks:**
- **AAPL**: Apple Inc.
- **META**: Meta Platforms
- **NVDA**: NVIDIA Corporation
- **SPY**: S&P 500 ETF
- **TSLA**: Tesla Inc.

### Sentiment Data (`sentiment/`)

Filtered sentiment scores extracted from financial news and social media using FinBERT-Tone. These files contain reliability-filtered sentiment (confidence > 0.1) that is used in the final training data.

**Columns:**
- `date`: Date (YYYY-MM-DD format)
- `weighted_sentiment`: Sentiment score [-1, 1]
  - Negative: Bearish sentiment
  - Positive: Bullish sentiment
  - Zero: Neutral

**Sentiment Sources:**
- Financial news: MarketWatch, Google News, Kaggle datasets (70% weight)
- Social media: Reddit, StockTwits, Twitter/X (30% weight)

**Sentiment Model:**
- Model: FinBERT-Tone (ProsusAI/finbert-tone)
- Architecture: BERT-base fine-tuned on financial text
- Output: 3-class (positive, negative, neutral) → converted to continuous [-1, 1]

## Data Generation Pipeline

The training datasets are generated using the script in `notebooks/02_generate_training_data.ipynb`:

1. **Load base training data** with technical indicators
2. **Load filtered sentiment scores** from `sentiment/` folder
3. **Merge** sentiment with stock data by date
4. **Fill missing values** using forward/backward fill
5. **Clean and validate** the final dataset
6. **Save** to `training_data/` folder

To regenerate the training data:

```bash
jupyter notebook notebooks/02_generate_training_data.ipynb
```

## Data Statistics

### Training Data Size

| Stock | Rows | Size | Date Range |
|-------|------|------|------------|
| AAPL | 1,267 | 177 KB | 2020-01-02 to 2024-12-31 |
| META | 1,267 | 170 KB | 2020-01-02 to 2024-12-31 |
| NVDA | 1,267 | 173 KB | 2020-01-02 to 2024-12-31 |
| SPY | 1,267 | 174 KB | 2020-01-02 to 2024-12-31 |
| TSLA | 1,267 | 187 KB | 2020-01-02 to 2024-12-31 |

### Sentiment Data Coverage

| Stock | Sentiment Records | Coverage |
|-------|------------------|----------|
| AAPL | 1,825 | 2020-2024 |
| META | 1,781 | 2020-2024 |
| NVDA | 1,823 | 2020-2024 |
| SPY | 1,772 | 2020-2024 |
| TSLA | 1,827 | 2020-2024 |

Note: Sentiment data covers more dates than trading data (includes weekends/holidays). All sentiment values are reliability-filtered (confidence > 0.1).

## Feature Engineering

### Technical Indicators (20 features)

10 combinations of technical indicators, each with 2 features:
- **Signal value**: {-1, 0, 1} indicating bearish/neutral/bullish
- **Days since change**: Number of days since last signal change

**Indicator Types:**
- EMA (Exponential Moving Average): 12, 20, 50-day
- SMA (Simple Moving Average): 8, 12-day
- MACD (Moving Average Convergence Divergence)
- Price crossovers and momentum signals

### Sentiment Features (1 feature)

- **Weighted Sentiment Score**: Reliability-filtered weighted sentiment (70% news + 30% social media, confidence > 0.1)

### Total Features

- 1 price feature (Close)
- 20 technical features (10 combinations × 2)
- 1 sentiment feature
- **Total: 22 features** for prediction

## Data Quality

### Missing Values

- Training data: < 1% missing (mostly recent dates)
- Filled using forward-fill then backward-fill strategy
- Final datasets have **zero missing values**

### Data Validation

All datasets have been validated for:
- ✅ No duplicate dates
- ✅ Correct date formatting and ordering
- ✅ Reasonable value ranges (sentiment in [-1, 1], prices > 0)
- ✅ Consistent feature count across all stocks
- ✅ Proper train/test temporal split

### Temporal Integrity

- **No data leakage**: Test data (2024) completely unseen during training
- **No look-ahead bias**: All features computed using only past information
- **Realistic trading simulation**: Non-overlapping positions with transaction fees

## Usage in Training Scripts

All training scripts use the `training_data/` folder:

```python
# Load training data
stock = "NVDA"
df = pd.read_csv(f"dataset/training_data/{stock}_data_model_training.csv")

# Parse dates
df["__DateDT__"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df["Year"] = df["__DateDT__"].dt.year

# Split train/test
train_df = df[df["Year"].between(2020, 2023)]
test_df = df[df["Year"] == 2024]
```

## Regenerating Sentiment Data

To regenerate sentiment data from raw news/social media (requires raw text data):

1. Collect raw news articles and social media posts
2. Run FinBERT-Tone sentiment analysis
3. Aggregate by date with reliability weighting
4. Save to `sentiment/` folder
5. Run training data generation notebook

See the main repository README for data collection methodology.

## License & Attribution

- **Stock price data**: Yahoo Finance (public domain)
- **News data**: Various sources (MarketWatch, Google News, Kaggle)
- **Social media data**: Reddit (via PRAW API), StockTwits, Twitter/X
- **Sentiment model**: FinBERT-Tone (MIT License, ProsusAI/HuggingFace)

## Citation

If you use this dataset, please cite our paper:

```bibtex
@inproceedings{multimodal-stock-prediction-2026,
  title={Multimodal Stock Movement Prediction: A Systematic Comparison of Task-Specific and Foundation Models},
  author={Anonymous Authors},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2026}
}
```

## Contact

For questions about the dataset, please open an issue on GitHub or contact the authors (details provided upon paper acceptance).
