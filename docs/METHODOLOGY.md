# Methodology Documentation

This document provides detailed technical documentation of the methodology used in our multimodal stock prediction framework.

## Table of Contents

1. [Problem Formulation](#problem-formulation)
2. [Data Collection](#data-collection)
3. [Feature Engineering](#feature-engineering)
4. [Model Training](#model-training)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Trading Simulation](#trading-simulation)

## Problem Formulation

### Binary Classification Task

We formulate stock prediction as a binary classification problem:

**Given:**
- Historical price sequence: $P_t = \{p_{t-T+1}, ..., p_t\}$
- Technical indicators: $X_t \in \mathbb{R}^{d_x}$
- Sentiment scores: $S_t \in \mathbb{R}^{d_s}$

**Predict:**
```
y_{t+h} = 1 if (p_{t+h} - p_t) / p_t > θ
         = 0 otherwise
```

Where:
- $h \in \{2, 3, ..., 10\}$: prediction horizon (days)
- $θ$: threshold optimized via Bayesian optimization
- $y_{t+h}$: binary direction (1=up, 0=down)

### Multi-Horizon Prediction

We train separate models for each horizon h ∈ {2, 3, ..., 10} days:
- **Short-term**: 2-4 days (high frequency, more noise)
- **Medium-term**: 5-7 days (balanced signal-to-noise)
- **Longer-term**: 8-10 days (lower frequency, stronger signals)

## Data Collection

### 1. Financial News

**Sources:**
- MarketWatch: Real-time financial news
- Google News: Aggregated news from multiple sources
- Kaggle Datasets: Historical news archives

**Collection Period:** 2020-2024

**Processing:**
- Remove duplicates by title similarity
- Filter by stock ticker mentions
- Extract publication timestamp
- Clean HTML and special characters

**Sample Count:**
- AAPL: ~15,000 articles
- META: ~12,000 articles
- NVDA: ~18,000 articles
- SPY: ~10,000 articles
- TSLA: ~20,000 articles

### 2. Social Media

**Sources:**
- Reddit (r/WallStreetBets, r/Stocks): Retail investor sentiment
- StockTwits: Stock-specific discussions
- Twitter/X: Real-time market sentiment

**Collection Method:**
- Reddit API (PRAW): Posts and comments
- StockTwits API: Stock-specific messages
- Twitter API: Ticker mentions and hashtags

**Data Fields:**
- Text content
- Timestamp
- User metadata (for quality filtering)
- Engagement metrics (likes, retweets, comments)

### 3. Stock Data

**Source:** Yahoo Finance API

**Data Fields:**
- Open, High, Low, Close (OHLC)
- Volume
- Adjusted Close (for splits/dividends)

**Frequency:** Daily

**Period:** 2020-01-01 to 2024-12-31

## Feature Engineering

### 1. Technical Indicators

We compute 10 technical indicator combinations:

#### Moving Average Crossovers

**EMA (Exponential Moving Average):**
```python
EMA_12 = prices.ewm(span=12).mean()
EMA_20 = prices.ewm(span=20).mean()
EMA_50 = prices.ewm(span=50).mean()
```

**Crossover Signals:**
- Price > EMA_20: Bullish signal
- Price < EMA_20: Bearish signal
- EMA_12 > EMA_20: Golden cross
- EMA_12 < EMA_20: Death cross

**SMA (Simple Moving Average):**
```python
SMA_8 = prices.rolling(window=8).mean()
SMA_12 = prices.rolling(window=12).mean()
```

#### MACD (Moving Average Convergence Divergence)

```python
EMA_12 = prices.ewm(span=12).mean()
EMA_26 = prices.ewm(span=26).mean()
MACD_line = EMA_12 - EMA_26
Signal_line = MACD_line.ewm(span=9).mean()

# Crossover signal
MACD_signal = 1 if MACD_line > Signal_line else -1
```

#### Technical Indicator Combinations

1. **Combi 1**: Crossover (Price & EMA 20) + Crossover (Price & EMA 50)
2. **Combi 2**: Crossover (Price & EMA 50) + MACD Crossover
3. **Combi 3**: SMA Crossover (8 & 12) + Crossover (Price & EMA 20)
4. **Combi 4**: SMA Crossover (8 & 12) + Crossover (Price & EMA 50)
5. **Combi 5**: SMA Crossover (8 & 12) + MACD Crossover
6. **Combi 6**: Crossover (Price & EMA 20) + MACD Crossover
7. **Combi 7**: EMA Crossover (12 & 20) + MACD Crossover
8. **Combi 8**: SMA Crossover (8 & 12) + Crossover (Price & EMA 200)
9. **Combi 9**: EMA Crossover (12 & 20) + Crossover (Price & EMA 200)
10. **Combi 10**: EMA Crossover (12 & 20) + Crossover (Price & EMA 20)

Each combination includes:
- Binary signal: {-1, 0, 1}
- Days since last signal change

### 2. Sentiment Analysis

#### FinBERT-Tone Model

**Model:** ProsusAI/finbert-tone (HuggingFace)

**Architecture:**
- Base: BERT-base
- Fine-tuned on financial text
- Output: 3-class sentiment (positive, negative, neutral)

**Processing:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert-tone")

def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)

    # scores: [negative, neutral, positive]
    sentiment_score = scores[0][2].item() - scores[0][0].item()  # positive - negative
    return sentiment_score  # Range: [-1, 1]
```

#### Sentiment Aggregation

**Daily Aggregation:**
```python
# News sentiment (higher reliability)
news_sentiment = news_df.groupby('date')['sentiment'].mean()

# Social media sentiment (lower reliability)
social_sentiment = social_df.groupby('date')['sentiment'].mean()

# Weighted aggregation (optimal weights from ablation study)
ALPHA_NEWS = 0.70
ALPHA_SOCIAL = 0.30

daily_sentiment = ALPHA_NEWS * news_sentiment + ALPHA_SOCIAL * social_sentiment
```

**Filtering Low-Quality Sentiment:**
```python
# Filter out neutral/low-confidence sentiments
MIN_CONFIDENCE = 0.1
filtered_sentiment = daily_sentiment[abs(daily_sentiment) > MIN_CONFIDENCE]
```

### 3. Feature Vector Construction

Final feature vector for time $t$:

```python
features = {
    # Price features
    'Close': close_price,

    # Technical indicators (10 combinations × 2 features)
    'Combi_1_Signal': signal_1,
    'Combi_1_Days': days_since_signal_1,
    ...
    'Combi_10_Signal': signal_10,
    'Combi_10_Days': days_since_signal_10,

    # Sentiment features
    'Weighted_Sentiment': weighted_sentiment,
    'Filtered_Sentiment': filtered_sentiment,
}
```

Total features: 1 (price) + 20 (technical) + 2 (sentiment) = 23 features

## Model Training

### 1. Data Splitting

**Temporal Split (No Data Leakage):**
- Training: 2020-2023 (4 years, ~1,006 days)
- Testing: 2024 (1 year, ~251 days)

**Validation Strategy:**
```python
# Time-series cross-validation
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X_train):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    # Train and validate
```

### 2. Hyperparameter Optimization

**Framework:** Optuna (Bayesian Optimization)

**Configuration:**
```python
import optuna

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()
)

study.optimize(objective, n_trials=100, timeout=3600)
```

**Optimization Metrics:**
- Primary: ROC-AUC (discrimination)
- Secondary: Sharpe Ratio (profitability)

### 3. Model Architectures

#### XGBoost

```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42
}
```

#### LightGBM

```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'random_state': 42
}
```

#### LSTM

```python
model = Sequential([
    LSTM(units=128, return_sequences=True, input_shape=(seq_len, n_features)),
    Dropout(0.3),
    LSTM(units=64, return_sequences=False),
    Dropout(0.3),
    Dense(units=32, activation='relu'),
    Dropout(0.2),
    Dense(units=1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)
```

### 4. Early Stopping & Regularization

```python
# XGBoost early stopping
eval_set = [(X_train, y_train), (X_val, y_val)]
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    early_stopping_rounds=50,
    verbose=False
)

# LSTM early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop]
)
```

## Evaluation Metrics

### Classification Metrics

**1. Accuracy:**
```python
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**2. ROC-AUC:**
```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_pred_proba)
```

**3. Confusion Matrix:**
```
              Predicted
              Down   Up
Actual Down   TN     FP
       Up     FN     TP
```

### Trading Performance Metrics

**1. Win Rate:**
```python
win_rate = (number_of_profitable_trades) / (total_trades)
```

**2. Total Return:**
```python
# With transaction fees
fee_multiplier = 1 - (fee_bps / 10000)
equity = 1.0

for trade in trades:
    gross_return = (exit_price / entry_price) - 1
    net_return = (1 + gross_return) * fee_multiplier - 1
    equity *= (1 + net_return)

total_return = equity - 1
```

**3. Sharpe Ratio:**
```python
# Annualized risk-adjusted return
trade_returns = np.array([...])  # Individual trade returns
sharpe = (
    trade_returns.mean() / trade_returns.std()
    * np.sqrt(len(trade_returns))
)
```

## Trading Simulation

### Non-Overlapping Backtest

**Strategy:** Long-Short (always in market)

```python
def non_overlap_backtest(test_df, predictions, horizon, fee_bps=10):
    prices = test_df.sort_values('Date')['Close'].values
    fee_mult = 1 - fee_bps / 10000

    equity = 1.0
    trades = []
    i = 0

    while i + horizon < len(prices):
        entry_price = prices[i]
        exit_price = prices[i + horizon]
        prediction = predictions[i]

        # Determine position
        if prediction == 1:
            # Go long
            gross_return = (exit_price / entry_price) - 1
        else:
            # Go short
            gross_return = (entry_price / exit_price) - 1

        # Apply transaction fees
        net_return = (1 + gross_return) * fee_mult - 1
        equity *= (1 + net_return)
        trades.append(net_return)

        # Move to next non-overlapping position
        i += horizon

    return {
        'equity': equity,
        'total_return': equity - 1,
        'trades': trades,
        'win_rate': (np.array(trades) > 0).mean()
    }
```

### Transaction Costs

**Fee Structure:**
- Round-trip fee: 10 basis points (0.10%)
- Entry fee: 5 bps
- Exit fee: 5 bps
- Includes: Commission + slippage estimate

**Impact on Returns:**
```python
# Example: 10% gross return
gross_return = 0.10
fee_multiplier = 1 - 0.001  # 10 bps
net_return = (1 + 0.10) * 0.999 - 1 = 0.0989  # 9.89%

# Fee cost: 0.10 - 0.0989 = 0.0011 = 11 bps
```

### Position Sizing

**Equal Weight Strategy:**
- Each trade: 100% of capital
- No leverage
- Binary: Either long or short

**Alternative Strategies (Future Work):**
- Kelly Criterion sizing
- Risk parity allocation
- Volatility-adjusted positions

## Model Selection Criteria

### ROC-AUC Selection

For each stock, select model with highest test AUC:
```python
best_model = models[np.argmax([m.test_auc for m in models])]
```

**Characteristics:**
- Maximizes discrimination ability
- Better for risk management applications
- May not optimize for trading profitability

### Sharpe Ratio Selection

For each stock, select model with highest Sharpe:
```python
best_model = models[np.argmax([m.sharpe_ratio for m in models])]
```

**Characteristics:**
- Maximizes risk-adjusted returns
- Better for trading applications
- Directly optimizes profitability

### Trade-offs

| Criterion | Pros | Cons |
|-----------|------|------|
| AUC | Better discrimination, More stable | Lower returns |
| Sharpe | Better trading performance, Higher returns | More variable |

## Reproducibility

### Random Seeds

```python
RANDOM_SEED = 42

# Python
random.seed(RANDOM_SEED)

# NumPy
np.random.seed(RANDOM_SEED)

# TensorFlow
tf.random.set_seed(RANDOM_SEED)

# PyTorch
torch.manual_seed(RANDOM_SEED)

# Optuna
sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
```

### Deterministic Operations

```python
# TensorFlow
tf.config.experimental.enable_op_determinism()

# PyTorch
torch.use_deterministic_algorithms(True)
```

## References

For detailed algorithm descriptions and theoretical foundations, please refer to our paper and the following key references:

1. FinBERT: Araci (2019) - Financial sentiment analysis
2. Chronos-2: Ansari et al. (2025) - Time-series foundation model
3. FinCast: Zhu et al. (2025) - Financial forecasting foundation model
4. XGBoost: Chen & Guestrin (2016) - Gradient boosting framework
5. LSTM: Hochreiter & Schmidhuber (1997) - Long short-term memory networks
