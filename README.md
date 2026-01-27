# Multimodal Stock Movement Prediction: A Systematic Comparison of Task-Specific and Foundation Models

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-FF6F00.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**IJCNN 2026 Submission**

This repository contains the official implementation and datasets for our paper: **"Multimodal Stock Movement Prediction: A Systematic Comparison of Task-Specific and Foundation Models"**

## 📄 Abstract

Predicting short-term stock price movements remains challenging due to market volatility and the influence of both quantitative and qualitative factors. We propose a multimodal framework that integrates technical indicators with FinBERT-based sentiment analysis from financial news and social media for 2-10 day stock direction prediction. Seven model families—LSTM, Random Forest, XGBoost, LightGBM, Gradient Boosting, Logistic Regression, and SVM—are systematically evaluated with Bayesian hyperparameter optimization across five stocks (AAPL, META, NVDA, SPY, TSLA) spanning 2020-2024.

Our key findings:
- **ROC-AUC vs Sharpe trade-off**: AUC-optimized models achieve strong discriminative performance (0.697 average, peak 0.793), while Sharpe-optimized models deliver superior risk-adjusted returns (2.05 average Sharpe, 66.8% win rate)
- **Sentiment aggregation**: Reliability-weighted sentiment (70% news, 30% social media) substantially outperforms equal weighting (2.05 vs. -0.25 Sharpe)
- **Foundation model comparison**: Our task-specific approach achieves 2.05 Sharpe, significantly outperforming fine-tuned Chronos-2 (0.90) and FinCast (0.16)
- **Complementary features**: Technical features primarily drive discrimination while sentiment features primarily drive profitability

## 🌟 Key Contributions

1. **Multimodal Framework**: Integration of FinBERT-Tone sentiment from financial news and social media with technical indicators (OHLCV data) for 2-10 day stock direction prediction

2. **Systematic Evaluation**: Comprehensive comparison of seven model families with Bayesian hyperparameter optimization, revealing critical trade-offs between ROC-AUC and Sharpe ratio selection criteria

3. **Sentiment Aggregation**: Demonstration that reliability-weighted sentiment aggregation (70% news, 30% social media) substantially improves trading performance, with ablation studies revealing complementary roles of technical and sentiment features

4. **Foundation Model Benchmark**: First systematic comparison with state-of-the-art time-series foundation models (FinCast and Chronos-2) in both zero-shot and fine-tuned settings for financial prediction tasks

## 📊 Dataset

We provide preprocessed datasets for five stocks spanning 2020-2024:

- **AAPL** (Apple Inc.)
- **META** (Meta Platforms)
- **NVDA** (NVIDIA Corporation)
- **SPY** (S&P 500 ETF)
- **TSLA** (Tesla Inc.)

### Data Features

Each stock dataset (`dataset/training_data/*_data_model_training.csv`) includes:

- **Price Data**: Close prices, historical trends
- **Sentiment Feature**: FinBERT-Tone reliability-filtered weighted sentiment (70% news, 30% social media) from:
  - Financial news (MarketWatch, Google News, Kaggle datasets)
  - Social media (Reddit, StockTwits, Twitter/X)
- **Technical Indicators**: 10 combinations of trading signals:
  - EMA/Price Crossovers (12, 20, 50-day)
  - SMA Crossovers (8, 12, 20-day)
  - MACD Crossover signals
  - Prediction horizons (2-10 days)

### Data Split
- **Training**: 2020-2023 (1,006 samples average)
- **Testing**: 2024 (251 samples average)

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
CUDA 11.8+ (for GPU acceleration, optional)
```

### Installation

1. Clone the repository:
```bash
https://github.com/inflaton/Multimodal-Stock-Movement-Prediction.git
cd Multimodal-Stock-Movement-Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

#### Optional: XGBoost Support (macOS)

XGBoost requires the OpenMP library (`libomp`) on macOS. If you encounter an XGBoost import error:

```bash
# Install OpenMP library via Homebrew
brew install libomp

# Reinstall XGBoost
pip install --upgrade xgboost
```

### FinCast Baseline Setup

To run FinCast baseline experiments, you need to set up the FinCast environment separately:

```bash
# Clone the FinCast repository
git clone https://github.com/inflaton/FinCast-fts.git
cd FinCast-fts

# Create conda environment and install dependencies
bash ./env_setup.sh && bash ./dep_install.sh

# Return to main project directory
cd ..
```

This will create a conda environment named `fincast_v1` with Python 3.11 and all required dependencies including PyTorch 2.5.0 with CUDA 12.4 support.

**Note**: FinCast requires:
- Python 3.11+
- PyTorch 2.5+
- CUDA 11.8+ for GPU acceleration (PyTorch 2.5.0 supports compute capabilities sm_50, sm_80, sm_86, sm_89, sm_90, sm_90a)
- Conda for environment management

**GPU Compatibility**: Newer GPUs like NVIDIA GB10 (sm_121 Blackwell architecture) are not supported by PyTorch 2.5.0. For these GPUs:
- Set `export CUDA_VISIBLE_DEVICES=""` to run on CPU, or
- Install PyTorch 2.9+ nightly/from source with sm_121 support

### Quick Start

#### 1. Train Task-Specific Models

Train models with hyperparameter optimization:

```bash
# Train XGBoost models
python scripts/xgboost_hyperparameter_tuning.py --stock NVDA

# Train all models for a specific stock
for model in xgboost lightgbm random_forest logistic_regression svm gradient_boosting; do
    python scripts/${model}_hyperparameter_tuning.py --stock NVDA
done

# Train LSTM models
python scripts/lstm_hyperparameter_tuning.py --stock NVDA
```

#### 2. Run Baseline Models

**Chronos-2 Baselines:**

```bash
# Zero-shot
python scripts/chronos_baseline.py --all-stocks

# Fine-tuned (price only)
python scripts/chronos_finetune.py --all-stocks

# Fine-tuned (with sentiment)
python scripts/chronos_finetune.py --all-stocks --use-covariates
```

**FinCast Baselines:**

```bash
conda activate fincast_v1

# Zero-shot
python scripts/fincast_baseline.py --all-stocks

# Fine-tuning (trains models)
python scripts/fincast_finetune.py --all-stocks

# Fine-tuned inference (evaluates fine-tuned models)
python scripts/fincast_finetuned_inference.py --all-stocks
```

#### 3. Analyze Results

Use Jupyter notebooks for comprehensive analysis:

```bash
jupyter notebook notebooks/02_our_results.ipynb      # Tuned models analysis
jupyter notebook notebooks/03_ablation_study.ipynb   # Ablation studies
jupyter notebook notebooks/04_baseline_results.ipynb # Baseline comparison
```

## 📈 Main Results

### Task-Specific Models vs Foundation Models

| Method | Selection | Accuracy | AUC | Trades | Win% | Sharpe |
|--------|-----------|----------|-----|--------|------|--------|
| **Ours (Multimodal)** | AUC | 0.496 | **0.697** | 33.6 | 38.2 | -1.29 |
| **Ours (Multimodal)** | Sharpe | 0.530 | 0.551 | 39.8 | **66.8** | **2.05** |
| Chronos-2 Fine-tuned + Sentiment | AUC | 0.534 | 0.542 | 56.0 | 52.7 | 0.24 |
| Chronos-2 Fine-tuned + Sentiment | Sharpe | 0.464 | 0.475 | 56.0 | 57.2 | **0.89** |
| FinCast Fine-tuned | AUC | **0.596** | **0.693** | 13.6 | 39.2 | -0.40 |
| FinCast Fine-tuned | Sharpe | 0.546 | 0.529 | 21.8 | 44.6 | **0.16** |

### Ablation Study: Sentiment Aggregation

| Configuration | AUC | Win% | Sharpe | Total Return |
|---------------|-----|------|--------|--------------|
| **Weighted (70% news, 30% social)** | 0.551 | **66.8** | **2.05** | **140.5%** |
| Equal Weight (50% news, 50% social) | 0.536 | 51.4 | -0.25 | -16.0% |
| News Only (100% news) | 0.547 | 61.0 | 1.50 | 96.8% |
| Social Only (100% social) | 0.516 | 45.3 | -1.04 | -57.9% |

### Feature Ablation

| Feature Set | AUC | Win% | Sharpe |
|-------------|-----|------|--------|
| **Technical + Sentiment (Full)** | **0.551** | **66.8** | **2.05** |
| Technical Only | 0.531 | 64.9 | 1.88 |
| Sentiment Only | 0.516 | 58.2 | 0.43 |

## 🏗️ Repository Structure

```
.
├── dataset/                        # All datasets
│   ├── training_data/            # Final preprocessed training data
│   │   ├── AAPL_data_model_training.csv
│   │   ├── META_data_model_training.csv
│   │   ├── NVDA_data_model_training.csv
│   │   ├── SPY_data_model_training.csv
│   │   └── TSLA_data_model_training.csv
│   ├── sentiment/                # Sentiment analysis results
│   │   └── news_sentiment_finbert_tone_weighted_*.csv
│   └── README.md                 # Dataset documentation
│
├── scripts/                        # Training and evaluation scripts
│   ├── xgboost_hyperparameter_tuning.py
│   ├── lightgbm_hyperparameter_tuning.py
│   ├── random_forest_hyperparameter_tuning.py
│   ├── logistic_regression_hyperparameter_tuning.py
│   ├── svm_hyperparameter_tuning.py
│   ├── gradient_boosting_hyperparameter_tuning.py
│   ├── lstm_hyperparameter_tuning.py
│   ├── chronos_baseline.py
│   ├── chronos_finetune.py
│   ├── chronos_inference.py
│   ├── fincast_baseline.py
│   ├── fincast_finetune.py
│   ├── fincast_finetuned_inference.py
│   ├── ablation_study.py
│   ├── analyze_chronos_results.py
│   ├── analyze_tuned_results.py
│   └── analyze_ablation_results.py
│
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── 01_update_sentiments_for_training_data.ipynb
│   ├── 02_our_results.ipynb
│   ├── 03_ablation_study.ipynb
│   └── 04_baseline_results.ipynb
│
├── results/                        # Model results and metrics
│   ├── chronos_all_results_combined.csv
│   └── Filtered_*_hyperparameter_tuned_results.csv
│
├── FinCast-fts/                    # FinCast foundation model (git submodule)
│   ├── env_setup.sh              # Create conda environment
│   ├── dep_install.sh            # Install dependencies
│   ├── scripts/                  # FinCast training scripts
│   └── README.md                 # FinCast documentation
│
├── models/                         # Trained model checkpoints (to be added)
├── docs/                           # Documentation
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## 🔬 Methodology

### Model Training Pipeline

1. **Data Preprocessing**: Load and validate stock data with technical indicators and sentiment scores
2. **Hyperparameter Optimization**: Bayesian optimization (scikit-optimize) with 30 iterations per configuration
3. **Model Training**: Train on 2020-2023 data with early stopping
4. **Evaluation**: Test on 2024 out-of-sample data
5. **Trading Simulation**: Non-overlapping backtest with 10 bps transaction fees

### Evaluation Metrics

- **Classification**: Accuracy, ROC-AUC, Confusion Matrix
- **Trading Performance**:
  - Win Rate: Percentage of profitable trades
  - Sharpe Ratio: Risk-adjusted return (annualized)
  - Total Return: Cumulative return with fees
  - Number of Trades: Trading frequency

### Hyperparameter Search Spaces

**XGBoost:**
- `n_estimators`: [50, 500]
- `max_depth`: [3, 12]
- `learning_rate`: [0.01, 0.3]
- `subsample`: [0.5, 1.0]
- `colsample_bytree`: [0.5, 1.0]

**LSTM:**
- `units`: [32, 256]
- `dropout`: [0.1, 0.5]
- `learning_rate`: [0.0001, 0.01]
- `sequence_length`: [10, 60]

See individual training scripts for complete hyperparameter ranges.

## 📊 Reproducing Results

### Step-by-Step Guide

1. **Set Up FinCast Environment** (for baseline comparisons):

   ```bash
   # Clone and set up FinCast
   git clone https://github.com/inflaton/FinCast-fts.git
   cd FinCast-fts
   bash ./env_setup.sh && bash ./dep_install.sh
   cd ..
   ```

2. **Update Sentiment Data** (if needed):

   ```bash
   jupyter notebook notebooks/01_update_sentiments_for_training_data.ipynb
   ```

3. **Train All Task-Specific Models**:

   ```bash
   # Train all models for all stocks
   ./run_all_hyperparameter_tuning.sh
   ```

4. **Run Ablation Study**:

   ```bash
   # Run ablation experiments for all configurations
   ./run_ablation_study.sh
   ```

5. **Train Foundation Model Baselines**:

   ```bash
   # Run Chronos-2 baselines (zero-shot and fine-tuned, with/without sentiment)
   ./run_baselines_chronos-2.sh

   # Run FinCast baselines (zero-shot, fine-tuning, and inference)
   conda activate fincast_v1
   ./run_baselines_fincast.sh
   ```

6. **Analyze Results**:

   ```bash
   jupyter notebook notebooks/02_our_results.ipynb      # Our task-specific models results
   jupyter notebook notebooks/03_ablation_study.ipynb   # Ablation study analysis
   jupyter notebook notebooks/04_baseline_results.ipynb # Baseline models comparison
   ```

## 🎯 Key Findings

### 1. Selection Criterion Trade-offs

- **AUC-Selected Models**: Better discrimination (0.697 AUC), but lower profitability
- **Sharpe-Selected Models**: Better risk-adjusted returns (2.05 Sharpe, 66.8% win rate)

### 2. Sentiment Aggregation

- **Weighted aggregation** (70% news, 30% social) significantly outperforms equal weighting
- News sentiment is more reliable than social media sentiment
- Combined sentiment features provide incremental value over technical features alone

### 3. Feature Complementarity

- **Technical features**: Primarily drive discrimination (AUC)
- **Sentiment features**: Primarily drive profitability (Sharpe, Win Rate)
- **Combined features**: Achieve best overall performance

### 4. Foundation Model Limitations

- **Zero-shot performance**: Near random chance (AUC 0.477-0.560)
- **Fine-tuning helps**: But still underperforms task-specific models
- **Covariate paradox**: Adding sentiment to Chronos-2 degrades trading performance
- **Task-specific advantage**: Domain-specific optimization crucial for financial prediction

## 💻 Hardware Requirements

- **Minimum**: 16GB RAM, CPU-only training
- **Recommended**: 32GB RAM, NVIDIA GPU (8GB+ VRAM) for LSTM and foundation model training
- **Foundation Models**: NVIDIA GPU with CUDA 11.8+ (CUDA 12.4+ recommended for FinCast)
- **Training Time**:
  - Task-specific models: 1-4 hours per stock (CPU)
  - LSTM models: 2-6 hours per stock (GPU)
  - Chronos-2 baselines: 4-8 hours per stock (GPU)
  - FinCast baselines: 6-12 hours per stock (GPU)

## 📝 Citation

If you use this code or datasets in your research, please cite our paper:

```bibtex
@inproceedings{multimodal-stock-prediction-2026,
  title={Multimodal Stock Movement Prediction: A Systematic Comparison of Task-Specific and Foundation Models},
  author={Anonymous Authors},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2026}
}
```

## 🤝 Contributing

This is a research project submitted to IJCNN 2026. Upon acceptance, we will welcome contributions including:
- Bug fixes and improvements
- Additional baseline implementations
- Extended ablation studies
- New stock additions

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the authors (details will be provided upon acceptance).

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FinBERT** and **FinBERT-Tone** for sentiment analysis
- **Chronos-2** ([Amazon Science](https://github.com/amazon-science/chronos-forecasting)) for time-series foundation model baselines
- **FinCast** ([Zhu et al., CIKM 2025](https://arxiv.org/abs/2508.19609)) for financial time-series foundation model baselines
- **scikit-optimize** for Bayesian hyperparameter optimization
- **scikit-learn**, **XGBoost**, **LightGBM**, **TensorFlow** for machine learning implementations
- Financial data providers: Yahoo Finance, MarketWatch, Google News
- Social media data sources: Reddit, StockTwits, Twitter/X

## 📊 Paper Status

**Status**: Under Review at IJCNN 2026
**Submission Date**: [To be added]
**Paper ID**: [To be added]

---

**Note**: This is an anonymous submission for double-blind peer review. Author information and affiliations will be disclosed upon acceptance.
