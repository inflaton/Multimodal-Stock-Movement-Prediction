# Dataset Cleanup Summary

## Changes Made (2026-01-23)

### 1. Sentiment Files Cleanup

**Before:**
- 10 sentiment CSV files (5 unfiltered + 5 filtered)
- Total size: ~1.1 MB

**After:**
- 5 sentiment CSV files (filtered only)
- Total size: ~540 KB (50% reduction)

**Actions:**
✅ Removed unfiltered sentiment files:
- `news_sentiment_finbert_tone_weighted_aapl.csv` → REMOVED
- `news_sentiment_finbert_tone_weighted_meta.csv` → REMOVED
- `news_sentiment_finbert_tone_weighted_nvda.csv` → REMOVED
- `news_sentiment_finbert_tone_weighted_spy.csv` → REMOVED
- `news_sentiment_finbert_tone_weighted_tsla.csv` → REMOVED

✅ Renamed filtered files (removed `_filtered` suffix):
- `news_sentiment_finbert_tone_weighted_aapl_filtered.csv` → `news_sentiment_finbert_tone_weighted_aapl.csv`
- `news_sentiment_finbert_tone_weighted_meta_filtered.csv` → `news_sentiment_finbert_tone_weighted_meta.csv`
- `news_sentiment_finbert_tone_weighted_nvda_filtered.csv` → `news_sentiment_finbert_tone_weighted_nvda.csv`
- `news_sentiment_finbert_tone_weighted_spy_filtered.csv` → `news_sentiment_finbert_tone_weighted_spy.csv`
- `news_sentiment_finbert_tone_weighted_tsla_filtered.csv` → `news_sentiment_finbert_tone_weighted_tsla.csv`

### 2. Training Data Cleanup

**Before:**
- 24 columns per file
- Features: Price + 20 technical + 2 sentiment features
- Columns: `Date`, `Close`, `Weighted Sentiment Score`, `Filtered Sentiment Score`, `Combi 1-10`, `Combi Days 1-10`

**After:**
- 23 columns per file (-1 column)
- Features: Price + 20 technical + 1 sentiment feature
- Columns: `Date`, `Close`, `Weighted Sentiment Score`, `Combi 1-10`, `Combi Days 1-10`

**Actions:**
✅ For each stock (AAPL, META, NVDA, SPY, TSLA):
1. Copied `Filtered Sentiment Score` → `Weighted Sentiment Score` (overwrite)
2. Removed `Filtered Sentiment Score` column
3. Saved updated file

**Rationale:**
- `Filtered Sentiment Score` is the reliability-filtered version (confidence > 0.1)
- It represents the final sentiment values used in modeling
- No need to keep both columns - use filtered values as the primary sentiment feature

### 3. Documentation Updates

**Updated files:**
- ✅ `dataset/README.md` - Updated to reflect 5 sentiment files (not 10), simplified feature description
- ✅ `README.md` - Changed "Sentiment Features" → "Sentiment Feature" (singular)
- ✅ `docs/USAGE.md` - Updated data format description

**Key changes:**
- Feature count: 23 → 22 features per stock
- Sentiment columns: 2 → 1 column
- File organization: Clearer separation between training data and sentiment sources

## Final Dataset Structure

```
dataset/
├── training_data/          # 5 CSV files (~880 KB)
│   ├── AAPL_data_model_training.csv (1,267 rows × 23 cols)
│   ├── META_data_model_training.csv (1,267 rows × 23 cols)
│   ├── NVDA_data_model_training.csv (1,267 rows × 23 cols)
│   ├── SPY_data_model_training.csv  (1,267 rows × 23 cols)
│   └── TSLA_data_model_training.csv (1,267 rows × 23 cols)
│
├── sentiment/              # 5 CSV files (~540 KB)
│   ├── news_sentiment_finbert_tone_weighted_aapl.csv
│   ├── news_sentiment_finbert_tone_weighted_meta.csv
│   ├── news_sentiment_finbert_tone_weighted_nvda.csv
│   ├── news_sentiment_finbert_tone_weighted_spy.csv
│   └── news_sentiment_finbert_tone_weighted_tsla.csv
│
└── README.md               # Dataset documentation
```

## Benefits

1. **Simplified Structure**: Only one sentiment column, easier to understand
2. **Reduced Redundancy**: Eliminated duplicate unfiltered sentiment files
3. **Clearer Intent**: The single "Weighted Sentiment Score" now clearly represents the final filtered values
4. **Smaller Repository**: ~50% reduction in sentiment file size
5. **Better Documentation**: Updated all docs to reflect the simplified structure

## Feature Breakdown (Final)

### Training Data Columns (23 total)

1. **Date** - Trading date
2. **Close** - Closing price
3. **Weighted Sentiment Score** - Reliability-filtered weighted sentiment (70% news, 30% social media, confidence > 0.1)
4-23. **Technical Indicators** - 10 combinations × 2 features (signal + days)

### Total Features for ML Models: 22
- 1 price feature
- 1 sentiment feature
- 20 technical indicator features

## Verification

All changes verified successfully:
✅ 5 training data files updated (AAPL, META, NVDA, SPY, TSLA)
✅ 5 sentiment files renamed (removed _filtered suffix)
✅ 5 unfiltered sentiment files removed
✅ All documentation updated
✅ Column count reduced from 24 to 23 in all training files
