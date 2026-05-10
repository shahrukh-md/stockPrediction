# stockPrediction
Phase 1 — Data
  raw_ohlcv.csv + nifty50_macro.csv + nifty50_fundamentals.csv
  → nifty50_full_dataset.csv (177,853 rows)
  → nifty50_cleaned.csv (158,768 rows · 50 cols · 0 nulls)

Phase 2 — Feature Engineering
  → nifty50_v4_features.csv (146,672 rows · 60 cols)
    Added: return_3m/6m/12m, dist_52w_high, realvol_20d/60d,
           vol_ratio_2060, amihud_20d, excess_return_1m

Phase 3 — Feature Selection + Transformation
  → nifty50_v4_model_ready.csv (145,837 rows · 42 cols)
    Dropped 14 redundant cols, converted price levels to returns/ratios

Phase 4 — Cross-Sectional Normalization
  → nifty50_v4_cs_normalized.csv (7,317 monthly rows · 37 cols)
    Month-end snapshot, CS zscore per feature across 48 stocks

Phase 5 — Interaction Features
  → nifty50_v4_interaction.csv (7,317 rows · 47 cols)
    Added 11 interaction features:
    vol-adjusted momentum, mean reversion strength,
    quality momentum, value momentum, BB squeeze momentum,
    vol trend momentum, ATR momentum, MACD confirmations

Phase 6 — Optuna Tuning + Walk-Forward Training
  → results_interaction/
    24M rolling window · 132 folds · 40 Optuna trials per model
    Rank IC: 0.0921 (Ensemble) · 0.0987 (RF)
    Net alpha: ~10.4% annualized over Nifty