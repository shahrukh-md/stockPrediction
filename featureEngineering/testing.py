"""
V4 Final Feature Engineering Script
Based on EDA findings — drops redundant features, converts price levels to returns/ratios,
and produces the final clean feature set ready for model training.

Run: python feature_engineering_final_v4.py

Input : nifty50_v4_features.csv
Output: nifty50_v4_model_ready.csv
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

INPUT_FILE  = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_v4_features.csv"
OUTPUT_FILE = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_v4_model_ready.csv"

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("Loading nifty50_v4_features.csv...")
df = pd.read_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)
df = df.sort_values(["Ticker", "date"]).reset_index(drop=True)
print(f"  Shape   : {df.shape}")
print(f"  Tickers : {df['Ticker'].nunique()}")
print(f"  Dates   : {df['date'].min().date()} → {df['date'].max().date()}\n")

# ── 2. DROP REDUNDANT FEATURES ────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: DROP REDUNDANT FEATURES")
print("=" * 60)

drop_cols = [
    # MA cluster — keep only MA_200, drop the rest (r > 0.994 with each other)
    "MA_20", "MA_50", "EMA_20",

    # MACD — keep MACD_sig and MACD_diff, drop raw MACD (r=0.944 with MACD_sig)
    "MACD",

    # PE — keep forward_PE, drop trailing PE (r=0.965)
    "PE_ratio",

    # RSI — redundant with BB_pct (r=0.887), BB_pct is more directly interpretable
    "RSI_14",

    # excess_return_1m — r=0.98 with return_20d, completely redundant
    "excess_return_1m",

    # Bollinger raw levels — keep BB_width and BB_pct only
    "BB_upper", "BB_lower",

    # Price level columns — non-stationary, will be replaced with ratio/return versions
    "Nifty50_close", "Nifty50_MA50",

    # Raw price levels for macro — will be replaced with return versions below
    "USD_INR", "Gold_price", "Crude_oil_price",
]

# Only drop columns that actually exist
drop_cols = [c for c in drop_cols if c in df.columns]
df.drop(columns=drop_cols, inplace=True)
print(f"  Dropped {len(drop_cols)} redundant columns: {drop_cols}")
print(f"  Shape after drop: {df.shape}\n")

# ── 3. CONVERT PRICE LEVELS TO STATIONARY FEATURES ───────────────────────────
print("=" * 60)
print("STEP 3: CONVERT PRICE LEVELS TO STATIONARY FEATURES")
print("=" * 60)

df = df.sort_values(["Ticker", "date"]).reset_index(drop=True)

# 3a. close / MA_200 ratio — price relative to long-term trend
# > 1 means stock is above its 200-day average (uptrend)
# < 1 means below (downtrend or depressed)
df["price_to_ma200"] = df["close"] / df["MA_200"]
print("  Added: price_to_ma200  = close / MA_200")

# 3b. Macro features as returns (computed per date since macro is same for all stocks)
# Sort by date first for the diff to be correct
macro_df = df[["date", "Ticker"]].copy()

# We need to compute macro returns once per date (not per stock)
# Get unique date-level macro values first
macro_cols_raw = []

# Check which macro source columns we still have after the drop
# USD_INR, Gold_price, Crude_oil_price were dropped — need to recompute from original
# Instead, compute from the already-available Nifty50_return equivalent approach:
# We'll use the per-stock groupby but only keep the first occurrence per date for macro

# Re-load just the macro columns from original to get the raw price levels back
print("\n  Re-loading macro price levels from original file for return computation...")
df_orig = pd.read_csv(INPUT_FILE, parse_dates=["date"], low_memory=False,
                      usecols=["date", "Ticker", "USD_INR", "Gold_price", "Crude_oil_price"])
df_orig = df_orig.sort_values("date").drop_duplicates("date").reset_index(drop=True)

# Compute 1-day returns for macro series
df_orig["USD_INR_return"]      = df_orig["USD_INR"].pct_change()
df_orig["Gold_return"]         = df_orig["Gold_price"].pct_change()
df_orig["Crude_oil_return"]    = df_orig["Crude_oil_price"].pct_change()

# Compute 20-day (monthly) returns for macro
df_orig["USD_INR_return_20d"]  = df_orig["USD_INR"].pct_change(20)
df_orig["Gold_return_20d"]     = df_orig["Gold_price"].pct_change(20)
df_orig["Crude_return_20d"]    = df_orig["Crude_oil_price"].pct_change(20)

macro_return_cols = [
    "date", "USD_INR_return", "Gold_return", "Crude_oil_return",
    "USD_INR_return_20d", "Gold_return_20d", "Crude_return_20d"
]
macro_returns = df_orig[macro_return_cols]

# Merge macro returns back into main df
df = df.merge(macro_returns, on="date", how="left")

print("  Added: USD_INR_return      (daily USD/INR % change)")
print("  Added: Gold_return         (daily gold % change)")
print("  Added: Crude_oil_return    (daily crude % change)")
print("  Added: USD_INR_return_20d  (20-day USD/INR % change)")
print("  Added: Gold_return_20d     (20-day gold % change)")
print("  Added: Crude_return_20d    (20-day crude % change)")

# 3c. ATR as % of price — normalize ATR by close price for cross-stock comparability
# Raw ATR in rupees is not comparable across stocks (RELIANCE ATR vs NESTLEIND ATR)
df["atr_pct"] = df["ATR_14"] / df["close"]
print("  Added: atr_pct             = ATR_14 / close  (normalized ATR)")

# Drop raw ATR and MA_200 now that we have the ratio versions
df.drop(columns=["ATR_14", "MA_200"], inplace=True)
print("  Dropped: ATR_14, MA_200  (replaced by atr_pct, price_to_ma200)")

print(f"\n  Shape after conversions: {df.shape}\n")

# ── 4. WINSORIZE NEW RATIO/RETURN FEATURES ────────────────────────────────────
print("=" * 60)
print("STEP 4: WINSORIZE NEW FEATURES")
print("=" * 60)

def winsorize(series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

new_cols_to_winsorize = [
    "price_to_ma200",
    "USD_INR_return", "Gold_return", "Crude_oil_return",
    "USD_INR_return_20d", "Gold_return_20d", "Crude_return_20d",
    "atr_pct"
]

for col in new_cols_to_winsorize:
    if col in df.columns:
        df[col] = winsorize(df[col])
        print(f"  Winsorized: {col}")

# ── 5. HANDLE NaNs FROM NEW FEATURES ─────────────────────────────────────────
print(f"\n  NaNs before final drop: {df.isnull().sum().sum()}")
df.dropna(inplace=True)
print(f"  NaNs after  final drop: {df.isnull().sum().sum()}")
print(f"  Shape after NaN drop  : {df.shape}")

# ── 6. FINAL FEATURE LIST ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: FINAL FEATURE LIST")
print("=" * 60)

# Define the exact feature set for model training
FEATURE_COLS = [
    # Price structure
    "price_to_ma200",       # price relative to long-term trend (stationary)
    "BB_width",             # bollinger band width (volatility measure)
    "BB_pct",               # where price sits in BB band (0=lower, 1=upper)

    # Momentum (short to long)
    "return_1d",            # 1-day momentum
    "return_5d",            # 1-week momentum
    "return_10d",           # 2-week momentum
    "return_20d",           # 1-month momentum
    "return_3m",            # 3-month momentum
    "return_6m",            # 6-month momentum
    "return_12m",           # 12-month momentum (skip-month)

    # Mean reversion
    "dist_52w_high",        # distance from 52-week high (negative = below peak)
    "MACD_sig",             # MACD signal line (trend momentum)
    "MACD_diff",            # MACD histogram (momentum acceleration)

    # Volatility
    "realvol_20d",          # 20-day realized volatility
    "realvol_60d",          # 60-day realized volatility
    "vol_ratio_2060",       # short/long vol ratio (risk change signal)
    "atr_pct",              # ATR as % of price (normalized range)

    # Volume / liquidity
    "vol_ratio",            # volume vs 20-day average
    "amihud_20d",           # illiquidity ratio

    # Macro
    "Nifty50_return",       # market return (beta context)
    "India_10yr_yield",     # interest rate environment
    "USD_INR_return",       # currency pressure (daily)
    "USD_INR_return_20d",   # currency pressure (monthly)
    "Gold_return_20d",      # risk-off signal
    "Crude_return_20d",     # input cost / inflation signal

    # Fundamentals
    "forward_PE",           # valuation
    "profit_margin",        # profitability
    "operating_margin",     # operating efficiency
    "debt_to_equity",       # leverage
    "book_value",           # asset base
    "price_to_book",        # value factor
    "dividend_yield",       # income/value signal
    "beta",                 # market sensitivity
]

# Validate all feature cols exist
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"\n  WARNING — missing columns: {missing}")
else:
    print(f"\n  All {len(FEATURE_COLS)} feature columns confirmed present.")

TARGET_RAW   = "target_return_30d"   # regression target
TARGET_CS    = "target_cs_zscore"    # cross-sectional ranking target
META_COLS    = ["date", "Ticker", "close", "open", "high", "low", "volume"]

print(f"\n  Feature columns  : {len(FEATURE_COLS)}")
print(f"  Meta columns     : {META_COLS}")
print(f"  Targets          : {TARGET_RAW}, {TARGET_CS}")

# Print full feature list
print(f"\n  Full feature list:")
for i, col in enumerate(FEATURE_COLS):
    print(f"    {i+1:2}. {col}")

# ── 7. FINAL VALIDATION ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: FINAL VALIDATION")
print("=" * 60)

keep_cols = META_COLS + FEATURE_COLS + [TARGET_RAW, TARGET_CS]
keep_cols = [c for c in keep_cols if c in df.columns]
df_final = df[keep_cols].copy()

print(f"  Final shape     : {df_final.shape}")
print(f"  Tickers         : {df_final['Ticker'].nunique()}")
print(f"  Date range      : {df_final['date'].min().date()} → {df_final['date'].max().date()}")
print(f"  Total NaNs      : {df_final.isnull().sum().sum()}")
print(f"  % UP returns    : {(df_final[TARGET_RAW] > 0).mean()*100:.1f}%")
print(f"  % DOWN returns  : {(df_final[TARGET_RAW] <= 0).mean()*100:.1f}%")

print(f"\n  Target summary ({TARGET_RAW}):")
print(f"    Mean : {df_final[TARGET_RAW].mean():.4f}")
print(f"    Std  : {df_final[TARGET_RAW].std():.4f}")
print(f"    Min  : {df_final[TARGET_RAW].min():.4f}")
print(f"    Max  : {df_final[TARGET_RAW].max():.4f}")

print(f"\n  Target summary ({TARGET_CS}):")
print(f"    Mean : {df_final[TARGET_CS].mean():.4f}")
print(f"    Std  : {df_final[TARGET_CS].std():.4f}")
print(f"    Min  : {df_final[TARGET_CS].min():.4f}")
print(f"    Max  : {df_final[TARGET_CS].max():.4f}")

print(f"\n  Row count per ticker:")
print(df_final.groupby("Ticker").size().sort_values().to_string())

# ── 8. SAVE ────────────────────────────────────────────────────────────────────
print(f"\nSaving to {OUTPUT_FILE}...")
df_final.to_csv(OUTPUT_FILE, index=False)
print("Done. nifty50_v4_model_ready.csv saved.")
print("This file is ready for walk-forward model training.")