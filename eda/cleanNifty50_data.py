"""
Nifty50 V4 Cleaning Script
Run from terminal: python clean_nifty50_v4.py
Output: nifty50_v4_clean.csv in the same folder as input
"""

import pandas as pd
import numpy as np
import os

INPUT_FILE  = os.path.expanduser("~/Desktop/stock/nifty50_data/nifty50_full_dataset.csv")
OUTPUT_FILE = os.path.expanduser("~/Desktop/stock/nifty50_data/nifty50_v4_clean.csv")

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)
print(f"  Raw shape: {df.shape}")
print(f"  Columns  : {df.columns.tolist()}\n")

# ── 2. AUDIT ──────────────────────────────────────────────────────────────────
print("=== NULL AUDIT (before cleaning) ===")
null_counts = df.isnull().sum()
null_pct    = (null_counts / len(df) * 100).round(2)
audit = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
print(audit[audit.null_count > 0].to_string())
print()

print("=== ROW COUNT PER TICKER ===")
print(df.groupby("Ticker").size().sort_values().to_string())
print()

print(f"Date range : {df['date'].min()} → {df['date'].max()}")
print(f"Tickers    : {df['Ticker'].nunique()}\n")

# ── 3. FIX DTYPES ─────────────────────────────────────────────────────────────
print("Fixing dtypes...")
df["Ticker"] = df["Ticker"].str.strip()
df = df.sort_values(["Ticker", "date"]).reset_index(drop=True)

numeric_cols = df.columns.difference(["date", "Ticker"])
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ── 4. DROP COLUMNS NOT NEEDED FOR V4 ─────────────────────────────────────────
drop_cols = []

# Drop old binary classification target if it exists
for c in ["target", "target_label", "label", "direction", "binary_target"]:
    if c in df.columns:
        drop_cols.append(c)
        print(f"  Dropping binary target column: {c}")

# Drop target_price_30d — we only need the return, not the price
if "target_price_30d" in df.columns:
    drop_cols.append("target_price_30d")
    print("  Dropping target_price_30d (keeping target_return_30d)")

if drop_cols:
    df.drop(columns=drop_cols, inplace=True)

# ── 5. VALIDATE target_return_30d ─────────────────────────────────────────────
print("\n=== TARGET VALIDATION ===")
print(f"  target_return_30d null count : {df['target_return_30d'].isnull().sum()}")
print(f"  target_return_30d mean       : {df['target_return_30d'].mean():.4f}")
print(f"  target_return_30d std        : {df['target_return_30d'].std():.4f}")
print(f"  target_return_30d min        : {df['target_return_30d'].min():.4f}")
print(f"  target_return_30d max        : {df['target_return_30d'].max():.4f}")
print(f"  % positive returns           : {(df['target_return_30d'] > 0).mean()*100:.1f}%")

# ── 6. FORWARD-FILL WITHIN EACH STOCK, THEN DROP REMAINING NaNs ───────────────
print("\nForward-filling missing values within each stock...")
before = df.isnull().sum().sum()
df = df.groupby("Ticker", group_keys=False).apply(lambda g: g.ffill())
after_ffill = df.isnull().sum().sum()
print(f"  NaNs before ffill : {before:,}")
print(f"  NaNs after  ffill : {after_ffill:,}")

# Drop rows that still have NaNs (start-of-history rows that couldn't be filled)
df.dropna(inplace=True)
print(f"  NaNs after  drop  : {df.isnull().sum().sum()}")
print(f"  Shape after clean : {df.shape}")

# ── 7. WINSORIZE OUTLIERS ──────────────────────────────────────────────────────
print("\nWinsorizing outliers...")

def winsorize(series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

# Target: winsorize at 1st/99th percentile
df["target_return_30d"] = winsorize(df["target_return_30d"], 0.01, 0.99)
print("  target_return_30d winsorized at 1st/99th pct")

# MACD can be unbounded — winsorize tighter
for col in ["MACD", "MACD_diff"]:
    if col in df.columns:
        df[col] = winsorize(df[col], 0.005, 0.995)
        print(f"  {col} winsorized at 0.5th/99.5th pct")

# Volume ratio outliers
if "vol_ratio" in df.columns:
    df["vol_ratio"] = winsorize(df["vol_ratio"], 0.01, 0.99)
    print("  vol_ratio winsorized at 1st/99th pct")

# Fundamental outliers — PE can be negative/extreme
for col in ["PE_ratio", "forward_PE"]:
    if col in df.columns:
        df[col] = winsorize(df[col], 0.01, 0.99)
        print(f"  {col} winsorized at 1st/99th pct")

# ── 8. ADD KEY V4 FEATURES ────────────────────────────────────────────────────
print("\nAdding V4 features...")

df = df.sort_values(["Ticker", "date"]).reset_index(drop=True)

for ticker, g in df.groupby("Ticker"):
    idx = g.index

    # Momentum: 3M (~63 days), 6M (~126 days), 12M excluding last month (~231 days)
    df.loc[idx, "return_3m"]  = g["close"].pct_change(63)
    df.loc[idx, "return_6m"]  = g["close"].pct_change(126)
    df.loc[idx, "return_12m"] = g["close"].shift(21).pct_change(231)  # skip last month

    # 52-week high distance
    rolling_52w = g["close"].rolling(252, min_periods=126)
    df.loc[idx, "dist_52w_high"] = (g["close"] - rolling_52w.max()) / rolling_52w.max()

    # Realized volatility (annualized)
    daily_ret = g["close"].pct_change()
    df.loc[idx, "realvol_20d"] = daily_ret.rolling(20).std() * np.sqrt(252)
    df.loc[idx, "realvol_60d"] = daily_ret.rolling(60).std() * np.sqrt(252)

    # Volatility ratio
    df.loc[idx, "vol_ratio_2060"] = df.loc[idx, "realvol_20d"] / df.loc[idx, "realvol_60d"].replace(0, np.nan)

    # Amihud illiquidity: |daily_return| / (close * volume), 20-day average
    illiq = (daily_ret.abs() / (g["close"] * g["volume"]).replace(0, np.nan))
    df.loc[idx, "amihud_20d"] = illiq.rolling(20).mean()

    # Excess return vs Nifty (stock return - Nifty return)
    df.loc[idx, "excess_return_1m"] = g["return_20d"] - g["Nifty50_return"]

print("  return_3m, return_6m, return_12m")
print("  dist_52w_high")
print("  realvol_20d, realvol_60d, vol_ratio_2060")
print("  amihud_20d")
print("  excess_return_1m")

# ── 9. FINAL DROP OF NaNs INTRODUCED BY NEW FEATURES ─────────────────────────
before = len(df)
df.dropna(inplace=True)
print(f"\n  Dropped {before - len(df):,} rows after new feature NaNs")
print(f"  Final shape: {df.shape}")
print(f"  Tickers    : {df['Ticker'].nunique()}")
print(f"  Date range : {df['date'].min()} → {df['date'].max()}")

# ── 10. FINAL VALIDATION ───────────────────────────────────────────────────────
print("\n=== FINAL NULL CHECK ===")
remaining_nulls = df.isnull().sum().sum()
print(f"  Total NaNs: {remaining_nulls}")
assert remaining_nulls == 0, "NaNs still present — check above!"

print("\n=== TARGET DISTRIBUTION (final) ===")
print(f"  Mean   : {df['target_return_30d'].mean():.4f}")
print(f"  Std    : {df['target_return_30d'].std():.4f}")
print(f"  Min    : {df['target_return_30d'].min():.4f}")
print(f"  Max    : {df['target_return_30d'].max():.4f}")
print(f"  % UP   : {(df['target_return_30d'] > 0).mean()*100:.1f}%")
print(f"  % DOWN : {(df['target_return_30d'] <= 0).mean()*100:.1f}%")

print("\n=== FINAL COLUMNS ===")
for i, col in enumerate(df.columns.tolist()):
    print(f"  {i+1:3}. {col}")

# ── 11. SAVE ───────────────────────────────────────────────────────────────────
print(f"\nSaving to {OUTPUT_FILE} ...")
df.to_csv(OUTPUT_FILE, index=False)
print("Done.")