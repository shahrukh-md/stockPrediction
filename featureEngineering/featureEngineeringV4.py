"""
V4 Feature Engineering Script
Adds 9 new features on top of nifty50_cleaned.csv for ranked stock selection.

Run: python feature_engineering_v4.py

Output: nifty50_v4_features.csv
"""

import pandas as pd
import numpy as np

INPUT_FILE  = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_cleaned.csv"
OUTPUT_FILE = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_v4_features.csv"

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("Loading nifty50_cleaned.csv...")
df = pd.read_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)
df = df.sort_values(["Ticker", "date"]).reset_index(drop=True)
print(f"  Shape: {df.shape}")
print(f"  Tickers: {df['Ticker'].nunique()}")
print(f"  Date range: {df['date'].min()} → {df['date'].max()}\n")

# ── 2. ADD V4 FEATURES PER STOCK ──────────────────────────────────────────────
print("Adding V4 features per stock...")

results = []

for ticker, g in df.groupby("Ticker"):
    g = g.copy().sort_values("date").reset_index(drop=True)

    # ── MOMENTUM ──────────────────────────────────────────────────────────────
    # 3M (~63 trading days)
    g["return_3m"] = g["close"].pct_change(63)

    # 6M (~126 trading days)
    g["return_6m"] = g["close"].pct_change(126)

    # 12M excluding last month (~252 days total, skip last 21 days)
    # Classic Jegadeesh-Titman: return from t-252 to t-21
    g["return_12m"] = g["close"].shift(21).pct_change(231)

    # ── 52-WEEK HIGH DISTANCE ─────────────────────────────────────────────────
    # How far below the 52-week high is the current price (negative = below)
    rolling_52w_high = g["close"].rolling(252, min_periods=126).max()
    g["dist_52w_high"] = (g["close"] - rolling_52w_high) / rolling_52w_high

    # ── REALIZED VOLATILITY ───────────────────────────────────────────────────
    daily_ret = g["close"].pct_change()

    # 20-day realized vol (annualized)
    g["realvol_20d"] = daily_ret.rolling(20, min_periods=15).std() * np.sqrt(252)

    # 60-day realized vol (annualized)
    g["realvol_60d"] = daily_ret.rolling(60, min_periods=45).std() * np.sqrt(252)

    # Volatility ratio: rising ratio = increasing short-term risk
    g["vol_ratio_2060"] = g["realvol_20d"] / g["realvol_60d"].replace(0, np.nan)

    # ── AMIHUD ILLIQUIDITY ────────────────────────────────────────────────────
    # |daily_return| / rupee_volume, 20-day average
    # Higher = less liquid = harder to trade without moving the price
    rupee_vol = (g["close"] * g["volume"]).replace(0, np.nan)
    illiq = daily_ret.abs() / rupee_vol
    g["amihud_20d"] = illiq.rolling(20, min_periods=15).mean()

    # ── EXCESS RETURN VS NIFTY ────────────────────────────────────────────────
    # Stock's 20-day return minus Nifty's return over same period
    # Positive = outperforming the market
    g["excess_return_1m"] = g["return_20d"] - g["Nifty50_return"]

    # ── CROSS-SECTIONAL RANK FEATURES (added later in pipeline) ──────────────
    # Note: cross-sectional zscore of target is done at the monthly level
    # after all stocks are combined — done in step 4 below

    results.append(g)
    print(f"  {ticker}: done ({len(g)} rows)")

df = pd.concat(results, ignore_index=True)
print(f"\nAll features added. Shape: {df.shape}")

# ── 3. WINSORIZE NEW FEATURES ─────────────────────────────────────────────────
print("\nWinsorizing new features at 1st/99th percentile...")

def winsorize(series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    clipped = series.clip(lo, hi)
    return clipped

new_features = [
    "return_3m", "return_6m", "return_12m",
    "dist_52w_high", "realvol_20d", "realvol_60d",
    "vol_ratio_2060", "amihud_20d", "excess_return_1m"
]

for col in new_features:
    before_nulls = df[col].isnull().sum()
    df[col] = winsorize(df[col].dropna().reindex(df.index))
    print(f"  {col}: winsorized (nulls: {before_nulls})")

# ── 4. CROSS-SECTIONAL ZSCORE OF TARGET ──────────────────────────────────────
# Within each month, zscore the target_return_30d across all stocks
# This makes the model learn relative ranking within each period
print("\nAdding cross-sectional zscore of target per month...")

df["year_month"] = df["date"].dt.to_period("M")

def cross_sectional_zscore(group):
    mean = group["target_return_30d"].mean()
    std  = group["target_return_30d"].std()
    if std == 0 or pd.isna(std):
        group["target_cs_zscore"] = 0.0
    else:
        group["target_cs_zscore"] = (group["target_return_30d"] - mean) / std
    return group

df = df.groupby("year_month", group_keys=False).apply(cross_sectional_zscore)
df.drop(columns=["year_month"], inplace=True)

print(f"  target_cs_zscore added")
print(f"  Mean: {df['target_cs_zscore'].mean():.4f} (should be ~0)")
print(f"  Std : {df['target_cs_zscore'].std():.4f} (should be ~1)")

# ── 5. CROSS-SECTIONAL STOCK COVERAGE CHECK ───────────────────────────────────
print("\n=== STOCK COVERAGE PER MONTH (sample) ===")
coverage = df.groupby(df["date"].dt.to_period("M"))["Ticker"].count()
print(f"  Min stocks in a month : {coverage.min()}")
print(f"  Max stocks in a month : {coverage.max()}")
print(f"  Mean stocks per month : {coverage.mean():.1f}")
low_coverage = coverage[coverage < 40]
if len(low_coverage) > 0:
    print(f"\n  Months with < 40 stocks:")
    print(low_coverage.to_string())
else:
    print("  All months have >= 40 stocks — good.")

# ── 6. NULL AUDIT ON NEW FEATURES ─────────────────────────────────────────────
print("\n=== NULL AUDIT ON NEW FEATURES ===")
for col in new_features + ["target_cs_zscore"]:
    n = df[col].isnull().sum()
    pct = n / len(df) * 100
    print(f"  {col:25s}: {n:6,} nulls ({pct:.2f}%)")

# ── 7. HANDLE REMAINING NaNs IN NEW FEATURES ──────────────────────────────────
# These come from the warmup period of rolling calculations
# Forward-fill within each stock, then drop rows that still can't be filled
print("\nForward-filling remaining NaNs in new features...")
df = df.sort_values(["Ticker", "date"]).reset_index(drop=True)

for col in new_features:
    df[col] = df.groupby("Ticker")[col].transform(lambda x: x.ffill())

before = len(df)
df.dropna(subset=new_features, inplace=True)
print(f"  Dropped {before - len(df):,} rows (warmup period NaNs)")
print(f"  Final shape: {df.shape}")
print(f"  Tickers: {df['Ticker'].nunique()}")

# ── 8. FINAL VALIDATION ────────────────────────────────────────────────────────
print("\n=== FINAL NULL CHECK ===")
total_nulls = df.isnull().sum().sum()
print(f"  Total NaNs: {total_nulls}")

print("\n=== FINAL FEATURE LIST ({} columns) ===".format(len(df.columns)))
existing   = [c for c in df.columns if c not in new_features + ["target_cs_zscore"]]
added      = new_features + ["target_cs_zscore"]
print(f"\n  Existing features ({len(existing)}):")
for c in existing:
    print(f"    {c}")
print(f"\n  New V4 features ({len(added)}):")
for c in added:
    print(f"    {c}  ← NEW")

print("\n=== TARGET SUMMARY ===")
print(f"  target_return_30d  mean : {df['target_return_30d'].mean():.4f}")
print(f"  target_return_30d  std  : {df['target_return_30d'].std():.4f}")
print(f"  target_cs_zscore   mean : {df['target_cs_zscore'].mean():.4f}")
print(f"  target_cs_zscore   std  : {df['target_cs_zscore'].std():.4f}")
print(f"  % positive returns      : {(df['target_return_30d'] > 0).mean()*100:.1f}%")

print("\n=== ROW COUNT PER TICKER ===")
print(df.groupby("Ticker").size().sort_values().to_string())

# ── 9. SAVE ────────────────────────────────────────────────────────────────────
print(f"\nSaving to {OUTPUT_FILE}...")
df.to_csv(OUTPUT_FILE, index=False)
print("Done. File saved as nifty50_v4_features.csv")
print(f"Ready for EDA and model training.")