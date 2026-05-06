"""
Cross-Sectional Normalization Script
Normalizes all 33 features cross-sectionally (across stocks within each month)
so the model learns relative rankings, not absolute levels.

Run: python models/cross_sectional_norm.py

Input : nifty50_v4_model_ready.csv
Output: nifty50_v4_cs_normalized.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os

INPUT_FILE  = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_v4_model_ready.csv"
OUTPUT_FILE = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_v4_cs_normalized.csv"
PLOT_DIR    = "/Users/shahrukh/Desktop/stock/eda_output"
os.makedirs(PLOT_DIR, exist_ok=True)

FEATURE_COLS = [
    "price_to_ma200", "BB_width", "BB_pct",
    "return_1d", "return_5d", "return_10d", "return_20d",
    "return_3m", "return_6m", "return_12m",
    "dist_52w_high", "MACD_sig", "MACD_diff",
    "realvol_20d", "realvol_60d", "vol_ratio_2060", "atr_pct",
    "vol_ratio", "amihud_20d",
    "Nifty50_return", "India_10yr_yield",
    "USD_INR_return", "USD_INR_return_20d",
    "Gold_return_20d", "Crude_return_20d",
    "forward_PE", "profit_margin", "operating_margin",
    "debt_to_equity", "book_value", "price_to_book",
    "dividend_yield", "beta"
]

META_COLS    = ["date", "Ticker", "close", "open", "high", "low", "volume"]
TARGET_RAW   = "target_return_30d"
TARGET_CS    = "target_cs_zscore"

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("Loading nifty50_v4_model_ready.csv...")
df = pd.read_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)
df = df.sort_values(["date", "Ticker"]).reset_index(drop=True)
print(f"  Shape   : {df.shape}")
print(f"  Tickers : {df['Ticker'].nunique()}")
print(f"  Dates   : {df['date'].min().date()} → {df['date'].max().date()}\n")

# ── 2. EXTRACT MONTH-END SNAPSHOT ─────────────────────────────────────────────
# One row per stock per month (last trading day of each month)
# This fixes the overlapping 30-day window problem
print("=" * 60)
print("STEP 2: EXTRACT MONTH-END SNAPSHOT")
print("=" * 60)

df["year_month"] = df["date"].dt.to_period("M")
df_monthly = (
    df.groupby(["year_month", "Ticker"], as_index=False)
    .last()
    .sort_values(["year_month", "Ticker"])
    .reset_index(drop=True)
)

print(f"  Daily rows   : {len(df):,}")
print(f"  Monthly rows : {len(df_monthly):,}  (one per stock per month)")
print(f"  Months       : {df_monthly['year_month'].nunique()}")
print(f"  Avg stocks/month: {df_monthly.groupby('year_month')['Ticker'].count().mean():.1f}")

# Check coverage
coverage = df_monthly.groupby("year_month")["Ticker"].count()
low_months = coverage[coverage < 40]
print(f"  Months with < 40 stocks: {len(low_months)}")
if len(low_months) > 0:
    print(f"    {low_months.to_string()}")

# ── 3. CROSS-SECTIONAL ZSCORE NORMALIZATION ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: CROSS-SECTIONAL ZSCORE NORMALIZATION")
print("=" * 60)
print("  For each month: zscore each feature across all stocks")
print("  Formula: (stock_value - monthly_mean) / monthly_std\n")

df_norm = df_monthly.copy()

# Store pre-normalization stats for validation
pre_stats = {}
for col in FEATURE_COLS:
    pre_stats[col] = {
        "mean": df_monthly[col].mean(),
        "std":  df_monthly[col].std(),
        "min":  df_monthly[col].min(),
        "max":  df_monthly[col].max(),
    }

# Apply cross-sectional zscore per month for each feature
def cs_zscore_month(group, cols):
    for col in cols:
        mean = group[col].mean()
        std  = group[col].std()
        if std > 1e-8:
            group[col] = (group[col] - mean) / std
        else:
            group[col] = 0.0  # flat feature this month — no signal
    return group

df_norm = df_norm.groupby("year_month", group_keys=False).apply(
    cs_zscore_month, cols=FEATURE_COLS
)

print("  Normalization complete.")

# ── 4. VALIDATE NORMALIZATION ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: VALIDATION")
print("=" * 60)

print(f"\n  {'Feature':<25} {'Pre-mean':>10} {'Pre-std':>10} {'Post-mean':>10} {'Post-std':>10} {'Status'}")
print("  " + "-" * 75)

all_ok = True
for col in FEATURE_COLS:
    pre_mean = pre_stats[col]["mean"]
    pre_std  = pre_stats[col]["std"]
    post_mean = df_norm[col].mean()
    post_std  = df_norm[col].std()

    # After CS normalization: mean should be ~0, std should be ~1
    mean_ok = abs(post_mean) < 0.05
    std_ok  = abs(post_std - 1.0) < 0.15
    status  = "✓" if (mean_ok and std_ok) else "⚠ CHECK"
    if not (mean_ok and std_ok):
        all_ok = False

    print(f"  {col:<25} {pre_mean:>10.3f} {pre_std:>10.3f} {post_mean:>10.4f} {post_std:>10.4f}  {status}")

print()
if all_ok:
    print("  All features normalized correctly (mean ≈ 0, std ≈ 1)")
else:
    print("  Some features may need review — check ⚠ rows above")

# ── 5. ALSO NORMALIZE THE RAW TARGET (for model training) ─────────────────────
print("\n" + "=" * 60)
print("STEP 5: TARGET NORMALIZATION CHECK")
print("=" * 60)

# target_cs_zscore already computed — verify it's still correct after month-end snapshot
target_mean = df_norm[TARGET_CS].mean()
target_std  = df_norm[TARGET_CS].std()
print(f"  target_cs_zscore  mean: {target_mean:.4f}  (should be ~0)")
print(f"  target_cs_zscore  std : {target_std:.4f}   (should be ~1)")
print(f"  target_return_30d mean: {df_norm[TARGET_RAW].mean():.4f}")
print(f"  target_return_30d std : {df_norm[TARGET_RAW].std():.4f}")

# ── 6. MACRO FEATURES NOTE ────────────────────────────────────────────────────
# Nifty50_return, India_10yr_yield, USD_INR_return etc. are the same for all
# stocks in a given month. After CS normalization, they'll have std=0 for that
# month (since all stocks share the same macro value).
# These will be zeroed out per-month by CS norm — which is correct behaviour:
# they provide no cross-sectional discriminating power within a month.
# Their value is captured implicitly through the stock-level features they
# influence (e.g., high Nifty return affects return_20d for all stocks).
# We keep them as context features but flag them.

print("\n" + "=" * 60)
print("STEP 6: MACRO FEATURES AFTER CS NORMALIZATION")
print("=" * 60)

macro_features = [
    "Nifty50_return", "India_10yr_yield",
    "USD_INR_return", "USD_INR_return_20d",
    "Gold_return_20d", "Crude_return_20d"
]

print("\n  Macro features are identical across stocks within a month.")
print("  After CS normalization, they become 0 for every stock in that month.")
print("  This is expected — they carry no cross-sectional signal.")
print("  They will be DROPPED from the model feature set.\n")

for col in macro_features:
    post_mean = df_norm[col].mean()
    post_std  = df_norm[col].std()
    print(f"  {col:<25}: mean={post_mean:.6f}, std={post_std:.6f}  → drop")

# Drop macro features from normalized feature set
df_norm.drop(columns=macro_features, inplace=True)
FEATURE_COLS_FINAL = [c for c in FEATURE_COLS if c not in macro_features]
print(f"\n  Features after dropping macro: {len(FEATURE_COLS_FINAL)}")
print(f"  Final feature list: {FEATURE_COLS_FINAL}")

# ── 7. FINAL SHAPE AND NULL CHECK ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: FINAL VALIDATION")
print("=" * 60)

total_nulls = df_norm.isnull().sum().sum()
print(f"\n  Final shape     : {df_norm.shape}")
print(f"  Total NaNs      : {total_nulls}")
print(f"  Tickers         : {df_norm['Ticker'].nunique()}")
print(f"  Months          : {df_norm['year_month'].nunique()}")
print(f"  Date range      : {df_norm['date'].min().date()} → {df_norm['date'].max().date()}")
print(f"  Feature columns : {len(FEATURE_COLS_FINAL)}")
print(f"  % UP returns    : {(df_norm[TARGET_RAW] > 0).mean()*100:.1f}%")
assert total_nulls == 0, f"NaNs present: {total_nulls}"
print("  Zero NaN assertion passed ✓")

# ── 8. PLOT: BEFORE vs AFTER NORMALIZATION ────────────────────────────────────
print("\nGenerating before/after normalization plots...")

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
sample_features = [
    "return_3m", "realvol_60d", "dist_52w_high",
    "MACD_sig", "forward_PE", "price_to_book",
    "amihud_20d", "beta", "BB_pct"
]

for i, feat in enumerate(sample_features):
    ax = axes[i]
    # Pre-norm distribution (from original df_monthly)
    pre_vals = df_monthly[feat].dropna()
    post_vals = df_norm[feat].dropna()
    ax.hist(pre_vals, bins=60, alpha=0.5, color="#E24B4A", label="Before", density=True)
    ax.hist(post_vals, bins=60, alpha=0.5, color="#378ADD", label="After CS norm", density=True)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(feat, fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylabel("Density")

plt.suptitle("Before vs after cross-sectional normalization (sample features)", fontsize=13)
plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, "07_cs_normalization_before_after.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved: 07_cs_normalization_before_after.png")

# ── 9. SAVE FINAL FEATURE LIST FOR TRAINING ───────────────────────────────────
feature_list_path = "/Users/shahrukh/Desktop/stock/nifty50_data/feature_cols_final.txt"
with open(feature_list_path, "w") as f:
    for col in FEATURE_COLS_FINAL:
        f.write(col + "\n")
print(f"  Saved feature list to: feature_cols_final.txt")

# ── 10. SAVE ───────────────────────────────────────────────────────────────────
print(f"\nSaving to {OUTPUT_FILE}...")
df_norm.drop(columns=["year_month"], inplace=True)
df_norm.to_csv(OUTPUT_FILE, index=False)
print("Done. nifty50_v4_cs_normalized.csv saved.")
print("\nSummary:")
print(f"  Input rows (daily)    : 145,837")
print(f"  Output rows (monthly) : {len(df_norm):,}  ← one per stock per month")
print(f"  Features              : {len(FEATURE_COLS_FINAL)}  ← macro features correctly dropped")
print(f"  Targets               : target_return_30d, target_cs_zscore")
print(f"  Ready for             : walk-forward model training")