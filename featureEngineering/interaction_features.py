"""
V4 Interaction Features Script
Adds non-linear interaction features identified from EDA IC analysis.
These capture relationships individual features miss.

Run: python featureEngineering/interaction_features.py

Input : nifty50_v4_cs_normalized.csv  (monthly snapshot, CS normalized)
Output: nifty50_v4_interaction.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
from scipy import stats

INPUT_FILE  = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_v4_cs_normalized.csv"
OUTPUT_FILE = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_v4_interaction.csv"
PLOT_DIR    = "/Users/shahrukh/Desktop/stock/eda_output"
os.makedirs(PLOT_DIR, exist_ok=True)

TARGET     = "target_cs_zscore"
TARGET_RAW = "target_return_30d"

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading nifty50_v4_cs_normalized.csv...")
df = pd.read_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)
df = df.sort_values(["date", "Ticker"]).reset_index(drop=True)
print(f"  Shape   : {df.shape}")
print(f"  Tickers : {df['Ticker'].nunique()}")
print(f"  Months  : {df['date'].dt.to_period('M').nunique()}\n")

# ── ADD INTERACTION FEATURES ──────────────────────────────────────────────────
print("=" * 60)
print("ADDING INTERACTION FEATURES")
print("=" * 60)

# All features are already CS-normalized (mean=0, std=1 within each month)
# Interactions are computed on normalized values — this is correct

interactions = {}

# ── GROUP 1: VOL-ADJUSTED MOMENTUM ───────────────────────────────────────────
# Backed by Barroso & Santa-Clara (2015)
# Stocks with high momentum AND low volatility are the best picks
# High vol momentum stocks often reverse — this filters them out

# 6M momentum adjusted by 60-day realized vol
# Positive = strong upward momentum relative to its own volatility (Sharpe-like)
interactions["mom6m_vol_adj"] = df["return_6m"] / (df["realvol_60d"].abs() + 1e-8)

# 3M momentum adjusted by 20-day vol (short-term Sharpe)
interactions["mom3m_vol_adj"] = df["return_3m"] / (df["realvol_20d"].abs() + 1e-8)

# 12M momentum adjusted by 60-day vol (classic skip-month momentum factor)
interactions["mom12m_vol_adj"] = df["return_12m"] / (df["realvol_60d"].abs() + 1e-8)

print("  Added: mom6m_vol_adj   = return_6m / realvol_60d")
print("  Added: mom3m_vol_adj   = return_3m / realvol_20d")
print("  Added: mom12m_vol_adj  = return_12m / realvol_60d")

# ── GROUP 2: MEAN REVERSION STRENGTH ─────────────────────────────────────────
# dist_52w_high tells you HOW FAR below peak
# BB_pct tells you WHERE in the current band
# Combined: stocks far below 52w high AND at the bottom of their BB
# are the strongest mean reversion candidates

# dist_52w_high is negative (closer to 0 = near the high)
# BB_pct is 0-1 (lower = more oversold)
# We want: far below 52w high (very negative dist) AND low BB_pct
# Multiply: negative × low_BB_pct = more negative = stronger mean reversion signal
interactions["mean_rev_strength"] = df["dist_52w_high"] * (1 - df["BB_pct"])

# BB squeeze + momentum: when BB is narrow (low width) and momentum is positive
# → breakout candidate
interactions["bb_squeeze_mom"] = (1 / (df["BB_width"].abs() + 1e-8)) * df["return_20d"]

print("  Added: mean_rev_strength = dist_52w_high × (1 - BB_pct)")
print("  Added: bb_squeeze_mom    = (1/BB_width) × return_20d")

# ── GROUP 3: QUALITY-ADJUSTED MOMENTUM ───────────────────────────────────────
# High momentum stocks with good fundamentals outperform
# low momentum stocks regardless of fundamentals

# Momentum × profit quality: high 6M return + high profit margin
interactions["quality_momentum"] = df["return_6m"] * df["profit_margin"]

# Value × momentum: cheap stocks (low price_to_book) with positive momentum
# Classic value-momentum combination factor
interactions["value_momentum"] = df["return_3m"] * (-df["price_to_book"])
# Note: negative price_to_book so lower PB = more positive signal

print("  Added: quality_momentum  = return_6m × profit_margin")
print("  Added: value_momentum    = return_3m × (-price_to_book)")

# ── GROUP 4: RISK REGIME FEATURES ────────────────────────────────────────────
# Volatility ratio (20d/60d) tells if risk is rising or falling
# Combined with momentum: rising risk + negative momentum = avoid
# Falling risk + positive momentum = buy

# Vol trend × price trend
interactions["vol_trend_mom"] = df["vol_ratio_2060"] * df["return_20d"]

# ATR pct × momentum: high range stocks with strong momentum
interactions["atr_momentum"] = df["atr_pct"] * df["return_6m"]

print("  Added: vol_trend_mom     = vol_ratio_2060 × return_20d")
print("  Added: atr_momentum      = atr_pct × return_6m")

# ── GROUP 5: MACD CONFIRMATION ────────────────────────────────────────────────
# MACD signal confirms or contradicts momentum
# When both agree (both positive or both negative) = stronger signal

# MACD signal × 20d return: confirms trend direction
interactions["macd_mom_confirm"] = df["MACD_sig"] * df["return_20d"]

# MACD diff × vol: acceleration in low-vol environment = cleaner signal
interactions["macd_vol_confirm"] = df["MACD_diff"] * (1 / (df["realvol_20d"].abs() + 1e-8))

print("  Added: macd_mom_confirm  = MACD_sig × return_20d")
print("  Added: macd_vol_confirm  = MACD_diff / realvol_20d\n")

# ── ADD TO DATAFRAME ──────────────────────────────────────────────────────────
for name, series in interactions.items():
    df[name] = series

INTERACTION_COLS = list(interactions.keys())
print(f"  Total interaction features added: {len(INTERACTION_COLS)}")

# ── CROSS-SECTIONAL NORMALIZE THE NEW FEATURES ───────────────────────────────
print("\n" + "=" * 60)
print("CROSS-SECTIONAL NORMALIZING NEW FEATURES")
print("=" * 60)

df["year_month"] = df["date"].dt.to_period("M")

def cs_zscore_cols(group, cols):
    for col in cols:
        mean = group[col].mean()
        std  = group[col].std()
        if std > 1e-8:
            group[col] = (group[col] - mean) / std
        else:
            group[col] = 0.0
    return group

df = df.groupby("year_month", group_keys=False).apply(
    cs_zscore_cols, cols=INTERACTION_COLS
)
df.drop(columns=["year_month"], inplace=True)
print(f"  All {len(INTERACTION_COLS)} interaction features CS-normalized.\n")

# ── IC ANALYSIS OF NEW FEATURES ───────────────────────────────────────────────
print("=" * 60)
print("IC ANALYSIS — INTERACTION FEATURES vs BASE FEATURES")
print("=" * 60)

# Base features for comparison
base_features = [
    "return_6m", "realvol_60d", "dist_52w_high", "MACD_sig",
    "return_3m", "BB_pct", "return_12m", "realvol_20d"
]

all_features_to_check = base_features + INTERACTION_COLS
ic_scores = {}

for col in all_features_to_check:
    if col in df.columns:
        ic, _ = stats.spearmanr(df[col].fillna(0), df[TARGET])
        ic_scores[col] = ic

ic_series = pd.Series(ic_scores).sort_values(key=abs, ascending=False)

print(f"\n  {'Feature':<30} {'IC':>8}  {'Type'}")
print("  " + "-" * 55)
for feat, ic in ic_series.items():
    ftype = "INTERACTION" if feat in INTERACTION_COLS else "base"
    marker = " ←" if feat in INTERACTION_COLS and abs(ic) > 0.05 else ""
    print(f"  {feat:<30} {ic:>+8.4f}  {ftype}{marker}")

# Winsorize interaction features at 1st/99th percentile
print("\n" + "=" * 60)
print("WINSORIZING INTERACTION FEATURES")
print("=" * 60)

def winsorize(series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

for col in INTERACTION_COLS:
    df[col] = winsorize(df[col])
    print(f"  Winsorized: {col}")

# ── NULL CHECK ────────────────────────────────────────────────────────────────
print(f"\n  NaNs in new features: {df[INTERACTION_COLS].isnull().sum().sum()}")
df[INTERACTION_COLS] = df[INTERACTION_COLS].fillna(0)
print(f"  NaNs after fillna(0): {df[INTERACTION_COLS].isnull().sum().sum()}")
print(f"  Total NaNs in df    : {df.isnull().sum().sum()}")

# ── PLOT: IC COMPARISON ───────────────────────────────────────────────────────
print("\nGenerating IC comparison plot...")
fig, ax = plt.subplots(figsize=(14, 8))

colors_bar = []
for feat in ic_series.index:
    if feat in INTERACTION_COLS:
        colors_bar.append("#1D9E75")   # green for interaction
    else:
        colors_bar.append("#378ADD")   # blue for base

ax.barh(ic_series.index[::-1], ic_series.values[::-1],
        color=colors_bar[::-1], alpha=0.85)
ax.axvline(0, color="black", linewidth=1)
ax.axvline(0.05,  color="gray", linewidth=0.8, linestyle="--", label="|IC|=0.05")
ax.axvline(-0.05, color="gray", linewidth=0.8, linestyle="--")

# Legend patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#1D9E75", alpha=0.85, label="Interaction feature (new)"),
    Patch(facecolor="#378ADD", alpha=0.85, label="Base feature"),
]
ax.legend(handles=legend_elements)
ax.set_title("IC comparison: interaction features (green) vs base features (blue)")
ax.set_xlabel("Spearman IC vs target_cs_zscore")
plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, "08_interaction_feature_ic.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved: 08_interaction_feature_ic.png")

# ── FINAL FEATURE LIST ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL FEATURE LIST")
print("=" * 60)

BASE_FEATURES = [
    "price_to_ma200", "BB_width", "BB_pct",
    "return_1d", "return_5d", "return_10d", "return_20d",
    "return_3m", "return_6m", "return_12m",
    "dist_52w_high", "MACD_sig", "MACD_diff",
    "realvol_20d", "realvol_60d", "vol_ratio_2060", "atr_pct",
    "vol_ratio",
    "forward_PE", "profit_margin", "operating_margin",
    "debt_to_equity", "book_value", "price_to_book",
    "dividend_yield", "beta"
]

ALL_FEATURES = BASE_FEATURES + INTERACTION_COLS
missing = [c for c in ALL_FEATURES if c not in df.columns]
if missing:
    print(f"  WARNING: Missing: {missing}")
    ALL_FEATURES = [c for c in ALL_FEATURES if c in df.columns]

print(f"\n  Base features       : {len(BASE_FEATURES)}")
print(f"  Interaction features: {len(INTERACTION_COLS)}")
print(f"  Total features      : {len(ALL_FEATURES)}")

print(f"\n  Interaction features:")
for i, col in enumerate(INTERACTION_COLS):
    ic_val = ic_scores.get(col, 0)
    print(f"    {i+1:2}. {col:<30} IC: {ic_val:+.4f}")

# Save updated feature list
feature_list_path = "/Users/shahrukh/Desktop/stock/nifty50_data/feature_cols_with_interactions.txt"
with open(feature_list_path, "w") as f:
    for col in ALL_FEATURES:
        f.write(col + "\n")
print(f"\n  Feature list saved: feature_cols_with_interactions.txt")

# ── FINAL VALIDATION ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

META_COLS  = ["date", "Ticker", "close", "open", "high", "low", "volume"]
keep_cols  = META_COLS + ALL_FEATURES + [TARGET_RAW, TARGET]
keep_cols  = [c for c in keep_cols if c in df.columns]
df_final   = df[keep_cols].copy()

print(f"  Final shape   : {df_final.shape}")
print(f"  Total NaNs    : {df_final.isnull().sum().sum()}")
print(f"  Tickers       : {df_final['Ticker'].nunique()}")
assert df_final.isnull().sum().sum() == 0, "NaNs present!"
print("  Zero NaN assertion passed ✓")

# ── SAVE ──────────────────────────────────────────────────────────────────────
print(f"\nSaving to {OUTPUT_FILE}...")
df_final.to_csv(OUTPUT_FILE, index=False)
print("Done. nifty50_v4_interaction.csv saved.")
print(f"\nSummary:")
print(f"  Input  : nifty50_v4_cs_normalized.csv ({df.shape[0]:,} rows, 26 base features)")
print(f"  Output : nifty50_v4_interaction.csv   ({df_final.shape[0]:,} rows, {len(ALL_FEATURES)} features)")
print(f"  New interaction features: {len(INTERACTION_COLS)}")
print(f"\nNext step: re-run tune_v4.py pointing at nifty50_v4_interaction.csv")