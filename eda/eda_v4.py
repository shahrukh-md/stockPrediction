"""
Full EDA Script — nifty50_v4_features.csv
Generates all plots and saves them to a dedicated EDA folder.

Run: python eda_v4.py

Output: ~/Desktop/stock/eda_output/ (all plots as PNG)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore")
import os
from scipy import stats

# ── CONFIG ────────────────────────────────────────────────────────────────────
FILE       = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_v4_features.csv"
OUTPUT_DIR = "/Users/shahrukh/Desktop/stock/eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}")

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading nifty50_v4_features.csv...")
df = pd.read_csv(FILE, parse_dates=["date"], low_memory=False)
df = df.sort_values(["Ticker", "date"]).reset_index(drop=True)
print(f"  Shape   : {df.shape}")
print(f"  Tickers : {df['Ticker'].nunique()}")
print(f"  Dates   : {df['date'].min().date()} → {df['date'].max().date()}\n")

FEATURE_COLS = [
    "MA_20","MA_50","MA_200","EMA_20","RSI_14",
    "MACD","MACD_sig","MACD_diff","BB_width","BB_pct",
    "return_1d","return_5d","return_10d","return_20d",
    "vol_ratio","ATR_14",
    "USD_INR","Nifty50_return","India_10yr_yield",
    "Gold_price","Crude_oil_price",
    "PE_ratio","forward_PE","profit_margin","operating_margin",
    "debt_to_equity","book_value","price_to_book","dividend_yield","beta",
    "return_3m","return_6m","return_12m",
    "dist_52w_high","realvol_20d","realvol_60d",
    "vol_ratio_2060","amihud_20d","excess_return_1m"
]
TARGET = "target_return_30d"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATASET HEALTH
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1: DATASET HEALTH")
print("=" * 60)

# 1a. Row count per ticker
fig, ax = plt.subplots(figsize=(14, 6))
counts = df.groupby("Ticker").size().sort_values()
colors = ["#E24B4A" if v < 2000 else "#378ADD" for v in counts.values]
ax.barh(counts.index, counts.values, color=colors, height=0.7)
ax.axvline(counts.mean(), color="black", linestyle="--", linewidth=1, label=f"Mean: {counts.mean():.0f}")
ax.set_xlabel("Row count")
ax.set_title("Row count per ticker")
ax.legend()
for i, (ticker, val) in enumerate(counts.items()):
    ax.text(val + 20, i, str(val), va="center", fontsize=8)
plt.tight_layout()
save(fig, "01a_row_count_per_ticker.png")

# 1b. Data timeline per ticker
fig, ax = plt.subplots(figsize=(14, 8))
tickers_sorted = df.groupby("Ticker")["date"].min().sort_values().index
for i, ticker in enumerate(tickers_sorted):
    sub = df[df["Ticker"] == ticker]
    ax.barh(i, (sub["date"].max() - sub["date"].min()).days,
            left=sub["date"].min(), height=0.6, color="#378ADD", alpha=0.7)
ax.set_yticks(range(len(tickers_sorted)))
ax.set_yticklabels(tickers_sorted, fontsize=8)
ax.set_xlabel("Date")
ax.set_title("Data timeline per ticker")
plt.tight_layout()
save(fig, "01b_timeline_per_ticker.png")

print("  Section 1 done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TARGET ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 2: TARGET ANALYSIS")
print("=" * 60)

# 2a. Overall return distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(df[TARGET], bins=100, color="#378ADD", alpha=0.8, edgecolor="white", linewidth=0.3)
ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
ax.axvline(df[TARGET].mean(), color="#E24B4A", linewidth=1.5, linestyle="-", label=f"Mean: {df[TARGET].mean():.3f}")
ax.set_title("Distribution of 30-day forward return")
ax.set_xlabel("30-day forward return")
ax.set_ylabel("Frequency")
ax.legend()
pct_up = (df[TARGET] > 0).mean() * 100
ax.text(0.98, 0.95, f"UP: {pct_up:.1f}%\nDOWN: {100-pct_up:.1f}%",
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

ax = axes[1]
ax.hist(df["target_cs_zscore"], bins=100, color="#1D9E75", alpha=0.8, edgecolor="white", linewidth=0.3)
ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
ax.set_title("Cross-sectional zscore distribution")
ax.set_xlabel("target_cs_zscore")
ax.set_ylabel("Frequency")

plt.tight_layout()
save(fig, "02a_target_distribution.png")

# 2b. Return distribution by year
fig, ax = plt.subplots(figsize=(14, 6))
df["year"] = df["date"].dt.year
years = sorted(df["year"].unique())
data_by_year = [df[df["year"] == y][TARGET].values for y in years]
bp = ax.boxplot(data_by_year, labels=years, patch_artist=True, showfliers=False)
for patch in bp["boxes"]:
    patch.set_facecolor("#B5D4F4")
    patch.set_alpha(0.8)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.set_title("30-day forward return distribution by year")
ax.set_xlabel("Year")
ax.set_ylabel("30-day forward return")
plt.xticks(rotation=45)
plt.tight_layout()
save(fig, "02b_return_by_year.png")
df.drop(columns=["year"], inplace=True)

# 2c. Cross-sectional return spread per month
fig, ax = plt.subplots(figsize=(14, 5))
monthly_std = df.groupby(df["date"].dt.to_period("M"))[TARGET].std()
monthly_std.index = monthly_std.index.to_timestamp()
ax.fill_between(monthly_std.index, monthly_std.values, alpha=0.4, color="#378ADD")
ax.plot(monthly_std.index, monthly_std.values, color="#378ADD", linewidth=0.8)
ax.axhline(monthly_std.mean(), color="#E24B4A", linestyle="--", linewidth=1.5,
           label=f"Mean spread: {monthly_std.mean():.3f}")
ax.set_title("Cross-sectional return spread per month (std of returns across 48 stocks)")
ax.set_xlabel("Date")
ax.set_ylabel("Std of 30-day returns")
ax.legend()
plt.tight_layout()
save(fig, "02c_cross_sectional_spread.png")

# 2d. Return autocorrelation (average across stocks)
fig, ax = plt.subplots(figsize=(10, 5))
all_acf = []
for ticker, g in df.groupby("Ticker"):
    ret = g.set_index("date")[TARGET].resample("ME").last()
    if len(ret) > 30:
        acf_vals = [ret.autocorr(lag=i) for i in range(1, 13)]
        all_acf.append(acf_vals)
mean_acf = np.nanmean(all_acf, axis=0)
lags = range(1, 13)
colors_acf = ["#E24B4A" if v < 0 else "#378ADD" for v in mean_acf]
ax.bar(lags, mean_acf, color=colors_acf, alpha=0.8)
ax.axhline(0, color="black", linewidth=1)
ax.axhline(0.05, color="gray", linewidth=0.8, linestyle="--", label="±0.05 threshold")
ax.axhline(-0.05, color="gray", linewidth=0.8, linestyle="--")
ax.set_title("Average autocorrelation of monthly returns across stocks")
ax.set_xlabel("Lag (months)")
ax.set_ylabel("Autocorrelation")
ax.set_xticks(list(lags))
ax.legend()
plt.tight_layout()
save(fig, "02d_return_autocorrelation.png")

print("  Section 2 done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: FEATURE–TARGET IC
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 3: FEATURE–TARGET IC (Spearman)")
print("=" * 60)

# 3a. IC per feature (Spearman correlation with target)
ic_scores = {}
for col in FEATURE_COLS:
    if col in df.columns:
        ic, _ = stats.spearmanr(df[col].fillna(0), df[TARGET])
        ic_scores[col] = ic

ic_series = pd.Series(ic_scores).sort_values(key=abs, ascending=False)

print("\n  Top 15 features by |IC|:")
for feat, val in ic_series.head(15).items():
    print(f"    {feat:30s}: {val:.4f}")

fig, ax = plt.subplots(figsize=(14, 10))
colors_ic = ["#E24B4A" if v < 0 else "#378ADD" for v in ic_series.values]
ax.barh(ic_series.index[::-1], ic_series.values[::-1], color=colors_ic[::-1], alpha=0.85)
ax.axvline(0, color="black", linewidth=1)
ax.axvline(0.05, color="#1D9E75", linewidth=1.2, linestyle="--", label="|IC| = 0.05 threshold")
ax.axvline(-0.05, color="#1D9E75", linewidth=1.2, linestyle="--")
ax.set_title("Spearman IC: feature vs 30-day forward return (all features)")
ax.set_xlabel("Spearman IC")
ax.legend()
plt.tight_layout()
save(fig, "03a_ic_all_features.png")

# 3b. Rolling IC for top 8 features over time
top_features = ic_series.abs().sort_values(ascending=False).head(8).index.tolist()

fig, axes = plt.subplots(4, 2, figsize=(16, 16))
axes = axes.flatten()

for i, feat in enumerate(top_features):
    ax = axes[i]
    rolling_ic = []
    dates = []
    months = df["date"].dt.to_period("M").unique()
    window = 12  # 12-month rolling window

    for j in range(window, len(months)):
        period_months = months[j-window:j]
        sub = df[df["date"].dt.to_period("M").isin(period_months)]
        if len(sub) > 50:
            ic_val, _ = stats.spearmanr(sub[feat].fillna(0), sub[TARGET])
            rolling_ic.append(ic_val)
            dates.append(months[j].to_timestamp())

    rolling_ic = pd.Series(rolling_ic, index=dates)
    color = "#378ADD" if ic_scores.get(feat, 0) >= 0 else "#E24B4A"
    ax.fill_between(rolling_ic.index, rolling_ic.values, alpha=0.3, color=color)
    ax.plot(rolling_ic.index, rolling_ic.values, color=color, linewidth=1)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(0.05, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(-0.05, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"{feat}  (overall IC: {ic_scores.get(feat,0):.3f})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling 12M IC")

plt.suptitle("Rolling 12-month IC for top 8 features", fontsize=14, y=1.01)
plt.tight_layout()
save(fig, "03b_rolling_ic_top_features.png")

print("  Section 3 done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: FEATURE CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 4: FEATURE CORRELATION HEATMAP")
print("=" * 60)

valid_feats = [c for c in FEATURE_COLS if c in df.columns]
corr_matrix = df[valid_feats].corr(method="pearson")

# Flag highly correlated pairs
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        val = corr_matrix.iloc[i, j]
        if abs(val) > 0.85:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], val))

print(f"\n  Highly correlated pairs (|r| > 0.85): {len(high_corr_pairs)}")
for a, b, v in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
    print(f"    {a:30s} ↔ {b:30s}: {v:.3f}")

fig, ax = plt.subplots(figsize=(18, 16))
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = ax.imshow(corr_matrix.values, cmap="RdBu_r", norm=norm, aspect="auto")
ax.set_xticks(range(len(valid_feats)))
ax.set_yticks(range(len(valid_feats)))
ax.set_xticklabels(valid_feats, rotation=90, fontsize=8)
ax.set_yticklabels(valid_feats, fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.03)
ax.set_title("Feature correlation matrix (Pearson)")
plt.tight_layout()
save(fig, "04a_feature_correlation_heatmap.png")

# 4b. Mutual information scores
from sklearn.feature_selection import mutual_info_regression

print("\n  Computing mutual information scores...")
X_mi = df[valid_feats].fillna(0)
y_mi = df[TARGET]
mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)
mi_series = pd.Series(mi_scores, index=valid_feats).sort_values(ascending=False)

print("\n  Top 15 features by mutual information:")
for feat, val in mi_series.head(15).items():
    print(f"    {feat:30s}: {val:.4f}")

fig, ax = plt.subplots(figsize=(14, 10))
mi_colors = ["#378ADD"] * len(mi_series)
ax.barh(mi_series.index[::-1], mi_series.values[::-1], color=mi_colors, alpha=0.85)
ax.set_title("Mutual information: feature vs 30-day forward return")
ax.set_xlabel("Mutual information score")
plt.tight_layout()
save(fig, "04b_mutual_information.png")

print("  Section 4 done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: TEMPORAL + LEAKAGE CHECKS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 5: TEMPORAL + STRUCTURAL BREAK CHECK")
print("=" * 60)

# 5a. Mean feature value over time for key features
key_temporal = ["return_20d", "RSI_14", "realvol_20d", "excess_return_1m",
                "return_3m", "return_6m", "dist_52w_high", "vol_ratio_2060"]

fig, axes = plt.subplots(4, 2, figsize=(16, 14))
axes = axes.flatten()

monthly_means = df.groupby(df["date"].dt.to_period("M"))[key_temporal].mean()
monthly_means.index = monthly_means.index.to_timestamp()

crisis_dates = {
    "Demonetization\n(Nov 2016)": "2016-11-01",
    "COVID\n(Mar 2020)":          "2020-03-01",
    "Rate hike\n(May 2022)":      "2022-05-01",
}

for i, feat in enumerate(key_temporal):
    ax = axes[i]
    ax.plot(monthly_means.index, monthly_means[feat], color="#378ADD", linewidth=0.9)
    ax.fill_between(monthly_means.index, monthly_means[feat], alpha=0.2, color="#378ADD")
    for label, date in crisis_dates.items():
        ax.axvline(pd.Timestamp(date), color="#E24B4A", linewidth=1, linestyle="--", alpha=0.7)
        ax.text(pd.Timestamp(date), ax.get_ylim()[1] * 0.9, label,
                fontsize=7, color="#E24B4A", rotation=90, va="top")
    ax.set_title(f"{feat} — monthly mean over time")
    ax.set_xlabel("Date")

plt.suptitle("Feature stability over time (structural break check)", fontsize=14, y=1.01)
plt.tight_layout()
save(fig, "05a_feature_temporal_stability.png")

# 5b. Leakage check — correlation of target with itself shifted by 1 day
print("\n  Leakage check: correlation of target_return_30d with lag-1 self...")
df_sorted = df.sort_values(["Ticker", "date"])
df_sorted["target_lag1"] = df_sorted.groupby("Ticker")["target_return_30d"].shift(1)
leak_corr = df_sorted[["target_return_30d", "target_lag1"]].corr().iloc[0, 1]
print(f"    Lag-1 autocorrelation of target: {leak_corr:.4f}")
if abs(leak_corr) > 0.5:
    print("    WARNING: High autocorrelation — possible leakage!")
else:
    print("    OK: No obvious leakage detected.")
df.drop(columns=["target_lag1"], errors="ignore", inplace=True)

print("  Section 5 done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: CROSS-SECTIONAL COVERAGE
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 6: CROSS-SECTIONAL COVERAGE")
print("=" * 60)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 6a. Stocks per month
monthly_count = df.groupby(df["date"].dt.to_period("M"))["Ticker"].nunique()
monthly_count.index = monthly_count.index.to_timestamp()
ax = axes[0]
ax.fill_between(monthly_count.index, monthly_count.values, alpha=0.4, color="#1D9E75")
ax.plot(monthly_count.index, monthly_count.values, color="#1D9E75", linewidth=0.8)
ax.axhline(40, color="#E24B4A", linestyle="--", linewidth=1.5, label="40-stock threshold")
ax.set_title("Number of stocks available per month")
ax.set_ylabel("Stock count")
ax.set_xlabel("Date")
ax.legend()
low_months = monthly_count[monthly_count < 40]
print(f"  Months with < 40 stocks: {len(low_months)}")
if len(low_months) > 0:
    print(low_months.to_string())

# 6b. Mean cross-sectional zscore spread per month
monthly_zstd = df.groupby(df["date"].dt.to_period("M"))["target_cs_zscore"].std()
monthly_zstd.index = monthly_zstd.index.to_timestamp()
ax = axes[1]
ax.fill_between(monthly_zstd.index, monthly_zstd.values, alpha=0.4, color="#7F77DD")
ax.plot(monthly_zstd.index, monthly_zstd.values, color="#7F77DD", linewidth=0.8)
ax.axhline(monthly_zstd.mean(), color="#E24B4A", linestyle="--", linewidth=1.5,
           label=f"Mean: {monthly_zstd.mean():.2f}")
ax.set_title("Cross-sectional zscore std per month (ranking signal width)")
ax.set_ylabel("Std of target_cs_zscore")
ax.set_xlabel("Date")
ax.legend()

plt.tight_layout()
save(fig, "06a_cross_sectional_coverage.png")

print("  Section 6 done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("EDA SUMMARY")
print("=" * 60)

print(f"\n  Dataset        : {df.shape[0]:,} rows × {df.shape[1]} cols × {df['Ticker'].nunique()} stocks")
print(f"  Date range     : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Total NaNs     : {df.isnull().sum().sum()}")
print(f"  % UP returns   : {(df[TARGET] > 0).mean()*100:.1f}%")
print(f"  % DOWN returns : {(df[TARGET] <= 0).mean()*100:.1f}%")

print(f"\n  Top 5 features by |IC|:")
for feat, val in ic_series.abs().sort_values(ascending=False).head(5).items():
    direction = "+" if ic_scores[feat] > 0 else "-"
    print(f"    {feat:30s}: {direction}{abs(val):.4f}")

print(f"\n  High correlation pairs (|r| > 0.85): {len(high_corr_pairs)}")

print(f"\n  All plots saved to: {OUTPUT_DIR}")
print("\nEDA complete. Open eda_output/ folder to view all charts.")