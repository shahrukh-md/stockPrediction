"""
V4 Walk-Forward Model Training Script
Ranked Stock Selection — Nifty 50

Config:
  - Rolling 36-month training window
  - Target: target_cs_zscore (relative ranking)
  - Portfolio: Top 10 long, Bottom 10 short
  - Models: LightGBM, XGBoost, Random Forest (ensemble)
  - Evaluation: Rank IC, Hit Rate, Long-Short Spread, vs Nifty benchmark

Run: python models/train_v4.py

Output:
  results/fold_results.csv
  results/monthly_portfolio.csv
  results/summary.txt
  results/plots/
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
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE   = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_v4_cs_normalized.csv"
RESULTS_DIR  = "/Users/shahrukh/Desktop/stock/results"
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_WINDOW = 36       # rolling months
TOP_N        = 10       # long top N stocks
BOTTOM_N     = 10       # short bottom N stocks
TARGET       = "target_cs_zscore"
TARGET_RAW   = "target_return_30d"
RANDOM_STATE = 42

FEATURE_COLS = [
    "price_to_ma200", "BB_width", "BB_pct",
    "return_1d", "return_5d", "return_10d", "return_20d",
    "return_3m", "return_6m", "return_12m",
    "dist_52w_high", "MACD_sig", "MACD_diff",
    "realvol_20d", "realvol_60d", "vol_ratio_2060", "atr_pct",
    "vol_ratio",
    # amihud_20d dropped — zero variance after CS normalization
    "forward_PE", "profit_margin", "operating_margin",
    "debt_to_equity", "book_value", "price_to_book",
    "dividend_yield", "beta"
]

# ── MODELS ────────────────────────────────────────────────────────────────────
def get_models():
    return {
        "LightGBM": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            max_features=0.6,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

# ── METRICS ───────────────────────────────────────────────────────────────────
def compute_metrics(actual, predicted, actual_raw, top_n=10, bottom_n=10):
    """
    Compute all ranking evaluation metrics for one fold.
    actual     : actual cs_zscore values
    predicted  : model predicted scores
    actual_raw : actual raw 30-day returns
    """
    n = len(actual)
    if n < top_n + bottom_n:
        return None

    # Rank IC — Spearman correlation between predicted and actual scores
    rank_ic, _ = stats.spearmanr(predicted, actual)

    # Sort by predicted score (descending)
    sorted_idx      = np.argsort(predicted)[::-1]
    top_idx         = sorted_idx[:top_n]
    bottom_idx      = sorted_idx[-bottom_n:]

    # Hit rate — % of top-N stocks with positive raw return
    top_returns     = actual_raw[top_idx]
    hit_rate        = (top_returns > 0).mean()

    # Portfolio returns
    top_return      = top_returns.mean()
    bottom_return   = actual_raw[bottom_idx].mean()
    ls_spread       = top_return - bottom_return

    return {
        "rank_ic":       rank_ic,
        "hit_rate":      hit_rate,
        "top10_return":  top_return,
        "bot10_return":  bottom_return,
        "ls_spread":     ls_spread,
        "n_stocks":      n,
    }

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading nifty50_v4_cs_normalized.csv...")
df = pd.read_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)
df = df.sort_values(["date", "Ticker"]).reset_index(drop=True)
df["year_month"] = df["date"].dt.to_period("M")

print(f"  Shape   : {df.shape}")
print(f"  Tickers : {df['Ticker'].nunique()}")
print(f"  Months  : {df['year_month'].nunique()}")

# Validate feature cols
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"\nWARNING: Missing feature columns: {missing}")
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
print(f"  Features: {len(FEATURE_COLS)}\n")

# Get sorted list of all months
all_months = sorted(df["year_month"].unique())
total_months = len(all_months)
print(f"  Total months      : {total_months}")
print(f"  Train window      : {TRAIN_WINDOW} months (rolling)")
print(f"  First test month  : {all_months[TRAIN_WINDOW]}")
print(f"  Last test month   : {all_months[-1]}")
print(f"  Total folds       : {total_months - TRAIN_WINDOW}\n")

# ── WALK-FORWARD TRAINING LOOP ────────────────────────────────────────────────
print("=" * 60)
print("WALK-FORWARD TRAINING")
print("=" * 60)

fold_results   = []    # per-fold metrics
monthly_portfolio = [] # which stocks selected each month
feature_importances = {name: np.zeros(len(FEATURE_COLS)) for name in ["LightGBM", "XGBoost", "RandomForest"]}
fi_counts = {name: 0 for name in ["LightGBM", "XGBoost", "RandomForest"]}

total_folds = total_months - TRAIN_WINDOW

for fold_idx in range(total_folds):
    train_months = all_months[fold_idx : fold_idx + TRAIN_WINDOW]
    test_month   = all_months[fold_idx + TRAIN_WINDOW]

    # Split data
    train_df = df[df["year_month"].isin(train_months)].copy()
    test_df  = df[df["year_month"] == test_month].copy()

    if len(test_df) < 20:
        print(f"  Fold {fold_idx+1:3d} | {test_month} | SKIP — only {len(test_df)} stocks")
        continue

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET].values
    X_test  = test_df[FEATURE_COLS].values

    actual_cs  = test_df[TARGET].values
    actual_raw = test_df[TARGET_RAW].values

    # Train all three models + collect predictions
    fold_preds = {}
    fold_row = {
        "fold":       fold_idx + 1,
        "test_month": str(test_month),
        "n_stocks":   len(test_df),
        "nifty_return": test_df[TARGET_RAW].mean(),  # equal-weight market return
    }

    models = get_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        fold_preds[name] = preds

        metrics = compute_metrics(actual_cs, preds, actual_raw, TOP_N, BOTTOM_N)
        if metrics:
            for k, v in metrics.items():
                fold_row[f"{name}_{k}"] = v

        # Accumulate feature importances
        if hasattr(model, "feature_importances_"):
            feature_importances[name] += model.feature_importances_
            fi_counts[name] += 1

    # Ensemble: average predictions from all three models
    ensemble_preds = np.mean([fold_preds[n] for n in models.keys()], axis=0)
    metrics_ens = compute_metrics(actual_cs, ensemble_preds, actual_raw, TOP_N, BOTTOM_N)
    if metrics_ens:
        for k, v in metrics_ens.items():
            fold_row[f"Ensemble_{k}"] = v

    # Record monthly portfolio (top 10 + bottom 10 by ensemble)
    sorted_idx  = np.argsort(ensemble_preds)[::-1]
    top_tickers = test_df.iloc[sorted_idx[:TOP_N]]["Ticker"].values
    bot_tickers = test_df.iloc[sorted_idx[-BOTTOM_N:]]["Ticker"].values

    monthly_portfolio.append({
        "month":          str(test_month),
        "long_portfolio": list(top_tickers),
        "short_portfolio":list(bot_tickers),
        "long_return":    actual_raw[sorted_idx[:TOP_N]].mean(),
        "short_return":   actual_raw[sorted_idx[-BOTTOM_N:]].mean(),
        "ls_spread":      actual_raw[sorted_idx[:TOP_N]].mean() - actual_raw[sorted_idx[-BOTTOM_N:]].mean(),
        "nifty_return":   actual_raw.mean(),
    })

    fold_results.append(fold_row)

    # Progress print every 12 folds (yearly)
    if (fold_idx + 1) % 12 == 0 or fold_idx == total_folds - 1:
        ens_ic = fold_row.get("Ensemble_rank_ic", float("nan"))
        ens_spread = fold_row.get("Ensemble_ls_spread", float("nan"))
        print(f"  Fold {fold_idx+1:3d}/{total_folds} | {test_month} | "
              f"Ensemble IC: {ens_ic:+.3f} | L/S Spread: {ens_spread:+.4f}")

print("\nWalk-forward training complete.\n")

# ── RESULTS AGGREGATION ───────────────────────────────────────────────────────
print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

results_df = pd.DataFrame(fold_results)
results_df.to_csv(os.path.join(RESULTS_DIR, "fold_results.csv"), index=False)

portfolio_df = pd.DataFrame(monthly_portfolio)
portfolio_df.to_csv(os.path.join(RESULTS_DIR, "monthly_portfolio.csv"), index=False)

model_names = ["LightGBM", "XGBoost", "RandomForest", "Ensemble"]

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("V4 WALK-FORWARD RESULTS SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Total folds      : {len(results_df)}")
summary_lines.append(f"Training window  : {TRAIN_WINDOW} months (rolling)")
summary_lines.append(f"Target           : {TARGET}")
summary_lines.append(f"Features         : {len(FEATURE_COLS)}")
summary_lines.append(f"Portfolio        : Top {TOP_N} long / Bottom {BOTTOM_N} short")
summary_lines.append("")

print(f"\n  {'Model':<15} {'Mean IC':>10} {'IC>0 %':>10} {'Hit Rate':>10} {'L/S Spread':>12} {'Top10 Ret':>10}")
print("  " + "-" * 65)
summary_lines.append(f"{'Model':<15} {'Mean IC':>10} {'IC>0 %':>10} {'Hit Rate':>10} {'L/S Spread':>12} {'Top10 Ret':>10}")
summary_lines.append("-" * 65)

for name in model_names:
    ic_col     = f"{name}_rank_ic"
    hr_col     = f"{name}_hit_rate"
    sp_col     = f"{name}_ls_spread"
    ret_col    = f"{name}_top10_return"

    if ic_col not in results_df.columns:
        continue

    mean_ic    = results_df[ic_col].mean()
    ic_pos_pct = (results_df[ic_col] > 0).mean() * 100
    mean_hr    = results_df[hr_col].mean()
    mean_sp    = results_df[sp_col].mean()
    mean_ret   = results_df[ret_col].mean()

    line = f"  {name:<15} {mean_ic:>+10.4f} {ic_pos_pct:>9.1f}% {mean_hr:>9.1f}% {mean_sp:>+11.4f} {mean_ret:>+9.4f}"
    print(line)
    summary_lines.append(line)

# Benchmark
nifty_mean = results_df["nifty_return"].mean()
summary_lines.append("")
summary_lines.append(f"Nifty benchmark (equal-weight): {nifty_mean:+.4f} per month")
print(f"\n  Nifty benchmark (equal-weight all 48): {nifty_mean:+.4f} per month")

# IC interpretation
ens_ic = results_df["Ensemble_rank_ic"].mean() if "Ensemble_rank_ic" in results_df.columns else 0
summary_lines.append("")
summary_lines.append("IC interpretation:")
summary_lines.append("  IC > 0.10 : Excellent — strong ranking signal")
summary_lines.append("  IC > 0.05 : Good — meaningful predictive power")
summary_lines.append("  IC > 0.02 : Marginal — weak but present")
summary_lines.append("  IC < 0.02 : Poor — model not ranking well")
summary_lines.append(f"\n  Your Ensemble IC: {ens_ic:+.4f}")

# Save summary
with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))
print(f"\n  Summary saved to results/summary.txt")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
print("\nGenerating result plots...")

# Plot 1: Rolling Rank IC over time
fig, ax = plt.subplots(figsize=(16, 5))
for name, color in zip(model_names, ["#378ADD", "#E24B4A", "#1D9E75", "#7F3FBF"]):
    ic_col = f"{name}_rank_ic"
    if ic_col in results_df.columns:
        ax.plot(results_df["test_month"], results_df[ic_col],
                label=name, color=color, linewidth=1.2, alpha=0.8)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.axhline(0.05, color="gray", linewidth=0.8, linestyle=":", label="IC=0.05 threshold")
ax.axhline(-0.05, color="gray", linewidth=0.8, linestyle=":")
ax.set_title("Rank IC per fold — all models")
ax.set_xlabel("Test month")
ax.set_ylabel("Rank IC (Spearman)")
ax.legend()
plt.xticks(rotation=45)
step = max(1, len(results_df) // 12)
ax.set_xticks(results_df["test_month"].values[::step])
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "01_rank_ic_over_time.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 01_rank_ic_over_time.png")

# Plot 2: Cumulative long-short return vs Nifty
fig, ax = plt.subplots(figsize=(16, 6))
for name, color in zip(["Ensemble", "LightGBM"], ["#7F3FBF", "#378ADD"]):
    ret_col = f"{name}_top10_return"
    if ret_col in results_df.columns:
        cumret = (1 + results_df[ret_col]).cumprod()
        ax.plot(results_df["test_month"], cumret,
                label=f"{name} Top-10 Long", color=color, linewidth=1.5)

# Nifty cumulative
nifty_cum = (1 + results_df["nifty_return"]).cumprod()
ax.plot(results_df["test_month"], nifty_cum,
        label="Nifty (equal-weight)", color="#E24B4A", linewidth=1.5, linestyle="--")
ax.axhline(1, color="black", linewidth=0.8, linestyle=":")
ax.set_title("Cumulative return: Top-10 portfolio vs Nifty benchmark")
ax.set_xlabel("Month")
ax.set_ylabel("Cumulative return (₹1 invested)")
ax.legend()
plt.xticks(rotation=45)
ax.set_xticks(results_df["test_month"].values[::step])
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "02_cumulative_return_vs_nifty.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 02_cumulative_return_vs_nifty.png")

# Plot 3: Long-short spread over time
fig, ax = plt.subplots(figsize=(16, 5))
sp_col = "Ensemble_ls_spread"
if sp_col in results_df.columns:
    colors_bar = ["#1D9E75" if v > 0 else "#E24B4A" for v in results_df[sp_col]]
    ax.bar(range(len(results_df)), results_df[sp_col], color=colors_bar, alpha=0.8, width=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(results_df[sp_col].mean(), color="#7F3FBF", linewidth=1.5,
               linestyle="--", label=f"Mean spread: {results_df[sp_col].mean():+.4f}")
    ax.set_title("Ensemble long-short spread per month (Top 10 - Bottom 10)")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Long-short spread")
    ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "03_ls_spread_over_time.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 03_ls_spread_over_time.png")

# Plot 4: Feature importance (Ensemble average)
fig, ax = plt.subplots(figsize=(14, 8))
avg_fi = np.zeros(len(FEATURE_COLS))
count  = 0
for name in ["LightGBM", "XGBoost", "RandomForest"]:
    if fi_counts[name] > 0:
        avg_fi += feature_importances[name] / fi_counts[name]
        count  += 1
if count > 0:
    avg_fi /= count
    fi_series = pd.Series(avg_fi, index=FEATURE_COLS).sort_values(ascending=True)
    colors_fi = ["#378ADD"] * len(fi_series)
    ax.barh(fi_series.index, fi_series.values, color=colors_fi, alpha=0.85)
    ax.set_title("Average feature importance across all models and folds")
    ax.set_xlabel("Importance score")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "04_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 04_feature_importance.png")

# Plot 5: Hit rate over time
fig, ax = plt.subplots(figsize=(16, 5))
hr_col = "Ensemble_hit_rate"
if hr_col in results_df.columns:
    ax.plot(results_df["test_month"], results_df[hr_col] * 100,
            color="#378ADD", linewidth=1.2)
    ax.axhline(50, color="black", linewidth=1, linestyle="--", label="50% baseline")
    ax.axhline(results_df[hr_col].mean() * 100, color="#E24B4A", linewidth=1.5,
               linestyle="--", label=f"Mean hit rate: {results_df[hr_col].mean()*100:.1f}%")
    ax.fill_between(results_df["test_month"],
                    results_df[hr_col] * 100, 50,
                    where=results_df[hr_col] > 0.5,
                    alpha=0.2, color="#1D9E75")
    ax.set_title("Hit rate of Top-10 portfolio per month (% with positive return)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Hit rate (%)")
    ax.legend()
    plt.xticks(rotation=45)
    ax.set_xticks(results_df["test_month"].values[::step])
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "05_hit_rate_over_time.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 05_hit_rate_over_time.png")

print(f"\nAll results saved to: {RESULTS_DIR}")
print("Training complete. Check results/summary.txt for full metrics.")