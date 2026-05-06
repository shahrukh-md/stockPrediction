"""
V4 Fixed Training Script
Fixes all fixable issues identified in model review:

1. Hit rate display bug fixed (0.6% → 60%)
2. Transaction cost simulation added (0.3% round-trip per rebalance)
3. Regime filter added (reduces positions in high-correlation regimes)
4. Look-ahead bias flag added to summary
5. Turnover calculation added per month

Run: python models/train_v4_fixed.py

Output:
  results_fixed/fold_results.csv
  results_fixed/monthly_portfolio.csv
  results_fixed/summary.txt
  results_fixed/plots/
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
RESULTS_DIR  = "/Users/shahrukh/Desktop/stock/results_fixed"
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_WINDOW       = 36      # rolling months
TOP_N              = 10      # long top N stocks
BOTTOM_N           = 10      # short bottom N stocks
TARGET             = "target_cs_zscore"
TARGET_RAW         = "target_return_30d"
RANDOM_STATE       = 42

# ── FIX 2: TRANSACTION COST CONFIG ───────────────────────────────────────────
# Round-trip cost per stock traded (in and out)
# Brokerage: 0.10% + STT: 0.10% + exchange: 0.003% + impact: 0.10% ≈ 0.30%
TRANSACTION_COST_PCT = 0.003   # 0.3% per round-trip per stock

FEATURE_COLS = [
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

# ── MODELS ────────────────────────────────────────────────────────────────────
def get_models():
    return {
        "LightGBM": LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=RANDOM_STATE, verbose=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=RANDOM_STATE, verbosity=0,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            max_features=0.6, random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

# ── FIX 3: REGIME FILTER ──────────────────────────────────────────────────────
def compute_regime(train_df, threshold=0.65):
    """
    Detect high-correlation regime using average pairwise return correlation.
    When all stocks move together (corr > threshold), ranking signal collapses.
    Returns: 'normal' or 'high_correlation'

    threshold=0.65 means: if avg pairwise correlation of 20d returns across
    all stocks exceeds 0.65, we are in a high-correlation (crisis) regime.
    """
    # Use last 3 months of training data for regime detection
    recent = train_df.tail(3 * 48)  # approx last 3 months
    if "return_20d" not in recent.columns or len(recent) < 20:
        return "normal", 1.0

    # Pivot to get return_20d per stock
    try:
        pivot = recent.pivot_table(
            index="date", columns="Ticker", values="return_20d"
        )
        if pivot.shape[1] < 5:
            return "normal", 1.0
        corr_matrix = pivot.corr()
        # Average of upper triangle (excluding diagonal)
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        avg_corr = upper.stack().mean()

        if avg_corr > threshold:
            # High correlation regime — reduce position sizing
            regime = "high_correlation"
            # Position scale: 0.5 at corr=0.65, 0.0 at corr=1.0
            scale = max(0.0, 1.0 - (avg_corr - threshold) / (1.0 - threshold))
        else:
            regime = "normal"
            scale = 1.0

        return regime, round(avg_corr, 4), round(scale, 4)
    except Exception:
        return "normal", 0.0, 1.0

# ── FIX 1+2+3: METRICS WITH COST AND REGIME ───────────────────────────────────
def compute_metrics(actual_cs, predicted, actual_raw, top_n, bottom_n,
                    prev_long=None, prev_short=None, position_scale=1.0):
    """
    Compute all ranking metrics with:
    - FIX 1: hit rate correctly formatted as percentage
    - FIX 2: transaction costs applied based on portfolio turnover
    - FIX 3: position scale applied from regime filter
    """
    n = len(actual_cs)
    if n < top_n + bottom_n:
        return None

    # Rank IC
    rank_ic, _ = stats.spearmanr(predicted, actual_cs)

    # Sort by predicted score
    sorted_idx   = np.argsort(predicted)[::-1]
    top_idx      = sorted_idx[:top_n]
    bottom_idx   = sorted_idx[-bottom_n:]

    top_tickers    = np.arange(n)[top_idx]    # positional indices
    bottom_tickers = np.arange(n)[bottom_idx]

    # Raw returns
    top_returns    = actual_raw[top_idx]
    bottom_returns = actual_raw[bottom_idx]

    # ── FIX 1: hit rate as proper % ──────────────────────────────────────────
    hit_rate_pct = (top_returns > 0).mean() * 100  # now in %, e.g. 60.0

    # ── FIX 2: TRANSACTION COSTS ─────────────────────────────────────────────
    # Compute turnover vs previous month's portfolio
    if prev_long is not None:
        prev_long_set  = set(prev_long)
        prev_short_set = set(prev_short) if prev_short else set()
        curr_long_set  = set(top_idx.tolist())
        curr_short_set = set(bottom_idx.tolist())

        # Stocks entering/exiting long portfolio
        long_entries  = len(curr_long_set - prev_long_set)
        long_exits    = len(prev_long_set - curr_long_set)
        short_entries = len(curr_short_set - prev_short_set)
        short_exits   = len(prev_short_set - curr_short_set)

        # Turnover as fraction of portfolio
        long_turnover  = (long_entries + long_exits) / (2 * top_n)
        short_turnover = (short_entries + short_exits) / (2 * bottom_n)
        total_turnover = (long_turnover + short_turnover) / 2
    else:
        # First month — full portfolio construction cost
        long_turnover  = 1.0
        short_turnover = 1.0
        total_turnover = 1.0

    # Cost = turnover × round-trip cost
    cost_long  = long_turnover  * TRANSACTION_COST_PCT
    cost_short = short_turnover * TRANSACTION_COST_PCT

    # Net returns after costs
    top_return_gross    = top_returns.mean()
    bottom_return_gross = bottom_returns.mean()
    top_return_net      = top_return_gross    - cost_long
    bottom_return_net   = bottom_return_gross + cost_short  # short side: cost reduces profit

    # ── FIX 3: REGIME SCALE ──────────────────────────────────────────────────
    # Scale down positions in high-correlation regimes
    top_return_net    = top_return_net    * position_scale
    bottom_return_net = bottom_return_net * position_scale

    ls_spread_gross = top_return_gross - bottom_return_gross
    ls_spread_net   = top_return_net   - bottom_return_net

    return {
        "rank_ic":            rank_ic,
        "hit_rate_pct":       hit_rate_pct,          # FIX 1: now proper %
        "top10_return_gross": top_return_gross,
        "top10_return_net":   top_return_net,         # FIX 2: after costs
        "bot10_return_gross": bottom_return_gross,
        "bot10_return_net":   bottom_return_net,
        "ls_spread_gross":    ls_spread_gross,
        "ls_spread_net":      ls_spread_net,          # FIX 2+3: after costs+regime
        "turnover":           total_turnover,
        "cost_drag":          cost_long,
        "position_scale":     position_scale,         # FIX 3: regime scale
        "n_stocks":           n,
        "top_idx":            top_idx.tolist(),
        "bottom_idx":         bottom_idx.tolist(),
    }

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading nifty50_v4_cs_normalized.csv...")
df = pd.read_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)
df = df.sort_values(["date", "Ticker"]).reset_index(drop=True)
df["year_month"] = df["date"].dt.to_period("M")

print(f"  Shape   : {df.shape}")
print(f"  Tickers : {df['Ticker'].nunique()}")
print(f"  Months  : {df['year_month'].nunique()}")

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"  WARNING: Missing features: {missing}")
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
print(f"  Features: {len(FEATURE_COLS)}")

all_months   = sorted(df["year_month"].unique())
total_months = len(all_months)
total_folds  = total_months - TRAIN_WINDOW

print(f"\n  Train window  : {TRAIN_WINDOW} months (rolling)")
print(f"  Total folds   : {total_folds}")
print(f"  First test    : {all_months[TRAIN_WINDOW]}")
print(f"  Last test     : {all_months[-1]}")
print(f"\n  Fixes applied:")
print(f"    FIX 1 — Hit rate display bug corrected")
print(f"    FIX 2 — Transaction costs: {TRANSACTION_COST_PCT*100:.1f}% round-trip per stock")
print(f"    FIX 3 — Regime filter: reduce positions when avg pairwise corr > 0.65")
print(f"    FIX 4 — Look-ahead bias disclosed in summary\n")

# ── WALK-FORWARD LOOP ─────────────────────────────────────────────────────────
print("=" * 60)
print("WALK-FORWARD TRAINING")
print("=" * 60)

fold_results      = []
monthly_portfolio = []
feature_importances = {n: np.zeros(len(FEATURE_COLS)) for n in ["LightGBM","XGBoost","RandomForest"]}
fi_counts = {n: 0 for n in ["LightGBM","XGBoost","RandomForest"]}

prev_long  = None
prev_short = None

for fold_idx in range(total_folds):
    train_months = all_months[fold_idx : fold_idx + TRAIN_WINDOW]
    test_month   = all_months[fold_idx + TRAIN_WINDOW]

    train_df = df[df["year_month"].isin(train_months)].copy()
    test_df  = df[df["year_month"] == test_month].copy().reset_index(drop=True)

    if len(test_df) < 20:
        continue

    # ── FIX 3: DETECT REGIME ──────────────────────────────────────────────────
    regime_result  = compute_regime(train_df)
    regime         = regime_result[0]
    avg_corr       = regime_result[1]
    position_scale = regime_result[2]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET].values
    X_test  = test_df[FEATURE_COLS].values
    actual_cs  = test_df[TARGET].values
    actual_raw = test_df[TARGET_RAW].values

    fold_row = {
        "fold":           fold_idx + 1,
        "test_month":     str(test_month),
        "n_stocks":       len(test_df),
        "nifty_return":   actual_raw.mean(),
        "regime":         regime,
        "avg_pairwise_corr": avg_corr,
        "position_scale": position_scale,
    }

    fold_preds = {}
    models = get_models()

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        fold_preds[name] = preds

        metrics = compute_metrics(
            actual_cs, preds, actual_raw,
            TOP_N, BOTTOM_N, prev_long, prev_short, position_scale
        )
        if metrics:
            for k, v in metrics.items():
                if k not in ["top_idx", "bottom_idx"]:
                    fold_row[f"{name}_{k}"] = v

        if hasattr(model, "feature_importances_"):
            feature_importances[name] += model.feature_importances_
            fi_counts[name] += 1

    # Ensemble
    ensemble_preds = np.mean([fold_preds[n] for n in models.keys()], axis=0)
    metrics_ens = compute_metrics(
        actual_cs, ensemble_preds, actual_raw,
        TOP_N, BOTTOM_N, prev_long, prev_short, position_scale
    )

    if metrics_ens:
        for k, v in metrics_ens.items():
            if k not in ["top_idx", "bottom_idx"]:
                fold_row[f"Ensemble_{k}"] = v

        # Update prev portfolio for next fold's turnover calculation
        prev_long  = metrics_ens["top_idx"]
        prev_short = metrics_ens["bottom_idx"]

        # Monthly portfolio record
        sorted_idx  = np.argsort(ensemble_preds)[::-1]
        top_tickers = test_df.iloc[sorted_idx[:TOP_N]]["Ticker"].values
        bot_tickers = test_df.iloc[sorted_idx[-BOTTOM_N:]]["Ticker"].values

        monthly_portfolio.append({
            "month":          str(test_month),
            "regime":         regime,
            "position_scale": position_scale,
            "long_portfolio": ", ".join(top_tickers),
            "short_portfolio":", ".join(bot_tickers),
            "long_return_gross":  actual_raw[sorted_idx[:TOP_N]].mean(),
            "long_return_net":    metrics_ens["top10_return_net"],
            "short_return_gross": actual_raw[sorted_idx[-BOTTOM_N:]].mean(),
            "ls_spread_gross":    metrics_ens["ls_spread_gross"],
            "ls_spread_net":      metrics_ens["ls_spread_net"],
            "turnover":           metrics_ens["turnover"],
            "cost_drag":          metrics_ens["cost_drag"],
            "nifty_return":       actual_raw.mean(),
        })

    fold_results.append(fold_row)

    if (fold_idx + 1) % 12 == 0 or fold_idx == total_folds - 1:
        ens_ic     = fold_row.get("Ensemble_rank_ic", float("nan"))
        ens_spread = fold_row.get("Ensemble_ls_spread_net", float("nan"))
        regime_str = f"[{regime[:4].upper()}|scale={position_scale:.1f}]"
        print(f"  Fold {fold_idx+1:3d}/{total_folds} | {test_month} | "
              f"IC: {ens_ic:+.3f} | Net L/S: {ens_spread:+.4f} | {regime_str}")

print("\nWalk-forward training complete.\n")

# ── RESULTS ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

results_df   = pd.DataFrame(fold_results)
portfolio_df = pd.DataFrame(monthly_portfolio)

results_df.to_csv(os.path.join(RESULTS_DIR, "fold_results.csv"), index=False)
portfolio_df.to_csv(os.path.join(RESULTS_DIR, "monthly_portfolio.csv"), index=False)

model_names = ["LightGBM", "XGBoost", "RandomForest", "Ensemble"]

print(f"\n  {'Model':<15} {'Mean IC':>9} {'IC>0%':>7} {'Hit Rate':>10} {'Gross Ret':>11} {'Net Ret':>9} {'Net Spread':>11}")
print("  " + "-" * 80)

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("V4 FIXED WALK-FORWARD RESULTS SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Total folds       : {len(results_df)}")
summary_lines.append(f"Training window   : {TRAIN_WINDOW} months (rolling)")
summary_lines.append(f"Transaction cost  : {TRANSACTION_COST_PCT*100:.1f}% round-trip")
summary_lines.append(f"Features          : {len(FEATURE_COLS)}")
summary_lines.append(f"Portfolio         : Top {TOP_N} long / Bottom {BOTTOM_N} short")
summary_lines.append("")

for name in model_names:
    ic_col     = f"{name}_rank_ic"
    hr_col     = f"{name}_hit_rate_pct"      # FIX 1
    sp_col     = f"{name}_ls_spread_net"     # FIX 2+3
    sp_gross   = f"{name}_ls_spread_gross"
    ret_gross  = f"{name}_top10_return_gross"
    ret_net    = f"{name}_top10_return_net"  # FIX 2+3

    if ic_col not in results_df.columns:
        continue

    mean_ic      = results_df[ic_col].mean()
    ic_pos_pct   = (results_df[ic_col] > 0).mean() * 100
    mean_hr      = results_df[hr_col].mean()          # FIX 1: proper %
    mean_ret_g   = results_df[ret_gross].mean()
    mean_ret_n   = results_df[ret_net].mean()
    mean_sp_n    = results_df[sp_col].mean()

    line = (f"  {name:<15} {mean_ic:>+9.4f} {ic_pos_pct:>6.1f}% "
            f"{mean_hr:>9.1f}% {mean_ret_g:>+10.4f} {mean_ret_n:>+8.4f} {mean_sp_n:>+10.4f}")
    print(line)
    summary_lines.append(line)

nifty_mean = results_df["nifty_return"].mean()
regime_counts = results_df["regime"].value_counts()
high_corr_months = regime_counts.get("high_correlation", 0)

print(f"\n  Nifty benchmark (equal-weight): {nifty_mean:+.4f} per month")
print(f"  High-correlation regime months: {high_corr_months}/{len(results_df)}")
print(f"  Avg monthly turnover: {portfolio_df['turnover'].mean()*100:.1f}%")
print(f"  Avg monthly cost drag: {portfolio_df['cost_drag'].mean()*100:.3f}%")

# ── FIX 4: LOOK-AHEAD BIAS DISCLOSURE ─────────────────────────────────────────
summary_lines.append("")
summary_lines.append("=" * 60)
summary_lines.append("KNOWN LIMITATIONS (important disclosures)")
summary_lines.append("=" * 60)
summary_lines.append("")
summary_lines.append("1. SURVIVORSHIP BIAS (High severity)")
summary_lines.append("   Universe = current Nifty 50 constituents only.")
summary_lines.append("   Stocks removed from Nifty 50 due to poor performance")
summary_lines.append("   are absent from training data. Backtest returns are")
summary_lines.append("   likely overstated as a result.")
summary_lines.append("")
summary_lines.append("2. LOOK-AHEAD BIAS IN FUNDAMENTALS (Medium-High severity)")
summary_lines.append("   Fundamental features (PE, profit_margin, etc.) are")
summary_lines.append("   point-in-time snapshots, not timestamped quarterly data.")
summary_lines.append("   Quarterly earnings are published 45-60 days after quarter")
summary_lines.append("   end. Model may have 'seen' data not available at trade time.")
summary_lines.append("")
summary_lines.append("3. BACKTEST OVERFITTING (High severity)")
summary_lines.append("   Feature selection, window length, and portfolio size were")
summary_lines.append("   all chosen by observing this dataset. Forward performance")
summary_lines.append("   will likely be lower than backtest results.")
summary_lines.append("")
summary_lines.append("4. TRANSACTION COSTS (Modelled — 0.3% round-trip)")
summary_lines.append("   Brokerage + STT + exchange + impact cost approximated.")
summary_lines.append("   Actual costs may vary by broker and stock liquidity.")
summary_lines.append("")
summary_lines.append("5. SMALL UNIVERSE (Low-Medium severity)")
summary_lines.append("   48 stocks is a small cross-sectional universe.")
summary_lines.append("   Planned expansion to Nifty 100 in V4.1.")

summary_lines.append("")
summary_lines.append("=" * 60)
summary_lines.append("IC INTERPRETATION GUIDE")
summary_lines.append("=" * 60)
summary_lines.append("  IC > 0.10 : Excellent")
summary_lines.append("  IC > 0.05 : Good — meaningful predictive power")
summary_lines.append("  IC > 0.02 : Marginal")
summary_lines.append("  IC < 0.02 : Poor")

with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))
print(f"\n  Summary saved to results_fixed/summary.txt")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")
step = max(1, len(results_df) // 12)

# Plot 1: Rank IC over time
fig, ax = plt.subplots(figsize=(16, 5))
for name, color in zip(model_names, ["#378ADD","#E24B4A","#1D9E75","#7F3FBF"]):
    ic_col = f"{name}_rank_ic"
    if ic_col in results_df.columns:
        ax.plot(results_df["test_month"], results_df[ic_col],
                label=name, color=color, linewidth=1.2, alpha=0.8)
# Shade high-correlation regime months
for _, row in results_df[results_df["regime"]=="high_correlation"].iterrows():
    ax.axvspan(row["test_month"], row["test_month"], alpha=0.15, color="red")
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.axhline(0.05, color="gray", linewidth=0.8, linestyle=":", label="IC=0.05")
ax.axhline(-0.05, color="gray", linewidth=0.8, linestyle=":")
ax.set_title("Rank IC per fold — all models (red shading = high-correlation regime)")
ax.set_xlabel("Test month")
ax.set_ylabel("Rank IC (Spearman)")
ax.legend()
plt.xticks(rotation=45)
ax.set_xticks(results_df["test_month"].values[::step])
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "01_rank_ic_over_time.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 01_rank_ic_over_time.png")

# Plot 2: Gross vs Net cumulative return vs Nifty
fig, ax = plt.subplots(figsize=(16, 6))
ret_gross = results_df["Ensemble_top10_return_gross"]
ret_net   = results_df["Ensemble_top10_return_net"]
nifty_ret = results_df["nifty_return"]
ax.plot(results_df["test_month"], (1 + ret_gross).cumprod(),
        label="Ensemble Top-10 (Gross)", color="#7F3FBF", linewidth=1.8)
ax.plot(results_df["test_month"], (1 + ret_net).cumprod(),
        label="Ensemble Top-10 (Net of costs)", color="#378ADD", linewidth=1.8, linestyle="--")
ax.plot(results_df["test_month"], (1 + nifty_ret).cumprod(),
        label="Nifty benchmark", color="#E24B4A", linewidth=1.5, linestyle=":")
ax.axhline(1, color="black", linewidth=0.8, linestyle=":")
ax.set_title("Cumulative return: Gross vs Net (after 0.3% transaction costs) vs Nifty")
ax.set_xlabel("Month")
ax.set_ylabel("Cumulative return (₹1 invested)")
ax.legend()
plt.xticks(rotation=45)
ax.set_xticks(results_df["test_month"].values[::step])
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "02_cumulative_gross_vs_net_vs_nifty.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 02_cumulative_gross_vs_net_vs_nifty.png")

# Plot 3: Net L/S spread per month
fig, ax = plt.subplots(figsize=(16, 5))
sp_net = results_df["Ensemble_ls_spread_net"]
colors_bar = ["#1D9E75" if v > 0 else "#E24B4A" for v in sp_net]
ax.bar(range(len(results_df)), sp_net, color=colors_bar, alpha=0.8, width=0.8)
ax.axhline(0, color="black", linewidth=1)
ax.axhline(sp_net.mean(), color="#7F3FBF", linewidth=1.5, linestyle="--",
           label=f"Mean net spread: {sp_net.mean():+.4f}")
ax.set_title("Ensemble net long-short spread per month (after transaction costs)")
ax.set_xlabel("Fold")
ax.set_ylabel("Net L/S spread")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "03_net_ls_spread.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 03_net_ls_spread.png")

# Plot 4: Feature importance
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
    ax.barh(fi_series.index, fi_series.values, color="#378ADD", alpha=0.85)
    ax.set_title("Average feature importance across all models and folds")
    ax.set_xlabel("Importance score")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "04_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 04_feature_importance.png")

# Plot 5: Hit rate over time — FIX 1 applied
fig, ax = plt.subplots(figsize=(16, 5))
hr = results_df["Ensemble_hit_rate_pct"]   # FIX 1: now proper %
ax.plot(results_df["test_month"], hr, color="#378ADD", linewidth=1.2)
ax.axhline(50, color="black", linewidth=1, linestyle="--", label="50% baseline")
ax.axhline(hr.mean(), color="#E24B4A", linewidth=1.5, linestyle="--",
           label=f"Mean hit rate: {hr.mean():.1f}%")
ax.fill_between(results_df["test_month"], hr, 50,
                where=hr > 50, alpha=0.2, color="#1D9E75")
ax.set_ylim(0, 100)
ax.set_title("Hit rate of Top-10 portfolio (% stocks with positive return) — FIXED")
ax.set_xlabel("Month")
ax.set_ylabel("Hit rate (%)")
ax.legend()
plt.xticks(rotation=45)
ax.set_xticks(results_df["test_month"].values[::step])
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "05_hit_rate_fixed.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 05_hit_rate_fixed.png")

# Plot 6: Monthly turnover
fig, ax = plt.subplots(figsize=(16, 4))
ax.bar(range(len(portfolio_df)), portfolio_df["turnover"] * 100,
       color="#378ADD", alpha=0.7, width=0.8)
ax.axhline(portfolio_df["turnover"].mean() * 100, color="#E24B4A",
           linewidth=1.5, linestyle="--",
           label=f"Mean turnover: {portfolio_df['turnover'].mean()*100:.1f}%")
ax.set_title("Monthly portfolio turnover (%)")
ax.set_xlabel("Fold")
ax.set_ylabel("Turnover (%)")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "06_monthly_turnover.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 06_monthly_turnover.png")

print(f"\nAll results saved to: {RESULTS_DIR}")
print("Done.")