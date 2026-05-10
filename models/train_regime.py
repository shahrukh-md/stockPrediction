"""
V4 Regime Layer + Walk-Forward Training
Detects market regimes using:
  1. Average pairwise correlation of 48 stocks (20-day rolling)
  2. Nifty 50 realized volatility (20-day rolling)
  3. Composite regime score → position scale

Run: python models/train_regime.py

Input:
  nifty50_v4_model_ready.csv   ← daily data for regime detection
  nifty50_v4_interaction.csv   ← monthly data for model training
  results_interaction/best_params.json  ← tuned hyperparameters

Output:
  results_regime/regime_history.csv
  results_regime/fold_results.csv
  results_regime/monthly_portfolio.csv
  results_regime/summary.txt
  results_regime/plots/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os, json
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# ── CONFIG ────────────────────────────────────────────────────────────────────
DAILY_FILE   = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_v4_model_ready.csv"
MONTHLY_FILE = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_v4_interaction.csv"
PARAMS_FILE  = "/Users/shahrukh/Desktop/stock/results_interaction/best_params.json"
RESULTS_DIR  = "/Users/shahrukh/Desktop/stock/results_regime"
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TOP_N            = 10
BOTTOM_N         = 10
TARGET           = "target_cs_zscore"
TARGET_RAW       = "target_return_30d"
RANDOM_STATE     = 42
TRANSACTION_COST = 0.003
TRAIN_WINDOW     = 24   # best window from interaction tuning

# Regime thresholds
CORR_NORMAL    = 0.50   # below this = normal regime
CORR_CRISIS    = 0.55   # above this = crisis regime
VOL_NORMAL     = 0.15   # below this = normal vol (annualized)
VOL_CRISIS     = 0.28   # above this = crisis vol (annualized)
LOOKBACK_DAYS  = 20     # days to look back for regime detection

FEATURE_COLS = [
    # Base (26)
    "price_to_ma200", "BB_width", "BB_pct",
    "return_1d", "return_5d", "return_10d", "return_20d",
    "return_3m", "return_6m", "return_12m",
    "dist_52w_high", "MACD_sig", "MACD_diff",
    "realvol_20d", "realvol_60d", "vol_ratio_2060", "atr_pct",
    "vol_ratio",
    "forward_PE", "profit_margin", "operating_margin",
    "debt_to_equity", "book_value", "price_to_book",
    "dividend_yield", "beta",
    # Interaction (11)
    "mom6m_vol_adj", "mom3m_vol_adj", "mom12m_vol_adj",
    "mean_rev_strength", "bb_squeeze_mom",
    "quality_momentum", "value_momentum",
    "vol_trend_mom", "atr_momentum",
    "macd_mom_confirm", "macd_vol_confirm",
]

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading data...")
daily_df = pd.read_csv(DAILY_FILE, parse_dates=["date"], low_memory=False)
daily_df = daily_df.sort_values(["date", "Ticker"]).reset_index(drop=True)

monthly_df = pd.read_csv(MONTHLY_FILE, parse_dates=["date"], low_memory=False)
monthly_df = monthly_df.sort_values(["date", "Ticker"]).reset_index(drop=True)
monthly_df["year_month"] = monthly_df["date"].dt.to_period("M")

# Load tuned params
with open(PARAMS_FILE, "r") as f:
    best_params = json.load(f)

missing = [c for c in FEATURE_COLS if c not in monthly_df.columns]
if missing:
    print(f"  WARNING — missing: {missing}")
    FEATURE_COLS = [c for c in FEATURE_COLS if c in monthly_df.columns]

print(f"  Daily  : {daily_df.shape}  ({daily_df['Ticker'].nunique()} tickers)")
print(f"  Monthly: {monthly_df.shape} ({monthly_df['Ticker'].nunique()} tickers)")
print(f"  Features: {len(FEATURE_COLS)}\n")

# ── REGIME DETECTION ──────────────────────────────────────────────────────────
print("=" * 60)
print("BUILDING REGIME HISTORY")
print("=" * 60)

def compute_regime_for_month(test_month_period, daily_df, lookback=20):
    """
    For a given test month, look back `lookback` trading days
    in daily data and compute:
      1. Average pairwise return correlation across all stocks
      2. Nifty 50 realized volatility (annualized)
    Returns: regime label, avg_corr, nifty_vol, position_scale
    """
    # Get the last date before the test month starts
    test_month_start = test_month_period.to_timestamp()

    # All daily dates before this test month
    available = daily_df[daily_df["date"] < test_month_start]["date"].unique()
    available = sorted(available)

    if len(available) < lookback:
        return "normal", 0.0, 0.0, 1.0

    # Use last `lookback` trading days
    lookback_dates = available[-lookback:]
    window_df = daily_df[daily_df["date"].isin(lookback_dates)].copy()

    # ── Signal 1: Average pairwise correlation ────────────────────────────────
    try:
        pivot = window_df.pivot_table(
            index="date", columns="Ticker", values="return_1d"
        )
        # Drop tickers with too many NaNs
        pivot = pivot.dropna(axis=1, thresh=int(lookback * 0.7))
        if pivot.shape[1] < 5:
            avg_corr = 0.0
        else:
            corr_matrix = pivot.corr()
            upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            avg_corr = float(corr_matrix.where(upper).stack().mean())
            if np.isnan(avg_corr):
                avg_corr = 0.0
    except Exception:
        avg_corr = 0.0

    # ── Signal 2: Nifty realized vol ─────────────────────────────────────────
    try:
        nifty_returns = window_df.drop_duplicates("date").set_index("date")["Nifty50_return"]
        nifty_vol = float(nifty_returns.std() * np.sqrt(252))
        if np.isnan(nifty_vol):
            nifty_vol = 0.0
    except Exception:
        nifty_vol = 0.0

    # ── Composite regime score (0 = normal, 1 = crisis) ──────────────────────
    # Normalize each signal to 0-1 range
    corr_score = max(0.0, min(1.0, (avg_corr - CORR_NORMAL) / (CORR_CRISIS - CORR_NORMAL)))
    vol_score  = max(0.0, min(1.0, (nifty_vol  - VOL_NORMAL) / (VOL_CRISIS  - VOL_NORMAL)))

    # Weighted composite: correlation gets 60%, vol gets 40%
    composite_score = 0.6 * corr_score + 0.4 * vol_score

    # Position scale: 1.0 at score=0, 0.0 at score=1.0
    position_scale = max(0.0, 1.0 - composite_score)
    position_scale = round(position_scale, 3)

    # Regime label
    if composite_score < 0.25:
        regime = "normal"
    elif composite_score < 0.60:
        regime = "elevated"
    else:
        regime = "crisis"

    return regime, round(avg_corr, 4), round(nifty_vol, 4), position_scale

# Build regime history for all months
all_months = sorted(monthly_df["year_month"].unique())
regime_history = []

print(f"  Computing regime for {len(all_months)} months...")
for month in all_months:
    regime, avg_corr, nifty_vol, scale = compute_regime_for_month(
        month, daily_df, LOOKBACK_DAYS
    )
    regime_history.append({
        "year_month":     str(month),
        "regime":         regime,
        "avg_pairwise_corr": avg_corr,
        "nifty_vol_20d":  nifty_vol,
        "position_scale": scale,
    })

regime_df = pd.DataFrame(regime_history)
regime_df.to_csv(os.path.join(RESULTS_DIR, "regime_history.csv"), index=False)

# Summary
counts = regime_df["regime"].value_counts()
print(f"\n  for CORR_CRISIS {CORR_CRISIS} and VOL_CRISIS {VOL_CRISIS} months:")
print(f"\n  Regime distribution across {len(regime_df)} months:")
for r, c in counts.items():
    pct = c / len(regime_df) * 100
    avg_scale = regime_df[regime_df["regime"]==r]["position_scale"].mean()
    print(f"    {r:<12}: {c:3d} months ({pct:.1f}%)  avg scale={avg_scale:.2f}")

crisis_months = regime_df[regime_df["regime"] == "crisis"]["year_month"].tolist()
elevated_months = regime_df[regime_df["regime"] == "elevated"]["year_month"].tolist()
print(f"\n  Crisis months   : {crisis_months}")
print(f"  Elevated months : {elevated_months[:10]}{'...' if len(elevated_months)>10 else ''}")

# ── MODELS ────────────────────────────────────────────────────────────────────
def get_models():
    return {
        "LightGBM":    LGBMRegressor(**best_params["LightGBM"]),
        "XGBoost":     XGBRegressor(**best_params["XGBoost"]),
        "RandomForest":RandomForestRegressor(**best_params["RandomForest"]),
    }

# ── METRICS ───────────────────────────────────────────────────────────────────
def compute_metrics(actual_cs, predicted, actual_raw,
                    prev_long=None, prev_short=None, position_scale=1.0):
    n = len(actual_cs)
    if n < TOP_N + BOTTOM_N:
        return None

    ic = float(stats.spearmanr(predicted, actual_cs)[0])
    if np.isnan(ic):
        ic = 0.0

    sorted_idx = np.argsort(predicted)[::-1]

    # Scale portfolio size with regime
    n_long  = max(1, int(TOP_N  * position_scale))
    n_short = max(1, int(BOTTOM_N * position_scale))

    top_idx = sorted_idx[:n_long]
    bot_idx = sorted_idx[-n_short:] if n_short > 0 else np.array([], dtype=int)

    top_ret = actual_raw[top_idx]
    bot_ret = actual_raw[bot_idx] if len(bot_idx) > 0 else np.array([0.0])

    hit_rate = (top_ret > 0).mean() * 100

    if prev_long is not None:
        lt = len(set(top_idx.tolist()) - set(prev_long)) / max(n_long, 1)
        st = len(set(bot_idx.tolist()) - set(prev_short)) / max(n_short, 1)
        turnover = (lt + st) / 2
    else:
        turnover = 1.0

    cost    = turnover * TRANSACTION_COST * position_scale
    top_net = top_ret.mean() * position_scale - cost
    bot_net = bot_ret.mean() * position_scale + cost if len(bot_idx) > 0 else 0.0
    ls_net  = top_net - bot_net

    return {
        "rank_ic":        ic,
        "hit_rate_pct":   hit_rate,
        "top10_gross":    top_ret.mean(),
        "top10_net":      top_net,
        "bot10_gross":    bot_ret.mean() if len(bot_ret) > 0 else 0.0,
        "ls_spread_net":  ls_net,
        "turnover":       turnover,
        "position_scale": position_scale,
        "n_long":         n_long,
        "n_short":        n_short,
        "top_idx":        top_idx.tolist(),
        "bottom_idx":     bot_idx.tolist(),
    }

# ── WALK-FORWARD WITH REGIME LAYER ────────────────────────────────────────────
print("\n" + "=" * 60)
print("WALK-FORWARD TRAINING WITH REGIME LAYER")
print("=" * 60)

total_folds = len(all_months) - TRAIN_WINDOW
print(f"  Window : {TRAIN_WINDOW}M rolling")
print(f"  Folds  : {total_folds}")
print(f"  Features: {len(FEATURE_COLS)}\n")

# Build regime lookup dict
regime_lookup = {
    row["year_month"]: row
    for _, row in regime_df.iterrows()
}

fold_results      = []
monthly_portfolio = []
fi = {n: np.zeros(len(FEATURE_COLS)) for n in ["LightGBM","XGBoost","RandomForest"]}
fi_c = {n: 0 for n in fi}
prev_long = prev_short = None

for i in range(total_folds):
    tr_months = all_months[i : i + TRAIN_WINDOW]
    te_month  = all_months[i + TRAIN_WINDOW]

    tr = monthly_df[monthly_df["year_month"].isin(tr_months)]
    te = monthly_df[monthly_df["year_month"] == te_month].reset_index(drop=True)

    if len(te) < 20:
        continue

    # Get regime for this test month
    regime_info    = regime_lookup.get(str(te_month), {})
    regime         = regime_info.get("regime", "normal")
    avg_corr       = regime_info.get("avg_pairwise_corr", 0.0)
    nifty_vol      = regime_info.get("nifty_vol_20d", 0.0)
    position_scale = regime_info.get("position_scale", 1.0)

    X_tr = tr[FEATURE_COLS].values
    y_tr = tr[TARGET].values
    X_te = te[FEATURE_COLS].values
    a_cs = te[TARGET].values
    a_rw = te[TARGET_RAW].values

    fold_row = {
        "fold":         i + 1,
        "test_month":   str(te_month),
        "n_stocks":     len(te),
        "nifty_return": a_rw.mean(),
        "regime":       regime,
        "avg_corr":     avg_corr,
        "nifty_vol":    nifty_vol,
        "position_scale": position_scale,
    }

    fold_preds = {}
    for name, model in get_models().items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        fold_preds[name] = preds

        m = compute_metrics(a_cs, preds, a_rw, prev_long, prev_short, position_scale)
        if m:
            for k, v in m.items():
                if k not in ["top_idx", "bottom_idx"]:
                    fold_row[f"{name}_{k}"] = v

        if hasattr(model, "feature_importances_"):
            fi[name] += model.feature_importances_
            fi_c[name] += 1

    # Ensemble
    ens = np.mean([fold_preds[n] for n in fold_preds], axis=0)
    m_e = compute_metrics(a_cs, ens, a_rw, prev_long, prev_short, position_scale)

    if m_e:
        for k, v in m_e.items():
            if k not in ["top_idx", "bottom_idx"]:
                fold_row[f"Ensemble_{k}"] = v
        prev_long  = m_e["top_idx"]
        prev_short = m_e["bottom_idx"]

        si = np.argsort(ens)[::-1]
        n_long  = m_e["n_long"]
        n_short = m_e["n_short"]

        monthly_portfolio.append({
            "month":            str(te_month),
            "regime":           regime,
            "position_scale":   position_scale,
            "avg_corr":         avg_corr,
            "nifty_vol":        nifty_vol,
            "n_long":           n_long,
            "n_short":          n_short,
            "long_portfolio":   ", ".join(te.iloc[si[:n_long]]["Ticker"].values),
            "short_portfolio":  ", ".join(te.iloc[si[-n_short:]]["Ticker"].values) if n_short > 0 else "",
            "long_return_gross":a_rw[si[:n_long]].mean(),
            "long_return_net":  m_e["top10_net"],
            "ls_spread_net":    m_e["ls_spread_net"],
            "turnover":         m_e["turnover"],
            "nifty_return":     a_rw.mean(),
        })

    fold_results.append(fold_row)

    if (i + 1) % 12 == 0 or i == total_folds - 1:
        ens_ic = fold_row.get("Ensemble_rank_ic", float("nan"))
        ens_sp = fold_row.get("Ensemble_ls_spread_net", float("nan"))
        regime_str = f"[{regime.upper()[:4]}|corr={avg_corr:.2f}|scale={position_scale:.1f}]"
        print(f"  Fold {i+1:3d}/{total_folds} | {te_month} | "
              f"IC: {ens_ic:+.3f} | Net L/S: {ens_sp:+.4f} | {regime_str}")

print("\nWalk-forward complete.\n")

# ── RESULTS ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("RESULTS — REGIME-AWARE MODEL")
print("=" * 60)

results_df   = pd.DataFrame(fold_results)
portfolio_df = pd.DataFrame(monthly_portfolio)
results_df.to_csv(os.path.join(RESULTS_DIR, "fold_results.csv"), index=False)
portfolio_df.to_csv(os.path.join(RESULTS_DIR, "monthly_portfolio.csv"), index=False)

# Load previous results for comparison
prev_path = "/Users/shahrukh/Desktop/stock/results_interaction/fold_results.csv"
prev_df   = pd.read_csv(prev_path) if os.path.exists(prev_path) else None

model_names = ["LightGBM", "XGBoost", "RandomForest", "Ensemble"]
summary_lines = ["="*60, "V4 REGIME-AWARE RESULTS", "="*60, "",
    f"Features        : {len(FEATURE_COLS)} (26 base + 11 interaction)",
    f"Training window : {TRAIN_WINDOW}M rolling",
    f"Total folds     : {len(results_df)}",
    f"Regime signals  : avg pairwise corr (60%) + Nifty vol (40%)",
    f"Lookback        : {LOOKBACK_DAYS} trading days", ""]

print(f"\n  {'Model':<15} {'Mean IC':>9} {'IC>0%':>7} {'Hit Rate':>10} "
      f"{'Net Ret':>9} {'Net Spread':>11} {'vs prev':>10}")
print("  " + "-" * 85)

for name in model_names:
    ic_c = f"{name}_rank_ic"
    hr_c = f"{name}_hit_rate_pct"
    sp_c = f"{name}_ls_spread_net"
    rn_c = f"{name}_top10_net"
    if ic_c not in results_df.columns:
        continue

    mean_ic = results_df[ic_c].mean()
    ic_pos  = (results_df[ic_c] > 0).mean() * 100
    mean_hr = results_df[hr_c].mean()
    mean_rn = results_df[rn_c].mean()
    mean_sp = results_df[sp_c].mean()

    delta_str = ""
    if prev_df is not None and ic_c in prev_df.columns:
        delta = mean_ic - prev_df[ic_c].mean()
        delta_str = f"Δ {delta:+.4f}"

    target = " ← TARGET" if mean_ic >= 0.10 else ""
    line = (f"  {name:<15} {mean_ic:>+9.4f} {ic_pos:>6.1f}% "
            f"{mean_hr:>9.1f}% {mean_rn:>+8.4f} {mean_sp:>+10.4f} "
            f"{delta_str:>10}{target}")
    print(line)
    summary_lines.append(line)

nifty_mean  = results_df["nifty_return"].mean()
ens_ic_mean = results_df.get("Ensemble_rank_ic", pd.Series([0])).mean()
regime_counts = results_df["regime"].value_counts()

print(f"\n  Nifty benchmark : {nifty_mean:+.4f}/month")
print(f"  Regime breakdown:")
for r, c in regime_counts.items():
    avg_ic = results_df[results_df["regime"]==r]["Ensemble_rank_ic"].mean()
    print(f"    {r:<12}: {c:3d} months | avg Ensemble IC: {avg_ic:+.4f}")

summary_lines += ["", f"Nifty benchmark: {nifty_mean:+.4f}/month",
    "Regime breakdown:"]
for r, c in regime_counts.items():
    avg_ic = results_df[results_df["regime"]==r]["Ensemble_rank_ic"].mean()
    summary_lines.append(f"  {r}: {c} months | avg IC: {avg_ic:+.4f}")

with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))
print(f"\n  Summary saved.")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")
step = max(1, len(results_df) // 12)
colors_m = {"LightGBM":"#378ADD","XGBoost":"#E24B4A",
             "RandomForest":"#1D9E75","Ensemble":"#7F3FBF"}

# Plot 1: Regime history
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

ax = axes[0]
ax.plot(regime_df["year_month"], regime_df["avg_pairwise_corr"],
        color="#378ADD", linewidth=1.2, label="Avg pairwise corr")
ax.axhline(CORR_NORMAL, color="orange", linestyle="--",
           linewidth=1, label=f"Normal threshold ({CORR_NORMAL})")
ax.axhline(CORR_CRISIS, color="red", linestyle="--",
           linewidth=1, label=f"Crisis threshold ({CORR_CRISIS})")
ax.fill_between(regime_df["year_month"],
                regime_df["avg_pairwise_corr"],
                CORR_NORMAL,
                where=regime_df["avg_pairwise_corr"] > CORR_NORMAL,
                alpha=0.2, color="red")
ax.set_title("Average pairwise correlation across 48 stocks (20-day rolling)")
ax.set_ylabel("Avg correlation")
ax.legend(fontsize=8)
ax.set_xticks(regime_df["year_month"].values[::12])
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

ax = axes[1]
ax.plot(regime_df["year_month"], regime_df["nifty_vol_20d"],
        color="#E24B4A", linewidth=1.2, label="Nifty 20d realized vol")
ax.axhline(VOL_NORMAL, color="orange", linestyle="--",
           linewidth=1, label=f"Normal threshold ({VOL_NORMAL:.0%})")
ax.axhline(VOL_CRISIS, color="red", linestyle="--",
           linewidth=1, label=f"Crisis threshold ({VOL_CRISIS:.0%})")
ax.fill_between(regime_df["year_month"],
                regime_df["nifty_vol_20d"],
                VOL_NORMAL,
                where=regime_df["nifty_vol_20d"] > VOL_NORMAL,
                alpha=0.2, color="red")
ax.set_title("Nifty 50 realized volatility (20-day rolling, annualized)")
ax.set_ylabel("Annualized vol")
ax.legend(fontsize=8)
ax.set_xticks(regime_df["year_month"].values[::12])
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

ax = axes[2]
scale_colors = {"normal":"#1D9E75","elevated":"#F5A623","crisis":"#E24B4A"}
bar_colors   = [scale_colors.get(r,"#378ADD") for r in regime_df["regime"]]
ax.bar(range(len(regime_df)), regime_df["position_scale"],
       color=bar_colors, alpha=0.85, width=0.8)
ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Position scale by regime (green=normal, orange=elevated, red=crisis)")
ax.set_ylabel("Position scale")
ax.set_ylim(0, 1.1)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="#1D9E75", label="Normal"),
    Patch(color="#F5A623", label="Elevated"),
    Patch(color="#E24B4A", label="Crisis"),
])

plt.suptitle("Regime detection history", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "01_regime_history.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: 01_regime_history.png")

# Plot 2: IC over time with regime shading
fig, ax = plt.subplots(figsize=(16, 6))
for name, color in colors_m.items():
    ic_col = f"{name}_rank_ic"
    if ic_col in results_df.columns:
        ax.plot(results_df["test_month"], results_df[ic_col],
                label=name, color=color, linewidth=1.2, alpha=0.85)

# Shade regime months
for _, row in results_df.iterrows():
    if row["regime"] == "crisis":
        ax.axvspan(row["test_month"], row["test_month"],
                   alpha=0.25, color="#E24B4A")
    elif row["regime"] == "elevated":
        ax.axvspan(row["test_month"], row["test_month"],
                   alpha=0.15, color="#F5A623")

ax.axhline(0,    color="black", linewidth=1,   linestyle="--")
ax.axhline(0.05, color="gray",  linewidth=0.8, linestyle=":")
ax.axhline(0.10, color="green", linewidth=1,   linestyle="-.", label="IC=0.10 target")
ax.set_title("Rank IC with regime shading (red=crisis, orange=elevated)")
ax.set_ylabel("Rank IC")
ax.legend(fontsize=9)
ax.set_xticks(results_df["test_month"].values[::step])
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "02_ic_with_regime.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: 02_ic_with_regime.png")

# Plot 3: Cumulative return — regime vs no-regime vs Nifty
fig, ax = plt.subplots(figsize=(16, 6))
if "Ensemble_top10_net" in results_df.columns:
    ax.plot(results_df["test_month"],
            (1 + results_df["Ensemble_top10_net"]).cumprod(),
            label="Regime-aware (net)", color="#7F3FBF", linewidth=2)

if prev_df is not None and "Ensemble_top10_net" in prev_df.columns:
    prev_aligned = prev_df[prev_df["test_month"].isin(
        results_df["test_month"].values)]
    if len(prev_aligned) > 0:
        ax.plot(prev_aligned["test_month"],
                (1 + prev_aligned["Ensemble_top10_net"]).cumprod(),
                label="No regime filter (net)", color="#378ADD",
                linewidth=1.5, linestyle="--")

ax.plot(results_df["test_month"],
        (1 + results_df["nifty_return"]).cumprod(),
        label="Nifty benchmark", color="#E24B4A",
        linewidth=1.5, linestyle=":")
ax.axhline(1, color="black", linewidth=0.8)
ax.set_title("Cumulative return: regime-aware vs no-regime vs Nifty")
ax.set_xlabel("Month")
ax.set_ylabel("₹1 invested")
ax.legend()
ax.set_xticks(results_df["test_month"].values[::step])
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "03_cumulative_regime_vs_baseline.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: 03_cumulative_regime_vs_baseline.png")

# Plot 4: IC by regime box plot
fig, ax = plt.subplots(figsize=(8, 6))
regime_order = ["normal", "elevated", "crisis"]
ic_by_regime = [
    results_df[results_df["regime"] == r]["Ensemble_rank_ic"].dropna().values
    for r in regime_order
    if r in results_df["regime"].values
]
labels_present = [r for r in regime_order if r in results_df["regime"].values]
bp = ax.boxplot(ic_by_regime, labels=labels_present,
                patch_artist=True, showfliers=True)
regime_colors = {"normal":"#1D9E75", "elevated":"#F5A623", "crisis":"#E24B4A"}
for patch, label in zip(bp["boxes"], labels_present):
    patch.set_facecolor(regime_colors.get(label, "#378ADD"))
    patch.set_alpha(0.7)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.axhline(0.10, color="green", linewidth=0.8, linestyle=":", label="IC=0.10")
ax.set_title("Ensemble Rank IC distribution by regime")
ax.set_ylabel("Rank IC")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "04_ic_by_regime.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: 04_ic_by_regime.png")

# Plot 5: Position scale over time
fig, ax = plt.subplots(figsize=(16, 4))
scale_bar_colors = [scale_colors.get(r, "#378ADD")
                    for r in results_df["regime"]]
ax.bar(range(len(results_df)), results_df["position_scale"],
       color=scale_bar_colors, alpha=0.85, width=0.8)
ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Position scale per month — regime filter in action")
ax.set_ylabel("Scale (1.0 = full, 0.0 = flat)")
ax.set_ylim(0, 1.1)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "05_position_scale.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: 05_position_scale.png")

print(f"\nAll results saved to: {RESULTS_DIR}")
print("Done.")