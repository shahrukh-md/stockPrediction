"""
V4 Optuna Tuning + Window Search — WITH INTERACTION FEATURES
Identical to tune_v4.py but uses nifty50_v4_interaction.csv (37 features)

Run: python models/tune_v4_interaction.py

Output:
  results_interaction/best_params.json
  results_interaction/window_search.csv
  results_interaction/fold_results.csv
  results_interaction/monthly_portfolio.csv
  results_interaction/summary.txt
  results_interaction/plots/
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
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE   = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_v4_interaction.csv"
RESULTS_DIR  = "/Users/shahrukh/Desktop/stock/results_interaction"
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TOP_N            = 10
BOTTOM_N         = 10
TARGET           = "target_cs_zscore"
TARGET_RAW       = "target_return_30d"
RANDOM_STATE     = 42
TRANSACTION_COST = 0.003
WINDOW_CANDIDATES= [24, 30, 36, 48]
N_TRIALS         = 40

# ── FEATURE COLS: 26 base + 11 interaction = 37 total ────────────────────────
FEATURE_COLS = [
    # Base features (26)
    "price_to_ma200", "BB_width", "BB_pct",
    "return_1d", "return_5d", "return_10d", "return_20d",
    "return_3m", "return_6m", "return_12m",
    "dist_52w_high", "MACD_sig", "MACD_diff",
    "realvol_20d", "realvol_60d", "vol_ratio_2060", "atr_pct",
    "vol_ratio",
    "forward_PE", "profit_margin", "operating_margin",
    "debt_to_equity", "book_value", "price_to_book",
    "dividend_yield", "beta",

    # Interaction features (11)
    "mom6m_vol_adj",       # return_6m / realvol_60d
    "mom3m_vol_adj",       # return_3m / realvol_20d
    "mom12m_vol_adj",      # return_12m / realvol_60d
    "mean_rev_strength",   # dist_52w_high × (1 - BB_pct)
    "bb_squeeze_mom",      # (1/BB_width) × return_20d
    "quality_momentum",    # return_6m × profit_margin
    "value_momentum",      # return_3m × (-price_to_book)
    "vol_trend_mom",       # vol_ratio_2060 × return_20d
    "atr_momentum",        # atr_pct × return_6m
    "macd_mom_confirm",    # MACD_sig × return_20d
    "macd_vol_confirm",    # MACD_diff / realvol_20d
]

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading nifty50_v4_interaction.csv...")
df = pd.read_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)
df = df.sort_values(["date", "Ticker"]).reset_index(drop=True)
df["year_month"] = df["date"].dt.to_period("M")

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"  WARNING — missing features: {missing}")
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

all_months   = sorted(df["year_month"].unique())
total_months = len(all_months)

print(f"  Shape    : {df.shape}")
print(f"  Tickers  : {df['Ticker'].nunique()}")
print(f"  Months   : {total_months}")
print(f"  Features : {len(FEATURE_COLS)} (26 base + 11 interaction)\n")

# ── CORE METRIC ───────────────────────────────────────────────────────────────
def rank_ic(actual_cs, predicted):
    ic, _ = stats.spearmanr(predicted, actual_cs)
    return float(ic) if not np.isnan(ic) else 0.0

def compute_fold_metrics(actual_cs, predicted, actual_raw,
                         prev_long=None, prev_short=None):
    n = len(actual_cs)
    if n < TOP_N + BOTTOM_N:
        return None
    ic         = rank_ic(actual_cs, predicted)
    sorted_idx = np.argsort(predicted)[::-1]
    top_idx    = sorted_idx[:TOP_N]
    bot_idx    = sorted_idx[-BOTTOM_N:]
    top_ret    = actual_raw[top_idx]
    bot_ret    = actual_raw[bot_idx]
    hit_rate   = (top_ret > 0).mean() * 100

    if prev_long is not None:
        lt = len(set(top_idx.tolist()) - set(prev_long)) / TOP_N
        st = len(set(bot_idx.tolist())  - set(prev_short)) / BOTTOM_N
        turnover = (lt + st) / 2
    else:
        turnover = 1.0

    cost    = turnover * TRANSACTION_COST
    top_net = top_ret.mean() - cost
    bot_net = bot_ret.mean() + cost

    return {
        "rank_ic":       ic,
        "hit_rate_pct":  hit_rate,
        "top10_gross":   top_ret.mean(),
        "top10_net":     top_net,
        "ls_spread_net": top_net - bot_net,
        "turnover":      turnover,
        "top_idx":       top_idx.tolist(),
        "bottom_idx":    bot_idx.tolist(),
    }

def walk_forward_eval(model_fn, window, n_folds=None):
    total = total_months - window
    if n_folds:
        total = min(total, n_folds)
    ics = []
    for i in range(total):
        tr_months = all_months[i : i + window]
        te_month  = all_months[i + window]
        tr = df[df["year_month"].isin(tr_months)]
        te = df[df["year_month"] == te_month].reset_index(drop=True)
        if len(te) < 20:
            continue
        m = model_fn()
        m.fit(tr[FEATURE_COLS].values, tr[TARGET].values)
        ic = rank_ic(te[TARGET].values, m.predict(te[FEATURE_COLS].values))
        ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0

# ── STEP 1: WINDOW SEARCH ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: WINDOW LENGTH SEARCH")
print("=" * 60)

window_results = []
for w in WINDOW_CANDIDATES:
    def rf_fn(w=w):
        return RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            max_features=0.6, random_state=RANDOM_STATE, n_jobs=-1
        )
    ic = walk_forward_eval(rf_fn, w, n_folds=36)
    window_results.append({"window": w, "mean_ic": ic})
    print(f"  Window {w:2d}M → Mean IC: {ic:+.4f}")

window_df   = pd.DataFrame(window_results).sort_values("mean_ic", ascending=False)
best_window = int(window_df.iloc[0]["window"])
print(f"\n  Best window: {best_window}M (IC: {window_df.iloc[0]['mean_ic']:+.4f})")
window_df.to_csv(os.path.join(RESULTS_DIR, "window_search.csv"), index=False)

# ── STEP 2: OPTUNA TUNING ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: OPTUNA HYPERPARAMETER TUNING")
print("=" * 60)
print(f"Window: {best_window}M | Trials: {N_TRIALS} per model | Folds: 30\n")

best_params = {}

# LightGBM
print("Tuning LightGBM...")
def lgbm_obj(trial):
    p = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 600),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 10, 120),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 60),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        random_state=RANDOM_STATE, verbose=-1,
    )
    return walk_forward_eval(lambda: LGBMRegressor(**p), best_window, 30)

lgbm_study = optuna.create_study(direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
lgbm_study.optimize(lgbm_obj, n_trials=N_TRIALS)
best_params["LightGBM"] = {**lgbm_study.best_params, "random_state": RANDOM_STATE, "verbose": -1}
print(f"  Best IC: {lgbm_study.best_value:+.4f}")
print(f"  Params : {lgbm_study.best_params}")

# XGBoost
print("\nTuning XGBoost...")
def xgb_obj(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, 600),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 20),
        gamma            = trial.suggest_float("gamma", 0.0, 1.0),
        random_state=RANDOM_STATE, verbosity=0,
    )
    return walk_forward_eval(lambda: XGBRegressor(**p), best_window, 30)

xgb_study = optuna.create_study(direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
xgb_study.optimize(xgb_obj, n_trials=N_TRIALS)
best_params["XGBoost"] = {**xgb_study.best_params, "random_state": RANDOM_STATE, "verbosity": 0}
print(f"  Best IC: {xgb_study.best_value:+.4f}")
print(f"  Params : {xgb_study.best_params}")

# Random Forest
print("\nTuning Random Forest...")
def rf_obj(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, 500),
        max_depth        = trial.suggest_int("max_depth", 3, 12),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 20),
        max_features     = trial.suggest_float("max_features", 0.3, 1.0),
        min_samples_split= trial.suggest_int("min_samples_split", 2, 20),
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    return walk_forward_eval(lambda: RandomForestRegressor(**p), best_window, 30)

rf_study = optuna.create_study(direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
rf_study.optimize(rf_obj, n_trials=N_TRIALS)
best_params["RandomForest"] = {**rf_study.best_params, "random_state": RANDOM_STATE, "n_jobs": -1}
print(f"  Best IC: {rf_study.best_value:+.4f}")
print(f"  Params : {rf_study.best_params}")

with open(os.path.join(RESULTS_DIR, "best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)
print(f"\n  Best params saved to results_interaction/best_params.json")

# ── STEP 3: FULL WALK-FORWARD ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: FULL WALK-FORWARD WITH TUNED PARAMS + INTERACTION FEATURES")
print("=" * 60)

total_folds = total_months - best_window
print(f"Window: {best_window}M | Folds: {total_folds} | Features: {len(FEATURE_COLS)}\n")

def get_models():
    return {
        "LightGBM":    LGBMRegressor(**best_params["LightGBM"]),
        "XGBoost":     XGBRegressor(**best_params["XGBoost"]),
        "RandomForest":RandomForestRegressor(**best_params["RandomForest"]),
    }

fold_results      = []
monthly_portfolio = []
fi = {n: np.zeros(len(FEATURE_COLS)) for n in ["LightGBM","XGBoost","RandomForest"]}
fi_c = {n: 0 for n in fi}
prev_long = prev_short = None

for i in range(total_folds):
    tr_months = all_months[i : i + best_window]
    te_month  = all_months[i + best_window]
    tr = df[df["year_month"].isin(tr_months)]
    te = df[df["year_month"] == te_month].reset_index(drop=True)
    if len(te) < 20:
        continue

    X_tr = tr[FEATURE_COLS].values
    y_tr = tr[TARGET].values
    X_te = te[FEATURE_COLS].values
    a_cs = te[TARGET].values
    a_rw = te[TARGET_RAW].values

    fold_row  = {"fold": i+1, "test_month": str(te_month),
                 "n_stocks": len(te), "nifty_return": a_rw.mean()}
    fold_preds = {}

    for name, model in get_models().items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        fold_preds[name] = preds
        m = compute_fold_metrics(a_cs, preds, a_rw, prev_long, prev_short)
        if m:
            for k, v in m.items():
                if k not in ["top_idx","bottom_idx"]:
                    fold_row[f"{name}_{k}"] = v
        if hasattr(model, "feature_importances_"):
            fi[name] += model.feature_importances_
            fi_c[name] += 1

    ens = np.mean([fold_preds[n] for n in fold_preds], axis=0)
    m_e = compute_fold_metrics(a_cs, ens, a_rw, prev_long, prev_short)
    if m_e:
        for k, v in m_e.items():
            if k not in ["top_idx","bottom_idx"]:
                fold_row[f"Ensemble_{k}"] = v
        prev_long  = m_e["top_idx"]
        prev_short = m_e["bottom_idx"]
        si = np.argsort(ens)[::-1]
        monthly_portfolio.append({
            "month":            str(te_month),
            "long_portfolio":   ", ".join(te.iloc[si[:TOP_N]]["Ticker"].values),
            "short_portfolio":  ", ".join(te.iloc[si[-BOTTOM_N:]]["Ticker"].values),
            "long_return_gross":a_rw[si[:TOP_N]].mean(),
            "long_return_net":  m_e["top10_net"],
            "ls_spread_net":    m_e["ls_spread_net"],
            "turnover":         m_e["turnover"],
            "nifty_return":     a_rw.mean(),
        })
    fold_results.append(fold_row)

    if (i+1) % 12 == 0 or i == total_folds-1:
        ens_ic = fold_row.get("Ensemble_rank_ic", float("nan"))
        ens_sp = fold_row.get("Ensemble_ls_spread_net", float("nan"))
        print(f"  Fold {i+1:3d}/{total_folds} | {te_month} | "
              f"Ensemble IC: {ens_ic:+.3f} | Net L/S: {ens_sp:+.4f}")

print("\nWalk-forward complete.\n")

# ── RESULTS ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("RESULTS — TUNED + INTERACTION FEATURES")
print("=" * 60)

results_df   = pd.DataFrame(fold_results)
portfolio_df = pd.DataFrame(monthly_portfolio)
results_df.to_csv(os.path.join(RESULTS_DIR, "fold_results.csv"), index=False)
portfolio_df.to_csv(os.path.join(RESULTS_DIR, "monthly_portfolio.csv"), index=False)

# Compare vs previous best (tuned without interactions)
prev_path = "/Users/shahrukh/Desktop/stock/results_tuned/fold_results.csv"
prev_df   = pd.read_csv(prev_path) if os.path.exists(prev_path) else None

model_names = ["LightGBM", "XGBoost", "RandomForest", "Ensemble"]
summary_lines = ["="*60, "V4 RESULTS — TUNED + INTERACTION FEATURES", "="*60, "",
    f"Features       : {len(FEATURE_COLS)} (26 base + 11 interaction)",
    f"Best window    : {best_window}M",
    f"Optuna trials  : {N_TRIALS} per model",
    f"Total folds    : {len(results_df)}", ""]

print(f"\n  {'Model':<15} {'Mean IC':>9} {'IC>0%':>7} {'Hit Rate':>10} "
      f"{'Gross':>9} {'Net':>9} {'Net Spread':>11} {'vs prev':>10}")
print("  " + "-" * 90)

for name in model_names:
    ic_c = f"{name}_rank_ic"
    hr_c = f"{name}_hit_rate_pct"
    sp_c = f"{name}_ls_spread_net"
    rg_c = f"{name}_top10_gross"
    rn_c = f"{name}_top10_net"
    if ic_c not in results_df.columns:
        continue

    mean_ic  = results_df[ic_c].mean()
    ic_pos   = (results_df[ic_c] > 0).mean() * 100
    mean_hr  = results_df[hr_c].mean()
    mean_rg  = results_df[rg_c].mean()
    mean_rn  = results_df[rn_c].mean()
    mean_sp  = results_df[sp_c].mean()

    delta_str = ""
    if prev_df is not None and ic_c in prev_df.columns:
        delta = mean_ic - prev_df[ic_c].mean()
        delta_str = f"Δ {delta:+.4f}"

    target_hit = " ← TARGET HIT" if mean_ic >= 0.10 else ""
    line = (f"  {name:<15} {mean_ic:>+9.4f} {ic_pos:>6.1f}% "
            f"{mean_hr:>9.1f}% {mean_rg:>+8.4f} {mean_rn:>+8.4f} "
            f"{mean_sp:>+10.4f} {delta_str:>10}{target_hit}")
    print(line)
    summary_lines.append(line)

nifty_mean  = results_df["nifty_return"].mean()
ens_ic_mean = results_df.get("Ensemble_rank_ic", pd.Series([0])).mean()
print(f"\n  Nifty benchmark : {nifty_mean:+.4f}/month")
print(f"  Avg turnover    : {portfolio_df['turnover'].mean()*100:.1f}%")

summary_lines += ["", f"Nifty benchmark : {nifty_mean:+.4f}/month",
    f"Ensemble Mean IC: {ens_ic_mean:+.4f}",
    "TARGET HIT" if ens_ic_mean >= 0.10 else f"IC={ens_ic_mean:.4f} — close. Consider Nifty 100 expansion next."]

with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))

# ── PLOTS ─────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")
step = max(1, len(results_df) // 12)
colors = {"LightGBM":"#378ADD","XGBoost":"#E24B4A","RandomForest":"#1D9E75","Ensemble":"#7F3FBF"}

# Plot 1: IC over time
fig, ax = plt.subplots(figsize=(16, 5))
for name, color in colors.items():
    col = f"{name}_rank_ic"
    if col in results_df.columns:
        ax.plot(results_df["test_month"], results_df[col],
                label=name, color=color, linewidth=1.2, alpha=0.85)
ax.axhline(0,    color="black", linewidth=1,   linestyle="--")
ax.axhline(0.05, color="gray",  linewidth=0.8, linestyle=":", label="IC=0.05")
ax.axhline(0.10, color="green", linewidth=1.2, linestyle="-.", label="IC=0.10 target")
ax.set_title(f"Rank IC — tuned + interaction features (window={best_window}M)")
ax.set_ylabel("Rank IC")
ax.set_xticks(results_df["test_month"].values[::step])
ax.legend(fontsize=9)
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "01_rank_ic.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: 01_rank_ic.png")

# Plot 2: Cumulative return
fig, ax = plt.subplots(figsize=(16, 6))
if "Ensemble_top10_gross" in results_df.columns:
    ax.plot(results_df["test_month"],
            (1+results_df["Ensemble_top10_gross"]).cumprod(),
            label="Ensemble Top-10 (Gross)", color="#7F3FBF", linewidth=2)
if "Ensemble_top10_net" in results_df.columns:
    ax.plot(results_df["test_month"],
            (1+results_df["Ensemble_top10_net"]).cumprod(),
            label="Ensemble Top-10 (Net)", color="#378ADD", linewidth=2, linestyle="--")
ax.plot(results_df["test_month"],
        (1+results_df["nifty_return"]).cumprod(),
        label="Nifty benchmark", color="#E24B4A", linewidth=1.5, linestyle=":")
ax.axhline(1, color="black", linewidth=0.8)
ax.set_title("Cumulative return — interaction features model vs Nifty")
ax.set_xlabel("Month")
ax.set_ylabel("₹1 invested")
ax.legend()
ax.set_xticks(results_df["test_month"].values[::step])
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "02_cumulative_return.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: 02_cumulative_return.png")

# Plot 3: Feature importance — highlight interaction features
fig, ax = plt.subplots(figsize=(14, 10))
avg_fi = np.zeros(len(FEATURE_COLS))
cnt = 0
for name in ["LightGBM","XGBoost","RandomForest"]:
    if fi_c[name] > 0:
        avg_fi += fi[name] / fi_c[name]
        cnt += 1
if cnt > 0:
    avg_fi /= cnt
    fi_s = pd.Series(avg_fi, index=FEATURE_COLS).sort_values(ascending=True)
    bar_colors = ["#1D9E75" if f in [
        "mom6m_vol_adj","mom3m_vol_adj","mom12m_vol_adj",
        "mean_rev_strength","bb_squeeze_mom","quality_momentum",
        "value_momentum","vol_trend_mom","atr_momentum",
        "macd_mom_confirm","macd_vol_confirm"
    ] else "#378ADD" for f in fi_s.index]
    ax.barh(fi_s.index, fi_s.values, color=bar_colors, alpha=0.85)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#1D9E75", alpha=0.85, label="Interaction feature"),
        Patch(facecolor="#378ADD", alpha=0.85, label="Base feature"),
    ])
    ax.set_title("Feature importance — green = interaction features")
    ax.set_xlabel("Importance score")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "03_feature_importance.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: 03_feature_importance.png")

# Plot 4: Optuna progress
fig, ax = plt.subplots(figsize=(10, 5))
for study, name, color in [
    (lgbm_study, "LightGBM", "#378ADD"),
    (xgb_study,  "XGBoost",  "#E24B4A"),
    (rf_study,   "RF",       "#1D9E75"),
]:
    vals = [t.value for t in study.trials if t.value is not None]
    ax.plot(np.maximum.accumulate(vals), label=name, color=color, linewidth=1.5)
ax.axhline(0.10, color="green", linewidth=1, linestyle="-.", label="IC=0.10 target")
ax.set_title("Optuna optimization — best IC per trial")
ax.set_xlabel("Trial")
ax.set_ylabel("Best IC so far")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "04_optuna_progress.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: 04_optuna_progress.png")

print(f"\nAll results saved to: {RESULTS_DIR}")
print("Done.")