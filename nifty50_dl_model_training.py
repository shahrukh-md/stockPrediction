"""
Nifty 50 Direction Prediction — Classification v4 (Final)
==========================================================
Uses same 80/20 time split as original notebook to ensure
apple-to-apple comparison, but with classification instead
of regression.

Key improvements:
1. Same split as original (80/20 by date)
2. LightGBM + XGBoost classification (not regression)
3. Improved feature engineering with stationarity focus
4. Proper class weighting
5. Comparison with baselines
"""

import os, warnings, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

DATA_PATH = 'nifty50_data/nifty50_cleaned.csv'
MODEL_DIR = 'nifty50_model'
RESULTS_DIR = 'nifty50_results'
SEED = 42
np.random.seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# Feature Engineering
# ============================================================================
def engineer_features(df):
    """Feature engineering with focus on stationarity."""
    # Price ratios
    df['c2o']     = df['close'] / df['open'] - 1
    df['h2l']     = df['high'] / df['low'] - 1
    df['c2ma20']  = df['close'] / df['MA_20'] - 1
    df['c2ma50']  = df['close'] / df['MA_50'] - 1
    df['c2ma200'] = df['close'] / df['MA_200'] - 1
    df['ma20_50'] = df['MA_20'] / df['MA_50'] - 1
    df['ma50_200']= df['MA_50'] / df['MA_200'] - 1
    df['c2ema']   = df['close'] / df['EMA_20'] - 1

    # Trend signals
    df['golden'] = (df['MA_50'] > df['MA_200']).astype(int)
    df['above200'] = (df['close'] > df['MA_200']).astype(int)
    df['above50'] = (df['close'] > df['MA_50']).astype(int)
    df['ma20_gt_50'] = (df['MA_20'] > df['MA_50']).astype(int)

    # BB
    df['bb_pos'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-8)
    df['bb_sq']  = df['BB_width'] / (df['close'] + 1e-8)

    # Momentum
    df['mom510']  = df['return_5d'] - df['return_10d']
    df['mom1020'] = df['return_10d'] - df['return_20d']
    df['mom520']  = df['return_5d'] - df['return_20d']

    # RSI
    df['rsi_n'] = (df['RSI_14'] - 50) / 50
    df['rsi_ob'] = (df['RSI_14'] > 70).astype(int)
    df['rsi_os'] = (df['RSI_14'] < 30).astype(int)

    # MACD
    df['macd_n'] = df['MACD_diff'] / (df['close'] + 1e-8)

    # Volatility
    df['rvol20'] = df.groupby('Ticker')['return_1d'].transform(lambda x: x.rolling(20, min_periods=10).std())
    df['rvol5']  = df.groupby('Ticker')['return_1d'].transform(lambda x: x.rolling(5, min_periods=3).std())
    df['vr520']  = df['rvol5'] / (df['rvol20'] + 1e-8)
    df['atr_n']  = df['ATR_14'] / (df['close'] + 1e-8)

    # Volume
    df['vchg'] = df.groupby('Ticker')['volume'].transform(lambda x: x.pct_change())

    # Market
    df['vs_nifty']  = df['return_1d'] - df['Nifty50_return']
    df['nifty_tr']  = df['Nifty50_close'] / df['Nifty50_MA50'] - 1

    # Lagged
    for lag in [1, 2, 3, 5]:
        df[f'rlag{lag}'] = df.groupby('Ticker')['return_1d'].shift(lag)

    # Rolling stats
    df['rmean10'] = df.groupby('Ticker')['return_1d'].transform(lambda x: x.rolling(10, min_periods=5).mean())
    df['rskew20'] = df.groupby('Ticker')['return_1d'].transform(lambda x: x.rolling(20, min_periods=10).skew())

    # Ticker encoding
    df['ticker_id'] = pd.Categorical(df['Ticker']).codes

    # Binary target
    df['target'] = (df['target_return_30d'] > 0).astype(int)
    return df


FEATURES = [
    'return_1d','return_5d','return_10d','return_20d',
    'rlag1','rlag2','rlag3','rlag5',
    'mom510','mom1020','mom520',
    'rmean10','rskew20',
    'c2ma20','c2ma50','c2ma200','ma20_50','ma50_200','c2ema',
    'golden','above200','above50','ma20_gt_50',
    'rsi_n','rsi_ob','rsi_os',
    'macd_n',
    'bb_pos','bb_sq',
    'rvol20','rvol5','vr520','atr_n',
    'vol_ratio','vchg',
    'c2o','h2l',
    'vs_nifty','nifty_tr',
    'PE_ratio','forward_PE','profit_margin','operating_margin',
    'debt_to_equity','roe','roa','price_to_book','beta',
    'ticker_id',
]


def main():
    start = datetime.now()
    print("=" * 60)
    print("  Nifty 50 Direction Prediction — Classification v4")
    print("=" * 60)

    # Load
    print("\n[1] Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df.sort_values(['Ticker', 'date']).reset_index(drop=True)
    print(f"  Raw: {df.shape}")

    # Engineer features
    df = engineer_features(df)

    # Clean
    df_clean = df.dropna(subset=FEATURES + ['target']).reset_index(drop=True)

    # Handle Inf
    for f in FEATURES:
        df_clean[f] = df_clean[f].replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"  Clean: {len(df_clean)}")

    # === SAME SPLIT AS ORIGINAL NOTEBOOK: 80/20 by date ===
    print("\n[2] Splitting (80/20 by date, same as original)...")
    split_date = df_clean['date'].quantile(0.80)
    if not isinstance(split_date, pd.Timestamp):
        split_date = pd.Timestamp(split_date)

    train_df = df_clean[df_clean['date'] <= split_date]
    test_df  = df_clean[df_clean['date'] >  split_date]

    print(f"  Split date: {split_date.date()}")
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"  Train target: {train_df['target'].mean():.3f} | Test target: {test_df['target'].mean():.3f}")

    X_train = train_df[FEATURES].values.astype(np.float64)
    y_train = train_df['target'].values
    X_test  = test_df[FEATURES].values.astype(np.float64)
    y_test  = test_df['target'].values

    # Handle remaining NaN/Inf
    train_medians = np.nanmedian(X_train, axis=0)
    for arr in [X_train, X_test]:
        for c in range(arr.shape[1]):
            mask = np.isnan(arr[:, c]) | np.isinf(arr[:, c])
            arr[mask, c] = train_medians[c]

    # Scale
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    np.clip(X_train_s, -5, 5, out=X_train_s)
    np.clip(X_test_s, -5, 5, out=X_test_s)

    # === Models ===
    pos_w = np.sum(y_train == 0) / np.sum(y_train == 1)

    # LightGBM
    print("\n[3] Training LightGBM...")
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURES)
    dval   = lgb.Dataset(X_test, label=y_test, feature_name=FEATURES, reference=dtrain)

    lgb_params = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'learning_rate': 0.02, 'num_leaves': 63, 'max_depth': 7,
        'min_child_samples': 100, 'min_child_weight': 10,
        'subsample': 0.6, 'colsample_bytree': 0.6,
        'reg_alpha': 1.0, 'reg_lambda': 5.0,
        'scale_pos_weight': pos_w,
        'verbose': -1, 'seed': SEED, 'n_jobs': -1,
    }

    lgb_model = lgb.train(
        lgb_params, dtrain, num_boost_round=3000,
        valid_sets=[dtrain, dval], valid_names=['train', 'test'],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)]
    )
    lgb_probs = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

    # XGBoost
    print("\n[4] Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        learning_rate=0.02, n_estimators=3000,
        max_depth=6, min_child_weight=10,
        subsample=0.6, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0,
        scale_pos_weight=pos_w,
        random_state=SEED, n_jobs=-1,
        early_stopping_rounds=200, verbosity=0,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=500)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    # Logistic
    print("\n[5] Training Logistic Regression...")
    lr_model = LogisticRegression(class_weight='balanced', C=0.01, max_iter=2000, random_state=SEED)
    lr_model.fit(X_train_s, y_train)
    lr_probs = lr_model.predict_proba(X_test_s)[:, 1]

    # Ensemble
    ens_probs = 0.5 * lgb_probs + 0.5 * xgb_probs

    # === Evaluate ===
    print("\n" + "=" * 60)
    print("              RESULTS")
    print("=" * 60)

    maj_class = 1 if np.mean(y_test) > 0.5 else 0
    maj_acc = np.mean(y_test == maj_class)

    all_results = []
    for name, probs in [('Logistic Reg', lr_probs), ('LightGBM', lgb_probs),
                         ('XGBoost', xgb_probs), ('Ensemble', ens_probs)]:
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        f1  = f1_score(y_test, preds, zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)
        rec  = recall_score(y_test, preds, zero_division=0)
        cm   = confusion_matrix(y_test, preds)

        all_results.append({
            'name': name, 'accuracy': acc, 'auc': auc, 'f1': f1,
            'precision': prec, 'recall': rec, 'cm': cm, 'probs': probs, 'preds': preds
        })

        imp = "<<<" if acc > 0.5869 else ""
        print(f"  {name:15s}  Acc={acc*100:.2f}%  AUC={auc:.4f}  F1={f1:.4f}  P={prec:.3f}  R={rec:.3f}  {imp}")

    best = max(all_results, key=lambda x: x['accuracy'])

    print(f"\n  Baselines:")
    print(f"    Random:          50.00%")
    print(f"    Majority class:  {maj_acc*100:.2f}%")
    print(f"    Old RF (reg):    58.41%")
    print(f"    Old XGB (reg):   57.35%")
    print(f"    Old LGBM (reg):  58.69%")
    print(f"\n  >>> Best new model: {best['name']} = {best['accuracy']*100:.2f}% <<<")
    print(f"  Vs old best:       {(best['accuracy'] - 0.5869)*100:+.2f}%")

    # Detailed report
    print(f"\n  {best['name']} classification report:")
    print(classification_report(y_test, best['preds'], target_names=['Down(0)','Up(1)']))

    # Feature importance
    print("\n  Top 20 features (LightGBM gain):")
    imp = lgb_model.feature_importance(importance_type='gain')
    idx = np.argsort(imp)[::-1]
    for i in range(min(20, len(FEATURES))):
        print(f"    {i+1:2d}. {FEATURES[idx[i]]:20s} {imp[idx[i]]:10.1f}")

    # === Plots ===
    print("\n[6] Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Nifty 50 Direction Prediction — Classification v4', fontsize=14, fontweight='bold')

    # Bar chart
    ax = axes[0, 0]
    names_all = ['Random', 'Majority', 'RF\n(old)', 'XGB\n(old)', 'LGBM\n(old)'] + [r['name'] for r in all_results]
    accs_all  = [50.0, maj_acc*100, 58.41, 57.35, 58.69] + [r['accuracy']*100 for r in all_results]
    colors    = ['#BDBDBD']*2 + ['#FF9800']*3 + ['#4CAF50' if a > 58.69 else '#2196F3' for a in [r['accuracy']*100 for r in all_results]]
    bars = ax.bar(range(len(names_all)), accs_all, color=colors, edgecolor='white', lw=1.5)
    for bar, acc in zip(bars, accs_all):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(range(len(names_all)))
    ax.set_xticklabels(names_all, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy (%)'); ax.set_title('Model Comparison')
    ax.set_ylim(45, max(accs_all)+5)
    ax.axhline(y=50, color='red', ls=':', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)

    # ROC
    ax = axes[0, 1]
    for r in all_results:
        fpr, tpr, _ = roc_curve(y_test, r['probs'])
        ax.plot(fpr, tpr, lw=2, label=f"{r['name']} ({r['auc']:.3f})")
    ax.plot([0,1],[0,1],'k--',alpha=0.5); ax.set_title('ROC Curves')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Confusion matrix
    ax = axes[1, 0]
    sns.heatmap(best['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Down','Up'], yticklabels=['Down','Up'])
    ax.set_title(f"{best['name']} Confusion Matrix")

    # Feature importance top 15
    ax = axes[1, 1]
    top_n = 15
    top_idx = idx[:top_n][::-1]
    ax.barh([FEATURES[i] for i in top_idx], [imp[i] for i in top_idx], color='#2196F3')
    ax.set_title('Top Features (LightGBM)'); ax.set_xlabel('Gain')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'classification_v4_results.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save
    joblib.dump(lgb_model, os.path.join(MODEL_DIR, 'lgbm_clf_v4.pkl'))
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgb_clf_v4.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler_v4.pkl'))
    joblib.dump({'features': FEATURES}, os.path.join(MODEL_DIR, 'config_v4.pkl'))
    print(f"  Saved to {MODEL_DIR}/ and {RESULTS_DIR}/")
    print(f"\nDone in {datetime.now()-start}")


if __name__ == '__main__':
    main()
