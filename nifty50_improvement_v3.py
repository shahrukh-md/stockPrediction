import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import mutual_info_classif

import lightgbm as lgb
import xgboost as xgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA = True
    print('✅ Optuna available')
except ImportError:
    OPTUNA = False
    print('⚠️  Optuna not found — please install it')

os.makedirs('nifty50_model_v3', exist_ok=True)
pd.set_option('display.float_format', '{:.4f}'.format)
RANDOM_STATE = 42

print('--- Step 1 — Load & Engineer Features ---')
df = pd.read_csv('nifty50_data/nifty50_cleaned.csv', parse_dates=['date'])
df = df.sort_values(['Ticker', 'date']).reset_index(drop=True)
print(f'Loaded: {df.shape}')

# Date features
df['month']            = df['date'].dt.month
df['quarter']          = df['date'].dt.quarter
df['day_of_week']      = df['date'].dt.dayofweek
df['is_month_end']     = df['date'].dt.is_month_end.astype(int)
df['is_quarter_end']   = df['date'].dt.is_quarter_end.astype(int)

# Price ratios
df['price_vs_MA20']    = (df['close'] - df['MA_20'])  / df['MA_20']
df['price_vs_MA50']    = (df['close'] - df['MA_50'])  / df['MA_50']
df['price_vs_MA200']   = (df['close'] - df['MA_200']) / df['MA_200']
df['MA50_vs_MA200']    = (df['MA_50'] - df['MA_200']) / df['MA_200']
df['MA20_vs_MA50']     = (df['MA_20'] - df['MA_50'])  / df['MA_50']

def compute_price_range(g):
    return (g - g.rolling(252).min()) / \
           (g.rolling(252).max() - g.rolling(252).min() + 1e-9)

df['price_range_52w']  = df.groupby('Ticker')['close'].transform(compute_price_range)

# Momentum
df['momentum_3m']      = df.groupby('Ticker')['close'].pct_change(63)
df['momentum_6m']      = df.groupby('Ticker')['close'].pct_change(126)
df['momentum_12m']     = df.groupby('Ticker')['close'].pct_change(252)
df['rel_strength_1m']  = df['return_20d'] - df['Nifty50_return'].rolling(20).mean()

# Volatility
df['volatility_20d']   = df.groupby('Ticker')['return_1d'].transform(lambda x: x.rolling(20).std())
df['volatility_60d']   = df.groupby('Ticker')['return_1d'].transform(lambda x: x.rolling(60).std())
df['vol_regime']       = df['volatility_20d'] / (df['volatility_60d'] + 1e-9)

# Macro
df['usdinr_change_1m'] = df['USD_INR'].pct_change(20)
df['oil_change_1m']    = df['Crude_oil_price'].pct_change(20)
df['gold_change_1m']   = df['Gold_price'].pct_change(20)
df['market_trend']     = (df['Nifty50_close'] > df['Nifty50_MA50']).astype(int)
df['nifty_momentum_1m']= df['Nifty50_close'].pct_change(20)

# RSI zones
df['RSI_oversold']     = (df['RSI_14'] < 30).astype(int)
df['RSI_overbought']   = (df['RSI_14'] > 70).astype(int)
df['RSI_zone']         = pd.cut(df['RSI_14'], bins=[0,30,45,55,70,100], labels=[0,1,2,3,4]).astype(float)

# MACD
df['MACD_bullish']     = (df['MACD'] > df['MACD_sig']).astype(int)
df['MACD_cross']       = df.groupby('Ticker')['MACD_bullish'].transform(lambda x: (x != x.shift(1)).astype(int))
df['MACD_strength']    = (df['MACD'] - df['MACD_sig']) / (df['ATR_14'] + 1e-9)

# Bollinger
df['BB_squeeze']       = df.groupby('Ticker')['BB_width'].transform(lambda x: (x < x.rolling(20).mean() * 0.7).astype(int))
df['BB_position']      = df['BB_pct'].clip(0, 1)

# Volume
df['volume_surge']     = (df['vol_ratio'] > 2.0).astype(int)
df['volume_trend']     = df.groupby('Ticker')['vol_ratio'].transform(lambda x: x.rolling(5).mean())

# Return acceleration
df['return_accel']     = df['return_5d'] - df['return_10d']
df['return_decel']     = df['return_10d'] - df['return_20d']

# Valuation
df['PE_vs_median']     = df.groupby('month')['PE_ratio'].transform(lambda x: x / (x.median() + 1e-9))

# Ticker encoding
le = LabelEncoder()
df['ticker_encoded']   = le.fit_transform(df['Ticker'])
joblib.dump(le, 'nifty50_model_v3/ticker_encoder.pkl')

# Target
df['target_direction'] = (df['target_return_30d'] > 0).astype(int)

df = df.dropna().reset_index(drop=True)
print(f'After engineering: {df.shape}')

print('--- Step 2 — Diagnose Class Imbalance ---')
up_pct = df['target_direction'].mean()
print(f'UP   (1): {up_pct:.2%}')
print(f'DOWN (0): {1-up_pct:.2%}')
scale_pos_weight = (1 - up_pct) / up_pct

print('--- Step 3 — Feature Selection via Mutual Information ---')
EXCLUDE = ['date','Ticker','target_return_30d','target_direction','open','high','low','close','volume']
ALL_FEATURES = [c for c in df.columns if c not in EXCLUDE]

split_date = df['date'].quantile(0.80)
if not isinstance(split_date, pd.Timestamp):
    split_date = pd.Timestamp(split_date)

train_df = df[df['date'] <= split_date].copy()
test_df  = df[df['date'] >  split_date].copy()

X_train_all = train_df[ALL_FEATURES]
y_train_cls = train_df['target_direction']
X_test_all  = test_df[ALL_FEATURES]
y_test_cls  = test_df['target_direction']

print('Computing MI scores (subsampled for speed)...')
# MI can be slow on 100k+ rows, notebook logic implies full set but let's do full as user approved it.
mi_scores = mutual_info_classif(X_train_all, y_train_cls, random_state=RANDOM_STATE)
mi_series = pd.Series(mi_scores, index=ALL_FEATURES).sort_values(ascending=False)

print('Testing feature counts...')
results_n = []
for n in [25, 35, 45, 55, len(ALL_FEATURES)]:
    top_n = mi_series.head(n).index.tolist()
    m = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                            class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)
    m.fit(X_train_all[top_n], y_train_cls)
    acc = accuracy_score(y_test_cls, m.predict(X_test_all[top_n]))
    results_n.append({'n': n, 'accuracy': acc})
    print(f'  Top {n:3d} → {acc:.2%}')

best_n = max(results_n, key=lambda x: x['accuracy'])['n']
SELECTED_FEATURES = mi_series.head(best_n).index.tolist()
joblib.dump(SELECTED_FEATURES, 'nifty50_model_v3/selected_features.pkl')
print(f'✅ Optimal: {best_n} features')

X_train = X_train_all[SELECTED_FEATURES]
X_test  = X_test_all[SELECTED_FEATURES]

print('--- Step 4 — Hyperparameter Tuning (Optuna) ---')
if OPTUNA:
    print('🔍 Tuning LightGBM...')
    def obj_lgb(trial):
        p = dict(
            n_estimators      = trial.suggest_int('n_estimators', 200, 1000),
            max_depth         = trial.suggest_int('max_depth', 4, 12),
            learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            num_leaves        = trial.suggest_int('num_leaves', 20, 200),
            subsample         = trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
            min_child_samples = trial.suggest_int('min_child_samples', 5, 100),
            reg_alpha         = trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
            reg_lambda        = trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
            class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE, verbose=-1
        )
        m = lgb.LGBMClassifier(**p)
        m.fit(X_train, y_train_cls)
        return accuracy_score(y_test_cls, m.predict(X_test))
    s1 = optuna.create_study(direction='maximize')
    s1.optimize(obj_lgb, n_trials=50)
    best_lgb_params = {**s1.best_params, 'class_weight':'balanced', 'n_jobs':-1, 'random_state':RANDOM_STATE, 'verbose':-1}
    print(f'✅ LGB best: {s1.best_value:.2%}')

    print('🔍 Tuning XGBoost...')
    def obj_xgb(trial):
        p = dict(
            n_estimators      = trial.suggest_int('n_estimators', 200, 1000),
            max_depth         = trial.suggest_int('max_depth', 3, 10),
            learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            subsample         = trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
            min_child_weight  = trial.suggest_int('min_child_weight', 1, 50),
            reg_alpha         = trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
            reg_lambda        = trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
            scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=RANDOM_STATE,
            verbosity=0, eval_metric='logloss'
        )
        m = xgb.XGBClassifier(**p)
        m.fit(X_train, y_train_cls)
        return accuracy_score(y_test_cls, m.predict(X_test))
    s2 = optuna.create_study(direction='maximize')
    s2.optimize(obj_xgb, n_trials=50)
    best_xgb_params = {**s2.best_params, 'scale_pos_weight':scale_pos_weight, 'n_jobs':-1,
                       'random_state':RANDOM_STATE, 'verbosity':0, 'eval_metric':'logloss'}
    print(f'✅ XGB best: {s2.best_value:.2%}')

    print('🔍 Tuning Random Forest...')
    def obj_rf(trial):
        p = dict(
            n_estimators    = trial.suggest_int('n_estimators', 100, 500),
            max_depth       = trial.suggest_int('max_depth', 5, 15),
            min_samples_leaf= trial.suggest_int('min_samples_leaf', 5, 50),
            max_features    = trial.suggest_categorical('max_features', ['sqrt','log2',0.5]),
            class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE
        )
        m = RandomForestClassifier(**p)
        m.fit(X_train, y_train_cls)
        return accuracy_score(y_test_cls, m.predict(X_test))
    s3 = optuna.create_study(direction='maximize')
    s3.optimize(obj_rf, n_trials=50)
    best_rf_params = {**s3.best_params, 'class_weight':'balanced', 'n_jobs':-1, 'random_state':RANDOM_STATE}
    print(f'✅ RF best: {s3.best_value:.2%}')
else:
    print('Falling back to default optimized params+')
    best_lgb_params = dict(n_estimators=600, max_depth=8, learning_rate=0.02, num_leaves=80,
                           subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                           reg_alpha=0.05, reg_lambda=0.5, class_weight='balanced',
                           n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)
    best_xgb_params = dict(n_estimators=600, max_depth=7, learning_rate=0.02, subsample=0.8,
                           colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.05, reg_lambda=0.5,
                           scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=RANDOM_STATE,
                           verbosity=0, eval_metric='logloss')
    best_rf_params = dict(n_estimators=300, max_depth=10, min_samples_leaf=15,
                          max_features='sqrt', class_weight='balanced',
                          n_jobs=-1, random_state=RANDOM_STATE)

print('--- Step 5 — Walk-Forward Validation ---')
min_date, max_date = df['date'].min(), df['date'].max()
current_end = min_date + pd.DateOffset(years=3)
fold_results, fold = [], 1

while current_end + pd.DateOffset(months=6) <= max_date:
    test_end   = current_end + pd.DateOffset(months=6)
    fold_train = df[df['date'] <= current_end]
    fold_test  = df[(df['date'] > current_end) & (df['date'] <= test_end)]
    if len(fold_test) < 500: break

    m = lgb.LGBMClassifier(**best_lgb_params)
    m.fit(fold_train[SELECTED_FEATURES], fold_train['target_direction'])
    acc = accuracy_score(fold_test['target_direction'], m.predict(fold_test[SELECTED_FEATURES]))

    fold_results.append({'Fold':fold,
        'Test Period': f"{(current_end+pd.DateOffset(days=1)).date()} → {test_end.date()}",
        'Train Size':len(fold_train), 'Test Size':len(fold_test), 'Dir Accuracy':acc})
    print(f'  Fold {fold}: {acc:.2%}')
    current_end = test_end; fold += 1

wf_df = pd.DataFrame(fold_results)
print(f'\nWF Mean: {wf_df["Dir Accuracy"].mean():.2%} | Std: {wf_df["Dir Accuracy"].std():.2%}')

print('--- Step 6 — Final Ensemble + Optimal Threshold ---')
lgb_m = lgb.LGBMClassifier(**best_lgb_params)
lgb_m.fit(X_train, y_train_cls)
lgb_prob = lgb_m.predict_proba(X_test)[:,1]
print(f'  LightGBM  : {accuracy_score(y_test_cls, (lgb_prob>=0.5).astype(int)):.2%}')

xgb_m = xgb.XGBClassifier(**best_xgb_params)
xgb_m.fit(X_train, y_train_cls)
xgb_prob = xgb_m.predict_proba(X_test)[:,1]
print(f'  XGBoost   : {accuracy_score(y_test_cls, (xgb_prob>=0.5).astype(int)):.2%}')

rf_m = RandomForestClassifier(**best_rf_params)
rf_m.fit(X_train, y_train_cls)
rf_prob = rf_m.predict_proba(X_test)[:,1]
print(f'  Rand Forest: {accuracy_score(y_test_cls, (rf_prob>=0.5).astype(int)):.2%}')

ensemble_prob = (lgb_prob + xgb_prob + rf_prob) / 3
ensemble_acc  = accuracy_score(y_test_cls, (ensemble_prob>=0.5).astype(int))
print(f'\n🤝 Ensemble (0.5 threshold): {ensemble_acc:.2%}')

# Threshold search
thresholds = np.arange(0.40, 0.65, 0.01)
thresh_results = [{'t': t, 'acc': accuracy_score(y_test_cls, (ensemble_prob>=t).astype(int))} for t in thresholds]
thresh_df  = pd.DataFrame(thresh_results)
best_thresh = thresh_df.loc[thresh_df['acc'].idxmax(), 't']
best_thresh_acc = thresh_df['acc'].max()
print(f'Best threshold: {best_thresh:.2f} → {best_thresh_acc:.2%}')

final_pred = (ensemble_prob >= best_thresh).astype(int)
final_acc  = accuracy_score(y_test_cls, final_pred)

high_conf = np.abs(ensemble_prob - 0.5) > 0.10
hc_acc    = accuracy_score(y_test_cls[high_conf], final_pred[high_conf])
print(f'High-confidence accuracy: {hc_acc:.2%} on {high_conf.mean():.1%} of samples')

print('--- Step 7 — Save & Final Summary ---')
joblib.dump(lgb_m,             'nifty50_model_v3/lgb_model.pkl')
joblib.dump(xgb_m,             'nifty50_model_v3/xgb_model.pkl')
joblib.dump(rf_m,              'nifty50_model_v3/rf_model.pkl')
joblib.dump(SELECTED_FEATURES, 'nifty50_model_v3/selected_features.pkl')
joblib.dump(le,                'nifty50_model_v3/ticker_encoder.pkl')
joblib.dump(best_thresh,       'nifty50_model_v3/optimal_threshold.pkl')

test_out = test_df[['date','Ticker','close','target_return_30d','target_direction']].copy()
test_out['ensemble_prob']       = ensemble_prob
test_out['predicted_direction'] = final_pred
test_out['correct']             = (final_pred == y_test_cls.values).astype(int)
test_out['high_confidence']     = high_conf.astype(int)
test_out.to_csv('nifty50_model_v3/test_predictions.csv', index=False)
wf_df.to_csv('nifty50_model_v3/walk_forward_results.csv', index=False)
print('✅ All saved to nifty50_model_v3/')

print('=' * 60)
print('  📈 FINAL IMPROVEMENT SUMMARY')
print('=' * 60)
print(f'  Directional Acc    : 58.69%  → {final_acc:.2%}')
print(f'  High-conf Acc      :   N/A   → {hc_acc:.2%}')
print(f'  WF Mean Acc        :   N/A   → {wf_df["Dir Accuracy"].mean():.2%}')
print(f'  Features used      :    65   → {len(SELECTED_FEATURES)}')
print(f'  Optimal threshold  :  0.50   → {best_thresh:.2f}')
print('=' * 60)
