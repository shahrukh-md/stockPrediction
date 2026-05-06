import pandas as pd
import numpy as np

INPUT_FILE = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_with_targets.csv"
OUTPUT_FILE = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_normalized.csv"

def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)
    
    # Exclude metadata, targets, and macro features (which have no cross-sectional variance)
    exclude_cols = [
        'date', 'Ticker', 'close', 'high', 'low', 'open', 'volume', 
        'target_return_30d', 'target_cs_zscore', # Legacy targets
        'forward_return_20d', 'vol_adj_return_20d', 'target_pctRank_20d' # New targets
    ]
    
    macro_cols = [
        'USD_INR', 'Nifty50_close', 'Nifty50_return', 'Nifty50_MA50', 
        'India_10yr_yield', 'Gold_price', 'Crude_oil_price'
    ]
    
    exclude_cols.extend(macro_cols)
    
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"Applying cross-sectional z-score normalization on {len(feature_cols)} features...")
    
    # Group by date to perform cross-sectional z-scoring
    def cs_zscore(group):
        for col in feature_cols:
            std = group[col].std()
            if std > 1e-8 and not pd.isna(std):
                group[col] = (group[col] - group[col].mean()) / std
            else:
                # If standard deviation is 0 or NaN (e.g. constant value across all stocks), set z-score to 0
                group[col] = 0.0
        return group
        
    df = df.groupby("date", group_keys=False).apply(cs_zscore)
    
    print(f"Saving normalized dataset to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Cross-sectional normalization complete.")

if __name__ == "__main__":
    main()
