import pandas as pd
import numpy as np
import os

INPUT_FILE  = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_v4_features.csv"
OUTPUT_FILE = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_with_targets.csv"

def main():
    print(f"Loading {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("Input file not found. Falling back to cleaned data...")
        input_path = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_v4_clean.csv"
    else:
        input_path = INPUT_FILE
    
    df = pd.read_csv(input_path, parse_dates=["date"], low_memory=False)
    df = df.sort_values(["Ticker", "date"]).reset_index(drop=True)
    
    print("Constructing targets...")
    results = []
    
    for ticker, group in df.groupby("Ticker"):
        g = group.copy()
        
        # 1. Forward Return 20d (1 month)
        # Shift close price backwards by 20 days to get the future price
        # (future_price / current_price) - 1
        g["forward_return_20d"] = g["close"].shift(-20) / g["close"] - 1
        
        # 2. Volatility Adjusted Return
        # Using the standard deviation of daily returns over the last 20 days
        daily_returns = g["close"].pct_change()
        real_vol_20d = daily_returns.rolling(20, min_periods=15).std() * np.sqrt(252)
        
        # Avoid division by zero
        real_vol_20d = real_vol_20d.replace(0, np.nan)
        g["vol_adj_return_20d"] = g["forward_return_20d"] / real_vol_20d
        
        # 3. Cross-sectional percentile rank will be done after all tickers are combined
        
        results.append(g)
        
    df = pd.concat(results, ignore_index=True)
    
    # 4. Cross-Sectional Ranking and Normalization
    print("Applying cross-sectional normalizations...")
    
    # Group by date for cross-sectional operations
    def cs_features(g):
        # Percentile rank of the simple forward return
        g["target_pctRank_20d"] = g["forward_return_20d"].rank(pct=True)
        return g
        
    df = df.groupby("date", group_keys=False).apply(cs_features)
    
    # Drop rows where target is null (the last 20 days of data for each stock)
    target_nulls = df["forward_return_20d"].isnull().sum()
    print(f"Dropping {target_nulls} rows due to NaN forward returns (end of dataset).")
    df = df.dropna(subset=["forward_return_20d", "vol_adj_return_20d"])
    
    print(f"Final shape: {df.shape}")
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Target construction complete.")

if __name__ == "__main__":
    main()
