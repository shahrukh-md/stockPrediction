"""
Patch: Fix cross-sectional zscore in nifty50_v4_features.csv
Run: python fix_cs_zscore.py
"""

import pandas as pd
import numpy as np

FILE = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_v4_features.csv"

print("Loading...")
df = pd.read_csv(FILE, parse_dates=["date"], low_memory=False)
df = df.sort_values(["Ticker", "date"]).reset_index(drop=True)
print(f"  Shape: {df.shape}")

# Drop old incorrect zscore
df.drop(columns=["target_cs_zscore"], inplace=True)

# Assign year-month
df["year_month"] = df["date"].dt.to_period("M")

# Month-end snapshot: last row per stock per month
monthly_snapshot = (
    df.groupby(["year_month", "Ticker"], as_index=False)
    .last()
)[["year_month", "Ticker", "target_return_30d"]]

# Verify coverage
coverage = monthly_snapshot.groupby("year_month")["Ticker"].count()
print(f"\n  Min stocks per month : {coverage.min()}")
print(f"  Max stocks per month : {coverage.max()}")
print(f"  Mean stocks per month: {coverage.mean():.1f}")

# Compute cross-sectional zscore per month across stocks
def cs_zscore(group):
    mean = group["target_return_30d"].mean()
    std  = group["target_return_30d"].std()
    if std == 0 or pd.isna(std):
        group["target_cs_zscore"] = 0.0
    else:
        group["target_cs_zscore"] = (group["target_return_30d"] - mean) / std
    return group

monthly_snapshot = monthly_snapshot.groupby(
    "year_month", group_keys=False
).apply(cs_zscore)

# Merge zscore back to daily df
zscore_map = monthly_snapshot[["year_month", "Ticker", "target_cs_zscore"]]
df = df.merge(zscore_map, on=["year_month", "Ticker"], how="left")
df.drop(columns=["year_month"], inplace=True)

# Validate
print(f"\n=== ZSCORE VALIDATION ===")
print(f"  Nulls : {df['target_cs_zscore'].isnull().sum()}")
print(f"  Mean  : {df['target_cs_zscore'].mean():.4f}  (should be ~0)")
print(f"  Std   : {df['target_cs_zscore'].std():.4f}   (should be ~1)")
print(f"  Min   : {df['target_cs_zscore'].min():.4f}")
print(f"  Max   : {df['target_cs_zscore'].max():.4f}")

total_nulls = df.isnull().sum().sum()
print(f"\n  Total NaNs : {total_nulls}")
assert total_nulls == 0, "NaNs present — check merge!"
print(f"  Final shape: {df.shape}")

# Save
print(f"\nSaving...")
df.to_csv(FILE, index=False)
print("Done. nifty50_v4_features.csv updated with correct cross-sectional zscore.")