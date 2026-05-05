"""
Audit script for nifty50_cleaned.csv
Run: python audit_cleaned.py
"""

import pandas as pd
import numpy as np

FILE = "/Users/shahrukh/Desktop/stock/nifty50_data/nifty50_cleaned.csv"

print("Loading...")
df = pd.read_csv(FILE, parse_dates=["date"], low_memory=False)
print(f"Shape: {df.shape}")
print(f"Tickers: {df['Ticker'].nunique()}")
print(f"Date range: {df['date'].min()} → {df['date'].max()}")

print("\n=== NULL COUNTS PER COLUMN ===")
null_counts = df.isnull().sum()
null_pct    = (null_counts / len(df) * 100).round(2)
audit = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
print(audit.to_string())

print("\n=== COLUMNS WITH ZERO NULLS ===")
print(audit[audit.null_count == 0].index.tolist())

print("\n=== COLUMNS WITH >10% NULLS ===")
print(audit[audit.null_pct > 10].to_string())

print("\n=== ROW COUNT PER TICKER ===")
print(df.groupby("Ticker").size().sort_values().to_string())

print("\n=== ALL COLUMNS ===")
for i, col in enumerate(df.columns):
    print(f"  {i+1:3}. {col}")