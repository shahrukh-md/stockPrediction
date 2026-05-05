"""
============================================================
  Nifty 50 - Complete Stock Data Collector
  Features: OHLCV, Technical Indicators, Fundamentals, Macro
============================================================
  Install dependencies:
      pip install yfinance pandas numpy ta pandas-datareader requests fredapi
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
import warnings
import time
import os

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
END_DATE   = datetime.today()
START_DATE = END_DATE - timedelta(days=5 * 365)   # 5 years of data
OUTPUT_DIR = "nifty50_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# ALL 50 NIFTY STOCKS  (Yahoo Finance tickers end with .NS)
# ============================================================
NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS", "WIPRO.NS",
    "LT.NS", "AXISBANK.NS", "KOTAKBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS",
    "ONGC.NS", "TECHM.NS", "HCLTECH.NS", "NESTLEIND.NS", "BAJAJFINSV.NS",
    "TMPV.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "COALINDIA.NS", "DRREDDY.NS", "DIVISLAB.NS", "CIPLA.NS", "EICHERMOT.NS",
    "HEROMOTOCO.NS", "BRITANNIA.NS", "SBILIFE.NS", "HDFCLIFE.NS", "INDUSINDBK.NS",
    "BPCL.NS", "GRASIM.NS", "HINDALCO.NS", "APOLLOHOSP.NS", "BAJAJ-AUTO.NS",
    "M&M.NS", "TATACONSUM.NS", "UPL.NS", "SHREECEM.NS", "ITC.NS"
]

# ============================================================
# STEP 1 — DOWNLOAD PRICE & VOLUME DATA (OHLCV)
# ============================================================
def download_price_data(tickers, start, end):
    print("\n📥 Downloading OHLCV price data for all 50 stocks...")
    all_data = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                print(f"  ⚠️  No data for {ticker}, skipping.")
                continue
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df["Ticker"] = ticker
            df.reset_index(inplace=True)
            all_data.append(df)
            print(f"  ✅ {ticker} — {len(df)} rows")
            time.sleep(0.3)   # be polite to Yahoo servers
        except Exception as e:
            print(f"  ❌ Error fetching {ticker}: {e}")

    combined = pd.concat(all_data, ignore_index=True)
    combined.rename(columns={"Date": "date", "Open": "open", "High": "high",
                              "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
    return combined

# ============================================================
# STEP 2 — COMPUTE TECHNICAL INDICATORS
# ============================================================
def add_technical_indicators(df):
    print("\n⚙️  Computing technical indicators...")
    result = []

    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values("date").copy()
        close = group["close"]
        high  = group["high"]
        low   = group["low"]
        vol   = group["volume"]

        # --- Moving Averages ---
        group["MA_20"]  = SMAIndicator(close, window=20).sma_indicator()
        group["MA_50"]  = SMAIndicator(close, window=50).sma_indicator()
        group["MA_200"] = SMAIndicator(close, window=200).sma_indicator()
        group["EMA_20"] = EMAIndicator(close, window=20).ema_indicator()

        # --- RSI ---
        group["RSI_14"] = RSIIndicator(close, window=14).rsi()

        # --- MACD ---
        macd_obj          = MACD(close)
        group["MACD"]     = macd_obj.macd()
        group["MACD_sig"] = macd_obj.macd_signal()
        group["MACD_diff"]= macd_obj.macd_diff()

        # --- Bollinger Bands ---
        bb = BollingerBands(close, window=20, window_dev=2)
        group["BB_upper"]  = bb.bollinger_hband()
        group["BB_lower"]  = bb.bollinger_lband()
        group["BB_middle"] = bb.bollinger_mavg()
        group["BB_width"]  = bb.bollinger_wband()
        group["BB_pct"]    = bb.bollinger_pband()

        # --- Price Returns ---
        group["return_1d"]  = close.pct_change(1)
        group["return_5d"]  = close.pct_change(5)
        group["return_10d"] = close.pct_change(10)
        group["return_20d"] = close.pct_change(20)

        # --- Volume MA ---
        group["vol_MA_20"] = vol.rolling(20).mean()
        group["vol_ratio"] = vol / group["vol_MA_20"]

        # --- Average True Range (volatility) ---
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        group["ATR_14"] = tr.rolling(14).mean()

        # --- TARGET: 30-day future return (what we want to predict) ---
        group["target_return_30d"] = close.shift(-30) / close - 1
        group["target_price_30d"]  = close.shift(-30)

        result.append(group)

    print("  ✅ Technical indicators computed.")
    return pd.concat(result, ignore_index=True)

# ============================================================
# STEP 3 — FUNDAMENTALS (via yfinance)
# ============================================================
def get_fundamentals(tickers):
    print("\n🏢 Fetching fundamentals for all 50 stocks...")
    records = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            records.append({
                "Ticker"            : ticker,
                "PE_ratio"          : info.get("trailingPE"),
                "forward_PE"        : info.get("forwardPE"),
                "EPS_trailing"      : info.get("trailingEps"),
                "EPS_forward"       : info.get("forwardEps"),
                "revenue_growth"    : info.get("revenueGrowth"),
                "earnings_growth"   : info.get("earningsGrowth"),
                "profit_margin"     : info.get("profitMargins"),
                "operating_margin"  : info.get("operatingMargins"),
                "debt_to_equity"    : info.get("debtToEquity"),
                "current_ratio"     : info.get("currentRatio"),
                "roe"               : info.get("returnOnEquity"),
                "roa"               : info.get("returnOnAssets"),
                "book_value"        : info.get("bookValue"),
                "price_to_book"     : info.get("priceToBook"),
                "dividend_yield"    : info.get("dividendYield"),
                "market_cap"        : info.get("marketCap"),
                "52w_high"          : info.get("fiftyTwoWeekHigh"),
                "52w_low"           : info.get("fiftyTwoWeekLow"),
                "beta"              : info.get("beta"),
            })
            print(f"  ✅ {ticker}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  ❌ Error for {ticker}: {e}")

    df = pd.DataFrame(records)
    print(f"  ✅ Fundamentals fetched for {len(df)} stocks.")
    return df

# ============================================================
# STEP 4 — MACRO DATA (USD/INR, Nifty 50, India Inflation)
# ============================================================
def get_macro_data(start, end):
    print("\n🌍 Fetching macro data...")
    macro_df = pd.DataFrame()

    # --- USD/INR via yfinance ---
    try:
        usdinr = yf.download("INR=X", start=start, end=end, progress=False, auto_adjust=True)
        usdinr.columns = [c[0] if isinstance(c, tuple) else c for c in usdinr.columns]
        macro_df["USD_INR"] = usdinr["Close"]
        print("  ✅ USD/INR exchange rate")
    except Exception as e:
        print(f"  ❌ USD/INR: {e}")

    # --- Nifty 50 Index ---
    try:
        nifty = yf.download("^NSEI", start=start, end=end, progress=False, auto_adjust=True)
        nifty.columns = [c[0] if isinstance(c, tuple) else c for c in nifty.columns]
        macro_df["Nifty50_close"]   = nifty["Close"]
        macro_df["Nifty50_return"]  = nifty["Close"].pct_change()
        macro_df["Nifty50_MA50"]    = nifty["Close"].rolling(50).mean()
        print("  ✅ Nifty 50 index")
    except Exception as e:
        print(f"  ❌ Nifty50: {e}")

    # --- India 10-yr Bond Yield (proxy for interest rates) ---
    # NIFTYGS10YR.NS = NSE Nifty G-Sec 10yr index — confirmed working on Yahoo Finance
    bond_tickers = ["NIFTYGS10YR.NS", "IN10Y.NS", "^INBMK10Y"]
    bond_fetched = False
    for bond_ticker in bond_tickers:
        try:
            bond = yf.download(bond_ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not bond.empty:
                bond.columns = [c[0] if isinstance(c, tuple) else c for c in bond.columns]
                macro_df["India_10yr_yield"] = bond["Close"]
                print(f"  ✅ India 10-yr bond yield (via {bond_ticker})")
                bond_fetched = True
                break
        except Exception:
            continue
    if not bond_fetched:
        # Fallback: fill with RBI repo rate as static proxy (6.5% as of 2024)
        print("  ⚠️  India bond yield unavailable from Yahoo. Using RBI repo rate proxy (6.5%)")
        macro_df["India_10yr_yield"] = 6.5

    # --- Gold price (safe haven indicator) ---
    try:
        gold = yf.download("GC=F", start=start, end=end, progress=False, auto_adjust=True)
        gold.columns = [c[0] if isinstance(c, tuple) else c for c in gold.columns]
        macro_df["Gold_price"] = gold["Close"]
        print("  ✅ Gold price")
    except Exception as e:
        print(f"  ❌ Gold: {e}")

    # --- Crude Oil (important for India) ---
    try:
        oil = yf.download("CL=F", start=start, end=end, progress=False, auto_adjust=True)
        oil.columns = [c[0] if isinstance(c, tuple) else c for c in oil.columns]
        macro_df["Crude_oil_price"] = oil["Close"]
        print("  ✅ Crude oil price")
    except Exception as e:
        print(f"  ❌ Crude oil: {e}")

    macro_df.index.name = "date"
    macro_df.reset_index(inplace=True)

    # flatten MultiIndex columns if present
    if isinstance(macro_df.columns, pd.MultiIndex):
        macro_df.columns = ["_".join(filter(None, c)) if isinstance(c, tuple) else c
                            for c in macro_df.columns]

    macro_df["date"] = pd.to_datetime(macro_df["date"])
    print(f"  ✅ Macro data shape: {macro_df.shape}")
    return macro_df

# ============================================================
# STEP 5 — MERGE EVERYTHING
# ============================================================
def merge_all(price_tech_df, fundamentals_df, macro_df):
    print("\n🔗 Merging all datasets...")

    price_tech_df["date"] = pd.to_datetime(price_tech_df["date"])
    macro_df["date"]      = pd.to_datetime(macro_df["date"])

    # Merge price+technical with macro on date
    merged = pd.merge(price_tech_df, macro_df, on="date", how="left")

    # Merge with fundamentals on Ticker (point-in-time snapshot)
    merged = pd.merge(merged, fundamentals_df, on="Ticker", how="left")

    # Forward-fill macro (weekends/holidays have no macro data)
    macro_cols = [c for c in macro_df.columns if c != "date"]
    merged[macro_cols] = merged[macro_cols].ffill()

    print(f"  ✅ Final dataset shape: {merged.shape}")
    return merged

# ============================================================
# STEP 6 — SAVE
# ============================================================
def save_data(df, fundamentals_df, macro_df):
    print("\n💾 Saving files...")

    # Main combined dataset
    main_path = os.path.join(OUTPUT_DIR, "nifty50_full_dataset.csv")
    df.to_csv(main_path, index=False)
    print(f"  ✅ Full dataset      → {main_path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")

    # Fundamentals snapshot
    fund_path = os.path.join(OUTPUT_DIR, "nifty50_fundamentals.csv")
    fundamentals_df.to_csv(fund_path, index=False)
    print(f"  ✅ Fundamentals      → {fund_path}")

    # Macro data
    macro_path = os.path.join(OUTPUT_DIR, "nifty50_macro.csv")
    macro_df.to_csv(macro_path, index=False)
    print(f"  ✅ Macro data        → {macro_path}")

    # Per-stock CSVs
    per_stock_dir = os.path.join(OUTPUT_DIR, "per_stock")
    os.makedirs(per_stock_dir, exist_ok=True)
    for ticker, group in df.groupby("Ticker"):
        path = os.path.join(per_stock_dir, f"{ticker.replace('.NS','')}.csv")
        group.to_csv(path, index=False)
    print(f"  ✅ Per-stock CSVs    → {per_stock_dir}/")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  🚀 Nifty 50 Data Collector — Starting")
    print(f"  📅 Period: {START_DATE.date()} → {END_DATE.date()}")
    print("=" * 60)

    # 1. Price data
    price_df = download_price_data(NIFTY50_TICKERS, START_DATE, END_DATE)
    price_df.to_csv(os.path.join(OUTPUT_DIR, "raw_ohlcv.csv"), index=False)

    # 2. Technical indicators
    price_tech_df = add_technical_indicators(price_df)

    # 3. Fundamentals
    fundamentals_df = get_fundamentals(NIFTY50_TICKERS)

    # 4. Macro data
    macro_df = get_macro_data(START_DATE, END_DATE)

    # 5. Merge
    final_df = merge_all(price_tech_df, fundamentals_df, macro_df)

    # 6. Save
    save_data(final_df, fundamentals_df, macro_df)

    # Summary
    print("\n" + "=" * 60)
    print("  ✅ ALL DONE!")
    print(f"  📊 Total rows     : {len(final_df):,}")
    print(f"  📋 Total features : {final_df.shape[1]}")
    print(f"  🏢 Stocks covered : {final_df['Ticker'].nunique()}")
    print(f"  📁 Output folder  : ./{OUTPUT_DIR}/")
    print("=" * 60)

    print("\n📋 FEATURE SUMMARY:")
    print(final_df.dtypes.to_string())

    print("\n📈 SAMPLE DATA (first 3 rows):")
    print(final_df.head(3).to_string())

    return final_df

if __name__ == "__main__":
    df = main()