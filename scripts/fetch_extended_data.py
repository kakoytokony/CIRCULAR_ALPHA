"""Fetch extended market, volume, spillover, Dot-Com historical, and FRED credit data.

Run once from the repository root:
    python scripts/fetch_extended_data.py

Outputs (all saved under data/raw/ for reproducible offline execution):
  1. data/raw/volumes_core.csv       — Daily Volume + Adj Close for the 12 core tickers + SPY (2023-01-01..present)
  2. data/raw/spillover_tiers.csv    — Daily Adj Close for Tier 3–4 spillover proxies (QQQ, SOXX, HYG, LQD, XLU, VST, CEG, EMB)
  3. data/raw/dotcom_1998_2003.csv   — Daily Adj Close for ^GSPC (S&P 500) & ^IXIC (Nasdaq Composite) during 1998–2003
  4. data/raw/fred_credit_spreads.csv— Daily US HY OAS (BAMLH0A0HYM2), US IG OAS (BAMLC0A0CM), 10Y Treasury (DGS10)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CORE_TICKERS = [
    "NVDA", "AMD", "AVGO", "MSFT", "AMZN", "GOOGL",
    "META", "ORCL", "CRWV", "APLD", "GLXY", "BX", "SPY",
]

SPILLOVER_TICKERS = [
    "QQQ",   # Nasdaq-100 (Broad Tech)
    "SOXX",  # iShares Semiconductor ETF
    "HYG",   # iShares High Yield Corporate Bond ETF
    "LQD",   # iShares Investment Grade Corporate Bond ETF
    "XLU",   # Utilities Select Sector SPDR
    "VST",   # Vistra Corp (IPP / Power)
    "CEG",   # Constellation Energy (IPP / Power)
    "EMB",   # iShares J.P. Morgan USD Emerging Markets Bond ETF
]

DOTCOM_TICKERS = ["^GSPC", "^IXIC"]

FRED_SERIES = {
    "hy_oas_pct": "BAMLH0A0HYM2",
    "ig_oas_pct": "BAMLC0A0CM",
    "ust_10y_pct": "DGS10",
}


def download_yf_panel(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )
    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    frames = []
    for t in tickers:
        if len(tickers) == 1:
            df_t = raw
        else:
            if t not in raw.columns.get_level_values(0):
                print(f"[WARN] Ticker {t} missing from Yahoo Finance response.")
                continue
            df_t = raw[t]
        price_col = "Adj Close" if "Adj Close" in df_t.columns else "Close"
        cols = [c for c in [price_col, "Volume"] if c in df_t.columns]
        sub = df_t[cols].dropna(subset=[price_col]).reset_index()
        sub = sub.rename(columns={"Date": "date", price_col: "adjusted_close", "Volume": "volume"})
        sub["date"] = pd.to_datetime(sub["date"]).dt.strftime("%Y-%m-%d")
        sub["ticker"] = t
        sub["source"] = f"https://finance.yahoo.com/quote/{t}/history/"
        sub["retrieved_date"] = retrieved
        frames.append(sub)
    if not frames:
        raise RuntimeError(f"No data downloaded for {tickers}")
    return pd.concat(frames, ignore_index=True)


def download_fred_spreads(start: str = "2023-01-01") -> pd.DataFrame:
    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    merged: pd.DataFrame | None = None
    for col_name, series_id in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        df = pd.read_csv(url)
        date_col = df.columns[0]
        val_col = df.columns[1]
        df = df.rename(columns={date_col: "date", val_col: col_name})
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        df = df[df["date"] >= start].copy()
        merged = df if merged is None else pd.merge(merged, df, on="date", how="outer")
    assert merged is not None
    merged = merged.sort_values("date").reset_index(drop=True)
    merged["source"] = "https://fred.stlouisfed.org/"
    merged["retrieved_date"] = retrieved
    return merged


def main() -> None:
    print("1/4 Downloading core tickers (price + volume)...")
    core_df = download_yf_panel(CORE_TICKERS, start="2023-01-01")
    core_df.to_csv(RAW_DIR / "volumes_core.csv", index=False)
    print(f"    Saved {len(core_df):,} rows -> data/raw/volumes_core.csv")

    print("2/4 Downloading Tier 3-4 spillover proxies...")
    spill_df = download_yf_panel(SPILLOVER_TICKERS, start="2023-01-01")
    spill_df.to_csv(RAW_DIR / "spillover_tiers.csv", index=False)
    print(f"    Saved {len(spill_df):,} rows -> data/raw/spillover_tiers.csv")

    print("3/4 Downloading 1998-2003 Dot-Com historical benchmarks (^GSPC, ^IXIC)...")
    dotcom_df = download_yf_panel(DOTCOM_TICKERS, start="1998-01-01", end="2003-12-31")
    dotcom_df.to_csv(RAW_DIR / "dotcom_1998_2003.csv", index=False)
    print(f"    Saved {len(dotcom_df):,} rows -> data/raw/dotcom_1998_2003.csv")

    print("4/4 Downloading FRED daily credit spreads (HY OAS, IG OAS, 10Y UST)...")
    fred_df = download_fred_spreads(start="2023-01-01")
    fred_df.to_csv(RAW_DIR / "fred_credit_spreads.csv", index=False)
    print(f"    Saved {len(fred_df):,} rows -> data/raw/fred_credit_spreads.csv")

    print("Done! All extended datasets are ready in data/raw/.")


if __name__ == "__main__":
    main()
