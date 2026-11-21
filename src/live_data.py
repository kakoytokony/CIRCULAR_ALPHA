"""Live Yahoo Finance price loader with seamless offline CSV fallback.

Usage:
  - In Streamlit: load_prices_with_live_option(live_mode=True) attempts yfinance
    download for all public tickers + SPY and falls back cleanly to data/raw/*.csv.
  - CLI daily refresh: python -m src.live_data --refresh
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from data import BASE_DIR, load_all_prices, load_companies


def fetch_yfinance_prices(
    tickers: list[str],
    start_date: str = "2023-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily adjusted close prices from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance") from exc

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned an empty dataset.")

    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    frames = []
    if len(tickers) == 1:
        t = tickers[0]
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        sub = raw[[col]].dropna().reset_index()
        sub.columns = ["date", "adjusted_close"]
        sub["ticker"] = t
        sub["currency"] = "USD"
        sub["source"] = f"https://finance.yahoo.com/quote/{t}/history/"
        sub["retrieved_date"] = retrieved
        frames.append(sub)
    else:
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                continue
            df_t = raw[t]
            col = "Adj Close" if "Adj Close" in df_t.columns else "Close"
            sub = df_t[[col]].dropna().reset_index()
            sub.columns = ["date", "adjusted_close"]
            sub["ticker"] = t
            sub["currency"] = "USD"
            sub["source"] = f"https://finance.yahoo.com/quote/{t}/history/"
            sub["retrieved_date"] = retrieved
            frames.append(sub)

    if not frames:
        raise RuntimeError("No valid ticker series parsed from Yahoo Finance response.")

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["adjusted_close"] = pd.to_numeric(out["adjusted_close"], errors="coerce")
    out = out.dropna(subset=["date", "adjusted_close"])
    out = out[out["adjusted_close"] > 0]
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_prices_with_live_option(
    live_mode: bool = False,
    start_date: str = "2023-01-01",
) -> tuple[pd.DataFrame, str]:
    """Load prices either from saved offline CSVs or live Yahoo Finance with fallback."""
    offline = load_all_prices()
    if not live_mode:
        last_dt = str(offline["date"].max().date()) if not offline.empty else "N/A"
        return offline, f"Offline CSV snapshot (reproducible, through {last_dt})"

    companies = load_companies()
    tickers = sorted(
        set(companies["public_ticker"].dropna().astype(str).str.strip().tolist()) | {"SPY"}
    )
    tickers = [t for t in tickers if t and t.lower() != "nan"]
    try:
        live_df = fetch_yfinance_prices(tickers, start_date=start_date)
        last_dt = str(live_df["date"].max().date())
        return live_df, f"Live Yahoo Finance (refreshed {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}, through {last_dt})"
    except Exception as exc:
        last_dt = str(offline["date"].max().date()) if not offline.empty else "N/A"
        return offline, f"Offline fallback (live fetch unavailable: {exc}; snapshot through {last_dt})"


def refresh_local_csvs(start_date: str = "2023-01-01") -> dict:
    """Refresh per-ticker CSV snapshots in data/raw/ for scheduled daily updates."""
    companies = load_companies()
    id_map = dict(zip(companies["public_ticker"].fillna("").astype(str).str.strip(), companies["company_id"]))
    tickers = sorted(set(t for t in id_map if t) | {"SPY"})
    live_df = fetch_yfinance_prices(tickers, start_date=start_date)
    raw_dir = BASE_DIR / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    updated = {}
    for t, grp in live_df.groupby("ticker"):
        out = grp.copy()
        out["company_id"] = id_map.get(t, "BENCHMARK" if t == "SPY" else "")
        out = out[["date", "company_id", "ticker", "adjusted_close", "currency", "source", "retrieved_date"]]
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(raw_dir / f"{t}.csv", index=False)
        updated[t] = len(out)
    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh Circular Alpha Yahoo Finance price snapshots.")
    parser.add_argument("--refresh", action="store_true", help="Download latest prices and overwrite data/raw/*.csv")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    args = parser.parse_args()
    if args.refresh:
        res = refresh_local_csvs(start_date=args.start)
        print(f"Refreshed {len(res)} tickers: {res}")
