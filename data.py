"""Local CSV data layer for Circular Alpha. No calculations and no network calls."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def locate_file(filename: str) -> Path:
    """Find saved project CSVs in root or data/raw; never downloads data."""
    candidates = [BASE_DIR / filename, BASE_DIR / "data" / "raw" / filename]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(BASE_DIR.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Required file '{filename}' was not found in {BASE_DIR}.")


def read_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(locate_file(filename))


def load_companies() -> pd.DataFrame:
    return read_csv("companies_clean.csv")


def load_relationships() -> pd.DataFrame:
    return read_csv("relationships_final.csv")


def load_financials() -> pd.DataFrame:
    return read_csv("financials.csv")


def load_dictionary() -> pd.DataFrame:
    try:
        return read_csv("data_dictionary_synced.csv")
    except FileNotFoundError:
        return pd.DataFrame()


def load_all_prices() -> pd.DataFrame:
    """Load saved daily adjusted-price CSV snapshots; supports offline startup."""
    excluded = {
        "companies_clean.csv", "relationships_final.csv", "financials.csv",
        "data_dictionary_synced.csv",
        # v3 extended datasets (scripts/fetch_extended_data.py) are read only by the
        # v3 modules; excluding them keeps the legacy tabs on the exact same price panel.
        "volumes_core.csv", "spillover_tiers.csv", "dotcom_1998_2003.csv",
        "fred_credit_spreads.csv",
    }
    frames = []
    for path in sorted(BASE_DIR.rglob("*.csv")):
        if path.name.lower() in excluded:
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        if not {"date", "ticker", "adjusted_close"}.issubset(frame.columns):
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["ticker"] = frame["ticker"].astype("string").str.strip()
        frame["adjusted_close"] = pd.to_numeric(frame["adjusted_close"], errors="coerce")
        frame = frame.dropna(subset=["date", "ticker", "adjusted_close"])
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["date", "company_id", "ticker", "adjusted_close", "currency", "source", "retrieved_date"])
    prices = pd.concat(frames, ignore_index=True, sort=False)
    prices = prices.drop_duplicates(subset=["date", "ticker"], keep="last")
    return prices.sort_values(["date", "ticker"]).reset_index(drop=True)
