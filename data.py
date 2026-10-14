
from pathlib import Path

import pandas as pd

# Folder containing the project's original CSV files.
DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"


def load_companies() -> pd.DataFrame:
    """Load the cleaned company dataset."""
    path = DATA_DIR / "companies_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Companies file not found: {path}")
    return pd.read_csv(path)


def load_relationships() -> pd.DataFrame:
    """Load company relationships."""
    path = DATA_DIR / "relationships_final.csv"
    if not path.exists():
        raise FileNotFoundError(f"Relationships file not found: {path}")
    return pd.read_csv(path)


def load_financials() -> pd.DataFrame:
    """Load financial data."""
    path = DATA_DIR / "financials.csv"
    if not path.exists():
        raise FileNotFoundError(f"Financials file not found: {path}")
    return pd.read_csv(path)


def load_data_dictionary() -> pd.DataFrame:
    """Load the data dictionary."""
    path = DATA_DIR / "data_dictionary_synced.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data dictionary file not found: {path}")
    return pd.read_csv(path)


def load_price_history() -> pd.DataFrame:
    """Load historical adjusted closing prices for all available tickers."""
    tickers = [
        "AMD", "AMZN", "AVGO", "CRWV", "GOOGL",
        "META", "MSFT", "NVDA", "ORCL", "SPY",
    ]

    frames = []

    for ticker in tickers:
        path = DATA_DIR / f"{ticker}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Price file not found: {path}")

        df = pd.read_csv(path)

        required_columns = {"date", "ticker", "adjusted_close"}
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{ticker}.csv is missing columns: {sorted(missing)}"
            )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["adjusted_close"] = pd.to_numeric(
            df["adjusted_close"], errors="coerce"
        )

        df = df.dropna(subset=["date", "adjusted_close"]).copy()
        frames.append(df)

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["ticker", "date"])

    return prices.reset_index(drop=True)


    print("DATA.PY LOADED")
print("PRICE FUNCTION EXISTS:", callable(load_price_history))
