"""Loaders for the v3 extended datasets produced by ``scripts/fetch_extended_data.py``.

Every loader is offline-only (reads ``data/raw/*.csv``) and never raises when a
file is missing: it returns an empty frame plus a human-readable status string so
the Streamlit app can flag the missing live series instead of crashing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data import BASE_DIR

RAW_DIR = BASE_DIR / "data" / "raw"

EXTENDED_FILES = {
    "volumes_core": "volumes_core.csv",
    "spillover_tiers": "spillover_tiers.csv",
    "dotcom": "dotcom_1998_2003.csv",
    "fred": "fred_credit_spreads.csv",
}

CORE_TICKERS = ["NVDA", "AMD", "AVGO", "MSFT", "AMZN", "GOOGL", "META", "ORCL", "CRWV", "APLD", "GLXY", "BX"]
SPILLOVER_TICKERS = ["QQQ", "SOXX", "HYG", "LQD", "XLU", "VST", "CEG", "EMB"]
DOTCOM_TICKERS = ["^GSPC", "^IXIC"]
FRED_COLUMNS = ["hy_oas_pct", "ig_oas_pct", "ust_10y_pct"]


@dataclass(frozen=True)
class LoadStatus:
    name: str
    ok: bool
    rows: int
    first_date: str
    last_date: str
    message: str


def _status(name: str, df: pd.DataFrame, path: Path, err: str | None = None) -> LoadStatus:
    if err is not None or df.empty:
        return LoadStatus(name, False, 0, "", "", err or f"{path.name} is empty")
    d = pd.to_datetime(df["date"])
    return LoadStatus(
        name, True, len(df), str(d.min().date()), str(d.max().date()),
        f"{path.name}: {len(df):,} rows, {d.min().date()} → {d.max().date()}",
    )


def load_long_panel(key: str, raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, LoadStatus]:
    """Load a long (date, ticker, adjusted_close, volume) panel."""
    path = raw_dir / EXTENDED_FILES[key]
    if not path.exists():
        msg = f"MISSING: {path.name} — run `python scripts/fetch_extended_data.py`"
        return pd.DataFrame(columns=["date", "ticker", "adjusted_close", "volume"]), _status(key, pd.DataFrame(), path, msg)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "adjusted_close"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df, _status(key, df, path)


def load_fred(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, LoadStatus]:
    path = raw_dir / EXTENDED_FILES["fred"]
    if not path.exists():
        msg = f"MISSING: {path.name} — run `python scripts/fetch_extended_data.py`"
        return pd.DataFrame(columns=["date", *FRED_COLUMNS]), _status("fred", pd.DataFrame(), path, msg)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in FRED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df, _status("fred", df, path)


def fred_series_bps(fred: pd.DataFrame, column: str = "hy_oas_pct") -> pd.Series:
    """FRED OAS (percent) as a date-indexed series in basis points."""
    if fred.empty or column not in fred.columns:
        return pd.Series(dtype=float)
    return (fred.set_index("date")[column] * 100.0).dropna()


def wide(panel: pd.DataFrame, value: str = "adjusted_close") -> pd.DataFrame:
    """Pivot a long panel to a date x ticker matrix."""
    if panel.empty:
        return pd.DataFrame()
    return panel.pivot_table(index="date", columns="ticker", values=value).sort_index()


def all_statuses(raw_dir: Path = RAW_DIR) -> list[LoadStatus]:
    out = [load_long_panel(k, raw_dir)[1] for k in ("volumes_core", "spillover_tiers", "dotcom")]
    out.append(load_fred(raw_dir)[1])
    return out


def equal_weight_index(prices_wide: pd.DataFrame, tickers: list[str], base: float = 100.0) -> pd.Series:
    """Equal-weight index of available tickers (dynamic availability mask, daily rebalanced).

    Each day's index return is the mean of the simple returns of the tickers that
    have a price on both t-1 and t, so names with shorter listings (CRWV, GLXY)
    join on their second trading day without back-filling.
    """
    cols = [t for t in tickers if t in prices_wide.columns]
    if not cols:
        return pd.Series(dtype=float)
    rets = prices_wide[cols].pct_change(fill_method=None)
    ew = rets.mean(axis=1, skipna=True)
    first = prices_wide[cols].dropna(how="all").index.min()
    ew = ew.loc[first:].fillna(0.0)
    ew.iloc[0] = 0.0
    return base * (1.0 + ew).cumprod()


def equal_weight_volume(dollar_volume_wide: pd.DataFrame, tickers: list[str]) -> pd.Series:
    cols = [t for t in tickers if t in dollar_volume_wide.columns]
    return dollar_volume_wide[cols].sum(axis=1, min_count=1) if cols else pd.Series(dtype=float)


def safe_last(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else float("nan")


__all__ = [
    "CORE_TICKERS", "SPILLOVER_TICKERS", "DOTCOM_TICKERS", "FRED_COLUMNS", "EXTENDED_FILES",
    "LoadStatus", "load_long_panel", "load_fred", "fred_series_bps", "wide", "all_statuses",
    "equal_weight_index", "equal_weight_volume", "safe_last", "RAW_DIR",
]
