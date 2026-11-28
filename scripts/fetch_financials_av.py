"""Refresh data/raw/financials.csv from Alpha Vantage (yfinance fallback).

Usage (one-off, key only in the environment — never commit it):
    ALPHAVANTAGE_API_KEY=... python scripts/fetch_financials_av.py

The key is read from the ALPHAVANTAGE_API_KEY environment variable or, if absent,
from a local `.env` file (git-ignored). It is passed to `requests` as a query
parameter and is never printed: request errors are reported by exception type only,
because their messages can contain the full URL.

For every listed ticker in data/raw/companies_clean.csv:
  * latest quarterly BALANCE_SHEET → cash_and_equivalents, total_debt
    (shortLongTermDebtTotal, else shortTermDebt + longTermDebt), net_debt;
  * quarterly INCOME_STATEMENT → TTM EBITDA (sum of the last 4 quarters; if fewer
    than 4 are reported, the latest quarter × 4, flagged in `ebitda_basis`).
If no key is set, or Alpha Vantage rate-limits / returns no data for a ticker, that
ticker falls back to yfinance quarterly statements. Each row records `data_source`
and `fiscal_date_ending`. Private entities keep empty (NaN) numbers.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
RAW = BASE_DIR / "data" / "raw"
OUT = RAW / "financials.csv"
AV_URL = "https://www.alphavantage.co/query"
SLEEP_S = 13.0          # free tier: ≤ 5 requests/minute
COLUMNS = ["company_id", "period_end", "period_type", "total_debt", "cash_and_equivalents", "net_debt",
           "ebitda", "currency", "source_url", "retrieved_date", "data_source", "fiscal_date_ending", "ebitda_basis"]


def _api_key() -> str:
    key = os.environ.get("ALPHAVANTAGE_API_KEY", "").strip()
    env = BASE_DIR / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ALPHAVANTAGE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


class RateLimited(Exception):
    pass


def _num(v) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _av(function: str, symbol: str, key: str) -> dict:
    try:
        r = requests.get(AV_URL, params={"function": function, "symbol": symbol, "apikey": key}, timeout=30)
        data = r.json()
    except Exception as exc:  # never echo the message: it can contain the URL with the key
        raise RuntimeError(f"request failed ({type(exc).__name__})") from None
    if "Note" in data or "Information" in data:
        raise RateLimited(function)
    if "Error Message" in data or not data.get("quarterlyReports"):
        raise RuntimeError(f"no {function} data")
    return data


def from_alpha_vantage(symbol: str, key: str) -> dict:
    bs = _av("BALANCE_SHEET", symbol, key)
    time.sleep(SLEEP_S)
    inc = _av("INCOME_STATEMENT", symbol, key)
    q = bs["quarterlyReports"][0]
    cash = _num(q.get("cashAndCashEquivalentsAtCarryingValue"))
    if np.isnan(cash):
        cash = _num(q.get("cashAndShortTermInvestments"))
    debt = _num(q.get("shortLongTermDebtTotal"))
    if np.isnan(debt):
        parts = [_num(q.get("shortTermDebt")), _num(q.get("longTermDebt"))]
        debt = np.nansum(parts) if not all(np.isnan(parts)) else np.nan
    qs = inc["quarterlyReports"]
    e = [_num(x.get("ebitda")) for x in qs[:4]]
    if len(e) == 4 and not any(np.isnan(e)):
        ebitda, basis = float(sum(e)), "TTM (4 quarters)"
    elif e and not np.isnan(e[0]):
        ebitda, basis = 4.0 * e[0], "latest quarter x4 (annualised)"
    else:
        ebitda, basis = np.nan, "not reported"
    return {"fiscal_date_ending": q["fiscalDateEnding"], "currency": q.get("reportedCurrency") if q.get("reportedCurrency") not in (None, "None", "") else "USD",
            "cash_and_equivalents": cash, "total_debt": debt, "ebitda": ebitda, "ebitda_basis": basis,
            "data_source": "alpha_vantage",
            "source_url": f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}"}


def _row(df: pd.DataFrame, names: list[str], col) -> float:
    for n in names:
        if n in df.index:
            return _num(df.loc[n, col])
    return np.nan


def from_yfinance(symbol: str) -> dict:
    import yfinance as yf
    t = yf.Ticker(symbol)
    bs = t.quarterly_balance_sheet
    inc = t.quarterly_income_stmt
    if bs is None or bs.empty:
        raise RuntimeError("no yfinance balance sheet")
    col = bs.columns[0]
    cash = _row(bs, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"], col)
    debt = _row(bs, ["Total Debt"], col)
    ebitda, basis = np.nan, "not reported"
    if inc is not None and not inc.empty:
        for name in ["EBITDA", "Normalized EBITDA"]:
            if name in inc.index:
                vals = [_num(v) for v in inc.loc[name].iloc[:4]]
                if len(vals) == 4 and not any(np.isnan(vals)):
                    ebitda, basis = float(sum(vals)), "TTM (4 quarters)"
                elif vals and not np.isnan(vals[0]):
                    ebitda, basis = 4.0 * vals[0], "latest quarter x4 (annualised)"
                break
    return {"fiscal_date_ending": str(pd.Timestamp(col).date()), "currency": "USD",
            "cash_and_equivalents": cash, "total_debt": debt, "ebitda": ebitda, "ebitda_basis": basis,
            "data_source": "yfinance", "source_url": f"https://finance.yahoo.com/quote/{symbol}/balance-sheet/"}


RECENT_IPO_LABEL = "Recent IPO (no quarterly history)"


def _no_ticker_provenance(company: pd.Series) -> dict:
    """No-ticker rows are never fetched; recent IPOs (e.g. Anthropic) are labelled as such."""
    if str(company.get("public_exchange", "")).strip() == RECENT_IPO_LABEL:
        return {"data_source": "recent IPO (no quarterly history)", "ebitda_basis": "n/a (no quarterly history)"}
    return {"data_source": "private (no public filings)", "ebitda_basis": "n/a (private)"}


def main() -> int:
    companies = pd.read_csv(RAW / "companies_clean.csv")
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=COLUMNS)
    old = old.set_index(old["company_id"].astype(str))
    key = _api_key()
    av_ok = bool(key)
    print("Alpha Vantage key:", "found (value hidden)" if av_ok else "NOT set -> yfinance fallback")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = []
    for _, c in companies.iterrows():
        cid = str(c["company_id"])
        tick = str(c["public_ticker"]).strip() if pd.notna(c["public_ticker"]) else ""
        if not tick:
            # Private entity: keep it explicitly empty — never fabricate numbers.
            prev = old.loc[cid] if cid in old.index else {}
            rows.append({"company_id": cid, "period_end": prev.get("period_end", np.nan) if len(prev) else np.nan,
                         "period_type": "FY", "currency": "USD",
                         "source_url": prev.get("source_url", np.nan) if len(prev) else np.nan,
                         "retrieved_date": today, **_no_ticker_provenance(c)})
            continue
        rec, err = None, ""
        if av_ok:
            try:
                rec = from_alpha_vantage(tick, key)
            except RateLimited:
                av_ok = False
                err = "Alpha Vantage rate limit reached"
            except RuntimeError as exc:
                err = str(exc)
            time.sleep(SLEEP_S)
        if rec is None:
            try:
                rec = from_yfinance(tick)
            except Exception as exc:
                print(f"  {tick}: FAILED ({err or 'no key'}; yfinance {type(exc).__name__}) — keeping previous row")
                if cid in old.index:
                    prev = old.loc[cid].to_dict(); prev["company_id"] = cid
                    rows.append(prev)
                continue
        net = rec["total_debt"] - rec["cash_and_equivalents"] if not (np.isnan(rec["total_debt"]) or np.isnan(rec["cash_and_equivalents"])) else np.nan
        rows.append({"company_id": cid, "period_end": rec["fiscal_date_ending"], "period_type": "Q (EBITDA TTM)",
                     "total_debt": rec["total_debt"], "cash_and_equivalents": rec["cash_and_equivalents"], "net_debt": net,
                     "ebitda": rec["ebitda"], "currency": rec["currency"], "source_url": rec["source_url"],
                     "retrieved_date": today, "data_source": rec["data_source"],
                     "fiscal_date_ending": rec["fiscal_date_ending"], "ebitda_basis": rec["ebitda_basis"]})
        print(f"  {tick}: {rec['data_source']} {rec['fiscal_date_ending']} debt={rec['total_debt']:.3e} "
              f"cash={rec['cash_and_equivalents']:.3e} ebitda={rec['ebitda']:.3e} ({rec['ebitda_basis']})" + (f" [{err}]" if err else ""))
    out = pd.DataFrame(rows).reindex(columns=COLUMNS)
    for c in ["total_debt", "cash_and_equivalents", "net_debt", "ebitda"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(0).astype("Int64")
    out.to_csv(OUT, index=False)
    print(f"Saved {len(out)} rows -> {OUT.relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
