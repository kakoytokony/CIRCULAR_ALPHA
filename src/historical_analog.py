"""2000–2002 Dot-Com / Telecom vendor-financing analog and the positioning playbook.

Verified benchmark constants (METHODOLOGY_AND_SPEC.md §3.1) live in
``DOTCOM_BENCHMARKS`` with mandatory sources; ``measured_drawdowns`` recomputes
the index figures from ``data/raw/dotcom_1998_2003.csv`` so the app shows the
data-derived number next to the cited one (tests assert they agree).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.credit_signals import SourcedValue

SEC_2004 = "SEC Press Release 2004-67 (May 17, 2004); SEC Litigation Releases"
IDX = "Nasdaq / S&P Dow Jones Indices daily closing data"


def _sv(key, value, unit, en, ru, source, as_of) -> SourcedValue:
    return SourcedValue(key, float(value), unit, en, ru, source, as_of)


DOTCOM_BENCHMARKS: dict[str, SourcedValue] = {v.key: v for v in [
    _sv("ixic_peak", 5048.62, "index", "Nasdaq Composite peak (2000-03-10)", "Пик Nasdaq Composite (10.03.2000)", IDX, "2000-03-10"),
    _sv("ixic_trough", 1114.11, "index", "Nasdaq Composite trough (2002-10-09)", "Дно Nasdaq Composite (09.10.2002)", IDX, "2002-10-09"),
    _sv("ixic_dd_pct", -77.9, "%", "Nasdaq Composite drawdown", "Просадка Nasdaq Composite", IDX, "2002-10-09"),
    _sv("ndx_peak", 4704.73, "index", "Nasdaq-100 peak (2000-03-27)", "Пик Nasdaq-100 (27.03.2000)", IDX, "2000-03-27"),
    _sv("ndx_trough", 804.64, "index", "Nasdaq-100 trough (2002-10-07)", "Дно Nasdaq-100 (07.10.2002)", IDX, "2002-10-07"),
    _sv("ndx_dd_pct", -82.9, "%", "Nasdaq-100 drawdown", "Просадка Nasdaq-100", IDX, "2002-10-07"),
    _sv("spx_peak", 1527.46, "index", "S&P 500 peak (2000-03-24)", "Пик S&P 500 (24.03.2000)", IDX, "2000-03-24"),
    _sv("spx_trough", 776.76, "index", "S&P 500 trough (2002-10-09)", "Дно S&P 500 (09.10.2002)", IDX, "2002-10-09"),
    _sv("spx_dd_pct", -49.1, "%", "S&P 500 drawdown", "Просадка S&P 500", IDX, "2002-10-09"),
    _sv("spx_recovery_years", 7.2, "years", "S&P 500 recovery (May 30, 2007)", "Восстановление S&P 500 (30.05.2007)", IDX, "2007-05-30"),
    _sv("lucent_customer_financing_bn", 8.1, "$bn", "Lucent customer-financing commitments (Sep 30, 2000)", "Обязательства Lucent по финансированию клиентов (30.09.2000)", SEC_2004, "2000-09-30"),
    _sv("lucent_bad_debt_bn", 2.2, "$bn", "Lucent bad-debt provisions FY2001", "Резервы Lucent по безнадёжным долгам FY2001", SEC_2004, "2001"),
    _sv("lucent_improper_revenue_bn", 1.148, "$bn", "Lucent improperly recognised FY2000 revenue (SEC)", "Неправомерно признанная выручка Lucent FY2000 (SEC)", SEC_2004, "2004-05-17"),
    _sv("lucent_pretax_overstatement_m", 470, "$m", "Lucent pre-tax income overstatement", "Завышение доналоговой прибыли Lucent", SEC_2004, "2004-05-17"),
    _sv("lucent_penalty_m", 25, "$m", "Lucent SEC penalty", "Штраф SEC для Lucent", SEC_2004, "2004-05-17"),
    _sv("clec_bankruptcies", 47, "count", "CLEC bankruptcies 2000–2003", "Банкротства CLEC 2000–2003", "Industry / bankruptcy records (spec §3.1)", "2003"),
]}


def measured_drawdowns(dotcom_panel: pd.DataFrame) -> pd.DataFrame:
    """Peak-to-trough drawdown per ticker computed from the downloaded closes."""
    rows = []
    for t, g in dotcom_panel.groupby("ticker"):
        p = g.set_index("date")["adjusted_close"].sort_index()
        peak_dt = p.idxmax()
        trough_dt = p.loc[peak_dt:].idxmin()
        months = (trough_dt.year - peak_dt.year) * 12 + (trough_dt.month - peak_dt.month)
        rows.append({"ticker": t, "peak_date": peak_dt.date(), "peak": float(p[peak_dt]),
                     "trough_date": trough_dt.date(), "trough": float(p[trough_dt]),
                     "drawdown_pct": 100.0 * (float(p[trough_dt]) / float(p[peak_dt]) - 1.0), "months": months})
    return pd.DataFrame(rows)


def spx_nasdaq_elasticity(measured: pd.DataFrame) -> float:
    m = measured.set_index("ticker")["drawdown_pct"]
    return float(m["^GSPC"] / m["^IXIC"])


def aligned_paths(dotcom_price: pd.Series, ai_price: pd.Series, pre_days: int = 504) -> pd.DataFrame:
    """Both series indexed to 100 at ``pre_days`` before their own running peak.

    Dot-com: aligned on the March 2000 peak. AI core: aligned on its highest close
    to date (no future information used). x = trading days relative to peak.
    """
    out = []
    for label, p in (("Nasdaq Composite 1998–2003", dotcom_price), ("AI Core (equal-weight, 12 names)", ai_price)):
        p = p.dropna()
        k = int(np.argmax(p.to_numpy()))
        start = max(0, k - pre_days)
        seg = p.iloc[start:]
        out.append(pd.DataFrame({"days_from_peak": np.arange(len(seg)) - (k - start),
                                 "index": 100.0 * seg.to_numpy() / seg.iloc[0], "series": label,
                                 "date": seg.index}))
    return pd.concat(out, ignore_index=True)


STRUCTURAL_COMPARISON = [
    # (dimension_en, dimension_ru, 1999–2002, 2025–2026, parallel_or_difference)
    ("Vendor financing", "Вендорное финансирование",
     "Lucent: $8.1B customer-financing commitments (Sep 2000); $2.2B bad-debt provisions FY2001",
     "Broadcom-guaranteed $63B Anthropic SPV; Nvidia $105B Ohio backstop; Nvidia lease backstop of CoreWeave to 2032",
     "Parallel"),
    ("Fragile customers", "Хрупкие клиенты",
     "47 CLECs bankrupt 2000–2003 (WinStar, Covad, NorthPoint)",
     "Neoclouds & loss-making AI labs: CoreWeave 2031s >1,000 bps; equity −23% YTD",
     "Parallel"),
    ("Equity supply wave", "Волна предложения акций",
     "IPO lock-up expiries and insider selling (Ofek & Richardson, 2003)",
     "Anthropic S-1; Meta $25B SPO ($5B chipmaker anchor)",
     "Parallel"),
    ("Revenue quality", "Качество выручки",
     "SEC: Lucent improperly recognised $1.148B FY2000 revenue",
     "Chip revenue partly supported by sellers' own guarantees and equity stakes (WSJ, Nov 1, 2026)",
     "Parallel (no fraud finding today)"),
    ("Core buyers' cash flow", "Денежный поток ключевых покупателей",
     "Telecom carriers financed capex with debt",
     "Hyperscalers fund most capex from operating cash; tracked debt $629B concentrated in ORCL/META/AMZN",
     "Difference (stronger today)"),
    ("Collateral life", "Срок жизни залога",
     "Fiber: multi-decade useful life",
     "GPUs: roughly 3–6 year economic life",
     "Difference (weaker today)"),
    ("Regulatory brake", "Регуляторный тормоз",
     "None comparable",
     "Nov 3, 2026: 42/45 local DC restrictions passed/leading; moratoria in TX, OH, GA, NV, NM",
     "Difference (supply discipline)"),
]


@dataclass
class CostAssumptions:
    """Explicit, user-editable cost assumptions (not market quotes)."""
    transaction_cost_bps: float = 10.0       # same default as the GA2 Rule Desk
    borrow_fee_gc_pct: float = 0.5           # assumption: easy-to-borrow large cap
    borrow_fee_htb_pct: float = 10.0         # assumption: hard-to-borrow neocloud
    put_hedge_cost_pct: float = 0.0          # user enters a quoted premium if used
    horizon_months: int = 6


HTB_TICKERS = {"CRWV", "APLD", "GLXY"}


def position_cost_pct(ticker: str, side: str, c: CostAssumptions) -> float:
    """Round-trip cost over the horizon: 2 × transaction cost + borrow fee × horizon (shorts)."""
    rt = 2.0 * c.transaction_cost_bps / 100.0
    if side != "short":
        return rt
    fee = c.borrow_fee_htb_pct if ticker in HTB_TICKERS else c.borrow_fee_gc_pct
    return rt + fee * c.horizon_months / 12.0


def positioning_matrix(
    prices_wide: pd.DataFrame,
    bond_matrix: pd.DataFrame | None,
    triggers: pd.DataFrame,
    costs: CostAssumptions,
) -> pd.DataFrame:
    """Rules → bucket per ticker, with live metrics from data/raw (spec §Module 3)."""
    trig = dict(zip(triggers["key"], triggers["status"]))
    credit_red = trig.get("hy_hpc") == "RED" or trig.get("crwv") == "RED"
    underwriting_red = trig.get("supply") == "RED"
    debt = {}
    if bond_matrix is not None:
        col = "jpm_printed_bn" if "jpm_printed_bn" in bond_matrix.columns else "total_bn"
        debt = dict(zip(bond_matrix["issuer"], bond_matrix[col]))
    rows = []
    lr = np.log(prices_wide).diff()
    for t in ["CRWV", "APLD", "GLXY", "NVDA", "AVGO", "AMD", "ORCL", "META", "GOOGL", "AMZN", "MSFT", "BX"]:
        if t not in prices_wide.columns:
            continue
        p = prices_wide[t].dropna()
        last = p.iloc[-1]
        dd_52w = 100.0 * (last / p.iloc[-252:].max() - 1.0)
        vol63 = float(lr[t].dropna().iloc[-63:].std() * np.sqrt(252) * 100.0)
        if t in {"CRWV", "APLD", "GLXY"}:
            bucket, side = "Short / Avoid (high fragility)", "short"
            why = "Neocloud/HPC builder; CRWV 2031s >1,000 bps; Texas pipeline exposed to moratorium" if credit_red else "Neocloud, credit trigger not RED"
        elif t in {"NVDA", "AVGO"}:
            active = credit_red and underwriting_red
            bucket = "Tactical underweight / hedge (guarantee exposure)" if active else "Neutral (hedge rule inactive)"
            side = "hedge" if active else "hold"
            why = "SPV/backstop exposure ($63B AVGO SPV; $105B NVDA Ohio); hedge via puts/pairs, not naked shorts"
        elif t == "AMD":
            bucket, side = "Neutral / monitor", "hold"
            why = "Chipmaker; equity anchor in Meta SPO; no disclosed guarantee in pack"
        elif t in {"MSFT", "AMZN"}:
            bucket, side = "Defensive / relative long (fortress balance sheet)", "long"
            why = f"Tracked debt ${debt.get(t, float('nan')):.0f}B; rose >6% after Blue Wave (WSJ, Nov 5)"
        elif t in {"ORCL", "META"}:
            bucket, side = "Avoid as long (debt concentration)", "hold"
            why = f"Largest tracked debt in ecosystem: ${debt.get(t, float('nan')):.0f}B"
        elif t == "GOOGL":
            bucket, side = "Neutral / monitor", "hold"
            why = f"Tracked debt ${debt.get(t, float('nan')):.0f}B; Fluidstack lease guarantees"
        else:
            bucket, side = "Monitor (private-credit markdown risk)", "hold"
            why = "Private credit participant in Anthropic $23B non-bank tranche"
        rows.append({"ticker": t, "bucket": bucket, "side": side, "rationale": why,
                     "tracked_debt_bn": debt.get(t, np.nan), "dd_from_52w_high_pct": round(dd_52w, 1),
                     "vol_63d_pct": round(vol63, 1),
                     "cost_over_horizon_pct": round(position_cost_pct(t, "short" if side == "short" else side, costs), 2)})
    return pd.DataFrame(rows)
