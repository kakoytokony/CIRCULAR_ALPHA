"""Versioned, source-cited credit & circularity constants plus the research-pack loader.

Two layers, both traceable:

1. ``load_research_pack()`` parses ``data/circular_alpha_research_pack.xlsx``
   (J.P. Morgan North America Credit Research Oct 29 & Nov 4 2026, WSJ, Bloomberg).
   Formula cells (totals, premiums) are recomputed here from their components
   because openpyxl does not evaluate Excel formulas.
2. ``SNAPSHOT`` holds the same Oct–Nov 2026 figures as typed constants, each with
   a mandatory ``source`` string, for numbers that appear only in prose in the
   pack or in METHODOLOGY_AND_SPEC.md §3.2. ``tests/test_methodology_v3.py``
   asserts that every constant that also exists in the workbook matches it.

Nothing in this module is estimated, extrapolated or invented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from data import BASE_DIR

PACK_PATH = BASE_DIR / "data" / "circular_alpha_research_pack.xlsx"
PACK_VERSION = "research_pack_2026-11 (JPM Oct 29 & Nov 4 2026 / WSJ / Bloomberg)"

JPM_OCT29 = "J.P. Morgan NA Credit Research, AI Capex Funding: Bond Matrix 3.0 (Oct 29, 2026)"
JPM_FIG10 = "J.P. Morgan NA Credit Research, Bond Matrix 3.0, Figure 10 (Oct 29, 2026)"
JPM_NOV4 = "J.P. Morgan NA Credit Research, Early Thoughts On Blue Wave Implications (Nov 4, 2026)"
WSJ_NOV1 = "WSJ, Mackintosh, The Tangled Web of AI Bets Is Reason to Fear Tech Stocks (Nov 1, 2026)"
WSJ_OCT30 = "WSJ The Journal podcast, Is AI Running Out of Money to Borrow? (Oct 30, 2026)"
WSJ_NOV5 = "WSJ, Demos, How the Data Center Backlash Could Play Into the Hands of Big Tech (Nov 5, 2026)"
BBG_OCT13 = "Bloomberg, Donnan, The New Circular AI Finance Trend Is Global and Spooky (Oct 13, 2026)"


@dataclass(frozen=True)
class SourcedValue:
    key: str
    value: float
    unit: str
    label_en: str
    label_ru: str
    source: str
    as_of: str


def _sv(key, value, unit, en, ru, source, as_of) -> SourcedValue:
    return SourcedValue(key, float(value), unit, en, ru, source, as_of)


SNAPSHOT: dict[str, SourcedValue] = {v.key: v for v in [
    _sv("total_tracked_debt_bn", 629, "$bn", "Total tracked hyperscaler & DC debt", "Совокупный отслеживаемый долг гиперскейлеров и ЦОД", JPM_OCT29, "2026-10-29"),
    _sv("total_tracked_debt_jul_bn", 602, "$bn", "Total tracked debt, July 2026", "Отслеживаемый долг, июль 2026", JPM_OCT29, "2026-07"),
    _sv("hy_hpc_spread_bps", 503, "bps", "J.P. Morgan HY HPC index spread", "Спред индекса HY HPC J.P. Morgan", JPM_OCT29, "2026-10-29"),
    _sv("hy_hpc_change_mom_bps", 244, "bps", "HY HPC spread change, 1 month", "Изменение спреда HY HPC за месяц", JPM_OCT29, "2026-10-29"),
    _sv("hy_hpc_peak_bps", 544, "bps", "HY HPC spread peak", "Пик спреда HY HPC", JPM_OCT29, "2026-10"),
    _sv("hy_hpc_index_weight_pct", 3.44, "%", "HPC weight in JPM HY index", "Вес HPC в индексе JPM HY", JPM_OCT29, "2026-10-29"),
    _sv("hy_hpc_index_weight_prior_pct", 1.07, "%", "HPC weight in JPM HY index (start of 2026)", "Вес HPC в индексе JPM HY (начало 2026)", JPM_OCT29, "2026-01"),
    _sv("lev_loans_hpc_bps", 667, "bps", "Leveraged loans HPC (3y) spread", "Спред Leveraged Loans HPC (3y)", JPM_OCT29, "2026-10-29"),
    _sv("hg_tech_spread_bps", 127, "bps", "HG Technology spread", "Спред HG Technology", JPM_OCT29, "2026-10-29"),
    _sv("hyperscaler_10y_plus_bps", 168.8, "bps", "Hyperscaler 10y+ spread", "Спред гиперскейлеров 10+ лет", JPM_FIG10, "2026-10-29"),
    _sv("juli_10y_plus_bps", 107.2, "bps", "JULI 10y+ spread", "Спред JULI 10+ лет", JPM_FIG10, "2026-10-29"),
    _sv("hyperscaler_10y_basis_end2025_bps", 15.8, "bps", "Hyperscaler 10y+ basis vs JULI, end-2025", "Базис гиперскейлеров 10+ лет к JULI, конец 2025", JPM_FIG10, "2025-12-31"),
    _sv("hyperscaler_10s30s_bps", 44, "bps", "Hyperscaler 10s30s curve", "Кривая 10s30s гиперскейлеров", JPM_FIG10, "2026-10-29"),
    _sv("ig_market_10s30s_bps", 14, "bps", "Rest of IG market 10s30s curve", "Кривая 10s30s остального рынка IG", JPM_FIG10, "2026-10-29"),
    _sv("crwv_2031_spread_bps", 1000, "bps", "CoreWeave 2031 9% bond spread (>1,000 bps)", "Спред облигаций CoreWeave 2031 9% (>1000 б.п.)", WSJ_NOV1, "2026-11-01"),
    _sv("crwv_2031_spread_jun_bps", 400, "bps", "CoreWeave 2031 spread, June 2026 (~400 bps)", "Спред CoreWeave 2031, июнь 2026 (~400 б.п.)", WSJ_NOV1, "2026-06"),
    _sv("crwv_equity_ytd_pct", -23, "%", "CoreWeave equity YTD", "Акции CoreWeave с начала года", WSJ_NOV1, "2026-11-01"),
    _sv("anthropic_debt_bn", 63, "$bn", "Anthropic Broadcom-guaranteed SPV debt", "Долг Anthropic через SPV с гарантией Broadcom", WSJ_OCT30, "2026-10-30"),
    _sv("nvda_ohio_backstop_bn", 105, "$bn", "Nvidia backstop, SB Energy Ohio DC", "Гарантия Nvidia по ЦОД SB Energy в Огайо", WSJ_NOV1, "2026-11-01"),
    _sv("meta_spo_bn", 25, "$bn", "Meta equity offering", "Размещение акций Meta", WSJ_NOV1, "2026-11-01"),
    _sv("meta_spo_chip_anchor_bn", 5, "$bn", "Chipmaker anchor in Meta SPO", "Якорное участие чипмейкеров в SPO Meta", WSJ_NOV1, "2026-11-01"),
    _sv("volta_deal_bn", 5, "$bn", "Volta Infrastructure deal size", "Размер сделки Volta Infrastructure", JPM_OCT29, "2026-10"),
    _sv("volta_underwritten_bn", 1.3, "$bn", "Volta deal underwritten by JPM & GS", "Часть Volta, гарантированная JPM и GS", JPM_OCT29, "2026-10"),
    _sv("referenda_passed", 42, "count", "Local DC-restriction ballots passed/leading", "Местные инициативы против ЦОД прошли/лидируют", JPM_NOV4, "2026-11-04"),
    _sv("referenda_total", 45, "count", "Local DC-restriction ballots on Nov 3", "Местные инициативы против ЦОД 3 ноября", JPM_NOV4, "2026-11-04"),
    _sv("texas_hy_pipeline_share", 0.78, "share", "Public HY DC pipeline in Texas", "Доля публичного HY-пайплайна ЦОД в Техасе", JPM_NOV4, "2026-11-04"),
    _sv("wgmi_post_election_pct", -9, "%", "WGMI ETF move, day after election (~)", "ETF WGMI на следующий день после выборов (~)", WSJ_NOV5, "2026-11-05"),
    _sv("sox_post_election_pct", -3, "%", "PHLX Semiconductor index post-election", "Индекс PHLX Semiconductor после выборов", WSJ_NOV5, "2026-11-05"),
    _sv("amzn_msft_post_election_pct", 6, "%", "AMZN & MSFT post-election (>+6%)", "AMZN и MSFT после выборов (>+6%)", WSJ_NOV5, "2026-11-05"),
    _sv("us_chip_imports_dec23_bn", 11, "$bn/month", "U.S. chip imports, Dec 2023", "Импорт чипов в США, дек. 2023", BBG_OCT13, "2023-12"),
    _sv("us_chip_imports_jul26_bn", 71, "$bn/month", "U.S. chip imports, Jul 2026", "Импорт чипов в США, июль 2026", BBG_OCT13, "2026-07"),
]}

JPM_PRINTED_TOTALS_BN = {"Other": 111.0, "MSFT": 44.0}

# Baseline implied by the reported level and 1-month change: 503 - 244 = 259 bps.
HY_HPC_BASELINE_BPS = SNAPSHOT["hy_hpc_spread_bps"].value - SNAPSHOT["hy_hpc_change_mom_bps"].value


# ---------------------------------------------------------------------------
# Research-pack parser
# ---------------------------------------------------------------------------
@dataclass
class ResearchPack:
    bond_matrix: pd.DataFrame
    tenor_curve: pd.DataFrame
    credit_segments: pd.DataFrame
    blue_wave_states: pd.DataFrame
    landmark_deals: pd.DataFrame
    dotcom_benchmarks: pd.DataFrame
    literature: pd.DataFrame
    kpis: pd.DataFrame
    source_path: str
    loaded_from_workbook: bool
    notes: list[str] = field(default_factory=list)


def _rows(ws) -> list[list]:
    return [[c.value for c in row] for row in ws.iter_rows()]


def _block(rows: list[list], header_first_cell: str, n_cols: int) -> pd.DataFrame:
    """Return the table whose header row starts with ``header_first_cell``.

    The block ends at the first fully blank row. Formula strings are kept as
    None so downstream code recomputes them from components.
    """
    start = next(i for i, r in enumerate(rows) if r and r[0] == header_first_cell)
    header = [str(h) for h in rows[start][:n_cols]]
    body = []
    for r in rows[start + 1:]:
        vals = list(r[:n_cols]) + [None] * max(0, n_cols - len(r))
        if all(v is None for v in vals):
            break
        body.append([None if isinstance(v, str) and v.startswith("=") else v for v in vals])
    return pd.DataFrame(body, columns=header)


def load_research_pack(path: Path = PACK_PATH) -> ResearchPack:
    """Parse the verified Oct–Nov 2026 research pack workbook."""
    import openpyxl  # imported lazily so the app can start without it

    wb = openpyxl.load_workbook(path, data_only=False, read_only=True)
    dash = _rows(wb["Dashboard_BondMatrix"])
    credit = _rows(wb["Credit_BlueWave_Deals"])
    dotcom = _rows(wb["DotCom_2000_Scenarios"])
    wb.close()

    bm = _block(dash, "Issuer / Group", 4)
    bm = bm[~bm["Issuer / Group"].astype(str).str.match(r"^(TOTAL ECOSYSTEM|Note)")].copy()
    bm.columns = ["issuer", "direct_ig_bn", "dc_ig_bn", "dc_hy_bn"]
    for c in ["direct_ig_bn", "dc_ig_bn", "dc_hy_bn"]:
        bm[c] = pd.to_numeric(bm[c], errors="coerce").fillna(0.0)
    bm["total_bn"] = bm[["direct_ig_bn", "dc_ig_bn", "dc_hy_bn"]].sum(axis=1)
    bm["share_pct"] = 100.0 * bm["total_bn"] / bm["total_bn"].sum()
    # J.P. Morgan's printed chart totals differ by $1B for two rows (component rounding;
    # see the workbook note under the matrix). Keep both so every figure is traceable.
    bm["jpm_printed_bn"] = bm["issuer"].map(JPM_PRINTED_TOTALS_BN).fillna(bm["total_bn"])
    bm = bm.sort_values("total_bn", ascending=False).reset_index(drop=True)

    tc = _block(dash, "Tenor Bucket", 5)
    tc.columns = ["tenor", "juli_bps", "hyperscaler_bps", "premium_bps", "note"]
    tc["juli_bps"] = pd.to_numeric(tc["juli_bps"], errors="coerce")
    tc["hyperscaler_bps"] = pd.to_numeric(tc["hyperscaler_bps"], errors="coerce")
    tc["premium_bps"] = (tc["hyperscaler_bps"] - tc["juli_bps"]).round(1)

    kp = _block(dash, "Metric", 4)
    kp.columns = ["metric", "level", "context", "source"]
    # Recompute the two formula KPIs (=E21 total debt, =D30 10y+ premium).
    kp.loc[kp["metric"].str.contains("Total Tracked", na=False), "level"] = float(bm["total_bn"].sum())
    kp.loc[kp["metric"].str.contains("10y\\+", na=False), "level"] = float(tc["premium_bps"].iloc[-1])

    seg = _block(credit, "Segment / Index", 5)
    seg.columns = ["segment", "level_bps", "change_bps", "peak_or_range", "context"]
    st_ = _block(credit, "State", 5)
    st_.columns = ["state", "governor_elect", "policy", "exposed", "horizon"]
    deals = _block(credit, "Deal / Signal", 5)
    deals.columns = ["deal", "amount", "parties", "mechanism", "source"]

    dc = _block(dotcom, "Benchmark / Entity", 6)
    dc.columns = ["benchmark", "peak", "trough", "drawdown", "note", "source"]
    lit = _block(dotcom, "Authors (Year)", 4)
    lit.columns = ["authors", "title", "journal", "role"]

    return ResearchPack(
        bond_matrix=bm, tenor_curve=tc, credit_segments=seg, blue_wave_states=st_,
        landmark_deals=deals, dotcom_benchmarks=dc, literature=lit, kpis=kp,
        source_path=str(path), loaded_from_workbook=True,
        notes=["Totals and tenor premiums recomputed from workbook components (formula cells)."],
    )


def load_research_pack_safe(path: Path = PACK_PATH) -> tuple[ResearchPack | None, str]:
    try:
        return load_research_pack(path), f"Research pack loaded: {Path(path).name} ({PACK_VERSION})"
    except Exception as exc:  # missing file or openpyxl not installed
        return None, f"Research pack unavailable ({exc}); using source-cited SNAPSHOT constants only."


# ---------------------------------------------------------------------------
# Trigger thresholds (METHODOLOGY_AND_SPEC.md §4.1.3) — pre-registered, v3.0
# ---------------------------------------------------------------------------
TRIGGER_THRESHOLDS = {
    "hy_hpc_spread_bps": {"amber": 450.0, "red": 600.0},
    "hyperscaler_10y_basis_bps": {"amber": 45.0, "red": 75.0},
    "crwv_2031_spread_bps": {"amber": 700.0, "red": 1000.0},
}
SWITCH_OFF = {"hy_hpc_spread_below_bps": 350.0, "bsadf_below": "cv90"}


def traffic_light(value: float, amber: float, red: float, red_inclusive: bool = False) -> str:
    if pd.isna(value):
        return "N/A"
    if value > red or (red_inclusive and value >= red):
        return "RED"
    if value >= amber:
        return "AMBER"
    return "GREEN"


def current_credit_inputs(pack: ResearchPack | None = None) -> dict[str, float]:
    """Return the latest credit levels, preferring the workbook over constants."""
    out = {
        "hy_hpc_spread_bps": SNAPSHOT["hy_hpc_spread_bps"].value,
        "hyperscaler_10y_basis_bps": round(SNAPSHOT["hyperscaler_10y_plus_bps"].value - SNAPSHOT["juli_10y_plus_bps"].value, 1),
        "crwv_2031_spread_bps": SNAPSHOT["crwv_2031_spread_bps"].value,
        "volta_underwritten_share": SNAPSHOT["volta_underwritten_bn"].value / SNAPSHOT["volta_deal_bn"].value,
    }
    if pack is not None:
        seg = pack.credit_segments.set_index("segment")
        if "J.P. Morgan HY HPC Index" in seg.index:
            out["hy_hpc_spread_bps"] = float(seg.loc["J.P. Morgan HY HPC Index", "level_bps"])
        if "CoreWeave 2031 9% Bond" in seg.index:
            out["crwv_2031_spread_bps"] = float(seg.loc["CoreWeave 2031 9% Bond", "level_bps"])
        out["hyperscaler_10y_basis_bps"] = float(pack.tenor_curve["premium_bps"].iloc[-1])
    return out


def supply_friction_flags() -> list[dict]:
    """Ofek–Richardson (2003) equity-supply + underwriting-friction events (sourced)."""
    v = SNAPSHOT
    share = v["volta_underwritten_bn"].value / v["volta_deal_bn"].value
    return [
        {"event": "Anthropic IPO S-1 filed alongside $63B SPV debt", "active": True, "source": WSJ_OCT30},
        {"event": f"Meta ${v['meta_spo_bn'].value:.0f}B SPO (${v['meta_spo_chip_anchor_bn'].value:.0f}B chipmaker anchor)", "active": True, "source": WSJ_NOV1},
        {"event": f"Volta: only {share:.0%} of ${v['volta_deal_bn'].value:.0f}B underwritten", "active": share < 0.5, "source": JPM_OCT29},
    ]
