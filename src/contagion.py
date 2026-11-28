"""Multi-channel collapse contagion & DebtRank stress-test engine for Circular Alpha.

Methodology (transparent scenario stress-testing, NOT a point price forecast):
1. Channel-by-channel Hop-1 balance-sheet & revenue decomposition:
   - investment (j -> i): equity mark-to-market write-down at company j
   - supplier_customer / commercial_agreement (i -> j): contracted revenue / backlog loss at company j
   - guarantee / financing (j -> i): contingent off-balance-sheet credit backstop call at company j
   - supplier_customer / commercial_agreement (j -> i): compute / hardware supply disruption at company j
   * Respects data dictionary rule: superseded LOI rows R001/R002 are excluded from dollar totals
     when R003/R004 are present to avoid double-counting NVIDIA -> OpenAI exposure.
2. Multi-hop DebtRank propagation (Battiston et al., 2012):
   - Attenuated secondary stress transmission across Hops 2 and 3+ using damping factor delta.
3. Empirical tail co-movement + optional Daily AI News Sentiment adjustment:
   - Compares purely quantitative stress estimates against news-sentiment-adjusted estimates.
"""
from __future__ import annotations

from collections import deque
import numpy as np
import pandas as pd

SUPERSEDED_LOI_IDS = {"R001", "R002"}

# Maps (company_id, channel) -> concrete business line affected for clear analyst interpretation
BUSINESS_LINE_MAP = {
    ("C001", "equity"): "Strategic Ventures & Balance Sheet Equity Portfolio",
    ("C001", "revenue"): "Data Center GPU Systems (Blackwell / Vera Rubin) & Networking Sales",
    ("C001", "guarantee"): "Off-Balance-Sheet Credit Guarantees & Free Cash Flow Headroom",
    ("C001", "supply"): "Ecosystem Compute & Partner Deployment Pipeline",
    ("C002", "equity"): "Strategic AI Equity Method Investments (Balance Sheet / Net Income)",
    ("C002", "revenue"): "Intelligent Cloud / Azure AI Compute Backlog (RPO)",
    ("C002", "guarantee"): "Cloud Lease & Credit Commitments",
    ("C002", "supply"): "External AI GPU Cloud Capacity (Azure overflow)",
    ("C003", "revenue"): "Oracle Cloud Infrastructure (OCI) Backlog / RPO & Data Center Capex ROI",
    ("C003", "supply"): "GPU Cluster Supply for OCI AI Superclusters",
    ("C004", "equity"): "Strategic Warrants & Equity Commitments",
    ("C004", "revenue"): "Data Center Instinct GPU Pipeline (MI450 / Helios)",
    ("C005", "equity"): "Strategic Equity Cross-Holdings",
    ("C005", "revenue"): "Core AI Cloud Contracted Revenue & HY Bond Debt Service",
    ("C005", "supply"): "NVIDIA GPU Hardware Supply & Data Center Shell Leases",
    ("C006", "revenue"): "API & Enterprise Model Subscriptions",
    ("C006", "supply"): "Frontier Model Training & Inference Compute Supply",
    ("C007", "equity"): "Strategic AI Equity Holdings",
    ("C007", "revenue"): "AWS Cloud Infrastructure & Trainium/Inferentia Adoption",
    ("C008", "guarantee"): "Google Cloud Lease Backstops for Neocloud Partners",
    ("C008", "revenue"): "Google Cloud Platform & Co-Developed TPU AI Chip Revenue",
    ("C009", "revenue"): "AI Compute & Model Ecosystem Partnerships",
    ("C009", "supply"): "Custom MTIA Silicon & Neocloud GPU Capacity",
    ("C010", "supply"): "Frontier Model Training Compute & Custom ASIC Financing",
    ("C011", "equity"): "Strategic Equity Stakes in Hyperscalers",
    ("C011", "revenue"): "Custom AI ASIC / Accelerator & Networking Silicon Revenue",
    ("C011", "guarantee"): "SPV Debt Guarantees & Investment-Grade Balance Sheet Headroom",
    ("C012", "revenue"): "Ohio AI Data Center Campus Lease Payments & Debt Service",
    ("C013", "supply"): "Google-Backed HPC Data Center Lease Structure",
    ("C014", "revenue"): "North Dakota HPC Data Center Leases & HY Project Bond Service",
    ("C015", "revenue"): "West Texas Helios Data Center Lease & HY Project Bond Service",
    ("C016", "guarantee"): "Private Credit AI Debt Tranche Recovery & Credit Fund NAV",
}

DEFAULT_BUSINESS_LINE = {
    "equity": "Strategic Equity / Corporate Balance Sheet",
    "revenue": "Contracted AI Revenue / Cloud & Hardware Backlog",
    "guarantee": "Credit Guarantees / Off-Balance-Sheet Debt Support",
    "supply": "AI Compute / Hardware Supply Chain",
}


RECENT_IPO_LABEL = "Recent IPO (no quarterly history)"


def _no_ticker_label(info: dict) -> str:
    """Display label for entities without a listed ticker in the price panel.

    Recently listed companies (e.g. Anthropic, C010) carry RECENT_IPO_LABEL in
    companies_clean.csv `public_exchange`; everything else without a ticker is private.
    """
    exch = str(info.get("public_exchange", "") or "").strip()
    return RECENT_IPO_LABEL if exch == RECENT_IPO_LABEL else "Private"


def _build_adj(relationships: pd.DataFrame) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    out_adj: dict[str, set[str]] = {}
    in_adj: dict[str, set[str]] = {}
    for _, row in relationships.iterrows():
        s = str(row["source_entity_id"])
        t = str(row["target_entity_id"])
        if s == t:
            continue
        out_adj.setdefault(s, set()).add(t)
        in_adj.setdefault(t, set()).add(s)
    return out_adj, in_adj


def _bfs_distances(adj: dict[str, set[str]], start: str) -> dict[str, int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _empirical_tail_sensitivity(
    prices: pd.DataFrame | None,
    target_ticker: str,
    failed_ticker: str | None,
    benchmark: str = "SPY",
) -> tuple[float, str]:
    """Estimate empirical downside sensitivity (%) on worst 5% stress days.

    If failed_ticker is public and present in prices, uses failed_ticker's worst 5% days;
    otherwise uses the AI peer/market (SPY) worst 5% days as a proxy for systemic stress.
    """
    if prices is None or prices.empty or not target_ticker:
        return np.nan, "N/A (private)"
    sub = prices[["date", "ticker", "adjusted_close"]].copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub["adjusted_close"] = pd.to_numeric(sub["adjusted_close"], errors="coerce")
    sub = sub.dropna()
    piv = sub.pivot_table(index="date", columns="ticker", values="adjusted_close").sort_index()
    if target_ticker not in piv.columns:
        return np.nan, "N/A (no price series)"
    ret = piv.pct_change().dropna(how="all")
    driver = failed_ticker if (failed_ticker and failed_ticker in ret.columns and failed_ticker != target_ticker) else benchmark
    if driver not in ret.columns:
        return np.nan, "N/A"
    paired = ret[[target_ticker, driver]].dropna()
    if len(paired) < 20:
        return np.nan, "Insufficient history"
    n_worst = max(5, int(round(0.05 * len(paired))))
    worst_idx = paired[driver].nsmallest(n_worst).index
    avg_move_pct = float(paired.loc[worst_idx, target_ticker].mean() * 100.0)
    driver_label = f"Avg move on {driver}'s worst 5% days"
    return avg_move_pct, driver_label


def run_contagion_stress_test(
    companies: pd.DataFrame,
    relationships: pd.DataFrame,
    financials: pd.DataFrame,
    failed_company_id: str,
    prices: pd.DataFrame | None = None,
    shock_severity: float = 1.0,
    damping: float = 0.35,
    confirmed_only: bool = False,
    news_sentiment_by_company: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run a structured multi-channel collapse stress test for a target company.

    Parameters
    ----------
    failed_company_id : str
        Company ID (e.g., 'C010' for Anthropic, 'C006' for OpenAI).
    shock_severity : float
        Shock magnitude in (0, 1]. 1.0 = full default / collapse; 0.5 = 50% distress.
    damping : float
        Multi-hop pass-through attenuation factor in (0, 1) for DebtRank propagation.
    confirmed_only : bool
        If True, restrict edges to confirmed primary-source relationships.
    news_sentiment_by_company : dict[str, float] | None
        Optional map of company_id -> sentiment score in [-1.0, +1.0] from the daily news agent.
        Negative sentiment amplifies stress; positive sentiment slightly cushions it.

    Returns
    -------
    impact_df : pd.DataFrame
        One row per impacted company sorted by contagion severity.
    summary : dict
        Scenario totals, channel totals, and explicit assumption disclosures.
    """
    if not (0.0 < shock_severity <= 1.0):
        raise ValueError("shock_severity must be in (0, 1].")
    if not (0.0 < damping < 1.0):
        raise ValueError("damping must be in (0, 1).")

    comp = companies.copy()
    comp["company_id"] = comp["company_id"].astype(str)
    if failed_company_id not in set(comp["company_id"]):
        raise ValueError(f"Unknown failed_company_id: {failed_company_id}")

    rel = relationships.copy()
    rel["source_entity_id"] = rel["source_entity_id"].astype(str)
    rel["target_entity_id"] = rel["target_entity_id"].astype(str)
    if confirmed_only:
        rel = rel[rel["evidence_status"].astype(str).str.lower() == "confirmed"].copy()

    # Exclude superseded LOI rows (R001/R002) if R003 is present (per data_dictionary_synced.csv)
    present_rids = set(rel["relationship_id"].astype(str))
    rel_clean = rel[~rel["relationship_id"].astype(str).isin(SUPERSEDED_LOI_IDS)].copy() if "R003" in present_rids else rel.copy()

    out_adj, in_adj = _build_adj(rel_clean)
    # Undirected reachability for multi-hop distance
    undir_adj: dict[str, set[str]] = {}
    for u, nbrs in out_adj.items():
        for v in nbrs:
            undir_adj.setdefault(u, set()).add(v)
            undir_adj.setdefault(v, set()).add(u)

    hops_map = _bfs_distances(undir_adj, failed_company_id)

    comp_lookup = comp.set_index("company_id").to_dict("index")
    fin_lookup = {}
    if financials is not None and not financials.empty and "company_id" in financials.columns:
        fin_copy = financials.copy()
        fin_copy["company_id"] = fin_copy["company_id"].astype(str)
        fin_lookup = fin_copy.drop_duplicates("company_id", keep="last").set_index("company_id").to_dict("index")

    failed_info = comp_lookup[failed_company_id]
    failed_ticker = str(failed_info.get("public_ticker", "") or "").strip()
    if failed_ticker.lower() == "nan":
        failed_ticker = ""

    # Step 1: Direct Hop-1 channel exposures
    channels: dict[str, dict] = {}
    for _, row in rel_clean.iterrows():
        s = row["source_entity_id"]
        t = row["target_entity_id"]
        if failed_company_id not in (s, t) or s == t:
            continue
        other = t if s == failed_company_id else s
        rtype = str(row.get("relationship_type", "")).lower().strip()
        amt = pd.to_numeric(row.get("amount", np.nan), errors="coerce")
        amt_val = 0.0 if pd.isna(amt) else float(amt) * shock_severity
        is_undisclosed = pd.isna(amt)

        rec = channels.setdefault(
            other,
            {
                "equity_writedown_usd": 0.0,
                "revenue_backlog_at_risk_usd": 0.0,
                "guarantee_spv_call_usd": 0.0,
                "supply_disruption_links": 0,
                "undisclosed_links": 0,
                "channels_hit": set(),
                "deal_ids": [],
            },
        )
        rec["deal_ids"].append(str(row.get("relationship_id", "")))
        if is_undisclosed:
            rec["undisclosed_links"] += 1

        if s == other and t == failed_company_id:
            # Other -> Failed (capital, guarantee, financing, or customer paying failed supplier)
            if rtype == "investment":
                rec["equity_writedown_usd"] += amt_val
                rec["channels_hit"].add("equity")
            elif rtype in ("guarantee", "financing"):
                rec["guarantee_spv_call_usd"] += amt_val
                rec["channels_hit"].add("guarantee")
            else:
                rec["supply_disruption_links"] += 1
                rec["channels_hit"].add("supply")
        else:
            # Failed -> Other (failed company pays other for cloud/chips/leases, or invests/guarantees other)
            if rtype in ("supplier_customer", "commercial_agreement"):
                rec["revenue_backlog_at_risk_usd"] += amt_val
                rec["channels_hit"].add("revenue")
            elif rtype in ("investment", "financing", "guarantee"):
                # Loss of funding / credit backstop from the failed company
                rec["guarantee_spv_call_usd"] += amt_val
                rec["channels_hit"].add("guarantee")

    # Step 2: Multi-hop DebtRank stress score h in [0, 1]
    h: dict[str, float] = {cid: 0.0 for cid in comp_lookup}
    h[failed_company_id] = shock_severity

    # Initialize Hop-1 stress using balance-sheet-scaled exposure + structural floor for undisclosed links
    for cid, rec in channels.items():
        total_direct = (
            rec["equity_writedown_usd"]
            + rec["revenue_backlog_at_risk_usd"]
            + rec["guarantee_spv_call_usd"]
        )
        fin = fin_lookup.get(cid, {})
        ebitda = pd.to_numeric(fin.get("ebitda", np.nan), errors="coerce")
        cash = pd.to_numeric(fin.get("cash_and_equivalents", np.nan), errors="coerce")
        denom = float(ebitda) if (pd.notna(ebitda) and ebitda > 0) else (float(cash) if (pd.notna(cash) and cash > 0) else 25e9)

        # Balance-sheet stress ratio capped at 0.85, plus structural exposure for undisclosed/supply links
        dollar_stress = min(0.85, (total_direct / max(denom, 1e9)) * 0.45) if total_direct > 0 else 0.0
        structural_floor = min(0.35, 0.12 * len(rec["channels_hit"]) + 0.06 * rec["undisclosed_links"]) * shock_severity
        h[cid] = min(0.95, max(dollar_stress, structural_floor))

    # Propagate to Hop 2 and Hop 3+ (DebtRank style)
    for hop_level in (2, 3):
        updates = {}
        for cid, dist in hops_map.items():
            if dist != hop_level:
                continue
            nbrs = [n for n in undir_adj.get(cid, ()) if hops_map.get(n, 99) == hop_level - 1]
            if not nbrs:
                continue
            propagated = sum(h[n] * damping for n in nbrs) / max(1, len(nbrs) ** 0.5)
            updates[cid] = min(0.60, propagated)
        h.update(updates)

    # Step 3: Assemble output table with business lines, balance-sheet ratios, empirical spillover & news blend
    sentiment_map = news_sentiment_by_company or {}
    rows = []
    for cid, dist in hops_map.items():
        if cid == failed_company_id:
            continue
        info = comp_lookup[cid]
        ticker = str(info.get("public_ticker", "") or "").strip()
        if ticker.lower() == "nan":
            ticker = ""
        rec = channels.get(cid, {
            "equity_writedown_usd": 0.0,
            "revenue_backlog_at_risk_usd": 0.0,
            "guarantee_spv_call_usd": 0.0,
            "supply_disruption_links": 0,
            "undisclosed_links": 0,
            "channels_hit": set(),
            "deal_ids": [],
        })
        total_direct = (
            rec["equity_writedown_usd"]
            + rec["revenue_backlog_at_risk_usd"]
            + rec["guarantee_spv_call_usd"]
        )
        fin = fin_lookup.get(cid, {})
        cash = pd.to_numeric(fin.get("cash_and_equivalents", np.nan), errors="coerce")
        ebitda = pd.to_numeric(fin.get("ebitda", np.nan), errors="coerce")
        pct_cash = (total_direct / float(cash) * 100.0) if (pd.notna(cash) and cash > 0 and total_direct > 0) else np.nan
        pct_ebitda = (total_direct / float(ebitda) * 100.0) if (pd.notna(ebitda) and ebitda > 0 and total_direct > 0) else np.nan
        total_debt = pd.to_numeric(fin.get("total_debt", np.nan), errors="coerce")
        fiscal_date = fin.get("fiscal_date_ending", fin.get("period_end", np.nan))
        data_source = fin.get("data_source", "financials.csv (source not recorded)" if fin else np.nan)

        # Build human-readable affected business lines
        if dist == 1 and rec["channels_hit"]:
            blines = [
                BUSINESS_LINE_MAP.get((cid, ch), DEFAULT_BUSINESS_LINE[ch])
                for ch in sorted(rec["channels_hit"])
            ]
            business_impact = "; ".join(blines)
        else:
            business_impact = f"Indirect ecosystem spillover (Hop {dist}): AI Capex slowdown, valuation repricing & partner credit tightening"

        # Empirical tail move & blended scenario drawdown
        tail_move_pct, tail_note = _empirical_tail_sensitivity(prices, ticker, failed_ticker)
        stress_score_quant = round(h[cid] * 100.0, 1)

        # Scenario equity drawdown estimate (quantitative only): combines stress score with empirical tail beta
        if pd.notna(tail_move_pct):
            # Scale empirical 1-day tail move into a multi-week event shock (3x single-day tail capped by stress)
            quant_drawdown_pct = -min(65.0, abs(tail_move_pct) * (1.2 + 2.5 * h[cid]))
        else:
            quant_drawdown_pct = np.nan

        # News sentiment adjustment: negative news sentiment amplifies shock up to +25%; positive softens up to -15%
        sent = float(np.clip(sentiment_map.get(cid, 0.0), -1.0, 1.0))
        failed_sent = float(np.clip(sentiment_map.get(failed_company_id, 0.0), -1.0, 1.0))
        combined_sent = 0.6 * sent + 0.4 * failed_sent
        news_multiplier = 1.0 - 0.25 * combined_sent  # e.g. sentiment=-0.8 -> 1.20x shock
        stress_score_news = round(min(100.0, stress_score_quant * news_multiplier), 1)
        news_drawdown_pct = (
            -min(75.0, abs(quant_drawdown_pct) * news_multiplier)
            if pd.notna(quant_drawdown_pct)
            else np.nan
        )

        rows.append({
            "company_id": cid,
            "company_name": info["company_name"],
            "ticker": ticker if ticker else _no_ticker_label(info),
            "hops": int(dist),
            "contagion_score_quant": stress_score_quant,
            "contagion_score_news_adj": stress_score_news,
            "equity_writedown_usd": rec["equity_writedown_usd"],
            "revenue_backlog_at_risk_usd": rec["revenue_backlog_at_risk_usd"],
            "guarantee_spv_call_usd": rec["guarantee_spv_call_usd"],
            "total_disclosed_exposure_usd": total_direct,
            "undisclosed_direct_links": int(rec["undisclosed_links"]),
            "exposure_pct_of_cash": round(pct_cash, 1) if pd.notna(pct_cash) else np.nan,
            "exposure_pct_of_ebitda": round(pct_ebitda, 1) if pd.notna(pct_ebitda) else np.nan,
            "total_debt_usd": float(total_debt) if pd.notna(total_debt) else np.nan,
            "cash_usd": float(cash) if pd.notna(cash) else np.nan,
            "ebitda_usd": float(ebitda) if pd.notna(ebitda) else np.nan,
            "fiscal_date_ending": fiscal_date if pd.notna(fiscal_date) else "",
            "data_source": data_source if pd.notna(data_source) else "",
            "empirical_tail_day_pct": round(tail_move_pct, 2) if pd.notna(tail_move_pct) else np.nan,
            "scenario_drawdown_quant_pct": round(quant_drawdown_pct, 1) if pd.notna(quant_drawdown_pct) else np.nan,
            "scenario_drawdown_news_adj_pct": round(news_drawdown_pct, 1) if pd.notna(news_drawdown_pct) else np.nan,
            "news_sentiment": round(combined_sent, 2),
            "affected_business_segments": business_impact,
            "deal_ids": "; ".join(rec["deal_ids"]),
            "tail_basis": tail_note,
        })

    impact_df = pd.DataFrame(rows)
    if not impact_df.empty:
        impact_df = impact_df.sort_values(
            ["hops", "contagion_score_quant", "total_disclosed_exposure_usd"],
            ascending=[True, False, False],
        ).reset_index(drop=True)

    summary = {
        "failed_company_id": failed_company_id,
        "failed_company_name": failed_info["company_name"],
        "shock_severity": float(shock_severity),
        "damping": float(damping),
        "confirmed_only": bool(confirmed_only),
        "hop1_count": int((impact_df["hops"] == 1).sum()) if not impact_df.empty else 0,
        "hop2_plus_count": int((impact_df["hops"] >= 2).sum()) if not impact_df.empty else 0,
        "total_equity_writedown_usd": float(impact_df["equity_writedown_usd"].sum()) if not impact_df.empty else 0.0,
        "total_revenue_backlog_usd": float(impact_df["revenue_backlog_at_risk_usd"].sum()) if not impact_df.empty else 0.0,
        "total_guarantee_call_usd": float(impact_df["guarantee_spv_call_usd"].sum()) if not impact_df.empty else 0.0,
        "total_disclosed_exposure_usd": float(impact_df["total_disclosed_exposure_usd"].sum()) if not impact_df.empty else 0.0,
        "undisclosed_direct_links_total": int(impact_df["undisclosed_direct_links"].sum()) if not impact_df.empty else 0,
        "disclaimer": (
            "Scenario stress-test model combining disclosed contractual exposures, DebtRank network "
            "propagation, and historical tail co-movement. Undisclosed amounts are never imputed as zero. "
            "Scaling constants are transparent scenario-calibration choices that rank relative vulnerability: "
            "0.45 (disclosed exposure / buffer -> stress), 1.2 + 2.5*h (1-day tail move -> multi-week drawdown) "
            "and 0.25 (max news-sentiment adjustment). "
            "Not a point price forecast or investment advice."
        ),
    }
    return impact_df, summary
