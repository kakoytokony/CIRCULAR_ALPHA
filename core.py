"""Pure calculations for Circular Alpha. No UI and no network access."""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


def format_money(value, currency="USD") -> str:
    """Format a monetary value using the app's existing units and precision."""
    if pd.isna(value):
        return "N/A"
    value = float(value)
    if abs(value) >= 1_000_000_000_000:
        return f"{currency} {value / 1_000_000_000_000:,.2f}T"
    if abs(value) >= 1_000_000_000:
        return f"{currency} {value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"{currency} {value / 1_000_000:,.2f}M"
    return f"{currency} {value:,.2f}"


def line_midpoint(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float]:
    """Return the midpoint of a straight line."""
    return (x0 + x1) / 2, (y0 + y1) / 2


def arc_midpoint(
    x0: float, y0: float, x1: float, y1: float, repeat: int = 0
) -> tuple[float, float]:
    """Return the midpoint of one of the network's quadratic arcs."""
    bend = 0.20 + 0.12 * repeat
    cx = (x0 + x1) / 2 - bend * (y1 - y0)
    cy = (y0 + y1) / 2 + bend * (x1 - x0)
    t = 0.5
    return (
        (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1,
        (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1,
    )


def curved_arc_points(
    x0: float, y0: float, x1: float, y1: float, repeat: int = 0
) -> list[tuple[float, float]]:
    """Return the 31 sampled points for one of the network's quadratic arcs."""
    bend = 0.20 + 0.12 * repeat
    cx = (x0 + x1) / 2 - bend * (y1 - y0)
    cy = (y0 + y1) / 2 + bend * (x1 - x0)
    return [
        (
            (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1,
            (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1,
        )
        for i in range(31)
        for t in [i / 30]
    ]


def validate_data(companies: pd.DataFrame, relationships: pd.DataFrame) -> dict:
    company_required = {"company_id", "company_name"}
    relationship_required = {
        "relationship_id", "source_entity_id", "target_entity_id", "evidence_status"
    }
    missing_company = company_required - set(companies.columns)
    missing_relationship = relationship_required - set(relationships.columns)
    if missing_company:
        raise ValueError(f"Companies table is missing columns: {sorted(missing_company)}")
    if missing_relationship:
        raise ValueError(
            f"Relationships table is missing columns: {sorted(missing_relationship)}"
        )

    company_ids = companies["company_id"]
    known_ids = set(company_ids.dropna())
    sources, targets = relationships["source_entity_id"], relationships["target_entity_id"]
    status_counts = (
        relationships["evidence_status"].fillna("missing").astype(str).value_counts().to_dict()
    )
    return {
        "company_rows": int(len(companies)),
        "relationship_rows": int(len(relationships)),
        "duplicate_company_ids": int(company_ids.duplicated().sum()),
        "duplicate_relationship_ids": int(relationships["relationship_id"].duplicated().sum()),
        "unknown_source_ids": int((~sources.isin(known_ids) & sources.notna()).sum()),
        "unknown_target_ids": int((~targets.isin(known_ids) & targets.notna()).sum()),
        "missing_company_cells": int(companies.isna().sum().sum()),
        "missing_relationship_cells": int(relationships.isna().sum().sum()),
        "evidence_status_counts": status_counts,
    }


def prepare_relationships(
    companies: pd.DataFrame, relationships: pd.DataFrame, confirmed_only: bool = True
) -> pd.DataFrame:
    required_company = {"company_id", "company_name"}
    required_relationship = {"source_entity_id", "target_entity_id", "evidence_status"}
    if required_company - set(companies.columns):
        raise ValueError(
            f"Companies table is missing columns: {sorted(required_company - set(companies.columns))}"
        )
    if required_relationship - set(relationships.columns):
        raise ValueError(
            "Relationships table is missing columns: "
            f"{sorted(required_relationship - set(relationships.columns))}"
        )
    if companies["company_id"].duplicated().any():
        raise ValueError("Company IDs must be unique.")

    result = relationships.copy()
    if confirmed_only:
        result = result[result["evidence_status"].astype(str).str.lower() == "confirmed"].copy()
    lookup = companies[["company_id", "company_name"]]
    source_lookup = lookup.rename(
        columns={"company_id": "source_entity_id", "company_name": "source_company"}
    )
    target_lookup = lookup.rename(
        columns={"company_id": "target_entity_id", "company_name": "target_company"}
    )
    result = result.merge(
        source_lookup, on="source_entity_id", how="left", validate="many_to_one"
    )
    result = result.merge(
        target_lookup, on="target_entity_id", how="left", validate="many_to_one"
    )
    return result.reset_index(drop=True)


def company_explorer_tables(
    companies: pd.DataFrame,
    relationships: pd.DataFrame,
    financials: pd.DataFrame,
    selected_company_id,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare readable relationship and financial tables for one company."""
    related = relationships[
        (relationships["source_entity_id"] == selected_company_id)
        | (relationships["target_entity_id"] == selected_company_id)
    ].copy()
    names = companies[["company_id", "company_name"]]
    related = related.merge(
        names.rename(columns={"company_id": "source_entity_id", "company_name": "source_name"}),
        on="source_entity_id", how="left",
    )
    related = related.merge(
        names.rename(columns={"company_id": "target_entity_id", "company_name": "target_name"}),
        on="target_entity_id", how="left",
    )
    relationship_columns = [
        "relationship_id", "source_name", "target_name", "relationship_type",
        "announcement_date", "amount", "currency", "evidence_status",
        "source_title", "source_url",
    ]
    related = related.reindex(columns=relationship_columns).fillna("")

    if "company_id" in financials.columns:
        financial_view = financials[financials["company_id"] == selected_company_id].copy()
        financial_view = financial_view.merge(
            names, on="company_id", how="left"
        )
    else:
        financial_view = financials.iloc[0:0].copy()
    return related, financial_view.fillna("")


def missing_values_report(
    companies: pd.DataFrame, relationships: pd.DataFrame, financials: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for table_name, frame in [
        ("Companies", companies),
        ("Relationships", relationships),
        ("Financials", financials),
    ]:
        for column in frame.columns:
            missing_count = int(frame[column].isna().sum())
            rows.append({
                "table": table_name,
                "column": column,
                "missing_count": missing_count,
                "missing_percent": round(100 * missing_count / len(frame), 2) if len(frame) else 0.0,
            })
    return pd.DataFrame(rows, columns=["table", "column", "missing_count", "missing_percent"])


def data_quality_summary(
    validation: dict, missing_report: pd.DataFrame
) -> tuple[list[tuple[str, int]], str, pd.DataFrame]:
    """Prepare plain-language metrics, verdict, and missing-cell details."""
    evidence_counts = validation.get("evidence_status_counts", {})
    normalized_counts = {
        str(key).lower(): int(value) for key, value in evidence_counts.items()
    }
    duplicate_ids = int(validation.get("duplicate_company_ids", 0)) + int(
        validation.get("duplicate_relationship_ids", 0)
    )
    unknown_links = int(validation.get("unknown_source_ids", 0)) + int(
        validation.get("unknown_target_ids", 0)
    )
    pairs = [
        ("Companies", int(validation.get("company_rows", 0))),
        ("Relationships", int(validation.get("relationship_rows", 0))),
        ("Confirmed relationships", normalized_counts.get("confirmed", 0)),
        ("Provisional relationships", normalized_counts.get("provisional", 0)),
        ("Duplicate IDs found", duplicate_ids),
        ("Links pointing to unknown companies", unknown_links),
        ("Empty cells in company data", int(validation.get("missing_company_cells", 0))),
        ("Empty cells in relationship data", int(validation.get("missing_relationship_cells", 0))),
    ]

    problems = []
    if duplicate_ids:
        problems.append(f"{duplicate_ids} duplicate IDs")
    if unknown_links:
        problems.append(f"{unknown_links} links point to unknown companies")
    if problems:
        verdict = "Problems found: " + "; ".join(problems) + "."
    else:
        verdict = "All IDs are unique and every link points to a known company."

    cleaned = missing_report.rename(
        columns={
            "table": "Table",
            "column": "Column",
            "missing_count": "Empty cells",
            "missing_percent": "Share empty (%)",
        }
    ).copy()
    cleaned["Share empty (%)"] = cleaned["Share empty (%)"].round(1)
    cleaned = cleaned[cleaned["Empty cells"] > 0]
    cleaned = cleaned[["Table", "Column", "Empty cells", "Share empty (%)"]]
    return pairs, verdict, cleaned.sort_values(
        "Share empty (%)", ascending=False
    ).reset_index(drop=True)


def calculate_market_metrics(prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    """Calculate price return, annualized daily volatility and beta to benchmark.

    Uses adjusted close prices. Returns are measured from each ticker's first to
    last available observation. Beta uses only overlapping daily-return dates.
    """
    output_columns = [
        "ticker", "latest_price", "total_return_pct", "annualized_volatility_pct",
        "beta_vs_spy", "start_date", "end_date", "observations",
    ]
    required = {"date", "ticker", "adjusted_close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Price data is missing columns: {sorted(missing)}")

    data = prices[["date", "ticker", "adjusted_close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["ticker"] = data["ticker"].astype("string").str.strip()
    data["adjusted_close"] = pd.to_numeric(data["adjusted_close"], errors="coerce")
    data = data.dropna(subset=["date", "ticker", "adjusted_close"])
    data = data[data["ticker"] != ""]
    if data.empty:
        return pd.DataFrame(columns=output_columns)
    if (data["adjusted_close"] <= 0).any():
        raise ValueError("Adjusted closing prices must be positive.")
    if data.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker observations found.")

    price_table = data.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()
    if benchmark not in price_table.columns:
        raise ValueError(f"Benchmark {benchmark} was not found.")
    daily_returns = price_table.pct_change(fill_method=None)
    results = []
    for ticker in price_table.columns:
        series = price_table[ticker].dropna()
        if series.empty:
            continue
        ticker_returns = daily_returns[ticker].dropna()
        volatility = (
            float(ticker_returns.std(ddof=1) * np.sqrt(252) * 100)
            if len(ticker_returns) > 1 else np.nan
        )
        if ticker == benchmark:
            beta, observations = 1.0, int(len(ticker_returns))
        else:
            paired = pd.concat(
                [daily_returns[ticker].rename("asset"),
                 daily_returns[benchmark].rename("market")],
                axis=1, join="inner"
            ).dropna()
            observations, beta = int(len(paired)), np.nan
            if observations >= 2:
                variance = paired["market"].var(ddof=1)
                if variance > 0:
                    beta = float(paired["asset"].cov(paired["market"]) / variance)
        results.append({
            "ticker": str(ticker),
            "latest_price": float(series.iloc[-1]),
            "total_return_pct": float((series.iloc[-1] / series.iloc[0] - 1) * 100),
            "annualized_volatility_pct": volatility,
            "beta_vs_spy": float(beta) if pd.notna(beta) else np.nan,
            "start_date": str(series.index.min().date()),
            "end_date": str(series.index.max().date()),
            "observations": observations,
        })
    return pd.DataFrame(results, columns=output_columns).sort_values("ticker").reset_index(drop=True)


def normalized_performance(
    prices: pd.DataFrame, tickers: list[str] | None = None,
    start_date=None, end_date=None, base_value: float = 100.0
) -> pd.DataFrame:
    """Return chart-ready adjusted-price series rebased to base_value per ticker."""
    required = {"date", "ticker", "adjusted_close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Price data is missing columns: {sorted(missing)}")
    data = prices[["date", "ticker", "adjusted_close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["adjusted_close"] = pd.to_numeric(data["adjusted_close"], errors="coerce")
    data = data.dropna(subset=["date", "ticker", "adjusted_close"])
    if tickers is not None:
        data = data[data["ticker"].isin(tickers)]
    if start_date is not None:
        data = data[data["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        data = data[data["date"] <= pd.to_datetime(end_date)]
    data = data[data["adjusted_close"] > 0].sort_values(["ticker", "date"]).copy()
    if data.empty:
        data["normalized_value"] = pd.Series(dtype=float)
        return data
    first = data.groupby("ticker")["adjusted_close"].transform("first")
    data["normalized_value"] = data["adjusted_close"] / first * float(base_value)
    return data


def calculate_cycle_metrics(
    companies: pd.DataFrame, relationships: pd.DataFrame, confirmed_only: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Count simple directed cycles in the company network.

    A cycle is a structural pattern, not proof of wrongdoing or a cash-flow total.
    """
    named = prepare_relationships(companies, relationships, confirmed_only=confirmed_only)
    named = named.dropna(subset=["source_company", "target_company"])
    graph = nx.DiGraph()
    graph.add_nodes_from(companies["company_id"].dropna().astype(str))
    graph.add_edges_from(
        zip(named["source_entity_id"].astype(str), named["target_entity_id"].astype(str))
    )
    cycles = []
    for cycle in nx.simple_cycles(graph):
        if len(cycle) < 2:
            continue
        cycles.append(cycle)
    names = companies.set_index("company_id")["company_name"].to_dict()
    cycle_rows = []
    for cycle in cycles:
        cycle_rows.append({
            "cycle_length": len(cycle),
            "companies_in_cycle": " → ".join([str(names.get(node, node)) for node in cycle] + [str(names.get(cycle[0], cycle[0]))]),
            "company_ids": " → ".join(cycle + [cycle[0]]),
        })
    cycle_frame = pd.DataFrame(cycle_rows, columns=["cycle_length", "companies_in_cycle", "company_ids"])
    summary = {
        "company_count": int(graph.number_of_nodes()),
        "relationship_count": int(graph.number_of_edges()),
        "cycle_count": int(len(cycle_rows)),
        "companies_in_cycles": int(len({node for cycle in cycles for node in cycle})),
        "confirmed_only": bool(confirmed_only),
    }
    return cycle_frame.sort_values(["cycle_length", "companies_in_cycle"]).reset_index(drop=True) if not cycle_frame.empty else cycle_frame, summary


def failure_exposure(
    companies: pd.DataFrame,
    relationships: pd.DataFrame,
    failed_company_id,
    confirmed_only: bool = True,
) -> pd.DataFrame:
    """Describe structural reachability around a selected failed company."""
    company_ids = companies["company_id"].dropna().astype(str)
    failed_id = str(failed_company_id)
    if failed_id not in set(company_ids):
        raise ValueError(f"Failed company ID not found: {failed_company_id}")

    named = prepare_relationships(companies, relationships, confirmed_only=confirmed_only)
    named = named.dropna(subset=["source_company", "target_company"])
    graph = nx.DiGraph()
    graph.add_nodes_from(company_ids)
    graph.add_edges_from(
        zip(named["source_entity_id"].astype(str), named["target_entity_id"].astype(str))
    )
    if failed_id not in graph or graph.degree(failed_id) == 0:
        raise ValueError(f"Failed company is not in the selected graph: {failed_company_id}")

    from_failed = nx.single_source_shortest_path_length(graph, failed_id)
    to_failed = nx.single_source_shortest_path_length(graph.reverse(copy=False), failed_id)
    names = dict(
        zip(
            companies.loc[companies["company_id"].notna(), "company_id"].astype(str),
            companies.loc[companies["company_id"].notna(), "company_name"],
        )
    )
    direct_ids = {}
    for _, relationship in named.iterrows():
        source = str(relationship["source_entity_id"])
        target = str(relationship["target_entity_id"])
        if failed_id not in (source, target) or source == target:
            continue
        other = target if source == failed_id else source
        relationship_id = relationship.get("relationship_id")
        if pd.notna(relationship_id) and str(relationship_id).strip():
            direct_ids.setdefault(other, []).append(str(relationship_id))

    rows = []
    for company_id in sorted(set(from_failed) | set(to_failed)):
        if company_id == failed_id:
            continue
        paid_distance = from_failed.get(company_id)
        pays_distance = to_failed.get(company_id)
        distances = [distance for distance in (paid_distance, pays_distance) if distance is not None]
        if not distances:
            continue
        hops = min(distances)
        if pays_distance is not None and (paid_distance is None or pays_distance < paid_distance):
            link_type = "pays into the failed company"
        elif paid_distance is not None and (pays_distance is None or paid_distance < pays_distance):
            link_type = "is paid by the failed company"
        else:
            link_type = "both"
        color_group = "hop1" if hops == 1 else "hop2" if hops == 2 else "hop3_plus"
        rows.append({
            "company_id": company_id,
            "company_name": names.get(company_id, company_id),
            "hops": int(hops),
            "link_type": link_type,
            "direct_relationship_ids": "; ".join(direct_ids.get(company_id, [])),
            "color_group": color_group,
        })
    return pd.DataFrame(
        rows,
        columns=[
            "company_id", "company_name", "hops", "link_type",
            "direct_relationship_ids", "color_group",
        ],
    ).sort_values(["hops", "company_name"]).reset_index(drop=True) if rows else pd.DataFrame(
        columns=[
            "company_id", "company_name", "hops", "link_type",
            "direct_relationship_ids", "color_group",
        ]
    )


def cycles_through_company(
    companies: pd.DataFrame,
    relationships: pd.DataFrame,
    company_id,
    confirmed_only: bool = True,
) -> int:
    """Return the number of detected simple cycles containing a company."""
    if str(company_id) not in set(companies["company_id"].dropna().astype(str)):
        raise ValueError(f"Company ID not found: {company_id}")
    cycle_table, _ = calculate_cycle_metrics(
        companies, relationships, confirmed_only=confirmed_only
    )
    company_id = str(company_id)
    return int(
        sum(
            company_id in {part.strip() for part in str(value).split("→")}
            for value in cycle_table["company_ids"]
        )
    )


def failure_exposure_display(
    exposure: pd.DataFrame, failed_company_name: str
) -> pd.DataFrame:
    """Convert structural exposure data into labels suitable for display."""
    rows = []
    for _, row in exposure.iterrows():
        hops = int(row["hops"])
        if hops == 1:
            distance = "Directly linked"
        elif hops == 2:
            distance = "2 steps away"
        else:
            distance = "3 or more steps away"
        link_type = row["link_type"]
        if link_type == "pays into the failed company":
            linked_as = f"Sends money to {failed_company_name}"
        elif link_type == "is paid by the failed company":
            linked_as = f"Receives money from {failed_company_name}"
        else:
            linked_as = f"Both ways with {failed_company_name}"
        deal_ids = row.get("direct_relationship_ids", "")
        rows.append({
            "Company": row["company_name"],
            "Distance": distance,
            "How it is linked": linked_as,
            "Deal IDs": "" if pd.isna(deal_ids) else str(deal_ids),
            "_hops": hops,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["Company", "Distance", "How it is linked", "Deal IDs"])
    return (
        result.sort_values(["_hops", "Company"])
        [["Company", "Distance", "How it is linked", "Deal IDs"]]
        .reset_index(drop=True)
    )


def failure_summary_text(
    exposure: pd.DataFrame, failed_company_name: str, cycle_count: int
) -> str:
    """Build a plain-English structural summary of failure exposure."""
    direct_count = int((exposure["hops"] == 1).sum())
    two_step_count = int((exposure["hops"] == 2).sum())
    three_plus_count = int((exposure["hops"] >= 3).sum())

    if direct_count == 0:
        direct_phrase = "no companies are directly linked to it"
    else:
        direct_noun = "company" if direct_count == 1 else "companies"
        direct_verb = "is" if direct_count == 1 else "are"
        direct_phrase = f"{direct_count} {direct_noun} {direct_verb} directly linked to it"

    clauses = [f"If {failed_company_name} failed, {direct_phrase}"]
    if two_step_count:
        two_noun = "company" if two_step_count == 1 else "companies"
        two_verb = "is" if two_step_count == 1 else "are"
        prefix = " and " if direct_count else ", and "
        clauses.append(f"{prefix}{two_step_count} more {two_noun} {two_verb} two steps away")
    if three_plus_count:
        prefix = " and " if two_step_count or direct_count else ", and "
        clauses.append(f"{prefix}{three_plus_count} more are three or more steps away")

    cycle_noun = "loop" if cycle_count == 1 else "loops"
    return (
        "".join(clauses)
        + f". {cycle_count} recorded {cycle_noun} pass through it. "
        "This is a structural view, not a forecast."
    )



# ===========================================================================
# Новые функции: риск-доходность, Sharpe, просадка, корзины компаний
# ===========================================================================

TRADING_DAYS = 252  # число торговых дней в году, как в calculate_market_metrics


def sharpe_ratio(returns: pd.Series, risk_free_annual_pct: float = 0.0) -> float:
    """Годовой коэффициент Шарпа по дневным доходностям.

    Дневная безрисковая ставка = годовая ставка (в процентах) / 100 / 252.
    Sharpe = среднее(доходность - безрисковая) / стандартное отклонение * sqrt(252).
    Возвращает NaN, если наблюдений меньше 2 или разброс равен нулю.
    """
    clean = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if len(clean) < 2:
        return float("nan")
    daily_rf = risk_free_annual_pct / 100.0 / TRADING_DAYS
    excess = clean - daily_rf
    spread = excess.std(ddof=1)
    if not spread > 0:
        return float("nan")
    return float(excess.mean() / spread * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    """Максимальная просадка (отрицательная доля, например -0.2 = -20%).

    Считается по накопленной стоимости портфеля, которая стартует с 1.0.
    """
    clean = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + clean.to_numpy())])
    peak = np.maximum.accumulate(wealth)
    return float((wealth / peak - 1.0).min())


def beta_vs_benchmark(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Бета = ковариация(актив, рынок) / дисперсия(рынка), по общим датам."""
    paired = pd.concat(
        [pd.Series(asset_returns).rename("asset"), pd.Series(benchmark_returns).rename("market")],
        axis=1, join="inner",
    ).dropna()
    if len(paired) < 2:
        return float("nan")
    variance = paired["market"].var(ddof=1)
    if not variance > 0:
        return float("nan")
    return float(paired["asset"].cov(paired["market"]) / variance)


def cycle_member_tickers(
    companies: pd.DataFrame, relationships: pd.DataFrame, confirmed_only: bool = True
) -> list[str]:
    """Тикеры ПУБЛИЧНЫХ компаний, которые входят хотя бы в один найденный цикл."""
    cycle_table, _ = calculate_cycle_metrics(companies, relationships, confirmed_only=confirmed_only)
    member_ids: set[str] = set()
    for text in cycle_table["company_ids"]:
        member_ids.update(part.strip() for part in str(text).split("→"))
    members = companies[companies["company_id"].astype(str).isin(member_ids)]
    tickers = members["public_ticker"].dropna().astype(str).str.strip()
    return sorted({ticker for ticker in tickers if ticker})


def compare_groups(
    prices: pd.DataFrame,
    groups: dict,
    benchmark: str = "SPY",
    start_date=None,
    end_date=None,
    risk_free_annual_pct: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """Сравнить равновзвешенные корзины акций между собой и с бенчмарком.

    groups: словарь вида {"Cycle group": ["NVDA", "MSFT"], "Comparison group": [...]}.
    Корзина пересобирается в равные веса каждый день (дневная ребалансировка).
    Все корзины и бенчмарк считаются на ОДНИХ И ТЕХ ЖЕ датах: берутся только дни,
    когда есть цены у всех используемых тикеров. Число отброшенных дней возвращается в info.
    """
    required = {"date", "ticker", "adjusted_close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Price data is missing columns: {sorted(missing)}")
    if not groups:
        raise ValueError("At least one group is required.")
    for name, tickers in groups.items():
        if not tickers:
            raise ValueError(f"Group '{name}' has no tickers.")
        if benchmark in name:
            raise ValueError("Group names must not contain the benchmark ticker.")

    data = prices[["date", "ticker", "adjusted_close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["ticker"] = data["ticker"].astype("string").str.strip()
    data["adjusted_close"] = pd.to_numeric(data["adjusted_close"], errors="coerce")
    data = data.dropna(subset=["date", "ticker", "adjusted_close"])
    if (data["adjusted_close"] <= 0).any():
        raise ValueError("Adjusted closing prices must be positive.")
    if data.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker observations found.")
    if start_date is not None:
        data = data[data["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        data = data[data["date"] <= pd.to_datetime(end_date)]

    table = data.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()
    needed = sorted({benchmark} | {t for tickers in groups.values() for t in tickers})
    absent = [t for t in needed if t not in table.columns]
    if absent:
        raise ValueError(f"Tickers not found in price data: {absent}")

    table = table[needed]
    first_dates = {t: table[t].first_valid_index() for t in needed}
    benchmark_dates = int(table[benchmark].notna().sum())
    common = table.dropna()
    if len(common) < 3:
        raise ValueError("Not enough overlapping dates to compare the groups.")
    daily = common.pct_change().dropna()

    series = {name: daily[list(tickers)].mean(axis=1) for name, tickers in groups.items()}
    series[benchmark] = daily[benchmark]
    members = {**{name: list(t) for name, t in groups.items()}, benchmark: [benchmark]}

    rows = []
    for name, returns in series.items():
        rows.append({
            "group": name,
            "tickers": ", ".join(members[name]),
            "return_observations": int(len(returns)),
            "total_return_pct": float(((1 + returns).prod() - 1) * 100),
            "annualized_volatility_pct": float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100),
            "sharpe_ratio": sharpe_ratio(returns, risk_free_annual_pct),
            "beta_vs_spy": 1.0 if name == benchmark else beta_vs_benchmark(returns, daily[benchmark]),
            "max_drawdown_pct": max_drawdown(returns) * 100,
        })
    limiting = max(first_dates, key=lambda t: first_dates[t])
    info = {
        "window_start": str(common.index.min().date()),
        "window_end": str(common.index.max().date()),
        "price_dates_used": int(len(common)),
        "price_dates_dropped": benchmark_dates - int(len(common)),
        "limiting_ticker": limiting,
        "limiting_start": str(first_dates[limiting].date()),
        "risk_free_annual_pct": float(risk_free_annual_pct),
    }
    info["daily_returns"] = pd.DataFrame(series)
    return pd.DataFrame(rows), info


def worst_day_comovement(
    prices: pd.DataFrame,
    failed_ticker: str,
    worst_n: int = 20,
    benchmark: str = "SPY",
) -> tuple[pd.DataFrame, dict]:
    """Measure other-ticker returns on a ticker's worst common-window days."""
    required = {"date", "ticker", "adjusted_close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Price data is missing columns: {sorted(missing)}")
    data = prices[["date", "ticker", "adjusted_close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["ticker"] = data["ticker"].astype("string").str.strip()
    data["adjusted_close"] = pd.to_numeric(data["adjusted_close"], errors="coerce")
    data = data.dropna(subset=["date", "ticker", "adjusted_close"])
    data = data[data["ticker"] != ""]
    if data.empty or failed_ticker not in set(data["ticker"]):
        raise ValueError(f"Failed ticker {failed_ticker} was not found in price data.")
    if benchmark not in set(data["ticker"]):
        raise ValueError(f"Benchmark {benchmark} was not found in price data.")
    if (data["adjusted_close"] <= 0).any():
        raise ValueError("Adjusted closing prices must be positive.")
    if data.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker observations found.")

    table = data.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()
    common = table.dropna()
    daily_returns = common.pct_change(fill_method=None).dropna()
    if len(daily_returns) < worst_n:
        raise ValueError(f"Fewer than {worst_n} common return days are available.")

    failed_returns = daily_returns[failed_ticker]
    worst_dates = failed_returns.nsmallest(worst_n).index
    other_dates = daily_returns.index.difference(worst_dates)
    rows = []
    for ticker in daily_returns.columns:
        if ticker == failed_ticker:
            continue
        selected = daily_returns.loc[worst_dates, ticker]
        other = daily_returns.loc[other_dates, ticker]
        rows.append({
            "ticker": str(ticker),
            "avg_return_on_those_days_pct": float(selected.mean() * 100),
            "avg_return_other_days_pct": float(other.mean() * 100) if len(other) else np.nan,
            "days_down_share_pct": float((selected < 0).mean() * 100),
            "days_used": int(len(selected)),
        })
    result = pd.DataFrame(
        rows,
        columns=[
            "ticker", "avg_return_on_those_days_pct", "avg_return_other_days_pct",
            "days_down_share_pct", "days_used",
        ],
    ).sort_values("ticker").reset_index(drop=True)
    summary = {
        "failed_ticker": str(failed_ticker),
        "failed_ticker_avg_return_pct": float(failed_returns.loc[worst_dates].mean() * 100),
        "window_start": str(common.index.min().date()),
        "window_end": str(common.index.max().date()),
        "days_used": int(len(daily_returns)),
        "worst_days_used": int(len(worst_dates)),
    }
    return result, summary


def worst_day_display(comovement: pd.DataFrame) -> pd.DataFrame:
    """Return plain-language, formatted columns for worst-day co-movement."""
    return pd.DataFrame({
        "Stock": comovement["ticker"],
        "Average move on those days": comovement["avg_return_on_those_days_pct"].map(
            lambda value: f"{value:.1f}%"
        ),
        "Average move on all other days": comovement["avg_return_other_days_pct"].map(
            lambda value: f"{value:.1f}%" if pd.notna(value) else "N/A"
        ),
        "Share of those days it also fell": comovement["days_down_share_pct"].map(
            lambda value: f"{value:.1f}%"
        ),
    })


def worst_day_text(comovement: pd.DataFrame, summary: dict) -> str:
    """Describe worst-day co-movement in one readable sentence."""
    ticker = summary["failed_ticker"]
    days = summary["worst_days_used"]
    failed_average = summary["failed_ticker_avg_return_pct"]
    parts = [
        f"{row.ticker} moved on average {row.avg_return_on_those_days_pct:.1f}%"
        for row in comovement.itertuples()
    ]
    if len(parts) == 1:
        other_text = parts[0]
    elif len(parts) == 2:
        other_text = " and ".join(parts)
    else:
        other_text = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return (
        f"On the {days} days when {ticker} fell the most (average {failed_average:.1f}%), "
        f"{other_text}."
    )



    
def growth_of_base(daily_returns: pd.DataFrame, start_date, base_value: float = 100.0) -> pd.DataFrame:
    """Рост base_value по дневным доходностям каждой серии (для графика).

    В день start_date значение равно base_value; дальше оно умножается на (1 + доходность).
    Возвращает длинную таблицу с колонками: date, series, value.
    """
    if daily_returns.empty:
        raise ValueError("No daily returns to chart.")
    growth = (1.0 + daily_returns).cumprod() * float(base_value)
    start_row = pd.DataFrame(
        [[float(base_value)] * growth.shape[1]],
        columns=growth.columns,
        index=[pd.to_datetime(start_date)],
    )
    growth = pd.concat([start_row, growth])
    growth.index.name = "date"
    return growth.reset_index().melt(id_vars="date", var_name="series", value_name="value")

    

def price_data_summary(prices: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на тикер: первая и последняя дата, число наблюдений, валюта, источник, дата выгрузки."""
    required = {"ticker", "date", "adjusted_close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Price data is missing columns: {sorted(missing)}")
    grouped = prices.groupby("ticker")
    summary = grouped.agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        observations=("adjusted_close", "count"),
    ).reset_index()
    for column, label, limit in [
        ("currency", "currency", None),
        ("source", "source", 2),
        ("retrieved_date", "pull_date", None),
    ]:
        if column in prices.columns:
            texts = grouped[column].agg(
                lambda s, n=limit: ", ".join(sorted(set(s.dropna().astype(str)))[:n])
            )
            summary = summary.merge(texts.rename(label).reset_index(), on="ticker", how="left")
    return summary


def describe_cycle_amounts(
    companies: pd.DataFrame,
    relationships: pd.DataFrame,
    cycle_table: pd.DataFrame,
    confirmed_only: bool = True,
    data_dictionary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add one readable, per-step amount description to the cycle table.

    Amounts are deliberately kept separate: different relationship rows are
    descriptions of links, not values that should be added together.
    """
    if data_dictionary is None or data_dictionary.empty:
        raise ValueError("The data dictionary is required to identify the amount unit.")
    dictionary_rows = data_dictionary[
        data_dictionary["field_or_term"].astype(str).str.strip().eq("amount (relationships)")
    ]
    if dictionary_rows.empty:
        raise ValueError("The data dictionary does not define the amount unit.")
    amount_unit = str(dictionary_rows.iloc[0]["unit_or_allowed_values"]).strip()
    if not amount_unit or amount_unit.lower() == "nan":
        raise ValueError("The amount unit in the data dictionary is unclear.")

    required = {"source_entity_id", "target_entity_id", "evidence_status"}
    missing = required - set(relationships.columns)
    if missing:
        raise ValueError(f"Relationships table is missing columns: {sorted(missing)}")
    if {"company_id", "company_name"} - set(companies.columns):
        raise ValueError("Companies table must contain company_id and company_name.")

    names = companies.set_index("company_id")["company_name"].to_dict()
    selected = relationships.copy()
    if confirmed_only:
        selected = selected[
            selected["evidence_status"].astype(str).str.lower().eq("confirmed")
        ]

    def format_cycle_amount(value, currency) -> str:
        if pd.isna(value):
            return "Not disclosed"
        currency_text = "" if pd.isna(currency) else str(currency).strip()
        formatted = format_money(value, currency_text or "USD")
        if currency_text.upper() == "USD" and formatted.startswith("USD "):
            formatted = "$" + formatted[4:]
        number_text = formatted[1:] if formatted.startswith("$") else formatted.split(" ", 1)[-1]
        suffix = number_text[-1] if number_text[-1:] in {"T", "B", "M"} else ""
        numeric_text = number_text[:-1] if suffix else number_text
        try:
            numeric_text = f"{float(numeric_text.replace(',', '')):.1f}"
            formatted = ("$" if formatted.startswith("$") else f"{currency_text} ") + numeric_text + suffix
        except ValueError:
            pass
        return formatted

    def format_amount(value) -> str:
        if pd.isna(value):
            return "amount not disclosed"
        number = float(value)
        return f"{number:,.0f}" if number.is_integer() else f"{number:,.2f}".rstrip("0").rstrip(".")

    descriptions = []
    for _, cycle_row in cycle_table.iterrows():
        ids = [part.strip() for part in str(cycle_row["company_ids"]).split("→")]
        steps = []
        for source_id, target_id in zip(ids, ids[1:]):
            matches = selected[
                selected["source_entity_id"].astype(str).eq(source_id)
                & selected["target_entity_id"].astype(str).eq(target_id)
            ]
            for _, relationship in matches.iterrows():
                currency = relationship.get("currency", "")
                currency_text = "" if pd.isna(currency) else str(currency).strip()
                relationship_type = relationship.get("relationship_type", "")
                type_text = "" if pd.isna(relationship_type) else str(relationship_type).strip()
                detail = f"{format_amount(relationship.get('amount', pd.NA))}"
                if currency_text:
                    detail += f" {currency_text}"
                elif amount_unit:
                    detail += f" ({amount_unit})"
                if type_text:
                    detail += f" ({type_text})"
                steps.append(
                    f"{names.get(source_id, source_id)} → {names.get(target_id, target_id)}: {detail}"
                )
        descriptions.append("; ".join(steps) if steps else "No matching relationships")

    display_cycle_table = cycle_table.copy()
    display_cycle_table["company_ids"] = display_cycle_table["company_ids"].astype(str).str.replace(
        chr(8594), "".join(chr(value) for value in (1074, 8224, 8217)), regex=False
    )
    display_descriptions = []
    for _, cycle_row in display_cycle_table.iterrows():
        ids = [part.strip() for part in str(cycle_row["company_ids"]).split("в†’")]
        display_steps = []
        for source_id, target_id in zip(ids, ids[1:]):
            matches = selected[
                selected["source_entity_id"].astype(str).eq(source_id)
                & selected["target_entity_id"].astype(str).eq(target_id)
            ]
            relationship_steps = []
            for _, relationship in matches.iterrows():
                currency = relationship.get("currency", "")
                currency_text = "" if pd.isna(currency) else str(currency).strip()
                notes = relationship.get("notes", "")
                notes_text = "" if pd.isna(notes) else str(notes)
                amount = relationship.get("amount", pd.NA)
                if pd.isna(amount) and "shares" in notes_text.lower():
                    amount_text = "in shares, no dollar value"
                else:
                    amount_text = format_cycle_amount(amount, currency_text)
                if "up to" in notes_text.lower() and amount_text not in {
                    "Not disclosed", "in shares, no dollar value"
                }:
                    amount_text = f"up to {amount_text}"
                relationship_type = relationship.get("relationship_type", "")
                type_text = "" if pd.isna(relationship_type) else str(relationship_type).strip()
                type_text = type_text.lower().replace("_", " ")
                if type_text == "supplier customer":
                    type_text = "purchase"
                relationship_steps.append(
                    f"{amount_text}, {type_text}" if type_text else amount_text
                )
            display_steps.extend(relationship_steps)
        numbered_steps = [
            f"{index}) {step}" for index, step in enumerate(display_steps, start=1)
        ]
        display_descriptions.append(
            " | ".join(numbered_steps) if numbered_steps else "No matching relationships"
        )

    result = cycle_table[["cycle_length", "companies_in_cycle"]].copy()
    result["Amounts by step (same order as the cycle)"] = display_descriptions
    return result
