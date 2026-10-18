"""Pure calculations for Circular Alpha. No UI and no network access."""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


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

    result = cycle_table[["cycle_length", "companies_in_cycle"]].copy()
    result["relationship_amounts"] = descriptions
    return result
