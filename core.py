
import pandas as pd


def validate_data(
    companies: pd.DataFrame,
    relationships: pd.DataFrame,
) -> dict:
    """Check company IDs and relationship references."""
    required_company_columns = {"company_id", "company_name"}
    required_relationship_columns = {
        "relationship_id",
        "source_entity_id",
        "target_entity_id",
        "evidence_status",
    }

    missing_company_columns = required_company_columns - set(companies.columns)
    missing_relationship_columns = (
        required_relationship_columns - set(relationships.columns)
    )

    if missing_company_columns:
        raise ValueError(
            f"Companies table is missing columns: {sorted(missing_company_columns)}"
        )

    if missing_relationship_columns:
        raise ValueError(
            "Relationships table is missing columns: "
            f"{sorted(missing_relationship_columns)}"
        )

    company_ids = set(companies["company_id"].dropna())

    return {
        "company_rows": len(companies),
        "relationship_rows": len(relationships),
        "duplicate_company_ids": int(
            companies["company_id"].duplicated().sum()
        ),
        "unknown_source_ids": int(
            (~relationships["source_entity_id"].isin(company_ids)).sum()
        ),
        "unknown_target_ids": int(
            (~relationships["target_entity_id"].isin(company_ids)).sum()
        ),
        "missing_company_cells": int(companies.isna().sum().sum()),
        "missing_relationship_cells": int(
            relationships.isna().sum().sum()
        ),
        "evidence_status_counts": {
            str(key): int(value)
            for key, value in relationships["evidence_status"]
            .value_counts(dropna=False)
            .items()
        },
    }


def prepare_relationships(
    companies: pd.DataFrame,
    relationships: pd.DataFrame,
    confirmed_only: bool = True,
) -> pd.DataFrame:
    """Attach company names to relationships without changing source data."""
    company_lookup = companies[["company_id", "company_name"]].copy()

    if company_lookup["company_id"].duplicated().any():
        raise ValueError("Company IDs must be unique before joining relationships.")

    result = relationships.copy()

    if confirmed_only:
        result = result.loc[
            result["evidence_status"].eq("confirmed")
        ].copy()

    result = result.merge(
        company_lookup.rename(
            columns={
                "company_id": "source_entity_id",
                "company_name": "source_company",
            }
        ),
        on="source_entity_id",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        company_lookup.rename(
            columns={
                "company_id": "target_entity_id",
                "company_name": "target_company",
            }
        ),
        on="target_entity_id",
        how="left",
        validate="many_to_one",
    )

    return result.reset_index(drop=True)


def missing_values_report(
    companies: pd.DataFrame,
    relationships: pd.DataFrame,
    financials: pd.DataFrame,
) -> pd.DataFrame:
    """Report missing cells by table and column."""
    reports = []

    for table_name, frame in [
        ("Companies", companies),
        ("Relationships", relationships),
        ("Financials", financials),
    ]:
        for column in frame.columns:
            missing_count = int(frame[column].isna().sum())

            reports.append(
                {
                    "table": table_name,
                    "column": column,
                    "missing_count": missing_count,
                    "missing_percent": round(
                        100 * missing_count / len(frame), 2
                    ) if len(frame) else 0.0,
                }
            )

    return pd.DataFrame(reports)


import numpy as np


def calculate_market_metrics(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """Calculate return, annualized volatility and beta vs benchmark."""

    required = {"date", "ticker", "adjusted_close"}
    missing = required - set(prices.columns)

    if missing:
        raise ValueError(
            f"Price data is missing columns: {sorted(missing)}"
        )

    data = prices[["date", "ticker", "adjusted_close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["adjusted_close"] = pd.to_numeric(
        data["adjusted_close"], errors="coerce"
    )
    data = data.dropna(
        subset=["date", "ticker", "adjusted_close"]
    )

    if (data["adjusted_close"] <= 0).any():
        raise ValueError("Adjusted closing prices must be positive.")

    if data.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker observations found.")

    price_table = data.pivot(
        index="date",
        columns="ticker",
        values="adjusted_close",
    ).sort_index()

    if benchmark not in price_table.columns:
        raise ValueError(f"Benchmark {benchmark} was not found.")

    daily_returns = price_table.pct_change(fill_method=None)
    benchmark_returns = daily_returns[benchmark]
    results = []

    for ticker in price_table.columns:
        if ticker == benchmark:
            continue

        paired = pd.concat(
            [daily_returns[ticker], benchmark_returns],
            axis=1,
            keys=["stock", "benchmark"],
        ).dropna()

        if len(paired) < 2:
            continue

        stock_returns = paired["stock"]
        market_returns = paired["benchmark"]

        market_variance = market_returns.var(ddof=1)

        beta = (
            stock_returns.cov(market_returns) / market_variance
            if market_variance > 0
            else np.nan
        )

        first_date = paired.index[0]
        last_date = paired.index[-1]

        first_price = price_table.loc[first_date, ticker]
        last_price = price_table.loc[last_date, ticker]

        results.append(
            {
                "ticker": ticker,
                "period_start": first_date.date(),
                "period_end": last_date.date(),
                "observations": len(paired),
                "total_return_pct": (
                    (last_price / first_price - 1) * 100
                ),
                "annualized_volatility_pct": (
                    stock_returns.std(ddof=1) * np.sqrt(252) * 100
                ),
                "beta_vs_spy": beta,
            }
        )

    return pd.DataFrame(results).sort_values(
        "total_return_pct",
        ascending=False,
    ).reset_index(drop=True)
