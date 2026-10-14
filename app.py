

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

from data import (
    load_companies,
    load_relationships,
    load_financials,
    load_price_history,
)
from core import (
    validate_data,
    prepare_relationships,
    calculate_market_metrics,
)


st.set_page_config(
    page_title="Circular Alpha",
    page_icon="📊",
    layout="wide",
)

st.title("Circular Alpha")
st.write(
    "Explore the financial resilience and business relationships "
    "of companies in the AI infrastructure ecosystem."
)


@st.cache_data
def get_data():
    companies = load_companies()
    relationships = load_relationships()
    financials = load_financials()
    prices = load_price_history()

    validation = validate_data(companies, relationships)
    confirmed = prepare_relationships(
        companies, relationships, confirmed_only=True
    )
    market_metrics = calculate_market_metrics(prices)

    return (
        companies,
        relationships,
        financials,
        prices,
        validation,
        confirmed,
        market_metrics,
    )


try:
    (
        companies,
        relationships,
        financials,
        prices,
        validation,
        confirmed,
        market_metrics,
    ) = get_data()

    # Main dashboard metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Companies", len(companies))
    col2.metric("All relationships", len(relationships))
    col3.metric("Confirmed relationships", len(confirmed))
    col4.metric(
        "Companies with financial data",
        financials["company_id"].nunique(),
    )

    st.divider()

    # Data quality
    st.header("Data quality")

    q1, q2, q3 = st.columns(3)
    q1.metric(
        "Duplicate company IDs",
        validation.get("duplicate_company_ids", 0),
    )
    q2.metric(
        "Unknown source IDs",
        validation.get("unknown_source_ids", 0),
    )
    q3.metric(
        "Unknown target IDs",
        validation.get("unknown_target_ids", 0),
    )

    st.success("Structural data checks completed.")
    st.caption(
        "Missing values are not automatically errors. "
        "Their meaning depends on the field and its source."
    )

    st.divider()

    # Company explorer
    st.header("Company explorer")

    if "sector" in companies.columns:
        sectors = sorted(
            companies["sector"].dropna().astype(str).unique().tolist()
        )
        selected_sector = st.selectbox(
            "Filter by sector",
            ["All"] + sectors,
        )

        company_view = companies.copy()
        if selected_sector != "All":
            company_view = company_view[
                company_view["sector"].astype(str) == selected_sector
            ]
    else:
        company_view = companies.copy()

    st.dataframe(
        company_view,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    # Relationship explorer
    st.header("Relationship explorer")

    include_provisional = st.checkbox(
        "Include provisional relationships",
        value=False,
    )

    relationship_view = prepare_relationships(
        companies,
        relationships,
        confirmed_only=not include_provisional,
    )

    st.dataframe(
        relationship_view,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Confirmed and provisional relationships are distinguished "
        "using the evidence status in the source dataset."
    )

    st.divider()

    # Financial data
    st.header("Financial data")

    st.dataframe(
        financials,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Financial values are shown as supplied in the source files. "
        "Check reporting periods and sources before comparing companies."
    )

    st.divider()

    # Historical market performance
    st.header("Market performance")

    st.write(
        "Historical price-based metrics for available publicly traded "
        "companies. SPY is used as the market benchmark."
    )

    st.caption(
        "These historical measures describe past performance and risk; "
        "they do not predict future returns. CRWV has a shorter price history."
    )

    if not market_metrics.empty:
        display_metrics = market_metrics.copy()

        percent_columns = [
            "total_return_pct",
            "annualized_volatility_pct",
        ]
        for column in percent_columns:
            if column in display_metrics.columns:
                display_metrics[column] = display_metrics[column].round(2)

        if "beta_vs_spy" in display_metrics.columns:
            display_metrics["beta_vs_spy"] = (
                display_metrics["beta_vs_spy"].round(2)
            )

        st.subheader("Risk and return summary")

        st.dataframe(
            display_metrics,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Total return and annualized volatility are percentages. "
            "Beta compares each stock's daily returns with SPY."
        )

        
        # Select tickers for the normalized performance chart
        available_tickers = sorted(
            prices["ticker"].dropna().unique().tolist()
        )

        default_tickers = [
            ticker
            for ticker in ["NVDA", "AMD", "GOOGL", "SPY"]
            if ticker in available_tickers
        ]

        selected_tickers = st.multiselect(
            "Choose tickers to compare",
            options=available_tickers,
            default=default_tickers,
        )

        if selected_tickers:
            chart_prices = prices[
                prices["ticker"].isin(selected_tickers)
            ].copy()

            chart_data = chart_prices.pivot(
                index="date",
                columns="ticker",
                values="adjusted_close",
            ).sort_index()

            # Normalize each ticker to 100 at its first available date
            base_values = chart_data.apply(
                lambda column: (
                    column.dropna().iloc[0]
                    if not column.dropna().empty
                    else float("nan")
                )
            )

            normalized_data = chart_data.divide(
                base_values, axis="columns"
            ) * 100

            st.subheader("Normalized investment performance")
            st.write(
                "Each ticker starts at 100 on its first available date. "
                "This allows comparison of relative price growth."
            )

            st.line_chart(
                normalized_data,
                width="stretch",
            )

            st.caption(
                "A value of 150 represents a 50% increase from the "
                "starting price; 80 represents a 20% decrease. "
                "Different tickers may have different start dates. "
                "This chart excludes dividends and is not a forecast."
            )
        else:
            st.info("Select at least one ticker to display the chart.")

    else:
        st.warning("No market metrics are available to display.")

except Exception as error:
    st.error("The app could not load or process the data.")
    st.exception(error)
