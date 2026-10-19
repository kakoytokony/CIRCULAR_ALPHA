"""Circular Alpha — local Streamlit MVP. Calculations live in core.py."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from data import load_all_prices, load_companies, load_dictionary, load_financials, load_relationships
from core import (
    calculate_cycle_metrics,
    calculate_market_metrics,
    compare_groups,
    company_explorer_tables,
    cycle_member_tickers,
    cycles_through_company,
    data_quality_summary,
    describe_cycle_amounts,
    arc_midpoint,
    curved_arc_points,
    failure_exposure,
    failure_exposure_display,
    failure_summary_text,
    format_money,
    growth_of_base,
    line_midpoint,
    missing_values_report,
    normalized_performance,
    prepare_relationships,
    price_data_summary,
    validate_data,
    worst_day_comovement,
    worst_day_display,
    worst_day_text,
)

st.set_page_config(page_title="Circular Alpha | QIM", page_icon="🔗", layout="wide")

st.title("🔗 Circular Alpha")
st.caption("QIM MVP | Public-company relationships, circularity patterns and market risk")
st.info(
    "**Research question:** Do recorded directed financial relationships form circular "
    "patterns among selected public companies, and what market-risk context helps an investor "
    "interpret those companies? A detected cycle is a pattern for review, not proof of wrongdoing "
    "or a forecast."
)

try:
    companies = load_companies()
    relationships = load_relationships()
    financials = load_financials()
    prices = load_all_prices()
    dictionary = load_dictionary()
except Exception as exc:
    st.error(f"Could not load project data: {exc}")
    st.info("Check the saved CSV files in the project root or data/raw/.")
    st.stop()

try:
    validation = validate_data(companies, relationships)
except ValueError as exc:
    st.error(f"Data validation failed: {exc}")
    st.stop()

if prices.empty:
    st.error("No valid saved adjusted-price data was found.")
    st.stop()

# Source status: current app deliberately uses local saved snapshots, not live API data.
st.caption("Data mode: saved local CSV snapshots (offline-capable). No live download is attempted by this MVP.")

# Top-level summary metrics
status_counts = relationships["evidence_status"].fillna("missing").astype(str).str.lower().value_counts()
confirmed_count = int(status_counts.get("confirmed", 0))
provisional_count = int(status_counts.get("provisional", 0))
cycle_table_confirmed, cycle_summary = calculate_cycle_metrics(companies, relationships, confirmed_only=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Companies", f"{validation['company_rows']:,}")
c2.metric("Recorded relationships", f"{validation['relationship_rows']:,}")
c3.metric("Confirmed relationships", f"{confirmed_count:,}")
c4.metric("Confirmed-link cycles", f"{cycle_summary['cycle_count']:,}")
st.caption("Cycle counts depend on the recorded links and evidence filter. They are not a measure of illegality or financial loss.")

# Primary feature: graph-cycle explorer
st.divider()
st.header("1. Circular relationship explorer")
confirmed_only = st.toggle("Use confirmed relationships only", value=True)
cycle_table, cycle_summary = calculate_cycle_metrics(companies, relationships, confirmed_only=confirmed_only)
left, right = st.columns([1, 2])
with left:
    st.metric("Directed cycles found", cycle_summary["cycle_count"])
    st.metric("Companies participating in cycles", cycle_summary["companies_in_cycles"])
    st.metric("Links in selected network", cycle_summary["relationship_count"])
    st.write(
        "A directed cycle exists when following the direction of recorded relationships "
        "returns to the starting company. This MVP counts simple cycles in the selected graph. "
        "Cycles may reflect commercial interdependence and require source-level review."
    )
with right:
    if cycle_table.empty:
        st.info("No directed cycles were found with the current evidence filter.")
    else:
        cycle_display = describe_cycle_amounts(
            companies,
            relationships,
            cycle_table,
            confirmed_only=confirmed_only,
            data_dictionary=dictionary,
        )
        st.dataframe(cycle_display, use_container_width=True, hide_index=True)
        st.caption("Cycle order is a graph representation; it does not imply that money moved in a closed loop at the same time. Amounts are as announced and are not additive; many relationships have no disclosed amount.")

st.subheader("Relationship evidence")
relationship_view = prepare_relationships(companies, relationships, confirmed_only=confirmed_only)
show_cols = [
    col for col in [
        "relationship_id", "source_company", "target_company", "relationship_type",
        "announcement_date", "amount", "currency", "evidence_status", "source_title",
        "source_url", "notes",
    ] if col in relationship_view.columns
]
st.dataframe(relationship_view[show_cols], use_container_width=True, hide_index=True)
st.caption("Amounts can represent ceilings, commitments or purchases; they are not automatically realised cash and must not be added across different relationship types.")


# Interactive company network — blue company nodes and relationship lines.
st.divider()
st.header("Company relationship network")
st.write(
    "Explore reported investments, supplier relationships, and other links between companies. "
    "Hover over a blue company node for its connection count, or hover near the middle of a line "
    "to inspect the relationship type and evidence status."
)
stress_company_options = ["None"] + companies["company_name"].dropna().astype(str).sort_values().tolist()
selected_stress_company = st.selectbox(
    "Stress test: if this company fails",
    options=stress_company_options,
    index=0,
    key="network_stress_company",
)
if selected_stress_company != "None":
    stress_company_ids = dict(
        zip(companies["company_name"].astype(str), companies["company_id"])
    )
    selected_stress_company_id = stress_company_ids[selected_stress_company]
    try:
        failure_table = failure_exposure(
            companies, relationships, selected_stress_company_id, confirmed_only=False
        )
    except ValueError:
        failure_table = None
    failure_colors_by_name = {} if failure_table is None else dict(
        zip(failure_table["company_name"], failure_table["color_group"])
    )
    failure_colors_by_name[selected_stress_company] = "failed"
    stress_cycle_count = None if failure_table is None else cycles_through_company(
        companies, relationships, selected_stress_company_id, confirmed_only=False
    )
    failure_display_table = None if failure_table is None else failure_exposure_display(
        failure_table, selected_stress_company
    )
    failure_summary = None if failure_table is None else failure_summary_text(
        failure_table, selected_stress_company, stress_cycle_count
    )
else:
    failure_table = None
    failure_colors_by_name = {}
    stress_cycle_count = None
    failure_display_table = None
    failure_summary = None

network_data = prepare_relationships(companies, relationships, confirmed_only=False)
network_data = network_data.dropna(subset=["source_company", "target_company"]).copy()
status_options = sorted(network_data["evidence_status"].dropna().astype(str).unique())
selected_network_statuses = st.multiselect(
    "Filter network by evidence status",
    options=status_options,
    default=status_options,
    key="network_status_filter",
)
network_data = network_data[
    network_data["evidence_status"].astype(str).isin(selected_network_statuses)
].copy()

if network_data.empty:
    st.info("No relationships match the selected filters.")
else:
    graph = nx.DiGraph()
    for _, row in network_data.iterrows():
        source, target = row["source_company"], row["target_company"]
        if pd.notna(source) and pd.notna(target):
            graph.add_edge(str(source), str(target))

    positions = nx.spring_layout(graph, seed=42, k=1.2)
    edge_x, edge_y = [], []
    hover_x, hover_y, hover_text = [], [], []
    for _, row in network_data.iterrows():
        source, target = str(row["source_company"]), str(row["target_company"])
        if source not in positions or target not in positions:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        midpoint_x, midpoint_y = line_midpoint(x0, y0, x1, y1)
        hover_x.append(midpoint_x)
        hover_y.append(midpoint_y)
        details = [
            f"<b>{source} → {target}</b>",
            f"Relationship: {row.get('relationship_type', 'N/A')}",
            f"Evidence status: {row.get('evidence_status', 'N/A')}",
        ]
        rid = row.get("relationship_id", "")
        if pd.notna(rid) and str(rid).strip():
            details.append(f"ID: {rid}")
        amount, currency = row.get("amount", float("nan")), row.get("currency", "USD")
        if pd.notna(amount):
            try:
                details.append(f"Amount: {format_money(float(amount), str(currency) if pd.notna(currency) else 'USD')}")
            except (TypeError, ValueError):
                pass
        title, url = row.get("source_title", ""), row.get("source_url", "")
        if pd.notna(title) and str(title).strip():
            details.append(f"Source: {title}")
        if pd.notna(url) and str(url).startswith("http"):
            details.append(f"<a href='{url}' target='_blank'>Open source</a>")
        hover_text.append("<br>".join(details))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.2, color="#9CA3AF"),
        hoverinfo="skip", showlegend=False,
    )
    edge_hover_trace = go.Scatter(
        x=hover_x, y=hover_y, mode="markers",
        marker=dict(size=14, color="rgba(0,0,0,0.01)"),
        customdata=hover_text, hovertemplate="%{customdata}<extra></extra>",
        showlegend=False, name="Relationship details",
    )
    node_x, node_y, node_text, node_hover = [], [], [], []
    for node in graph.nodes():
        x, y = positions[node]
        node_x.append(x); node_y.append(y); node_text.append(node)
        node_hover.append(f"<b>{node}</b><br>Connections: {graph.degree(node)}")
    node_marker = dict(size=22, color="#2563EB", line=dict(width=1, color="white"))
    if selected_stress_company != "None" and failure_table is not None:
        stress_colors = {
            "failed": "#DC2626",
            "hop1": "#F97316",
            "hop2": "#FACC15",
            "hop3_plus": "#BAE6FD",
            "not_connected": "#9CA3AF",
        }
        node_marker["color"] = [
            stress_colors[failure_colors_by_name.get(node, "not_connected")]
            for node in graph.nodes()
        ]
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=node_text,
        textposition="top center", hovertext=node_hover,
        hovertemplate="%{hovertext}<extra></extra>",
        marker=node_marker,
        showlegend=False, name="Companies",
    )
    network_fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace])
    network_fig.update_layout(
        height=700, showlegend=False, hovermode="closest",
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
    )
    # Curved arrows: every relationship is drawn as an arc that bends away from the straight line,
    # with an arrowhead at the target company. Repeated deals between the same two companies get
    # their own, more strongly bent arcs. Two-way pairs bend to opposite sides automatically.
    network_fig.data[0].visible = False  # hide the old straight lines
    arc_style = {
        "confirmed": dict(color="#374151", width=1.8, dash="solid"),
        "other": dict(color="#9CA3AF", width=1.4, dash="dot"),
    }
    arc_x = {"confirmed": [], "other": []}
    arc_y = {"confirmed": [], "other": []}
    new_hover_x, new_hover_y = [], []
    pair_counter = {}
    for _, row in network_data.iterrows():
        source, target = str(row["source_company"]), str(row["target_company"])
        if source not in positions or target not in positions:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        repeat = pair_counter.get((source, target), 0)
        pair_counter[(source, target)] = repeat + 1
        arc_points = curved_arc_points(x0, y0, x1, y1, repeat)
        xs = [point[0] for point in arc_points]
        ys = [point[1] for point in arc_points]
        group = "confirmed" if str(row["evidence_status"]).lower() == "confirmed" else "other"
        arc_x[group].extend(xs + [None])
        arc_y[group].extend(ys + [None])
        midpoint_x, midpoint_y = arc_midpoint(x0, y0, x1, y1, repeat)
        new_hover_x.append(midpoint_x)
        new_hover_y.append(midpoint_y)
        network_fig.add_annotation(
            x=xs[-1], y=ys[-1], ax=xs[-4], ay=ys[-4],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, text="",
            arrowhead=2, arrowsize=1.6, arrowwidth=1.6,
            arrowcolor=arc_style[group]["color"], standoff=13,
        )
    for group in ("other", "confirmed"):
        network_fig.add_trace(go.Scatter(
            x=arc_x[group], y=arc_y[group], mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(color=arc_style[group]["color"], width=arc_style[group]["width"],
                      dash=arc_style[group]["dash"], shape="linear"),
        ))
    network_fig.data[1].x = new_hover_x  # move the hover points onto the arcs
    network_fig.data[1].y = new_hover_y
    st.plotly_chart(network_fig, use_container_width=True, config={"displaylogo": False})
    st.caption(
        "Blue nodes represent companies; lines represent directed relationships. "
        "Hover near the middle of a line to inspect its evidence. A recorded link is not by itself proof of a completed cash transfer."
    )
    st.caption(
        "Colours: red = selected company, orange = directly linked, yellow = 2 steps away, light blue = 3 or more steps away, grey = not connected."
    )
    if selected_stress_company != "None" and failure_display_table is not None:
        st.write(failure_summary)
        st.dataframe(failure_display_table, use_container_width=True, hide_index=True)
        st.metric("Cycles through this company", stress_cycle_count)
        st.caption(
            "Structural view only: it shows which companies are linked to the selected one and how many steps away. It is not a forecast of price moves, and it does not mean a failure would spread. Amounts on direct links are not added up."
        )
        stress_tickers_by_name = dict(
            zip(companies["company_name"].astype(str), companies["public_ticker"])
        )
        stress_ticker = stress_tickers_by_name.get(selected_stress_company)
        available_tickers = set(prices["ticker"].dropna().astype(str))
        if pd.isna(stress_ticker) or not str(stress_ticker).strip() or str(stress_ticker) not in available_tickers:
            st.info(
                "This company has no stock price in our data (private company), so its historical behaviour cannot be shown."
            )
        else:
            try:
                stress_comovement, stress_comovement_info = worst_day_comovement(
                    prices, str(stress_ticker), worst_n=20, benchmark="SPY"
                )
            except ValueError as exc:
                st.info(str(exc))
            else:
                st.subheader("How the other public stocks behaved on this company's worst days")
                st.write(worst_day_text(stress_comovement, stress_comovement_info))
                st.dataframe(
                    worst_day_display(stress_comovement),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "Historical co-movement, not a forecast and not proof of cause. Part of every move is the whole market falling at the same time, so compare with SPY. Companies without a stock price (for example OpenAI and Anthropic) cannot be shown."
                )

    
    # Compact source list: show one source first, expand to see the rest
    st.subheader("Sources & Evidence")

    source_columns = [
        col for col in [
            "relationship_id",
            "source_company",
            "target_company",
            "relationship_type",
            "announcement_date",
            "evidence_status",
            "source_title",
            "source_url",
        ]
        if col in network_data.columns
    ]

    sources = network_data[source_columns].copy()

    if "source_url" in sources.columns:
        sources["source_url"] = (
            sources["source_url"].fillna("").astype(str).str.strip()
        )
        sources = sources[
            sources["source_url"].str.startswith(("https://", "http://"))
        ].copy()

    # Remove repeated source URLs to avoid showing duplicate links
    if "source_url" in sources.columns:
        sources = sources.drop_duplicates(subset=["source_url"])

    if sources.empty:
        st.info("No source links are available for the displayed relationships.")
    else:
        first_source = sources.iloc[0]

        # Always show the first source
        first_title = first_source.get("source_title", "")
        if pd.isna(first_title) or not str(first_title).strip():
            first_title = (
                f"{first_source.get('source_company', 'Unknown company')} → "
                f"{first_source.get('target_company', 'Unknown company')}"
            )

        st.markdown(f"**{first_title}**")

        first_url = first_source["source_url"]
        st.link_button("Open original source ↗", first_url)

        remaining_sources = sources.iloc[1:]

        # Keep all other sources hidden until the user expands the section
        if not remaining_sources.empty:
            with st.expander(
                f"Show {len(remaining_sources)} more sources ▾",
                expanded=False,
            ):
                for _, source in remaining_sources.iterrows():
                    title = source.get("source_title", "")
                    if pd.isna(title) or not str(title).strip():
                        title = (
                            f"{source.get('source_company', 'Unknown company')} → "
                            f"{source.get('target_company', 'Unknown company')}"
                        )

                    st.markdown(f"**{title}**")

                    st.caption(
                        f"{source.get('source_company', 'Unknown company')} → "
                        f"{source.get('target_company', 'Unknown company')} | "
                        f"Type: {source.get('relationship_type', 'Not specified')} | "
                        f"Evidence: {source.get('evidence_status', 'Not specified')}"
                    )

                    source_date = source.get("announcement_date", "")
                    if pd.notna(source_date) and str(source_date).strip():
                        st.caption(f"Announcement date: {source_date}")

                    st.link_button(
                        "Open original source ↗",
                        source["source_url"],
                        key=f"source_{source.get('relationship_id', '')}_{source['source_url']}",
                    )

# ---------------------------------------------------------------------------
# Headline: companies inside cycles vs other public companies vs SPY
# ---------------------------------------------------------------------------
st.divider()
st.header("Headline: companies inside cycles vs other public companies")
st.write(
    "The headline number is the Sharpe ratio (return per unit of risk, after a risk-free rate) of an "
    "equal-weight basket of the public companies that appear in the detected cycles. It is compared "
    "with a basket of the other public companies and with SPY. This is a descriptive comparison of a "
    "very small group, not evidence that cycles cause returns."
)

price_tickers = set(prices["ticker"].dropna().astype(str))
cycle_group = [
    t for t in cycle_member_tickers(companies, relationships, confirmed_only=confirmed_only)
    if t in price_tickers
]
other_public = sorted(
    t for t in companies["public_ticker"].dropna().astype(str).str.strip().unique()
    if t in price_tickers and t != "SPY" and t not in cycle_group
)

g1, g2, g3 = st.columns(3)
min_day = prices["date"].min().date()
max_day = prices["date"].max().date()
with g1:
    risk_free_pct = st.number_input(
        "Risk-free rate, % per year (your assumption)",
        min_value=0.0, max_value=20.0, value=4.0, step=0.25, key="grp_risk_free",
    )
with g2:
    start_day = st.date_input(
        "Start date", value=min_day, min_value=min_day, max_value=max_day, key="grp_start"
    )
with g3:
    end_day = st.date_input(
        "End date", value=max_day, min_value=min_day, max_value=max_day, key="grp_end"
    )

default_comparison = [t for t in other_public if t != "CRWV"]
comparison_group = st.multiselect(
    "Comparison group (public companies outside the cycles)",
    other_public, default=default_comparison, key=f"grp_comparison_{confirmed_only}",
)
st.caption(
    "Cycle group (from the evidence filter above): "
    + (", ".join(cycle_group) if cycle_group else "none")
    + ". Companies with a short price history (for example CRWV) shrink the common window "
    "for every series, so CRWV is left out of the comparison group by default."
)

if not cycle_group:
    st.info("No public companies are in cycles with the current evidence filter.")
elif not comparison_group:
    st.warning("Choose at least one company for the comparison group.")
elif start_day >= end_day:
    st.warning("The start date must be before the end date.")
else:
    try:
        group_table, group_info = compare_groups(
            prices,
            {"Cycle group": cycle_group, "Comparison group": comparison_group},
            benchmark="SPY",
            start_date=start_day,
            end_date=end_day,
            risk_free_annual_pct=risk_free_pct,
        )
    except ValueError as exc:
        st.error(f"Could not compare the groups: {exc}")
    else:
        rows = group_table.set_index("group")
        m1, m2, m3 = st.columns(3)
        m1.metric("Sharpe ratio, cycle group", f"{rows.loc['Cycle group', 'sharpe_ratio']:.2f}")
        m1.caption(
            f"Comparison group: {rows.loc['Comparison group', 'sharpe_ratio']:.2f} | "
            f"SPY: {rows.loc['SPY', 'sharpe_ratio']:.2f}"
        )
        m2.metric("Beta vs SPY, cycle group", f"{rows.loc['Cycle group', 'beta_vs_spy']:.2f}")
        m2.caption(f"Comparison group: {rows.loc['Comparison group', 'beta_vs_spy']:.2f}")
        m3.metric("Max drawdown, cycle group", f"{rows.loc['Cycle group', 'max_drawdown_pct']:.1f}%")
        m3.caption(
            f"Comparison group: {rows.loc['Comparison group', 'max_drawdown_pct']:.1f}% | "
            f"SPY: {rows.loc['SPY', 'max_drawdown_pct']:.1f}%"
        )
        st.dataframe(group_table, use_container_width=True, hide_index=True)
        st.caption(
            f"Window: {group_info['window_start']} to {group_info['window_end']} "
            f"({group_info['price_dates_used']} price dates used, {group_info['price_dates_dropped']} dropped "
            "because not every ticker has a price on that day). Equal weights, rebalanced daily, USD adjusted prices. "
            f"The risk-free rate of {group_info['risk_free_annual_pct']:.2f}% per year is your assumption, not downloaded data. "
            "Private companies that appear in cycles have no stock prices and are not in either basket."
        )
        if group_info["price_dates_dropped"] > 0:
            st.warning(
                f"{group_info['limiting_ticker']} has prices only from {group_info['limiting_start']}, "
                "so earlier dates were dropped for every series in this comparison."
            )
        
        growth_long = growth_of_base(
            group_info["daily_returns"], group_info["window_start"], base_value=100.0
        )
        growth_fig = px.line(
            growth_long, x="date", y="value", color="series",
            labels={"date": "Date", "value": "Value of $100", "series": "Series"},
        )
        growth_fig.update_layout(
            height=420, hovermode="x unified",
            yaxis_title="Value of $100 (all series start together on the first common date)",
        )
        st.plotly_chart(growth_fig, use_container_width=True)
        st.caption(
            "Equal-weight baskets rebalanced daily; all three lines start on the same date, "
            "so they are directly comparable. Historical, not a forecast."
        )

# Market data: headline number with units/period and benchmark
st.divider()
st.header("2. Market performance and risk")
all_tickers = sorted(prices["ticker"].dropna().astype(str).unique().tolist())
default_tickers = [ticker for ticker in ["NVDA", "MSFT", "AMZN", "GOOGL", "META", "SPY"] if ticker in all_tickers]
if not default_tickers:
    default_tickers = all_tickers[:min(5, len(all_tickers))]
chosen_tickers = st.multiselect("Choose companies / benchmark", all_tickers, default=default_tickers)
metrics = calculate_market_metrics(prices, benchmark="SPY")
selected_metrics = metrics[metrics["ticker"].isin(chosen_tickers)].copy()
benchmark_row = metrics.loc[metrics["ticker"] == "SPY"]
if not selected_metrics.empty:
    headline_ticker = st.selectbox(
        "Headline company (total return over its available history)",
        selected_metrics["ticker"].tolist(),
        index=0,
    )
    headline = selected_metrics.loc[selected_metrics["ticker"] == headline_ticker].iloc[0]
    period_label = f"{headline['start_date']} to {headline['end_date']}"
    h1, h2, h3 = st.columns(3)
    h1.metric(f"{headline_ticker} total return", f"{headline['total_return_pct']:.2f}%", period_label)
    h2.metric("Annualized volatility", f"{headline['annualized_volatility_pct']:.2f}%" if pd.notna(headline["annualized_volatility_pct"]) else "N/A", "252 trading-day convention")
    h3.metric("Beta vs SPY", f"{headline['beta_vs_spy']:.2f}" if pd.notna(headline["beta_vs_spy"]) else "N/A", "overlapping daily returns")
    st.dataframe(selected_metrics, use_container_width=True, hide_index=True)
else:
    st.warning("Choose at least one ticker to display market metrics.")

st.subheader("Growth of $100 versus the benchmark")
if chosen_tickers:
    chart_data = normalized_performance(prices, tickers=chosen_tickers, base_value=100.0)
    if not chart_data.empty:
        fig = px.line(
            chart_data, x="date", y="normalized_value", color="ticker",
            labels={"date": "Date", "normalized_value": "Value of $100", "ticker": "Ticker"},
        )
        fig.update_layout(height=450, hovermode="x unified", yaxis_title="Value of $100 (each series rebased to its own first date)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No observations are available for the selected tickers.")

# Company explorer
st.divider()
st.header("3. Company and financial-data explorer")
company_options = companies["company_name"].dropna().sort_values().tolist()
selected_name = st.selectbox("Select company", company_options)
selected_company = companies.loc[companies["company_name"] == selected_name].iloc[0]
left, right = st.columns([1, 2])
with left:
    st.write(f"**Ticker:** {selected_company.get('public_ticker', 'N/A')}")
    st.write(f"**Sector:** {selected_company.get('sector', 'N/A')}")
    st.write(f"**Network role:** {selected_company.get('network_role', 'N/A')}")
    source_url = selected_company.get("source_url", "")
    if pd.notna(source_url) and str(source_url).startswith("http"):
        st.markdown(f"[Company source]({source_url})")
with right:
    selected_id = selected_company["company_id"]
    with st.expander(
        "Raw data for the selected company (relationships and financial statements)",
        expanded=False,
    ):
        related, financial_view = company_explorer_tables(
            companies, relationships, financials, selected_id
        )
        st.dataframe(related, use_container_width=True, hide_index=True)
        if financial_view.empty:
            st.info("No financial statements are stored for this company (private or not available).")
        else:
            st.dataframe(financial_view, use_container_width=True, hide_index=True)

with st.expander("Data quality report"):
    missing_report = missing_values_report(companies, relationships, financials)
    quality_pairs, quality_verdict, quality_missing = data_quality_summary(
        validation, missing_report
    )
    if quality_verdict.startswith("All IDs"):
        st.success(quality_verdict)
    else:
        st.warning(quality_verdict)
    for metric_group in (quality_pairs[:4], quality_pairs[4:]):
        metric_columns = st.columns(4)
        for metric_column, (label, value) in zip(metric_columns, metric_group):
            metric_column.metric(label, value)
    st.dataframe(quality_missing, use_container_width=True, hide_index=True)
    st.caption(
        "Empty cells are not always errors. For example, private companies have no ticker, and an amount is left empty when the source does not disclose it (we never write zero instead). Data dictionary: see below."
    )
    if not dictionary.empty:
        st.subheader("Data dictionary")
        st.dataframe(dictionary, use_container_width=True, hide_index=True)

# Honesty / limitations panel
st.divider()
with st.expander("Data, methods and limitations — read before interpreting", expanded=True):
    st.markdown(
        """
        - **Price source:** source URL is stored in each local price CSV where supplied (the current files indicate Yahoo Finance history pages); the app does not fetch new data live.
        - **Pull date:** read from each CSV's `retrieved_date` field when present; inspect the source file for the per-ticker date.
        - **Prices:** the app uses the supplied `adjusted_close` field. Its adjusted-price definition depends on the upstream provider.
        - **Currency:** all prices are in USD; no currency conversion is used.
        - **Frequency:** daily observations; annualized volatility uses √252.
        - **Window:** the market table uses each ticker's own history; the group comparison uses only dates on which every selected ticker has a price (the number of dropped dates is shown under the comparison table).
        - **Dates dropped:** invalid date/price rows are removed by the loader; the group comparison drops dates where any selected ticker has no price; beta uses only overlapping daily-return dates.
        - **Survivorship:** this universe contains selected companies that were chosen for this project, not a survivorship-free investment universe.
        - **Interpretation:** historical performance is descriptive, not a forecast and not investment advice.
        """
    )
    st.dataframe(price_data_summary(prices), use_container_width=True, hide_index=True)

st.caption("Circular Alpha | QIM MVP — local research tool, not investment advice.")
