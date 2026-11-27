"""Circular Alpha v2 — Quantitative Investment Management (KBTU Business School).

Multi-tab Streamlit product combining:
1. Network & Circularity Explorer (reused & expanded from Assignment 1)
2. Collapse Contagion Stress-Test Simulator (DebtRank + Balance-Sheet Channels + News Sentiment)
3. Cluster Lens (K-Means risk map w/ 200-copy noise check + Ward residual correlation clustering)
4. Rule Lab & Evidence Pack (GA2: Lagged SMA Trend Filter vs Buy-and-Hold, E1-E8, Canary check)
5. Daily AI News & Sentiment Agent (Scheduled 10:00 Asia/Almaty with 3-tier credibility filter)
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.mascot import render_mascot

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=League+Spartan:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, [data-testid="stAppViewContainer"] { background:#FFFFFF; color:#02273A; }
*, .stMarkdown, .stText, p, label, input, button { font-family:'Plus Jakarta Sans', sans-serif; }
h1,h2,h3,h4,h5,h6,[data-testid="stMetricValue"] { font-family:'League Spartan', sans-serif !important; color:#311C60 !important; letter-spacing:-.02em; }
[data-testid="stHeader"] { background:rgba(255,255,255,.94); }
[data-testid="stSidebar"] { background:#F7F3FC; border-right:1px solid #DEBEEF; }
[data-testid="stAlert"], .stInfo, .stSuccess, .stWarning { background:#F7F3FC; border:1px solid #DEBEEF; border-radius:16px; box-shadow:none; }
[data-baseweb="tab-list"] { gap:8px; background:transparent; padding:8px 0; }
[data-baseweb="tab"] { border-radius:999px; background:#F7F3FC; color:#311C60; border:1px solid #DEBEEF; padding:8px 16px; }
[data-baseweb="tab"][aria-selected="true"] { background:#4B279B; color:#FFFFFF; }
[data-testid="stMetric"] { border:1px solid #DEBEEF; border-radius:16px; padding:16px; background:#FFFFFF; transition:transform .2s ease, box-shadow .2s ease; }
[data-testid="stMetric"]:hover { transform:translateY(-3px); box-shadow:0 8px 18px rgba(75,39,155,.10); }
.stButton > button { border-radius:999px; border:1px solid #4B279B; color:#4B279B; background:#FFFFFF; }
.stButton > button:hover { background:#4B279B; color:#FFFFFF; }
[data-testid="stDataFrame"] { border:1px solid #DEBEEF; border-radius:14px; }
</style>
""", unsafe_allow_html=True)

px.defaults.color_discrete_sequence = ["#4B279B", "#AB8FBB", "#59529C", "#D9385A"]
px.defaults.color_continuous_scale = ["#F7F3FC", "#AB8FBB", "#4B279B", "#D9385A"]

from core import (
    arc_midpoint,
    calculate_cycle_metrics,
    calculate_market_metrics,
    compare_groups,
    curved_arc_points,
    cycle_member_tickers,
    describe_cycle_amounts,
    format_money,
    growth_of_base,
    prepare_relationships,
    validate_data,
)
from data import (
    BASE_DIR,
    load_companies,
    load_dictionary,
    load_financials,
    load_relationships,
)
from src.backtest import (
    build_equal_weight_basket,
    cost_sensitivity_and_breakeven,
    downside_risk_summary,
    paired_block_bootstrap_sharpe_diff,
    parameter_neighbors_check,
    run_canary_lookahead_test,
    run_rule_backtest,
    subperiod_and_startdate_check,
    summarize_strategy,
)
from src.clustering import (
    adjusted_rand_index,
    build_risk_features,
    evaluate_k_with_null_band,
    hierarchical_residual_clustering,
    kmeans_multistart,
    oos_cluster_portfolio_comparison,
    standardize_features,
)
from src.contagion import run_contagion_stress_test
from src.v3_tabs import render_analog_tab, render_bubble_tab, render_domino_tab
from src.live_data import load_prices_with_live_option
from src.news_agent import (
    get_company_sentiment_map,
    load_news_feed,
    run_daily_update,
)


# These four analyses are deterministic and expensive; keep their results across
# Streamlit reruns so tab switches do not recompute the quantitative evidence.
_evaluate_k_with_null_band = evaluate_k_with_null_band
_paired_block_bootstrap_sharpe_diff = paired_block_bootstrap_sharpe_diff
_hierarchical_residual_clustering = hierarchical_residual_clustering
_oos_cluster_portfolio_comparison = oos_cluster_portfolio_comparison


@st.cache_data(show_spinner=False)
def evaluate_k_with_null_band(*args, **kwargs):
    return _evaluate_k_with_null_band(*args, **kwargs)


@st.cache_data(show_spinner=False)
def paired_block_bootstrap_sharpe_diff(*args, **kwargs):
    return _paired_block_bootstrap_sharpe_diff(*args, **kwargs)


@st.cache_data(show_spinner=False)
def hierarchical_residual_clustering(*args, **kwargs):
    return _hierarchical_residual_clustering(*args, **kwargs)


@st.cache_data(show_spinner=False)
def oos_cluster_portfolio_comparison(*args, **kwargs):
    return _oos_cluster_portfolio_comparison(*args, **kwargs)

st.set_page_config(
    page_title="Circular Alpha | AI Contagion, Clustering & Rule Desk",
    page_icon="🔗",
    layout="wide",
)

# --- UI language helper (calculations, tickers, formulas, and raw columns stay unchanged) ---
lang = "English"
with st.sidebar:
    lang = st.radio("Language / Язык", ["English", "Русский"], horizontal=True)

def tr(en: str, ru: str) -> str:
    return ru if lang == "Русский" else en

# --- Sidebar Controls ---
with st.sidebar:
    st.title(tr("🔗 Circular Alpha v2", "🔗 Circular Alpha v2"))
    st.caption(tr("Quantitative Investment Management · KBTU Business School", "Количественное управление инвестициями · KBTU Business School"))
    st.divider()
    live_mode = st.toggle(
        tr("Live Yahoo Finance prices", "Котировки Yahoo Finance в реальном времени"),
        value=False,
        help=tr("Off = reproducible local CSV snapshot (required for GA2 grading). On = live yfinance fetch with offline fallback.", "Выкл. — воспроизводимый локальный CSV-снимок (нужен для проверки GA2). Вкл. — загрузка yfinance с переходом к локальным данным при сбое."),
    )

with st.sidebar:
    # Scenario-specific evidence and sentiment controls live in the hero tab.
    confirmed_only_global = False
    include_unverified_news = False
    st.divider()
    st.caption(tr(
        "**Honest-Result Notice:** Contagion figures are scenario stress-tests, not point price forecasts. "
        "Backtests are historical with error bars. Not investment advice.",
        "**Честное предупреждение:** показатели заражения — сценарный стресс-тест, а не точный прогноз цены. Бэктесты основаны на истории и имеют погрешность. Это не инвестиционная рекомендация."
    )
    )

# --- Load Data ---
companies = load_companies()
relationships = load_relationships()
financials = load_financials()
dictionary = load_dictionary()
prices, price_status = load_prices_with_live_option(live_mode=live_mode)
news_feed = load_news_feed()
sentiment_map = get_company_sentiment_map(news_feed, include_unverified=include_unverified_news)
validation = validate_data(companies, relationships)

st.title(tr("Circular Alpha — AI Ecosystem Contagion, Cluster Lens & Rule Desk", "Circular Alpha — эффект домино в ИИ-индустрии, кластеры риска и тест правил"))
st.caption(f"{tr('Price data', 'Данные цен')}: {price_status} | {tr('Daily News Agent', 'Ежедневный новостной агент')}: {news_feed.get('last_run_almaty', 'N/A')}")

# First-time visitor orientation
st.info(tr(
    "**The 30-Second Story**\n\n"
    "Big AI companies invest in each other, buy chips and cloud from each other, and guarantee each other’s debt — this is **circular financing**.\n\n"
    "If one major hub stumbles, losses can ripple across balance sheets, cloud backlogs, and stock prices.\n\n"
    "Use the tabs below to simulate a collapse, see how many independent bets AI stocks really represent (**Cluster Lens**), test a systematic risk-filter rule (**Rule Lab**), and track verified daily AI news.",
    "**История за 30 секунд**\n\nКрупные компании ИИ инвестируют друг в друга, покупают друг у друга чипы и облачные услуги и гарантируют чужие долги — так работает **круговое финансирование**.\n\nЕсли один крупный узел даст сбой, потери могут быстро пройти по балансам, облачным заказам и котировкам акций.\n\nНа вкладках ниже можно смоделировать обвал, оценить реальное число независимых ставок (**Кластеры риска**), проверить правило управления риском (**Тест правила**) и следить за проверенными новостями ИИ."
))

# Top KPI bar
cycle_table_top, cycle_summary_top = calculate_cycle_metrics(
    companies, relationships, confirmed_only=confirmed_only_global
)
k1, k2, k3, k4 = st.columns(4)
k1.metric(tr("Tracked AI Entities", "Компании и организации ИИ"), f"{validation['company_rows']}")
k2.metric(tr("Recorded Links", "Зафиксированные связи"), f"{validation['relationship_rows']}")
k3.metric(tr("Directed Loops Found", "Найденные циклы"), f"{cycle_summary_top['cycle_count']}")
k4.metric(tr("Verified News Signals", "Проверенные сигналы новостей"), f"{sum(1 for x in news_feed.get('items', []) if x.get('include_in_base_forecast'))}")

legacy_tab_labels = [
    "💥 Collapse Contagion Simulator",
    "🧩 Cluster Lens (QIM)",
    "📐 Rule Lab & Evidence (GA2)",
    "🔗 Network & Loops (GA1)",
    "📰 Daily AI News Agent (10:00 Almaty)",
][:3] + ["Daily AI News Agent (10:00 Almaty)"]
tab_contagion, tab_cluster, tab_rule, tab_news, tab_bubble, tab_domino, tab_analog = st.tabs([
    tr("💥 Collapse Contagion Simulator", "💥 Симулятор обвала"),
    tr("🧩 Cluster Lens (QIM)", "🧩 Кластеры риска"),
    tr("📐 Rule Lab & Evidence (GA2)", "📐 Тест правила (GA2)"),
    tr("📰 Daily AI News Agent (10:00 Almaty)", "📰 ИИ-агент новостей (10:00 Алматы)"),
    tr("🫧 Bubble Radar & Triggers", "🫧 Радар пузыря и триггеры"),
    tr("📉 Domino & S&P 500 Forecast", "📉 Домино и прогноз S&P 500"),
    tr("🕰️ 2000–2002 Analog & Playbook", "🕰️ Аналог 2000–2002 и план действий"),
])
tab_network = tab_contagion  # legacy evidence block is rendered in the unified hero tab

# ===========================================================================
# TAB 1: COLLAPSE CONTAGION SIMULATOR
# ===========================================================================
with tab_contagion:
    st.header(tr("Multi-Channel Collapse Contagion Simulator", "Симулятор эффекта домино: что будет при обвале компании"))
    st.write(
        tr("Select any company in the AI ecosystem to model how a collapse or severe distress propagates across **equity write-downs**, **contracted cloud/chip revenue backlog**, **off-balance-sheet SPV/debt guarantees**, and **multi-hop DebtRank spillovers** — comparing the **Quantitative-Only** scenario against the **Quantitative + Daily News Sentiment** adjusted scenario.", "Выберите компанию и посмотрите, как её обвал или тяжёлый стресс может пройти по **списаниям инвестиций**, **потере облачных и чиповых контрактов**, **долговым гарантиям и SPV** и многошаговым эффектам DebtRank — в сравнении с моделью, скорректированной по ежедневным новостям.")
    )

    comp_names = companies["company_name"].astype(str).tolist()
    default_idx = comp_names.index("Anthropic PBC") if "Anthropic PBC" in comp_names else 0
    c_sel, c_sev, c_damp = st.columns([2, 1, 1])
    with c_sel:
        selected_name = st.selectbox(tr("Company experiencing collapse / distress", "Компания, переживающая обвал или стресс"), comp_names, index=default_idx)
    with c_sev:
        shock_pct = st.slider(tr("Shock severity (%)", "Сила шока (%)"), min_value=25, max_value=100, value=100, step=5)
    with c_damp:
        damping = st.slider(tr("Multi-hop pass-through (δ)", "Передача эффекта по цепочке (δ)"), min_value=0.15, max_value=0.60, value=0.35, step=0.05)

    c_filter, c_sent = st.columns([2.3, 1.5])
    with c_filter:
        evidence_filter = st.radio(tr("Data & Source Filter", "Фильтр данных и источников"), [tr("Verified Primary Only (SEC / Company IR — 18 confirmed links)", "Только проверенные первичные источники (SEC / IR компаний — 18 подтверждённых связей)"), tr("Verified + Tier-1 Institutional/Press (42 links, incl. JPM/WSJ/Bloomberg Oct–Nov 2026)", "Проверенные + институциональные СМИ Tier 1 (42 связи, включая JPM/WSJ/Bloomberg, октябрь–ноябрь 2026)")], index=1)
    with c_sent:
        apply_news = st.toggle(tr("Apply Daily AI News Sentiment Adjustment", "Учитывать настроение ежедневных новостей ИИ"), value=True)
    confirmed_only_scenario = evidence_filter.startswith("Verified Primary Only")
    sentiment_for_run = sentiment_map if apply_news else {}
    cid_map = dict(zip(companies["company_name"].astype(str), companies["company_id"].astype(str)))
    failed_cid = cid_map[selected_name]

    impact_df, impact_summary = run_contagion_stress_test(
        companies=companies,
        relationships=relationships,
        financials=financials,
        failed_company_id=failed_cid,
        prices=prices,
        shock_severity=shock_pct / 100.0,
        damping=damping,
        confirmed_only=confirmed_only_scenario,
        news_sentiment_by_company=sentiment_for_run,
    )

    st.caption(tr("🔴 Selected collapse · 🟠 Directly impacted (Hop 1) · 🟡 Indirect spillover (Hop 2) · ⚪ Unconnected / Hop 3+", "🔴 Выбранная компания · 🟠 прямой эффект (шаг 1) · 🟡 косвенный эффект (шаг 2) · ⚪ нет связи / шаг 3+"))
    st.markdown(tr("**Edge legend:** 🔵 Equity investment · 🟢 Cloud / chip revenue contract · 🟠 Debt guarantee / SPV financing · solid = confirmed primary · dotted = provisional secondary", "**Легенда связей:** 🔵 инвестиции в акции · 🟢 контракт на облако или чипы · 🟠 долговая гарантия / SPV · сплошная линия = подтверждённый первичный источник · пунктир = предварительные данные"))
    net_df = prepare_relationships(companies, relationships, confirmed_only=confirmed_only_scenario).dropna(subset=["source_company", "target_company"])
    if not net_df.empty:
        graph = nx.DiGraph()
        graph.add_edges_from((str(r.source_company), str(r.target_company)) for r in net_df.itertuples())
        pos = nx.spring_layout(graph, seed=42, k=2.4, iterations=120)
        hop_by_name = {selected_name: 0}
        if not impact_df.empty:
            hop_by_name.update(dict(zip(impact_df.company_name, impact_df.hops)))
        edge_colors = {"investment": "#2563EB", "supplier_customer": "#0F766E", "commercial_agreement": "#14B8A6", "guarantee": "#DC2626", "financing": "#F59E0B"}
        fig_net, annotations, pair_cnt = go.Figure(), [], {}
        for _, row in net_df.iterrows():
            s, t = str(row.source_company), str(row.target_company)
            if s not in pos or t not in pos:
                continue
            rep = pair_cnt.get((s, t), 0); pair_cnt[(s, t)] = rep + 1
            points = curved_arc_points(*pos[s], *pos[t], rep)
            connected = selected_name in (s, t) or hop_by_name.get(s, 99) <= 1 or hop_by_name.get(t, 99) <= 1
            color = edge_colors.get(str(row.relationship_type).lower(), "#64748B")
            dash = "solid" if str(row.evidence_status).lower() == "confirmed" else "dot"
            fig_net.add_trace(go.Scatter(x=[p[0] for p in points], y=[p[1] for p in points], mode="lines", line=dict(color=color, width=2.8 if connected else 1.2, dash=dash), opacity=1.0 if connected else 0.15, hoverinfo="skip", showlegend=False))
            mid = len(points) // 2; p0, p1 = points[max(0, mid - 1)], points[mid]
            mx, my = arc_midpoint(*pos[s], *pos[t], rep)
            amount = format_money(row.amount) if pd.notna(row.amount) else "Not disclosed"
            hover = f"<b>{s} → {t}</b><br>Relationship: {row.relationship_type}<br>Amount: {amount}<br>Evidence: {row.evidence_status}<br>Source: {row.source_title}"
            fig_net.add_trace(go.Scatter(x=[mx], y=[my], mode="markers", marker=dict(size=14, color="rgba(0,0,0,0.01)"), customdata=[hover], hovertemplate="%{customdata}<extra></extra>", showlegend=False))
            annotations.append(dict(x=p1[0], y=p1[1], ax=p0[0], ay=p0[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.4, standoff=14, arrowcolor=color, opacity=1.0 if connected else 0.2))
        names = list(graph.nodes())
        colors, sizes = [], []
        for name in names:
            hop = hop_by_name.get(name, 99)
            if name == selected_name: colors.append("#B91C1C"); sizes.append(36)
            elif hop == 1: colors.append("#F97316"); sizes.append(26)
            elif hop == 2: colors.append("#FBBF24"); sizes.append(20)
            else: colors.append("#CBD5E1"); sizes.append(15)
        fig_net.add_trace(go.Scatter(x=[pos[n][0] for n in names], y=[pos[n][1] for n in names], mode="markers+text", text=names, textposition="top center", textfont=dict(size=11), marker=dict(size=sizes, color=colors, line=dict(width=1.5, color="white")), hovertext=[f"<b>{n}</b><br>Hop: {hop_by_name.get(n, 'Unconnected')}" for n in names], hovertemplate="%{hovertext}<extra></extra>", showlegend=False))
        fig_net.update_layout(height=620, plot_bgcolor="white", paper_bgcolor="white", xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=35, r=35, t=20, b=20), annotations=annotations)
        st.plotly_chart(fig_net, use_container_width=True)
    st.info(tr(
            "**What does this mean for a normal investor?** Arrows show who pays or guarantees whom. Dimmed lines hide background noise so the domino chain around the stressed company is easy to follow.",
            "**Что это значит для обычного инвестора?** Стрелки показывают, кто кому платит или гарантирует долг. Бледные линии скрывают фоновый шум, чтобы была видна цепочка домино вокруг проблемной компании."
        ))
    if not apply_news:
        st.warning(tr("Quantitative-only view is active. Tier-3 rumors remain excluded by default.", "Включён режим только количественной модели. Слухи Tier 3 по-прежнему исключены из базового сценария."))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(tr("Directly Hit (Hop 1)", "Прямо затронуты (шаг 1)"), impact_summary["hop1_count"], f"+{impact_summary['hop2_plus_count']} {tr('indirect (Hop 2+)', 'косвенно (шаг 2+)')}")
    m2.metric(tr("Equity Write-Downs", "Списание инвестиций (акции)"), format_money(impact_summary["total_equity_writedown_usd"]))
    m3.metric(tr("Revenue / Backlog at Risk", "Потеря контрактов и выручки"), format_money(impact_summary["total_revenue_backlog_usd"]))
    m4.metric(tr("Guarantee / SPV Calls", "Срабатывание долговых гарантий (SPV)"), format_money(impact_summary["total_guarantee_call_usd"]))

    if impact_summary["undisclosed_direct_links_total"] > 0:
        st.caption(
            f"Note: {impact_summary['undisclosed_direct_links_total']} direct link(s) have undisclosed dollar amounts "
            "(left un-imputed, never treated as $0). Superseded LOI rows R001/R002 are excluded to avoid double counting."
        )

    if not impact_df.empty:
        st.info(tr(
            "**Everyday glossary:** Equity Write-Down = lost investment value; Revenue / Backlog = cancelled cloud or chip orders; Guarantee / SPV Call = paying someone else’s debt; Estimated Drawdown = how far the stock could fall in this stress scenario.",
            "**Простой словарь:** Equity Write-Down = потерянная стоимость инвестиций; Revenue / Backlog = отменённые заказы на облако или чипы; Guarantee / SPV Call = необходимость оплатить чужой долг; Estimated Drawdown = насколько акция может упасть в этом стресс-сценарии."
        ))
        st.subheader(tr("Top Impacted Companies — Executive Impact Cards", "Наиболее уязвимые компании — ключевые показатели"))
        ranked_impact = impact_df.sort_values("contagion_score_news_adj", ascending=False).head(2)
        channel_labels = {
            "equity_writedown_usd": "equity write-down",
            "revenue_backlog_at_risk_usd": "backlog",
            "guarantee_spv_call_usd": "guarantee call",
        }

        def primary_channel(impact_row):
            channel = max(channel_labels, key=lambda column: float(impact_row.get(column, 0) or 0))
            return channel_labels[channel]

        top_one = ranked_impact.iloc[0]
        top_two = ranked_impact.iloc[1] if len(ranked_impact) > 1 else None
        top_two_text = (
            f" and #2 **{top_two['company_name']}** ({primary_channel(top_two)})"
            if top_two is not None
            else ""
        )
        worst_quant = float(impact_df["scenario_drawdown_quant_pct"].min())
        worst_news = float(impact_df["scenario_drawdown_news_adj_pct"].min())
        news_shift = worst_news - worst_quant
        news_shift_text = (
            f"worsens the worst estimated drawdown by {abs(news_shift):.1f} percentage points"
            if news_shift < 0
            else f"changes the worst estimated drawdown by {abs(news_shift):.1f} percentage points"
        )
        st.info(tr(
            f"**What Happens in Plain English** {selected_name} takes a **{shock_pct}% shock**, hitting {int(impact_summary['hop1_count'])} direct (Hop 1) companies and {int(impact_summary['hop2_plus_count'])} indirect (Hop 2+) companies. The most exposed are #1 **{top_one['company_name']}** ({primary_channel(top_one)}){top_two_text}. Verified daily news sentiment {news_shift_text}, moving the worst estimated drawdown from {worst_quant:.1f}% to {worst_news:.1f}%.",
            f"**Простыми словами:** компания {selected_name} получает шок **{shock_pct}%**, который напрямую затрагивает {int(impact_summary['hop1_count'])} компаний и косвенно — ещё {int(impact_summary['hop2_plus_count'])}. Сильнее всего рискует #1 **{top_one['company_name']}** ({primary_channel(top_one)}). Проверенное настроение новостей {news_shift_text}; худшая оценка просадки меняется с {worst_quant:.1f}% до {worst_news:.1f}%."
        ))
        card_cols = st.columns(min(5, len(impact_df.head(5))))
        for card, (_, row) in zip(card_cols, impact_df.head(5).iterrows()):
            with card:
                with st.container(border=True):
                    st.markdown(f"**{row['company_name']}** · `{row['ticker']}`")
                    st.caption(f"{tr('Hop', 'Шаг')} {int(row['hops'])} · Quant {row['contagion_score_quant']:.1f} vs News {row['contagion_score_news_adj']:.1f}")
                    st.write(f"{tr('Equity Write-Down', 'Списание инвестиций (акции)')}: **{format_money(row['equity_writedown_usd'])}**")
                    st.write(f"{tr('Revenue / Backlog', 'Потеря контрактов и выручки')}: **{format_money(row['revenue_backlog_at_risk_usd'])}**")
                    st.write(f"{tr('Guarantee / SPV Call', 'Срабатывание долговых гарантий (SPV)')}: **{format_money(row['guarantee_spv_call_usd'])}**")
                    st.write(f"Cash / EBITDA: **{row['exposure_pct_of_cash']:.1f}% / {row['exposure_pct_of_ebitda']:.1f}%**" if pd.notna(row['exposure_pct_of_cash']) and pd.notna(row['exposure_pct_of_ebitda']) else tr("Cash / EBITDA: not disclosed", "Cash / EBITDA: данных нет"))
                    st.caption(str(row['affected_business_segments']))
                    quant_dd = float(row["scenario_drawdown_quant_pct"])
                    news_dd = float(row["scenario_drawdown_news_adj_pct"])
                    st.metric(
                        tr("Estimated equity drawdown (quant)", "Оценка просадки акций"),
                        f"{quant_dd:.1f}%",
                        delta=f"{news_dd - quant_dd:+.1f} pp (news: {news_dd:.1f}%)",
                        delta_color="normal",
                    )

    with st.expander(tr("How this forecast is built: data sources, formulas & credibility filters", "Как построен расчёт: источники, формулы и проверка достоверности")):
        st.markdown("""
**Data.** SEC EDGAR financials, official company IR/press releases (`confirmed`), and Tier-1 institutional/financial press (`provisional`, including JPMorgan, WSJ and Bloomberg). Superseded LOIs R001/R002 are excluded; undisclosed amounts remain `Not disclosed`, never imputed as $0.

**Propagation.** Hop-1 channels are decomposed into equity, revenue/backlog and guarantee/SPV exposures, scaled by Cash/EBITDA. Hops 2–3+ use the Battiston et al. (2012) DebtRank-style pass-through with damping δ.

**Market tail.** Yahoo Finance adjusted prices (live or reproducible CSV snapshot) estimate downside co-movement on the 5% worst stress days.

**News filter.** `src/news_agent.py` runs daily at 10:00 Asia/Almaty and separates Tier-1/Tier-2 credible signals from Tier-3 rumors; Tier-3 items are flagged and excluded from the base forecast.

**Disclaimer.** This is a scenario stress-test model, not a point price prediction or investment advice.
""")
    with st.expander(tr("Raw relationship evidence & directed loop list", "Исходные связи и список циклов")):
        st.dataframe(net_df[[c for c in ["source_company", "target_company", "relationship_type", "amount", "evidence_status", "source_title"] if c in net_df.columns]], use_container_width=True, hide_index=True)
        if not cycle_table_top.empty:
            st.dataframe(describe_cycle_amounts(companies, relationships, cycle_table_top, confirmed_only=confirmed_only_scenario, data_dictionary=dictionary), use_container_width=True, hide_index=True)

    if impact_df.empty:
        st.info(tr("No connected companies under the current evidence filter. Try turning off 'Confirmed primary links only' in the sidebar.", "При текущем фильтре подтверждённых источников связанных компаний нет. Попробуйте отключить в боковой панели режим «Только подтверждённые первичные связи»."))
    else:
        # Comparison chart: Quant-only vs News-adjusted contagion score
        chart_df = impact_df.head(10).melt(
            id_vars=["company_name", "hops"],
            value_vars=["contagion_score_quant", "contagion_score_news_adj"],
            var_name="Model",
            value_name="Stress Score (0-100)",
        )
        chart_df["Model"] = chart_df["Model"].map({
            "contagion_score_quant": "Quantitative Only (Balance Sheet + DebtRank)",
            "contagion_score_news_adj": "Quantitative + Daily News Sentiment",
        })
        fig_cont = px.bar(
            chart_df,
            x="Stress Score (0-100)",
            y="company_name",
            color="Model",
            barmode="group",
            orientation="h",
            color_discrete_sequence=["#1D4ED8", "#B91C1C"],
            title=tr(f"Projected Contagion Stress if {selected_name} Suffers a {shock_pct}% Shock", f"Стресс для рынка при шоке {shock_pct}% у компании {selected_name}"),
        )
        fig_cont.update_layout(height=420, yaxis={"categoryorder": "total ascending"}, yaxis_title="")
        st.plotly_chart(fig_cont, use_container_width=True)
        st.info(tr(
            "**Why two bars?** The blue bar is pure balance-sheet and network math; the red bar adds today’s verified news mood. Tier-3 rumors are excluded by default.",
            "**Почему два столбца?** Синий — это чистая математика балансов и сети, красный добавляет сегодняшнее проверенное настроение новостей. Слухи Tier 3 по умолчанию исключены."
        ))

        # Detailed breakdown table
        st.subheader(tr("Company-by-Company Financial & Business-Line Impact", "Финансовый и операционный эффект по компаниям"))
        display_cols = [
            "company_name", "ticker", "hops",
            "contagion_score_quant", "contagion_score_news_adj",
            "equity_writedown_usd", "revenue_backlog_at_risk_usd", "guarantee_spv_call_usd",
            "exposure_pct_of_cash", "exposure_pct_of_ebitda",
            "scenario_drawdown_quant_pct", "scenario_drawdown_news_adj_pct",
            "affected_business_segments",
        ]
        disp = impact_df[display_cols].copy()
        for money_col in ["equity_writedown_usd", "revenue_backlog_at_risk_usd", "guarantee_spv_call_usd"]:
            disp[money_col] = disp[money_col].apply(lambda v: format_money(v) if v > 0 else "—")
        disp = disp.rename(columns={
            "company_name": "Company",
            "ticker": "Ticker",
            "hops": "Hop",
            "contagion_score_quant": "Quant Score",
            "contagion_score_news_adj": "News-Adj Score",
            "equity_writedown_usd": "Equity Write-Down",
            "revenue_backlog_at_risk_usd": "Revenue / Backlog",
            "guarantee_spv_call_usd": "Guarantee / SPV Call",
            "exposure_pct_of_cash": "% of Cash",
            "exposure_pct_of_ebitda": "% of EBITDA",
            "scenario_drawdown_quant_pct": "Quant Scenario DD (%)",
            "scenario_drawdown_news_adj_pct": "News-Adj DD (%)",
            "affected_business_segments": "Affected Business Segments",
        })
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.caption(impact_summary["disclaimer"])

# ===========================================================================
# TAB 2: CLUSTER LENS (KBTU QIM LECTURE)
# ===========================================================================
with tab_cluster:
    st.info(tr(
        "**Plain-language warning:** Cluster analysis does **not** predict tomorrow’s stock price. It is a hidden-risk radar: are 9 AI stocks really 9 separate bets, or a few crowded groups?",
        "**Важное предупреждение:** кластерный анализ **не предсказывает** цену акции завтра. Это радар скрытого риска: 9 AI-акций — это действительно 9 отдельных ставок или несколько перегруженных групп?"
    ))
    st.header(tr("Cluster Lens: Letting the AI Market Sort Itself", "Кластеры риска: как устроен AI-рынок"))
    st.caption(tr("Why this matters: This tab shows whether a basket of AI stocks is really many independent bets or a few risks wearing different ticker symbols.", "Зачем это нужно: вкладка показывает, действительно ли корзина AI-акций состоит из независимых ставок или скрывает несколько одинаковых рисков под разными тикерами."))
    st.write(tr(
        "Implements the KBTU QIM Cluster Analysis methodology: z-score standardization, multi-start K-Means (`nstart=100`), silhouette verification against **200 column-shuffled null copies**, Ward hierarchical clustering on **SPY market-removed residual correlations** ($d = \\sqrt{2(1 - \\rho)}$), Adjusted Rand Index (ARI) time-stability check, and out-of-sample portfolio testing.",
        "Здесь применена методика кластерного анализа KBTU QIM: стандартизация z-оценками, K-Means со 100 стартами, проверка силуэта на **200 перемешанных копиях**, кластеризация Ward по остаточным корреляциям после удаления влияния SPY, проверка стабильности ARI и тест портфеля вне выборки."
    ))

    cl_k = st.slider(tr("Number of clusters (k)", "Число кластеров (k)"), min_value=2, max_value=4, value=3)
    feats = build_risk_features(prices, benchmark="SPY")
    z_feats, _, _ = standardize_features(feats)
    labels_k, _, wcss_k = kmeans_multistart(z_feats.to_numpy(), k=cl_k, nstart=100, seed=42)
    feats_plot = feats.copy()
    feats_plot["Cluster"] = [f"Cluster {c + 1}" for c in labels_k]
    feats_plot["Ticker"] = feats_plot.index

    col_left, col_right = st.columns(2)
    with col_left:
        fig_risk = px.scatter(
            feats_plot,
            x="ann_vol_pct",
            y="beta_vs_spy",
            color="Cluster",
            text="Ticker",
            size="ann_vol_pct",
            title=tr(f"K-Means Risk Map (k={cl_k}, standardized 4 features, nstart=100)", f"Карта рисков K-Means (k={cl_k}, 4 стандартизированных признака, nstart=100)"),
            labels={"ann_vol_pct": tr("Annualized Volatility (%)", "Годовая волатильность (%)"), "beta_vs_spy": "Beta vs SPY"},
        )
        fig_risk.update_traces(textposition="top center")
        fig_risk.update_layout(height=400)
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_right:
        k_eval = evaluate_k_with_null_band(z_feats.to_numpy(), k_values=[2, 3, 4], n_shuffles=200, nstart=50, seed=42)
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=k_eval["k"], y=k_eval["silhouette_real"], mode="lines+markers",
            name=tr("Real AI Stocks", "Реальные AI-акции"), line=dict(color="#1D4ED8", width=3),
        ))
        fig_sil.add_trace(go.Scatter(
            x=k_eval["k"], y=k_eval["silhouette_null_95"], mode="lines+markers",
            name=tr("Shuffled Null (95th pct, 200 copies)", "Перемешанный шум (95-й перцентиль, 200 копий)"), line=dict(color="#94A3B8", dash="dash"),
            fill="tozeroy", fillcolor="rgba(148,163,184,0.2)",
        ))
        fig_sil.update_layout(
            title=tr("Noise Check: Real Silhouette vs. 200 Shuffled Copies", "Проверка на шум: реальный силуэт и 200 перемешанных копий"),
            xaxis_title=tr("Number of clusters (k)", "Число кластеров (k)"),
            yaxis_title=tr("Average Silhouette", "Средний показатель силуэта"),
            height=400,
        )
        st.plotly_chart(fig_sil, use_container_width=True)
        cluster_story = []
        for cluster_name, cluster_rows in feats_plot.groupby("Cluster", sort=True):
            members = ", ".join(cluster_rows["Ticker"].astype(str).tolist())
            avg_vol = float(cluster_rows["ann_vol_pct"].mean())
            cluster_story.append(f"{cluster_name}: {members} (average volatility {avg_vol:.2f}%).")
        selected_k_eval = k_eval.loc[k_eval["k"] == cl_k].iloc[0]
        above_noise = bool(selected_k_eval["above_noise_band"])
        noise_beating_k = k_eval.loc[k_eval["above_noise_band"], "k"].astype(int).tolist()
        noise_result = (
            f"Selected k={cl_k} has silhouette {selected_k_eval['silhouette_real']:.3f} "
            f"versus the 200-shuffle noise band at {selected_k_eval['silhouette_null_95']:.3f}; "
            f"it {'beats' if above_noise else 'falls inside'} random noise."
        )
        if not above_noise and noise_beating_k:
            noise_result += f" A k that beats the noise band is {', '.join(map(str, noise_beating_k))}."
        st.info(tr(
            f"**K-Means result for k={cl_k}:** {' '.join(cluster_story)} {noise_result}",
            f"**Результат K-Means для k={cl_k}:** {' '.join(cluster_story)} {noise_result}",
        ))
        if False:
            st.info(tr(
            "The three groups are different combinations of volatility, market sensitivity, and risk features. The grey band is a **lie detector**: algorithms can cluster random noise, so only points above the grey 200-shuffle band show real structure.",
            "Три группы — это разные сочетания волатильности, чувствительности к рынку и других признаков риска. Серая полоса — **детектор лжи**: алгоритм может сгруппировать случайный шум, поэтому реальная структура есть только у точек выше серой полосы 200 перестановок."
        ))

    st.subheader(tr("Hierarchical Ward Clustering on SPY Market-Removed Residuals", "Иерархическая кластеризация Ward по доходностям без влияния SPY"))
    hier = hierarchical_residual_clustering(prices, benchmark="SPY", k=cl_k)

    # Time stability ARI
    f_early = build_risk_features(prices, start_date="2023-01-03", end_date="2024-12-31")
    f_late = build_risk_features(prices, start_date="2025-01-01", end_date="2026-10-08")
    common_t = sorted(set(f_early.index) & set(f_late.index))
    z_e, _, _ = standardize_features(f_early.loc[common_t])
    z_l, _, _ = standardize_features(f_late.loc[common_t])
    le, _, _ = kmeans_multistart(z_e.to_numpy(), k=cl_k, nstart=100, seed=42)
    ll, _, _ = kmeans_multistart(z_l.to_numpy(), k=cl_k, nstart=100, seed=42)
    ari_time = adjusted_rand_index(le, ll)

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Effective Independent Bets", f"{hier['effective_independent_bets']} / {len(hier['tickers'])}")
    h2.metric("Mean Raw vs. Residual Corr", f"{hier['mean_raw_corr']:.2f} → {hier['mean_residual_corr']:.2f}")
    h3.metric("Within vs. Across Cluster Corr", f"{hier['within_cluster_corr']:.2f} vs. {hier['across_cluster_corr']:.2f}")
    h4.metric("Time-Stability ARI (23–24 vs 25–26)", f"{ari_time:.2f}")

    fig_hm = px.imshow(
        hier["raw_corr"],
        text_auto=".2f",
        color_continuous_scale="Blues",
        zmin=0, zmax=1,
        title=tr("Cluster-Sorted Daily Return Correlation Matrix (Ward on Market-Removed Residuals)", "Матрица корреляций дневных доходностей по кластерам (Ward после удаления влияния рынка)"),
    )
    fig_hm.update_layout(height=460)
    st.plotly_chart(fig_hm, use_container_width=True)
    raw_corr = hier["raw_corr"]
    pair_values = [
        (float(raw_corr.iloc[i, j]), str(raw_corr.index[i]), str(raw_corr.columns[j]))
        for i in range(len(raw_corr)) for j in range(i + 1, len(raw_corr.columns))
    ]
    pair_story = "; ".join(
        f"{left}–{right} ({corr:.3f})" for corr, left, right in sorted(pair_values, reverse=True)[:2]
    )
    st.info(tr(
        f"**What does this mean?** The {len(hier['tickers'])}-ticker set has {hier['effective_independent_bets']} effective independent bets. Mean raw correlation is {hier['mean_raw_corr']:.3f}, mean residual correlation is {hier['mean_residual_corr']:.3f}; within-cluster correlation is {hier['within_cluster_corr']:.3f} versus {hier['across_cluster_corr']:.3f} across clusters. The highest raw-correlation pairs are {pair_story}. Time-stability ARI is {ari_time:.3f}.",
        f"**Что это значит?** Набор из {len(hier['tickers'])} тикеров содержит {hier['effective_independent_bets']} эффективных независимых ставок. Средняя исходная корреляция {hier['mean_raw_corr']:.3f}, остаточная {hier['mean_residual_corr']:.3f}; корреляция внутри кластеров {hier['within_cluster_corr']:.3f} против {hier['across_cluster_corr']:.3f} между кластерами. Пары с наибольшей исходной корреляцией: {pair_story}. ARI временной стабильности равен {ari_time:.3f}.",
    ))
    if False:
        st.info(tr(
        f"**What does this mean?** Holding 9 AI stocks behaves like holding only ~{hier['effective_independent_bets']} independent bets. The heatmap highlights tight co-movement among chipmakers, big tech, and infrastructure; ARI {ari_time:.2f} means groups drift over time, so re-check them regularly.",
        f"**Что это значит?** Портфель из 9 AI-акций ведёт себя примерно как ~{hier['effective_independent_bets']} независимых ставок. Тепловая карта показывает тесное движение производителей чипов, big tech и инфраструктуры; ARI {ari_time:.2f} означает, что группы со временем меняются — их нужно регулярно проверять."
    ))

    st.subheader(tr("Out-of-Sample Portfolio Check (Fit 2023–2024, Test 2025–2026)", "Проверка портфеля вне выборки (обучение 2023–2024, тест 2025–2026)"))
    oos_df, oos_info = oos_cluster_portfolio_comparison(prices, benchmark="SPY", split_date="2025-01-01", k=cl_k)
    st.dataframe(oos_df, use_container_width=True, hide_index=True)
    sharpe_col = next(col for col in oos_df.columns if "sharpe" in str(col).lower())
    dd_col = next(col for col in oos_df.columns if "drawdown" in str(col).lower())
    portfolio_col = next(col for col in oos_df.columns if "portfolio" in str(col).lower() or "strategy" in str(col).lower())
    best_row = oos_df.loc[oos_df[sharpe_col].idxmax()]
    crp_row = oos_df[oos_df[portfolio_col].astype(str).str.contains("Cluster Risk Parity", case=False, na=False)].iloc[0]
    oos_story = (
        f"Train: {oos_info['train_window']} | Out-of-Sample Test: {oos_info['test_window']}. "
        f"Highest-Sharpe portfolio {best_row[portfolio_col]} delivered OOS Sharpe {best_row[sharpe_col]:.3f} "
        f"and Max Drawdown {best_row[dd_col]:.2f}%; Cluster Risk Parity delivered OOS Sharpe "
        f"{crp_row[sharpe_col]:.3f} and Max Drawdown {crp_row[dd_col]:.2f}%."
    )
    st.caption(tr(oos_story, oos_story))
    st.info(tr(
        f"Out-of-sample, {best_row[portfolio_col]} has Sharpe {best_row[sharpe_col]:.3f} and Max Drawdown {best_row[dd_col]:.2f}%, while Cluster Risk Parity has Sharpe {crp_row[sharpe_col]:.3f} and Max Drawdown {crp_row[dd_col]:.2f}%.",
        f"Вне выборки {best_row[portfolio_col]} имеет Sharpe {best_row[sharpe_col]:.3f} и максимальную просадку {best_row[dd_col]:.2f}%, а Cluster Risk Parity — Sharpe {crp_row[sharpe_col]:.3f} и максимальную просадку {crp_row[dd_col]:.2f}%.",
    ))
    if False:
        st.caption(tr(
        f"Train: {oos_info['train_window']} | Out-of-Sample Test: {oos_info['test_window']}. Honest takeaway: clustering reveals hidden concentration (~6 real bets out of 9 AI stocks) and shows cluster drift over time (low ARI), while simple 1/N remains hard to beat out-of-sample.",
        f"Обучение: {oos_info['train_window']} | Тест вне выборки: {oos_info['test_window']}. Честный вывод: кластеризация показывает скрытую концентрацию (примерно 6 независимых ставок вместо 9) и изменение групп во времени, а простой 1/N трудно превзойти вне выборки."
    ))
    if False:
        st.info(tr(
        "Equal Weight (1/N) beat Cluster Risk Parity out-of-sample. That is the honest lesson: clustering helps spot concentration risk, not magically boost returns.",
        "Equal Weight (1/N) обошёл Cluster Risk Parity вне выборки. Честный вывод: кластеризация помогает увидеть концентрацию риска, но не является волшебным усилителем доходности."
    ))

# ===========================================================================
# TAB 3: RULE LAB & EVIDENCE PACK (GA2)
# ===========================================================================
with tab_rule:
    st.header(tr("The Rule Desk: Lagged SMA Trend-Risk Filter (GA2)", "Тест правила: отстающий SMA-фильтр тренда и риска (GA2)"))
    st.info(tr(
        "**The rule in one sentence:** step into safe cash whenever the AI basket drops below its 100-day average.",
        "**Правило в одном предложении:** переходить в безопасные денежные средства, когда корзина AI опускается ниже своей 100-дневной средней."
    ))
    st.caption(tr("Why this matters: This tab tests whether a simple, pre-defined rule can reduce downside without relying on hindsight.", "Зачем это нужно: вкладка проверяет, снижает ли простое заранее заданное правило потери без подглядывания в будущее."))
    st.write(tr(
        "Tests whether applying an end-of-day Simple Moving Average trend filter (switching to cash at the risk-free rate when below SMA) improves risk-adjusted performance or reduces drawdowns on the Equal-Weight AI Cycle Basket (**AMD, MSFT, NVDA**). Position is strictly lagged (`position[t] = signal[t-1]`).",
        "Проверяет, помогает ли фильтр тренда на основе простой скользящей средней: если корзина ниже SMA, стратегия переходит в деньги по безрисковой ставке. Позиция всегда запаздывает на один день (`position[t] = signal[t-1]`) и тестируется на равновзвешенной корзине (**AMD, MSFT, NVDA**)."
    ))

    basket_df = build_equal_weight_basket(prices, ["AMD", "MSFT", "NVDA"])
    price_series = basket_df["basket_price"]
    canary_ok = run_canary_lookahead_test(price_series, lookback=20, split_idx=120)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        rule_L = st.select_slider(tr("SMA Lookback (days)", "Период SMA (дни)"), options=[60, 80, 100, 120, 140], value=100)
    with r2:
        rule_cost = st.select_slider(tr("Transaction Cost (bps / switch)", "Комиссия за сделку (б.п. / переключение)"), options=[0.0, 10.0, 25.0, 50.0], value=10.0)
    with r3:
        rule_rf = st.number_input(tr("Cash Risk-Free Rate (% / yr)", "Безрисковая ставка денег (% годовых)"), 0.0, 15.0, 4.0, 0.25)
    with r4:
        st.metric(tr("Canary Look-Ahead Test", "Проверка на заглядывание вперёд"), tr("PASS ✅", "ПРОЙДЕН ✅") if canary_ok else tr("FAIL ❌", "ОШИБКА ❌"), tr("5 train trials · test opened 1x", "5 обучающих прогонов · тест открыт один раз"))

    bt = run_rule_backtest(price_series, lookback=rule_L, cost_bps=rule_cost, risk_free_annual_pct=rule_rf)
    e1 = summarize_strategy(bt, risk_free_annual_pct=rule_rf)
    e2 = paired_block_bootstrap_sharpe_diff(bt["net_return"], bt["benchmark_return"], risk_free_annual_pct=rule_rf)
    e3, breakeven_bps = cost_sensitivity_and_breakeven(price_series, lookback=rule_L, risk_free_annual_pct=rule_rf)
    e4 = subperiod_and_startdate_check(price_series, lookback=rule_L, cost_bps=rule_cost, risk_free_annual_pct=rule_rf)
    e5 = parameter_neighbors_check(price_series, center_lookback=rule_L, cost_bps=rule_cost, risk_free_annual_pct=rule_rf)
    e8 = downside_risk_summary(bt)
    e1_columns = {str(column).lower(): column for column in e1.columns}
    strategy_col = next(column for name, column in e1_columns.items() if "strategy" in name or "portfolio" in name)
    def e1_metric(row_terms, column_terms):
        column = next(column for name, column in e1_columns.items() if any(term in name for term in column_terms))
        row = e1[e1[strategy_col].astype(str).str.contains("|".join(row_terms), case=False, na=False)].iloc[0]
        return float(row[column])
    rule_cagr, benchmark_cagr = e1_metric(["rule"], ["cagr"]), e1_metric(["benchmark", "buy"], ["cagr"])
    rule_vol, benchmark_vol = e1_metric(["rule"], ["vol"]), e1_metric(["benchmark", "buy"], ["vol"])
    rule_dd, benchmark_dd = e1_metric(["rule"], ["drawdown"]), e1_metric(["benchmark", "buy"], ["drawdown"])
    rule_switches = e1_metric(["rule"], ["switch"])
    e8_columns = {str(column).lower(): column for column in e8.columns}
    worst_month_col = next(column for name, column in e8_columns.items() if "worst" in name and "month" in name)
    var_col = next(column for name, column in e8_columns.items() if "var" in name)
    e8_row = e8.iloc[0]
    st.info(tr(
        f"SMA lookback {rule_L} days, transaction cost {rule_cost:.1f} bps per switch, and cash risk-free rate {rule_rf:.2f}% per year. Rule versus benchmark CAGR: {rule_cagr:.2f}% vs {benchmark_cagr:.2f}%; volatility: {rule_vol:.2f}% vs {benchmark_vol:.2f}%; Max Drawdown: {rule_dd:.2f}% vs {benchmark_dd:.2f}%; switches: {rule_switches:.0f}. E2 Sharpe difference is {e2['sharpe_diff']:.3f} with CI [{e2['ci_low']:.3f}, {e2['ci_high']:.3f}], zero inside CI: {bool(e2['zero_inside'])}; break-even cost: {breakeven_bps} bps. E8 worst month is {e8_row[worst_month_col]:.2f}% and 5% VaR is {e8_row[var_col]:.2f}%.",
        f"Период SMA {rule_L} дней, комиссия {rule_cost:.1f} б.п. за переключение и безрисковая ставка {rule_rf:.2f}% годовых. CAGR правила против бенчмарка: {rule_cagr:.2f}% против {benchmark_cagr:.2f}%; волатильность: {rule_vol:.2f}% против {benchmark_vol:.2f}%; максимальная просадка: {rule_dd:.2f}% против {benchmark_dd:.2f}%; переключений: {rule_switches:.0f}. Разница Sharpe E2: {e2['sharpe_diff']:.3f}, ДИ [{e2['ci_low']:.3f}, {e2['ci_high']:.3f}], ноль внутри ДИ: {bool(e2['zero_inside'])}; безубыточная комиссия: {breakeven_bps} б.п. Худший месяц E8: {e8_row[worst_month_col]:.2f}%, VaR 5%: {e8_row[var_col]:.2f}%.",
    ))

    st.info(f"**{tr('Honest Verdict (E2 Paired Block Bootstrap)', 'Честный вывод (парный блочный bootstrap E2)')}:** {e2['sentence']}")

    st.subheader(tr("E1. Headline Performance Table (Same Window)", "E1. Основные результаты (один и тот же период)"))
    st.dataframe(e1, use_container_width=True, hide_index=True)
    st.info(tr(
        "E1–E2 ask whether the result beats luck: headline returns are compared with buy-and-hold and a bootstrap test. The honest verdict is that the rule does not reliably beat buy-and-hold on return, but it meaningfully reduces volatility and crash depth.",
        "E1–E2 проверяют, лучше ли результат случайности: доходность сравнивается с buy-and-hold и bootstrap-тестом. Честный вердикт: правило не превосходит buy-and-hold по доходности стабильно, но заметно снижает волатильность и глубину обвала."
    ))

    eq_df = pd.DataFrame({
        "Date": bt.index,
        f"SMA-{rule_L} Rule (Net, {rule_cost:.0f} bps)": bt["rule_wealth"].to_numpy() * 100,
        "Buy-and-Hold AI Cycle Basket": bt["benchmark_wealth"].to_numpy() * 100,
    }).melt(id_vars="Date", var_name="Strategy", value_name="Value of $100")
    fig_eq = px.line(eq_df, x="Date", y="Value of $100", color="Strategy", color_discrete_sequence=["#1D4ED8", "#64748B"])
    fig_eq.update_layout(height=380, hovermode="x unified")
    st.plotly_chart(fig_eq, use_container_width=True)

    c_e3, c_e8 = st.columns(2)
    with c_e3:
        st.subheader(tr(f"E3. Cost Sensitivity (Break-even: {breakeven_bps} bps)", f"E3. Чувствительность к комиссиям (безубыточность: {breakeven_bps} б.п.)"))
        st.dataframe(e3, use_container_width=True, hide_index=True)
    with c_e8:
        st.subheader(tr("E8. Downside Risk (Worst Month & 5% VaR)", "E8. Риск падения (худший месяц и VaR 5%)"))
        st.dataframe(e8, use_container_width=True, hide_index=True)
    st.info(tr(
        "E3–E8 test the uncomfortable details: can trading fees kill the edge, is performance a stable plateau or a lucky spike, do subperiods and nearby parameters agree, and how bad was the worst month? The same honest verdict holds: lower risk, but no reliable return victory over buy-and-hold.",
        "E3–E8 проверяют неприятные детали: съедают ли комиссии эффект, это устойчивое плато или удачный пик, согласуются ли периоды и соседние параметры, и насколько плохим был худший месяц. Вердикт тот же: риск ниже, но надёжной победы по доходности над buy-and-hold нет."
    ))

    c_e4, c_e5 = st.columns(2)
    with c_e4:
        st.subheader(tr("E4. Start-Date & Subperiod Check", "E4. Проверка дат начала и подпериодов"))
        st.dataframe(e4, use_container_width=True, hide_index=True)
    with c_e5:
        st.subheader(tr("E5. Parameter Neighbours (Plateau vs. Spike)", "E5. Соседние параметры (устойчивое плато или случайный пик)"))
        st.dataframe(e5, use_container_width=True, hide_index=True)

    # Load script-generated E6 & trials ledger
    e6_path = BASE_DIR / "evidence" / "e6_splits.csv"
    trials_path = BASE_DIR / "trials.csv"
    if e6_path.exists():
        st.subheader(tr("E6. Chronological Train / Validation / Test Split (Frozen Script Evidence)", "E6. Хронологическое разделение train / validation / test (зафиксированные результаты)"))
        st.dataframe(pd.read_csv(e6_path), use_container_width=True, hide_index=True)
    if trials_path.exists():
        with st.expander(tr("E7. Full Trial Ledger (trials.csv — never delete a row)", "E7. Полный журнал испытаний (trials.csv — строки не удалять)")):
            st.dataframe(pd.read_csv(trials_path), use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 4: NETWORK & LOOPS (REUSED FROM ASSIGNMENT 1)
# ===========================================================================
if False:  # Legacy Network & Loops tab retired; evidence now lives in Tab 1.
    st.header("AI Ecosystem Relationship Graph & Directed Cycles")
    left_n, right_n = st.columns([1, 2])
    with left_n:
        st.metric("Directed cycles found", cycle_summary_top["cycle_count"])
        st.metric("Companies in cycles", cycle_summary_top["companies_in_cycles"])
        st.metric("Links in selected graph", cycle_summary_top["relationship_count"])
        st.write(
            "Reused from Assignment 1 and expanded with Oct–Nov 2026 provisional relationships "
            "(Broadcom–Anthropic $63B SPV, Chipmakers–Meta $5B cornerstone stake, Google–Fluidstack, Applied Digital, Galaxy)."
        )
    with right_n:
        if cycle_table_top.empty:
            st.info("No directed cycles found with current filter.")
        else:
            cycle_disp = describe_cycle_amounts(
                companies, relationships, cycle_table_top,
                confirmed_only=confirmed_only_global, data_dictionary=dictionary,
            )
            st.dataframe(cycle_disp, use_container_width=True, hide_index=True)

    net_df = prepare_relationships(companies, relationships, confirmed_only=confirmed_only_global)
    net_df = net_df.dropna(subset=["source_company", "target_company"])
    if not net_df.empty:
        G = nx.DiGraph()
        for _, row in net_df.iterrows():
            G.add_edge(str(row["source_company"]), str(row["target_company"]))
        pos = nx.spring_layout(G, seed=42, k=1.35)

        arc_x, arc_y, hover_x, hover_y, hover_txt = [], [], [], [], []
        pair_cnt = {}
        for _, row in net_df.iterrows():
            s, t = str(row["source_company"]), str(row["target_company"])
            if s not in pos or t not in pos:
                continue
            rep = pair_cnt.get((s, t), 0)
            pair_cnt[(s, t)] = rep + 1
            pts = curved_arc_points(*pos[s], *pos[t], rep)
            arc_x.extend([p[0] for p in pts] + [None])
            arc_y.extend([p[1] for p in pts] + [None])
            mx, my = arc_midpoint(*pos[s], *pos[t], rep)
            hover_x.append(mx)
            hover_y.append(my)
            amt_str = format_money(row["amount"]) if pd.notna(row.get("amount")) else "Not disclosed"
            hover_txt.append(f"<b>{s} → {t}</b><br>Type: {row.get('relationship_type')}<br>Amount: {amt_str}<br>Status: {row.get('evidence_status')}")

        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(x=arc_x, y=arc_y, mode="lines", line=dict(width=1.5, color="#64748B"), hoverinfo="skip", showlegend=False))
        fig_net.add_trace(go.Scatter(x=hover_x, y=hover_y, mode="markers", marker=dict(size=12, color="rgba(0,0,0,0.01)"), customdata=hover_txt, hovertemplate="%{customdata}<extra></extra>", showlegend=False))
        fig_net.add_trace(go.Scatter(
            x=[pos[n][0] for n in G.nodes()],
            y=[pos[n][1] for n in G.nodes()],
            mode="markers+text",
            text=list(G.nodes()),
            textposition="top center",
            marker=dict(size=20, color="#1D4ED8", line=dict(width=1, color="white")),
            hovertext=[f"<b>{n}</b><br>Degree: {G.degree(n)}" for n in G.nodes()],
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        ))
        fig_net.update_layout(
            height=580, plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_net, use_container_width=True)

# ===========================================================================
# TAB 5: DAILY AI NEWS & SENTIMENT AGENT (10:00 ALMATY)
# ===========================================================================
with tab_news:
    news_items = news_feed.get("items", [])
    verified_count = sum(bool(item.get("include_in_base_forecast", False)) for item in news_items)
    flagged_count = len(news_items) - verified_count
    st.info(tr(
        f"Live feed contains {verified_count} verified items and {flagged_count} flagged rumor items. The latest scheduled run is {news_feed.get('last_run_almaty', 'N/A')}; verified sentiment feeds the contagion scenario.",
        f"В текущей ленте {verified_count} проверенных материалов и {flagged_count} отмеченных как слух материалов. Последний запуск по расписанию: {news_feed.get('last_run_almaty', 'N/A')}; проверенное настроение передаётся в сценарий заражения.",
    ))
    st.header(tr("Daily AI News Intelligence & Credibility Agent", "ИИ-агент новостей и проверки достоверности"))
    st.info(tr(
        "At 10:00 AM Almaty time, the agent separates verified institutional and primary sources (Tier 1–2) from unverified rumors (Tier 3). Tier-3 items are quarantined with a warning badge; verified sentiment feeds Tab 1.",
        "В 10:00 по Алматы агент отделяет проверенные институциональные и первичные источники (Tier 1–2) от непроверенных слухов (Tier 3). Tier 3 помещаются в карантин с предупреждающим значком, а проверенное настроение передаётся во вкладку 1."
    ))
    st.caption(tr("Why this matters: This tab separates verified AI news from rumors and shows how credible signals can change the stress test.", "Зачем это нужно: агент отделяет проверенные новости ИИ от слухов и показывает, как надёжные сигналы меняют стресс-тест."))
    st.write(tr(
        "Runs automatically every day at **10:00 AM Asia/Almaty (05:00 UTC)** via GitHub Actions "
        "(`.github/workflows/daily_news_agent.yml`), scanning AI infrastructure, circular financing, and data-center "
        "credit headlines. Every item is graded into **Tier 1 (Verified Primary/Institutional)**, **Tier 2 (Trade Press)**, "
        "or **Tier 3 (⚠️ Unverified / Rumor — excluded from the base contagion model)**.",
        "Агент запускается каждый день в **10:00 по Алматы (05:00 UTC)** через GitHub Actions. Каждая новость получает уровень: **Tier 1 (проверенный первичный или институциональный источник)**, **Tier 2 (отраслевая пресса)** или **Tier 3 (⚠️ непроверенный слух — исключён из базовой модели заражения)**."
    ))

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button(tr("Run Live News Scan Now", "Запустить проверку новостей сейчас")):
            with st.spinner(tr("Scanning RSS feeds & grading source credibility...", "Сканируем RSS и оцениваем надёжность источников…")):
                news_feed = run_daily_update()
                st.success(f"{tr('Updated at', 'Обновлено')}: {news_feed['last_run_almaty']}")
    with col_info:
        st.caption(f"{tr('Last scheduled run', 'Последний запуск по расписанию')}: **{news_feed.get('last_run_almaty', 'N/A')}** | {tr('Schedule', 'Расписание')}: {news_feed.get('schedule', tr('Daily 10:00 Asia/Almaty', 'Ежедневно в 10:00 по Алматы'))}")

    # Show per-company sentiment summary
    cid_to_name = dict(zip(companies["company_id"].astype(str), companies["company_name"].astype(str)))
    if sentiment_map:
        sent_df = pd.DataFrame([
            {"Company": cid_to_name.get(cid, cid), "News Sentiment (-1 to +1)": sc}
            for cid, sc in sentiment_map.items()
        ]).sort_values("News Sentiment (-1 to +1)")
        st.subheader(tr("Active News Sentiment by Company (Feeds Contagion Simulator)", "Текущее настроение новостей по компаниям (передаётся в симулятор)"))
        fig_sent = px.bar(
            sent_df, x="News Sentiment (-1 to +1)", y="Company", orientation="h",
            color="News Sentiment (-1 to +1)", color_continuous_scale="RdYlGn", range_color=[-1, 1],
        )
        fig_sent.update_layout(height=340, yaxis_title="")
        st.plotly_chart(fig_sent, use_container_width=True)

    st.subheader(tr("Daily Filtered Intelligence Feed", "Ежедневная лента проверенной информации"))
    for item in news_feed.get("items", []):
        badge = item.get("credibility_badge", "")
        title = item.get("title", "")
        src = item.get("source_name", "")
        dt = item.get("published_date", "")
        sc = item.get("sentiment_score", 0.0)
        affected = ", ".join(cid_to_name.get(c, c) for c in item.get("affected_company_ids", []))
        if not item.get("include_in_base_forecast", False):
            st.warning(
                f"**{badge} · {title}**  \n"
                f"Source: *{src}* ({dt}) | Sentiment: `{sc:+.2f}` | Entities: {affected}  \n"
                f"{item.get('analytical_takeaway', '')}"
            )
        else:
            st.markdown(
                f"**{badge} · [{title}]({item.get('source_url', '#')})**  \n"
                f"Source: *{src}* ({dt}) | Sentiment: `{sc:+.2f}` | Entities: {affected}  \n"
                f"{item.get('analytical_takeaway', '')}"
            )
            st.divider()

# ===========================================================================
# TABS 5-7: CIRCULAR ALPHA v3 (Bubble Radar, Domino & S&P 500, 2000–2002 Analog)
# ===========================================================================
_v3_context: dict = {}
with tab_bubble:
    _v3_context.update(render_bubble_tab(tr) or {})
with tab_domino:
    _v3_context.update(render_domino_tab(tr) or {})
with tab_analog:
    _v3_context.update(render_analog_tab(tr) or {})

# Floating offline guide: values are assembled from the live Tab 1 scenario.
_mascot_context = {"current_tab_insights": "Cluster Lens, Rule Lab, and News Agent are available in the tabs."}
if "selected_name" in locals():
    _mascot_context.update({"selected_company": selected_name, "shock_pct": shock_pct})
if "impact_df" in locals() and not impact_df.empty:
    _mascot_context["top_impacted_company"] = str(impact_df.iloc[0].get("company_name", "connected companies"))
if "impact_summary" in locals():
    _mascot_context["tab_1_insights"] = f"{impact_summary.get('hop1_count', 0)} direct and {impact_summary.get('hop2_plus_count', 0)} indirect impacts"
if "impact_summary" in locals():
    _mascot_context["top_impacted_dollar_channels"] = {
        "equity_write_downs": impact_summary.get("total_equity_writedown_usd", "unavailable"),
        "revenue_backlog": impact_summary.get("total_revenue_backlog_usd", "unavailable"),
        "guarantees_spv": impact_summary.get("total_guarantee_call_usd", "unavailable"),
    }
if "hier" in locals():
    _mascot_context["cluster_lens"] = {"effective_bets": hier.get("effective_independent_bets", "unavailable"), "ARI": locals().get("ari_time", "unavailable")}
if "e2" in locals():
    _mascot_context["rule_lab"] = {"Sharpe_difference": e2.get("sharpe_diff", "unavailable"), "CI": [e2.get("ci_low", "unavailable"), e2.get("ci_high", "unavailable")]}
_mascot_context["verified_news_count"] = sum(bool(item.get("include_in_base_forecast", False)) for item in news_feed.get("items", []))
_mascot_context.update(_v3_context)  # v3: appended keys only; existing keys unchanged
st.session_state.alpha_context = _mascot_context
render_mascot(lang, _mascot_context)
