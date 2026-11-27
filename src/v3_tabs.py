"""Rendering for the three v3 tabs (Bubble Radar, Domino & S&P 500, 2000–2002 Analog).

Kept out of ``app.py`` so the existing tabs, theme and Alpha-chan hooks stay
untouched; ``app.py`` only calls ``render_*`` inside the new tab containers.
Every number shown is computed here from ``data/raw`` / the research pack, or
read from a source-cited constant whose source is printed in a caption.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import load_all_prices, load_companies, load_financials, load_relationships
from src import bubble_monitor as bm
from src import credit_signals as cs
from src import historical_analog as ha
from src import macro_contagion as mc
from src.extended_data import (
    CORE_TICKERS,
    all_statuses,
    equal_weight_index,
    equal_weight_volume,
    fred_series_bps,
    load_fred,
    load_long_panel,
    wide,
)

# Same palette as app.py's Plotly defaults / existing charts.
PURPLE, LILAC, INDIGO, ROSE = "#4B279B", "#AB8FBB", "#59529C", "#D9385A"
BLUE, SLATE, RED_ = "#1D4ED8", "#64748B", "#B91C1C"
LIGHT = {"RED": "🔴", "AMBER": "🟠", "GREEN": "🟢", "N/A": "⚪"}

T = Callable[[str, str], str]
BUCKET_RU = {
    "Short / Avoid (high fragility)": "Шорт / избегать (высокая хрупкость)",
    "Tactical underweight / hedge (guarantee exposure)": "Тактический недовес / хедж (риск гарантий)",
    "Neutral (hedge rule inactive)": "Нейтрально (правило хеджа неактивно)",
    "Neutral / monitor": "Нейтрально / наблюдать",
    "Defensive / relative long (fortress balance sheet)": "Защитный / относительный лонг (крепкий баланс)",
    "Avoid as long (debt concentration)": "Не держать в лонг (концентрация долга)",
    "Monitor (private-credit markdown risk)": "Наблюдать (риск переоценки частного кредита)",
}

CITES = {
    "psy": "Phillips, Shi & Yu (2015), *International Economic Review* 56(4), 1043–1078",
    "gsy": "Greenwood, Shleifer & You (2019), *Journal of Financial Economics* 131(1), 20–43",
    "gz": "Gilchrist & Zakrajšek (2012), *American Economic Review* 102(4), 1692–1720",
    "or": "Ofek & Richardson (2003), *Journal of Finance* 58(3), 1113–1137",
    "aot": "Acemoglu, Ozdaglar & Tahbaz-Salehi (2015), *American Economic Review* 105(2), 564–608",
    "dr": "Battiston et al. (2012), *Scientific Reports* 2, 541",
    "dy": "Diebold & Yilmaz (2014), *Journal of Econometrics* 182(1), 119–134",
    "bglp": "Billio, Getmansky, Lo & Pelizzon (2012), *Journal of Financial Economics* 104(3), 535–559",
}


# ---------------------------------------------------------------------------
# Cached data & heavy computations
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _panels():
    core, s1 = load_long_panel("volumes_core")
    if core.empty:
        # Offline fallback: legacy per-ticker CSV snapshot (prices only, no volume).
        legacy = load_all_prices()
        if not legacy.empty:
            core = legacy[["date", "ticker", "adjusted_close"]].assign(volume=np.nan)
            core["date"] = pd.to_datetime(core["date"])
    spill, s2 = load_long_panel("spillover_tiers")
    dot, s3 = load_long_panel("dotcom")
    fred, s4 = load_fred()
    return core, spill, dot, fred, [s1, s2, s3, s4]


@st.cache_data(show_spinner=False)
def _pack():
    return cs.load_research_pack_safe()


@st.cache_data(show_spinner=False)
def _cv(n_obs: int, min_window: int, n_sims: int):
    return bm.simulate_bsadf_critical_values(n_obs, min_window=min_window, n_sims=n_sims)


@st.cache_data(show_spinner=False)
def _calibration():
    _, _, dot, _, _ = _panels()
    if dot.empty:
        return None
    ix = dot[dot["ticker"] == "^IXIC"].set_index("date")
    return bm.calibrate_on_dotcom(ix["adjusted_close"], ix["adjusted_close"] * ix["volume"])


@st.cache_data(show_spinner=False)
def _rolling_tci(tier_key: str, window: int, p: int, H: int):
    R = _tier_returns(tier_key)
    return mc.rolling_tci(R, p=p, H=H, window=window, step=5)


def _tier_returns(tier_key: str) -> pd.DataFrame:
    core, spill, _, _, _ = _panels()
    pw = pd.concat([wide(core), wide(spill)], axis=1)
    tiers = mc.TIERS_CORE if tier_key == "core" else mc.TIERS_EXTENDED
    return mc.tier_returns(pw, tiers)


def _ai_core():
    core, _, _, fred, _ = _panels()
    pw = wide(core)
    dv = wide(core.assign(dv=core["adjusted_close"] * core["volume"]), "dv")
    ai = equal_weight_index(pw, CORE_TICKERS)
    aidv = equal_weight_volume(dv, CORE_TICKERS).reindex(ai.index)
    return pw, ai, aidv, fred_series_bps(fred)


def _missing_banner(tr: T, statuses) -> bool:
    missing = [s for s in statuses if not s.ok]
    for s in missing:
        st.warning(tr(f"⚠️ Live series unavailable — {s.message}", f"⚠️ Ряд данных недоступен — {s.message}"))
    return bool(missing)


def _fmt_pct(x: float) -> str:
    return f"{x:+.1f}%" if np.isfinite(x) else "n/a"


# Plain-language meaning of each trigger (what a RED/AMBER light means in real life).
TRIGGER_STORY = {
    "hy_hpc": ("Lenders to AI data-center builders now demand {cur} over Treasuries, {chg} in one month. Junk-bond buyers are pricing real default risk.",
               "Кредиторы строителей ИИ-ЦОД требуют {cur} сверх казначейских облигаций, {chg} за месяц. Покупатели мусорных облигаций закладывают реальный риск дефолта."),
    "basis": ("Even Big Tech's own 10-year+ bonds pay {cur} more than comparable investment-grade bonds — investors want extra pay to lend long-term for AI capex.",
              "Даже 10+-летние облигации крупнейших техкомпаний платят на {cur} больше сопоставимых облигаций инвестиционного уровня — инвесторы хотят надбавку за долгосрочное финансирование ИИ-капвложений."),
    "crwv": ("CoreWeave bonds at {cur} = lenders demanding distress-level interest from the biggest neocloud.",
             "Облигации CoreWeave на уровне {cur} = кредиторы требуют от крупнейшего неооблака процент уровня бедствия."),
    "gsadf": ("AI stock prices are going vertical like a classic bubble ({cur}).",
              "Цены ИИ-акций уходят в вертикальный рост, как в классическом пузыре ({cur})."),
    "supply": ("A flood of new shares (Anthropic IPO filing, Meta's $25B offering) while banks refuse to fully underwrite GPU loans — the pattern that ended the 2000 boom.",
               "Поток новых акций (заявка Anthropic на IPO, размещение Meta на $25 млрд) при отказе банков полностью гарантировать GPU-кредиты — именно так закончился бум 2000 года."),
}


def _status_caption(tr: T, statuses) -> None:
    ok = " · ".join(s.message for s in statuses if s.ok)
    if ok:
        st.caption(tr(f"Data: {ok}. Source: Yahoo Finance (yfinance) & FRED via scripts/fetch_extended_data.py.",
                      f"Данные: {ok}. Источник: Yahoo Finance (yfinance) и FRED через scripts/fetch_extended_data.py."))


# ---------------------------------------------------------------------------
# Tab A — Bubble Radar & Triggers
# ---------------------------------------------------------------------------
def render_bubble_tab(tr: T) -> dict:
    st.header(tr("Bubble Radar & Triggers", "Радар пузыря и триггеры"))
    verdict_box = st.container()   # filled at the end, once every metric below is computed
    st.info(tr(
        "**The question:** is AI-stock pricing behaving like a bubble, how likely is a crash, and which objective credit signals would tell us it is starting? "
        "Three lenses: a statistical *explosiveness* test on prices, a crash-risk score built from the four warning signs found in a century of industry booms, and a five-light credit & supply checklist.",
        "**Вопрос:** ведут ли себя цены ИИ-акций как пузырь, насколько вероятен обвал и какие объективные кредитные сигналы покажут, что он начался? "
        "Три линзы: статистический тест на *взрывной* рост цен, индекс риска обвала по четырём признакам из столетней истории отраслевых бумов и светофор из пяти кредитных сигналов и сигналов предложения."))
    core, _, dot, fred, statuses = _panels()
    if _missing_banner(tr, statuses[:1]) and core.empty:
        return {}
    pw, ai, aidv, hy = _ai_core()
    pack, pack_msg = _pack()

    c1, c2, c3 = st.columns(3)
    with c1:
        series_choice = st.radio(tr("Price series tested", "Тестируемый ряд"), [tr("AI Core (equal-weight 12)", "Ядро ИИ (12 акций, равные веса)"), "SPY (S&P 500)"], horizontal=True)
    with c2:
        n_sims = st.select_slider(tr("Monte Carlo paths for critical values", "Число Монте-Карло траекторий для критических значений"), options=[100, 200, 400], value=200)
    with c3:
        st.caption(tr("Lag order K = 0 · minimum window r₀ = 0.01 + 1.8/√T · seed 20151043 (fixed)", "Порядок лагов K = 0 · минимальное окно r₀ = 0.01 + 1.8/√T · seed 20151043 (фиксирован)"))
    price = ai if series_choice.startswith(("AI", "Ядро")) else pw["SPY"].dropna()
    w0 = bm.psy_min_window(len(price))
    cv = _cv(len(price), w0, n_sims)
    ex = bm.run_exuberance_test(price, cv=cv)
    last = ex.table.iloc[-1]

    # ---- crash score
    calib = _calibration()
    feats = bm.gsy_features(ai, aidv, hy if not hy.empty else None, pw.get("SPY"))
    st.subheader(tr("1 · Crash-risk score (Greenwood–Shleifer–You attributes)", "1 · Индекс риска обвала (признаки Гринвуда–Шлейфера–Ю)"))
    wc = st.columns(4)
    labels = [("z_vol", tr("Volatility weight", "Вес волатильности")), ("z_turn", tr("Turnover weight", "Вес оборота")),
              ("z_acc", tr("Acceleration weight", "Вес ускорения")), ("z_cred", tr("Credit (HY OAS) weight", "Вес кредита (HY OAS)"))]
    weights = {k: wc[i].slider(lab, 0.0, 1.0, bm.DEFAULT_WEIGHTS[k], 0.05, key=f"v3w_{k}") for i, (k, lab) in enumerate(labels)}
    risk = bm.crash_risk_table(feats, calib, weights) if calib else None

    k1, k2, k3, k4 = st.columns(4)
    if risk is not None and risk["score"].notna().any():
        rlast = risk.dropna(subset=["score"]).iloc[-1]
        k1.metric(tr("Composite stress score Sₜ", "Сводный индекс стресса Sₜ"), f"{rlast['score']:+.2f} σ")
        k2.metric(tr("Calibrated crash-risk index", "Калиброванный индекс риска обвала"), f"{100 * rlast['p_crash']:.0f}%",
                  help=tr("Probability of a ≥40% fall within 24 months under a logistic link fitted on the Nasdaq 1998–2003 episode. Upward-biased level — read with the odds multiplier.",
                          "Вероятность падения ≥40% в течение 24 месяцев по логистической связи, откалиброванной на эпизоде Nasdaq 1998–2003. Уровень завышен — смотрите вместе с множителем шансов."))
        k3.metric(tr("Crash odds vs. average day", "Шансы обвала vs средний день"), f"×{rlast['odds_mult']:.2f}")
        k4.metric(tr("2-year run-up (raw / net of SPY)", "Рост за 2 года (абс. / сверх SPY)"),
                  f"{100 * rlast['runup_2y']:.0f}% / {100 * rlast.get('runup_2y_net', np.nan):.0f}%",
                  tr("GSY boom condition: ≥100%", "Условие бума GSY: ≥100%"), delta_color="off")
        zcols = [c for c in ["z_vol", "z_turn", "z_acc", "z_cred"] if c in risk.columns]
        zdf = risk[zcols + ["score"]].dropna(subset=["score"]).reset_index().melt(id_vars="date", var_name="Attribute", value_name="z")
        names = {"z_vol": tr("Volatility (63d)", "Волатильность (63д)"), "z_turn": tr("Dollar-volume surge (21d/252d)", "Всплеск долларового объёма (21д/252д)"),
                 "z_acc": tr("Acceleration (last vs prior 126d)", "Ускорение (последние vs предыдущие 126д)"), "z_cred": tr("HY OAS credit stress", "Кредитный стресс HY OAS"), "score": "Sₜ"}
        zdf["Attribute"] = zdf["Attribute"].map(names)
        fig = px.line(zdf, x="date", y="z", color="Attribute", color_discrete_sequence=[LILAC, INDIGO, SLATE, ROSE, PURPLE])
        fig.update_layout(height=360, hovermode="x unified", yaxis_title=tr("z-score (trailing 504d)", "z-оценка (скользящие 504 дня)"), xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.info(tr(
            f"**What does this mean?** The **{100 * rlast['p_crash']:.0f}%** headline looks alarming, but read it carefully: the model learned from the 1999–2001 Nasdaq, "
            f"when **{calib.positive_rate:.0%}** of days were followed by a ≥40% crash — so its starting level is high by construction. "
            f"The honest number is the odds: today's mix of volatility, trading frenzy, price acceleration and credit stress makes a crash about **×{rlast['odds_mult']:.2f} as likely as on an average-stress day**. "
            f"Also note the AI basket rose {100 * rlast['runup_2y']:.0f}% in two years — " + ("below" if rlast["runup_2y"] < bm.GSY_RUNUP else "above") + " the 100% run-up that defines a classic boom in the research.",
            f"**Что это значит?** Заголовочные **{100 * rlast['p_crash']:.0f}%** выглядят пугающе, но читать их нужно осторожно: модель обучена на Nasdaq 1999–2001, "
            f"когда за **{calib.positive_rate:.0%}** дней следовал обвал ≥40%, поэтому её исходный уровень высок по построению. "
            f"Честная цифра — шансы: нынешнее сочетание волатильности, ажиотажа в торгах, ускорения цен и кредитного стресса делает обвал примерно **в ×{rlast['odds_mult']:.2f} раза вероятнее, чем в день со средним стрессом**. "
            f"Кроме того, ИИ-корзина выросла на {100 * rlast['runup_2y']:.0f}% за два года — " + ("ниже" if rlast["runup_2y"] < bm.GSY_RUNUP else "выше") + " порога роста 100%, которым в исследовании определяется классический бум."))
        st.caption(tr(
            f"Calibration: θ₀ = {calib.theta0:.3f}, θ₁ = {calib.theta1:.3f}; n = {calib.n_obs} days ({calib.sample_start} → {calib.sample_end}), crash-label rate {calib.positive_rate:.0%}; attributes {', '.join(calib.attributes_used)}. {calib.note} "
            f"Credit attribute = ICE BofA US HY OAS (FRED BAMLH0A0HYM2), latest {hy.iloc[-1]:.0f} bps. Turnover uses dollar volume (shares outstanding not in data). Sources: {CITES['gsy']}; {CITES['gz']}.",
            f"Калибровка: θ₀ = {calib.theta0:.3f}, θ₁ = {calib.theta1:.3f}; n = {calib.n_obs} дней ({calib.sample_start} → {calib.sample_end}), доля меток обвала {calib.positive_rate:.0%}; признаки {', '.join(calib.attributes_used)}. "
            f"Калибровка по одному эпизоду (Nasdaq ^IXIC) с перекрывающимися 24-месячными метками; выборка окружает обвал 2000 года, поэтому уровень вероятности завышен, информативен наклон θ₁. Это калиброванный индекс стресса, а не прогноз. "
            f"Кредитный признак — ICE BofA US HY OAS (FRED BAMLH0A0HYM2), последнее значение {hy.iloc[-1]:.0f} б.п. Оборот — долларовый объём (числа акций нет в данных). Источники: {CITES['gsy']}; {CITES['gz']}.") if not hy.empty else "")
    else:
        st.warning(tr("Not enough history to compute the crash-risk score.", "Недостаточно истории для расчёта индекса риска обвала."))

    # ---- BSADF chart
    st.subheader(tr("2 · Explosive-price detector (BSADF / GSADF)", "2 · Детектор взрывного роста цен (BSADF / GSADF)"))
    e1, e2, e3 = st.columns(3)
    e1.metric("BSADFₜ", f"{last['bsadf']:.2f}", tr(f"CV95 {last['cv95']:.2f}", f"КЗ95 {last['cv95']:.2f}"), delta_color="off")
    e2.metric("GSADF", f"{ex.gsadf:.2f}", tr(f"CV95 {ex.gsadf_cv95:.2f}", f"КЗ95 {ex.gsadf_cv95:.2f}"), delta_color="off")
    e3.metric(tr("Regime today", "Режим сегодня"), tr("EXPLOSIVE", "ВЗРЫВНОЙ") if last["explosive"] else tr("Not explosive", "Не взрывной"))
    tb = ex.table.dropna(subset=["bsadf"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tb["date"], y=tb["bsadf"], name="BSADFₜ", line=dict(color=PURPLE, width=2)))
    fig.add_trace(go.Scatter(x=tb["date"], y=tb["cv95"], name=tr("95% critical value", "95% критическое значение"), line=dict(color=ROSE, dash="dash")))
    fig.add_trace(go.Scatter(x=tb["date"], y=tb["cv90"], name=tr("90% critical value (switch-off)", "90% критическое значение (выключение)"), line=dict(color=LILAC, dash="dot")))
    for _, epi in ex.episodes.iterrows():
        fig.add_vrect(x0=epi["start"], x1=epi["end"], fillcolor=ROSE, opacity=0.12, line_width=0)
    fig.update_layout(height=360, hovermode="x unified", xaxis_title="", yaxis_title="ADF t-stat")
    st.plotly_chart(fig, use_container_width=True)
    above = bool(last["bsadf"] > last["cv95"])
    st.info(tr(
        "**What does this mean?** This test checks whether prices are rising smoothly or **going vertical like a classic bubble**. "
        + (f"Today's reading ({last['bsadf']:.2f}) is **above** the red line ({last['cv95']:.2f}): prices are in a parabolic phase."
           if above else
           f"Today's reading ({last['bsadf']:.2f}) is **below** the red line ({last['cv95']:.2f}), meaning equities are **not** in a parabolic blow-off right now; the shaded band shows the last time they were."),
        "**Что это значит?** Тест проверяет, растут ли цены плавно или **уходят в вертикаль, как в классическом пузыре**. "
        + (f"Сегодняшнее значение ({last['bsadf']:.2f}) **выше** красной линии ({last['cv95']:.2f}): цены в параболической фазе."
           if above else
           f"Сегодняшнее значение ({last['bsadf']:.2f}) **ниже** красной линии ({last['cv95']:.2f}): акции сейчас **не** в параболическом взлёте; заштрихованная полоса показывает, когда это было в последний раз.")))
    epi_txt = "; ".join(f"{e.start.date()} → {e.end.date()} ({e.days}d)" for e in ex.episodes.itertuples()) or tr("none", "нет")
    st.caption(tr(
        f"Shaded = date-stamped explosive episodes (BSADF > CV95 for ≥5 days): {epi_txt}. Critical values: {cv['n_sims']} fixed-seed random-walk simulations of length {len(price)}; min window {ex.min_window} days. Signal used by the rules is lagged one day. Source: {CITES['psy']}.",
        f"Заштрихованы взрывные эпизоды (BSADF > КЗ95 не менее 5 дней): {epi_txt}. Критические значения: {cv['n_sims']} симуляций случайного блуждания с фиксированным seed, длина {len(price)}; минимальное окно {ex.min_window} дней. Сигнал для правил запаздывает на один день. Источник: {CITES['psy']}."))

    # ---- trigger checklist
    st.subheader(tr("3 · Early-warning trigger checklist", "3 · Светофор ранних предупреждений"))
    credit = cs.current_credit_inputs(pack)
    ai_ex = ex if series_choice.startswith(("AI", "Ядро")) else bm.run_exuberance_test(ai, cv=_cv(len(ai), bm.psy_min_window(len(ai)), n_sims))
    ai_last = ai_ex.table.iloc[-1]
    trig = bm.trigger_checklist(credit, ai_last["bsadf"], ai_last["cv95"], cs.supply_friction_flags())
    show = pd.DataFrame({
        tr("Status", "Статус"): [f"{LIGHT[s]} {s}" for s in trig["status"]],
        tr("Trigger", "Триггер"): trig["trigger_en"] if tr("en", "ru") == "en" else trig["trigger_ru"],
        tr("Current", "Сейчас"): trig["current"], tr("Rule", "Правило"): trig["rule"], tr("Source", "Источник"): trig["source"],
    })
    st.dataframe(show, use_container_width=True, hide_index=True)
    reds = int((trig["status"] == "RED").sum())
    ambers = int((trig["status"] == "AMBER").sum())
    lit = trig[trig["status"].isin(["RED", "AMBER"])]
    if not lit.empty:
        lines = [f"- {LIGHT[r.status]} **{r.trigger_en if tr('en', 'ru') == 'en' else r.trigger_ru}:** "
                 + tr(*TRIGGER_STORY[r.key]).format(cur=r.current, chg=f"{credit['hy_hpc_spread_bps'] - cs.HY_HPC_BASELINE_BPS:+.0f} bps") for r in lit.itertuples()]
        st.markdown(tr("**What the lit-up lights mean in real life:**", "**Что означают горящие сигналы в реальной жизни:**") + "\n" + "\n".join(lines))
    sw = bm.switch_off_status(credit["hy_hpc_spread_bps"], ai_last["bsadf"], ai_last["cv90"])
    s1, s2, s3 = st.columns(3)
    s1.metric(tr("RED triggers", "Красных триггеров"), f"{reds} / 5")
    s2.metric(tr("Short/hedge rule", "Правило шорта/хеджа"), tr("OFF (switched off)", "ВЫКЛ") if sw["switch_off"] else tr("ON", "ВКЛ"))
    s3.metric(tr("HY HPC vs 1-month-ago baseline", "HY HPC vs база месяц назад"), f"{credit['hy_hpc_spread_bps']:.0f} bps",
              f"+{credit['hy_hpc_spread_bps'] - cs.HY_HPC_BASELINE_BPS:.0f} bps vs {cs.HY_HPC_BASELINE_BPS:.0f}", delta_color="inverse")
    st.info(tr(
        "**Switch-off rule (RULE_CARD v3.0):** the short/hedge stance turns OFF only when **both** HY HPC spreads compress below 350 bps **and** AI-core BSADFₜ falls below its 90% critical value. "
        f"Today: credit normalised = {sw['credit_normalised']}, equity exuberance normalised = {sw['equity_normalised']}.",
        "**Правило выключения (RULE_CARD v3.0):** позиция шорт/хедж выключается, только когда **одновременно** спред HY HPC сжался ниже 350 б.п. **и** BSADFₜ ядра ИИ опустился ниже 90% критического значения. "
        f"Сегодня: кредит нормализовался = {sw['credit_normalised']}, взрывной рост акций нормализовался = {sw['equity_normalised']}."))
    st.caption(tr(f"Credit levels — {pack_msg}. Supply-friction events per {CITES['or']}.", f"Кредитные уровни — {pack_msg}. События предложения акций — по {CITES['or']}."))

    # ---- tenor curve & bond matrix (pack)
    if pack is not None:
        st.subheader(tr("4 · Hyperscaler credit repricing (J.P. Morgan, Oct 29, 2026)", "4 · Переоценка кредита гиперскейлеров (J.P. Morgan, 29 октября 2026)"))
        cl, cr_ = st.columns(2)
        with cl:
            tc = pack.tenor_curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tc["tenor"], y=tc["juli_bps"], name=tr("JULI index", "Индекс JULI"), mode="lines+markers", line=dict(color=SLATE)))
            fig.add_trace(go.Scatter(x=tc["tenor"], y=tc["hyperscaler_bps"], name=tr("Hyperscalers", "Гиперскейлеры"), mode="lines+markers+text",
                                     text=[f"+{p:.1f}" for p in tc["premium_bps"]], textposition="top center", line=dict(color=PURPLE)))
            fig.update_layout(height=340, yaxis_title=tr("Spread to UST (bps)", "Спред к UST (б.п.)"), title=tr("Spread by tenor", "Спред по срокам"),
                              yaxis_range=[50, float(tc["hyperscaler_bps"].max()) + 25])
            st.plotly_chart(fig, use_container_width=True)
        with cr_:
            b = pack.bond_matrix.melt(id_vars="issuer", value_vars=["direct_ig_bn", "dc_ig_bn", "dc_hy_bn"], var_name="Type", value_name="$bn")
            b["Type"] = b["Type"].map({"direct_ig_bn": tr("Direct IG", "Прямой IG"), "dc_ig_bn": tr("DC landlord IG", "IG арендодателей ЦОД"), "dc_hy_bn": tr("DC landlord HY", "HY арендодателей ЦОД")})
            fig = px.bar(b, y="issuer", x="$bn", color="Type", orientation="h", color_discrete_sequence=[PURPLE, LILAC, ROSE],
                         title=tr(f"Tracked debt: ${pack.bond_matrix['total_bn'].sum():.0f}B", f"Отслеживаемый долг: ${pack.bond_matrix['total_bn'].sum():.0f} млрд"))
            fig.update_layout(height=340, yaxis={"categoryorder": "total ascending"}, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        st.caption(tr(f"Source: {cs.JPM_OCT29}; tenor premium recomputed as hyperscaler − JULI (end-2025 10y+ premium: +{cs.SNAPSHOT['hyperscaler_10y_basis_end2025_bps'].value} bps). Component sums give Other $110B / MSFT $45B; J.P. Morgan prints $111B / $44B (rounding).",
                      f"Источник: {cs.JPM_OCT29}; премия по срокам пересчитана как гиперскейлеры − JULI (премия 10+ лет на конец 2025: +{cs.SNAPSHOT['hyperscaler_10y_basis_end2025_bps'].value} б.п.). Суммы компонентов: Прочие $110 млрд / MSFT $45 млрд; у J.P. Morgan напечатано $111 / $44 млрд (округление)."))
    with verdict_box:
        equity_hot = bool(ai_last["bsadf"] > ai_last["cv95"])
        hedge_on = not sw["switch_off"]
        box = st.warning if hedge_on else st.info
        box(tr(
            "### ⏱️ Executive verdict in 30 seconds\n"
            + (f"**Status right now:** AI **stock prices are not in an explosive melt-up** (bubble test {ai_last['bsadf']:.2f}, below the {ai_last['cv95']:.2f} alarm line), "
               if not equity_hot else
               f"**Status right now:** AI **stock prices are in an explosive, bubble-like phase** (bubble test {ai_last['bsadf']:.2f}, above the {ai_last['cv95']:.2f} alarm line), ")
            + f"**but the bond/credit market is already flashing stress: {reds} of 5 triggers are RED and {ambers} AMBER.** "
            + ("So the defensive / hedge rule stays **ON**.\n\n" if hedge_on else "The defensive / hedge rule is **OFF**.\n\n")
            + f"**Why this matters:** bond investors usually panic before stock investors. When neocloud bonds like CoreWeave's blow out past {credit['crwv_2031_spread_bps']:,.0f} bps "
              "and banks pull back from underwriting GPU loans, cracks are forming beneath the surface even while share prices look calm.\n\n"
            + "**Roadmap:** **Step 1** — check the alarm (this tab) → **Step 2** — see how the domino hits the S&P 500 (next tab) → **Step 3** — compare with 2000 and see who to short / avoid (last tab).",
            "### ⏱️ Вердикт за 30 секунд\n"
            + (f"**Статус сейчас:** **цены ИИ-акций не во взрывном росте** (тест на пузырь {ai_last['bsadf']:.2f}, ниже линии тревоги {ai_last['cv95']:.2f}), "
               if not equity_hot else
               f"**Статус сейчас:** **цены ИИ-акций во взрывной, пузыреподобной фазе** (тест на пузырь {ai_last['bsadf']:.2f}, выше линии тревоги {ai_last['cv95']:.2f}), ")
            + f"**но рынок облигаций и кредита уже подаёт сигналы стресса: {reds} из 5 триггеров КРАСНЫЕ и {ambers} ЖЁЛТЫЕ.** "
            + ("Поэтому защитное правило / хедж остаётся **ВКЛЮЧЁННЫМ**.\n\n" if hedge_on else "Защитное правило / хедж **ВЫКЛЮЧЕНО**.\n\n")
            + f"**Почему это важно:** держатели облигаций обычно паникуют раньше акционеров. Когда облигации неооблаков вроде CoreWeave улетают за {credit['crwv_2031_spread_bps']:,.0f} б.п., "
              "а банки отказываются гарантировать GPU-кредиты, под поверхностью появляются трещины, даже если котировки акций выглядят спокойно.\n\n"
            + "**Маршрут:** **Шаг 1** — проверьте сигнал тревоги (эта вкладка) → **Шаг 2** — посмотрите, как домино бьёт по S&P 500 (следующая вкладка) → **Шаг 3** — сравните с 2000 годом и узнайте, кого шортить / избегать (последняя вкладка)."))
    with st.expander(tr("How this model works (for methodology defense)", "Как работает модель (для защиты методологии)")):
        st.markdown(tr(
            f"- **Explosiveness:** right-tailed ADF regressions Δyₜ = α + βyₜ₋₁ + εₜ over every backward window; BSADFₜ is the supremum, compared with Monte Carlo random-walk critical values ({CITES['psy']}).\n"
            f"- **Crash-risk score:** Sₜ = Σ wₖZₖ over volatility, dollar-volume turnover, acceleration and HY OAS; P = σ(θ₀ + θ₁Sₜ) fitted on Nasdaq 1999–2001 complete cases ({CITES['gsy']}; {CITES['gz']}).\n"
            f"- **Triggers:** pre-registered thresholds from RULE_CARD v3.0 on J.P. Morgan Oct 29 / Nov 4 2026 levels; supply friction per {CITES['or']}.\n"
            "- **No look-ahead:** every statistic uses data ≤ t and every trading signal is lagged one day (canary tests in tests/).",
            f"- **Взрывной рост:** правосторонние ADF-регрессии Δyₜ = α + βyₜ₋₁ + εₜ на всех обратных окнах; BSADFₜ — супремум, сравнивается с критическими значениями Монте-Карло для случайного блуждания ({CITES['psy']}).\n"
            f"- **Индекс риска обвала:** Sₜ = Σ wₖZₖ по волатильности, долларовому обороту, ускорению и HY OAS; P = σ(θ₀ + θ₁Sₜ), оценка на полных наблюдениях Nasdaq 1999–2001 ({CITES['gsy']}; {CITES['gz']}).\n"
            f"- **Триггеры:** заранее заданные пороги RULE_CARD v3.0 на уровнях J.P. Morgan 29 окт. / 4 нояб. 2026; трение предложения — по {CITES['or']}.\n"
            "- **Без заглядывания вперёд:** каждая статистика использует данные ≤ t, каждый торговый сигнал запаздывает на день (canary-тесты в tests/)."))
    _status_caption(tr, statuses)
    out = {"bubble_radar": {"red_triggers": reds, "bsadf_ai_core": round(float(ai_last["bsadf"]), 2), "cv95": round(float(ai_last["cv95"]), 2),
                            "hedge_rule_on": not sw["switch_off"]}}
    if risk is not None and risk["score"].notna().any():
        out["bubble_radar"].update({"stress_score": round(float(rlast["score"]), 2), "crash_risk_index_pct": round(100 * float(rlast["p_crash"]), 1), "odds_multiplier": round(float(rlast["odds_mult"]), 2)})
    st.session_state["v3_triggers"] = trig
    return out


# ---------------------------------------------------------------------------
# Tab B — Domino & S&P 500 forecast
# ---------------------------------------------------------------------------
def render_domino_tab(tr: T) -> dict:
    st.header(tr("Domino & S&P 500 Forecast", "Эффект домино и прогноз S&P 500"))
    st.info(tr(
        "**Why doesn't a neocloud problem stay inside AI?** Because the same few companies lend to, guarantee, and buy from each other — so a default at the edge lands on the balance sheets of the giants that back it. "
        "Small shocks get absorbed by this web; large ones travel through it (Acemoglu, Ozdaglar & Tahbaz-Salehi, 2015).\n\n"
        "**The four contagion rings:**\n"
        "1. **Neoclouds** (CoreWeave, Applied Digital, Galaxy) — heavily indebted GPU landlords; the first to crack.\n"
        "2. **Chipmakers who guaranteed their debt** (Broadcom's $63B Anthropic SPV, Nvidia's $105B Ohio backstop) — a default turns into their bill.\n"
        "3. **Hyperscalers & bond markets** — Big Tech loses customers and investments; credit spreads widen for everyone.\n"
        "4. **S&P 500 & power/utilities** — AI names are a large slice of the index, and data-center power buyers lose demand.",
        "**Почему проблема неооблака не остаётся внутри ИИ?** Потому что несколько одних и тех же компаний кредитуют друг друга, дают друг другу гарантии и покупают друг у друга — дефолт на периферии ложится на балансы гигантов, которые его поддерживают. "
        "Малые шоки эта сеть гасит, большие — передаёт дальше (Acemoglu, Ozdaglar & Tahbaz-Salehi, 2015).\n\n"
        "**Четыре кольца заражения:**\n"
        "1. **Неооблака** (CoreWeave, Applied Digital, Galaxy) — закредитованные арендодатели GPU; трескаются первыми.\n"
        "2. **Чипмейкеры, гарантировавшие их долг** (SPV Broadcom для Anthropic на $63 млрд, гарантия Nvidia в Огайо на $105 млрд) — дефолт превращается в их счёт.\n"
        "3. **Гиперскейлеры и рынки облигаций** — крупнейшие техкомпании теряют клиентов и инвестиции, кредитные спреды расширяются для всех.\n"
        "4. **S&P 500 и энергетика/коммунальные компании** — ИИ-акции составляют большую часть индекса, а покупатели электроэнергии для ЦОД теряют спрос."))
    core, spill, _, _, statuses = _panels()
    _missing_banner(tr, statuses[:2])
    comp, rel, fin = load_companies(), load_relationships(), load_financials()

    # ---- DebtRank
    st.subheader(tr("1 · DebtRank: who is too central to fail?", "1 · DebtRank: кто слишком централен, чтобы упасть?"))
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        conf = st.toggle(tr("Confirmed links only", "Только подтверждённые связи"), value=False, key="v3_conf")
    with a2:
        lgd_rev = st.slider(tr("Revenue/backlog loss-given-default", "Потеря по выручке/контрактам (LGD)"), 0.0, 1.0, mc.DEFAULT_LGD["revenue"], 0.05, key="v3_lgd")
    with a3:
        pass_through = st.toggle(tr("Private nodes pass distress through", "Частные узлы передают стресс дальше"), value=False, key="v3_pt",
                                 help=tr("Off: entities without SEC financials in financials.csv (OpenAI, Anthropic — S-1 filed Oct 2026, no price history yet — SB Energy, Fluidstack, APLD, GLXY, BX) cannot absorb measured distress, so nothing is imputed.",
                                         "Выкл.: у организаций без данных SEC в financials.csv (OpenAI, Anthropic — S-1 подана в октябре 2026, истории цен пока нет — SB Energy, Fluidstack, APLD, GLXY, BX) нет измеримого буфера, поэтому стресс им не приписывается."))
    with a4:
        dd_unit = st.slider(tr("Equity drawdown per unit of distress (%)", "Просадка акций на единицу стресса (%)"), 10, 100, 50, 5, key="v3_ddu")
    net = mc.build_exposure_network(comp, rel, fin, confirmed_only=conf, lgd={"revenue": lgd_rev}, no_buffer_mode="pass_through" if pass_through else "block")
    scores = mc.debtrank_scores(net)
    cl, cr_ = st.columns([1.1, 1])
    with cl:
        fig = px.bar(scores.head(10), x="debtrank", y="company", orientation="h", color="buffer_basis",
                     color_discrete_sequence=[PURPLE, LILAC, ROSE], title=tr("Systemic impact if the node fails (share of ecosystem buffer distressed)", "Системный эффект при отказе узла (доля буфера экосистемы под стрессом)"))
        fig.update_layout(height=380, yaxis={"categoryorder": "total ascending"}, yaxis_title="", xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with cr_:
        names = dict(zip(comp["company_name"], comp["company_id"]))
        default = [n for n in ["CoreWeave, Inc.", "Anthropic PBC"] if n in names]
        picked = st.multiselect(tr("Shock simulator: nodes that default", "Симулятор шока: узлы, объявившие дефолт"), list(names), default=default, key="v3_pick")
        sev = st.slider(tr("Initial distress h(0)", "Начальный стресс h(0)"), 0.1, 1.0, 1.0, 0.1, key="v3_sev")
        sim = mc.shock_simulation(net, {names[n]: sev for n in picked})
        sim = sim[sim["h_final"] > 0]
        fig = px.bar(sim, x="h_final", y="company", orientation="h", color_discrete_sequence=[ROSE], title=tr("Final distress h(T)", "Итоговый стресс h(T)"))
        fig.update_layout(height=300, yaxis={"categoryorder": "total ascending"}, yaxis_title="", xaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    st.caption(tr(
        f"W[i,j] = min(1, LGD·A[i,j]/Eⱼ) from data/raw/relationships_final.csv ({len(net.used_links)} links with disclosed amounts; {net.undisclosed_links} undisclosed links not imputed; superseded LOIs R001/R002 excluded). "
        f"Eⱼ = EBITDA, else cash (data/raw/financials.csv). LGD: equity/guarantee/credit = 1.0, revenue = {lgd_rev:.2f} (assumption). Source: {CITES['dr']}; {CITES['aot']}.",
        f"W[i,j] = min(1, LGD·A[i,j]/Eⱼ) по data/raw/relationships_final.csv ({len(net.used_links)} связей с раскрытой суммой; {net.undisclosed_links} нераскрытых не заполнялись; заменённые LOI R001/R002 исключены). "
        f"Eⱼ = EBITDA, иначе денежные средства (data/raw/financials.csv). LGD: акции/гарантии/кредит = 1.0, выручка = {lgd_rev:.2f} (допущение). Источник: {CITES['dr']}; {CITES['aot']}."))
    st.caption(tr(
        "**Anthropic status:** Anthropic filed its IPO registration (S-1) in October 2026 (WSJ, Oct 30, 2026), so it is moving from private to public. "
        "Any listing is too recent to give usable trading history, so it remains **excluded from the 2023–2026 daily price panel** and from every time-series model "
        "(BSADF, crash score, Diebold–Yilmaz, scenario weights). In DebtRank it stays a node without an SEC buffer until audited public financials are added to financials.csv.",
        "**Статус Anthropic:** в октябре 2026 года Anthropic подала заявку на IPO (S-1) (WSJ, 30 октября 2026), то есть переходит из частной компании в публичную. "
        "Возможный листинг слишком свежий, чтобы дать пригодную историю торгов, поэтому компания по-прежнему **исключена из дневной ценовой панели 2023–2026** и из всех моделей временных рядов "
        "(BSADF, индекс риска обвала, Diebold–Yilmaz, веса сценариев). В DebtRank она остаётся узлом без буфера SEC, пока в financials.csv не появится аудированная публичная отчётность."))
    top_dr = scores.iloc[0]
    hit = sim[sim["h0"] == 0].sort_values("h_final", ascending=False)
    hit_txt = ", ".join(f"{r.company} ({r.h_final:.0%})" for r in hit.head(4).itertuples()) or tr("no one else", "никого больше")
    st.info(tr(
        f"**What does this mean?** The left chart ranks who would do the most damage by failing: **{top_dr['company']}** tops the list because many firms have money tied to it. "
        f"The right chart is the simulator: if {', '.join(picked) or 'nobody'} default, the firms that lose the biggest share of their financial cushion are {hit_txt}. "
        "100% means a firm's measurable buffer (EBITDA or cash) is fully wiped out by the losses it inherits.",
        f"**Что это значит?** Левый график показывает, чей отказ нанесёт наибольший ущерб: **{top_dr['company']}** во главе, потому что с ней связаны деньги многих компаний. "
        f"Правый — симулятор: если объявят дефолт {', '.join(picked) or 'никто'}, сильнее всего потеряют свою финансовую подушку {hit_txt}. "
        "100% означает, что измеримый буфер компании (EBITDA или денежные средства) полностью съеден унаследованными потерями."))
    with st.expander(tr("Exposure links used", "Использованные связи")):
        st.dataframe(net.used_links, use_container_width=True, hide_index=True)

    # ---- Diebold-Yilmaz
    st.subheader(tr("2 · Market spillovers (Diebold–Yilmaz connectedness)", "2 · Рыночные переливы (связность Diebold–Yilmaz)"))
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        system = st.radio(tr("System", "Система"), [tr("Core 4 tiers", "4 базовых уровня"), tr("Extended cross-market", "Расширенная межрыночная")], key="v3_sys")
    with d2:
        p = st.select_slider(tr("VAR lags p", "Лаги VAR p"), [1, 2, 3], 2, key="v3_p")
    with d3:
        H = st.select_slider(tr("Horizon H (days)", "Горизонт H (дни)"), [5, 10, 20], 10, key="v3_H")
    with d4:
        win = st.select_slider(tr("Window (days)", "Окно (дни)"), [126, 252, 378], 252, key="v3_win")
    tier_key = "core" if system.startswith(("Core", "4")) else "ext"
    R = _tier_returns(tier_key)
    dy = mc.diebold_yilmaz(R, p=p, H=H, window=win)
    h1, h2 = st.columns([1.15, 1])
    with h1:
        fig = px.imshow(dy["matrix"].round(1), text_auto=True, aspect="auto", color_continuous_scale=["#F7F3FC", LILAC, PURPLE, ROSE],
                        title=tr(f"Variance share from column (%) · TCI {dy['tci']:.1f}%", f"Доля дисперсии от столбца (%) · TCI {dy['tci']:.1f}%"))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    with h2:
        sm = dy["summary"].sort_values("net")
        fig = px.bar(sm, x="net", y="tier", orientation="h", color=np.where(sm["net"] > 0, tr("Net transmitter", "Чистый передатчик"), tr("Net receiver", "Чистый получатель")),
                     color_discrete_sequence=[ROSE, PURPLE], title=tr("Net directional spillover (To − From)", "Чистый направленный перелив (To − From)"))
        fig.update_layout(height=420, yaxis_title="", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    trans = dy["summary"].sort_values("net", ascending=False)
    givers = ", ".join(trans.loc[trans["net"] > 0, "tier"]) or "—"
    st.info(tr(
        f"**What does this mean?** Each cell answers: *how much of the surprise moves in the row's prices came from the column?* "
        f"Overall, **{dy['tci']:.0f}%** of daily surprises in these groups come from each other rather than from themselves — they move as one system. "
        f"Right now the shocks mainly flow **out of: {givers}** (net transmitters). The rolling line shows whether this interconnection is rising.",
        f"**Что это значит?** Каждая ячейка отвечает: *какая доля неожиданных движений цены строки пришла от столбца?* "
        f"В целом **{dy['tci']:.0f}%** дневных неожиданностей в этих группах приходят друг от друга, а не изнутри — они движутся как одна система. "
        f"Сейчас шоки в основном **исходят от: {givers}** (чистые передатчики). Скользящая линия показывает, растёт ли эта взаимосвязь."))
    roll = _rolling_tci(tier_key, win, p, H)
    fig = px.line(roll.reset_index(), x="index", y="TCI", color_discrete_sequence=[PURPLE], title=tr("Rolling Total Connectedness Index (strictly trailing window)", "Скользящий индекс общей связности (строго по прошлым данным)"))
    fig.update_layout(height=300, xaxis_title="", yaxis_title="TCI (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(tr(
        f"VAR({p}) on equal-weight daily log returns per tier, generalised FEVD at H = {H}, window {dy['start'].date()} → {dy['end'].date()} ({dy['n_obs']} days). Rolling TCI every 5 days. Tier 1 uses whichever of CRWV/APLD/GLXY were listed each day. Sources: {CITES['dy']}; {CITES['bglp']}.",
        f"VAR({p}) на равновзвешенных дневных лог-доходностях уровней, обобщённая FEVD при H = {H}, окно {dy['start'].date()} → {dy['end'].date()} ({dy['n_obs']} дней). Скользящий TCI каждые 5 дней. Уровень 1 — те из CRWV/APLD/GLXY, что торговались в этот день. Источники: {CITES['dy']}; {CITES['bglp']}."))

    # ---- Scenarios
    st.subheader(tr("3 · Three-scenario S&P 500 & Tech stress test", "3 · Стресс-тест S&P 500 и техсектора по трём сценариям"))
    pw = pd.concat([wide(core), wide(spill)], axis=1)
    lr = np.log(pw).diff()
    g1, g2 = st.columns([1, 2])
    with g1:
        targets = [t for t in ["SPY", "QQQ", "SOXX"] if t in lr.columns]
        target = st.radio(tr("Index", "Индекс"), targets, horizontal=True, key="v3_tgt")
        w_est = mc.implied_effective_weights(lr, target, mc.AI_IN_SPX)
        b_est = mc.downside_spill_beta(lr, target, w_est)
        w_df = pd.DataFrame({"ticker": w_est.index, "weight": w_est.round(4).values})
        w_edit = st.data_editor(w_df, hide_index=True, use_container_width=True, key=f"v3_w_{target}",
                                column_config={"weight": st.column_config.NumberColumn(tr("Effective weight", "Эффективный вес"), min_value=0.0, max_value=1.0, step=0.005, format="%.3f")})
        weights = pd.Series(w_edit["weight"].to_numpy(dtype=float), index=w_edit["ticker"])
        beta = st.slider("β_spill", 0.0, mc.DOTCOM_SPX_NASDAQ_ELASTICITY_CAP, float(round(b_est["beta"], 2)), 0.01, key=f"v3_b_{target}")
        st.caption(tr(
            f"Weights: non-negative returns-based style regression of {target} on the AI names over the last 252 days (Sharpe, 1992) — effective sensitivities, not official index weights. "
            f"Estimated β_spill = {b_est['beta_raw']:.3f} on {b_est['n_tail_days']} bottom-10% AI days, bounded to [0, {b_est['cap']}] (2000–02 S&P/Nasdaq elasticity).",
            f"Веса: неотрицательная регрессия стиля {target} на ИИ-акции за последние 252 дня (Sharpe, 1992) — эффективные чувствительности, а не официальные веса индекса. "
            f"Оценка β_spill = {b_est['beta_raw']:.3f} по {b_est['n_tail_days']} худшим 10% дням ИИ-корзины, ограничена [0, {b_est['cap']}] (эластичность S&P/Nasdaq 2000–02)."))
    results, firm_tables = [], {}
    with g2:
        tabs = st.tabs([tr("S1 · Mild", "S1 · Мягкий"), tr("S2 · Severe", "S2 · Тяжёлый"), tr("S3 · Tail", "S3 · Хвостовой")])
        for (key, sc), tb in zip(mc.SCENARIOS.items(), tabs):
            with tb:
                st.markdown(f"**{tr(sc['name_en'], sc['name_ru'])}**")
                cols = st.columns(4)
                shocks = {}
                for i, tier in enumerate(["T1 Neoclouds", "T2 Chipmakers", "T3 Hyperscalers", "Private credit"]):
                    shocks[tier] = cols[i].number_input(tier, -100.0, 50.0, float(sc["tier_shock_pct"][tier]), 2.5, key=f"v3_{key}_{tier}")
                sc_live = {**sc, "tier_shock_pct": shocks}
                fd = mc.firm_scenario_drawdowns(net, sc_live, dd_per_unit_distress=float(dd_unit))
                res = mc.index_stress_forecast(fd, weights, beta)
                firm_tables[key] = fd
                results.append({"scenario": key, "name": tr(sc["name_en"], sc["name_ru"]), **res})
                st.caption(sc["basis"])
                st.dataframe(fd, use_container_width=True, hide_index=True)
    rdf = pd.DataFrame(results)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rdf["scenario"], y=rdf["direct_pct"], name=tr("Direct AI-weight component", "Прямой вклад ИИ-весов"), marker_color=PURPLE))
    fig.add_trace(go.Bar(x=rdf["scenario"], y=rdf["spill_pct"], name=tr("Non-AI spillover component", "Перелив на не-ИИ часть"), marker_color=LILAC))
    fig.add_hline(y=ha.DOTCOM_BENCHMARKS["spx_dd_pct"].value, line_dash="dash", line_color=ROSE,
                  annotation_text=tr("S&P 500 2000–02: −49.1%", "S&P 500 2000–02: −49,1%"))
    fig.update_layout(barmode="relative", height=360, yaxis_title=tr(f"{target} scenario drawdown (%)", f"Сценарная просадка {target} (%)"),
                      title=tr(f"{target}: S1 {rdf['total_pct'][0]:.1f}% · S2 {rdf['total_pct'][1]:.1f}% · S3 {rdf['total_pct'][2]:.1f}%",
                               f"{target}: S1 {rdf['total_pct'][0]:.1f}% · S2 {rdf['total_pct'][1]:.1f}% · S3 {rdf['total_pct'][2]:.1f}%"))
    st.plotly_chart(fig, use_container_width=True)
    t1, t2, t3 = rdf["total_pct"].tolist()
    st.markdown(tr("**What the three scenarios look like in the real world** (default settings):", "**Как три сценария выглядят в реальной жизни** (настройки по умолчанию):"))
    s1c, s2c, s3c = st.columns(3)
    s1c.info(tr(
        f"**🟢 Mild — {target} {_fmt_pct(t1)}**\n\nState moratoria slow new data-center builds. Neocloud stocks fall 35–50%, chipmakers ~15%, Big Tech roughly flat. **Feels like a sector correction, not a market crash.**",
        f"**🟢 Мягкий — {target} {_fmt_pct(t1)}**\n\nМоратории штатов замедляют стройку ЦОД. Акции неооблаков падают на 35–50%, чипмейкеров на ~15%, крупнейшие техкомпании почти без изменений. **Ощущается как коррекция сектора, а не обвал рынка.**"))
    s2c.warning(tr(
        f"**🟠 Severe — {target} {_fmt_pct(t2)}**\n\nCoreWeave and Anthropic default; Broadcom's SPV guarantee is called and AMD / Microsoft write down their stakes. **The 2001 telecom bust in miniature: a painful tech bear market.**",
        f"**🟠 Тяжёлый — {target} {_fmt_pct(t2)}**\n\nCoreWeave и Anthropic объявляют дефолт; гарантия Broadcom по SPV исполняется, AMD и Microsoft списывают доли. **Телекомовский крах 2001 года в миниатюре: болезненный медвежий рынок в техсекторе.**"))
    s3c.error(tr(
        f"**🔴 Tail — {target} {_fmt_pct(t3)}**\n\nCredit markets seize up, OpenAI also fails, private credit marks down, AI stocks repeat 2000–02-sized losses. **A full bear market approaching the 2000–02 S&P 500 fall (−49.1%).**",
        f"**🔴 Хвостовой — {target} {_fmt_pct(t3)}**\n\nКредитные рынки замерзают, OpenAI тоже падает, частный кредит переоценивается вниз, ИИ-акции повторяют потери 2000–02. **Полноценный медвежий рынок, приближающийся к падению S&P 500 в 2000–02 (−49,1%).**"))
    with st.expander(tr("How this model works (for methodology defense)", "Как работает модель (для защиты методологии)")):
        st.markdown(tr(
        "**How to read it:** ΔIndex = Σ wᵢ·ΔPᵢ + β_spill·(1 − Σwᵢ)·ΔP_AI,avg. Firm drawdowns compound each tier's scenario shock with DebtRank distress: 1+ΔPᵢ = (1+tier shock)·(1 − k·Δhᵢ). "
        "All tier shocks and k are editable scenario assumptions anchored to the spec (S1) and to 2000–02 drawdowns (S3) — this is a stress test, not a point forecast.",
        "**Как читать:** ΔИндекса = Σ wᵢ·ΔPᵢ + β_spill·(1 − Σwᵢ)·ΔP_ИИ,ср. Просадки компаний объединяют шок уровня со стрессом DebtRank: 1+ΔPᵢ = (1+шок уровня)·(1 − k·Δhᵢ). "
        "Все шоки уровней и k — редактируемые сценарные допущения, привязанные к спецификации (S1) и просадкам 2000–02 (S3). Это стресс-тест, а не точечный прогноз."))
        st.markdown(tr(
            f"DebtRank: {CITES['dr']}. Spillovers: {CITES['dy']}; {CITES['bglp']}. Network phase transition: {CITES['aot']}.",
            f"DebtRank: {CITES['dr']}. Переливы: {CITES['dy']}; {CITES['bglp']}. Фазовый переход в сетях: {CITES['aot']}."))
    _status_caption(tr, statuses[:2])
    top = scores.iloc[0]
    return {"domino": {"index": target, "scenario_drawdowns_pct": {r["scenario"]: round(r["total_pct"], 1) for r in results},
                       "tci_pct": round(dy["tci"], 1), "most_central_node": str(top["company"])}}


# ---------------------------------------------------------------------------
# Tab C — 2000–2002 Analog & Playbook
# ---------------------------------------------------------------------------
def render_analog_tab(tr: T) -> dict:
    st.header(tr("2000–2002 Analog & Playbook", "Аналог 2000–2002 и план действий"))
    st.info(tr(
        "**The question:** what did the last vendor-financed tech boom look like when it broke, how is today similar or different, and what should a disciplined investor short, avoid or hold?",
        "**Вопрос:** как выглядел крах прошлого технологического бума на вендорном финансировании, чем сегодня похоже и чем отличается, и что дисциплинированному инвестору шортить, избегать или держать?"))
    st.markdown(tr(
        "#### 🎬 We've seen this movie before\n"
        "In 1999–2000, equipment makers such as **Lucent** and **Nortel** lent money to young telecom carriers (CLECs) so those carriers could buy their equipment. "
        "Sales looked spectacular — until the carriers could not pay. **47 CLECs went bankrupt (2000–2003)**, Lucent booked **$2.2B** of bad-debt provisions, and the SEC later found **$1.148B** of improperly recognised revenue.\n\n"
        "In 2026 the script rhymes: **chipmakers guarantee the debt of their own customers** — Broadcom backs a $63B Anthropic SPV, Nvidia backstops a $105B Ohio data center for OpenAI and promises to rent CoreWeave's unused capacity. "
        "The big difference: today's largest buyers (Microsoft, Amazon, Alphabet) earn far more cash than 2000's carriers — but GPUs wear out in 3–6 years, not decades like fiber.",
        "#### 🎬 Мы уже видели это кино\n"
        "В 1999–2000 годах производители оборудования, такие как **Lucent** и **Nortel**, кредитовали молодых телеком-операторов (CLEC), чтобы те покупали их оборудование. "
        "Продажи выглядели блестяще — пока операторы не перестали платить. **47 CLEC обанкротились (2000–2003)**, Lucent создала резервы на **$2,2 млрд** безнадёжных долгов, а SEC позже обнаружила **$1,148 млрд** неправомерно признанной выручки.\n\n"
        "В 2026 году сюжет рифмуется: **чипмейкеры гарантируют долги собственных клиентов** — Broadcom стоит за SPV Anthropic на $63 млрд, Nvidia гарантирует ЦОД OpenAI в Огайо на $105 млрд и обещает арендовать простаивающие мощности CoreWeave. "
        "Главное отличие: крупнейшие покупатели сегодня (Microsoft, Amazon, Alphabet) зарабатывают намного больше денег, чем операторы 2000 года, — но GPU изнашиваются за 3–6 лет, а не за десятилетия, как оптоволокно."))
    core, _, dot, _, statuses = _panels()
    _missing_banner(tr, [statuses[0], statuses[2]])
    pack, _ = _pack()

    if not dot.empty:
        m = ha.measured_drawdowns(dot)
        el = ha.spx_nasdaq_elasticity(m)
        mi = m.set_index("ticker")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(tr("Nasdaq Composite peak→trough", "Nasdaq Composite пик→дно"), f"{mi.loc['^IXIC', 'drawdown_pct']:.1f}%", f"{mi.loc['^IXIC', 'months']} {tr('months', 'мес.')}", delta_color="off")
        k2.metric(tr("S&P 500 peak→trough", "S&P 500 пик→дно"), f"{mi.loc['^GSPC', 'drawdown_pct']:.1f}%", f"{mi.loc['^GSPC', 'months']} {tr('months', 'мес.')}", delta_color="off")
        k3.metric(tr("S&P / Nasdaq elasticity", "Эластичность S&P / Nasdaq"), f"{el:.2f}×")
        k4.metric(tr("Nasdaq-100 drawdown (cited)", "Просадка Nasdaq-100 (источник)"), f"{ha.DOTCOM_BENCHMARKS['ndx_dd_pct'].value:.1f}%")
        pw, ai, _, _ = _ai_core()
        ix = dot[dot["ticker"] == "^IXIC"].set_index("date")["adjusted_close"]
        paths = ha.aligned_paths(ix, ai)
        fig = px.line(paths, x="days_from_peak", y="index", color="series", color_discrete_sequence=[ROSE, PURPLE], hover_data=["date"],
                      title=tr("Run-up paths aligned on each series' own peak (= 100 two years before the peak)", "Траектории роста, выровненные по собственному пику (= 100 за два года до пика)"))
        fig.add_vline(x=0, line_dash="dot", line_color=SLATE)
        fig.update_layout(height=380, xaxis_title=tr("Trading days relative to peak", "Торговые дни относительно пика"), yaxis_title="Index")
        st.plotly_chart(fig, use_container_width=True)
        ex_d = bm.run_exuberance_test(ix, cv=_cv(len(ix), bm.psy_min_window(len(ix)), 200))
        epi = "; ".join(f"{e.start.date()} → {e.end.date()}" for e in ex_d.episodes.itertuples())
        st.caption(tr(
            f"Measured from data/raw/dotcom_1998_2003.csv (Yahoo ^IXIC, ^GSPC): peaks {mi.loc['^IXIC', 'peak_date']} / {mi.loc['^GSPC', 'peak_date']}, troughs {mi.loc['^IXIC', 'trough_date']} / {mi.loc['^GSPC', 'trough_date']} — identical to the cited index records. "
            f"Validation: the same BSADF detector date-stamps Nasdaq explosive episodes {epi}. Nasdaq-100 figure: {ha.IDX}.",
            f"Рассчитано по data/raw/dotcom_1998_2003.csv (Yahoo ^IXIC, ^GSPC): пики {mi.loc['^IXIC', 'peak_date']} / {mi.loc['^GSPC', 'peak_date']}, дно {mi.loc['^IXIC', 'trough_date']} / {mi.loc['^GSPC', 'trough_date']} — совпадает с цитируемыми данными индексов. "
            f"Проверка: тот же детектор BSADF датирует взрывные эпизоды Nasdaq {epi}. Данные по Nasdaq-100: {ha.IDX}."))

    st.subheader(tr("Structural comparison: 1999–2002 telecom vs 2025–2026 AI", "Структурное сравнение: телеком 1999–2002 vs ИИ 2025–2026"))
    comp = pd.DataFrame(ha.STRUCTURAL_COMPARISON, columns=["en", "ru", "1999–2002", "2025–2026", tr("Parallel / difference", "Сходство / различие")])
    comp.insert(0, tr("Dimension", "Измерение"), comp["en"] if tr("en", "ru") == "en" else comp["ru"])
    st.dataframe(comp.drop(columns=["en", "ru"]), use_container_width=True, hide_index=True)
    lucent = [ha.DOTCOM_BENCHMARKS[k] for k in ["lucent_customer_financing_bn", "lucent_bad_debt_bn", "lucent_improper_revenue_bn", "lucent_pretax_overstatement_m", "lucent_penalty_m", "clec_bankruptcies"]]
    st.caption(" · ".join(f"{tr(v.label_en, v.label_ru)}: {v.value:g} {v.unit}" for v in lucent) + f". {tr('Sources', 'Источники')}: {ha.SEC_2004}; {CITES['or']}.")

    st.subheader(tr("Actionable positioning matrix", "Матрица позиционирования"))
    c1, c2, c3, c4 = st.columns(4)
    costs = ha.CostAssumptions(
        transaction_cost_bps=c1.number_input(tr("Transaction cost (bps/side)", "Комиссия (б.п./сторона)"), 0.0, 100.0, 10.0, 5.0, key="v3_tc"),
        borrow_fee_gc_pct=c2.number_input(tr("Borrow fee, easy-to-borrow (%/yr)", "Плата за заём, лёгкий (%/год)"), 0.0, 20.0, 0.5, 0.25, key="v3_gc"),
        borrow_fee_htb_pct=c3.number_input(tr("Borrow fee, neoclouds (%/yr)", "Плата за заём, неооблака (%/год)"), 0.0, 100.0, 10.0, 1.0, key="v3_htb"),
        horizon_months=int(c4.number_input(tr("Holding horizon (months)", "Горизонт удержания (мес.)"), 1, 24, 6, 1, key="v3_hz")),
    )
    trig = st.session_state.get("v3_triggers")
    if trig is None:
        pw0, ai0, _, _ = _ai_core()
        ex0 = bm.run_exuberance_test(ai0, cv=_cv(len(ai0), bm.psy_min_window(len(ai0)), 200))
        trig = bm.trigger_checklist(cs.current_credit_inputs(pack), ex0.table["bsadf"].iloc[-1], ex0.table["cv95"].iloc[-1], cs.supply_friction_flags())
    pm = ha.positioning_matrix(wide(core), pack.bond_matrix if pack else None, trig, costs)
    sides = {k: pm.loc[pm["side"] == k, "ticker"].tolist() for k in ("short", "hedge", "long")}
    avoid_long = pm.loc[pm["bucket"].str.startswith("Avoid as long"), "ticker"].tolist()
    htb_cost = pm.loc[pm["side"] == "short", "cost_over_horizon_pct"]
    htb_txt = f"{htb_cost.iloc[0]:.1f}%" if not htb_cost.empty else "n/a"
    j = lambda xs: ", ".join(xs) if xs else "—"
    st.success(tr(
        "### 🧭 So what should an investor actually do?\n"
        f"- **Avoid or short the weakest links:** {j(sides['short'])} — but only with small size: borrowing these shares costs about **{htb_txt}** over {costs.horizon_months} months, and short squeezes are violent.\n"
        f"- **Hedge, don't bet against, the guarantors:** {j(sides['hedge'])} — use protective puts or pair trades, since their profit margins are still strong.\n"
        f"- **Lean on fortress balance sheets:** {j(sides['long'])} — Microsoft has the smallest tracked AI debt ($44B) and Amazon holds more cash than debt (SEC 10-K); both rose >6% after the election as forced capex discipline helped them.\n"
        f"- **Don't add as longs:** {j(avoid_long)} — the largest AI-related debt piles in the market.\n"
        "- **Know when to stop:** the whole defensive stance switches **OFF** once AI junk-bond spreads fall below 350 bps **and** the bubble test cools below its 90% line.",
        "### 🧭 Так что же делать инвестору?\n"
        f"- **Избегать или шортить самые слабые звенья:** {j(sides['short'])} — но только небольшим объёмом: заём этих акций стоит около **{htb_txt}** за {costs.horizon_months} мес., а шорт-сквизы бывают жёсткими.\n"
        f"- **Хеджировать, а не ставить против гарантов:** {j(sides['hedge'])} — через защитные путы или парные сделки, ведь их маржа всё ещё высока.\n"
        f"- **Опираться на крепкие балансы:** {j(sides['long'])} — у Microsoft наименьший отслеживаемый ИИ-долг ($44 млрд), а у Amazon денежных средств больше, чем долга (10-K SEC); обе выросли более чем на 6% после выборов, так как вынужденная дисциплина капвложений им на руку.\n"
        f"- **Не наращивать в лонг:** {j(avoid_long)} — крупнейшие долги, связанные с ИИ.\n"
        "- **Знать, когда остановиться:** вся защитная позиция **ВЫКЛЮЧАЕТСЯ**, когда спреды мусорных облигаций ИИ опустятся ниже 350 б.п. **и** тест на пузырь остынет ниже своей 90% линии."))
    if tr("en", "ru") == "ru":
        pm["bucket"] = pm["bucket"].map(BUCKET_RU).fillna(pm["bucket"])
        pm["side"] = pm["side"].map({"short": "шорт", "hedge": "хедж", "hold": "держать/нейтрально", "long": "лонг"}).fillna(pm["side"])
    st.dataframe(pm.rename(columns={
        "ticker": "Ticker", "bucket": tr("Bucket", "Группа"), "side": tr("Stance", "Позиция"), "rationale": tr("Why (sourced)", "Почему (с источником)"),
        "tracked_debt_bn": tr("JPM tracked debt ($B)", "Долг по JPM ($ млрд)"), "dd_from_52w_high_pct": tr("From 52w high (%)", "От 52-нед. максимума (%)"),
        "vol_63d_pct": tr("63d vol (%)", "Волат. 63д (%)"), "cost_over_horizon_pct": tr("Cost over horizon (%)", "Издержки за горизонт (%)")}),
        use_container_width=True, hide_index=True)
    st.warning(tr(
        "**Cost & risk discipline:** shorting high-beta neoclouds carries elevated borrow fees and squeeze risk; chipmaker exposure is hedged with puts/pairs rather than naked shorts. "
        "Borrow fees above are user assumptions, not prime-broker quotes. Buckets switch automatically with the Bubble Radar triggers, and the whole stance turns OFF under the RULE_CARD v3.0 switch-off. Not investment advice.",
        "**Дисциплина издержек и риска:** шорт высокобета-неооблаков несёт высокую плату за заём и риск шорт-сквиза; экспозиция на чипмейкеров хеджируется путами/парами, а не голым шортом. "
        "Плата за заём выше — допущение пользователя, а не котировка брокера. Группы переключаются автоматически по триггерам радара, а вся позиция выключается по правилу RULE_CARD v3.0. Не инвестиционная рекомендация."))
    st.caption(tr("Live metrics from data/raw/volumes_core.csv; debt from the J.P. Morgan bond matrix (printed totals); Blue Wave reaction from WSJ (Nov 5, 2026).",
                  "Текущие метрики из data/raw/volumes_core.csv; долг из облигационной матрицы J.P. Morgan (напечатанные итоги); реакция на «Синюю волну» — WSJ (5 ноября 2026)."))
    if pack is not None:
        with st.expander(tr("'Blue Wave' state policies (Nov 3, 2026)", "Политика штатов после «Синей волны» (3 ноября 2026)")):
            st.dataframe(pack.blue_wave_states, use_container_width=True, hide_index=True)
            st.caption(f"{cs.JPM_NOV4}; 42/45 local ballots passed or leading.")
        with st.expander(tr("Literature reference matrix", "Матрица литературы")):
            st.dataframe(pack.literature, use_container_width=True, hide_index=True)
    return {"analog_playbook": {"short_avoid": sides["short"], "relative_long": sides["long"]}}
