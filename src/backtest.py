"""Group Assignment 2 (The Rule Desk) backtest engine & full E1-E8 evidence generator.

Non-negotiable honesty rules enforced by design:
1. Timing: position[t] = signal[t-1] (strictly lagged; zero look-ahead).
2. Cash earns the daily risk-free rate (annual_rf / 252).
3. Transaction costs: deducted on |position[t] - position[t-1]|.
4. Error bars: every Sharpe ratio has a standard error; Sharpe difference uses
   paired circular block bootstrap 95% CI.
5. Canary test helper verifies that perturbing future returns does not alter earlier positions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def build_equal_weight_basket(
    prices: pd.DataFrame,
    tickers: list[str],
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Build daily equal-weight basket returns and cumulative wealth index (starts at 100)."""
    data = prices[["date", "ticker", "adjusted_close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["adjusted_close"] = pd.to_numeric(data["adjusted_close"], errors="coerce")
    data = data.dropna()
    if start_date is not None:
        data = data[data["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        data = data[data["date"] <= pd.to_datetime(end_date)]

    piv = data.pivot_table(index="date", columns="ticker", values="adjusted_close").sort_index()
    missing = [t for t in tickers if t not in piv.columns]
    if missing:
        raise ValueError(f"Missing tickers in price data: {missing}")

    common = piv[tickers].dropna()
    if len(common) < 10:
        raise ValueError("Not enough overlapping price dates for basket.")

    daily_ret = common.pct_change().dropna()
    basket_ret = daily_ret.mean(axis=1)
    # Index starts at 100 on common.index[0]
    wealth = pd.Series(100.0, index=[common.index[0]])
    wealth = pd.concat([wealth, 100.0 * (1.0 + basket_ret).cumprod()])
    return pd.DataFrame({
        "basket_price": wealth,
        "basket_return": wealth.pct_change(),
    })


def compute_sma_signal_and_position(
    price_series: pd.Series,
    lookback: int = 100,
) -> tuple[pd.Series, pd.Series]:
    """Compute end-of-day SMA signal and strictly lagged next-day position.

    Signal at close of t: 1.0 if price[t] > SMA(lookback)[t], else 0.0.
    Position held during day t (earning return from t-1 to t): signal[t-1].
    During warm-up (before lookback observations exist), position is NaN and
    those warm-up dates are excluded identically for both rule and benchmark.
    """
    if lookback < 2:
        raise ValueError("lookback must be >= 2.")
    sma = price_series.rolling(window=lookback, min_periods=lookback).mean()
    signal = pd.Series(np.nan, index=price_series.index, dtype=float)
    valid = sma.notna()
    signal.loc[valid] = (price_series.loc[valid] > sma.loc[valid]).astype(float)
    position = signal.shift(1)
    return signal, position


def run_rule_backtest(
    price_series: pd.Series,
    lookback: int = 100,
    cost_bps: float = 10.0,
    risk_free_annual_pct: float = 4.0,
    eval_start=None,
    eval_end=None,
) -> pd.DataFrame:
    """Run the lagged SMA trend-filter backtest on a price/index series.

    Parameters
    ----------
    price_series : pd.Series
        Index-aligned adjusted close or basket wealth series.
    lookback : int
        SMA window in trading days.
    cost_bps : float
        One-way transaction cost in basis points per unit of position change (e.g., 10 bps = 0.0010).
    risk_free_annual_pct : float
        Annual risk-free rate (%) earned when position == 0 (cash).
    """
    series = price_series.dropna().sort_index()
    asset_ret = series.pct_change()
    signal, position = compute_sma_signal_and_position(series, lookback=lookback)

    df = pd.DataFrame({
        "price": series,
        "asset_return": asset_ret,
        "signal": signal,
        "position": position,
    }).dropna(subset=["asset_return", "position"])

    if eval_start is not None:
        df = df[df.index >= pd.to_datetime(eval_start)]
    if eval_end is not None:
        df = df[df.index <= pd.to_datetime(eval_end)]

    if len(df) < 2:
        raise ValueError("Evaluation window has fewer than 2 valid return observations.")

    daily_rf = risk_free_annual_pct / 100.0 / TRADING_DAYS
    c = cost_bps / 10_000.0

    # Position change on day t (first evaluated day assumes starting from 0 -> initial entry incurs cost if pos=1)
    pos_change = df["position"].diff().abs()
    pos_change.iloc[0] = 0.0  # evaluate steady-state switches within window; initial entry cost tracked if desired
    gross_ret = df["position"] * df["asset_return"] + (1.0 - df["position"]) * daily_rf
    net_ret = gross_ret - c * pos_change

    df["pos_change"] = pos_change
    df["gross_return"] = gross_ret
    df["net_return"] = net_ret
    df["benchmark_return"] = df["asset_return"]
    df["rule_wealth"] = (1.0 + df["net_return"]).cumprod()
    df["benchmark_wealth"] = (1.0 + df["benchmark_return"]).cumprod()
    return df


def sharpe_and_se(returns: pd.Series, risk_free_annual_pct: float = 4.0) -> tuple[float, float]:
    """Annualized Sharpe ratio and Lo (2002) analytical standard error."""
    r = pd.to_numeric(returns, errors="coerce").dropna()
    n = len(r)
    if n < 2:
        return float("nan"), float("nan")
    daily_rf = risk_free_annual_pct / 100.0 / TRADING_DAYS
    ex = r - daily_rf
    sd = float(ex.std(ddof=1))
    if sd <= 0:
        return float("nan"), float("nan")
    sr_daily = float(ex.mean() / sd)
    sr_ann = sr_daily * np.sqrt(TRADING_DAYS)
    # Lo (2002) i.i.d. SE of annualized Sharpe: sqrt((1 + 0.5 * SR_daily^2) / n) * sqrt(252)
    se_ann = float(np.sqrt((1.0 + 0.5 * sr_daily ** 2) / n) * np.sqrt(TRADING_DAYS))
    return float(sr_ann), se_ann


def summarize_strategy(
    bt: pd.DataFrame,
    risk_free_annual_pct: float = 4.0,
) -> pd.DataFrame:
    """E1: Headline table for Rule (net) and Benchmark on the exact same window."""
    n = len(bt)
    n_years = n / TRADING_DAYS

    def _stats(ret: pd.Series, is_rule: bool) -> dict:
        wealth = np.concatenate([[1.0], np.cumprod(1.0 + ret.to_numpy())])
        peak = np.maximum.accumulate(wealth)
        mdd = float((wealth / peak - 1.0).min() * 100.0)
        cagr = float((wealth[-1] ** (1.0 / n_years) - 1.0) * 100.0)
        vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100.0)
        sr, se = sharpe_and_se(ret, risk_free_annual_pct)
        time_in_mkt = float(bt["position"].mean() * 100.0) if is_rule else 100.0
        switches = int((bt["pos_change"] > 0).sum()) if is_rule else 0
        turnover_yr = float(bt["pos_change"].sum() / n_years) if is_rule else 0.0
        return {
            "Strategy": "Rule (SMA Trend Filter, Net)" if is_rule else "Benchmark (Buy-and-Hold Basket)",
            "Window": f"{bt.index.min().date()} to {bt.index.max().date()}",
            "CAGR (%)": round(cagr, 2),
            "Volatility (%)": round(vol, 2),
            "Sharpe": round(sr, 3),
            "Sharpe SE": round(se, 3),
            "Max Drawdown (%)": round(mdd, 2),
            "Time in Market (%)": round(time_in_mkt, 1),
            "Switches": switches,
            "Turnover / yr": round(turnover_yr, 2),
        }

    return pd.DataFrame([
        _stats(bt["net_return"], is_rule=True),
        _stats(bt["benchmark_return"], is_rule=False),
    ])


def paired_block_bootstrap_sharpe_diff(
    rule_returns: pd.Series,
    bench_returns: pd.Series,
    risk_free_annual_pct: float = 4.0,
    block_size: int = 20,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """E2: Paired circular block bootstrap 95% CI for (Sharpe_rule - Sharpe_bench)."""
    paired = pd.concat([rule_returns.rename("r"), bench_returns.rename("b")], axis=1).dropna()
    n = len(paired)
    if n < 5:
        raise ValueError("Need at least 5 paired observations for bootstrap.")

    r_arr = paired["r"].to_numpy()
    b_arr = paired["b"].to_numpy()
    daily_rf = risk_free_annual_pct / 100.0 / TRADING_DAYS

    def _sr(x: np.ndarray) -> float:
        ex = x - daily_rf
        sd = ex.std(ddof=1)
        return float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan

    obs_diff = _sr(r_arr) - _sr(b_arr)
    rng = np.random.default_rng(seed)
    blk = max(1, min(block_size, n))
    n_blocks = int(np.ceil(n / blk))

    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + blk) % n) for s in starts])[:n]
        diffs[i] = _sr(r_arr[idx]) - _sr(b_arr[idx])

    valid = diffs[~np.isnan(diffs)]
    ci_low = float(np.percentile(valid, 2.5))
    ci_high = float(np.percentile(valid, 97.5))
    zero_inside = bool(ci_low <= 0.0 <= ci_high)
    verdict = (
        f"Zero IS inside the 95% CI [{ci_low:+.3f}, {ci_high:+.3f}]: the Sharpe difference "
        f"({obs_diff:+.3f}) is NOT statistically distinguishable from zero."
        if zero_inside
        else f"Zero is NOT inside the 95% CI [{ci_low:+.3f}, {ci_high:+.3f}]: the Sharpe difference "
        f"({obs_diff:+.3f}) is distinguishable from zero."
    )
    return {
        "sharpe_diff": round(obs_diff, 3),
        "ci_low": round(ci_low, 3),
        "ci_high": round(ci_high, 3),
        "zero_inside": zero_inside,
        "block_size": blk,
        "n_boot": n_boot,
        "sentence": verdict,
    }


def cost_sensitivity_and_breakeven(
    price_series: pd.Series,
    lookback: int = 100,
    cost_grid_bps: list[float] | None = None,
    risk_free_annual_pct: float = 4.0,
    eval_start=None,
    eval_end=None,
) -> tuple[pd.DataFrame, float]:
    """E3: Net return & Sharpe across transaction cost grid + exact break-even cost (bps)."""
    if cost_grid_bps is None:
        cost_grid_bps = [0.0, 10.0, 25.0, 50.0]

    rows = []
    for c_bps in cost_grid_bps:
        bt = run_rule_backtest(
            price_series, lookback=lookback, cost_bps=c_bps,
            risk_free_annual_pct=risk_free_annual_pct,
            eval_start=eval_start, eval_end=eval_end,
        )
        n_years = len(bt) / TRADING_DAYS
        r_cagr = float(((1.0 + bt["net_return"]).prod() ** (1.0 / n_years) - 1.0) * 100.0)
        b_cagr = float(((1.0 + bt["benchmark_return"]).prod() ** (1.0 / n_years) - 1.0) * 100.0)
        r_sr, _ = sharpe_and_se(bt["net_return"], risk_free_annual_pct)
        b_sr, _ = sharpe_and_se(bt["benchmark_return"], risk_free_annual_pct)
        rows.append({
            "Cost per Switch (bps)": c_bps,
            "Rule Net CAGR (%)": round(r_cagr, 2),
            "Benchmark CAGR (%)": round(b_cagr, 2),
            "Net CAGR Diff (%)": round(r_cagr - b_cagr, 2),
            "Rule Net Sharpe": round(r_sr, 3),
            "Sharpe Diff": round(r_sr - b_sr, 3),
        })

    # Exact linear break-even cost (bps) where mean daily net return == mean daily benchmark return
    bt0 = run_rule_backtest(
        price_series, lookback=lookback, cost_bps=0.0,
        risk_free_annual_pct=risk_free_annual_pct,
        eval_start=eval_start, eval_end=eval_end,
    )
    mean_gross_adv = float((bt0["gross_return"] - bt0["benchmark_return"]).mean())
    mean_turnover = float(bt0["pos_change"].mean())
    if mean_turnover <= 0:
        breakeven_bps = float("nan")
    else:
        breakeven_bps = round((mean_gross_adv / mean_turnover) * 10_000.0, 1)

    return pd.DataFrame(rows), breakeven_bps


def subperiod_and_startdate_check(
    price_series: pd.Series,
    lookback: int = 100,
    cost_bps: float = 10.0,
    risk_free_annual_pct: float = 4.0,
    windows: list[tuple[str, str | None, str | None]] | None = None,
) -> pd.DataFrame:
    """E4: Evaluate rule vs benchmark across at least three start dates / subperiods."""
    if windows is None:
        windows = [
            ("Full window (post-warmup)", None, None),
            ("Start 2024-01-01", "2024-01-01", None),
            ("Start 2025-01-01", "2025-01-01", None),
            ("Subperiod 2023-2024", "2023-01-01", "2024-12-31"),
        ]
    rows = []
    for label, s, e in windows:
        bt = run_rule_backtest(
            price_series, lookback=lookback, cost_bps=cost_bps,
            risk_free_annual_pct=risk_free_annual_pct,
            eval_start=s, eval_end=e,
        )
        r_sr, r_se = sharpe_and_se(bt["net_return"], risk_free_annual_pct)
        b_sr, _ = sharpe_and_se(bt["benchmark_return"], risk_free_annual_pct)
        r_w = np.concatenate([[1.0], np.cumprod(1.0 + bt["net_return"].to_numpy())])
        b_w = np.concatenate([[1.0], np.cumprod(1.0 + bt["benchmark_return"].to_numpy())])
        r_mdd = float((r_w / np.maximum.accumulate(r_w) - 1.0).min() * 100.0)
        b_mdd = float((b_w / np.maximum.accumulate(b_w) - 1.0).min() * 100.0)
        rows.append({
            "Subperiod": label,
            "Window": f"{bt.index.min().date()} to {bt.index.max().date()}",
            "Rule Sharpe": round(r_sr, 3),
            "Bench Sharpe": round(b_sr, 3),
            "Sharpe Diff": round(r_sr - b_sr, 3),
            "Rule MaxDD (%)": round(r_mdd, 1),
            "Bench MaxDD (%)": round(b_mdd, 1),
        })
    return pd.DataFrame(rows)


def parameter_neighbors_check(
    price_series: pd.Series,
    center_lookback: int = 100,
    neighbor_grid: list[int] | None = None,
    cost_bps: float = 10.0,
    risk_free_annual_pct: float = 4.0,
    eval_start=None,
    eval_end=None,
) -> pd.DataFrame:
    """E5: Check parameter +/- 1 and 2 steps for plateau vs spike."""
    if neighbor_grid is None:
        neighbor_grid = [center_lookback - 40, center_lookback - 20, center_lookback, center_lookback + 20, center_lookback + 40]
    rows = []
    for L in neighbor_grid:
        bt = run_rule_backtest(
            price_series, lookback=L, cost_bps=cost_bps,
            risk_free_annual_pct=risk_free_annual_pct,
            eval_start=eval_start, eval_end=eval_end,
        )
        r_sr, r_se = sharpe_and_se(bt["net_return"], risk_free_annual_pct)
        b_sr, _ = sharpe_and_se(bt["benchmark_return"], risk_free_annual_pct)
        r_w = np.concatenate([[1.0], np.cumprod(1.0 + bt["net_return"].to_numpy())])
        r_mdd = float((r_w / np.maximum.accumulate(r_w) - 1.0).min() * 100.0)
        rows.append({
            "Lookback (days)": int(L),
            "Role": "Selected" if L == center_lookback else "Neighbour",
            "Rule Sharpe": round(r_sr, 3),
            "Sharpe SE": round(r_se, 3),
            "Sharpe Diff": round(r_sr - b_sr, 3),
            "Rule MaxDD (%)": round(r_mdd, 1),
            "Switches": int((bt["pos_change"] > 0).sum()),
        })
    return pd.DataFrame(rows)


def downside_risk_summary(bt: pd.DataFrame) -> pd.DataFrame:
    """E8: Worst calendar month and historical daily 5% VaR for rule and benchmark."""
    window = f"{bt.index.min().date()} to {bt.index.max().date()}"
    rows = []
    for label, col in [("Rule (SMA Trend Filter, Net)", "net_return"), ("Benchmark (Buy-and-Hold)", "benchmark_return")]:
        r = bt[col]
        monthly = (1.0 + r).resample("ME").prod() - 1.0
        worst_mo = float(monthly.min() * 100.0)
        worst_mo_date = str(monthly.idxmin().date())[:7]
        var_5 = float(np.percentile(r.to_numpy(), 5) * 100.0)
        rows.append({
            "Strategy": label,
            "Window": window,
            "Worst Month (%)": round(worst_mo, 2),
            "Worst Month Date": worst_mo_date,
            "Historical Daily 5% VaR (%)": round(var_5, 2),
        })
    return pd.DataFrame(rows)


def expected_max_noise_sharpe(n_trials: int, n_years: float) -> float:
    """E7: Approximate expected max Sharpe from N independent noise rules over T years.

    Formula: E[max_k SR_k] ≈ sqrt(2 * ln(N)) / sqrt(T_years) for N >= 2, else 0.0.
    """
    if n_trials < 2 or n_years <= 0:
        return 0.0
    return float(np.sqrt(2.0 * np.log(n_trials)) / np.sqrt(n_years))


def run_canary_lookahead_test(
    price_series: pd.Series,
    lookback: int = 20,
    split_idx: int = 60,
) -> bool:
    """Canary test: inject a +500% shock after split_idx; positions up to split_idx must NOT change."""
    orig = price_series.copy()
    perturbed = price_series.copy()
    perturbed.iloc[split_idx + 1 :] = perturbed.iloc[split_idx + 1 :] * 6.0

    _, pos_orig = compute_sma_signal_and_position(orig, lookback=lookback)
    _, pos_pert = compute_sma_signal_and_position(perturbed, lookback=lookback)

    # Position on split_idx uses signal from split_idx - 1 (which is before split_idx + 1)
    # Position on split_idx + 1 uses signal from split_idx (still unperturbed!)
    a = pos_orig.iloc[: split_idx + 2].dropna()
    b = pos_pert.iloc[: split_idx + 2].dropna()
    return bool(a.equals(b))
