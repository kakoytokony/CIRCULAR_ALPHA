"""Bubble Radar: real-time exuberance tests, crash-risk score and trigger checklist.

Methods (see METHODOLOGY_AND_SPEC.md §4.1):

* **BSADF / GSADF** — Phillips, Shi & Yu (2015, IER 56(4)). Right-tailed ADF
  regressions ``Δy_t = α + β y_{t-1} + ε_t`` over every backward-expanding window
  ending at t; ``BSADF_t = sup_{r1} ADF_{r1}^{t}``. Lag order K = 0 (logged in
  trials.csv). Computed in closed form from prefix sums, so each window is O(1).
  Finite-sample critical values come from a fixed-seed Monte Carlo of driftless
  Gaussian random walks of the same length (the ADF t-statistic is scale-free).
* **Crash-risk score** — Greenwood, Shleifer & You (2019, JFE 131(1)) attributes
  of run-ups that crash: volatility, turnover, acceleration, issuance/credit
  (credit proxied by the ICE BofA US HY OAS from FRED, after Gilchrist &
  Zakrajšek 2012). Each attribute is a trailing z-score; the composite
  ``S_t = Σ w_k Z_k`` maps to a probability with a logistic link whose (θ0, θ1)
  are fitted on the Nasdaq Composite 1998–2003 episode (labelled with the GSY
  crash definition: ≥40% fall within the next 24 months). That is one historical
  episode, so the probability is a *calibrated stress index*, not a forecast.

Look-ahead discipline: every statistic at date t uses data ≤ t only, and every
tradable *signal* is shifted by one day (``signal[t] = stat[t-1]``).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
GSY_HORIZON = 504          # 24 months
GSY_CRASH_DD = -0.40       # crash = drawdown of at least 40%
GSY_RUNUP = 1.00           # 100% two-year run-up
DEFAULT_WEIGHTS = {"z_vol": 0.25, "z_turn": 0.25, "z_acc": 0.25, "z_cred": 0.25}
Z_WINDOW = 504
Z_MIN_PERIODS = 126


# ---------------------------------------------------------------------------
# 1. Phillips–Shi–Yu BSADF / GSADF
# ---------------------------------------------------------------------------
def psy_min_window(n_obs: int) -> int:
    """PSY (2015) minimum window r0 = 0.01 + 1.8/sqrt(T), in observations."""
    return max(10, int(np.floor((0.01 + 1.8 / np.sqrt(n_obs)) * n_obs)))


def _prefix(a: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape[:-1] + (a.shape[-1] + 1,))
    np.cumsum(a, axis=-1, out=out[..., 1:])
    return out


def bsadf_series(log_price: np.ndarray, min_window: int | None = None) -> np.ndarray:
    """Backward SADF statistic for every t (NaN before the first full window).

    Accepts a 1-D array (one series) or a 2-D array (paths x time) so the Monte
    Carlo critical values are vectorised across paths.
    """
    y = np.asarray(log_price, dtype=float)
    single = y.ndim == 1
    if single:
        y = y[None, :]
    n_paths, T = y.shape
    w0 = psy_min_window(T) if min_window is None else int(min_window)
    y = y - y[:, :1]                     # causal centring (first observation only)
    x = y[:, :-1]                        # y_{p-1}, pair index p = 1..T-1
    d = np.diff(y, axis=1)               # Δy_p
    Px, Pd = _prefix(x), _prefix(d)
    Pxx, Pxd, Pdd = _prefix(x * x), _prefix(x * d), _prefix(d * d)
    out = np.full((n_paths, T), np.nan)
    for t in range(w0 - 1, T):
        # windows of y-observations [s, t] with t - s + 1 >= w0 -> pairs (s, t]
        s = np.arange(0, t - w0 + 2)
        lo, hi = s, t                    # prefix indices: pairs s..t-1 in 0-based pair arrays
        n = (hi - lo).astype(float)
        Sx = Px[:, hi][:, None] - Px[:, lo]
        Sd = Pd[:, hi][:, None] - Pd[:, lo]
        Sxx = Pxx[:, hi][:, None] - Pxx[:, lo]
        Sxd = Pxd[:, hi][:, None] - Pxd[:, lo]
        Sdd = Pdd[:, hi][:, None] - Pdd[:, lo]
        Cxx = Sxx - Sx * Sx / n
        Cxd = Sxd - Sx * Sd / n
        Cdd = Sdd - Sd * Sd / n
        with np.errstate(divide="ignore", invalid="ignore"):
            b = Cxd / Cxx
            ssr = np.maximum(Cdd - b * Cxd, 0.0)
            se = np.sqrt(ssr / (n - 2.0) / Cxx)
            stat = b / se
        stat = np.where(np.isfinite(stat), stat, -np.inf)
        out[:, t] = stat.max(axis=1)
    out[~np.isfinite(out)] = np.nan
    return out[0] if single else out


def adf_stat(log_price: np.ndarray) -> float:
    """Plain full-sample ADF(0) t-statistic on β (used in tests as a cross-check)."""
    y = np.asarray(log_price, dtype=float)
    x, d = y[:-1], np.diff(y)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, d, rcond=None)
    resid = d - X @ beta
    s2 = resid @ resid / (len(d) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(beta[1] / np.sqrt(cov[1, 1]))


def simulate_bsadf_critical_values(
    n_obs: int,
    min_window: int | None = None,
    n_sims: int = 200,
    seed: int = 20151043,
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.99),
    batch: int = 50,
) -> dict:
    """Fixed-seed Monte Carlo critical values under the random-walk null."""
    rng = np.random.default_rng(seed)
    w0 = psy_min_window(n_obs) if min_window is None else int(min_window)
    sims = []
    for start in range(0, n_sims, batch):
        m = min(batch, n_sims - start)
        paths = np.cumsum(rng.standard_normal((m, n_obs)), axis=1)
        sims.append(bsadf_series(paths, min_window=w0))
    sims = np.vstack(sims)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN warm-up columns
        cv_t = {q: np.nanquantile(sims, q, axis=0) for q in quantiles}
    gsadf = np.nanmax(sims, axis=1)
    cv_gsadf = {q: float(np.quantile(gsadf, q)) for q in quantiles}
    return {"cv_t": cv_t, "cv_gsadf": cv_gsadf, "n_sims": n_sims, "seed": seed, "min_window": w0, "n_obs": n_obs}


@dataclass
class ExuberanceResult:
    table: pd.DataFrame          # date, log_price, bsadf, cv90, cv95, explosive, signal
    gsadf: float
    gsadf_cv95: float
    min_window: int
    n_sims: int
    episodes: pd.DataFrame


def date_stamp_episodes(flag: pd.Series, min_length: int = 5) -> pd.DataFrame:
    """Origination/termination dates of explosive runs lasting ≥ min_length days."""
    rows, start, prev = [], None, None
    for dt, f in flag.items():
        if f and start is None:
            start = dt
        if not f and start is not None:
            run = flag.loc[start:prev]
            if len(run) >= min_length:
                rows.append({"start": start, "end": prev, "days": len(run)})
            start = None
        prev = dt
    if start is not None:
        run = flag.loc[start:]
        if len(run) >= min_length:
            rows.append({"start": start, "end": prev, "days": len(run), "ongoing": True})
    out = pd.DataFrame(rows, columns=["start", "end", "days", "ongoing"])
    out["ongoing"] = out["ongoing"].fillna(False).astype(bool) if not out.empty else out["ongoing"]
    return out


def run_exuberance_test(price: pd.Series, n_sims: int = 200, min_window: int | None = None, cv: dict | None = None) -> ExuberanceResult:
    price = price.dropna()
    y = np.log(price.to_numpy(dtype=float))
    w0 = psy_min_window(len(y)) if min_window is None else int(min_window)
    stat = bsadf_series(y, min_window=w0)
    cv = cv or simulate_bsadf_critical_values(len(y), min_window=w0, n_sims=n_sims)
    tbl = pd.DataFrame({
        "log_price": y, "bsadf": stat,
        "cv90": cv["cv_t"][0.90], "cv95": cv["cv_t"][0.95],
    }, index=price.index)
    tbl["explosive"] = tbl["bsadf"] > tbl["cv95"]
    tbl["signal"] = tbl["explosive"].shift(1, fill_value=False).astype(bool)   # tradable at t+1
    return ExuberanceResult(
        table=tbl, gsadf=float(np.nanmax(stat)), gsadf_cv95=cv["cv_gsadf"][0.95],
        min_window=w0, n_sims=cv["n_sims"], episodes=date_stamp_episodes(tbl["explosive"]),
    )


# ---------------------------------------------------------------------------
# 2. Greenwood–Shleifer–You attributes & composite score
# ---------------------------------------------------------------------------
def _trailing_z(x: pd.Series, window: int = Z_WINDOW, min_periods: int = Z_MIN_PERIODS) -> pd.Series:
    mu = x.rolling(window, min_periods=min_periods).mean()
    sd = x.rolling(window, min_periods=min_periods).std()
    return (x - mu) / sd.replace(0.0, np.nan)


def gsy_features(
    price: pd.Series,
    dollar_volume: pd.Series | None = None,
    credit_spread: pd.Series | None = None,
    market_price: pd.Series | None = None,
) -> pd.DataFrame:
    """Causal GSY attributes for one price index (all values use data ≤ t).

    * ``z_vol``  — 63-day realised annualised volatility, z-scored vs trailing 504 days.
    * ``z_turn`` — 21-day / 252-day average dollar volume, z-scored (dollar volume
      stands in for share turnover because shares outstanding are not in the data).
    * ``z_acc``  — convexity = log return of the last 126 days minus log return of
      the prior 126 days. ``r126/r252 > 0.5`` iff this is positive (for r252 > 0),
      but the difference stays finite when r252 ≈ 0.
    * ``z_cred`` — HY OAS level z-scored vs trailing 504 days (Gilchrist–Zakrajšek).
    * ``runup_2y`` / ``runup_2y_net`` — GSY conditioning variable (raw and net of market).
    """
    price = price.dropna().astype(float)
    lp = np.log(price)
    r = lp.diff()
    out = pd.DataFrame(index=price.index)
    out["vol_63"] = r.rolling(63, min_periods=63).std() * np.sqrt(TRADING_DAYS)
    out["z_vol"] = _trailing_z(out["vol_63"])
    if dollar_volume is not None:
        dv = dollar_volume.reindex(price.index)
        out["turn_ratio"] = dv.rolling(21, min_periods=21).mean() / dv.rolling(252, min_periods=252).mean()
        out["z_turn"] = _trailing_z(out["turn_ratio"])
    out["accel"] = (lp - lp.shift(126)) - (lp.shift(126) - lp.shift(252))
    out["z_acc"] = _trailing_z(out["accel"])
    if credit_spread is not None:
        cs = credit_spread.dropna()
        cs = cs.reindex(cs.index.union(price.index)).ffill().reindex(price.index)
        out["credit"] = cs
        out["z_cred"] = _trailing_z(cs)
    out["runup_2y"] = price / price.shift(GSY_HORIZON) - 1.0
    if market_price is not None:
        mp = market_price.reindex(price.index)
        out["runup_2y_net"] = out["runup_2y"] - (mp / mp.shift(GSY_HORIZON) - 1.0)
    return out


def composite_score(features: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """S_t = Σ w_k Z_k, renormalising weights over the attributes available that day."""
    weights = weights or DEFAULT_WEIGHTS
    cols = [c for c in weights if c in features.columns]
    if not cols:
        return pd.Series(np.nan, index=features.index)
    Z = features[cols]
    W = pd.DataFrame({c: np.where(Z[c].notna(), weights[c], 0.0) for c in cols}, index=Z.index)
    wsum = W.sum(axis=1).replace(0.0, np.nan)
    return (Z.fillna(0.0) * W).sum(axis=1) / wsum * sum(weights[c] for c in cols)


def forward_crash_label(price: pd.Series, horizon: int = GSY_HORIZON, dd: float = GSY_CRASH_DD) -> pd.Series:
    """1 if the price falls ≥ |dd| below P_t at any point in the next ``horizon`` days.

    NaN where the full forward window is not observed (no partial labels).
    """
    p = price.to_numpy(dtype=float)
    n = len(p)
    lab = np.full(n, np.nan)
    for i in range(n - horizon):
        lab[i] = float(p[i + 1:i + 1 + horizon].min() / p[i] - 1.0 <= dd)
    return pd.Series(lab, index=price.index)


def fit_logistic_ridge(s: np.ndarray, yv: np.ndarray, l2: float = 1.0, iters: int = 100) -> tuple[float, float]:
    """Ridge-penalised logistic regression P = σ(θ0 + θ1 s) by Newton–Raphson."""
    X = np.column_stack([np.ones_like(s), s])
    theta = np.zeros(2)
    pen = np.diag([0.0, l2])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ theta)))
        g = X.T @ (yv - p) - pen @ theta
        H = X.T @ (X * (p * (1 - p))[:, None]) + pen
        step = np.linalg.solve(H, g)
        theta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(theta[0]), float(theta[1])


@dataclass
class CrashCalibration:
    theta0: float
    theta1: float
    n_obs: int
    positive_rate: float
    sample_start: str
    sample_end: str
    attributes_used: list[str]
    note: str


def calibrate_on_dotcom(dotcom_price: pd.Series, dotcom_dollar_volume: pd.Series | None, weights: dict[str, float] | None = None, l2: float = 1.0) -> CrashCalibration:
    """Fit (θ0, θ1) on the Nasdaq Composite 1998–2003 GSY crash labels.

    Credit is unavailable for 1998–2003 in the repo (FRED's ICE BofA history is
    limited), so the calibration score renormalises over vol/turnover/accel.
    """
    feats = gsy_features(dotcom_price, dotcom_dollar_volume)
    used = [c for c in (weights or DEFAULT_WEIGHTS) if c in feats.columns]
    # Complete cases only: mixing days where only some attributes exist changes the
    # score's meaning over time (trial CAL01 in trials.csv flipped the slope sign).
    complete = feats[used].dropna()
    s = composite_score(complete, weights)
    lab = forward_crash_label(dotcom_price)
    df = pd.DataFrame({"s": s, "y": lab}).dropna()
    t0, t1 = fit_logistic_ridge(df["s"].to_numpy(), df["y"].to_numpy(), l2=l2)
    return CrashCalibration(
        theta0=t0, theta1=t1, n_obs=len(df), positive_rate=float(df["y"].mean()),
        sample_start=str(df.index.min().date()), sample_end=str(df.index.max().date()),
        attributes_used=used,
        note=("Single-episode calibration (Nasdaq Composite, ^IXIC, Yahoo Finance) with overlapping "
              "24-month labels; ridge λ=%.1f. The sample surrounds the 2000 bust, so the intercept "
              "(base rate) is upward-biased; the slope θ1 is the informative part. Treat as a "
              "calibrated stress index, not a forecast." % l2),
    )


def crash_probability(score: pd.Series | float, theta0: float, theta1: float):
    return 1.0 / (1.0 + np.exp(-(theta0 + theta1 * score)))


def odds_multiplier(score: pd.Series | float, theta1: float):
    """Crash odds relative to an average-stress day (S = 0): exp(θ1·S). Intercept-free."""
    return np.exp(theta1 * score)


def crash_risk_table(features: pd.DataFrame, calib: CrashCalibration, weights: dict[str, float] | None = None) -> pd.DataFrame:
    out = features.copy()
    out["score"] = composite_score(features, weights)
    out["p_crash"] = crash_probability(out["score"], calib.theta0, calib.theta1)
    out["odds_mult"] = odds_multiplier(out["score"], calib.theta1)
    out["score_signal"] = out["score"].shift(1)        # usable at t
    out["p_crash_signal"] = out["p_crash"].shift(1)
    return out


# ---------------------------------------------------------------------------
# 3. Five-point trigger checklist + switch-off
# ---------------------------------------------------------------------------
def trigger_checklist(credit: dict[str, float], bsadf_now: float, cv95_now: float, supply_flags: list[dict]) -> pd.DataFrame:
    from src.credit_signals import TRIGGER_THRESHOLDS as T, traffic_light

    hy = credit["hy_hpc_spread_bps"]
    basis = credit["hyperscaler_10y_basis_bps"]
    crwv = credit["crwv_2031_spread_bps"]
    supply_active = sum(bool(f["active"]) for f in supply_flags)
    rows = [
        ("hy_hpc", "HY HPC credit spread", "Спред HY HPC", f"{hy:.0f} bps",
         "RED > 600 · AMBER 450–600", traffic_light(hy, T["hy_hpc_spread_bps"]["amber"], T["hy_hpc_spread_bps"]["red"]),
         "J.P. Morgan (Oct 29, 2026)"),
        ("basis", "Hyperscaler 10y+ basis vs JULI", "Базис гиперскейлеров 10+ к JULI", f"+{basis:.1f} bps",
         "RED > 75 · AMBER 45–75", traffic_light(basis, T["hyperscaler_10y_basis_bps"]["amber"], T["hyperscaler_10y_basis_bps"]["red"]),
         "J.P. Morgan Fig. 10 (Oct 29, 2026)"),
        ("crwv", "Neocloud distress (CRWV 2031s)", "Стресс неооблаков (CRWV 2031)", f">{crwv:,.0f} bps",
         "RED ≥ 1,000 (reported as '>1,000')", traffic_light(crwv, T["crwv_2031_spread_bps"]["amber"], T["crwv_2031_spread_bps"]["red"], red_inclusive=True),
         "WSJ / J.P. Morgan (Oct–Nov 2026)"),
        ("gsadf", "BSADF explosive equity (AI core)", "BSADF: взрывной рост (ядро ИИ)", f"{bsadf_now:.2f} vs CV95 {cv95_now:.2f}",
         "RED if BSADF_t > CV95_t", "N/A" if np.isnan(bsadf_now) else ("RED" if bsadf_now > cv95_now else "GREEN"),
         "Computed: data/raw/volumes_core.csv"),
        ("supply", "Equity supply + underwriting friction", "Предложение акций и отказ андеррайтеров", f"{supply_active}/{len(supply_flags)} events",
         "RED if all events active (Ofek–Richardson)", "RED" if supply_active == len(supply_flags) else ("AMBER" if supply_active else "GREEN"),
         "WSJ / J.P. Morgan (Oct–Nov 2026)"),
    ]
    return pd.DataFrame(rows, columns=["key", "trigger_en", "trigger_ru", "current", "rule", "status", "source"])


def switch_off_status(hy_hpc_bps: float, bsadf_now: float, cv90_now: float) -> dict:
    """Hedge/short rule turns OFF only when BOTH credit and equity exuberance normalise."""
    from src.credit_signals import SWITCH_OFF

    credit_ok = hy_hpc_bps < SWITCH_OFF["hy_hpc_spread_below_bps"]
    equity_ok = (not np.isnan(bsadf_now)) and bsadf_now < cv90_now
    return {"credit_normalised": bool(credit_ok), "equity_normalised": bool(equity_ok), "switch_off": bool(credit_ok and equity_ok)}
