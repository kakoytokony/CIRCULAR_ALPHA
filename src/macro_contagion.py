"""Extended domino engine: DebtRank, Diebold–Yilmaz spillovers and the 3-scenario
S&P 500 / Tech stress test (METHODOLOGY_AND_SPEC.md §4.2).

This module is independent of ``src/contagion.py`` (the Tab-1 simulator) and does
not change it.

DebtRank (Battiston et al., 2012)
    Impact W[i, j] = min(1, LGD_channel * A[i, j] / E_j): the share of firm j's
    loss buffer wiped out if firm i fails. Exposure direction per
    ``relationships_final.csv`` ``direction_definition``:

    * investment / financing  s→t : s holds a claim on t  → A[t, s] += amount
    * guarantee               s→t : s backstops t         → A[t, s] += amount
    * supplier_customer / commercial_agreement s pays t  → A[s, t] += amount
      (t loses contracted revenue if s fails)
    * R006 (NVIDIA must buy CoreWeave's unsold capacity) is a contingent backstop
      and is treated like a guarantee: A[CoreWeave, NVIDIA].

    Superseded LOI rows R001/R002 are excluded (data dictionary rule), and
    undisclosed amounts are never imputed — they contribute zero dollars and are
    counted separately. Buffer E_j = EBITDA, else cash (same convention as
    ``src/contagion.py``). Entities with no SEC financials in ``financials.csv``
    have no measurable buffer: by default (``no_buffer_mode="block"``) they do not
    *receive* induced distress (their loss capacity is unknown, so nothing is
    imputed) but still propagate it when shocked as an epicentre. The optional
    ``"pass_through"`` mode instead sets W = 1 on any disclosed exposure. They
    carry zero value weight v_j either way.

Diebold–Yilmaz (2014) generalised FEVD connectedness on a VAR(p), with
Pesaran–Shin (1998) generalised impulse responses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SUPERSEDED_LOI_IDS = {"R001", "R002"}
CONTINGENT_BACKSTOP_IDS = {"R006"}
DEFAULT_LGD = {"equity": 1.0, "guarantee": 1.0, "credit": 1.0, "revenue": 0.25}

TIERS_CORE = {
    "T1 Neoclouds": ["CRWV", "APLD", "GLXY"],
    "T2 Chipmakers": ["NVDA", "AVGO", "AMD"],
    "T3 Hyperscalers": ["ORCL", "META", "AMZN", "GOOGL", "MSFT"],
    "T4 S&P 500": ["SPY"],
}
TIERS_EXTENDED = {
    **TIERS_CORE,
    "Nasdaq-100": ["QQQ"],
    "IG credit": ["LQD"],
    "HY credit": ["HYG"],
    "EM USD debt": ["EMB"],
    "Power / IPP": ["VST", "CEG"],
    "Utilities": ["XLU"],
    "Private credit": ["BX"],
}
AI_IN_SPX = ["NVDA", "MSFT", "AMZN", "GOOGL", "META", "AVGO", "ORCL", "AMD"]
DOTCOM_SPX_NASDAQ_ELASTICITY_CAP = 0.63   # -49.1% / -77.9% (historical_analog)


# ---------------------------------------------------------------------------
# DebtRank
# ---------------------------------------------------------------------------
@dataclass
class ExposureNetwork:
    ids: list[str]
    names: dict[str, str]
    tickers: dict[str, str]
    A: np.ndarray                 # A[i, j] = LGD-weighted $ exposure of j to i's failure
    buffer: np.ndarray            # E_j (NaN if unknown)
    buffer_basis: list[str]
    W: np.ndarray                 # impact matrix in [0, 1]
    v: np.ndarray                 # economic value weights (sum 1 over known buffers)
    undisclosed_links: int
    used_links: pd.DataFrame


def _channel(rtype: str, rid: str) -> str:
    if rid in CONTINGENT_BACKSTOP_IDS:
        return "guarantee"
    return {"investment": "equity", "financing": "credit", "guarantee": "guarantee"}.get(rtype, "revenue")


def build_exposure_network(
    companies: pd.DataFrame,
    relationships: pd.DataFrame,
    financials: pd.DataFrame,
    confirmed_only: bool = False,
    lgd: dict[str, float] | None = None,
    no_buffer_mode: str = "block",
) -> ExposureNetwork:
    lgd = {**DEFAULT_LGD, **(lgd or {})}
    ids = companies["company_id"].astype(str).tolist()
    idx = {c: k for k, c in enumerate(ids)}
    names = dict(zip(ids, companies["company_name"].astype(str)))
    tick = dict(zip(ids, companies["public_ticker"].fillna("").astype(str).str.strip()))
    rel = relationships.copy()
    rel["relationship_id"] = rel["relationship_id"].astype(str)
    if "R003" in set(rel["relationship_id"]):
        rel = rel[~rel["relationship_id"].isin(SUPERSEDED_LOI_IDS)]
    if confirmed_only:
        rel = rel[rel["evidence_status"].astype(str).str.lower() == "confirmed"]
    n = len(ids)
    A = np.zeros((n, n))
    undisclosed, used = 0, []
    for _, r in rel.iterrows():
        s, t = str(r["source_entity_id"]), str(r["target_entity_id"])
        if s not in idx or t not in idx or s == t:
            continue
        rtype = str(r["relationship_type"]).strip().lower()
        ch = _channel(rtype, r["relationship_id"])
        failer, exposed = (t, s) if ch in ("equity", "guarantee", "credit") else (s, t)
        amt = pd.to_numeric(r["amount"], errors="coerce")
        if pd.isna(amt):
            undisclosed += 1
            continue
        A[idx[failer], idx[exposed]] += lgd[ch] * float(amt)
        used.append({"relationship_id": r["relationship_id"], "fails": names[failer], "exposed": names[exposed],
                     "channel": ch, "amount_usd": float(amt), "lgd": lgd[ch], "evidence": r["evidence_status"]})
    fin = financials.copy()
    fin["company_id"] = fin["company_id"].astype(str)
    fin = fin.drop_duplicates("company_id", keep="last").set_index("company_id")
    buf, basis = np.full(n, np.nan), []
    for c in ids:
        e = pd.to_numeric(fin.loc[c, "ebitda"], errors="coerce") if c in fin.index else np.nan
        cash = pd.to_numeric(fin.loc[c, "cash_and_equivalents"], errors="coerce") if c in fin.index else np.nan
        if pd.notna(e) and e > 0:
            buf[idx[c]] = float(e); basis.append("EBITDA")
        elif pd.notna(cash) and cash > 0:
            buf[idx[c]] = float(cash); basis.append("Cash (EBITDA n/a)")
        else:
            basis.append("No SEC financials (no buffer)")
    with np.errstate(divide="ignore", invalid="ignore"):
        unknown = (A > 0).astype(float) if no_buffer_mode == "pass_through" else np.zeros_like(A)
        W = np.where(np.isnan(buf)[None, :], unknown, np.minimum(1.0, A / buf[None, :]))
    np.fill_diagonal(W, 0.0)
    known = np.nan_to_num(buf)
    v = known / known.sum() if known.sum() > 0 else np.zeros(n)
    return ExposureNetwork(ids, names, tick, A, buf, basis, W, v, undisclosed, pd.DataFrame(used))


def debtrank_propagate(W: np.ndarray, h0: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Battiston et al. (2012) DebtRank: each distressed node propagates once.

    States: U (undistressed), D (distressed, propagates next round), I (inactive).
    h_j(t+1) = min(1, h_j(t) + Σ_{i∈D(t)} W_ij h_i(t)) for j not inactive.
    """
    h = np.clip(np.asarray(h0, dtype=float), 0.0, 1.0).copy()
    n = len(h)
    state = np.where(h > 0, 1, 0)        # 0=U, 1=D, 2=I
    for _ in range(max_iter):
        active = state == 1
        if not active.any():
            break
        inc = W[active].T @ h[active]
        state[active] = 2
        recv = state != 2
        new_h = h.copy()
        new_h[recv] = np.minimum(1.0, h[recv] + inc[recv])
        newly = recv & (new_h > h)
        h = new_h
        state[newly] = 1
    return h


def debtrank_scores(net: ExposureNetwork, shock: float = 1.0) -> pd.DataFrame:
    """DebtRank_i = Σ_{j≠i} (h_j(T) − h_j(0)) v_j when node i alone receives ``shock``."""
    rows = []
    for k, cid in enumerate(net.ids):
        h0 = np.zeros(len(net.ids)); h0[k] = shock
        hT = debtrank_propagate(net.W, h0)
        induced = hT - h0
        induced[k] = 0.0
        rows.append({"company_id": cid, "company": net.names[cid], "ticker": net.tickers[cid] or "Private",
                     "debtrank": float(induced @ net.v), "nodes_hit": int((induced > 1e-9).sum()),
                     "buffer_basis": net.buffer_basis[k]})
    return pd.DataFrame(rows).sort_values("debtrank", ascending=False).reset_index(drop=True)


def shock_simulation(net: ExposureNetwork, shocked: dict[str, float]) -> pd.DataFrame:
    h0 = np.zeros(len(net.ids))
    for cid, s in shocked.items():
        h0[net.ids.index(cid)] = s
    hT = debtrank_propagate(net.W, h0)
    return pd.DataFrame({"company_id": net.ids, "company": [net.names[c] for c in net.ids],
                         "ticker": [net.tickers[c] or "Private" for c in net.ids],
                         "h0": h0, "h_final": hT, "buffer_basis": net.buffer_basis}).sort_values("h_final", ascending=False)


# ---------------------------------------------------------------------------
# Diebold–Yilmaz connectedness
# ---------------------------------------------------------------------------
def tier_returns(prices_wide: pd.DataFrame, tiers: dict[str, list[str]]) -> pd.DataFrame:
    """Equal-weight daily log returns per tier using available members each day."""
    lr = np.log(prices_wide).diff()
    out = {}
    for name, members in tiers.items():
        cols = [m for m in members if m in lr.columns]
        if cols:
            out[name] = lr[cols].mean(axis=1, skipna=True)
    return pd.DataFrame(out).dropna(how="any")


def fit_var(Y: np.ndarray, p: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """OLS VAR(p). Returns coefficient blocks (p, N, N) and residual covariance."""
    T, N = Y.shape
    X = np.hstack([np.ones((T - p, 1))] + [Y[p - k - 1:T - k - 1] for k in range(p)])
    Yt = Y[p:]
    B, *_ = np.linalg.lstsq(X, Yt, rcond=None)
    resid = Yt - X @ B
    sigma = resid.T @ resid / (len(Yt) - X.shape[1])
    Phi = np.stack([B[1 + k * N:1 + (k + 1) * N].T for k in range(p)])
    return Phi, sigma


def generalized_fevd(Phi: np.ndarray, sigma: np.ndarray, H: int = 10) -> np.ndarray:
    """Row-normalised generalised FEVD θ̃_ij(H) (Pesaran–Shin; Diebold–Yilmaz 2012/2014)."""
    p, N, _ = Phi.shape
    A = [np.eye(N)]
    for h in range(1, H):
        A.append(sum(Phi[k] @ A[h - k - 1] for k in range(min(p, h))))
    num = np.zeros((N, N)); den = np.zeros(N)
    sd = np.diag(sigma)
    for Ah in A:
        AS = Ah @ sigma
        num += AS ** 2 / sd[None, :]
        den += np.einsum("ij,jk,ik->i", Ah, sigma, Ah)
    theta = num / den[:, None]
    return theta / theta.sum(axis=1, keepdims=True)


def connectedness_table(theta: np.ndarray, names: list[str]) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    N = len(names)
    off = theta * (1 - np.eye(N)) * 100.0
    tci = off.sum() / N
    frm = off.sum(axis=1)
    to = off.sum(axis=0)
    mat = pd.DataFrame(theta * 100.0, index=names, columns=names)
    summ = pd.DataFrame({"tier": names, "from_others": frm, "to_others": to, "net": to - frm})
    return mat, float(tci), summ


def diebold_yilmaz(returns: pd.DataFrame, p: int = 2, H: int = 10, window: int | None = 252) -> dict:
    """Connectedness on the trailing ``window`` days ending at the last date."""
    R = returns.dropna()
    if window is not None:
        R = R.iloc[-window:]
    Phi, sigma = fit_var(R.to_numpy(), p)
    theta = generalized_fevd(Phi, sigma, H)
    mat, tci, summ = connectedness_table(theta, list(R.columns))
    return {"matrix": mat, "tci": tci, "summary": summ, "start": R.index.min(), "end": R.index.max(), "n_obs": len(R)}


def rolling_tci(returns: pd.DataFrame, p: int = 2, H: int = 10, window: int = 252, step: int = 5) -> pd.Series:
    """TCI at date t uses only the ``window`` returns ending at t (strictly trailing)."""
    R = returns.dropna()
    vals, dates = [], []
    arr = R.to_numpy()
    for end in range(window, len(R) + 1, step):
        Phi, sigma = fit_var(arr[end - window:end], p)
        theta = generalized_fevd(Phi, sigma, H)
        N = theta.shape[0]
        vals.append((theta * (1 - np.eye(N))).sum() / N * 100.0)
        dates.append(R.index[end - 1])
    return pd.Series(vals, index=pd.DatetimeIndex(dates), name="TCI")


# ---------------------------------------------------------------------------
# S&P 500 / Tech 3-scenario stress test
# ---------------------------------------------------------------------------
def nnls(X: np.ndarray, y: np.ndarray, iters: int = 500) -> np.ndarray:
    """Lawson–Hanson non-negative least squares (scipy-free fallback is fine here)."""
    try:
        from scipy.optimize import nnls as _nnls
        return _nnls(X, y)[0]
    except Exception:  # pragma: no cover - scipy is in requirements
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            g = X.T @ (y - X @ w)
            w = np.maximum(0.0, w + 1e-3 * g)
        return w


def implied_effective_weights(returns: pd.DataFrame, target: str, members: list[str], window: int = 252) -> pd.Series:
    """Returns-based style analysis (Sharpe 1992): non-negative exposures of the
    target index's daily returns to AI-core returns over the trailing window.

    These are *effective* sensitivities, not official index weights: because AI
    names co-move with the other ~490 S&P 500 stocks they will typically exceed
    the true cap weights. Users can override them in the UI.
    """
    cols = [m for m in members if m in returns.columns]
    R = returns[[target] + cols].dropna().iloc[-window:]
    w = nnls(R[cols].to_numpy(), R[target].to_numpy())
    return pd.Series(w, index=cols)


def downside_spill_beta(returns: pd.DataFrame, target: str, weights: pd.Series, tail_q: float = 0.10, cap: float = DOTCOM_SPX_NASDAQ_ELASTICITY_CAP) -> dict:
    """β_spill: slope of the non-AI residual (scaled to 1 − Σw) on the AI basket
    on the bottom-``tail_q`` AI-basket days; bounded to [0, cap] (dot-com 0.63)."""
    cols = list(weights.index)
    R = returns[[target] + cols].dropna()
    w = weights.to_numpy()
    wsum = w.sum()
    ai = R[cols].to_numpy() @ (w / wsum) if wsum > 0 else R[cols].mean(axis=1).to_numpy()
    non_ai_share = max(1e-6, 1.0 - wsum)
    resid = (R[target].to_numpy() - R[cols].to_numpy() @ w) / non_ai_share
    mask = ai <= np.quantile(ai, tail_q)
    x, yv = ai[mask], resid[mask]
    raw = float(np.cov(x, yv, ddof=1)[0, 1] / np.var(x, ddof=1)) if mask.sum() > 3 else 0.0
    return {"beta_raw": raw, "beta": float(np.clip(raw, 0.0, cap)), "n_tail_days": int(mask.sum()), "cap": cap}


SCENARIOS = {
    "S1": {
        "name_en": "Capex slowdown & 'Blue Wave' bottleneck (mild)",
        "name_ru": "Замедление Capex и барьеры «Синей волны» (мягкий)",
        "tier_shock_pct": {"T1 Neoclouds": -42.5, "T2 Chipmakers": -15.0, "T3 Hyperscalers": 0.0, "Private credit": 0.0},
        "epicenters": {},
        "basis": "Spec §4.2.3: neoclouds −35% to −50% (midpoint), chipmakers −15%, hyperscalers −5% to +5% (midpoint). No DebtRank default.",
    },
    "S2": {
        "name_en": "Circular vendor-financing unwind (2001 telecom analog, severe)",
        "name_ru": "Распад кругового вендорного финансирования (аналог телекома 2001, тяжёлый)",
        "tier_shock_pct": {"T1 Neoclouds": -50.0, "T2 Chipmakers": -15.0, "T3 Hyperscalers": 0.0, "Private credit": 0.0},
        "epicenters": {"C005": 1.0, "C010": 1.0},
        "basis": "Spec §4.2.3: a neocloud (CoreWeave) and an AI lab (Anthropic) default; SPV guarantees are called via DebtRank on top of S1's upper-end tier shocks.",
    },
    "S3": {
        "name_en": "Systemic credit & macro cascade (tail risk)",
        "name_ru": "Системный кредитный и макрокаскад (хвостовой риск)",
        "tier_shock_pct": {"T1 Neoclouds": -82.9, "T2 Chipmakers": -77.9, "T3 Hyperscalers": -49.1, "Private credit": -49.1},
        "epicenters": {"C005": 1.0, "C010": 1.0, "C006": 1.0},
        "basis": "2000–2002 envelope: Nasdaq-100 −82.9% (neoclouds), Nasdaq Composite −77.9% (chipmakers), S&P 500 −49.1% (hyperscalers & private credit) + DebtRank with OpenAI, Anthropic and CoreWeave defaulting.",
    },
}

TICKER_TIER = {t: tier for tier, ms in {**TIERS_CORE, "Private credit": ["BX"]}.items() for t in ms}


def firm_scenario_drawdowns(net: ExposureNetwork, scenario: dict, dd_per_unit_distress: float = 50.0) -> pd.DataFrame:
    """Firm drawdown compounding the tier shock with DebtRank-induced distress:

    1 + ΔP_i = (1 + tier_shock_i) × (1 − dd_per_unit_distress/100 × Δh_i),
    where Δh_i = h_i(T) − h_i(0). Epicentres (h_i(0) = 1) are set to −100%.
    """
    h0 = np.zeros(len(net.ids))
    for cid, s in scenario.get("epicenters", {}).items():
        if cid in net.ids:
            h0[net.ids.index(cid)] = s
    hT = debtrank_propagate(net.W, h0) if h0.any() else h0
    rows = []
    for k, cid in enumerate(net.ids):
        t = net.tickers[cid]
        if not t or t not in TICKER_TIER:
            continue
        base = scenario["tier_shock_pct"].get(TICKER_TIER[t], 0.0)
        induced = hT[k] - h0[k]
        if np.isnan(net.buffer[k]):
            induced = 0.0            # no SEC buffer: price impact = tier shock only (flagged)
        dd = -100.0 if h0[k] >= 1.0 else 100.0 * ((1 + base / 100.0) * (1 - dd_per_unit_distress / 100.0 * induced) - 1.0)
        rows.append({"ticker": t, "company": net.names[cid], "tier": TICKER_TIER[t], "tier_shock_pct": base,
                     "debtrank_distress": round(float(hT[k]), 4), "scenario_dd_pct": round(dd, 2),
                     "buffer_basis": net.buffer_basis[k]})
    return pd.DataFrame(rows)


def index_stress_forecast(firm_dd: pd.DataFrame, weights: pd.Series, beta_spill: float) -> dict:
    """ΔIndex = Σ w_i ΔP_i + β_spill (1 − Σ w_i) ΔP_AI,avg (spec §4.2.3)."""
    dd = firm_dd.set_index("ticker")["scenario_dd_pct"].reindex(weights.index).fillna(0.0)
    w = weights.astype(float)
    direct = float((w * dd).sum())
    wsum = float(w.sum())
    ai_avg = float((w * dd).sum() / wsum) if wsum > 0 else 0.0
    spill = float(beta_spill * max(0.0, 1.0 - wsum) * ai_avg)
    return {"direct_pct": direct, "spill_pct": spill, "total_pct": max(-100.0, direct + spill),
            "ai_weight_sum": wsum, "ai_avg_dd_pct": ai_avg, "beta_spill": beta_spill}
