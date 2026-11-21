"""Cluster Analysis module implementing KBTU QIM Cluster Analysis Lecture standards.

Covers all six audit checks from the lecture (Dias Kumarbekov, Fall 2026):
1. Standardization (z-scores) before Euclidean K-Means.
2. Noise check: Elbow (WCSS) and Average Silhouette vs. 200 column-shuffled copies (95th pct null band).
3. Multi-start K-Means (nstart >= 50, fixed seed) to avoid local optima.
4. Hierarchical Ward clustering on market-removed (SPY-residual) correlation distance d = sqrt(2*(1-rho)).
5. Stability check: Adjusted Rand Index (ARI) vs. network-role labels and across time subperiods.
6. Out-of-sample portfolio check: Equal Weight (1/N) vs. Inverse Volatility vs. Cluster Risk Parity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist, squareform

TRADING_DAYS = 252


def build_risk_features(
    prices: pd.DataFrame,
    tickers: list[str] | None = None,
    benchmark: str = "SPY",
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Compute per-stock risk features: annualized return, volatility, beta vs SPY, max drawdown."""
    data = prices[["date", "ticker", "adjusted_close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["adjusted_close"] = pd.to_numeric(data["adjusted_close"], errors="coerce")
    data = data.dropna()
    if start_date is not None:
        data = data[data["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        data = data[data["date"] <= pd.to_datetime(end_date)]

    piv = data.pivot_table(index="date", columns="ticker", values="adjusted_close").sort_index()
    if benchmark not in piv.columns:
        raise ValueError(f"Benchmark {benchmark} missing from prices.")

    if tickers is None:
        selected = [c for c in piv.columns if c != benchmark and piv[c].notna().sum() >= 200]
    else:
        selected = [t for t in tickers if t in piv.columns and t != benchmark]

    common = piv[selected + [benchmark]].dropna()
    if len(common) < 30:
        raise ValueError("Not enough common dates to compute clustering risk features.")

    daily = common.pct_change().dropna()
    mkt = daily[benchmark]
    mkt_var = float(mkt.var(ddof=1))

    rows = []
    n_years = len(daily) / TRADING_DAYS
    for t in selected:
        r = daily[t]
        wealth = np.concatenate([[1.0], np.cumprod(1.0 + r.to_numpy())])
        peak = np.maximum.accumulate(wealth)
        mdd = float((wealth / peak - 1.0).min() * 100.0)
        cagr = float((wealth[-1] ** (1.0 / n_years) - 1.0) * 100.0)
        vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100.0)
        beta = float(r.cov(mkt) / mkt_var) if mkt_var > 0 else np.nan
        rows.append({
            "ticker": t,
            "ann_return_pct": cagr,
            "ann_vol_pct": vol,
            "beta_vs_spy": beta,
            "max_drawdown_pct": mdd,
        })
    return pd.DataFrame(rows).set_index("ticker")


def standardize_features(
    features: pd.DataFrame,
    means: pd.Series | None = None,
    stds: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Z-score standardize columns: z = (x - mean) / std (ddof=1, matching R's scale())."""
    mu = features.mean(axis=0) if means is None else means
    sd = features.std(axis=0, ddof=1) if stds is None else stds
    safe_sd = sd.replace(0, np.nan)
    z = (features - mu) / safe_sd
    return z.fillna(0.0), mu, sd


def _kmeans_single(X: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    n = X.shape[0]
    # K-Means++ initialization
    centers = np.empty((k, X.shape[1]), dtype=float)
    first_idx = int(rng.integers(0, n))
    centers[0] = X[first_idx]
    for c in range(1, k):
        dists_sq = np.min(((X[:, None, :] - centers[None, :c, :]) ** 2).sum(axis=2), axis=1)
        total = dists_sq.sum()
        if total <= 0:
            idx = int(rng.integers(0, n))
        else:
            probs = dists_sq / total
            idx = int(rng.choice(n, p=probs))
        centers[c] = X[idx]

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = X[mask].mean(axis=0)
            else:
                centers[j] = X[int(rng.integers(0, n))]
    wcss = float(sum(((X[labels == j] - centers[j]) ** 2).sum() for j in range(k) if np.any(labels == j)))
    return labels, centers, wcss


def kmeans_multistart(
    X: np.ndarray, k: int, nstart: int = 100, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run K-Means with nstart random initializations and return the lowest-WCSS solution."""
    if k < 1 or k > X.shape[0]:
        raise ValueError(f"k={k} must be between 1 and n={X.shape[0]}.")
    rng = np.random.default_rng(seed)
    best_wcss = np.inf
    best_labels = None
    best_centers = None
    for _ in range(nstart):
        labels, centers, wcss = _kmeans_single(X, k, rng)
        if wcss < best_wcss:
            best_wcss = wcss
            best_labels = labels
            best_centers = centers
    return best_labels, best_centers, float(best_wcss)


def average_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Exact Rousseeuw (1987) average silhouette score in [-1, 1]."""
    unique = np.unique(labels)
    n = X.shape[0]
    if len(unique) < 2 or len(unique) >= n:
        return float("nan")
    dist_mat = squareform(pdist(X, metric="euclidean"))
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        own = labels == labels[i]
        own_count = int(own.sum())
        if own_count <= 1:
            scores[i] = 0.0
            continue
        a_i = float(dist_mat[i, own].sum() / (own_count - 1))
        b_i = min(
            float(dist_mat[i, labels == c].mean())
            for c in unique
            if c != labels[i]
        )
        denom = max(a_i, b_i)
        scores[i] = (b_i - a_i) / denom if denom > 0 else 0.0
    return float(scores.mean())


def evaluate_k_with_null_band(
    X: np.ndarray,
    k_values: list[int] | None = None,
    n_shuffles: int = 200,
    nstart: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate Elbow (WCSS) and Silhouette vs 95th percentile of column-shuffled null copies."""
    n = X.shape[0]
    if k_values is None:
        max_k = min(6, n - 1)
        k_values = list(range(2, max_k + 1))

    rng = np.random.default_rng(seed)
    rows = []
    for k in k_values:
        if k >= n:
            continue
        labels, _, wcss = kmeans_multistart(X, k, nstart=nstart, seed=seed + k)
        real_sil = average_silhouette(X, labels)

        null_sils = []
        for s in range(n_shuffles):
            X_shuf = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
            shuf_labels, _, _ = kmeans_multistart(X_shuf, k, nstart=10, seed=seed + 1000 * k + s)
            val = average_silhouette(X_shuf, shuf_labels)
            if not np.isnan(val):
                null_sils.append(val)
        null_95 = float(np.percentile(null_sils, 95)) if null_sils else np.nan
        rows.append({
            "k": int(k),
            "wcss": round(wcss, 4),
            "silhouette_real": round(real_sil, 4),
            "silhouette_null_95": round(null_95, 4),
            "above_noise_band": bool(real_sil > null_95) if pd.notna(null_95) else False,
        })
    return pd.DataFrame(rows)


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Exact Hubert & Arabie (1985) Adjusted Rand Index (1=identical, 0=chance)."""
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if len(a) != len(b) or len(a) < 2:
        return float("nan")

    _, a_idx = np.unique(a, return_inverse=True)
    _, b_idx = np.unique(b, return_inverse=True)
    contingency = np.zeros((a_idx.max() + 1, b_idx.max() + 1), dtype=np.int64)
    for i, j in zip(a_idx, b_idx):
        contingency[i, j] += 1

    def comb2(x):
        return x * (x - 1) / 2.0

    sum_comb_c = comb2(contingency).sum()
    sum_comb_a = comb2(contingency.sum(axis=1)).sum()
    sum_comb_b = comb2(contingency.sum(axis=0)).sum()
    total_comb = comb2(len(a))
    if total_comb == 0:
        return 0.0
    expected = (sum_comb_a * sum_comb_b) / total_comb
    max_idx = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_idx - expected
    if denom == 0:
        return 1.0 if sum_comb_c == expected else 0.0
    return float((sum_comb_c - expected) / denom)


def hierarchical_residual_clustering(
    prices: pd.DataFrame,
    tickers: list[str] | None = None,
    benchmark: str = "SPY",
    k: int = 3,
    start_date=None,
    end_date=None,
) -> dict:
    """Ward hierarchical clustering on market-removed (SPY residual) correlation distance.

    Implements Mantegna (1999) distance d = sqrt(2 * (1 - rho)) after regressing out SPY.
    Returns linkage matrix, cluster labels, sorted correlation matrices, within/across-cluster
    correlations, and effective number of independent bets.
    """
    data = prices[["date", "ticker", "adjusted_close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["adjusted_close"] = pd.to_numeric(data["adjusted_close"], errors="coerce")
    data = data.dropna()
    if start_date is not None:
        data = data[data["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        data = data[data["date"] <= pd.to_datetime(end_date)]

    piv = data.pivot_table(index="date", columns="ticker", values="adjusted_close").sort_index()
    if benchmark not in piv.columns:
        raise ValueError(f"Benchmark {benchmark} missing from prices.")
    if tickers is None:
        selected = [c for c in piv.columns if c != benchmark and piv[c].notna().sum() >= 200]
    else:
        selected = [t for t in tickers if t in piv.columns and t != benchmark]

    common = piv[selected + [benchmark]].dropna()
    daily = common.pct_change().dropna()
    mkt = daily[benchmark].to_numpy()
    X_mkt = np.column_stack([np.ones(len(mkt)), mkt])

    residuals = pd.DataFrame(index=daily.index, columns=selected, dtype=float)
    for t in selected:
        y = daily[t].to_numpy()
        coef, _, _, _ = np.linalg.lstsq(X_mkt, y, rcond=None)
        residuals[t] = y - X_mkt @ coef

    raw_corr = daily[selected].corr()
    res_corr = residuals.corr()

    # Mantegna (1999) distance on market-removed residuals
    dist_mat = np.sqrt(np.clip(2.0 * (1.0 - res_corr.to_numpy()), 0.0, 4.0))
    np.fill_diagonal(dist_mat, 0.0)
    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method="ward")

    n_clusters = min(k, len(selected))
    cluster_ids = fcluster(Z, t=n_clusters, criterion="maxclust")
    order = leaves_list(Z)
    ordered_tickers = [selected[i] for i in order]

    # Within-cluster vs across-cluster mean raw correlation
    within_vals, across_vals = [], []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            val = float(raw_corr.iloc[i, j])
            if cluster_ids[i] == cluster_ids[j]:
                within_vals.append(val)
            else:
                across_vals.append(val)

    # Effective number of independent bets from raw correlation eigenvalues (Meucci / entropy)
    eigvals = np.linalg.eigvalsh(raw_corr.to_numpy())
    eigvals = np.clip(eigvals, 1e-12, None)
    p = eigvals / eigvals.sum()
    n_eff_bets = float(np.exp(-np.sum(p * np.log(p))))

    n_pairs = len(selected) * (len(selected) - 1) / 2
    mean_raw_corr = float((raw_corr.to_numpy().sum() - len(selected)) / (2 * n_pairs)) if n_pairs > 0 else np.nan
    mean_res_corr = float((res_corr.to_numpy().sum() - len(selected)) / (2 * n_pairs)) if n_pairs > 0 else np.nan

    clusters_df = pd.DataFrame({
        "ticker": selected,
        "ward_cluster": cluster_ids,
    }).sort_values(["ward_cluster", "ticker"]).reset_index(drop=True)

    return {
        "tickers": selected,
        "ordered_tickers": ordered_tickers,
        "linkage_matrix": Z,
        "clusters": clusters_df,
        "raw_corr": raw_corr.loc[ordered_tickers, ordered_tickers],
        "residual_corr": res_corr.loc[ordered_tickers, ordered_tickers],
        "mean_raw_corr": round(mean_raw_corr, 3),
        "mean_residual_corr": round(mean_res_corr, 3),
        "within_cluster_corr": round(float(np.mean(within_vals)), 3) if within_vals else np.nan,
        "across_cluster_corr": round(float(np.mean(across_vals)), 3) if across_vals else np.nan,
        "effective_independent_bets": round(n_eff_bets, 2),
        "window_start": str(common.index.min().date()),
        "window_end": str(common.index.max().date()),
    }


def oos_cluster_portfolio_comparison(
    prices: pd.DataFrame,
    tickers: list[str] | None = None,
    benchmark: str = "SPY",
    split_date: str = "2025-01-01",
    k: int = 3,
    risk_free_annual_pct: float = 4.0,
) -> tuple[pd.DataFrame, dict]:
    """Out-of-sample test (Lecture Part D): fit clusters before split_date, evaluate after.

    Compares Equal Weight (1/N), Inverse Volatility, Cluster Risk Parity, and SPY.
    """
    train_res = hierarchical_residual_clustering(
        prices, tickers=tickers, benchmark=benchmark, k=k, end_date=pd.to_datetime(split_date) - pd.Timedelta(days=1)
    )
    selected = train_res["tickers"]
    clusters_map = dict(zip(train_res["clusters"]["ticker"], train_res["clusters"]["ward_cluster"]))

    data = prices[["date", "ticker", "adjusted_close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["adjusted_close"] = pd.to_numeric(data["adjusted_close"], errors="coerce")
    piv = data.dropna().pivot_table(index="date", columns="ticker", values="adjusted_close").sort_index()
    common = piv[selected + [benchmark]].dropna()
    daily = common.pct_change().dropna()

    train_r = daily[daily.index < pd.to_datetime(split_date)]
    test_r = daily[daily.index >= pd.to_datetime(split_date)]
    if len(train_r) < 30 or len(test_r) < 30:
        raise ValueError("Insufficient train/test observations for OOS cluster portfolio test.")

    # Weights computed strictly from train_r (no look-ahead)
    n = len(selected)
    w_eq = pd.Series(1.0 / n, index=selected)

    vols = train_r[selected].std(ddof=1)
    w_invvol = (1.0 / vols) / (1.0 / vols).sum()

    # Cluster Risk Parity: inverse-variance across clusters, inverse-vol within each cluster
    w_crp = pd.Series(0.0, index=selected)
    cluster_ids = sorted(set(clusters_map.values()))
    cluster_vols = {}
    within_weights = {}
    for cid in cluster_ids:
        members = [t for t in selected if clusters_map[t] == cid]
        m_inv = (1.0 / vols[members]) / (1.0 / vols[members]).sum()
        within_weights[cid] = m_inv
        c_ret = (train_r[members] * m_inv).sum(axis=1)
        cluster_vols[cid] = float(c_ret.std(ddof=1))
    c_inv_var = {cid: 1.0 / max(v, 1e-8) for cid, v in cluster_vols.items()}
    c_total = sum(c_inv_var.values())
    for cid in cluster_ids:
        c_weight = c_inv_var[cid] / c_total
        for t, wt in within_weights[cid].items():
            w_crp[t] = c_weight * wt

    daily_rf = risk_free_annual_pct / 100.0 / TRADING_DAYS
    n_years = len(test_r) / TRADING_DAYS

    portfolios = {
        "Equal Weight (1/N)": (test_r[selected] * w_eq).sum(axis=1),
        "Inverse Volatility": (test_r[selected] * w_invvol).sum(axis=1),
        "Cluster Risk Parity": (test_r[selected] * w_crp).sum(axis=1),
        benchmark: test_r[benchmark],
    }

    rows = []
    for name, r in portfolios.items():
        wealth = np.concatenate([[1.0], np.cumprod(1.0 + r.to_numpy())])
        peak = np.maximum.accumulate(wealth)
        mdd = float((wealth / peak - 1.0).min() * 100.0)
        cagr = float((wealth[-1] ** (1.0 / n_years) - 1.0) * 100.0)
        vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100.0)
        excess = r - daily_rf
        sr = float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS)) if excess.std(ddof=1) > 0 else np.nan
        rows.append({
            "Portfolio": name,
            "OOS CAGR (%)": round(cagr, 2),
            "OOS Vol (%)": round(vol, 2),
            "OOS Sharpe": round(sr, 2),
            "OOS Max Drawdown (%)": round(mdd, 2),
        })

    info = {
        "train_window": f"{train_r.index.min().date()} to {train_r.index.max().date()}",
        "test_window": f"{test_r.index.min().date()} to {test_r.index.max().date()}",
        "weights_crp": w_crp.round(4).to_dict(),
    }
    return pd.DataFrame(rows), info
