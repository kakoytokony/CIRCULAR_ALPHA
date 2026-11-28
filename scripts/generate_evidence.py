"""Programmatic evidence generator for Group Assignment 2 (The Rule Desk) & Cluster Lens.

Generates all E1-E8 tables, trials.csv, and publication-ready PNG figures into evidence/
from the saved reproducible CSV snapshots (zero manual typing of numbers).
Run with: python3 scripts/generate_evidence.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import load_all_prices, load_companies, load_relationships
from src.backtest import (
    build_equal_weight_basket,
    cost_sensitivity_and_breakeven,
    downside_risk_summary,
    expected_max_noise_sharpe,
    paired_block_bootstrap_sharpe_diff,
    parameter_neighbors_check,
    run_canary_lookahead_test,
    run_rule_backtest,
    sharpe_and_se,
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

EVIDENCE_DIR = ROOT / "evidence"
CYCLE_TICKERS = ["AMD", "MSFT", "NVDA"]
RF_ANNUAL_PCT = 4.0
SELECTED_LOOKBACK = 100
BASE_COST_BPS = 10.0

TRAIN_START, TRAIN_END = "2023-01-03", "2024-12-31"
VAL_START, VAL_END = "2025-01-01", "2025-12-31"
TEST_START, TEST_END = "2026-01-01", "2026-10-08"


def upsert_trials(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Merge GA2 trial rows into the existing ledger without deleting any other row.

    Rows whose ``config_id`` this script regenerates (T01-T05, V01, TEST01) are
    updated in place; every other row (e.g. V3-*) is kept unchanged and in order;
    config_ids not yet in the ledger are appended at the end.
    """
    if not path.exists():
        return new_rows
    old = pd.read_csv(path, dtype=str, keep_default_na=False)
    new = new_rows.astype(object).where(new_rows.notna(), "").astype(str)
    by_id = new.set_index("config_id")
    columns = list(dict.fromkeys(list(old.columns) + list(new.columns)))
    rows = []
    for _, row in old.iterrows():
        cid = row["config_id"]
        rows.append(by_id.loc[cid].to_dict() | {"config_id": cid} if cid in by_id.index else row.to_dict())
    seen = set(old["config_id"])
    rows += [r for r in new.to_dict("records") if r["config_id"] not in seen]
    return pd.DataFrame(rows).reindex(columns=columns).fillna("")


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_all_prices()
    basket_df = build_equal_weight_basket(prices, CYCLE_TICKERS)
    price_series = basket_df["basket_price"]

    # Canary test (must pass before generating any evidence)
    assert run_canary_lookahead_test(price_series, lookback=20, split_idx=120), "CANARY TEST FAILED!"

    # Full-sample post-warmup backtest (for E1, E2, E3, E4, E5, E8)
    bt_full = run_rule_backtest(
        price_series, lookback=SELECTED_LOOKBACK, cost_bps=BASE_COST_BPS,
        risk_free_annual_pct=RF_ANNUAL_PCT,
    )
    e1 = summarize_strategy(bt_full, risk_free_annual_pct=RF_ANNUAL_PCT)
    e2 = paired_block_bootstrap_sharpe_diff(
        bt_full["net_return"], bt_full["benchmark_return"],
        risk_free_annual_pct=RF_ANNUAL_PCT, block_size=20, n_boot=2000, seed=42,
    )
    e3, breakeven_bps = cost_sensitivity_and_breakeven(
        price_series, lookback=SELECTED_LOOKBACK, cost_grid_bps=[0.0, 10.0, 25.0, 50.0],
        risk_free_annual_pct=RF_ANNUAL_PCT,
    )
    e4 = subperiod_and_startdate_check(
        price_series, lookback=SELECTED_LOOKBACK, cost_bps=BASE_COST_BPS,
        risk_free_annual_pct=RF_ANNUAL_PCT,
    )
    e5 = parameter_neighbors_check(
        price_series, center_lookback=SELECTED_LOOKBACK,
        neighbor_grid=[60, 80, 100, 120, 140],
        cost_bps=BASE_COST_BPS, risk_free_annual_pct=RF_ANNUAL_PCT,
    )
    e8 = downside_risk_summary(bt_full)

    # E6: Chronological Train / Validation / Test split (Test opened ONCE)
    split_rows = []
    trial_rows = []
    # Train trials: 5 lookbacks looked at during development
    for i, L in enumerate([60, 80, 100, 120, 140], start=1):
        bt_tr = run_rule_backtest(
            price_series, lookback=L, cost_bps=BASE_COST_BPS,
            risk_free_annual_pct=RF_ANNUAL_PCT, eval_start=TRAIN_START, eval_end=TRAIN_END,
        )
        sr, _ = sharpe_and_se(bt_tr["net_return"], RF_ANNUAL_PCT)
        ny = len(bt_tr) / 252.0
        cagr = float(((1 + bt_tr["net_return"]).prod() ** (1 / ny) - 1) * 100)
        w = np.concatenate([[1.0], np.cumprod(1 + bt_tr["net_return"].to_numpy())])
        mdd = float((w / np.maximum.accumulate(w) - 1).min() * 100)
        trial_rows.append({
            "date": "2026-11-14",
            "config_id": f"T{i:02d}",
            "rule_version": "sma_trend_v1",
            "param": f"L={L}d,c=10bps",
            "window_start": str(bt_tr.index.min().date()),
            "window_end": str(bt_tr.index.max().date()),
            "sharpe": round(sr, 4),
            "cagr": round(cagr, 2),
            "mdd": round(mdd, 2),
            "sample": "train",
            "comment": "Selected parameter" if L == SELECTED_LOOKBACK else "Neighbour parameter check on train",
        })

    # Validation & Test (opened once for selected L=100)
    for split_name, s, e, cid, comment in [
        ("train", TRAIN_START, TRAIN_END, None, None),
        ("validation", VAL_START, VAL_END, "V01", "Frozen L=100 validation check"),
        ("test", TEST_START, TEST_END, "TEST01", "Final holdout test opened ONCE on 2026-11-21"),
    ]:
        bt_s = run_rule_backtest(
            price_series, lookback=SELECTED_LOOKBACK, cost_bps=BASE_COST_BPS,
            risk_free_annual_pct=RF_ANNUAL_PCT, eval_start=s, eval_end=e,
        )
        r_sr, r_se = sharpe_and_se(bt_s["net_return"], RF_ANNUAL_PCT)
        b_sr, _ = sharpe_and_se(bt_s["benchmark_return"], RF_ANNUAL_PCT)
        ny = len(bt_s) / 252.0
        r_cagr = float(((1 + bt_s["net_return"]).prod() ** (1 / ny) - 1) * 100)
        b_cagr = float(((1 + bt_s["benchmark_return"]).prod() ** (1 / ny) - 1) * 100)
        rw = np.concatenate([[1.0], np.cumprod(1 + bt_s["net_return"].to_numpy())])
        bw = np.concatenate([[1.0], np.cumprod(1 + bt_s["benchmark_return"].to_numpy())])
        r_mdd = float((rw / np.maximum.accumulate(rw) - 1).min() * 100)
        b_mdd = float((bw / np.maximum.accumulate(bw) - 1).min() * 100)
        split_rows.append({
            "Split": split_name.capitalize(),
            "Window": f"{bt_s.index.min().date()} to {bt_s.index.max().date()}",
            "Obs (days)": len(bt_s),
            "Rule CAGR (%)": round(r_cagr, 2),
            "Bench CAGR (%)": round(b_cagr, 2),
            "Rule Sharpe": round(r_sr, 3),
            "Rule Sharpe SE": round(r_se, 3),
            "Bench Sharpe": round(b_sr, 3),
            "Rule MaxDD (%)": round(r_mdd, 2),
            "Bench MaxDD (%)": round(b_mdd, 2),
        })
        if cid is not None:
            trial_rows.append({
                "date": "2026-11-21",
                "config_id": cid,
                "rule_version": "sma_trend_v1",
                "param": f"L={SELECTED_LOOKBACK}d,c=10bps",
                "window_start": str(bt_s.index.min().date()),
                "window_end": str(bt_s.index.max().date()),
                "sharpe": round(r_sr, 4),
                "cagr": round(r_cagr, 2),
                "mdd": round(r_mdd, 2),
                "sample": split_name,
                "comment": comment,
            })

    e6 = pd.DataFrame(split_rows)
    trials_df = upsert_trials(ROOT / "trials.csv", pd.DataFrame(trial_rows))
    trials_df.to_csv(ROOT / "trials.csv", index=False)

    # E7: Trial ledger summary
    in_sample = trials_df[trials_df["sample"] == "train"]
    n_configs = len(in_sample)
    train_years = len(
        run_rule_backtest(price_series, lookback=SELECTED_LOOKBACK, eval_start=TRAIN_START, eval_end=TRAIN_END)
    ) / 252.0
    e7 = {
        "total_configurations_tried": int(n_configs),
        "test_period_openings": 1,
        "best_in_sample_sharpe": round(float(in_sample["sharpe"].max()), 3),
        "median_in_sample_sharpe": round(float(in_sample["sharpe"].median()), 3),
        "expected_max_noise_sharpe": round(expected_max_noise_sharpe(n_configs, train_years), 3),
    }

    # Save CSV evidence tables
    e1.to_csv(EVIDENCE_DIR / "e1_headline.csv", index=False)
    e3.to_csv(EVIDENCE_DIR / "e3_costs.csv", index=False)
    e4.to_csv(EVIDENCE_DIR / "e4_subperiods.csv", index=False)
    e5.to_csv(EVIDENCE_DIR / "e5_neighbors.csv", index=False)
    e6.to_csv(EVIDENCE_DIR / "e6_splits.csv", index=False)
    e8.to_csv(EVIDENCE_DIR / "e8_downside.csv", index=False)

    # Cluster Analysis evidence (KBTU Lecture)
    feats = build_risk_features(prices, benchmark="SPY")
    z_feats, _, _ = standardize_features(feats)
    k_eval = evaluate_k_with_null_band(z_feats.to_numpy(), k_values=[2, 3, 4], n_shuffles=200, nstart=50, seed=42)
    k_labels, _, _ = kmeans_multistart(z_feats.to_numpy(), k=3, nstart=100, seed=42)
    feats_out = feats.copy()
    feats_out["kmeans_cluster"] = k_labels + 1
    feats_out.to_csv(EVIDENCE_DIR / "cluster_risk_features.csv")
    k_eval.to_csv(EVIDENCE_DIR / "cluster_k_eval.csv", index=False)

    hier = hierarchical_residual_clustering(prices, benchmark="SPY", k=3)
    hier["clusters"].to_csv(EVIDENCE_DIR / "cluster_ward_groups.csv", index=False)

    # Time stability ARI: 2023-2024 vs 2025-2026
    f_early = build_risk_features(prices, start_date="2023-01-03", end_date="2024-12-31")
    f_late = build_risk_features(prices, start_date="2025-01-01", end_date="2026-10-08")
    common_t = sorted(set(f_early.index) & set(f_late.index))
    z_e, _, _ = standardize_features(f_early.loc[common_t])
    z_l, _, _ = standardize_features(f_late.loc[common_t])
    lab_e, _, _ = kmeans_multistart(z_e.to_numpy(), k=3, nstart=100, seed=42)
    lab_l, _, _ = kmeans_multistart(z_l.to_numpy(), k=3, nstart=100, seed=42)
    ari_time = round(adjusted_rand_index(lab_e, lab_l), 3)

    oos_port, oos_info = oos_cluster_portfolio_comparison(prices, benchmark="SPY", split_date="2025-01-01", k=3)
    oos_port.to_csv(EVIDENCE_DIR / "cluster_oos_portfolio.csv", index=False)

    # Save master JSON summary
    master = {
        "rule_name": f"SMA-{SELECTED_LOOKBACK} Trend Filter on Equal-Weight AI Cycle Basket (AMD, MSFT, NVDA)",
        "cost_bps": BASE_COST_BPS,
        "risk_free_annual_pct": RF_ANNUAL_PCT,
        "e2_bootstrap": e2,
        "e3_breakeven_bps": breakeven_bps,
        "e7_trials": e7,
        "cluster_summary": {
            "tickers": hier["tickers"],
            "mean_raw_corr": hier["mean_raw_corr"],
            "mean_residual_corr": hier["mean_residual_corr"],
            "within_cluster_corr": hier["within_cluster_corr"],
            "across_cluster_corr": hier["across_cluster_corr"],
            "effective_independent_bets": hier["effective_independent_bets"],
            "ari_2023_2024_vs_2025_2026": ari_time,
        },
    }
    (EVIDENCE_DIR / "evidence_summary.json").write_text(json.dumps(master, indent=2), encoding="utf-8")

    # Figures
    # 1. Equity curve
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(bt_full.index, bt_full["rule_wealth"] * 100, label=f"SMA-{SELECTED_LOOKBACK} Rule (Net, 10 bps)", color="#1D4ED8", lw=2)
    ax.plot(bt_full.index, bt_full["benchmark_wealth"] * 100, label="Buy-and-Hold Cycle Basket", color="#64748B", lw=1.8, ls="--")
    ax.set_title(f"Growth of $100: SMA-{SELECTED_LOOKBACK} Rule vs. Buy-and-Hold AI Cycle Basket")
    ax.set_ylabel("Wealth ($)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / "fig_equity_curve.png", dpi=150)
    plt.close(fig)

    # 2. Parameter neighbours
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = ["#1D4ED8" if r == "Selected" else "#94A3B8" for r in e5["Role"]]
    ax.bar(e5["Lookback (days)"].astype(str), e5["Rule Sharpe"], color=colors)
    ax.axhline(e1.loc[1, "Sharpe"], color="#B91C1C", ls="--", lw=1.5, label=f"Benchmark Sharpe ({e1.loc[1, 'Sharpe']:.2f})")
    ax.set_title("Parameter Neighbours (E5): Net Sharpe Ratio Across Lookbacks")
    ax.set_xlabel("SMA Lookback Window (Trading Days)")
    ax.set_ylabel("Annualized Net Sharpe")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / "fig_neighbors.png", dpi=150)
    plt.close(fig)

    # 3. Ward dendrogram
    fig, ax = plt.subplots(figsize=(8, 4))
    dendrogram(hier["linkage_matrix"], labels=hier["tickers"], ax=ax, color_threshold=0)
    ax.set_title("Ward Hierarchical Clustering on Market-Removed (SPY Residual) Returns")
    ax.set_ylabel("Mantegna Correlation Distance")
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / "fig_dendrogram.png", dpi=150)
    plt.close(fig)

    # 4. Sorted correlation heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(hier["raw_corr"].to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(hier["ordered_tickers"])))
    ax.set_yticks(range(len(hier["ordered_tickers"])))
    ax.set_xticklabels(hier["ordered_tickers"], rotation=45, ha="right")
    ax.set_yticklabels(hier["ordered_tickers"])
    ax.set_title(f"Cluster-Sorted Correlation Matrix (Eff. Bets: {hier['effective_independent_bets']})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / "fig_cluster_heatmap.png", dpi=150)
    plt.close(fig)

    print("=== EVIDENCE PACK GENERATED ===")
    print("\n--- E1 HEADLINE ---")
    print(e1.to_string(index=False))
    print("\n--- E2 BOOTSTRAP ---")
    print(e2)
    print("\n--- E3 COSTS (breakeven bps:", breakeven_bps, ") ---")
    print(e3.to_string(index=False))
    print("\n--- E4 SUBPERIODS ---")
    print(e4.to_string(index=False))
    print("\n--- E5 NEIGHBOURS ---")
    print(e5.to_string(index=False))
    print("\n--- E6 SPLITS ---")
    print(e6.to_string(index=False))
    print("\n--- E7 TRIALS ---")
    print(e7)
    print("\n--- E8 DOWNSIDE ---")
    print(e8.to_string(index=False))
    print("\n--- CLUSTERING ---")
    print(k_eval.to_string(index=False))
    print("Cluster summary:", master["cluster_summary"])
    print("\n--- OOS CLUSTER PORTFOLIO ---")
    print(oos_port.to_string(index=False))


if __name__ == "__main__":
    main()
