"""Comprehensive GA2 (The Rule Desk), Cluster Lens, Contagion & News Agent tests.

Runs in one command with EITHER:
  python3 -m unittest discover -s tests -v
  pytest -v
"""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from data import load_all_prices, load_companies, load_financials, load_relationships
from src.backtest import (
    compute_sma_signal_and_position,
    cost_sensitivity_and_breakeven,
    expected_max_noise_sharpe,
    paired_block_bootstrap_sharpe_diff,
    run_canary_lookahead_test,
    run_rule_backtest,
    sharpe_and_se,
    summarize_strategy,
)
from src.clustering import (
    adjusted_rand_index,
    average_silhouette,
    build_risk_features,
    hierarchical_residual_clustering,
    kmeans_multistart,
    standardize_features,
)
from src.contagion import run_contagion_stress_test
from src.news_agent import (
    classify_source_credibility,
    get_company_sentiment_map,
    load_news_feed,
    score_financial_sentiment,
)


class TestGA2BacktestCore(unittest.TestCase):
    def test_hand_computed_toy_backtest_lagged_position_cash_and_costs(self):
        """Hand-computed toy test (GA2 Criterion B2):
        Prices (lookback=2):
          Day 0: 100 -> SMA=NaN, sig=NaN, pos=NaN
          Day 1: 110 -> SMA=105, sig=1 (110 > 105), pos=NaN
          Day 2: 121 -> ret=+10%, SMA=115.5, sig=1, pos=sig[1]=1 -> gross=+10%, pos_change=0 (first eval day) -> net=+10%
          Day 3: 108.9 -> ret=-10%, SMA=114.95, sig=0 (108.9 < 114.95), pos=sig[2]=1 -> gross=-10%, pos_change=0 -> net=-10%
          Day 4: 98.01 -> ret=-10%, SMA=103.455, sig=0, pos=sig[3]=0 (CASH!)
            With rf=25.2%/yr -> daily_rf = 0.001 (0.1%).
            pos_change = |0 - 1| = 1. With cost=50 bps (0.005):
            gross = 0.001, net = 0.001 - 0.005 = -0.004 (-0.4%).
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        prices = pd.Series([100.0, 110.0, 121.0, 108.9, 98.01], index=dates)
        bt = run_rule_backtest(prices, lookback=2, cost_bps=50.0, risk_free_annual_pct=25.2)

        self.assertEqual(len(bt), 3)  # Days 2, 3, 4
        self.assertEqual(bt["position"].tolist(), [1.0, 1.0, 0.0])
        self.assertAlmostEqual(bt["gross_return"].iloc[0], 0.10, places=6)
        self.assertAlmostEqual(bt["gross_return"].iloc[1], -0.10, places=6)
        self.assertAlmostEqual(bt["gross_return"].iloc[2], 0.001, places=6)
        self.assertAlmostEqual(bt["net_return"].iloc[2], -0.004, places=6)

        # Known-answer max drawdown check: rule wealth = [1.0, 1.10, 0.99, 0.98604]
        # Peak = 1.10, trough = 0.98604 -> MDD = (0.98604 / 1.10 - 1) * 100 = -10.36%
        summary = summarize_strategy(bt, risk_free_annual_pct=25.2)
        self.assertAlmostEqual(summary.loc[0, "Max Drawdown (%)"], -10.36, places=2)
        self.assertEqual(int(summary.loc[0, "Switches"]), 1)

    def test_canary_lookahead_test_passes(self):
        """Canary test (GA2 Criterion B2): perturbing future prices must NOT alter earlier positions."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        rng = np.random.default_rng(7)
        prices = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, 100)), index=dates)
        self.assertTrue(run_canary_lookahead_test(prices, lookback=10, split_idx=50))

    def test_known_answer_transaction_costs_reduce_net_return_monotonically(self):
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        # Oscillating series forces multiple switches
        vals = [100.0 + (10.0 if (i // 4) % 2 == 0 else -10.0) for i in range(40)]
        prices = pd.Series(vals, index=dates)
        cost_tbl, _ = cost_sensitivity_and_breakeven(prices, lookback=5, cost_grid_bps=[0.0, 10.0, 50.0])
        cagrs = cost_tbl["Rule Net CAGR (%)"].tolist()
        self.assertGreater(cagrs[0], cagrs[1])
        self.assertGreater(cagrs[1], cagrs[2])

    def test_bootstrap_and_noise_sharpe_formulas(self):
        rng = np.random.default_rng(12)
        r = pd.Series(rng.normal(0.001, 0.015, 252))
        b = pd.Series(rng.normal(0.001, 0.015, 252))
        res = paired_block_bootstrap_sharpe_diff(r, b, n_boot=300, seed=42)
        self.assertLessEqual(res["ci_low"], res["ci_high"])
        self.assertGreater(expected_max_noise_sharpe(5, 2.0), 0.0)
        sr, se = sharpe_and_se(r, risk_free_annual_pct=0.0)
        self.assertGreater(se, 0.0)


class TestClusterLens(unittest.TestCase):
    def test_zscore_standardization_exact(self):
        df = pd.DataFrame({"vol": [16.5, 16.7, 23.0, 46.7, 57.8], "beta": [0.0, -0.1, 0.9, 2.1, 2.3]})
        z, mu, sd = standardize_features(df)
        self.assertAlmostEqual(float(z["vol"].mean()), 0.0, places=10)
        self.assertAlmostEqual(float(z["vol"].std(ddof=1)), 1.0, places=10)
        # Matches KBTU lecture Slide 7: Tesla z = (57.8 - 32.14) / 18.95 = 1.354
        self.assertAlmostEqual(float(z["vol"].iloc[-1]), 1.354, places=2)

    def test_kmeans_and_ari_and_ward_residual_clustering(self):
        # Two cleanly separated clusters in 2D
        X = np.array([[-5.0, -5.0], [-4.8, -5.1], [-5.2, -4.9], [5.0, 5.0], [5.1, 4.9], [4.9, 5.2]])
        labels, _, _ = kmeans_multistart(X, k=2, nstart=50, seed=42)
        self.assertAlmostEqual(adjusted_rand_index(labels, np.array([0, 0, 0, 1, 1, 1])), 1.0, places=6)
        self.assertGreater(average_silhouette(X, labels), 0.9)

        prices = load_all_prices()
        hier = hierarchical_residual_clustering(prices, benchmark="SPY", k=3)
        self.assertLess(hier["mean_residual_corr"], hier["mean_raw_corr"])
        self.assertGreater(hier["effective_independent_bets"], 1.0)


class TestContagionAndNewsAgent(unittest.TestCase):
    def test_anthropic_collapse_hits_broadcom_guarantee_and_msft_backlog(self):
        companies = load_companies()
        relationships = load_relationships()
        financials = load_financials()
        prices = load_all_prices()

        impact, summary = run_contagion_stress_test(
            companies, relationships, financials,
            failed_company_id="C010",  # Anthropic PBC
            prices=prices, shock_severity=1.0, confirmed_only=False,
        )
        by_id = impact.set_index("company_id")
        # Broadcom (C011) has $63B SPV guarantee + $63B chip backlog
        self.assertIn("C011", by_id.index)
        self.assertAlmostEqual(by_id.loc["C011", "guarantee_spv_call_usd"], 63e9, delta=1e6)
        # Microsoft (C002) has $30B Azure commitment + $5B equity write-down
        self.assertAlmostEqual(by_id.loc["C002", "revenue_backlog_at_risk_usd"], 30e9, delta=1e6)
        self.assertAlmostEqual(by_id.loc["C002", "equity_writedown_usd"], 5e9, delta=1e6)
        self.assertGreater(summary["hop1_count"], 3)

    def test_negative_news_sentiment_amplifies_contagion_score(self):
        companies = load_companies()
        relationships = load_relationships()
        financials = load_financials()
        impact_neg, _ = run_contagion_stress_test(
            companies, relationships, financials,
            failed_company_id="C010", confirmed_only=False,
            news_sentiment_by_company={"C010": -0.8, "C011": -0.8},
        )
        row = impact_neg.set_index("company_id").loc["C011"]
        self.assertGreater(row["contagion_score_news_adj"], row["contagion_score_quant"])

    def test_news_agent_credibility_and_rumor_exclusion(self):
        t1, _, inc1 = classify_source_credibility("https://www.wsj.com/tech/ai", "The Wall Street Journal", "Broadcom deal")
        t3, _, inc3 = classify_source_credibility("https://random.blog/post", "Anonymous", "Unconfirmed rumor about Nvidia")
        self.assertTrue(inc1)
        self.assertIn("Tier 1", t1)
        self.assertFalse(inc3)
        self.assertIn("Tier 3", t3)
        self.assertLess(score_financial_sentiment("Data center moratorium and distressed bond selloff"), 0.0)

        feed = load_news_feed()
        sent_clean = get_company_sentiment_map(feed, include_unverified=False)
        self.assertGreater(len(sent_clean), 0)


if __name__ == "__main__":
    unittest.main()
