"""v3 methodology tests: BSADF/GSADF, GSY crash score, triggers, DebtRank,
Diebold–Yilmaz, the S&P 500 scenario identity, and source-constant integrity.

Runs with either ``pytest -v`` or ``python -m unittest discover -s tests -v``.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from data import load_companies, load_financials, load_relationships
from src.bubble_monitor import (
    adf_stat,
    bsadf_series,
    calibrate_on_dotcom,
    composite_score,
    crash_risk_table,
    forward_crash_label,
    gsy_features,
    psy_min_window,
    run_exuberance_test,
    simulate_bsadf_critical_values,
    switch_off_status,
    trigger_checklist,
)
from src.credit_signals import (
    HY_HPC_BASELINE_BPS,
    SNAPSHOT,
    current_credit_inputs,
    load_research_pack,
    supply_friction_flags,
    traffic_light,
)
from src.extended_data import (
    CORE_TICKERS,
    equal_weight_index,
    equal_weight_volume,
    fred_series_bps,
    load_fred,
    load_long_panel,
    wide,
)
from src.historical_analog import DOTCOM_BENCHMARKS, measured_drawdowns, spx_nasdaq_elasticity
from src.macro_contagion import (
    AI_IN_SPX,
    SCENARIOS,
    TIERS_CORE,
    build_exposure_network,
    debtrank_propagate,
    debtrank_scores,
    diebold_yilmaz,
    downside_spill_beta,
    firm_scenario_drawdowns,
    fit_var,
    generalized_fevd,
    implied_effective_weights,
    index_stress_forecast,
    rolling_tci,
    tier_returns,
)


def _rw(n=400, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n))),
                     index=pd.bdate_range("2020-01-01", periods=n))


def _bubble(n=400, seed=2):
    """Random walk followed by a mildly explosive segment."""
    rng = np.random.default_rng(seed)
    y = np.cumsum(0.01 * rng.standard_normal(n))
    for t in range(300, n):
        y[t] = 1.03 * y[t - 1] + 0.002 + 0.002 * rng.standard_normal()
    return pd.Series(100 * np.exp(y - y.min() + 0.1), index=pd.bdate_range("2020-01-01", periods=n))


class TestBSADF(unittest.TestCase):
    def test_closed_form_matches_brute_force_adf(self):
        y = np.log(_rw(250).to_numpy())
        w0 = psy_min_window(len(y))
        b = bsadf_series(y, min_window=w0)
        t = 180
        brute = max(adf_stat(y[s:t + 1]) for s in range(0, t - w0 + 2))
        self.assertAlmostEqual(b[t], brute, places=8)

    def test_warmup_is_nan_and_rest_finite(self):
        y = np.log(_rw(200).to_numpy())
        w0 = psy_min_window(200)
        b = bsadf_series(y, min_window=w0)
        self.assertTrue(np.isnan(b[: w0 - 1]).all())
        self.assertTrue(np.isfinite(b[w0 - 1:]).all())

    def test_detects_explosive_segment(self):
        p = _bubble()
        cv = simulate_bsadf_critical_values(len(p), n_sims=60, seed=7)
        res = run_exuberance_test(p, cv=cv)
        self.assertTrue(res.table["explosive"].iloc[320:].any())
        self.assertGreater(res.gsadf, res.gsadf_cv95)

    def test_critical_values_reproducible_and_ordered(self):
        a = simulate_bsadf_critical_values(150, n_sims=40, seed=3)
        b = simulate_bsadf_critical_values(150, n_sims=40, seed=3)
        np.testing.assert_allclose(a["cv_t"][0.95], b["cv_t"][0.95], equal_nan=True)
        self.assertLess(a["cv_gsadf"][0.90], a["cv_gsadf"][0.99])

    def test_canary_bsadf_no_lookahead(self):
        p = _rw(300)
        t0 = 200
        q = p.copy()
        q.iloc[t0 + 1:] *= np.linspace(1.0, 3.0, len(q) - t0 - 1)   # perturb only the future
        w0 = psy_min_window(len(p))
        a = bsadf_series(np.log(p.to_numpy()), min_window=w0)
        b = bsadf_series(np.log(q.to_numpy()), min_window=w0)
        np.testing.assert_array_equal(a[: t0 + 1], b[: t0 + 1])


class TestCrashScore(unittest.TestCase):
    def setUp(self):
        self.p = _rw(900, seed=5)
        rng = np.random.default_rng(9)
        self.dv = pd.Series(1e9 * np.exp(rng.standard_normal(900) * 0.2), index=self.p.index)
        self.cs = pd.Series(300 + np.cumsum(rng.standard_normal(900)), index=self.p.index)

    def test_canary_features_and_score_no_lookahead(self):
        t0 = 700
        f1 = gsy_features(self.p, self.dv, self.cs)
        p2, dv2, cs2 = self.p.copy(), self.dv.copy(), self.cs.copy()
        p2.iloc[t0 + 1:] *= 5.0
        dv2.iloc[t0 + 1:] *= 10.0
        cs2.iloc[t0 + 1:] += 500.0
        f2 = gsy_features(p2, dv2, cs2)
        cols = ["z_vol", "z_turn", "z_acc", "z_cred"]
        pd.testing.assert_frame_equal(f1[cols].iloc[: t0 + 1], f2[cols].iloc[: t0 + 1])
        s1, s2 = composite_score(f1), composite_score(f2)
        pd.testing.assert_series_equal(s1.iloc[: t0 + 1], s2.iloc[: t0 + 1])

    def test_signal_is_lagged_one_day(self):
        f = gsy_features(self.p, self.dv, self.cs)
        from src.bubble_monitor import CrashCalibration
        cal = CrashCalibration(0.0, 1.0, 1, 0.5, "", "", [], "")
        t = crash_risk_table(f, cal)
        pd.testing.assert_series_equal(t["score_signal"], t["score"].shift(1), check_names=False)

    def test_probability_in_unit_interval(self):
        f = gsy_features(self.p, self.dv, self.cs)
        from src.bubble_monitor import CrashCalibration
        t = crash_risk_table(f, CrashCalibration(-1.0, 2.0, 1, 0.5, "", "", [], ""))
        p = t["p_crash"].dropna()
        self.assertTrue(((p > 0) & (p < 1)).all())

    def test_forward_label_definition(self):
        p = pd.Series([100, 90, 70, 59, 80, 100, 100.0])
        lab = forward_crash_label(p, horizon=3)
        self.assertEqual(lab.iloc[0], 1.0)     # 59 <= 60
        self.assertEqual(lab.iloc[1], 0.0)     # min 59 / 90 - 1 = -34%
        self.assertTrue(lab.iloc[4:].isna().all())

    def test_dotcom_calibration_positive_slope(self):
        panel, st = load_long_panel("dotcom")
        if not st.ok:
            self.skipTest(st.message)
        ix = panel[panel["ticker"] == "^IXIC"].set_index("date")
        cal = calibrate_on_dotcom(ix["adjusted_close"], ix["adjusted_close"] * ix["volume"])
        self.assertGreater(cal.theta1, 0.0)
        self.assertGreater(cal.n_obs, 300)
        self.assertEqual(cal.attributes_used, ["z_vol", "z_turn", "z_acc"])


class TestTriggers(unittest.TestCase):
    def test_traffic_light_thresholds(self):
        self.assertEqual(traffic_light(503, 450, 600), "AMBER")
        self.assertEqual(traffic_light(601, 450, 600), "RED")
        self.assertEqual(traffic_light(300, 450, 600), "GREEN")
        self.assertEqual(traffic_light(1000, 1000, 1000, red_inclusive=True), "RED")

    def test_spec_current_states(self):
        """Spec §4.1.3 lists HY HPC → AMBER, basis → AMBER, CRWV → RED, supply → RED."""
        cr = current_credit_inputs(load_research_pack())
        tc = trigger_checklist(cr, bsadf_now=0.0, cv95_now=1.0, supply_flags=supply_friction_flags())
        st = dict(zip(tc["key"], tc["status"]))
        self.assertEqual(st["hy_hpc"], "AMBER")
        self.assertEqual(st["basis"], "AMBER")
        self.assertEqual(st["crwv"], "RED")
        self.assertEqual(st["supply"], "RED")
        self.assertEqual(st["gsadf"], "GREEN")

    def test_canary_trigger_uses_only_past_bsadf(self):
        p = _rw(300)
        cv = simulate_bsadf_critical_values(300, n_sims=30, seed=11)
        q = p.copy(); q.iloc[201:] *= 4.0
        a = run_exuberance_test(p, cv=cv).table
        b = run_exuberance_test(q, cv=cv).table
        pd.testing.assert_series_equal(a["signal"].iloc[:202], b["signal"].iloc[:202])

    def test_switch_off_requires_both(self):
        self.assertFalse(switch_off_status(503, -1.0, 0.3)["switch_off"])
        self.assertFalse(switch_off_status(300, 2.0, 0.3)["switch_off"])
        self.assertTrue(switch_off_status(300, -1.0, 0.3)["switch_off"])


class TestDebtRank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = build_exposure_network(load_companies(), load_relationships(), load_financials())

    def test_bounds(self):
        for k in range(len(self.net.ids)):
            h0 = np.zeros(len(self.net.ids)); h0[k] = 1.0
            h = debtrank_propagate(self.net.W, h0)
            self.assertTrue(((h >= 0) & (h <= 1)).all())
        self.assertTrue(((self.net.W >= 0) & (self.net.W <= 1)).all())

    def test_monotonic_in_shock(self):
        k = self.net.ids.index("C010")
        prev = None
        for s in [0.1, 0.3, 0.6, 1.0]:
            h0 = np.zeros(len(self.net.ids)); h0[k] = s
            h = debtrank_propagate(self.net.W, h0)
            if prev is not None:
                self.assertTrue((h >= prev - 1e-12).all())
            prev = h

    def test_superseded_loi_excluded_and_undisclosed_not_imputed(self):
        used = set(self.net.used_links["relationship_id"])
        self.assertFalse({"R001", "R002"} & used)
        rel = load_relationships()
        undisclosed = rel[rel["amount"].isna() & ~rel["relationship_id"].isin(["R001", "R002"])]
        self.assertEqual(self.net.undisclosed_links, len(undisclosed))

    def test_known_exposure_broadcom_guarantee(self):
        """R031: $63B Broadcom guarantee vs $34.259B EBITDA → W = 1 (capped)."""
        i, j = self.net.ids.index("C010"), self.net.ids.index("C011")
        self.assertEqual(self.net.W[i, j], 1.0)

    def test_scores_nonnegative(self):
        s = debtrank_scores(self.net)
        self.assertTrue((s["debtrank"] >= 0).all())
        self.assertEqual(len(s), len(self.net.ids))


class TestSpilloverAndScenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2022-01-03", periods=400)
        common = rng.standard_normal(400)
        cls.R = pd.DataFrame({f"A{k}": 0.01 * (0.6 * common + rng.standard_normal(400)) for k in range(4)}, index=idx)

    def test_gfevd_rows_sum_to_one(self):
        Phi, sig = fit_var(self.R.to_numpy(), 2)
        th = generalized_fevd(Phi, sig, 10)
        np.testing.assert_allclose(th.sum(axis=1), 1.0)
        self.assertTrue((th >= 0).all())

    def test_tci_range_and_net_sums_zero(self):
        d = diebold_yilmaz(self.R)
        self.assertTrue(0 < d["tci"] < 100)
        self.assertAlmostEqual(d["summary"]["net"].sum(), 0.0, places=8)

    def test_canary_rolling_tci(self):
        t0 = 330
        a = rolling_tci(self.R, window=252, step=5)
        R2 = self.R.copy(); R2.iloc[t0 + 1:] *= 7.0
        b = rolling_tci(R2, window=252, step=5)
        cut = a.index <= self.R.index[t0]
        np.testing.assert_allclose(a[cut].to_numpy(), b[cut].to_numpy())

    def test_scenario_conservation_identity(self):
        net = build_exposure_network(load_companies(), load_relationships(), load_financials())
        w = pd.Series({"NVDA": 0.08, "MSFT": 0.06, "AVGO": 0.03, "AMZN": 0.04})
        for sc in SCENARIOS.values():
            fd = firm_scenario_drawdowns(net, sc)
            self.assertFalse(fd["scenario_dd_pct"].isna().any())
            self.assertTrue((fd["scenario_dd_pct"] >= -100).all())
            out = index_stress_forecast(fd, w, beta_spill=0.3)
            dd = fd.set_index("ticker")["scenario_dd_pct"].reindex(w.index)
            direct = float((w * dd).sum())
            spill = 0.3 * (1 - w.sum()) * direct / w.sum()
            self.assertAlmostEqual(out["direct_pct"], direct, places=10)
            self.assertAlmostEqual(out["total_pct"], direct + spill, places=10)
            self.assertFalse(any(np.isnan(v) for v in out.values()))

    def test_scenarios_are_ordered_on_real_data(self):
        v, st = load_long_panel("volumes_core")
        if not st.ok:
            self.skipTest(st.message)
        lr = np.log(wide(v)).diff()
        w = implied_effective_weights(lr, "SPY", AI_IN_SPX)
        b = downside_spill_beta(lr, "SPY", w)
        self.assertTrue(0.0 <= b["beta"] <= 0.63)
        self.assertTrue((w >= 0).all())
        net = build_exposure_network(load_companies(), load_relationships(), load_financials())
        tot = [index_stress_forecast(firm_scenario_drawdowns(net, SCENARIOS[k]), w, b["beta"])["total_pct"] for k in ("S1", "S2", "S3")]
        self.assertTrue(tot[0] > tot[1] > tot[2])


class TestSourcedConstants(unittest.TestCase):
    def test_every_constant_has_source(self):
        for d in (SNAPSHOT, DOTCOM_BENCHMARKS):
            for v in d.values():
                self.assertTrue(v.source.strip(), v.key)

    def test_snapshot_matches_workbook(self):
        pack = load_research_pack()
        self.assertAlmostEqual(pack.bond_matrix["total_bn"].sum(), SNAPSHOT["total_tracked_debt_bn"].value)
        self.assertAlmostEqual(pack.bond_matrix["direct_ig_bn"].sum(), 495)
        self.assertAlmostEqual(pack.bond_matrix["dc_ig_bn"].sum(), 70)
        self.assertAlmostEqual(pack.bond_matrix["dc_hy_bn"].sum(), 64)
        self.assertAlmostEqual(pack.tenor_curve["premium_bps"].iloc[-1], 61.6)
        seg = pack.credit_segments.set_index("segment")
        self.assertEqual(seg.loc["J.P. Morgan HY HPC Index", "level_bps"], SNAPSHOT["hy_hpc_spread_bps"].value)
        self.assertEqual(seg.loc["Leveraged Loans HPC (3y)", "level_bps"], SNAPSHOT["lev_loans_hpc_bps"].value)
        self.assertEqual(HY_HPC_BASELINE_BPS, 259.0)

    def test_dotcom_constants_match_downloaded_data(self):
        panel, st = load_long_panel("dotcom")
        if not st.ok:
            self.skipTest(st.message)
        m = measured_drawdowns(panel).set_index("ticker")
        self.assertAlmostEqual(m.loc["^IXIC", "peak"], DOTCOM_BENCHMARKS["ixic_peak"].value, places=1)
        self.assertAlmostEqual(m.loc["^IXIC", "trough"], DOTCOM_BENCHMARKS["ixic_trough"].value, places=1)
        self.assertAlmostEqual(m.loc["^GSPC", "peak"], DOTCOM_BENCHMARKS["spx_peak"].value, places=1)
        self.assertAlmostEqual(m.loc["^GSPC", "trough"], DOTCOM_BENCHMARKS["spx_trough"].value, places=1)
        self.assertAlmostEqual(round(m.loc["^IXIC", "drawdown_pct"], 1), DOTCOM_BENCHMARKS["ixic_dd_pct"].value)
        self.assertAlmostEqual(round(m.loc["^GSPC", "drawdown_pct"], 1), DOTCOM_BENCHMARKS["spx_dd_pct"].value)
        self.assertAlmostEqual(round(spx_nasdaq_elasticity(measured_drawdowns(panel)), 2), 0.63)


class TestRealDataPipelineCanary(unittest.TestCase):
    """Canary on the actual AI-core index built from data/raw/volumes_core.csv."""

    def test_real_ai_core_canary(self):
        v, st = load_long_panel("volumes_core")
        fred, fst = load_fred()
        if not st.ok:
            self.skipTest(st.message)
        pw = wide(v)
        dv = wide(v.assign(dv=v["adjusted_close"] * v["volume"]), "dv")
        ai = equal_weight_index(pw, CORE_TICKERS)
        aidv = equal_weight_volume(dv, CORE_TICKERS).reindex(ai.index)
        hy = fred_series_bps(fred) if fst.ok else None
        t0 = len(ai) - 60
        f1 = gsy_features(ai, aidv, hy)
        ai2 = ai.copy(); ai2.iloc[t0 + 1:] *= 0.3
        f2 = gsy_features(ai2, aidv, hy)
        s1, s2 = composite_score(f1), composite_score(f2)
        pd.testing.assert_series_equal(s1.iloc[: t0 + 1], s2.iloc[: t0 + 1])
        w0 = psy_min_window(len(ai))
        b1 = bsadf_series(np.log(ai.to_numpy()), w0)
        b2 = bsadf_series(np.log(ai2.to_numpy()), w0)
        np.testing.assert_array_equal(b1[: t0 + 1], b2[: t0 + 1])

    def test_tier_returns_built_for_all_core_tiers(self):
        v, st = load_long_panel("volumes_core")
        if not st.ok:
            self.skipTest(st.message)
        tr = tier_returns(wide(v), TIERS_CORE)
        self.assertEqual(list(tr.columns), list(TIERS_CORE))
        self.assertFalse(tr.isna().any().any())


if __name__ == "__main__":
    unittest.main()
