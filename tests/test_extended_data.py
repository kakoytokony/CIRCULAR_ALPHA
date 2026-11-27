"""Data-integrity tests for the v3 extended datasets and the research pack.

Covers: schema & completeness, date integrity, value sanity, cross-file
consistency, and look-ahead canaries for the bubble/trigger indicators.
Missing files fail loudly here (the app itself degrades gracefully instead).
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data import BASE_DIR, load_all_prices, load_companies, load_relationships
from src.extended_data import CORE_TICKERS, DOTCOM_TICKERS, SPILLOVER_TICKERS, equal_weight_index, wide

RAW = BASE_DIR / "data" / "raw"
PACK = BASE_DIR / "data" / "circular_alpha_research_pack.xlsx"
PANEL_FILES = {
    "volumes_core.csv": CORE_TICKERS + ["SPY"],
    "spillover_tiers.csv": SPILLOVER_TICKERS,
    "dotcom_1998_2003.csv": DOTCOM_TICKERS,
}
PANEL_COLS = {"date", "adjusted_close", "volume", "ticker", "source", "retrieved_date"}
FRED_COLS = {"date", "hy_oas_pct", "ig_oas_pct", "ust_10y_pct", "source", "retrieved_date"}
MAX_MISSING = 0.10       # after a series' first valid observation


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / name)


class TestSchemaAndCompleteness(unittest.TestCase):
    def test_files_exist_and_nonempty(self):
        for name in [*PANEL_FILES, "fred_credit_spreads.csv"]:
            p = RAW / name
            self.assertTrue(p.exists(), f"{name} missing: run python scripts/fetch_extended_data.py")
            self.assertGreater(len(pd.read_csv(p)), 100, name)
        self.assertTrue(PACK.exists())

    def test_panel_columns_and_tickers(self):
        for name, tickers in PANEL_FILES.items():
            df = _read(name)
            self.assertTrue(PANEL_COLS.issubset(df.columns), name)
            self.assertEqual(set(df["ticker"]), set(tickers), name)

    def test_fred_columns(self):
        self.assertTrue(FRED_COLS.issubset(_read("fred_credit_spreads.csv").columns))

    def test_research_pack_sheets_and_headers(self):
        import openpyxl
        wb = openpyxl.load_workbook(PACK, read_only=True)
        expected = {"Dashboard_BondMatrix", "Credit_BlueWave_Deals", "DotCom_2000_Scenarios",
                    "Repo_Companies", "Repo_Financials", "Repo_Relationships"}
        self.assertTrue(expected.issubset(set(wb.sheetnames)))
        wb.close()
        from src.credit_signals import load_research_pack
        pack = load_research_pack()
        self.assertEqual(len(pack.bond_matrix), 6)
        self.assertEqual(len(pack.tenor_curve), 5)
        self.assertEqual(len(pack.blue_wave_states), 5)
        self.assertGreaterEqual(len(pack.credit_segments), 6)
        self.assertGreaterEqual(len(pack.landmark_deals), 6)
        self.assertEqual(len(pack.literature), 8)


class TestDateIntegrity(unittest.TestCase):
    def test_panels_sorted_unique_not_future(self):
        today = pd.Timestamp.today().normalize()
        for name in PANEL_FILES:
            df = _read(name)
            d = pd.to_datetime(df["date"], errors="coerce")
            self.assertFalse(d.isna().any(), name)
            self.assertFalse(df.duplicated(["date", "ticker"]).any(), name)
            for t, g in df.assign(d=d).groupby("ticker"):
                self.assertTrue(g["d"].is_monotonic_increasing and g["d"].is_unique, f"{name}:{t}")
            self.assertLessEqual(d.max(), today, name)

    def test_fred_sorted_unique_not_future(self):
        d = pd.to_datetime(_read("fred_credit_spreads.csv")["date"])
        self.assertTrue(d.is_monotonic_increasing and d.is_unique)
        self.assertLessEqual(d.max(), pd.Timestamp.today().normalize())

    def test_dotcom_window(self):
        d = pd.to_datetime(_read("dotcom_1998_2003.csv")["date"])
        self.assertGreaterEqual(d.min(), pd.Timestamp("1998-01-01"))
        self.assertLessEqual(d.max(), pd.Timestamp("2003-12-31"))


class TestValueSanity(unittest.TestCase):
    def test_prices_positive_volume_nonnegative(self):
        for name in PANEL_FILES:
            df = _read(name)
            self.assertTrue((df["adjusted_close"] > 0).all(), name)
            self.assertTrue((df["volume"].dropna() >= 0).all(), name)
            self.assertLess(df["adjusted_close"].isna().mean(), MAX_MISSING, name)

    def test_fred_bounds_and_gaps(self):
        df = _read("fred_credit_spreads.csv")
        for c in ["hy_oas_pct", "ig_oas_pct"]:
            s = df[c].dropna()
            self.assertTrue(((s > 0) & (s < 30)).all(), c)
            first = df[c].first_valid_index()
            miss = df.loc[first:, c].isna().mean()
            # FRED rows are the union of three calendars, so holidays leave small gaps.
            self.assertLess(miss, MAX_MISSING, f"{c}: {miss:.1%} missing after first observation")
        u = df["ust_10y_pct"].dropna()
        self.assertTrue(((u > 0) & (u < 20)).all())

    def test_no_extreme_one_day_jumps_in_indices(self):
        df = _read("spillover_tiers.csv")
        for t in ["QQQ", "LQD", "HYG", "EMB", "XLU"]:
            r = df[df["ticker"] == t]["adjusted_close"].pct_change().abs()
            self.assertLess(r.max(), 0.25, t)


class TestCrossFileConsistency(unittest.TestCase):
    def test_model_tickers_present_in_downloads(self):
        from src.macro_contagion import AI_IN_SPX, TIERS_EXTENDED
        have = set(_read("volumes_core.csv")["ticker"]) | set(_read("spillover_tiers.csv")["ticker"])
        for tier, members in TIERS_EXTENDED.items():
            self.assertTrue(set(members).issubset(have), tier)
        self.assertTrue(set(AI_IN_SPX).issubset(have))

    def test_core_downloads_match_company_master(self):
        comp = load_companies()
        public = set(comp["public_ticker"].dropna().astype(str))
        self.assertEqual(public, set(CORE_TICKERS))

    def test_relationship_entities_map_to_companies(self):
        ids = set(load_companies()["company_id"].astype(str))
        rel = load_relationships()
        self.assertTrue(set(rel["source_entity_id"].astype(str)).issubset(ids))
        self.assertTrue(set(rel["target_entity_id"].astype(str)).issubset(ids))

    def test_extended_files_do_not_leak_into_legacy_price_panel(self):
        prices = load_all_prices()
        self.assertEqual(set(prices["ticker"]), set(CORE_TICKERS) | {"SPY"})
        legacy = pd.read_csv(RAW / "SPY.csv")
        self.assertEqual(int((prices["ticker"] == "SPY").sum()), len(legacy))

    def test_core_prices_close_to_legacy_snapshot(self):
        """Same Yahoo source: closes agree within dividend re-basing (< 5%)."""
        v = wide(pd.read_csv(RAW / "volumes_core.csv").assign(date=lambda d: pd.to_datetime(d["date"])))
        for t in ["NVDA", "MSFT", "SPY"]:
            leg = pd.read_csv(RAW / f"{t}.csv", parse_dates=["date"]).set_index("date")["adjusted_close"]
            m = pd.concat([v[t], leg], axis=1, join="inner").dropna()
            self.assertLess((m.iloc[:, 0] / m.iloc[:, 1] - 1).abs().max(), 0.05, t)


class TestLookAheadCanary(unittest.TestCase):
    """Indicators at date t must not change when data after t is modified."""

    def setUp(self):
        from src.extended_data import load_long_panel
        v, st = load_long_panel("volumes_core")
        self.assertTrue(st.ok, st.message)
        self.pw = wide(v)
        self.dv = wide(v.assign(dv=v["adjusted_close"] * v["volume"]), "dv")
        self.ai = equal_weight_index(self.pw, CORE_TICKERS)
        self.t0 = len(self.ai) - 100

    def test_gsadf_bsadf(self):
        from src.bubble_monitor import bsadf_series, psy_min_window
        y = np.log(self.ai.to_numpy())
        y2 = y.copy(); y2[self.t0 + 1:] += np.linspace(0, 2, len(y) - self.t0 - 1)
        w0 = psy_min_window(len(y))
        np.testing.assert_array_equal(bsadf_series(y, w0)[: self.t0 + 1], bsadf_series(y2, w0)[: self.t0 + 1])

    def test_gsy_probability(self):
        from src.bubble_monitor import CrashCalibration, crash_risk_table, gsy_features
        from src.extended_data import fred_series_bps, load_fred
        fred, _ = load_fred()
        hy = fred_series_bps(fred)
        cal = CrashCalibration(0.5, 0.8, 1, 0.5, "", "", [], "")
        a = crash_risk_table(gsy_features(self.ai, self.dv.sum(axis=1), hy, self.pw["SPY"]), cal)
        ai2 = self.ai.copy(); ai2.iloc[self.t0 + 1:] *= 2.0
        hy2 = hy.copy(); hy2[hy2.index > self.ai.index[self.t0]] += 400
        b = crash_risk_table(gsy_features(ai2, self.dv.sum(axis=1), hy2, self.pw["SPY"]), cal)
        pd.testing.assert_series_equal(a["p_crash"].iloc[: self.t0 + 1], b["p_crash"].iloc[: self.t0 + 1])

    def test_credit_triggers_independent_of_future_prices(self):
        from src.bubble_monitor import run_exuberance_test, simulate_bsadf_critical_values, trigger_checklist
        from src.credit_signals import current_credit_inputs, supply_friction_flags
        n = self.t0 + 1
        cv = simulate_bsadf_critical_values(len(self.ai), n_sims=20, seed=1)
        a = run_exuberance_test(self.ai, cv=cv).table
        ai2 = self.ai.copy(); ai2.iloc[n:] *= 3.0
        b = run_exuberance_test(ai2, cv=cv).table
        cr = current_credit_inputs()
        ta = trigger_checklist(cr, a["bsadf"].iloc[self.t0], a["cv95"].iloc[self.t0], supply_friction_flags())
        tb = trigger_checklist(cr, b["bsadf"].iloc[self.t0], b["cv95"].iloc[self.t0], supply_friction_flags())
        pd.testing.assert_frame_equal(ta, tb)


if __name__ == "__main__":
    unittest.main()
