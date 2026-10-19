import numpy as np
import pandas as pd
import pytest

from core import (
    arc_midpoint,
    calculate_market_metrics,
    company_explorer_tables,
    cycles_through_company,
    curved_arc_points,
    data_quality_summary,
    failure_exposure,
    failure_exposure_display,
    failure_summary_text,
    format_money,
    line_midpoint,
    missing_values_report,
    prepare_relationships,
    validate_data,
    worst_day_comovement,
    worst_day_display,
    worst_day_text,
)


def test_data_quality_summary_hand_calculated_synthetic():
    # SYNTHETIC data: made-up tables, used only for testing.
    validation = {
        "company_rows": 12,
        "relationship_rows": 30,
        "evidence_status_counts": {"confirmed": 18, "provisional": 12},
        "duplicate_company_ids": 0,
        "duplicate_relationship_ids": 0,
        "unknown_source_ids": 0,
        "unknown_target_ids": 0,
        "missing_company_cells": 3,
        "missing_relationship_cells": 0,
    }
    missing_report = pd.DataFrame({
        "table": ["Companies", "Relationships"],
        "column": ["public_ticker", "amount"],
        "missing_count": [3, 0],
        "missing_percent": [25.0, 0.0],
    })
    pairs, verdict, cleaned = data_quality_summary(validation, missing_report)
    assert pairs == [
        ("Companies", 12),
        ("Relationships", 30),
        ("Confirmed relationships", 18),
        ("Provisional relationships", 12),
        ("Duplicate IDs found", 0),
        ("Links pointing to unknown companies", 0),
        ("Empty cells in company data", 3),
        ("Empty cells in relationship data", 0),
    ]
    assert verdict == "All IDs are unique and every link points to a known company."
    assert cleaned.to_dict("records") == [{
        "Table": "Companies",
        "Column": "public_ticker",
        "Empty cells": 3,
        "Share empty (%)": 25.0,
    }]


def test_data_quality_summary_duplicate_verdict_synthetic():
    # SYNTHETIC data: made-up tables, used only for testing.
    validation = {
        "company_rows": 1,
        "relationship_rows": 0,
        "evidence_status_counts": {},
        "duplicate_company_ids": 1,
        "duplicate_relationship_ids": 0,
        "unknown_source_ids": 0,
        "unknown_target_ids": 0,
        "missing_company_cells": 0,
        "missing_relationship_cells": 0,
    }
    _, verdict, _ = data_quality_summary(
        validation,
        pd.DataFrame(columns=["table", "column", "missing_count", "missing_percent"]),
    )
    assert verdict == "Problems found: 1 duplicate IDs."


def test_company_explorer_tables_filters_and_renames_synthetic():
    # SYNTHETIC data: made-up companies, used only for testing.
    companies = pd.DataFrame({
        "company_id": ["A", "B"],
        "company_name": ["Alpha", "Beta"],
    })
    relationships = pd.DataFrame({
        "relationship_id": ["R1", "R2"],
        "source_entity_id": ["A", "B"],
        "target_entity_id": ["B", "A"],
        "relationship_type": ["investment", "purchase"],
        "announcement_date": [None, "2024-01-02"],
        "amount": [None, 10],
        "currency": ["USD", "USD"],
        "evidence_status": ["confirmed", "confirmed"],
        "source_title": ["Source 1", None],
        "source_url": [None, "https://example.test"],
        "source_excerpt": ["technical", "technical"],
    })
    financials = pd.DataFrame({
        "company_id": ["A", "B"],
        "cash": [None, 25],
        "period": ["2024", "2024"],
    })
    related, financial_view = company_explorer_tables(
        companies, relationships, financials, "A"
    )
    assert related.columns.tolist() == [
        "relationship_id", "source_name", "target_name", "relationship_type",
        "announcement_date", "amount", "currency", "evidence_status",
        "source_title", "source_url",
    ]
    assert related.loc[0, "source_name"] == "Alpha"
    assert related.loc[0, "target_name"] == "Beta"
    assert related.loc[0, "announcement_date"] == ""
    assert financial_view["company_id"].tolist() == ["A"]
    assert financial_view.loc[0, "cash"] == ""


def test_describe_cycle_amounts_uses_numbered_name_free_steps_synthetic():
    # SYNTHETIC data: made-up companies, used only for testing.
    companies = pd.DataFrame({"company_id": ["A", "B"], "company_name": ["Alpha", "Beta"]})
    relationships = pd.DataFrame({
        "source_entity_id": ["A", "B"],
        "target_entity_id": ["B", "A"],
        "relationship_type": ["investment", "supplier_customer"],
        "amount": [10_000_000_000, pd.NA],
        "currency": ["USD", "USD"],
        "evidence_status": ["confirmed", "confirmed"],
        "notes": ["", ""],
    })
    cycles = pd.DataFrame({
        "cycle_length": [2],
        "companies_in_cycle": ["Alpha в†’ Beta в†’ Alpha"],
        "company_ids": ["A в†’ B в†’ A"],
    })
    cycle_sep = "".join(chr(value) for value in (1074, 8224, 8217))
    cycles["companies_in_cycle"] = ["Alpha" + cycle_sep + " Beta" + cycle_sep + " Alpha"]
    cycles["company_ids"] = ["A" + cycle_sep + " B" + cycle_sep + " A"]
    result = describe_cycle_amounts(companies, relationships, cycles, data_dictionary=cycle_amount_dictionary())
    assert result.loc[0, "Amounts by step (same order as the cycle)"] == (
        "1) $10.0B, investment | 2) Not disclosed, purchase"
    )


def test_describe_cycle_amounts_marks_up_to_synthetic():
    # SYNTHETIC data: made-up companies, used only for testing.
    cycle_sep = "".join(chr(value) for value in (1074, 8224, 8217))
    companies = pd.DataFrame({"company_id": ["A", "B"], "company_name": ["Alpha", "Beta"]})
    relationships = pd.DataFrame({
        "source_entity_id": ["A"], "target_entity_id": ["A"],
        "relationship_type": ["investment"], "amount": [5_000_000_000],
        "currency": ["USD"], "evidence_status": ["confirmed"], "notes": ["Up to."],
    })
    cycles = pd.DataFrame({"cycle_length": [1], "companies_in_cycle": ["Alpha"], "company_ids": ["A → A"]})
    result = describe_cycle_amounts(companies, relationships, cycles, data_dictionary=cycle_amount_dictionary())
    assert result.loc[0, "Amounts by step (same order as the cycle)"] == "1) up to $5.0B, investment"


def test_describe_cycle_amounts_marks_shares_without_value_synthetic():
    # SYNTHETIC data: made-up companies, used only for testing.
    cycle_sep = "".join(chr(value) for value in (1074, 8224, 8217))
    companies = pd.DataFrame({"company_id": ["A", "B"], "company_name": ["Alpha", "Beta"]})
    relationships = pd.DataFrame({
        "source_entity_id": ["A"], "target_entity_id": ["A"],
        "relationship_type": ["investment"], "amount": [pd.NA],
        "currency": ["USD"], "evidence_status": ["confirmed"],
        "notes": ["Amount is in SHARES, not dollars"],
    })
    cycles = pd.DataFrame({"cycle_length": [1], "companies_in_cycle": ["Alpha"], "company_ids": ["A → A"]})
    result = describe_cycle_amounts(companies, relationships, cycles, data_dictionary=cycle_amount_dictionary())
    assert result.loc[0, "Amounts by step (same order as the cycle)"] == "1) in shares, no dollar value, investment"


def test_arc_midpoint_hand_calculated_synthetic():
    # SYNTHETIC: with endpoints (0, 0), (2, 0), bend 0.20 gives control (1, 0.4).
    # At t=0.5: 0.25*(0,0) + 0.5*(1,0.4) + 0.25*(2,0) = (1.0, 0.2).
    assert arc_midpoint(0.0, 0.0, 2.0, 0.0) == pytest.approx((1.0, 0.2))


def test_curved_arc_points_returns_synthetic_endpoints():
    # SYNTHETIC: a quadratic curve starts at its source and ends at its target.
    points = curved_arc_points(1.0, 2.0, 5.0, 6.0)
    assert len(points) == 31
    assert points[0] == pytest.approx((1.0, 2.0))
    assert points[-1] == pytest.approx((5.0, 6.0))


def test_format_money_hand_calculated_synthetic():
    # SYNTHETIC: 2,500,000,000 / 1,000,000,000 = 2.5, formatted to two decimals.
    assert format_money(2_500_000_000) == "USD 2.50B"


def test_line_midpoint_hand_calculated_synthetic():
    # SYNTHETIC: ((0 + 4) / 2, (2 + 6) / 2) = (2, 4).
    assert line_midpoint(0.0, 2.0, 4.0, 6.0) == pytest.approx((2.0, 4.0))


def synthetic_failure_data():
    # SYNTHETIC data: made-up companies, used only for testing
    companies = pd.DataFrame({
        "company_id": ["A", "B", "C", "D", "X"],
        "company_name": ["Alpha", "Beta", "Gamma", "Delta", "Unlinked"],
    })
    relationships = pd.DataFrame({
        "relationship_id": ["R1", "R2", "R3", "R4", "RP"],
        "source_entity_id": ["A", "B", "C", "C", "B"],
        "target_entity_id": ["B", "C", "A", "D", "D"],
        "evidence_status": ["confirmed", "confirmed", "confirmed", "confirmed", "provisional"],
    })
    return companies, relationships


def test_failure_exposure_hand_calculated_synthetic():
    companies, relationships = synthetic_failure_data()
    result = failure_exposure(companies, relationships, "B")
    rows = result.set_index("company_id")
    # SYNTHETIC hand calculation: A->B, B->C, C->A, C->D. If B fails,
    # A is hop 1 and pays into B, C is hop 1 and is paid by B, D is hop 2 and is paid by B; cycles through B = 1.
    assert rows.loc["A", ["hops", "link_type"]].tolist() == [1, "pays into the failed company"]
    assert rows.loc["C", ["hops", "link_type"]].tolist() == [1, "is paid by the failed company"]
    assert rows.loc["D", ["hops", "link_type"]].tolist() == [2, "is paid by the failed company"]
    assert cycles_through_company(companies, relationships, "B") == 1


def test_failure_exposure_ignores_provisional_synthetic():
    companies, relationships = synthetic_failure_data()
    result = failure_exposure(companies, relationships, "B", confirmed_only=True)
    assert "D" in set(result["company_id"])
    assert "RP" not in set(result["direct_relationship_ids"])


def test_failure_exposure_unknown_company_raises_synthetic():
    companies, relationships = synthetic_failure_data()
    with pytest.raises(ValueError, match="not found"):
        failure_exposure(companies, relationships, "UNKNOWN")


def test_failure_exposure_excludes_unconnected_company_synthetic():
    companies, relationships = synthetic_failure_data()
    result = failure_exposure(companies, relationships, "B")
    assert "X" not in set(result["company_id"])


def test_failure_exposure_display_hand_calculated_synthetic():
    companies, relationships = synthetic_failure_data()
    exposure = failure_exposure(companies, relationships, "B")
    display = failure_exposure_display(exposure, "Beta")
    # SYNTHETIC data: made-up companies, used only for testing.
    expected = pd.DataFrame({
        "Company": ["Alpha", "Gamma", "Delta"],
        "Distance": ["Directly linked", "Directly linked", "2 steps away"],
        "How it is linked": [
            "Sends money to Beta",
            "Receives money from Beta",
            "Receives money from Beta",
        ],
        "Deal IDs": ["R1", "R2", ""],
    })
    pd.testing.assert_frame_equal(display, expected)
    # Hand-written expected summary: two direct links, one company two steps away, and one loop.
    assert failure_summary_text(exposure, "Beta", 1) == (
        "If Beta failed, 2 companies are directly linked to it and 1 more company is two steps away. "
        "1 recorded loop pass through it. This is a structural view, not a forecast."
    )


def test_failure_summary_singular_wording_synthetic():
    # SYNTHETIC data: made-up companies, used only for testing.
    exposure = pd.DataFrame({
        "company_name": ["Alpha"],
        "hops": [1],
        "link_type": ["pays into the failed company"],
        "direct_relationship_ids": ["R1"],
    })
    assert failure_summary_text(exposure, "Beta", 1) == (
        "If Beta failed, 1 company is directly linked to it. "
        "1 recorded loop pass through it. This is a structural view, not a forecast."
    )


def test_failure_summary_no_direct_links_synthetic():
    # SYNTHETIC data: made-up companies, used only for testing.
    exposure = pd.DataFrame({
        "company_name": ["Gamma", "Delta", "Epsilon"],
        "hops": [2, 3, 3],
        "link_type": [
            "is paid by the failed company",
            "is paid by the failed company",
            "both",
        ],
        "direct_relationship_ids": ["", "", ""],
    })
    assert failure_summary_text(exposure, "Beta", 0) == (
        "If Beta failed, no companies are directly linked to it, and 1 more company is two steps away "
        "and 2 more are three or more steps away. 0 recorded loops pass through it. "
        "This is a structural view, not a forecast."
    )


def synthetic_worst_day_prices():
    # SYNTHETIC data: made-up prices, used only for testing
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    x_prices = [100.0, 90.0, 91.8, 86.292, 87.15492, 89.7695676]
    y_prices = [100.0, 95.0, 95.95, 93.0715, 93.0715, 94.93293]
    spy_prices = [100.0] * 6
    return pd.DataFrame({
        "date": list(dates) * 3,
        "ticker": ["X"] * 6 + ["Y"] * 6 + ["SPY"] * 6,
        "adjusted_close": x_prices + y_prices + spy_prices,
    })


def test_worst_day_comovement_hand_calculated_synthetic():
    prices = synthetic_worst_day_prices()
    result, summary = worst_day_comovement(prices, "X", worst_n=2)
    y = result.set_index("ticker").loc["Y"]
    # SYNTHETIC hand calculation: X's worst returns are -10% and -6%.
    # Y on those days averages (-5% + -3%) / 2 = -4.0%; other days average (1% + 0% + 2%) / 3 = 1.0%; down share = 100%.
    assert y["avg_return_on_those_days_pct"] == pytest.approx(-4.0)
    assert y["avg_return_other_days_pct"] == pytest.approx(1.0)
    assert y["days_down_share_pct"] == pytest.approx(100.0)
    assert y["days_used"] == 2
    assert summary["days_used"] == 5


def test_worst_day_comovement_unknown_ticker_raises_synthetic():
    # SYNTHETIC data: made-up prices, used only for testing
    with pytest.raises(ValueError, match="was not found"):
        worst_day_comovement(synthetic_worst_day_prices(), "NOPE", worst_n=2)


def test_worst_day_comovement_rejects_too_large_worst_n_synthetic():
    # SYNTHETIC data: made-up prices, used only for testing
    with pytest.raises(ValueError, match="Fewer than 6"):
        worst_day_comovement(synthetic_worst_day_prices(), "X", worst_n=6)


def test_worst_day_text_returns_exact_sentence_synthetic():
    # SYNTHETIC data: made-up prices, used only for testing
    prices = synthetic_worst_day_prices()
    result, summary = worst_day_comovement(prices, "X", worst_n=2)
    assert worst_day_text(result, summary) == (
        "On the 2 days when X fell the most (average -8.0%), "
        "SPY moved on average 0.0% and Y moved on average -4.0%."
    )


def sample_companies():
    return pd.DataFrame(
        {
            "company_id": ["A", "B"],
            "company_name": ["Alpha", "Beta"],
        }
    )


def sample_relationships():
    return pd.DataFrame(
        {
            "relationship_id": ["R1", "R2", "R3"],
            "source_entity_id": ["A", "A", "B"],
            "target_entity_id": ["B", "B", "A"],
            "evidence_status": ["confirmed", "provisional", "confirmed"],
        }
    )


def test_validate_data_counts_rows_and_statuses():
    result = validate_data(sample_companies(), sample_relationships())
    assert result["company_rows"] == 2
    assert result["relationship_rows"] == 3
    assert result["duplicate_company_ids"] == 0
    assert result["unknown_source_ids"] == 0
    assert result["unknown_target_ids"] == 0
    assert result["evidence_status_counts"] == {
        "confirmed": 2,
        "provisional": 1,
    }


def test_validate_data_detects_duplicate_and_unknown_ids():
    companies = pd.DataFrame(
        {"company_id": ["A", "A"], "company_name": ["Alpha", "Alpha duplicate"]}
    )
    relationships = pd.DataFrame(
        {
            "relationship_id": ["R1"],
            "source_entity_id": ["UNKNOWN"],
            "target_entity_id": ["A"],
            "evidence_status": ["confirmed"],
        }
    )
    result = validate_data(companies, relationships)
    assert result["duplicate_company_ids"] == 1
    assert result["unknown_source_ids"] == 1
    assert result["unknown_target_ids"] == 0


def test_validate_data_rejects_missing_required_columns():
    companies = pd.DataFrame({"company_id": ["A"]})
    relationships = sample_relationships()
    with pytest.raises(ValueError, match="Companies table is missing columns"):
        validate_data(companies, relationships)


def test_prepare_relationships_filters_confirmed_and_adds_names():
    result = prepare_relationships(
        sample_companies(), sample_relationships(), confirmed_only=True
    )
    assert len(result) == 2
    assert set(result["relationship_id"]) == {"R1", "R3"}
    assert result.loc[result["relationship_id"] == "R1", "source_company"].iloc[0] == "Alpha"
    assert result.loc[result["relationship_id"] == "R1", "target_company"].iloc[0] == "Beta"


def test_prepare_relationships_can_include_provisional():
    result = prepare_relationships(
        sample_companies(), sample_relationships(), confirmed_only=False
    )
    assert len(result) == 3
    assert "R2" in set(result["relationship_id"])


def test_prepare_relationships_rejects_duplicate_company_ids():
    companies = pd.DataFrame(
        {"company_id": ["A", "A"], "company_name": ["Alpha", "Duplicate"]}
    )
    with pytest.raises(ValueError, match="Company IDs must be unique"):
        prepare_relationships(companies, sample_relationships())


def test_missing_values_report_counts_missing_cells():
    companies = pd.DataFrame(
        {"company_id": ["A", None], "company_name": ["Alpha", "Beta"]}
    )
    relationships = pd.DataFrame({"relationship_id": ["R1", None]})
    financials = pd.DataFrame({"cash": [100.0, np.nan]})
    result = missing_values_report(companies, relationships, financials)

    row = result[(result["table"] == "Companies") & (result["column"] == "company_id")].iloc[0]
    assert row["missing_count"] == 1
    assert row["missing_percent"] == 50.0
    row = result[(result["table"] == "Financials") & (result["column"] == "cash")].iloc[0]
    assert row["missing_count"] == 1



def test_calculate_market_metrics_hand_calculated_return():
    prices = pd.DataFrame({
        "date": [
            "2024-01-01", "2024-01-02", "2024-01-03",
            "2024-01-01", "2024-01-02", "2024-01-03",
        ],
        "ticker": ["AAA", "AAA", "AAA", "SPY", "SPY", "SPY"],
        "adjusted_close": [100.0, 110.0, 120.0, 100.0, 101.0, 102.0],
    })

    result = calculate_market_metrics(prices)
    row = result.loc[result["ticker"] == "AAA"].iloc[0]

    # Полная доходность: (120 / 100 - 1) * 100 = 20%.
    assert row["total_return_pct"] == pytest.approx(20.0)
    assert row["observations"] == 2



def test_calculate_market_metrics_rejects_nonpositive_prices():
    prices = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "ticker": ["AAA", "SPY"],
            "adjusted_close": [0.0, 100.0],
        }
    )
    with pytest.raises(ValueError, match="must be positive"):
        calculate_market_metrics(prices)


def test_calculate_market_metrics_rejects_duplicate_date_ticker():
    prices = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "ticker": ["AAA", "AAA", "SPY"],
            "adjusted_close": [100.0, 101.0, 100.0],
        }
    )
    with pytest.raises(ValueError, match="Duplicate date/ticker"):
        calculate_market_metrics(prices)


def test_calculate_market_metrics_requires_benchmark():
    prices = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "ticker": ["AAA", "AAA"],
            "adjusted_close": [100.0, 101.0],
        }
    )
    with pytest.raises(ValueError, match="Benchmark SPY was not found"):
        calculate_market_metrics(prices)



# ---------------------------------------------------------------------------
# Новые тесты (ответы посчитаны ВРУЧНУЮ, расчёты — в приложении к записке)
# ---------------------------------------------------------------------------
from core import calculate_cycle_metrics, normalized_performance


def two_asset_prices():
    # SPY: 100 -> 110 -> 99 -> 108.9  (доходности +10%, -10%, +10%)
    # AAA: 100 -> 120 -> 96 -> 115.2  (доходности +20%, -20%, +20%, ровно в 2 раза больше SPY)
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    return pd.DataFrame({
        "date": dates + dates,
        "ticker": ["SPY"] * 4 + ["AAA"] * 4,
        "adjusted_close": [100.0, 110.0, 99.0, 108.9, 100.0, 120.0, 96.0, 115.2],
    })


def test_hand_calculated_beta_equals_two():
    # Бета = ковариация(AAA, SPY) / дисперсия(SPY). Так как AAA = 2 * SPY, бета = 2.
    row = calculate_market_metrics(two_asset_prices()).set_index("ticker").loc["AAA"]
    assert row["beta_vs_spy"] == pytest.approx(2.0)


def test_hand_calculated_total_return():
    # AAA: 115.2 / 100 - 1 = 15.2%.  SPY: 108.9 / 100 - 1 = 8.9%.
    result = calculate_market_metrics(two_asset_prices()).set_index("ticker")
    assert result.loc["AAA", "total_return_pct"] == pytest.approx(15.2)
    assert result.loc["SPY", "total_return_pct"] == pytest.approx(8.9)


def test_hand_calculated_annualized_volatility():
    # Доходности AAA: +0.2, -0.2, +0.2. Среднее = 0.0667.
    # Отклонения: 0.1333, -0.2667, 0.1333 -> квадраты: 0.017778, 0.071111, 0.017778
    # Сумма = 0.106667; делим на (n-1)=2 -> 0.053333; корень -> 0.230940.
    # Годовая волатильность = 0.230940 * sqrt(252) * 100 = 366.6%
    row = calculate_market_metrics(two_asset_prices()).set_index("ticker").loc["AAA"]
    assert row["annualized_volatility_pct"] == pytest.approx(366.6, rel=1e-3)


def test_spy_beta_is_one_and_observations_counted():
    row = calculate_market_metrics(two_asset_prices()).set_index("ticker").loc["SPY"]
    assert row["beta_vs_spy"] == pytest.approx(1.0)
    assert row["observations"] == 3


def test_normalized_performance_starts_at_100_for_each_ticker():
    # Тикер X: 50, 55, 60 -> 100, 110, 120.  Тикер Y: 200, 190 -> 100, 95.
    prices = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"],
        "ticker": ["X", "X", "X", "Y", "Y"],
        "adjusted_close": [50.0, 55.0, 60.0, 200.0, 190.0],
    })
    result = normalized_performance(prices)
    x = result[result["ticker"] == "X"]["normalized_value"].tolist()
    y = result[result["ticker"] == "Y"]["normalized_value"].tolist()
    assert x == pytest.approx([100.0, 110.0, 120.0])
    assert y == pytest.approx([100.0, 95.0])


def test_cycle_detection_finds_three_company_cycle():
    # A -> B -> C -> A: ровно один цикл длины 3. Связь D -> A цикл не создаёт.
    companies = pd.DataFrame({
        "company_id": ["A", "B", "C", "D"],
        "company_name": ["Aa", "Bb", "Cc", "Dd"],
    })
    relationships = pd.DataFrame({
        "relationship_id": ["R1", "R2", "R3", "R4"],
        "source_entity_id": ["A", "B", "C", "D"],
        "target_entity_id": ["B", "C", "A", "A"],
        "evidence_status": ["confirmed"] * 4,
    })
    table, summary = calculate_cycle_metrics(companies, relationships)
    assert summary["cycle_count"] == 1
    assert summary["companies_in_cycles"] == 3
    assert table.iloc[0]["cycle_length"] == 3


def test_cycle_detection_ignores_provisional_links_when_confirmed_only():
    # Обратная связь B -> A только provisional: при confirmed_only цикла нет, при выключенном фильтре есть.
    companies = pd.DataFrame({"company_id": ["A", "B"], "company_name": ["Aa", "Bb"]})
    relationships = pd.DataFrame({
        "relationship_id": ["R1", "R2"],
        "source_entity_id": ["A", "B"],
        "target_entity_id": ["B", "A"],
        "evidence_status": ["confirmed", "provisional"],
    })
    _, strict = calculate_cycle_metrics(companies, relationships, confirmed_only=True)
    _, loose = calculate_cycle_metrics(companies, relationships, confirmed_only=False)
    assert strict["cycle_count"] == 0
    assert loose["cycle_count"] == 1

    

# ---------------------------------------------------------------------------
# Тесты на Sharpe, просадку, корзины (ответы посчитаны ВРУЧНУЮ)
# ---------------------------------------------------------------------------
from core import (
    beta_vs_benchmark,
    compare_groups,
    cycle_member_tickers,
    max_drawdown,
    sharpe_ratio,
)


def test_hand_calculated_sharpe_without_risk_free():
    # Дневные доходности: 0.01 и 0.03. Среднее = 0.02.
    # Отклонения: -0.01 и +0.01 -> квадраты 0.0001 + 0.0001 = 0.0002; делим на (n-1)=1 -> 0.0002
    # Стандартное отклонение = sqrt(0.0002) = 0.0141421.
    # Дневной Sharpe = 0.02 / 0.0141421 = 1.41421; годовой = 1.41421 * sqrt(252) = 22.45
    assert sharpe_ratio(pd.Series([0.01, 0.03])) == pytest.approx(22.45, rel=1e-3)


def test_hand_calculated_sharpe_with_risk_free():
    # Безрисковая ставка 2.52% в год -> 0.0252 / 252 = 0.0001 в день.
    # Избыточные доходности: 0.0099 и 0.0299. Среднее = 0.0199. Отклонение прежнее: 0.0141421.
    # Дневной Sharpe = 0.0199 / 0.0141421 = 1.40714; годовой = 1.40714 * sqrt(252) = 22.34
    assert sharpe_ratio(pd.Series([0.01, 0.03]), risk_free_annual_pct=2.52) == pytest.approx(22.34, rel=1e-3)


def test_sharpe_is_nan_when_returns_do_not_vary():
    assert np.isnan(sharpe_ratio(pd.Series([0.01, 0.01, 0.01])))


def test_hand_calculated_max_drawdown():
    # Доходности: +10%, -20%, +5%. Стоимость: 1.00 -> 1.10 -> 0.88 -> 0.924.
    # Пик перед падением = 1.10, дно = 0.88. Просадка = 0.88 / 1.10 - 1 = -0.20 (-20%).
    assert max_drawdown(pd.Series([0.10, -0.20, 0.05])) == pytest.approx(-0.20)


def test_max_drawdown_counts_a_fall_from_the_start():
    # Доходности: -10%, +10%. Стоимость: 1.00 -> 0.90 -> 0.99. Просадка от старта = -10%.
    assert max_drawdown(pd.Series([-0.10, 0.10])) == pytest.approx(-0.10)


def test_beta_vs_benchmark_is_two_for_double_moves():
    market = pd.Series([0.10, -0.10, 0.10])
    assert beta_vs_benchmark(market * 2, market) == pytest.approx(2.0)


def test_cycle_member_tickers_returns_only_public_cycle_members():
    # Цикл A -> B -> C -> A. C частная (без тикера). D публичная, но в цикл не входит.
    companies = pd.DataFrame({
        "company_id": ["A", "B", "C", "D"],
        "company_name": ["Aa", "Bb", "Cc", "Dd"],
        "public_ticker": ["AAA", "BBB", None, "DDD"],
    })
    relationships = pd.DataFrame({
        "relationship_id": ["R1", "R2", "R3", "R4"],
        "source_entity_id": ["A", "B", "C", "D"],
        "target_entity_id": ["B", "C", "A", "A"],
        "evidence_status": ["confirmed"] * 4,
    })
    assert cycle_member_tickers(companies, relationships) == ["AAA", "BBB"]


def test_compare_groups_hand_calculated_two_stock_basket():
    # X: 100 -> 110 -> 121 (доходности +10%, +10%). Y: 100 -> 100 -> 110 (0%, +10%).
    # Корзина 50/50: день 1 = (10% + 0%) / 2 = 5%; день 2 = (10% + 10%) / 2 = 10%.
    # Итог = 1.05 * 1.10 - 1 = 15.5%.  SPY: 100 -> 101 -> 102.01 даёт +2.01%.
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    prices = pd.DataFrame({
        "date": dates * 3,
        "ticker": ["X"] * 3 + ["Y"] * 3 + ["SPY"] * 3,
        "adjusted_close": [100.0, 110.0, 121.0, 100.0, 100.0, 110.0, 100.0, 101.0, 102.01],
    })
    table, info = compare_groups(prices, {"Basket": ["X", "Y"]})
    basket = table.set_index("group").loc["Basket"]
    assert basket["total_return_pct"] == pytest.approx(15.5)
    assert table.set_index("group").loc["SPY", "total_return_pct"] == pytest.approx(2.01)
    assert info["price_dates_used"] == 3
    assert info["price_dates_dropped"] == 0


def test_compare_groups_reports_dropped_dates_and_limiting_ticker():
    # LATE торгуется только с 2024-01-03, поэтому 2 первых дня отбрасываются для всех.
    prices = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"] +
                ["2024-01-03", "2024-01-04", "2024-01-05"],
        "ticker": ["SPY"] * 5 + ["LATE"] * 3,
        "adjusted_close": [100.0, 101.0, 102.0, 103.0, 104.0, 50.0, 55.0, 60.0],
    })
    table, info = compare_groups(prices, {"Late group": ["LATE"]})
    assert info["price_dates_used"] == 3
    assert info["price_dates_dropped"] == 2
    assert info["limiting_ticker"] == "LATE"
    assert info["limiting_start"] == "2024-01-03"


def test_compare_groups_rejects_unknown_ticker():
    with pytest.raises(ValueError, match="Tickers not found"):
        compare_groups(two_asset_prices(), {"Group": ["NOPE"]})




# ---------------------------------------------------------------------------
# Тесты для графика роста (ответы посчитаны ВРУЧНУЮ)
# ---------------------------------------------------------------------------
from core import growth_of_base


def test_growth_of_base_hand_calculated():
    # Серия A: доходности +10%, +10% -> 100 -> 110 -> 121.
    # Серия B: доходности 0%, +10%    -> 100 -> 100 -> 110.
    daily = pd.DataFrame(
        {"A": [0.10, 0.10], "B": [0.0, 0.10]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    result = growth_of_base(daily, "2024-01-01", base_value=100.0)
    a = result[result["series"] == "A"].sort_values("date")["value"].tolist()
    b = result[result["series"] == "B"].sort_values("date")["value"].tolist()
    assert a == pytest.approx([100.0, 110.0, 121.0])
    assert b == pytest.approx([100.0, 100.0, 110.0])


def test_compare_groups_returns_daily_returns_for_the_chart():
    # Корзина X/Y из ручного расчёта: дневные доходности корзины 5% и 10%, SPY 1% и 1%.
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    prices = pd.DataFrame({
        "date": dates * 3,
        "ticker": ["X"] * 3 + ["Y"] * 3 + ["SPY"] * 3,
        "adjusted_close": [100.0, 110.0, 121.0, 100.0, 100.0, 110.0, 100.0, 101.0, 102.01],
    })
    _, info = compare_groups(prices, {"Basket": ["X", "Y"]})
    daily = info["daily_returns"]
    assert daily["Basket"].tolist() == pytest.approx([0.05, 0.10])
    assert daily["SPY"].tolist() == pytest.approx([0.01, 0.01])

    

# ---------------------------------------------------------------------------
# Тест для сводки по ценам (панель честности)
# ---------------------------------------------------------------------------
from core import price_data_summary


def test_price_data_summary_counts_per_ticker():
    # AAA: 3 наблюдения с 2024-01-01 по 2024-01-03. BBB: 1 наблюдение.
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-02"]),
        "ticker": ["AAA", "AAA", "AAA", "BBB"],
        "adjusted_close": [10.0, 11.0, 12.0, 20.0],
        "currency": ["USD"] * 4,
        "source": ["Yahoo"] * 4,
        "retrieved_date": ["2026-10-10"] * 4,
    })
    result = price_data_summary(prices).set_index("ticker")
    assert result.loc["AAA", "observations"] == 3
    assert str(result.loc["AAA", "first_date"].date()) == "2024-01-01"
    assert str(result.loc["AAA", "last_date"].date()) == "2024-01-03"
    assert result.loc["BBB", "observations"] == 1
    assert result.loc["AAA", "pull_date"] == "2026-10-10"
    assert result.loc["AAA", "currency"] == "USD"


from core import describe_cycle_amounts


def cycle_amount_dictionary():
    return pd.DataFrame({
        "field_or_term": ["amount (relationships)"],
        "unit_or_allowed_values": ["whole USD; blank if not disclosed"],
    })


def test_describe_cycle_amounts_lists_all_present_amounts():
    companies = pd.DataFrame({
        "company_id": ["A", "B"],
        "company_name": ["Alpha", "Beta"],
    })
    relationships = pd.DataFrame({
        "source_entity_id": ["A", "B"],
        "target_entity_id": ["B", "A"],
        "relationship_type": ["investment", "financing"],
        "amount": [100, 25],
        "currency": ["USD", "USD"],
        "evidence_status": ["confirmed", "confirmed"],
    })
    cycles = pd.DataFrame({
        "cycle_length": [2],
        "companies_in_cycle": ["Alpha → Beta → Alpha"],
        "company_ids": ["A → B → A"],
    })
    result = describe_cycle_amounts(companies, relationships, cycles, data_dictionary=cycle_amount_dictionary())
    # Known answer: Alpha -> Beta is 100 USD (investment); Beta -> Alpha is 25 USD (financing).
    assert result.loc[0, "Amounts by step (same order as the cycle)"] == "1) $100.0, investment | 2) $25.0, financing"
    if False: (
        "Alpha → Beta: 100 USD (investment); Beta → Alpha: 25 USD (financing)"
    )


def test_describe_cycle_amounts_marks_missing_amount():
    companies = pd.DataFrame({
        "company_id": ["A", "B"],
        "company_name": ["Alpha", "Beta"],
    })
    relationships = pd.DataFrame({
        "source_entity_id": ["A", "B"],
        "target_entity_id": ["B", "A"],
        "relationship_type": ["investment", "supplier_customer"],
        "amount": [100, pd.NA],
        "currency": ["USD", "USD"],
        "evidence_status": ["confirmed", "confirmed"],
    })
    cycles = pd.DataFrame({
        "cycle_length": [2],
        "companies_in_cycle": ["Alpha → Beta → Alpha"],
        "company_ids": ["A → B → A"],
    })
    result = describe_cycle_amounts(companies, relationships, cycles, data_dictionary=cycle_amount_dictionary())
    # Known answer: the second relationship has no disclosed amount.
    assert result.loc[0, "Amounts by step (same order as the cycle)"] == "1) $100.0, investment | 2) Not disclosed, purchase"
    if False: (
        "Alpha → Beta: 100 USD (investment); Beta → Alpha: amount not disclosed USD (supplier_customer)"
    )
    
