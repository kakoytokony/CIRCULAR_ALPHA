import numpy as np
import pandas as pd
import pytest

from core import (
    calculate_market_metrics,
    missing_values_report,
    prepare_relationships,
    validate_data,
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
    assert result.loc[0, "relationship_amounts"] == (
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
    assert result.loc[0, "relationship_amounts"] == (
        "Alpha → Beta: 100 USD (investment); Beta → Alpha: amount not disclosed USD (supplier_customer)"
    )
    
