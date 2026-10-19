# Product card - Circular Alpha

**Team:** Darina Baikazakova, Linara Mamraimova, Asselya Krymova
**Track:** 8, our own idea: circular financial relationships between AI companies. Twist: instead of a standard portfolio question we combine a relationship graph with market-risk numbers.

**User:** a KBTU student (or any retail investor) who holds, or is thinking of buying, US AI stocks priced in USD and wants to know whether the companies that sit inside "circular" funding loops (A invests in or guarantees B, B buys from A) carry a different risk-return profile from other AI companies.

**Question:** Do public companies that appear in confirmed circular financial relationships show a different risk-adjusted return, market sensitivity and worst fall than other public AI companies and than the S&P 500 (SPY)?

**Decision it supports:** whether to treat the "circular-deal" stocks as a separate risk bucket when sizing positions (for example, not assuming they diversify each other). The user can also pick one company and see which companies are linked to it, how many steps away, and how the other stocks moved on that company's worst days. It is **not** a buy or sell signal and not a claim that cycles cause returns.

**Headline number:** annualized Sharpe ratio of an equal-weight basket of the public companies that appear in confirmed cycles (daily returns, adjusted prices, USD), using a risk-free rate chosen by the user (default 4.0% per year), over the common window 2023-01-03 to 2026-10-08. With the default inputs: cycle group 1.64, comparison group 1.41, SPY 1.16.

**Benchmark:** SPY (S&P 500 ETF), and an equal-weight basket of the other public companies in our universe (AMZN, AVGO, GOOGL, META, ORCL; CRWV is left out by default because it only has prices from 2025-03-28).

## Course concepts used (at least 4, from at least 3 weeks)

| Concept | Week | Feature on screen | Test with known answer |
|---|---|---|---|
| Simple returns, total return | 2 | "Total return" in the market table and the group table | `test_hand_calculated_total_return` |
| Volatility (annualized, sqrt 252) | 3 | "Annualized volatility" | `test_hand_calculated_annualized_volatility` |
| Maximum drawdown | 4 | "Max drawdown" metric | `test_hand_calculated_max_drawdown` |
| Sharpe ratio vs a risk-free rate | 5 | Headline metric and the risk-free rate the user can change | `test_hand_calculated_sharpe_without_risk_free`, `test_hand_calculated_sharpe_with_risk_free` |
| Beta vs an index (SPY) | 6 | "Beta vs SPY" | `test_hand_calculated_beta_equals_two` |
| Portfolio weights, equal weight with daily rebalancing | 8 | Cycle group and comparison group baskets, growth-of-$100 chart | `test_compare_groups_hand_calculated_two_stock_basket`, `test_growth_of_base_hand_calculated` |

Network analysis (directed cycles in the relationship graph, the "what if this company fails" view) and the table of how stocks moved on a company's worst days are our own methods and are not counted as course concepts. Cycles are tested in `test_cycle_detection_finds_three_company_cycle` and `test_cycle_detection_ignores_provisional_links_when_confirmed_only`.

## Data contract

| Series | Source / ticker | Currency | Frequency | Adjusted? | Official cross-check |
|---|---|---|---|---|---|
| Prices: AMD, AMZN, AVGO, CRWV, GOOGL, META, MSFT, NVDA, ORCL | Yahoo Finance history pages, ticker as named | USD | Daily | Yes (`adjusted_close`, Yahoo definition) | Not done: the instructor said it is not required (see `DATA.md`) |
| Benchmark: SPY | Yahoo Finance, SPY | USD | Daily | Yes | Same as above |
| Companies (12) and relationships (30 rows, 18 confirmed / 12 provisional) | Hand-collected from company and SEC pages, with a source URL and excerpt on every row | USD where an amount is given | Event dates (2024-11-22 to 2026-08-26) | Not applicable | Each row has its own source URL; review date 2026-10-10 |
| Financials (debt, cash, net debt, EBITDA) | Company filings, URL in each row | USD | Annual | Not applicable | Source URL per row; not used in the headline number |

## One hand calculation we reproduce in a test

Two daily returns of 0.01 and 0.03 with a 0% risk-free rate: mean = 0.02, sample standard deviation = sqrt(0.0002) = 0.0141421, daily Sharpe = 1.41421, annualized Sharpe = 1.41421 x sqrt(252) = 22.45. Test: `test_hand_calculated_sharpe_without_risk_free`.