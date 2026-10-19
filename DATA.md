# DATA.md - Circular Alpha

All data are saved as CSV files in the project and loaded by `data.py`. The app never downloads anything, so it runs offline. 

## 1. Price series (used for returns, volatility, beta, Sharpe, drawdown)

| Ticker | Source | Currency | Frequency | Adjusted? | First date | Last date | Observations | Pull date | Official cross-check |
|---|---|---|---|---|---|---|---|---|---|
| AMD | https://finance.yahoo.com/quote/AMD/history/ | USD | Daily | Yes, `adjusted_close` | 2023-01-03 | 2026-10-08 | 945 | 2026-10-10 |
| AMZN | https://finance.yahoo.com/quote/AMZN/history/ | USD | Daily | Yes | 2023-01-03 | 2026-10-08 | 945 | 2026-10-10 |
| AVGO | https://finance.yahoo.com/quote/AVGO/history/ | USD | Daily | Yes | 2023-01-03 | 2026-10-08 | 945 | 2026-10-10 | 
| CRWV | https://finance.yahoo.com/quote/CRWV/history/ | USD | Daily | Yes | 2025-03-28 | 2026-10-08 | 385 | 2026-10-10 |
| GOOGL | https://finance.yahoo.com/quote/GOOGL/history/ | USD | Daily | Yes | 2023-01-03 | 2026-10-08 | 945 | 2026-10-10 |
| META | https://finance.yahoo.com/quote/META/history/ | USD | Daily | Yes | 2023-01-03 | 2026-10-08 | 945 | 2026-10-10 |
| MSFT | https://finance.yahoo.com/quote/MSFT/history/ | USD | Daily | Yes | 2023-01-03 | 2026-10-08 | 945 | 2026-10-10 |
| NVDA | https://finance.yahoo.com/quote/NVDA/history/ | USD | Daily | Yes | 2023-01-03 | 2026-10-08 | 945 | 2026-10-10 |
| ORCL | https://finance.yahoo.com/quote/ORCL/history/ | USD | Daily | Yes | 2023-01-03 | 2026-10-08 | 945 | 2026-10-10 |
| SPY (benchmark) | https://finance.yahoo.com/quote/SPY/history/ | USD | Daily | Yes | 2023-01-03 | 2026-10-08 | 945 | 2026-10-10 |

Checks run on these files: no missing values, no duplicate date/ticker rows, no non-positive prices, no gap between consecutive price dates longer than 4 calendar days, and the 9 non-CRWV tickers share exactly the same 945 dates.

## 2. Network data (used for cycle detection)

| File | Content | Rows | Source | Notes |
|---|---|---|---|---|
| `companies_clean.csv` | Companies in the project universe | 12 (9 public, 3 private) | Company pages (URL in each row) | Private: OpenAI, Anthropic, SB Energy. They have no stock price. |
| `relationships_final.csv` | Directed financial relationships | 30 (18 confirmed, 12 provisional) | URL, title and excerpt in every row | Announcement dates 2024-11-22 to 2026-08-26. Review date of all rows 2026-10-10. |
| `data_dictionary_synced.csv` | Field definitions and missing-value rules | 21 | Project team | See known issues below. |

Direction rule (from the dictionary): the edge points from the company that pays or commits money to the company that receives it. A cycle is a closed directed path, which is a structural pattern, not proof of wrongdoing and not a cash-flow total.

## 3. Financial statements (displayed, not used in the headline number)

`financials.csv` has one annual row per company with total debt, cash, net debt and EBITDA in USD and the filing URL. OpenAI and Anthropic are empty (private), SB Energy is empty except for the period, and EBITDA is blank for Microsoft and Alphabet. Amounts must not be summed across companies.

## 4. Data problems found and how they are handled

1. **CRWV has a short history** (from 2025-03-28). Including it shrinks the common window for every series from 945 to 385 price dates, so it is excluded from the comparison group by default and the app shows the number of dropped dates.
2. **Private companies sit in every confirmed cycle.** All 5 confirmed cycles pass through OpenAI or Anthropic, which have no prices. Only three public stocks (AMD, MSFT, NVDA) are in the cycle group, so the comparison is descriptive and cannot support a statistical conclusion.
3. **Evidence quality.** Judged by the web domain of the source URL, all 18 confirmed rows cite a primary source (company, SEC or investor-relations pages). The 9 rows that cite press or Wikipedia are all `provisional`. The app counts only `confirmed` rows by default.
4. **Missing amounts.** 12 of the 30 relationships have no amount, by design (the dictionary says to leave it empty when no amount was disclosed). Amounts are never added up.
5. **Repeated deals between the same two companies** (for example three OpenAI to CoreWeave agreements) are separate rows but one edge in the graph, so the app shows 15 links from 18 confirmed rows.
6. **Cycle count depends on hubs.** With all links included there are 26 cycles, mostly created by two hubs (OpenAI and Anthropic). The cycle count is not a score of "circularity".
7. **Yahoo is an unofficial source** and its adjusted-price definition belongs to Yahoo. That is why a cross-check against an official source is required (section 5).
8. **Known documentation issue:** the data dictionary mentions a file `network_core.py` and cycles of 2 or 3 companies. Cycles are actually computed in `core.py` and the code finds cycles of any length. 


## 5. Limits

Historical data only: not a forecast and not investment advice. The universe was chosen by the team and is not survivorship-free. All series are in USD, so no currency conversion is used. The risk-free rate is an assumption entered by the user, not downloaded data. Window of the headline number: 2023-01-03 to 2026-10-08 (a period of strong AI-related market growth).
