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

**Update 2026-11-28 — `financials.csv` refreshed by `scripts/fetch_financials_av.py`.** One row per company (16 rows). Listed companies now carry the latest quarterly balance sheet (`fiscal_date_ending`, 2026-07-31 to 2026-10-31) with `total_debt`, `cash_and_equivalents`, `net_debt = total_debt − cash` and trailing-twelve-month EBITDA (`ebitda_basis` says when only the latest quarter ×4 was available: AMZN). `data_source` records the provider per row:
- `alpha_vantage` (BALANCE_SHEET + INCOME_STATEMENT, free tier): **NVDA, MSFT**. `total_debt` = Alpha Vantage `shortLongTermDebtTotal`, which includes lease liabilities (MSFT $133.9B vs $57.2B debt + finance leases in yfinance). The free tier's 25-calls/day limit was reached after these two tickers.
- `yfinance` (quarterly statements, fallback): ORCL, AMD, CRWV, AMZN, GOOGL, META, AVGO, APLD, GLXY, BX. `total_debt` = Yahoo "Total Debt" (debt plus capital-lease obligations).
- Debt definitions therefore differ slightly between the two providers; compare companies with that caveat. MSFT's and NVDA's EBITDA were cross-checked against yfinance (MSFT TTM identical at $222.9B).
- APLD and GLXY report negative TTM EBITDA; BX reports no EBITDA. Models fall back to cash as the loss buffer in those cases.
- OpenAI, Anthropic, SB Energy and Fluidstack stay empty (`data_source = private (no public filings)`); nothing is imputed.
- The API key is read only from the `ALPHAVANTAGE_API_KEY` environment variable or a git-ignored `.env`; the app never calls the API and only reads the cached CSV.

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

---

## 6. v3 extended datasets (added 2026-11-27)

Fetched by `python scripts/fetch_extended_data.py` (Yahoo Finance via `yfinance`, FRED CSV endpoint) on 2026-11-27 and saved for offline, reproducible runs. `data.py` explicitly excludes these four files from the legacy price panel, so the four original tabs use exactly the same per-ticker CSVs as before.

| File | Content | Rows / window | Notes |
|---|---|---|---|
| `data/raw/volumes_core.csv` | Adj. close + volume, 12 core tickers + SPY | 11,586 · 2023-01-03 → 2026-11-27 | CRWV from 2025-03-28, GLXY from 2025-05-16 |
| `data/raw/spillover_tiers.csv` | QQQ, SOXX, HYG, LQD, XLU, VST, CEG, EMB | 7,840 · 2023-01-03 → 2026-11-27 | Tier 3–4 spillover proxies |
| `data/raw/dotcom_1998_2003.csv` | ^GSPC, ^IXIC | 3,014 · 1998-01-02 → 2003-12-30 | Reproduces the cited peaks/troughs exactly (tested) |
| `data/raw/fred_credit_spreads.csv` | HY OAS, IG OAS, 10Y UST (%) | 1,030 · 2023-01-02 → 2026-11-26 | ICE BofA OAS starts 2023-11-28 on FRED; rows are a union of calendars, so single-series gaps are holidays (tested < 10% after first value) |
| `data/circular_alpha_research_pack.xlsx` | J.P. Morgan / WSJ / Bloomberg Oct–Nov 2026 credit pack, 2000–02 benchmarks, literature | 6 sheets | Formula cells (totals, tenor premiums) are recomputed in `src/credit_signals.py` |

**Known data issues (v3).**
1. *Bond matrix rounding:* component sums give Other $110B / MSFT $45B; J.P. Morgan prints $111B / $44B. Both are kept (`total_bn`, `jpm_printed_bn`); totals still equal $495B + $70B + $64B = $629B.
2. *Volume adjustment:* the extended files use split-adjusted volumes; closes differ from the legacy CSVs by < 5% (dividend re-basing on a later retrieval date; tested).
3. *Turnover proxy:* shares outstanding are not in the data, so GSY "turnover" uses dollar volume.
4. *Private / unlisted entities:* OpenAI, Anthropic (S-1 filed Oct 2026, see item 5), SB Energy, Fluidstack have no SEC financials. (Since 2026-11-28 APLD, GLXY and BX are in `financials.csv`, so DebtRank now uses their buffers: cash where EBITDA is negative or missing.) DebtRank does not impute their buffers (they cannot absorb measured distress) and their scenario price impact is the tier shock only.
5. *Anthropic IPO status (updated 2026-11-27):* Anthropic filed its IPO registration statement (S-1) in October 2026 (WSJ, Oct 30, 2026), which also disclosed Broadcom's direct lending. Any listing is too recent to provide usable trading history, so Anthropic **remains excluded from the 2023–2026 daily price panel** and from all time-series models (BSADF, crash score, Diebold–Yilmaz, scenario weights). `companies_clean.csv` is deliberately left unchanged (still `is_public = False`, no ticker) so the existing tests and cycle counts stay reproducible; it is still a no-buffer node in DebtRank until audited public financials are added to `financials.csv`.
6. *Single credit snapshot:* J.P. Morgan HY HPC, tenor curve and CoreWeave spreads are one Oct–Nov 2026 observation, not a time series; only the FRED HY OAS enters the time-varying score.

**Survivorship bias.** The 12-ticker universe is today's survivors and winners; the 2000–2003 analog had 47 CLEC bankruptcies whose equities left the indices. Crash-risk calibration uses index-level data (^IXIC), which is itself survivor-weighted. CRWV and GLXY have short listing histories, so Tier 1 is an availability-masked equal-weight of whichever names traded each day.

**Look-ahead bias prevention.** BSADF, GSY z-scores (trailing windows), dollar-volume ratios, rolling Diebold–Yilmaz TCI and the NNLS effective weights use data ≤ t; every tradable signal is lagged one day. Canary tests perturb data after t₀ and assert zero change at t ≤ t₀ for BSADF, the crash-risk probability, trigger statuses and rolling TCI (`tests/test_methodology_v3.py`, `tests/test_extended_data.py`). Exception, stated openly: the dot-com logistic calibration uses forward 24-month labels from 1999–2001 — historical data that predate the AI sample entirely.

**Short-selling & borrow costs.** Shorting high-beta neoclouds (`CRWV`, `APLD`, `GLXY`) incurs elevated borrow fees, possible recalls and squeeze risk. The playbook therefore reports a cost-over-horizon column (2 × transaction cost + borrow fee × horizon) using user-entered assumptions, and chipmaker exposure is hedged via puts/pairs rather than naked shorts.
