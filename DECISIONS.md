# DECISIONS.md - Circular Alpha

Decision record: each major choice we made, the alternatives we considered, the evidence, and which AI advice we accepted or rejected.

## D1. Track: 8 (our own idea), not one of the seven standard tracks
- **Alternatives:** Track 4 (market exposure explorer) or Track 3 (rebalancing simulator), which fit the same data.
- **Evidence:** the project already had a company network, relationships and prices. A standard track would have thrown the network part away, and the network is the most interesting part of our question.
- **Condition:** track 8 is our own proposal and is subject to the instructor's approval.
- **AI advice:** accepted (the assistant pointed out that a self-chosen track needs approval).

## D2. Headline number: Sharpe ratio of an equal-weight basket of public companies that sit in confirmed cycles
- **Alternatives:** (a) the number of cycles as the headline; (b) a "circularity score" regressed on returns, as our first project description promised.
- **Evidence:** only three public stocks (AMD, MSFT, NVDA) are in confirmed cycles, so no statistical test of "cycles lead to returns" is possible. The cycle count is inflated by two private hubs (5 cycles with confirmed links, 26 with all links), so it is not a score of circularity.
- **Result:** a descriptive comparison of risk and return against other public AI stocks and SPY. The screen and the note say it is not a forecast and not proof of cause.
- **AI advice:** accepted (drop the claim of a predictive signal).

## D3. Only `confirmed` relationships by default (switch on screen)
- **Alternatives:** use all 30 relationships.
- **Evidence:** judged by the web domain of the source URL, all 18 confirmed rows cite company, SEC or investor-relations pages. The 9 rows that cite press or Wikipedia are all `provisional`.
- **Trade-off:** fewer cycles. The switch lets the user see the effect (with all links the cycle group becomes AMD, AMZN, CRWV, MSFT, NVDA).

## D4. Equal weights, rebalanced daily, for both baskets
- **Alternatives:** a buy-and-hold basket; market-cap weights.
- **Reason:** the simplest rule that needs no extra data, and it uses the weights and rebalancing topic of the course.
- **Limit:** daily rebalancing ignores trading costs.

## D5. All series compared on one common window; CRWV left out of the comparison group by default
- **Alternatives:** each series from its own first date (the table in section 2 of the app still does this).
- **Evidence:** CRWV has prices only from 2025-03-28 (385 dates against 945 for the others). Including it cuts the common window for every series and drops 560 dates. The screen shows how many dates were dropped and which ticker limits the window.

## D6. Risk-free rate typed by the user (default 4.00% per year), not downloaded
- **Alternatives:** download a rate from FRED (needs an API key).
- **Reason:** it keeps the app free of keys, so the offline run and the "no keys in the repository" check stay simple.
- **Limit:** the default 4% is our own round assumption, not data. The screen says it is the user's assumption. We tested how much it matters: with 0% the order of the three groups does not change (Sharpe 1.75, 1.56 and 1.43 for cycle group, comparison group and SPY).

## D7. Saved CSV files only; no live download in the app
- **Alternatives:** download prices from Yahoo (yfinance) and fall back to saved files.
- **Reason:** we pulled the data once on 2026-10-10 and saved it, so the app always runs offline and gives the same numbers every time.
- **Known gap:** the assignment describes `data.py` as downloading and saving raw files with a fallback. Ours only reads saved files. We say this openly instead of hiding it.

## D8. No independent cross-check of prices
- **Alternatives:** compare each series with an official source on two dates, as the brief suggests.
- **Reason:** we asked the instructor and he said it is not required.
- **Limit:** prices rely on Yahoo Finance alone, so a Yahoo error would go unnoticed. This is stated in `DATA.md` (sections 4, 5 and 6) and in the note.

## D9. All calculations in `core.py`; the screen only calls functions
- **Alternatives:** compute on the screen in `app.py` (this was partly the case at the start).
- **Evidence:** the brief caps the methods mark if the screen code contains calculations. We moved the price summary, the group comparison, the money formatting and the geometry of the curved arrows into `core.py` and covered them with tests.

## D10. Structural failure-exposure view, not a price-impact model
- **Alternatives:** a beta- or correlation-based shock model ("if this stock falls 30%, the others fall by X").
- **Reason:** we rejected it because OpenAI and Anthropic have no prices and the result would look like a forecast. Instead, the user picks a company and sees which other companies are linked to it, in which direction money flows, and how many steps away they are.
- **Limit:** the view shows links and distance only. It does not estimate price impact and it does not claim that a failure would spread.

## D11. Historical co-movement on the selected company's worst days
- **Alternatives:** a beta-times-shock forecast or a regression model.
- **Reason:** both look like forecasts. We prefer a plain historical fact that can be checked by hand: on the 20 days when the selected stock fell the most, what did the other public stocks and SPY do on average? SPY is shown so the user can see how much of the move is the whole market.
- **Limit:** co-movement is not cause, and the market effect is included in the result. Companies without a stock price cannot be shown.

## D12. Amounts are shown per relationship and never added up
- **Alternatives:** add the amounts along a cycle to get a "size of the loop".
- **Evidence:** 12 of 30 relationships have no amount, one is in shares, some are "up to" ceilings, and one deal has a funded part smaller than the announced amount. A total would be misleading.
- **Result:** the cycle table lists the amount of each step in order, with "Not disclosed", "up to" or "in shares" where that applies, and the screen says amounts are not additive.

## AI advice we rejected or changed
- We rejected a shock/forecast model for the stress test (D10, D11).
- We did not accept the first curved-arrow design and asked for arcs that go over the lines, like the examples we showed.
- We asked for the cycle table to show amounts instead of company ids, and later without repeating the company names, and we checked the shown amounts against `relationships_final.csv`.
- We told the coding assistant to change only the files we named and to leave all other documents alone.
- We asked the assistant to correct a placement instruction it had given wrongly (a code block had to go above a line, not after it).

---

## Group Assignment 2 & Final v2 Decisions (2026-11-21)

### D8. Trading Rule Choice: Lagged SMA-100 Trend-Risk Filter on the AI Cycle Basket
- **Date:** 2026-11-21
- **Decision:** Test a 100-day Simple Moving Average trend filter on the Equal-Weight AI Cycle Basket (`AMD, MSFT, NVDA`), holding cash at 4.00% annual risk-free rate when below SMA, with strict 1-day lag (`position[t] = signal[t-1]`).
- **Alternatives considered:** (a) 20-day volatility targeting; (b) dynamic switching between Cycle Group and Comparison Group.
- **Reason:** The SMA trend filter directly tests whether an investor can mitigate the Cycle Group's higher volatility (35.6%) and deeper drawdown (-36.8%) discovered in Assignment 1, with transparent, auditable timing and a clean parameter plateau ($L \in \{60, 80, 100, 120, 140\}$).

### D9. Holdout Test Period Opened Once
- **Date:** 2026-11-21
- **Decision:** Froze `L = 100` days and `c = 10 bps` on Train (`2023-05-26` to `2024-12-31`) and Validation (`2025-01-02` to `2025-12-31`), then opened the final Test window (`2026-01-02` to `2026-10-08`) **once** via `scripts/generate_evidence.py` and logged all 5 train trials + 1 validation + 1 test run in `trials.csv`.

### D10. Quantitative Collapse Contagion Methodology (DebtRank + Balance-Sheet Channels)
- **Date:** 2026-11-21
- **Decision:** Built a transparent multi-channel stress-test model separating equity write-downs, contracted revenue/backlog loss, and off-balance-sheet SPV/guarantee calls, combined with DebtRank multi-hop damping ($\delta = 0.35$) and historical 5% tail co-movement, rather than an opaque black-box ML predictor.
- **Reason:** Private hubs (OpenAI, Anthropic) have no stock prices and undisclosed amounts cannot be imputed as zero; excluding superseded LOI rows (`R001/R002`) prevents double-counting.

### D11. Cluster Analysis Integration (KBTU Lecture Standards)
- **Date:** 2026-11-21
- **Decision:** Implemented both K-Means on 4 standardized risk features (`nstart=100` + 200-copy shuffled null band) and Ward hierarchical clustering on SPY market-removed residual correlations ($d = \sqrt{2(1-\rho)}$), plus ARI time-stability and OOS portfolio checks.
- **Evidence:** At $k=3$, real silhouette (0.4378) exceeds the 95th percentile null band (0.4084); effective independent bets = 5.98 out of 9 AI stocks; ARI across 2023–2024 vs. 2025–2026 = 0.164 (confirming cluster drift).

### D12. Live Yahoo Finance & Daily 10:00 Almaty News Agent with Rumor Guardrail
- **Date:** 2026-11-21
- **Decision:** Added optional live `yfinance` refresh with offline CSV fallback (defaulting to offline snapshot for reproducible grading) and a GitHub Actions daily workflow (`05:00 UTC` = `10:00 Asia/Almaty`) running `src/news_agent.py` with a 3-tier source credibility rubric that flags and excludes Tier-3 rumors from the base contagion shock.