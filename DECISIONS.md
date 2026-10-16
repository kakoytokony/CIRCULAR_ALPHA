# DECISIONS.md - Circular Alpha

Decision record: each major choice, the alternatives, the evidence, and the AI advice accepted or rejected.
[FILL IN: check that this layout matches the format required in syllabus section 9.2. It follows the list in the assignment brief.]

## D1. Track: 8 (own idea), not one of the seven standard tracks
- **Alternatives:** Track 4 (market exposure explorer) or Track 3 (rebalancing simulator), which fit the same data.
- **Evidence:** the project already existed with a company network, relationships and prices. A standard track would have discarded the network part.
- **Condition:** track 8 needs the instructor's written approval. [FILL IN: date and where it is stored.]
- **AI advice:** accepted (AI pointed out that track 8 needs written approval).

## D2. Headline number: Sharpe ratio of an equal-weight basket of public companies that sit in confirmed cycles
- **Alternatives:** (a) number of cycles as the headline; (b) a "circularity score" regressed on returns, as the first project description promised.
- **Evidence:** only three public stocks (AMD, MSFT, NVDA) are in confirmed cycles, so no statistical test of "cycles -> returns" is possible. The cycle count is inflated by two private hubs (5 cycles with confirmed links, 26 with all links), so it is not a score of circularity.
- **Result:** descriptive comparison of risk and return against other public AI stocks and SPY. The screen and the notes say it is not a forecast and not proof of cause.
- **AI advice:** accepted (drop the claim of a predictive signal).

## D3. Only `confirmed` relationships by default (switch on screen)
- **Alternatives:** use all 30 relationships.
- **Evidence:** by the web domain of the source URL, all 18 confirmed rows cite company, SEC or investor-relations pages. The 9 rows citing press or Wikipedia are all `provisional`.
- **Trade-off:** fewer cycles; the switch lets the user see the effect (cycle group becomes AMD, AMZN, CRWV, MSFT, NVDA).

## D4. Equal weights, rebalanced daily, for both baskets
- **Alternatives:** buy-and-hold basket; market-cap weights.
- **Reason:** simplest rule with no extra data, and it uses the weights and rebalancing topic of the course.
- **Limit:** daily rebalancing ignores trading costs.

## D5. All series compared on one common window; CRWV excluded from the comparison group by default
- **Alternatives:** each series from its own first date (the older chart does this).
- **Evidence:** CRWV has prices only from 2025-03-28 (385 dates against 945 for the others). Including it cuts the common window for every series and drops 560 dates. The screen shows how many dates were dropped and which ticker limits the window.

## D6. Risk-free rate typed by the user (default 4.00% per year), not downloaded
- **Alternatives:** download a rate from FRED (needs an API key).
- **Reason:** keeps the app free of keys, so the offline run and the "no keys in the repository" check are simple.
- **Limit:** the default 4% is our assumption, not data. [FILL IN: say where the 4% comes from, or state that it is only a placeholder.] The screen says it is the user's assumption.

## D7. Saved CSV files only; no live download in the app
- **Alternatives:** download prices from Yahoo (yfinance) and fall back to saved files.
- **Reason:** the data were pulled once on 2026-10-10 and saved, so the app always runs offline and gives reproducible numbers.
- **Known gap:** the assignment describes `data.py` as downloading and saving raw files with a fallback. Ours only reads saved files. We state this openly instead of hiding it.

## D8. No independent cross-check of prices
- **Alternatives:** compare each series with an official source on two dates, as the brief asks.
- **Reason:** the instructor said it is not required. [FILL IN: date and how the instructor said it.]
- **Limit:** prices rely on Yahoo Finance alone. Stated in DATA.md sections 4, 5 and 6.

## D9. All calculations in `core.py`; the screen only calls functions
- **Alternatives:** compute on the screen in `app.py` (this was partly the case at the start).
- **Evidence:** the brief caps the methods score if the screen code contains calculations. We moved the price summary and the group comparison into `core.py` and covered them with tests.

## AI advice rejected
[FILL IN: write any AI suggestion you did not follow and why. If there was none, write "none".]
