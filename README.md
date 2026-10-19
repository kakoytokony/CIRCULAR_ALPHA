# Circular Alpha

**Circular Alpha** is a local Streamlit app that finds circular financial relationships (for example investor → AI company → chip supplier → investor) among selected AI-related companies. It compares the risk and return (Sharpe ratio) of the public stocks inside such loops with other public AI stocks and with SPY, using saved Yahoo Finance prices.

A detected loop is a pattern in public documents, not proof of wrongdoing. All results are historical, not a forecast and not investment advice.


## Install and run (one command)

Windows PowerShell (if `pip` or `python` is not recognized, use the full path to your `python.exe` instead of `python`):

```
python -m pip install -r requirements.txt; python -m streamlit run app.py
```

macOS / Linux / Git Bash:

```
pip install -r requirements.txt && streamlit run app.py
```

The app opens at http://localhost:8501. It works offline: it reads the saved CSV files, downloads nothing and needs no API key.

## Run the tests

```
python -m pytest
```

All tests should pass. At least three of them check answers calculated by hand; the calculation is written in a comment above each test. Test data are **synthetic** (made-up numbers) and are used only in tests. The app itself uses only the real saved data.

## What the app shows

1. **Circular relationship explorer:** directed loops found with NetworkX, a switch for confirmed-only relationships, and the amounts along each loop (shown per step, never added up).
2. **Company relationship network:** curved arrows from payer to receiver; hover a line for the relationship type, evidence status, amount and source link.
3. **Headline:** Sharpe ratio of an equal-weight basket of the public companies inside confirmed loops, compared with a basket of other public AI stocks and with SPY. The user can change the risk-free rate, the dates and the comparison group.
4. **Market performance and risk:** total return, annualized volatility, beta, maximum drawdown and a growth-of-$100 chart.
5. **Company and financial-data explorer, data-quality report and a limitations panel.**

## Project structure

| File | Role |
|---|---|
| `app.py` | Screen only: shows inputs, tables and charts. It calls functions from `core.py` and does no calculations. |
| `core.py` | All calculations as plain functions (returns, volatility, Sharpe, drawdown, beta, baskets, loop detection, chart geometry, money formatting). |
| `data.py` | Reads the saved CSV files. It does not download anything (a known gap, see `DECISIONS.md` D7). |
| `tests/test_core.py` | Automated tests for `core.py`. |
| `requirements.txt` | Python packages. |

## Data

Saved CSV files in the project folder (or `data/raw/`): one daily adjusted-price file per ticker (Yahoo Finance, USD, pulled 2026-10-10), `companies_clean.csv`, `relationships_final.csv`, `financials.csv` and `data_dictionary_synced.csv`. Sources, dates, problems found and limits are in `DATA.md`.

## Documents

| File | Content |
|---|---|
| `PRODUCT_CARD.md` | Who the product is for, what it does and does not do |
| `DATA.md` | Data sources, checks, problems and limits |
| `NOTE.pdf` | Analysis note (question, data, method, results, limits) with hand calculations in the annex |
| `DECISIONS.md` | Major choices, alternatives, evidence and AI advice accepted or rejected |
| `AI_USE.md` | Every use of AI: tool, place in the code, prompt, what we changed, how we verified |
| `CONTRIBUTIONS.md` | Who did what |

## What is reused

- **Reused (starting code):** No starter code from Week 4 X-Ray was used; the project was built from our own earlier version.

## API keys and secrets

None are needed and none are stored. `.env` is listed in `.gitignore` in case keys are added later.

## Known limits

Only 3 of our 9 public stocks (AMD, MSFT, NVDA) sit in confirmed loops, because every loop also passes through the private companies OpenAI or Anthropic, which have no stock price. So the comparison is descriptive and cannot prove cause.

## What the app shows (continued)

- **Structural stress test:** select a company in the network section to see reachable companies by hop distance and link direction if that company fails; this is not a price forecast.

- **Historical co-movement:** for a selected public company, compare how other stocks moved on its worst historical days; this is not proof of cause or a forecast.
