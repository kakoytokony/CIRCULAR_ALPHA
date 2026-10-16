# Circular Alpha

Circular Alpha is a local research tool that finds directed circular financial relationships (investments, purchases, guarantees) between selected AI-related companies and shows market risk context (return, volatility, beta vs SPY) for the public ones.
It is descriptive only: a detected cycle is a pattern for review, not proof of wrongdoing, not a forecast and not investment advice.

## Install and run

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder (the one that contains app.py).
3. Install the packages:

    pip install -r requirements.txt

4. Start the app:

    streamlit run app.py

The app opens in your browser. It works offline: it reads the saved CSV files in this folder and says so on screen.

## Run the tests

    python -m pytest

## Data

- companies_clean.csv, relationships_final.csv, financials.csv, data_dictionary_synced.csv: hand-collected from public sources; every relationship has a source link and an evidence status (confirmed / provisional).
- SPY.csv, NVDA.csv, MSFT.csv, etc.: daily adjusted close prices in USD from Yahoo Finance, pulled on 2026-10-10.

## Code structure

- data.py: loads the saved CSV files (no calculations, no internet).
- core.py: all calculations (pure functions, no screen code).
- app.py: the Streamlit screen; it only calls functions from core.py.
- test_core.py: automated tests with known answers.

## API keys

This version needs no API key
