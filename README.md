# 🔗 Circular Alpha v2 — AI Ecosystem Contagion, Cluster Lens & Rule Desk

**Quantitative Investment Management · KBTU Business School**  
*Team: Baikazakova Darina, Mamraimova Linara, Krymova Asselya*

Circular Alpha is a reproducible quantitative research platform and interactive Streamlit web application analyzing **circular financing in the AI infrastructure ecosystem** — combining network graph analysis, multi-channel collapse contagion stress-testing, K-Means & Ward residual cluster analysis, a look-ahead-free Rule Desk backtest (Group Assignment 2), live Yahoo Finance market data, and an automated **Daily 10:00 AM Almaty AI News Intelligence Agent**.

---

## Core Modules

1. **💥 Collapse Contagion Simulator (`src/contagion.py`)**
   - Models what happens across the AI supply and financing chain if any hub (e.g., **Anthropic**, **OpenAI**, **CoreWeave**, **NVIDIA**, **Broadcom**) suffers severe distress or collapse.
   - Decomposes Hop-1 exposure into **Equity Write-Downs**, **Contracted Revenue / Cloud & Chip Backlog at Risk**, and **Off-Balance-Sheet SPV / Debt Guarantee Calls**, scaled against SEC-reported Cash and EBITDA, plus **multi-hop DebtRank propagation** ($\delta$) and affected business segments.
   - Displays **Quantitative-Only** vs. **Quantitative + Daily News Sentiment** scenario estimates side by side.
2. **🧩 Cluster Lens (`src/clustering.py`)**
   - **K-Means Risk Map:** 4 z-score standardized risk features (CAGR, Volatility, Beta vs SPY, Max Drawdown), `nstart=100`, with Elbow and Average Silhouette validated against **200 column-shuffled null copies** (95th percentile noise band).
   - **Ward Hierarchical Clustering on Market-Removed Residuals:** Regresses daily returns on SPY, computes Mantegna (1999) distance $d = \sqrt{2(1 - \rho)}$, plots cluster-sorted correlation heatmaps, computes **Effective Number of Independent Bets**, **Adjusted Rand Index (ARI)** time-stability (2023–2024 vs. 2025–2026), and **out-of-sample portfolio comparison** (`1/N` vs. Inverse Volatility vs. Cluster Risk Parity).
3. **📐 Rule Lab & Evidence Pack (`src/backtest.py`, `evidence/`, `RULE_CARD.md`, `RESULT_CARD.md`)**
   - Strictly lagged `SMA-100` trend-risk filter on the Equal-Weight AI Cycle Basket (`AMD, MSFT, NVDA`) vs. Buy-and-Hold benchmark (`position[t] = signal[t-1]`, cash earns 4.00% risk-free rate).
   - Programmatically generates all GA2 evidence items **E1–E8** (`scripts/generate_evidence.py`), paired circular block bootstrap 95% CI, transaction-cost & break-even analysis, parameter-neighbour plateau check, chronological train/validation/test split (test opened once), and passing **canary look-ahead test**.
4. **📰 Daily AI News Intelligence Agent (`src/news_agent.py`, `.github/workflows/daily_news_agent.yml`)**
   - Scheduled automatically at **10:00 AM Asia/Almaty (`05:00 UTC`)** every day via GitHub Actions alongside a live Yahoo Finance price refresh (`src/live_data.py`).
   - Grades every source into **Tier 1 (Verified Primary/Institutional)**, **Tier 2 (Trade Press)**, or **Tier 3 (⚠️ Unverified / Rumor — flagged and excluded from base shock model)**.

---

## Quick Start (One Command)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all unit tests (including hand-computed toy test & look-ahead canary test)
pytest -v
# (or zero-dependency GA2/v2 test run: python3 -m unittest discover -s tests -p "test_ga2_and_v2.py" -v)

# 3. Regenerate the complete GA2 Evidence Pack (tables + PNG figures)
python scripts/generate_evidence.py

# 4. Launch the interactive web app locally
streamlit run app.py
```

## Deploying Live (Streamlit Community Cloud + Daily 10:00 Almaty Agent)
1. Push this repository to GitHub.
2. Enable **GitHub Actions** (read/write workflow permissions) so `.github/workflows/daily_news_agent.yml` runs automatically every day at **10:00 AM Asia/Almaty (`05:00 UTC`)** to refresh `data/news_feed.json` and `data/raw/*.csv`.
3. Connect the repository to [Streamlit Community Cloud](https://share.streamlit.io/) pointing to `app.py` for a live, shareable LinkedIn portfolio link.
4. In the Streamlit Cloud app settings, add a secret named `GEMINI_API_KEY`. Alpha-chan reads this server-side secret only; users do not provide API keys and no local process, tunnel, or laptop environment variable is required.

---

*Disclaimer: Academic quantitative research project for KBTU Business School. Contagion outputs are scenario stress-tests, not point price forecasts. Historical backtests do not guarantee future performance. Not investment advice.*
