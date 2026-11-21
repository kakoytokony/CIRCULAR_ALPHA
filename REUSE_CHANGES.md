# REUSE_CHANGES.md — Circular Alpha (Assignment 1 → Assignment 2 / Final v2)

## Reused from Assignment 1
- **Data loader (`data.py`):** Reused local offline-capable CSV loader for `companies_clean.csv`, `relationships_final.csv`, `financials.csv`, `data_dictionary_synced.csv`, and daily adjusted price snapshots (`data/raw/*.csv`).
- **Core calculations (`core.py`):** Reused all Assignment 1 functions (`validate_data`, `prepare_relationships`, `calculate_cycle_metrics`, `describe_cycle_amounts`, `calculate_market_metrics`, `compare_groups`, `growth_of_base`, `worst_day_comovement`, `sharpe_ratio`, `max_drawdown`, `beta_vs_benchmark`).
- **Network explorer & tests:** Preserved the curved-arrow directed relationship network in Tab 4 of `app.py` and kept all original Assignment 1 tests in `tests/test_core.py`.
- **All 18 confirmed relationships (`R001–R030`):** Kept intact so switching `Confirmed primary links only = On` reproduces Assignment 1 baseline results identically.

## New in Assignment 2 / Final v2
- **Unified interactive contagion graph, plain-language storytelling, and EN/RU language toggle (`app.py`):** Added a single interactive graph with arrowheads, investor-friendly explanations, and bilingual UI controls.
- **The Rule Desk (`src/backtest.py`, `scripts/generate_evidence.py`, `evidence/`, `RULE_CARD.md`, `RESULT_CARD.md`, `trials.csv`):** Added strictly lagged SMA trend-risk filter backtest, paired circular block bootstrap CI, cost sensitivity + break-even cost, subperiod/start-date checks, parameter-neighbour plateau checks, chronological train/validation/test split, downside risk (worst month & 5% VaR), and canary look-ahead test.
- **Collapse Contagion Simulator (`src/contagion.py`):** Upgraded Assignment 1's structural hop counter into a quantitative multi-channel stress-test engine decomposing equity write-downs, contracted revenue/backlog loss, off-balance-sheet SPV/debt guarantee calls, multi-hop DebtRank propagation, balance-sheet ratios (% of Cash / EBITDA), affected business segments, and Quantitative vs. News-Sentiment-Adjusted scenario drawdowns.
- **Cluster Lens (`src/clustering.py`):** Added full KBTU QIM Cluster Analysis module (z-score standardization, multi-start K-Means `nstart=100`, 200-copy column-shuffled null silhouette band, Ward hierarchical clustering on SPY market-removed residual correlations, effective number of independent bets, ARI time-stability check, and out-of-sample portfolio comparison).
- **Expanded Oct–Nov 2026 Ecosystem Data (`companies_clean.csv`, `relationships_final.csv`):** Added 4 new entities (`C013` Fluidstack, `C014` Applied Digital, `C015` Galaxy Digital, `C016` Blackstone) and 12 provisional Oct–Nov 2026 relationships (`R031–R042`) from J.P. Morgan Bond Matrix 3.0, WSJ, and Bloomberg (including Broadcom–Anthropic's $63B SPV, Chipmakers–Meta's $5B cornerstone stake, and Alphabet's Fluidstack/CoreWeave links).
- **Live Yahoo Finance + Daily 10:00 Almaty AI News Agent (`src/live_data.py`, `src/news_agent.py`, `data/news_feed.json`, `.github/workflows/daily_news_agent.yml`):** Added optional live Yahoo Finance price refresh (with offline CSV fallback) and an automated daily 10:00 Asia/Almaty (`05:00 UTC`) news-intelligence agent with a 3-tier credibility & rumor filter that feeds verified sentiment into the Contagion Simulator.
- **GA2 & v2 Test Suite (`tests/test_ga2_and_v2.py`):** Added hand-computed toy backtest, canary look-ahead test, known-answer cost/drawdown tests, clustering tests, contagion tests, and news-credibility tests runnable in one command (`python3 -m unittest discover -s tests` or `pytest`).

## Changed after feedback
- **From single-page MVP to 5-tab analytical product (`app.py`):** Reorganized the interface into dedicated English-language tabs so each workflow (Contagion, Clustering, Rule Lab, Network, Daily News) is clear, scannable, and portfolio-ready.
- **Alphabet zero-link gap resolved:** Added documented provisional Alphabet links (`R033`, `R034`, `R039`) from Oct 2026 J.P. Morgan and WSJ reporting.

## Dropped
- **None:** No Assignment 1 logic or confirmed data was discarded; all prior outputs remain reproducible under the `Confirmed primary links only` and `Offline CSV snapshot` toggles.
