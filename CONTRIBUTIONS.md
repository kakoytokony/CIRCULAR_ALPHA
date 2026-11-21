# CONTRIBUTIONS.md - Circular Alpha

Project: Circular Alpha (QIM Group Assignment 1, track 8)

| Member | Parts done |
|---|---|
| Mamraimova Linara | Conducted market research according to the latest news related to investments in the AI sector, gathered datasets |
| Baikazakova Darina | Completed quantitative and technical implementation of the project including fin calculations, backend logic, frontend development, and integration of the analytical complements into the application |
| Krymova Asselya | Tested the application's functionality, checked that the system worked as expected |


---

## Group Assignment 2 & Final v2 Responsibilities (Review & Defence Checklist)

| Member | Built / Expanded | Checked & Verified | Ready to Explain Without a Computer (Quiz 2 & Final) |
|---|---|---|---|
| **Baikazakova Darina** | Quantitative backtest engine (`src/backtest.py`), Cluster Lens (`src/clustering.py`), multi-tab Streamlit app (`app.py`) | Hand-computed toy backtest, canary look-ahead test, `evidence/` tables vs. `RESULT_CARD.md` | Lagged position timing (`position[t] = signal[t-1]`), Sharpe SE & block bootstrap CI, Ward residual clustering & noise band |
| **Mamraimova Linara** | Oct–Nov 2026 dataset expansion (`C013–C016`, `R031–R042` from J.P. Morgan/WSJ/Bloomberg), Contagion business-line mapping, Daily News Agent credibility rubric | Verified source URLs, evidence status (`confirmed` vs `provisional`), superseded LOI (`R001/R002`) exclusion, rumor flagging | Circular financing channels (equity, backlog, SPV guarantees), Anthropic–Broadcom $63B deal, Blue Wave DC implications |
| **Krymova Asselya** | Test suite (`tests/test_core.py`, `tests/test_ga2_and_v2.py`), evidence generation pipeline (`scripts/generate_evidence.py`), live Yahoo Finance & GitHub Actions workflow | Executed one-command tests, verified cost sensitivity / break-even cost, parameter-neighbour plateau, and `trials.csv` completeness | Canary test proof of no look-ahead, transaction cost impact, trial-ledger honesty & expected max noise Sharpe |
