# RULE_CARD.md — Circular Alpha (Group Assignment 2: The Rule Desk)

> **Team Review Notice (AI Policy — RED Fields):** Fields R3–R5 below are pre-aligned with `src/backtest.py` for technical accuracy. Before Gate G2 / Quiz 2 / Final Defence, every team member must review, verify, and be able to explain R3–R5 without a computer.

**R1. Question**
If an investor holding the Equal-Weight AI Cycle Basket (`AMD, MSFT, NVDA` — the three public companies inside confirmed circular-financing loops from Assignment 1) had applied a systematic daily Simple Moving Average (SMA) trend-risk filter, switching to cash at the risk-free rate during downtrends, would they have improved risk-adjusted returns or reduced drawdowns net of transaction costs, and how sure are we?

**R2. Universe and data**
- **Universe:** Equal-weight daily-rebalanced basket of `AMD`, `MSFT`, and `NVDA` (public confirmed-cycle members from Assignment 1).
- **Source:** Yahoo Finance daily adjusted closing prices (USD), saved locally in `data/raw/*.csv` (reproducible snapshot pulled 2026-10-10; live refresh optional via `src/live_data.py`).
- **Frequency & Dates:** Daily trading days, `2023-01-03` to `2026-10-08` (945 price dates; post-100-day warm-up evaluation window `2023-05-26` to `2026-10-08`, 845 return observations).
- **Chronological Split:** Train `2023-05-26` to `2024-12-31` (402 days), Validation `2025-01-02` to `2025-12-31` (250 days), Test `2026-01-02` to `2026-10-08` (193 days, opened once).

**R3. Signal (formula at close of t)**
Let $P_t$ be the cumulative wealth index of the daily-rebalanced equal-weight basket (`AMD, MSFT, NVDA`, starting at 100 on `2023-01-03`), and let $\text{SMA}_L(t) = \frac{1}{L}\sum_{k=0}^{L-1} P_{t-k}$ be its $L$-day simple moving average computed strictly from closing prices up to and including day $t$.
$$\text{signal}_t = \begin{cases} 1 & \text{if } P_t > \text{SMA}_L(t) \\ 0 & \text{otherwise} \end{cases}$$

**R4. Position held in t + 1 for each signal value (including the "out" position)**
- $\text{position}_{t+1} = \text{signal}_t$.
- When $\text{position}_{t+1} = 1$: hold 100% of the equal-weight AI Cycle Basket during day $t+1$.
- When $\text{position}_{t+1} = 0$ ("out" position): hold 100% cash earning the daily risk-free rate $r_{f,\text{daily}} = \frac{4.00\%}{252}$.
- One-way transaction cost $c$ (base case $10\text{ bps} = 0.0010$) is deducted whenever $|\text{position}_{t+1} - \text{position}_t| > 0$.

**R5. Timing (information → decision → return)**
1. **Information:** Closing adjusted prices on day $t$ become known at market close of day $t$.
2. **Decision:** $\text{SMA}_L(t)$ and $\text{signal}_t$ are computed at close of day $t$ using only prices up to $t$.
3. **Return:** The resulting $\text{position}_{t+1}$ is held over day $t+1$ and earns the return from close of $t$ to close of $t+1$ (no look-ahead; verified by the canary test in `tests/test_ga2_and_v2.py`).

**R6. Benchmark and why**
Buy-and-hold (100% always invested) in the exact same Equal-Weight AI Cycle Basket (`AMD, MSFT, NVDA`) over the exact same post-warm-up window (`2023-05-26` to `2026-10-08`). Using the same basket isolates the exact effect of the trend-risk timing rule rather than stock selection.

**R7. Parameters and ranges allowed**
- Lookback window $L \in \{60, 80, 100, 120, 140\}$ trading days (frozen selection: $L = 100$ trading days, ~5 months).
- Transaction cost grid: $c \in \{0, 10, 25, 50\}$ bps per unit of position change (base case: $10$ bps).
- Annual risk-free rate: $4.00\%$ (user-adjustable in Rule Lab).

**R8. When we drop the rule (decided before results)**
We drop the rule as a return-enhancing strategy if: (a) the 95% paired circular block bootstrap confidence interval for $(\text{Sharpe}_{\text{rule}} - \text{Sharpe}_{\text{benchmark}})$ includes zero, OR (b) net advantage disappears below $10$ bps per switch, OR (c) performance is a narrow single-parameter spike rather than a plateau across $L \in [60, 140]$. Even if dropped for alpha, we note honestly whether it reduces volatility and drawdown as a risk-budgeting tool.
