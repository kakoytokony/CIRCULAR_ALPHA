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

---

## v3.0 — Bubble Radar hedge/short rule (frozen 2026-11-27)

> Implemented in `src/bubble_monitor.py`, `src/credit_signals.py`, `src/historical_analog.py`; tested in `tests/test_methodology_v3.py` and `tests/test_extended_data.py`. Methodology: `METHODOLOGY_AND_SPEC.md` §4.1 and §4.3. **Team review required (RED field):** be able to explain V1–V6 without a computer.

**V1. Question.** Do objective equity-exuberance and credit-stress signals say that the AI-infrastructure trade is in a fragile regime, and if so, which names should be shorted/avoided, hedged, or held?

**V2. Signals at close of t (data ≤ t only).**
1. *BSADFₜ* (Phillips, Shi & Yu, 2015) on log levels of the equal-weight AI Core index (12 names, `data/raw/volumes_core.csv`): right-tailed ADF(K = 0) supremum over backward windows with minimum window r₀ = 0.01 + 1.8/√T. Critical values CV₉₀ₜ, CV₉₅ₜ from 200 fixed-seed (20151043) Gaussian random walks of the same length.
2. *Crash-risk score Sₜ* (Greenwood, Shleifer & You, 2019) = 0.25·Z_vol + 0.25·Z_turn + 0.25·Z_acc + 0.25·Z_cred, each a z-score vs. the trailing 504 days (min 126): 63-day realised vol; 21d/252d dollar volume; last-126d minus prior-126d log return; ICE BofA US HY OAS (FRED `BAMLH0A0HYM2`, Gilchrist & Zakrajšek, 2012). P(crash) = σ(θ₀ + θ₁Sₜ) with θ₀ = 1.936, θ₁ = 0.840 fitted on Nasdaq Composite 1999-07-02 → 2001-12-28 (n = 626, complete cases; crash = ≥40% fall within 504 days). The intercept is upward-biased by the bust-centred sample; the app therefore also shows the odds multiplier exp(θ₁Sₜ).
3. *Five-point checklist* (J.P. Morgan Oct 29 / Nov 4 2026 levels from `data/circular_alpha_research_pack.xlsx`): HY HPC spread RED > 600 / AMBER 450–600 bps; hyperscaler 10y+ basis vs JULI RED > 75 / AMBER 45–75 bps; CoreWeave 2031s RED ≥ 1,000 bps; AI Core BSADFₜ > CV₉₅ₜ; equity-supply + underwriting friction (Ofek & Richardson, 2003) RED when all three sourced events are active (Anthropic S-1, Meta $25B SPO, Volta 26% underwritten).

**V3. Position in t + 1.** `signal[t] = stat[t-1]` for every time-series statistic (asserted by canary tests).
- **Short / Avoid:** `CRWV`, `APLD`, `GLXY` while the HY HPC or CRWV trigger is RED.
- **Tactical underweight / hedge (puts or pairs, never naked shorts):** `NVDA`, `AVGO` only when a credit trigger **and** the supply/underwriting trigger are RED.
- **Defensive / relative long:** `MSFT`, `AMZN` (lowest tracked debt among hyperscalers: $44B / $120B J.P. Morgan printed totals).
- **Avoid as long:** `ORCL` ($134B), `META` ($130B) tracked debt.

**V4. Switch-off condition (decided before results).** The whole short/hedge stance turns **OFF** only when **both** (a) HY HPC spread < 350 bps **and** (b) AI Core BSADFₜ < CV₉₀ₜ. Status on 2026-11-27: (a) false (503 bps), (b) true → rule **ON**.

**V5. Costs.** 10 bps per side transaction cost (same as GA2); borrow fee assumptions 0.5%/yr easy-to-borrow and 10%/yr for neoclouds over a 6-month horizon (user-editable; not prime-broker quotes). Neocloud shorts also carry squeeze risk that the cost line does not capture.

**V6. Freeze tag.** Parameters above are frozen as `v3.0` (trial IDs `V3-*` in `trials.csv`). Any change becomes a new trial row; no row is deleted. The v3 rule has **no out-of-sample performance evaluation yet** — the credit levels are a single Oct–Nov 2026 snapshot, so the rule is a monitored regime classifier, not a backtested alpha claim.
