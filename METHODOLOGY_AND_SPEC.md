# CIRCULAR ALPHA v3: Academic Methodology, Literature Review & AI Implementation Specification

> **Document Purpose:** This file serves simultaneously as (1) the formal academic methodology and literature review for the **Circular Alpha** final dossier and defense, and (2) an unambiguous engineering specification for an AI coding assistant implementing these modules into `CIRCULAR_ALPHA-main`.
>
> **Zero-Fabrication Mandate:** Every academic reference, mathematical formula, historical 2000–2002 statistic, and October–November 2026 market figure in this document is verified against peer-reviewed literature, SEC filings, index data, or the attached J.P. Morgan / WSJ / Bloomberg credit research brief. **Do not invent, extrapolate, or hallucinate any additional statistics, citations, or fake data series during implementation.**

---

## 0. CRITICAL GUARDRAILS FOR THE AI CODING ASSISTANT

Before writing a single line of code, you **MUST** obey the following strict repository rules:

1. **DO NOT TOUCH ALPHA-CHAN (MASCOT):**
   - **Read-only files:** `src/mascot.py`, `assets/mascot.png`, and every Alpha-chan rendering call, dialogue hook, or UI container inside `app.py`.
   - Do not rename, restyle, reposition, or remove Alpha-chan.
2. **DO NOT ALTER CURRENT DESIGN, THEME, OR EXISTING TABS:**
   - **Read-only styling:** `.streamlit/config.toml` and all existing custom CSS blocks, fonts, color tokens, and layout containers in `app.py`.
   - All newly added tabs and charts **must reuse the existing Streamlit theme and Plotly styling helpers** already present in `app.py`.
   - Do not delete or rewrite existing tabs, existing charts, or existing evidence tables (`evidence/e1_headline.csv` through `evidence/e8_downside.csv`).
3. **PRESERVE BACKTEST CORE SEPARATION & ONE-COMMAND TESTS:**
   - **Do not break** `core.py`, `src/backtest.py`, `src/clustering.py`, `tests/test_core.py`, or `tests/test_ga2_and_v2.py`.
   - All new econometric and forecasting logic must live in **new, self-contained modules** under `src/` and be validated by a new test suite `tests/test_methodology_v3.py`. Running `pytest` must pass 100% of old and new tests with a single command.
4. **DATA TRACEABILITY (EXAM RUBRIC RULE):**
   - Per course grading rules (*"A number that cannot be traced to code and data earns nothing"*), every metric rendered on the dashboard must be computed dynamically from `data/raw/*.csv`, `data/raw/relationships_final.csv`, and `data/raw/financials.csv`, **or** loaded from an explicitly versioned, source-cited constants module (`src/historical_analog.py` / `src/credit_signals.py`) with exact source attribution displayed in the UI caption.

---

## 1. Executive Summary & Architectural Integration Plan

### 1.1 Addressing Professor & Rubric Feedback
The existing `CIRCULAR_ALPHA-main` platform tracks 12 core AI/infrastructure equities (`NVDA`, `AMD`, `AVGO`, `MSFT`, `AMZN`, `GOOGL`, `META`, `ORCL`, `CRWV`, `APLD`, `GLXY`, `BX`) plus `SPY`, using Ward hierarchical clustering (`src/clustering.py`), bilateral contagion (`src/contagion.py`), and a separated backtest core (`core.py`).

To elevate the project to full research-grade rigor, this specification adds **four missing pillars**:
1. **Quantitative Bubble Detection & Crash Probability** (answering: *What is the probability the bubble bursts, and what are the objective triggers?*).
2. **Extended Multi-Tier Domino Contagion** (propagating shocks beyond the 12 core companies to Tech ETFs, S&P 500, Investment Grade / High Yield credit, Power/Utilities, and Emerging Market debt).
3. **Historical Calibration (2000–2002 Dot-Com / Telecom Vendor-Financing Bust) & 3-Scenario S&P 500 Forecast** (building a transparent, auditable market stress-test forecast grounded in historical vendor-financing parallels).
4. **Actionable Positioning Matrix & Regime Switch-Off Rules** (mapping signals to *who to short / who to avoid / who to hold*, with explicit cost, look-ahead, and out-of-sample freeze discipline).

### 1.2 Recommended UI Integration Strategy (Why New Tabs Are Optimal)
**Recommendation:** Add **3 new dedicated tabs** to the main navigation in `app.py` rather than modifying existing tabs.
- **Why:** Existing tabs remain 100% intact for grading continuity, Alpha-chan's current state/placement is undisturbed, and a first-time visitor (or examiner) can immediately click through the exact logical sequence requested by the professor:
  - **New Tab A — `Bubble Radar & Triggers`**: Real-time GSADF exuberance indicator, Greenwood–Shleifer–You crash probability gauge, Credit Spread / Issuance early-warning checklist, and Rule Switch-Off status.
  - **New Tab B — `Domino & S&P 500 Forecast`**: `DebtRank` systemic centrality on `relationships_final.csv`, 4-Tier Diebold–Yilmaz spillover matrix, and the interactive 3-Scenario S&P 500 (`SPY`) & Tech (`QQQ` / `SOXX`) stress-test forecaster.
  - **New Tab C — `2000–2002 Analog & Playbook`**: Side-by-side structural comparison of the 2000–2002 Telecom Vendor-Financing Bust vs. 2026 AI Circular Financing, plus the Actionable Short / Avoid / Relative-Long Positioning Matrix.

---

## 2. Literature Review & Theoretical Foundations

Every modeling choice in Circular Alpha v3 is anchored in peer-reviewed financial economics literature:

### 2.1 Econometric Bubble Detection & Crash Probability
1. **Recursive Right-Tailed Unit-Root Tests (GSADF):**
   - **Phillips, P. C. B., Shi, S., & Yu, J. (2015).** *"Testing for Multiple Bubbles: Historical Episodes of Exuberance and Collapse in the S&P 500."* **International Economic Review**, 56(4), 1043–1078. DOI: `10.1111/iere.12132`. (Extending Phillips, Wu, & Yu, 2011, *IER*, 52(1), 201–226).
   - **Contribution:** Develops the Generalized Supremum Augmented Dickey-Fuller (GSADF) test and Backward SADF (BSADF) date-stamping strategy using flexible rolling windows. Unlike standard left-tailed stationarity tests, GSADF tests the null hypothesis of a random walk with asymptotically negligible drift against a mildly explosive alternative, consistently dating origination and termination of multiple bubbles in real time.
2. **Forecasting Industry Crashes ("Bubbles for Fama"):**
   - **Greenwood, R., Shleifer, A., & You, Y. (2019).** *"Bubbles for Fama."* **Journal of Financial Economics**, 131(1), 20–43. DOI: `10.1016/j.jfineco.2018.09.004`.
   - **Contribution:** Examining U.S. industry portfolios (1926–2014) and international sectors (1985–2014), the authors show that a sharp two-year price run-up ($\ge 100\%$ raw and net of market) does not automatically imply negative average forward returns, **but substantially raises the probability of a crash** (defined as a $\ge 40\%$ drawdown within 24 months). Crucially, four observable attributes of the run-up forecast which booms crash: **(1) elevated volatility, (2) high share turnover, (3) heavy equity/debt issuance, and (4) accelerating (convex) price paths**.
3. **Credit Spreads as Macro & Financial Leading Indicators:**
   - **Gilchrist, S., & Zakrajšek, E. (2012).** *"Credit Spreads and Business Cycle Fluctuations."* **American Economic Review**, 102(4), 1692–1720. DOI: `10.1257/aer.102.4.1692`.
   - **Contribution:** Constructs the GZ credit spread index from micro-level corporate bond data and decomposes it into expected default risk and the **Excess Bond Premium (EBP)**. An increase in EBP reflects a contraction in the financial sector's risk-bearing capacity, reliably preceding credit supply tightening and broad macroeconomic downturns.

### 2.2 Financial Networks, Circularity & Cross-Market Contagion
4. **Network Phase Transitions (Why Circularity Turns Fragile):**
   - **Acemoglu, D., Ozdaglar, A., & Tahbaz-Salehi, A. (2015).** *"Systemic Risk and Stability in Financial Networks."* **American Economic Review**, 105(2), 564–608. DOI: `10.1257/aer.20130456`.
   - **Contribution:** Proves that densely interconnected financial networks exhibit a **phase transition**: for small shocks, interconnections diversify and absorb losses; once a shock exceeds a critical threshold, the exact same dense interconnections (guarantees, cross-holdings, vendor credit) propagate systemic cascades across the entire network.
5. **Systemic Impact & "Too Central to Fail" (`DebtRank`):**
   - **Battiston, S., Puliga, M., Kaushik, R., Tasca, P., & Caldarelli, G. (2012).** *"DebtRank: Too Central to Fail? Financial Networks, the FED and Systemic Risk."* **Scientific Reports**, 2, 541. DOI: `10.1038/srep00541`.
   - **Contribution:** Introduces `DebtRank`, a recursive centrality metric measuring the fraction of total economic value in a directed financial network that becomes distressed following an initial shock to node $i$. It captures cascading distress *before* full technical default occurs.
6. **Cross-Market Volatility & Return Spillovers (VAR FEVD & Granger Networks):**
   - **Diebold, F. X., & Yilmaz, K. (2014).** *"On the network topology of variance decompositions: Measuring the connectedness of financial firms."* **Journal of Econometrics**, 182(1), 119–134. DOI: `10.1016/j.jeconom.2014.04.012`.
   - **Billio, M., Getmansky, M., Lo, A. W., & Pelizzon, L. (2012).** *"Econometric measures of connectedness and systemic risk in the finance and insurance sectors."* **Journal of Financial Economics**, 104(3), 535–559. DOI: `10.1016/j.jfineco.2011.12.010`.
   - **Contribution:** Diebold & Yilmaz (2014) use Generalized Forecast Error Variance Decompositions (GFEVD) from a Vector Autoregression (VAR) to build a directed, weighted spillover matrix (`From`, `To`, and `Net` connectedness) across assets and sectors. Billio et al. (2012) complement this with Principal Components Analysis (PCA) absorption ratios and pairwise Granger-causality networks to detect rising systemic fragility prior to crises.

### 2.3 Historical Bubble Burst Mechanics (2000–2002 Parallel)
7. **Equity Supply Shock & Bursting Mechanics:**
   - **Ofek, E., & Richardson, M. (2003).** *"DotCom Mania: The Rise and Fall of Internet Stock Prices."* **Journal of Finance**, 58(3), 1113–1137. DOI: `10.1111/1540-6261.00560`.
   - **Contribution:** Demonstrates that bubbles persist under heterogeneous investor beliefs and binding short-sale constraints, and **burst when equity supply expands rapidly** (via IPO lockup expirations, insider selling, and secondary offerings), documenting average post-lockup long-run excess returns as low as $-33\%$ for Internet stocks.

---

## 3. Empirical Calibration: 2000–2002 Parallel vs. Late-2026 AI Credit Reality

### 3.1 Verified 2000–2002 Dot-Com / Telecom Vendor-Financing Benchmarks
Use only these verified figures in `src/historical_analog.py`:

| Metric / Event | Verified Historical Figure | Source |
| :--- | :--- | :--- |
| **Nasdaq Composite Drawdown** | **$-77.9\%$** (5,048.62 on Mar 10, 2000 $\to$ 1,114.11 on Oct 9, 2002; 31 months; recovered Apr 23, 2015) | Nasdaq Daily Closing Data |
| **Nasdaq-100 Drawdown** | **$-82.9\%$** (4,704.73 on Mar 27, 2000 $\to$ 804.64 on Oct 7, 2002; recovered Nov 3, 2015) | Nasdaq Daily Closing Data |
| **S&P 500 Drawdown** | **$-49.1\%$** (1,527.46 on Mar 24, 2000 $\to$ 776.76 on Oct 9, 2002; 31 months; recovered May 30, 2007 — 7.2 years) | S&P Dow Jones Indices |
| **Implied S&P-to-Nasdaq Elasticity** | $\approx 0.63\times$ (Peak-to-trough S&P 500 drawdown relative to Nasdaq Composite) | Derived from closing levels above |
| **Lucent Vendor Financing & Fraud** | $\$8.1\text{B}$ customer financing commitments (as of Sep 30, 2000); $\$2.2\text{B}$ bad-debt provisions in FY2001; SEC charged Lucent with **$\$1.148\text{B}$** improper FY2000 revenue recognition & **$\$470\text{M}$** pre-tax income overstatement ($\$25\text{M}$ penalty) | SEC Press Release 2004-67 (May 17, 2004); SEC Litigation Releases |
| **Telecom Customer Failures** | 47 Competitive Local Exchange Carriers (CLECs) filed for bankruptcy (2000–2003), including WinStar (Apr 2001), Covad, and NorthPoint | Industry / Bankruptcy Records |

### 3.2 Verified October–November 2026 AI Credit & Circularity Signals
All late-2026 figures below are sourced strictly from the project's verified research brief (J.P. Morgan North America Credit Research, Oct 29 & Nov 4, 2026; *The Wall Street Journal*, Oct 30, Nov 1 & Nov 5, 2026; *Bloomberg News*, Nov 18, 2025 & Oct 13, 2026):

* **Total Tracked Hyperscaler & Data Center Debt:** **$\$629\text{B}$** as of late Oct 2026 ($+\$27\text{B}$ since July 2026): $\$495\text{B}$ direct IG debt, $\$70\text{B}$ IG data-center lessor debt, $\$64\text{B}$ HY data-center debt. Concentration: `ORCL` ($\$134\text{B}$), `META` ($\$130\text{B}$), `AMZN` ($\$120\text{B}$), `Other` ($\$111\text{B}$), `GOOGL` ($\$90\text{B}$), `MSFT` ($\$44\text{B}$). *(J.P. Morgan Bond Matrix 3.0, Oct 29, 2026)*.
* **High-Yield HPC Credit Stress:** J.P. Morgan HY HPC Index spread widened **$+244\text{ bps}$ MoM to $503\text{ bps}$** (peaking at $544\text{ bps}$); HPC share of the J.P. Morgan HY Index tripled from $1.07\%$ to $3.44\%$ ($\$53.3\text{B}$, 18 issuers). Leveraged Loans HPC (3y) widened $+207\text{ bps}$ in October to $667\text{ bps}$. *(J.P. Morgan, Oct 29, 2026)*.
* **Neocloud Distress (`CRWV`):** CoreWeave equity $-23\%$ YTD; its 2031 9% bonds widened past **$1,000\text{ bps}$** over UST (vs. $\sim 400\text{ bps}$ in June 2026). CoreWeave revenue concentration: Meta ($36\%$), OpenAI ($23\%$), Microsoft ($20\%$), Other ($11\%$), Google ($10\%$). *(J.P. Morgan, Oct 29, 2026; WSJ, Nov 1, 2026)*.
* **Hyperscaler Long-End Curve Repricing:** Hyperscaler 10y+ spreads reached **$168.8\text{ bps}$** ($+61.6\text{ bps}$ over the JULI index at $107.2\text{ bps}$, vs. $+15.8\text{ bps}$ at end-2025); hyperscaler 10s30s steepened to $44\text{ bps}$ vs. $14\text{ bps}$ for the broader IG market. *(J.P. Morgan Figure 10, Oct 29, 2026)*.
* **Circular SPV & Vendor Guarantees:**
  * **Anthropic / Broadcom:** Anthropic raised **$\$63\text{B}$** in corporate debt ($\$40\text{B}$ bank loans + $\$23\text{B}$ with Blackstone) via a Broadcom-guaranteed SPV to buy Broadcom/Google chips, alongside Anthropic's IPO S-1 filing disclosing direct Broadcom lending. *(WSJ, Oct 30 & Nov 1, 2026)*.
  * **Nvidia Guarantees:** Nvidia backstopped **$\$105\text{B}$** for SoftBank SB Energy's Ohio data center for OpenAI, holds a major equity stake in CoreWeave, and committed to lease CoreWeave's unused capacity through 2032. *(WSJ, Nov 1, 2026)*.
  * **Inverted Vendor Financing:** Nvidia, AMD, and Broadcom anchored **$\$5\text{B}$** of **Meta's $\$25\text{B}$** equity offering. *(WSJ, Nov 1, 2026)*.
  * **Underwriting Pullback:** On the $\$5\text{B}$ Volta Infrastructure deal, JPMorgan and Goldman Sachs underwrote only $\$1.3\text{B}$, leaving $\$3.7\text{B}$ of GPU-backed debt on a best-efforts basis. *(J.P. Morgan, Oct 29, 2026)*.
* **Global Macro Loop:** Monthly U.S. chip imports rose from $\$11\text{B}$ (Dec 2023) to $\$71\text{B}$ (Jul 2026); South Korea's current-account surplus surged from $\$33\text{B}$ (2023) toward a projected $\$450\text{B}$ (2026), recycling into U.S. equities ($\$17\text{B} \to \$69\text{B}$ over 12 months) rather than USTs, while hyperscaler borrowing crowded out EM USD sovereigns (EM yields up $\sim 1\text{ pp}$ in 2H26). *(Bloomberg News, Oct 13, 2026)*.
* **"Blue Wave" Regulatory Shock (Nov 3, 2026):** **42 of 45** local ballot initiatives restricting data centers passed or led; Democratic governors elected in TX, OH, GA, NV, NM proposed moratoria/pauses impacting the **$\sim 78\%$** of public HY data-center pipeline located in Texas (primarily hitting 2028–2029 builds). Post-election reaction: `WGMI` (mining/digital power ETF) fell $\sim 9\%$, PHLX Semiconductor Index fell $3\%$, while `AMZN` and `MSFT` rose $>6\%$ on forced Capex discipline. *(J.P. Morgan, Nov 4, 2026; WSJ, Nov 5, 2026)*.

---

## 4. Mathematical Methodology & Module Specifications

### Module 1: `src/bubble_monitor.py` (Bubble Probability & Triggers)

#### 4.1.1 Rolling BSADF / GSADF Exuberance Detector (Phillips, Shi, & Yu, 2015)
For a log-price series $y_t = \ln(P_t)$ (computed for an equal- or cap-weighted AI Core index and for `SPY`), estimate the rolling ADF regression over subsample $[r_1, r_2]$ with minimum window fraction $r_0 = 0.01 + 1.8/\sqrt{T}$:
$$\Delta y_t = \alpha_{r_1, r_2} + \beta_{r_1, r_2} y_{t-1} + \sum_{k=1}^{K} \gamma_{r_1, r_2, k} \Delta y_{t-k} + \varepsilon_t$$
The Backward SADF statistic at observation fraction $r_2$ is:
$$\text{BSADF}_{r_2}(r_0) = \sup_{r_1 \in [0, r_2 - r_0]} \left\{ \text{ADF}_{r_1}^{r_2} \right\}$$
- **Strict Look-Ahead Rule:** At date $t$, $\text{BSADF}_t$ uses **strictly** data from $[1, t]$.
- **Signal:** Flag an *Explosive Regime* when $\text{BSADF}_t$ exceeds the simulated 95% finite-sample right-tailed critical value (pre-computed via Monte Carlo random walk simulation with fixed seed for speed and reproducibility).

#### 4.1.2 Greenwood–Shleifer–You (2019) Crash Risk Score
Compute the four rolling, strictly causal (lagged $t-1$) features over a trailing window (e.g., 252 trading days for volatility/acceleration, 63 days vs. 252 days for turnover surge) from `data/raw/*.csv`:
1. **Volatility Surge ($Z_{\text{vol}, t}$):** Standardized 63-day realized annualized volatility relative to trailing 504-day distribution.
2. **Turnover / Volume Surge ($Z_{\text{turn}, t}$):** Standardized 21-day average dollar volume divided by 252-day average dollar volume.
3. **Price Path Acceleration ($Z_{\text{acc}, t}$):** Ratio of recent 126-day return to 252-day return (measuring late-stage convexity).
4. **Issuance & Credit Stress Proxy ($Z_{\text{cred}, t}$):** Standardized composite of tracked HY HPC credit spread ($503\text{ bps}$ vs. $259\text{ bps}$ baseline) and Hyperscaler 10y+ vs. JULI basis ($+61.6\text{ bps}$ vs. $+15.8\text{ bps}$ end-2025 baseline), derived from Gilchrist & Zakrajšek (2012).

Combine into a transparent **Composite Bubble Stress Score** $S_t$ and map to an interpretable **12–24 Month Crash Risk Probability** via a calibrated logistic link:
$$S_t = w_1 Z_{\text{vol}, t} + w_2 Z_{\text{turn}, t} + w_3 Z_{\text{acc}, t} + w_4 Z_{\text{cred}, t}, \qquad P(\text{Crash}_t) = \frac{1}{1 + \exp(-(\theta_0 + \theta_1 S_t))}$$
- **Transparency Mandate:** Expose the exact weights $(w_1, w_2, w_3, w_4)$ and calibration note in the UI and `RULE_CARD.md`. Clearly label this as a *calibrated stress-index probability* (not a black-box ML prediction).

#### 4.1.3 Early-Warning Trigger Checklist & Rule Switch-Off Criteria
Implement a deterministic 5-point traffic-light dashboard (`GREEN` / `AMBER` / `RED`):
1. **HY HPC Credit Spread Trigger:** `RED` if $> 600\text{ bps}$, `AMBER` if $450\text{–}600\text{ bps}$ (Current Oct 2026: $503\text{ bps}$ $\to$ `AMBER`).
2. **Hyperscaler 10y+ Basis vs. JULI:** `RED` if $> 75\text{ bps}$, `AMBER` if $45\text{–}75\text{ bps}$ (Current Oct 2026: $+61.6\text{ bps}$ $\to$ `AMBER`).
3. **Neocloud Distress Spread (`CRWV` 2031s):** `RED` if $> 1,000\text{ bps}$ (Current Oct 2026: $>1,000\text{ bps}$ $\to$ `RED`).
4. **GSADF Explosive Equity Signal:** `RED` if AI Core $\text{BSADF}_t > \text{CV}_{95\%}$.
5. **Supply / Underwriting Friction (Ofek–Richardson):** `RED` when mega-SPOs/IPOs coincide with bank underwriting pullbacks (Current Oct–Nov 2026: Anthropic S-1 + Meta $\$25\text{B}$ SPO + Volta $\$1.3\text{B}/\$5\text{B}$ partial underwrite $\to$ `RED`).

---

### Module 2: `src/macro_contagion.py` (Extended Domino & S&P 500 Forecast)

#### 4.2.1 `DebtRank` Centrality on `relationships_final.csv` (Battiston et al., 2012)
Using the directed exposure graph in `data/raw/relationships_final.csv` (and equity/debt anchors from `data/raw/financials.csv`), define the impact matrix $W_{ij} = \min\left(1, \frac{A_{ij}}{E_j}\right)$, where $A_{ij}$ is the dollar exposure/guarantee/revenue commitment of firm $j$ to firm $i$ and $E_j$ is firm $j$'s equity buffer.
Initialize distress $h_i(0) \in [0, 1]$ for an epicenter shock (e.g., neocloud default) and iterate:
$$h_j(t+1) = \min\left(1, \; h_j(t) + \sum_{i \in \mathcal{A}(t)} W_{ij} \, h_i(t)\right)$$
where $\mathcal{A}(t)$ is the set of active distressed nodes (each node propagates its shock once, avoiding infinite loops). Compute:
$$\text{DebtRank}_i = \sum_{j \neq i} \left(h_j(T) - h_j(0)\right) v_j$$
where $v_j$ is node $j$'s share of total ecosystem value. Output a ranked bar chart of **Most Systemically Central Nodes** and an interactive **Shock Simulator** (user picks which node defaults and shock severity $h_i(0)$).

#### 4.2.2 4-Tier Cross-Market Spillover Matrix (Diebold & Yilmaz, 2014)
Estimate a rolling VAR($p$) (default $p=2$, forecast horizon $H=10$ days, strictly causal trailing window of 252 trading days) on daily log returns of four concentric asset tiers constructed from `data/raw/*.csv`:
- **Tier 1 (Epicenter — Neoclouds):** Equal-weighted `CRWV`, `APLD`, `GLXY` (handling shorter histories gracefully via dynamic availability masks).
- **Tier 2 (Circular Guarantors — Chipmakers):** `NVDA`, `AVGO`, `AMD`.
- **Tier 3 (Hyperscalers):** `ORCL`, `META`, `AMZN`, `GOOGL`, `MSFT`.
- **Tier 4 (Broader Market):** `SPY` (S&P 500), plus optional live/static proxies if available (`BX` for Private Credit exposure).

Compute the Generalized Forecast Error Variance Decomposition (GFEVD) $\tilde{\theta}_{ij}^g(H)$ (normalized so $\sum_{j=1}^N \tilde{\theta}_{ij}^g(H) = 1$):
- **Total Connectedness Index (TCI):** $\text{TCI} = \frac{1}{N} \sum_{i \neq j} \tilde{\theta}_{ij}^g(H) \times 100$.
- **Directional Spillover `To` / `From` / `Net`** for each tier.
- Visualize as a clean **4x4 Directional Spillover Heatmap** and a **Rolling TCI Time-Series** showing how AI-to-`SPY` connectedness spikes during stress episodes.

#### 4.2.3 3-Scenario S&P 500 (`SPY`) & Tech Stress-Test Forecast
Compute auditable, transparent drawdown projections for `SPY` using two explicit components:
1. **Direct Index-Weight Component:**
   $$\Delta \text{SPY}_{\text{direct}} = \sum_{i \in \text{AI Core in S\&P}} w_{i,\text{SPY}} \cdot \Delta P_i$$
   where $w_{i,\text{SPY}}$ are transparent, editable S&P 500 weights (defaulting to current approximate index weights for `NVDA`, `MSFT`, `AMZN`, `GOOGL`, `META`, `AVGO`, `ORCL`, `AMD`) and $\Delta P_i$ are the firm-level scenario drawdowns from `DebtRank`.
2. **Non-AI Spillover Component:**
   $$\Delta \text{SPY}_{\text{total}} = \Delta \text{SPY}_{\text{direct}} + \beta_{\text{spill}} \cdot \left(1 - \sum_{i} w_{i,\text{SPY}}\right) \cdot \Delta P_{\text{AI, avg}}$$
   where $\beta_{\text{spill}}$ is empirically estimated from the rolling regression of non-AI `SPY` proxy returns on AI-cluster returns during downside regimes (bottom 10% tail days in `data/raw/`), bounded by the 2000–2002 historical S&P-to-Nasdaq elasticity ($\approx 0.63$).

Present **three clearly defined scenarios** (with user-adjustable sliders for sensitivity analysis):
- **Scenario 1 — Capex Slowdown & "Blue Wave" Bottleneck (Mild):** Regulatory moratoria slow 2028+ builds. Tier-1 neoclouds/Texas DC pipeline draw down sharply ($-35\%$ to $-50\%$), chipmakers pull back moderately ($-15\%$), while cash-rich hyperscalers (`MSFT`, `AMZN`) remain resilient ($-5\%$ to $+5\%$). Contained `SPY` impact.
- **Scenario 2 — Circular Vendor-Financing Unwind (2001 Telecom Analog — Severe):** A Tier-1 neocloud or AI lab defaults on lease/debt covenants, calling SPV guarantees on Tier-2 chipmakers (`NVDA`, `AVGO`) and blowing out HY HPC spreads ($>800\text{ bps}$). Propagated via `DebtRank` + downside $\beta_{\text{spill}}$.
- **Scenario 3 — Systemic Credit & Macro Cascade (Tail Risk):** Full Gilchrist–Zakrajšek EBP credit contraction + reversal of Asian tech-surplus equity inflows + private-credit markdowns (`BX`), approaching the 2000–2002 S&P 500 historical drawdown envelope (up to $-49.1\%$).

---

### Module 3: `src/historical_analog.py` (2000–2002 Analog & Actionable Playbook)

1. Store verified 2000–2002 benchmark metrics (Section 3.1) and October–November 2026 credit metrics (Section 3.2) in clean Python dataclasses with mandatory `source` strings.
2. Render a structured **Structural Comparison Table** (*1999–2002 Telecom/Dot-Com* vs. *2025–2026 AI Circular Web*) highlighting both parallels (vendor financing, CLECs vs. neoclouds/AI labs, IPO/SPO supply wave) and key differences (stronger hyperscaler cash flows today, but shorter 3–6 year GPU collateral life vs. multi-decade fiber).
3. Render the **Actionable Positioning Matrix (Who to Short / Avoid / Hold)** tied directly to live metrics:
   - **Short / Avoid (High Fragility):** Leveraged neoclouds (`CRWV`, `APLD`, `GLXY`) with $>1,000\text{ bps}$ credit stress or unfinanced Texas DC pipeline; unhedged bridge/BTM power names exposed to 2028–2029 delays.
   - **Tactical Underweight / Hedge (Guarantee Exposure):** Chipmakers (`NVDA`, `AVGO`) *only when* Credit & Underwriting Triggers are `RED` (hedge guarantee/SPV backstop risk via puts/pairs rather than naked shorts due to strong gross margins).
   - **Defensive / Relative Long (Fortress Balance Sheets):** Lower-debt hyperscalers (`MSFT` at $\$44\text{B}$ tracked debt vs. `ORCL` $\$134\text{B}$ and `META` $\$130\text{B}$; `AMZN`) and "assets in the ground" (already-built, permitted capacity with take-or-pay covenants).

---

## 5. Submission Checklist & Academic Integrity Compliance

To guarantee maximum points on the **Dossier (24 pts)**, **Fresh Series Drill (8 pts)**, and **Individual Defence (8 pts)**, the AI coding assistant must also update the following repository documentation files:

1. **`RULE_CARD.md`:** Append a versioned `v3.0` section logging the exact quantitative entry/exit rules for the Bubble Radar trigger (GSADF + Greenwood–Shleifer–You + Credit Spread thresholds) and explicitly defining the **Switch-Off Condition** (when the short/hedge rule turns off: e.g., HY HPC spreads compress below $350\text{ bps}$ and $\text{BSADF}_t$ falls below $\text{CV}_{90\%}$).
2. **`DATA.md`:** Document:
   - **Survivorship Bias:** The 12-ticker AI universe contains current winners/survivors (unlike the 47 delisted CLECs in 2000–2003); `CRWV` has a shorter listing history.
   - **Look-Ahead Bias Prevention:** All BSADF, volatility, turnover, and VAR FEVD windows are strictly causal (using data $\le t-1$ for signals applied at $t$).
   - **Short-Selling & Borrow Costs:** Explicitly note that shorting high-beta neoclouds (`CRWV`, `APLD`) incurs elevated borrow fees and squeeze risk, which must be accounted for in net-of-cost evaluation.
3. **`trials.csv`:** Log any new parameter configurations tested for the v3 indicators to maintain full transparency against data snooping.
4. **`AI_USE.md`:** Declare the use of AI tools for literature synthesis and v3 module scaffolding in accordance with the course academic integrity policy.
5. **`tests/test_methodology_v3.py`:** Add unit tests verifying:
   - **Canary Look-Ahead Test:** Perturbing future prices ($t > t_0$) produces **zero** change in `BSADF`, `Crash Risk Score`, or `Diebold–Yilmaz TCI` at $t \le t_0$.
   - **DebtRank Bounds Test:** All node distress values stay strictly in $[0, 1]$ and monotonic in initial shock size.
   - **Scenario Conservation Test:** S&P 500 forecast returns exact weighted sum + spillover term with no `NaN` values.

---

## 6. Complete Reference List (Verified)

### Peer-Reviewed Academic Literature
1. Acemoglu, D., Ozdaglar, A., & Tahbaz-Salehi, A. (2015). Systemic Risk and Stability in Financial Networks. *American Economic Review*, 105(2), 564–608. https://doi.org/10.1257/aer.20130456
2. Battiston, S., Puliga, M., Kaushik, R., Tasca, P., & Caldarelli, G. (2012). DebtRank: Too Central to Fail? Financial Networks, the FED and Systemic Risk. *Scientific Reports*, 2, 541. https://doi.org/10.1038/srep00541
3. Billio, M., Getmansky, M., Lo, A. W., & Pelizzon, L. (2012). Econometric measures of connectedness and systemic risk in the finance and insurance sectors. *Journal of Financial Economics*, 104(3), 535–559. https://doi.org/10.1016/j.jfineco.2011.12.010
4. Diebold, F. X., & Yilmaz, K. (2014). On the network topology of variance decompositions: Measuring the connectedness of financial firms. *Journal of Econometrics*, 182(1), 119–134. https://doi.org/10.1016/j.jeconom.2014.04.012
5. Gilchrist, S., & Zakrajšek, E. (2012). Credit Spreads and Business Cycle Fluctuations. *American Economic Review*, 102(4), 1692–1720. https://doi.org/10.1257/aer.102.4.1692
6. Greenwood, R., Shleifer, A., & You, Y. (2019). Bubbles for Fama. *Journal of Financial Economics*, 131(1), 20–43. https://doi.org/10.1016/j.jfineco.2018.09.004
7. Ofek, E., & Richardson, M. (2003). DotCom Mania: The Rise and Fall of Internet Stock Prices. *Journal of Finance*, 58(3), 1113–1137. https://doi.org/10.1111/1540-6261.00560
8. Phillips, P. C. B., Shi, S., & Yu, J. (2015). Testing for Multiple Bubbles: Historical Episodes of Exuberance and Collapse in the S&P 500. *International Economic Review*, 56(4), 1043–1078. https://doi.org/10.1111/iere.12132

### Institutional Research, Regulatory & Market Sources
9. U.S. Securities and Exchange Commission (SEC). (May 17, 2004). *Lucent Settles SEC Enforcement Action Charging the Company with $1.1 Billion Accounting Fraud* (Press Release 2004-67).
10. J.P. Morgan North America Credit Research (Hamid, T., Rosenbaum, N., Jantzen, N., Mantri, S., et al.). (October 29, 2026). *AI Capex Funding: Data Center – Hyperscaler Bond Matrix 3.0*.
11. J.P. Morgan North America Credit Research (Hamid, T., Kwan, K. L., Egan, T., et al.). (November 4, 2026). *AI Capex / Data Centers: Early Thoughts On Blue Wave Implications On DC & Power Funding*.
12. Mackintosh, J. (November 1, 2026). *The Tangled Web of AI Bets Is Reason to Fear Tech Stocks*. The Wall Street Journal.
13. Demos, T. (November 5, 2026). *How the Data Center Backlash Could Play Into the Hands of Big Tech*. The Wall Street Journal.
14. Tucker, E. (November 1, 2026). *Investors Are Hedging Against the Risks of the AI Boom*. The Wall Street Journal.
15. Knutson, R., & Saeedy, A. (October 30, 2026). *Is AI Running Out of Money to Borrow?* The Wall Street Journal (*The Journal* Podcast).
16. Donnan, S. (October 13, 2026). *The New Circular AI Finance Trend Is Global and Spooky*. Bloomberg News (*AI Today*).
17. Bloomberg News. (November 18, 2025). *OpenAI and Anthropic Tap Web of Circular AI Deals*.