# AI use disclosure - Circular Alpha

## Summary

We used AI tools to build this project, and this file lists every one of them, what each did and how we checked it. We picked the idea, decided what the app should show and say, ran every piece of code on our own laptops, and tested the results. The AI tools wrote most of the code and drafted most of the documents. We are responsible for everything in this repository.

## Tools we used

| Tool | What it did |
|---|---|
| Claude (Anthropic), GPT, Codex | wrote all the code blocks we pasted into `core.py`, `app.py` and `tests/test_core.py` (Sharpe ratio, maximum drawdown, beta, the group comparison, the growth chart, the first curved-arrow graph); explained every step of code to us and how vibecoding works. |
| Codex (VS Code extension) | Made the code changes that we asked for with written prompts: amounts along each cycle, moving the arithmetic out of `app.py` into `core.py`, the "what if this company fails" selector, the "worst days" table, and the display clean-ups listed below. |

People responsible for checking AI output: Linara (six), Darina (seven), Asselya (67)

## What we did ourselves

- We chose the idea and the question of our project
- We collected and checked the relationship data and the company list by hand. The sources and review dates are in `relationships_final.csv`.
- We downloaded the price files from Yahoo Finance on 2026-10-10 and saved them as CSV. 
- We ran every command and every test on our own laptops, pasted the code blocks into the files, fixed the indentation errors that came up, restarted the app and looked at every screen.
- We decided how the app should look: curved arrows from payer to receiver (we rejected the first straight-line version), a table of amounts along each cycle without repeating company names, a stress-test selector, and a table of how the other stocks behaved on a company's worst days.
- We checked the amounts shown on screen against `relationships_final.csv`. They match the file.
- We recomputed one number on the screen in a spreadsheet from the saved CSV: NVDA total return = last adjusted close 230.48 / first adjusted close 14.2673 - 1 = 1515.443%. The app shows 1515.44%. The file is `docs/check_nvda.xlsx`.
- We fixed the data dictionary (wrong file name, wrong cycle length) with find and replace.
- We kept the project structure we already had and told the assistants not to change anything else.
- We keep Codex away from the documents
- We created a shared GitHub repository for the team and used it to keep one version of the code and documents.
- We wrote the NOTE that we submitted till deadline 
- We wrote the requests and propmts to the AI ourselves, step by step
- We tested every change ourselves: we ran `pytest`, restarted the app, clicked through the screens and compared the result with the previous version.

## What the AI did

- The assistants wrote the code, step by step, from our requests and prompts.
- AI explained us the code and vibecoding processes 
- Created the drafts of md files 

## Log: what was done, where, how we checked

| Tool | Where | What we asked (short) | How we checked |
|---|---|---|---|
| Claude | `requirements.txt`, `.gitignore` | Fix the install | We ran the install and `pytest` on our laptop. |
| Claude | `tests/test_core.py`: 7 tests (beta, total return, volatility, `normalized_performance`, cycles) | Add tests | `pytest`; the hand calculations are written in the comments and in `NOTE.pdf`, annex A. |
| Codex | `core.py`: `sharpe_ratio`, `max_drawdown`, `beta_vs_benchmark`, `cycle_member_tickers`, `compare_groups` + 10 tests | Add Sharpe, drawdown and the basket comparison | `pytest`; recomputed cycle group return 717.33%, Sharpe 1.64 and beta 1.77 separately; our screen showed the same. |
| Codex | `app.py`: headline section with inputs (risk-free rate, dates, comparison group) | Connect the new functions to the screen | We changed the inputs and the numbers changed. |
| Codex | `core.py`: `growth_of_base`; `app.py`: growth chart | Add a chart that supports the headline | The end values of the lines (about 817, 440, 213) match the returns in the table. |
| Codex | `core.py`: `price_data_summary`; `app.py`: price table in the limitations panel | Move the calculation out of the screen file | Test added; the table appears on screen. |
| Codex and Claude | `core.py` / `app.py`: amounts column in the cycle table | Show the amounts along each cycle instead of company ids | We asked why all amounts looked like 30 billion; Claude then compared every amount with `relationships_final.csv` and they matched the file. |
| Claude | `app.py`: curved arrows on the network graph | Draw arrows from payer to receiver | We looked at the graph and asked for half-circle arcs instead of the first design. |
| Codex | `core.py`: `format_money`, `line_midpoint`, `arc_midpoint`, `curved_arc_points`; `app.py` | Move all arithmetic out of `app.py` (the brief caps the method mark if the screen code contains calculations) | `pytest` passed (37 tests); we restarted the app and the graph looked the same. |
| GPT | `core.py`: `failure_exposure`, `cycles_through_company`, display helpers; `app.py`: "Stress test: if this company fails" | Show which companies are linked to the selected one and how many steps away (structure only, not a forecast) | The hand-calculated test (A->B, B->C, C->A, C->D, B fails) passed; we tried several companies on screen. |
| Codex | `core.py`: `worst_day_comovement` and text helper; `app.py`: table of how other stocks moved on a company's worst days | Show historical co-movement | `pytest` |
|Codex | `app.py`, `core.py`: cycle amounts as plain text per step ("up to $10.0B, investment"), tidy section 3 (raw data in a collapsed block, only the selected company's financials), readable data-quality summary | Make three tables understandable for a normal user | [FILL IN: only keep the rows that were really applied; write the pytest result.] |
| Claude | `data_dictionary_synced.csv` | Fix wrong text in the data dictionary | We made the two replacements ourselves and searched the project for the old text. |

## Suggestions we rejected or changed

- We rejected a beta-times-shock model for the stress test, because it looks like a forecast and OpenAI and Anthropic have no prices. We use the structure of the links and the historical behaviour on the worst days instead (`DECISIONS.md`, D10 and D11). We decided to left this part of work on the end term.
- We did not accept the first curved-arrow design and asked for arcs that go over the lines, like the examples we showed.
- We wanted the cycle table to show amounts instead of company ids, and later without repeating the company names.
- We told Codex not to change anything outside the files we named and to skip any request to edit other files. When it asked for permission to edit other documents we pressed Skip.

## How we checked the work

Checks done by people:
- We ran the app after every change, with the network on and with the network off, and looked at every screen.
- We checked the amounts on the screen against `relationships_final.csv`.
- We recomputed one number in a spreadsheet.
- We performed hand- calculated tests

## Where AI-written code lives

| Location | What it contains |
|---|---|
| `core.py` | All calculations. Written by Claude and Codex from our requests.
| `app.py` | The screen only. Written by Claude and Codex; it calls functions from `core.py`. |
| `tests/test_core.py` | All tests. Written by Claude and Codex; hand calculations are in comments. |
| `data.py`, first versions of the above |

## What AI could not check

Claude could not open Yahoo, SEC or news pages from its environment, so it did not check the prices against any outside source and did not check whether the deals in `relationships_final.csv` are true. 


## Statement

We confirm that the idea, the choice of data, the decision on what the app shows and the interpretation of the results are our own decisions, that the AI tools listed above wrote most of the code and drafted most of the documents, that all use of AI is listed in this file, and that we are responsible for everything in this repository.

Members: Darina Baikazakova, Linara Mamraimova, Asselya Krymova

---

## Group Assignment 2 & Final v2 AI Use Log (2026-11-21)

| Date | Tool | Policy Tier | Task | What the team kept, verified, or must rewrite |
|---|---|---|---|---|
| 2026-11-21 | Codex | GREEN | `app.py` UI refactor: unified contagion graph with arrowheads, executive impact cards, plain-language storytelling, and EN/RU switcher | Kept and verified via `py_compile`. |
| 2026-11-21 | Codex | GREEN | Purple & White startup theme and draggable pixel mascot widget | Kept and verified via `py_compile`. |
| 2026-11-21 | Codex | GREEN | Alpha-chan AI chat, optional session-only API key, and local fallback | Kept; verified with the permitted compile check. |

| Date | Tool | Policy Tier | Task | What the team kept, verified, or must rewrite |
|---|---|---|---|---|
| 2026-11-21 | Gemini | GREEN | Repository architecture, `app.py` 5-tab English UI, `README.md`, `.github/workflows/daily_news_agent.yml`, `src/live_data.py`, `src/news_agent.py` | Kept and verified via `py_compile` and unit tests. |
| 2026-11-21 | Gemini | YELLOW | Drafted `src/backtest.py`, `src/clustering.py`, `src/contagion.py`, `scripts/generate_evidence.py`, and `tests/test_ga2_and_v2.py`; generated `evidence/` tables and figures | All 9 new unit tests (including hand-computed toy backtest and canary look-ahead test) executed and passed. Team to review every formula and line before submission. |
| 2026-11-21 | Gemini | RED (Draft for Team Rewrite) | Initial aligned drafts of `RULE_CARD.md` (R3–R5) and `RESULT_CARD.md` verdict/limits | **Action required by team (Darina, Linara, Asselya):** Rewrite R3–R5, the toy/canary explanation, and the verdict/limits wording in your own words and verify every number against `evidence/` prior to Quiz 2 and the Final Defence. |

---

## Circular Alpha v3 AI Use Log (2026-11-27)

| Date | Tool | Policy Tier | Task | What the team kept, verified, or must rewrite |
|---|---|---|---|---|
| 2026-11-27 | Claude Code | GREEN | Placed `METHODOLOGY_AND_SPEC.md`, `scripts/fetch_extended_data.py`, `data/circular_alpha_research_pack.xlsx`; ran the fetch script (Yahoo Finance + FRED) | Datasets validated by 18 tests in `tests/test_extended_data.py`; dot-com peaks/troughs match the cited index levels exactly. |
| 2026-11-27 | Claude Code | YELLOW | Drafted `src/bubble_monitor.py` (BSADF/GSADF, GSY score, triggers), `src/macro_contagion.py` (DebtRank, Diebold–Yilmaz, S&P 500 scenarios), `src/historical_analog.py`, `src/credit_signals.py`, `src/extended_data.py`, `src/v3_tabs.py` and three new tabs in `app.py` | BSADF closed form checked against brute-force ADF; 29 tests in `tests/test_methodology_v3.py` incl. canary look-ahead, DebtRank bounds/monotonicity and scenario conservation. Team to review every formula. |
| 2026-11-27 | Claude Code | YELLOW | Literature synthesis for citations shown in the app (taken from `METHODOLOGY_AND_SPEC.md`) | Claude could not open the publishers' pages to re-verify DOIs; a web search listed the GSY (2019) DOI as `10.1016/j.jfineco.2018.09.002` while the spec says `.004` — **team must check before submission**. |
| 2026-11-27 | Claude Code | RED (Draft for Team Rewrite) | `RULE_CARD.md` v3.0 section, `DATA.md` §6, `trials.csv` V3 rows | Rewrite V1–V6 in your own words; verify each number against the app and tests. |

Scenario tier shocks, LGD for revenue links (0.25), drawdown per unit of DebtRank distress (50%) and borrow fees are **assumptions**, labelled as such in the UI, not data.
