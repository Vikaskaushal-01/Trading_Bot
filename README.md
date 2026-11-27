# 📈 TradeAI — AI Paper-Trading Platform

A Streamlit web app for paper trading US stocks, crypto and Indian equities with
machine-learning signals, technical analysis, strategy backtesting and an AI assistant.

> Paper trading only — no real money moves. Signals are educational, not financial advice.

## Features

| Page | What you get |
|---|---|
| **Dashboard** | Net worth of both accounts, featured market cards with sparklines, top gainers/losers, full market table, **AI scanner** ranking every asset by composite score |
| **Live / Demo Trading** | Candlestick/line charts with EMA 20/50, SMA 200, Bollinger, RSI, MACD and on-chart lines for cost basis, stops, targets, limit orders and the AI price target · **Market & Limit orders** · **risk panel**: stop-loss (default 2×ATR), take-profit by reward:risk, trailing stop, risk-per-trade position sizing, ½-Kelly check · AI signal card · technical rating · positions with exit-rule editor, allocation donut and risk checks · pending orders · trade history with trigger reasons + CSV · deposit / withdraw / reset · watchlist · any Yahoo Finance ticker · 15s live mode that keeps stops & limits monitored |
| **Live Scalper** (opens first) | Persistent TradingView stream: price, bid/ask and live spread ticking every second, clock + candle countdown, tick sparkline · new BUY / SELL / WAIT at every candle close (or every 15/30 s, provisional) · BUY vs SELL success % from the most similar past setups, labelled by simulating your TP/SL/spread/max-hold on ~10,000 candles · targets in points or ATR × · filters: trend-only, trading hours, high-confidence (90% lower bound) · 26 inputs incl. 5m RSI, 15m trend, VWAP, today's & yesterday's high/low · walk-forward replay with equity curve · best-hours analysis · TP/SL optimizer · trade journal that follows your logged trades to TP/SL with live bid/ask · daily max-trades / loss / profit limits · sound + toast alerts · settings remembered between days |
| **AI Lab** | Full model diagnostics: calibrated P(up) vs usual up-rate, reliability (out-of-sample ROC-AUC), 10/50/90% price targets, signal-consensus breakdown, probability history on price, returns by conviction bucket, calibration curve, out-of-sample trading of the signal, 53-feature dashboard with percentiles, historical analogs, one-click comparison of 5 models |
| **Backtester** | SMA crossover, RSI mean reversion, MACD, Bollinger breakout, Donchian breakout and the **AI model** · stop-loss / take-profit / trailing stop with re-entry cooldown · CAGR, Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor, avg win/loss, trade distribution |
| **AI Assistant** | Groq-powered streaming chat grounded in your balances, holdings, live quotes and the focused asset's AI/technical signals; instant offline answers for balance/portfolio/movers; history persisted |

### AI model settings (sidebar, shared by every page)

| Setting | Options | Why it matters |
|---|---|---|
| Forecast horizon | 1 day · 1 week · 2 weeks · 1 month | Longer horizons are smoother and suit swing trading |
| Model | Ensemble · Gradient Boosting · Random Forest · Extra Trees · Logistic Regression | Compare them in the AI Lab |
| Training history | 3 · 5 · 10 years | More data vs. more relevant regime |
| Feature groups | Momentum · Trend · Volatility · Volume · Price structure · Market context · Calendar | Turn families on/off |
| Required edge | 1–15 pts | How far P(up) must beat the usual up-rate to trigger BUY/SELL |

### How the prediction works
* **53 scale-free features** (see `utils/features.py` for each one's meaning): multi-horizon returns, RSI 7/14,
  Stochastic, Williams %R, CCI, MACD; EMA/SMA distances, ADX/DI, trend persistence; ATR, realised volatility &
  regime, Bollinger %B/width, gaps; volume z-score, OBV slope, MFI, Chaikin money flow; 52-week high/low distance,
  Donchian position, candle anatomy; benchmark returns/trend, relative strength, beta, correlation, VIX.
* **Purged walk-forward validation** (5 folds, gap = horizon) — every metric is out-of-sample.
* **Platt calibration** so stated probabilities match historical frequencies; with no skill they collapse to
  the asset's usual up-rate, and the signal stays HOLD instead of inventing trades.
* **Quantile regression** for the 10th/50th/90th percentile price target; **nearest-neighbour analogs** for
  what similar setups did next.
* **Composite score** = AI (weighted by its reliability) + technical rating + analogs.

> Honest expectation: for liquid stocks and crypto, daily-to-monthly direction is close to unpredictable
> (out-of-sample ROC-AUC usually 0.48–0.55). The app measures and shows this instead of hiding it, and leans on
> risk management (stops, sizing) where prediction has no edge.

### Live Scalper: what the trade analysis found (Nov 2026)
Re-checked on fresh TradingView data (SPREADEX:SPX and FOREXCOM:SPXUSD, 1m and 3m, Nov 8–27 2026):

* The saved defaults (3m, 15-minute hold, trend filter) **did not hold up out of sample**: walk-forward
  ROC-AUC 0.48 BUY / 0.49 SELL. Their +104 pts replay came from 13 SELLs on one day.
* Across 64 settings, no single indicator had a stable link to the next 15 minutes (rank correlation ≤ 0.08,
  about half flipping sign between the two halves of the data), and no hour of the day had a significant drift.
### Currencies
Cash is held in USD. Non-USD assets (e.g. `.NS` stocks in INR) are converted at the live FX rate.

## Setup

```bash
python -m venv trading_bot
trading_bot\Scripts\activate          # Windows  (source trading_bot/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

Create `.env` (optional — enables the free-form AI assistant):

```
GROQ_API_KEY=your_key_here
MODEL_NAME=openai/gpt-oss-120b
```

Run:

```bash
streamlit run app.py
```

## Project structure

```
app.py                 entry point: theme + navigation
app_pages/             dashboard, trading (live), demo, ai_lab, backtest, assistant
utils/
  config.py            asset universe, paths, constants
  market.py            cached yfinance quotes/history, FX conversion
  indicators.py        technical indicators + technical rating
  features.py          53 prediction features with descriptions
  predictor.py         models, purged walk-forward validation, calibration, targets, analogs
  portfolio.py         account engine: orders, stop-loss/take-profit/trailing, limit orders, P&L
  backtest.py          backtester with protective exits and trade statistics
  chatbot.py           Groq assistant + instant local answers
  tv_feed.py           one-off TradingView candle download (+ Yahoo fallback)
  tv_stream.py         persistent TradingView stream: ticks, bid/ask, candles
  journal.py           scalp trade journal with live TP/SL tracking
  scalper.py           intraday outcome labelling, k-NN success rates, walk-forward replay
  trading_view.py      shared trading terminal UI
  chat_view.py         reusable chat panel
  ui.py                CSS theme, components, Plotly charts
data/                  account files (trading_memory.json, demo_memory.json), chat history
```

Old account files (`wallet` / `transactions` format) are migrated automatically on first run.
