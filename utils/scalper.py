"""Intraday scalping signal engine.

For every candle we know, we ask: "if I had bought (or sold) at this close — exiting at TP / SL, or after a fixed
number of minutes — did I make money after spread?" That labels every candle for both sides. For the current
candle two independent estimators then give the chance of success:

* **Similar setups (k-NN)** — how often a BUY / SELL worked from the most alike past candles.
* **ML model (gradient boosting)** — trained on every labelled candle to learn which indicator combinations
  lead to wins.

The final percentage is their average. Everything is also replayed walk-forward (each decision only uses candles
whose outcome was already known, models retrained on the past only) to show whether the percentages were
accurate and whether the signal made money.

Market-specific details come from the symbol's own trading session (e.g. SPREADEX:SPX trades 18:00–17:00
New York time), so "today", VWAP, the daily high/low and yesterday's levels reset when that market's day starts,
and S&P 500 cash-session context (09:30 New York open, opening range, gap from the previous close) is included.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

NY = "America/New_York"


@dataclass(frozen=True)
class ScalpParams:
    tp: float = 4.0              # take-profit: points, or ATR multiples when tp_mode == "atr" (net of spread)
    sl: float = 3.0              # stop-loss: same unit as tp
    spread: float = 0.6          # broker spread, price points
    horizon: int = 10            # max candles to hold before closing at market (TP/SL mode)
    k: int = 150                 # neighbours used for the similar-setups estimate
    min_edge: float = 0.05       # success rate must beat break-even by this much
    validate_frac: float = 0.35  # share of history replayed walk-forward
    tp_mode: str = "points"      # "points" | "atr"
    trend_filter: bool = False   # BUY only in an up-trend, SELL only in a down-trend (EMA 200 + 15m trend)
    session: tuple | None = None  # your trading hours: (start, end) minutes after 00:00 UTC; None = always
    high_conf: bool = False      # require the 90% lower confidence bound to clear break-even
    exit_mode: str = "tp_sl"     # "tp_sl" = exit at TP / SL (or max hold) | "time" = close after `hold` candles
    hold: int = 3                # candles held in "time" mode (3 × 1m = close after 3 minutes)
    use_ml: bool = True          # blend in the gradient-boosting model
    require_skill: bool = True   # only signal a side whose percentages proved accurate in the replay
    min_auc: float = 0.53        # "proved accurate" = walk-forward ROC-AUC at least this (0.50 = coin flip)

    @property
    def breakeven(self) -> float:
        """Win rate needed not to lose money. Time exits win and lose roughly equal amounts, so ~50%."""
        return 0.5 if self.exit_mode == "time" else self.sl / (self.tp + self.sl)

    @property
    def span(self) -> int:
        """Candles until every trade opened on a candle is finished."""
        return self.hold if self.exit_mode == "time" else self.horizon

    def label_key(self) -> tuple:
        """Everything that changes the trade outcomes / models (not the filters)."""
        return (self.tp, self.sl, self.spread, self.horizon, self.k, self.validate_frac, self.tp_mode,
                self.exit_mode, self.hold, self.use_ml)

    def decision_key(self) -> tuple:
        return (self.min_edge, self.trend_filter, self.session, self.high_conf, self.require_skill, self.min_auc)


@dataclass(frozen=True)
class MarketSession:
    """The symbol's own trading day, from TradingView (e.g. tz America/New_York, session '1800-1700')."""
    tz: str = "UTC"
    start_min: int = 0           # minutes after local midnight when the trading day begins
    us_index: bool = False       # add US cash-session context (09:30–16:00 New York)

    @classmethod
    def from_info(cls, info: dict, symbol: str = "") -> "MarketSession":
        tz = info.get("timezone") or "UTC"
        m = re.match(r"(\d{2})(\d{2})-", str(info.get("session") or ""))
        start = int(m.group(1)) * 60 + int(m.group(2)) if m else 0
        sym = symbol.upper()
        us = any(s in sym for s in ("SPX", "SPY", "ES1", "US500", "NSX", "NAS", "NDX", "US100", "DJI", "US30", "DOW"))
        return cls(tz=tz, start_min=start, us_index=us)
