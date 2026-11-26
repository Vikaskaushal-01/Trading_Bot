"""Real-time intraday candles from TradingView's chart websocket, with a yfinance fallback.

TradingView has no official public data API. This module speaks the same websocket protocol the
tradingview.com chart uses, as an anonymous (logged-out) viewer, so it gets exactly what a logged-out
chart shows: real-time for most CFDs / forex / crypto, delayed for some exchange-listed symbols.
Automated access may conflict with TradingView's terms of use — use it for personal research only.
If the websocket fails, intraday bars are taken from Yahoo Finance instead.
"""

from __future__ import annotations

import json
import random
import re
import string
import time

import pandas as pd
import streamlit as st

from utils.market import OHLCV, get_history

WS_URL = "wss://data.tradingview.com/socket.io/websocket"
ORIGIN = "https://www.tradingview.com"

# display name -> (TradingView symbol, Yahoo Finance fallback ticker)
PRESETS = {
    "S&P 500 (SPREADEX SPX)": ("SPREADEX:SPX", "ES=F"),
    "S&P 500 (OANDA SPX500USD)": ("OANDA:SPX500USD", "ES=F"),
    "S&P 500 (SPX500 CFD)": ("FOREXCOM:SPXUSD", "ES=F"),
    "Nasdaq 100 (NAS100 CFD)": ("FOREXCOM:NSXUSD", "NQ=F"),
    "Dow Jones (US30 CFD)": ("FOREXCOM:DJI", "YM=F"),
    "Gold (XAUUSD)": ("OANDA:XAUUSD", "GC=F"),
    "EUR/USD": ("FX:EURUSD", "EURUSD=X"),
    "Bitcoin (BTCUSDT)": ("BINANCE:BTCUSDT", "BTC-USD"),
    "Ethereum (ETHUSDT)": ("BINANCE:ETHUSDT", "ETH-USD"),
    "Nifty 50": ("NSE:NIFTY", "^NSEI"),
}

TIMEFRAMES = {"1m": "1", "3m": "3", "5m": "5", "15m": "15"}


def _frame(func: str, params: list) -> str:
    msg = json.dumps({"m": func, "p": params}, separators=(",", ":"))
    return f"~m~{len(msg)}~m~{msg}"


def _session(prefix: str) -> str:
    return prefix + "".join(random.choices(string.ascii_lowercase, k=12))
