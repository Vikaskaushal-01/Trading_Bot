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


def _tv_bars(symbol: str, resolution: str, n_bars: int, timeout: float = 15.0) -> tuple[pd.DataFrame, dict]:
    """Fetch up to `n_bars` candles from TradingView. Raises on failure."""
    from websockets.sync.client import connect

    cs = _session("cs_")
    info: dict = {}
    bars: dict[float, list] = {}
    with connect(WS_URL, additional_headers={"Origin": ORIGIN}, open_timeout=10, close_timeout=2,
                 max_size=2 ** 24) as ws:  # 5,000+ candles arrive in one frame larger than the 1 MB default
        for func, params in [
            ("set_auth_token", ["unauthorized_user_token"]),
            ("chart_create_session", [cs, ""]),
            ("resolve_symbol", [cs, "sds_sym_1", "=" + json.dumps({"symbol": symbol, "adjustment": "splits"})]),
            ("create_series", [cs, "sds_1", "s1", "sds_sym_1", resolution, n_bars, ""]),
        ]:
            ws.send(_frame(func, params))

        deadline = time.time() + timeout
        done = False
        while not done and time.time() < deadline:
            raw = ws.recv(timeout=max(0.5, deadline - time.time()))
            if re.fullmatch(r"~m~\d+~m~~h~\d+", raw):  # heartbeat: echo it back
                ws.send(raw)
                continue
            for part in re.split(r"~m~\d+~m~", raw):
                if not part.startswith("{"):
                    continue
                try:
                    msg = json.loads(part)
                except json.JSONDecodeError:
                    continue
                kind = msg.get("m")
                if kind == "symbol_resolved":
                    info = msg["p"][2]
                elif kind in ("timescale_update", "du"):
                    for bar in msg["p"][1].get("sds_1", {}).get("s", []):
                        v = bar["v"]
                        bars[v[0]] = v
                elif kind == "series_completed":
                    done = True
                elif kind in ("symbol_error", "series_error", "critical_error", "protocol_error"):
                    raise RuntimeError(f"TradingView {kind}: {msg.get('p')}")
    if not bars:
        raise RuntimeError("TradingView returned no candles")

    rows = [v + [0.0] * (6 - len(v)) for v in sorted(bars.values())]
    df = pd.DataFrame(rows, columns=["t"] + OHLCV)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("t"), unit="s", utc=True)).tz_convert(None)  # naive UTC
    df.index.name = None
    return df.astype(float), info
