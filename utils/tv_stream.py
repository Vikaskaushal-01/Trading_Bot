"""Persistent TradingView stream: live ticks (last, bid, ask) and candles, kept current in a background thread.

One stream per (symbol, timeframe) is shared by every browser tab through ``st.cache_resource``. The page reads
snapshots from memory, so the price ticker can refresh every second without touching the network. If the
socket drops, the thread reconnects with back-off and reloads history.
"""

from __future__ import annotations

import json
import random
import re
import string
import threading
import time

import pandas as pd
import streamlit as st

from utils.market import OHLCV
from utils.tv_feed import ORIGIN, WS_URL

HEARTBEAT = re.compile(r"~m~\d+~m~~h~\d+")
SPLIT = re.compile(r"~m~\d+~m~")
QUOTE_FIELDS = ["lp", "bid", "ask", "ch", "chp", "lp_time", "description", "update_mode", "high_price", "low_price",
                "open_price", "prev_close_price"]


def _frame(func: str, params: list) -> str:
    msg = json.dumps({"m": func, "p": params}, separators=(",", ":"))
    return f"~m~{len(msg)}~m~{msg}"


def _sid(prefix: str) -> str:
    return prefix + "".join(random.choices(string.ascii_lowercase, k=12))
