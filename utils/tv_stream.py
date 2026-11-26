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


class TVStream:
    def __init__(self, symbol: str, resolution: str, n_bars: int = 10000):
        self.symbol, self.resolution, self.n_bars = symbol, resolution, n_bars
        self._lock = threading.Lock()
        self._bars: dict[float, list] = {}
        self.quote: dict = {}
        self.info: dict = {}
        self.status = "connecting"
        self.error = ""
        self.history_ready = False
        self.last_msg = 0.0
        self.last_tick = 0.0
        self.connects = 0
        self.started = time.time()
        self.ticks: list[tuple[float, float]] = []  # (time, last price) — last ~10 min for the tick chart
        threading.Thread(target=self._run, name=f"tv-{symbol}-{resolution}", daemon=True).start()

    # ------------------------------------------------------------------ public
    def snapshot(self) -> tuple[pd.DataFrame, dict]:
        with self._lock:
            rows = [v[:6] + [0.0] * (6 - len(v)) for _, v in sorted(self._bars.items())]
            quote = dict(self.quote)
        df = pd.DataFrame(rows, columns=["t"] + OHLCV)
        if len(df):
            df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("t"), unit="s", utc=True)).tz_convert(None)
            df.index.name = None
            df = df.astype(float)
        else:
            df = pd.DataFrame(columns=OHLCV)
        return df, quote

    def quote_snapshot(self) -> dict:
        with self._lock:
            return dict(self.quote)

    def tick_history(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(self.ticks)

    @property
    def healthy(self) -> bool:
        return self.status == "live" and self.history_ready and time.time() - self.last_msg < 30

    # ------------------------------------------------------------------ worker
    def _run(self):
        backoff = 2
        while True:
            try:
                self._session()
                backoff = 2
            except Exception as exc:  # network error, protocol change, server closed …
                self.error = str(exc)[:200]
            self.status = "reconnecting"
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    def _session(self):
        from websockets.sync.client import connect

        cs, qs = _sid("cs_"), _sid("qs_")
        with connect(WS_URL, additional_headers={"Origin": ORIGIN}, open_timeout=10, close_timeout=2,
                     max_size=2 ** 24) as ws:
            self.connects += 1
            for func, params in [
                ("set_auth_token", ["unauthorized_user_token"]),
                ("quote_create_session", [qs]),
                ("quote_set_fields", [qs, *QUOTE_FIELDS]),
                ("quote_add_symbols", [qs, self.symbol]),
                ("chart_create_session", [cs, ""]),
                ("resolve_symbol", [cs, "sym", "=" + json.dumps({"symbol": self.symbol, "adjustment": "splits"})]),
                ("create_series", [cs, "sds_1", "s1", "sym", self.resolution, self.n_bars, ""]),
            ]:
                ws.send(_frame(func, params))
            self.status, self.error = "loading", ""
            while True:
                try:
                    raw = ws.recv(timeout=40)
                except TimeoutError:
                    raise RuntimeError("no data from TradingView for 40 s")
                self.last_msg = time.time()
                if HEARTBEAT.fullmatch(raw):
                    ws.send(raw)
                    continue
                for part in SPLIT.split(raw):
                    if part.startswith("{"):
                        self._handle(json.loads(part))

    def _handle(self, msg: dict):
        kind = msg.get("m")
        if kind == "qsd":
            values = msg["p"][1].get("v", {})
            with self._lock:
                self.quote.update(values)
                if "lp" in values:
                    now = time.time()
                    self.last_tick = now
                    self.ticks.append((now, float(values["lp"])))
                    cutoff = now - 600
                    while self.ticks and self.ticks[0][0] < cutoff:
                        self.ticks.pop(0)
        elif kind in ("timescale_update", "du"):
            series = msg["p"][1].get("sds_1", {}).get("s", [])
            with self._lock:
                for bar in series:
                    self._bars[bar["v"][0]] = bar["v"]
                if len(self._bars) > self.n_bars + 500:
                    for k in sorted(self._bars)[: len(self._bars) - self.n_bars]:
                        del self._bars[k]
        elif kind == "series_completed":
            self.history_ready, self.status = True, "live"
        elif kind == "symbol_resolved":
            self.info = msg["p"][2]
        elif kind in ("symbol_error", "series_error", "critical_error", "protocol_error"):
            raise RuntimeError(f"TradingView {kind}: {msg.get('p')}")


@st.cache_resource(show_spinner=False, max_entries=6)
def get_stream(symbol: str, resolution: str) -> TVStream:
    return TVStream(symbol, resolution)
