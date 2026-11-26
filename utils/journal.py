"""Scalp trade journal: trades the user placed at their broker, tracked live against TP / SL.

The app never places orders. When you take a trade at your broker, log it here and the journal follows it
with the live bid/ask, closes it at TP or SL, and keeps today's P&L so the daily risk limits can stop you.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime

from utils.config import DATA_DIR

JOURNAL_FILE = DATA_DIR / "scalp_journal.json"
_lock = threading.Lock()


def load() -> list[dict]:
    try:
        with open(JOURNAL_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save(trades: list[dict]):
    JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = JOURNAL_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(trades[-2000:], f, indent=1)
    tmp.replace(JOURNAL_FILE)


def open_trade(symbol: str, side: str, entry: float, tp: float | None, sl: float | None, lot: float, ppv: float,
               signal_success: float | None = None, close_after: float | None = None) -> dict:
    """tp / sl may be None for a time-exit trade; close_after = seconds until it is closed at market."""
    now = time.time()
    trade = {
        "id": uuid.uuid4().hex[:8], "symbol": symbol, "side": side, "entry": round(entry, 5),
        "tp": None if tp is None else round(tp, 5), "sl": None if sl is None else round(sl, 5),
        "close_at": now + close_after if close_after else None, "lot": lot, "ppv": ppv, "opened": now,
        "status": "open", "exit": None, "closed": None, "points": None, "usd": None,
        "signal_success": signal_success,
    }
    with _lock:
        trades = load()
        trades.append(trade)
        _save(trades)
    return trade
