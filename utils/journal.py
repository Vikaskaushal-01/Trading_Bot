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


def _close(t: dict, price: float, status: str):
    pts = (price - t["entry"]) if t["side"] == "BUY" else (t["entry"] - price)
    t.update(status=status, exit=round(price, 5), closed=time.time(), points=round(pts, 5),
             usd=round(pts * t["lot"] * t["ppv"], 4))


def update(symbol: str, bid: float | None, ask: float | None) -> list[dict]:
    """Close open trades whose TP / SL was reached. A BUY exits at bid, a SELL at ask. Returns closed trades."""
    if bid is None or ask is None:
        return []
    closed = []
    with _lock:
        trades = load()
        for t in trades:
            if t["status"] != "open" or t["symbol"] != symbol:
                continue
            tp, sl = t.get("tp"), t.get("sl")
            if t["side"] == "BUY":
                if tp is not None and bid >= tp:
                    _close(t, tp, "TP")
                elif sl is not None and bid <= sl:
                    _close(t, sl, "SL")
            else:
                if tp is not None and ask <= tp:
                    _close(t, tp, "TP")
                elif sl is not None and ask >= sl:
                    _close(t, sl, "SL")
            if t["status"] == "open" and t.get("close_at") and time.time() >= t["close_at"]:
                _close(t, bid if t["side"] == "BUY" else ask, "Time")
            if t["status"] != "open":
                closed.append(t)
        if closed:
            _save(trades)
    return closed


def close_now(trade_id: str, bid: float, ask: float):
    with _lock:
        trades = load()
        for t in trades:
            if t["id"] == trade_id and t["status"] == "open":
                _close(t, bid if t["side"] == "BUY" else ask, "Manual")
        _save(trades)


def delete(trade_id: str):
    with _lock:
        _save([t for t in load() if t["id"] != trade_id])


def today_stats(trades: list[dict], tz) -> dict:
    today = datetime.now(tz).date()
    done = [t for t in trades if t["status"] != "open" and datetime.fromtimestamp(t["closed"], tz).date() == today]
    opened = [t for t in trades if datetime.fromtimestamp(t["opened"], tz).date() == today]
    usd = sum(t["usd"] for t in done)
    wins = sum(1 for t in done if t["points"] > 0)
    return {"trades": len(opened), "closed": len(done), "usd": usd, "points": sum(t["points"] for t in done),
            "wins": wins, "win_rate": wins / len(done) if done else None}
