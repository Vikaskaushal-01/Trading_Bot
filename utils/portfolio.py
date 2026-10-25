"""Paper-trading account engine shared by the live and demo accounts.

Accounts are stored as JSON. Cash is always in USD; positions keep their native
currency and are valued in USD using the current FX rate.
"""

import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from utils.config import COMMISSION_RATE, STARTING_BALANCE, asset_name


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _default() -> dict:
    return {
        "cash": STARTING_BALANCE,
        "deposited": STARTING_BALANCE,
        "positions": {},
        "history": [],
        "watchlist": [],
        "orders": [],
    }


def _migrate(raw: dict) -> dict:
    """Convert the old {wallet, transactions} / {demo_wallet, demo_transactions} format."""
    if "cash" in raw:
        data = _default()
        data.update(raw)
        return data

    data = _default()
    data["cash"] = float(raw.get("wallet", raw.get("demo_wallet", STARTING_BALANCE)))
    data["deposited"] = STARTING_BALANCE
    for t in raw.get("transactions", raw.get("demo_transactions", [])):
        sym, qty, price = t["Symbol"], float(t["Quantity"]), float(t["Buy Price"])
        pos = data["positions"].setdefault(sym, {"name": t.get("Asset", sym), "qty": 0.0, "cost": 0.0})
        pos["qty"] += qty
        pos["cost"] += float(t.get("Investment", qty * price))
        data["history"].append({
            "time": "migrated", "side": "BUY", "symbol": sym, "name": pos["name"],
            "qty": qty, "price": price, "amount": float(t.get("Investment", qty * price)),
            "fee": 0.0, "pnl": None,
        })
    # Untracked balance changes (old "Add Money") count as deposits so P&L stays honest.
    invested = sum(p["cost"] for p in data["positions"].values())
    data["deposited"] = max(STARTING_BALANCE, data["cash"] + invested)
    return data


class Account:
    def __init__(self, path: Path, label: str):
        self.path = Path(path)
        self.label = label
        self.data = self._load()

    # ---------- persistence ---------- #

    def _load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            raw = {}
        data = _migrate(raw) if raw else _default()
        if data != raw:
            self._write(data)
        return data

    def _write(self, data: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)  # atomic: never leaves a half-written file

    def save(self):
        self._write(self.data)

    # ---------- accessors ---------- #

    @property
    def cash(self) -> float:
        return self.data["cash"]

    @property
    def positions(self) -> dict:
        return self.data["positions"]

    @property
    def history(self) -> list:
        return self.data["history"]

    @property
    def watchlist(self) -> list:
        return self.data["watchlist"]

    # ---------- actions (return (ok, message)) ---------- #

    def deposit(self, amount: float):
        if amount <= 0:
            return False, "Amount must be positive."
        self.data["cash"] += amount
        self.data["deposited"] += amount
        self._log("DEPOSIT", amount=amount)
        self.save()
        return True, f"Deposited ${amount:,.2f}"

    def withdraw(self, amount: float):
        if amount <= 0:
            return False, "Amount must be positive."
        if amount > self.cash + 1e-9:
            return False, "Insufficient cash."
        self.data["cash"] -= amount
        self.data["deposited"] -= amount
        self._log("WITHDRAW", amount=amount)
        self.save()
        return True, f"Withdrew ${amount:,.2f}"

    def buy(self, symbol: str, usd_amount: float, usd_price: float, name: str | None = None,
            exits: dict | None = None, note: str = ""):
        """Market buy. `exits` uses native-currency prices: {stop, target, trail_pct, ref}."""
        if usd_amount <= 0 or usd_price <= 0:
            return False, "Invalid order."
        fee = usd_amount * COMMISSION_RATE
        if usd_amount + fee > self.cash + 1e-9:
            return False, f"Insufficient cash (need ${usd_amount + fee:,.2f} incl. fee)."
        qty = usd_amount / usd_price
        pos = self.positions.setdefault(symbol, {"name": name or asset_name(symbol), "qty": 0.0, "cost": 0.0})
        pos["qty"] += qty
        pos["cost"] += usd_amount
        self.data["cash"] -= usd_amount + fee
        if exits:
            self._apply_exits(pos, exits)
        self._log("BUY", symbol=symbol, name=pos["name"], qty=qty, price=usd_price, amount=usd_amount, fee=fee, note=note)
        self.save()
        return True, f"Bought {qty:,.6f} {pos['name']} for ${usd_amount:,.2f}"

    def sell(self, symbol: str, qty: float, usd_price: float, note: str = ""):
        pos = self.positions.get(symbol)
        if not pos or qty <= 0:
            return False, "No position to sell."
        qty = min(qty, pos["qty"])
        avg_cost = pos["cost"] / pos["qty"]
        proceeds = qty * usd_price
        fee = proceeds * COMMISSION_RATE
        pnl = proceeds - fee - qty * avg_cost
        pos["qty"] -= qty
        pos["cost"] -= qty * avg_cost
        if pos["qty"] <= 1e-9:
            del self.positions[symbol]
        self.data["cash"] += proceeds - fee
        self._log("SELL", symbol=symbol, name=pos["name"], qty=qty, price=usd_price, amount=proceeds, fee=fee, pnl=pnl, note=note)
        self.save()
        reason = f"[{note}] " if note else ""
        return True, f"{reason}Sold {qty:,.6f} {pos['name']} for ${proceeds:,.2f} (P&L ${pnl:+,.2f})"

    # ---------- risk exits & pending orders ---------- #

    @staticmethod
    def _apply_exits(pos: dict, exits: dict):
        for key in ("stop", "target", "trail_pct"):
            if key in exits:
                pos[key] = exits[key] or None
        if exits.get("ref"):
            pos["high"] = max(pos.get("high") or 0, exits["ref"])

    def set_exits(self, symbol: str, stop=None, target=None, trail_pct=None, ref=None):
        pos = self.positions.get(symbol)
        if not pos:
            return False, "No such position."
        self._apply_exits(pos, {"stop": stop, "target": target, "trail_pct": trail_pct, "ref": ref})
        self.save()
        return True, f"Exit rules updated for {pos['name']}"

    @property
    def orders(self) -> list:
        return self.data.setdefault("orders", [])

    def place_limit(self, side: str, symbol: str, name: str, limit: float, currency: str,
                    amount: float | None = None, qty: float | None = None, exits: dict | None = None):
        """Queue a limit order: BUY fills when price <= limit, SELL when price >= limit."""
        if limit <= 0 or (side == "BUY" and not amount) or (side == "SELL" and not qty):
            return False, "Invalid limit order."
        if side == "SELL" and symbol not in self.positions:
            return False, "No position to sell."
        self.orders.append({
            "id": uuid.uuid4().hex[:8], "time": _now(), "side": side, "symbol": symbol, "name": name,
            "limit": limit, "currency": currency, "amount": amount, "qty": qty, "exits": exits or {},
        })
        self.save()
        return True, f"Limit {side.lower()} placed for {name} at {limit:,.2f} {currency}"

    def cancel_order(self, order_id: str):
        self.data["orders"] = [o for o in self.orders if o["id"] != order_id]
        self.save()
        return True, "Order cancelled"

    def process(self, native_prices: dict, fx: dict) -> list[str]:
        """Fill triggered limit orders and fire stop-loss / take-profit / trailing exits.

        native_prices: symbol -> latest price in the asset's own currency
        fx: symbol -> multiplier into USD
        """
        events, dirty = [], False
        for order in list(self.orders):
            sym, price = order["symbol"], native_prices.get(order["symbol"])
            if price is None:
                continue
            usd = price * fx.get(sym, 1.0)
            if order["side"] == "BUY" and price <= order["limit"]:
                self.data["orders"].remove(order)
                amount = min(order["amount"], self.cash / (1 + COMMISSION_RATE))
                if amount < 1.0:
                    ok, msg = False, "insufficient cash"
                else:
                    ok, msg = self.buy(sym, amount, usd, order["name"], exits={**order["exits"], "ref": price}, note="Limit")
                events.append(msg if ok else f"Limit buy for {order['name']} cancelled: {msg}")
                dirty = True
            elif order["side"] == "SELL" and price >= order["limit"]:
                self.data["orders"].remove(order)
                if sym in self.positions:
                    events.append(self.sell(sym, min(order["qty"], self.positions[sym]["qty"]), usd, note="Limit")[1])
                dirty = True

        for sym, pos in list(self.positions.items()):
            price = native_prices.get(sym)
            if price is None:
                continue
            usd = price * fx.get(sym, 1.0)
            trail_stop = None
            if pos.get("trail_pct"):
                new_high = max(pos.get("high") or price, price)
                dirty |= new_high != pos.get("high")
                pos["high"] = new_high
                trail_stop = new_high * (1 - pos["trail_pct"] / 100)
            stop = max([x for x in (pos.get("stop"), trail_stop) if x], default=None)
            if stop and price <= stop:
                reason = "Trailing stop" if trail_stop and stop == trail_stop else "Stop-loss"
                events.append(self.sell(sym, pos["qty"], usd, note=reason)[1])
            elif pos.get("target") and price >= pos["target"]:
                events.append(self.sell(sym, pos["qty"], usd, note="Take-profit")[1])
        if dirty and not events:
            self.save()
        return events

    def reset(self, balance: float = STARTING_BALANCE):
        self.data = _default()
        self.data["cash"] = self.data["deposited"] = balance
        self.save()

    def toggle_watch(self, symbol: str):
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
        else:
            self.watchlist.append(symbol)
        self.save()

    def _log(self, side, symbol="", name="", qty=0.0, price=0.0, amount=0.0, fee=0.0, pnl=None, note=""):
        self.history.append({
            "time": _now(), "side": side, "symbol": symbol, "name": name,
            "qty": qty, "price": price, "amount": amount, "fee": fee, "pnl": pnl, "note": note,
        })

    # ---------- valuation ---------- #

    def valuation(self, usd_prices: dict) -> dict:
        """usd_prices: symbol -> current USD price. Missing prices fall back to cost."""
        rows, holdings = [], 0.0
        for sym, pos in self.positions.items():
            avg = pos["cost"] / pos["qty"]
            price = usd_prices.get(sym, avg)
            value = pos["qty"] * price
            holdings += value
            rows.append({
                "symbol": sym, "name": pos["name"], "qty": pos["qty"], "avg_price": avg,
                "price": price, "value": value, "cost": pos["cost"],
                "pnl": value - pos["cost"], "pnl_pct": (price / avg - 1) * 100,
                "stop": pos.get("stop"), "target": pos.get("target"), "trail_pct": pos.get("trail_pct"),
            })
        equity = self.cash + holdings
        realized = sum(t["pnl"] or 0 for t in self.history if t["side"] == "SELL")
        return {
            "cash": self.cash, "holdings": holdings, "equity": equity,
            "deposited": self.data["deposited"],
            "total_pnl": equity - self.data["deposited"],
            "total_pnl_pct": (equity / self.data["deposited"] - 1) * 100 if self.data["deposited"] else 0,
            "unrealized": sum(r["pnl"] for r in rows),
            "realized": realized,
            "rows": sorted(rows, key=lambda r: -r["value"]),
        }
