"""Long-only strategy backtester on daily bars with stop-loss / take-profit / trailing stops.

Signals are computed on a day's close and acted on at that close; protective
exits are checked intrabar against the next days' low/high (gaps fill at the open).
After a protective exit the position re-opens once the cooldown has passed, if the
strategy is still long.
"""

import numpy as np
import pandas as pd

from utils.indicators import add_indicators

STRATEGIES = {
    "SMA Crossover": "Long while the fast moving average is above the slow one.",
    "RSI Mean Reversion": "Buy when RSI drops below the lower band, exit above the upper band.",
    "MACD Crossover": "Long while MACD is above its signal line.",
    "Bollinger Breakout": "Buy a close above the upper band, exit on a close below the middle band.",
    "Donchian Breakout": "Turtle-style: buy a close above the N-day high, exit below the M-day low.",
    "AI Model": "Uses the AI's out-of-sample probabilities: exit on a SELL signal, re-enter on a BUY.",
}


def _hold_until(entries: pd.Series, exits: pd.Series, start: float = 0.0) -> pd.Series:
    """Turn entry/exit event flags into a 0/1 position series."""
    state = pd.Series(np.nan, index=entries.index)
    state[entries] = 1.0
    state[exits & ~entries] = 0.0
    return state.ffill().fillna(start)


def signals(df: pd.DataFrame, strategy: str, params: dict) -> pd.Series:
    close = df["Close"]
    if strategy == "SMA Crossover":
        fast = close.rolling(params.get("fast", 20)).mean()
        slow = close.rolling(params.get("slow", 50)).mean()
        return (fast > slow).astype(float)
    if strategy == "RSI Mean Reversion":
        rsi = df["RSI"]
        return _hold_until(rsi < params.get("lower", 30), rsi > params.get("upper", 70))
    if strategy == "MACD Crossover":
        return (df["MACD"] > df["MACD_signal"]).astype(float)
    if strategy == "Bollinger Breakout":
        mid = (df["BB_upper"] + df["BB_lower"]) / 2
        return _hold_until(close > df["BB_upper"], close < mid)
    if strategy == "Donchian Breakout":
        upper = df["High"].rolling(params.get("entry", 20)).max().shift(1)
        lower = df["Low"].rolling(params.get("exit", 10)).min().shift(1)
        return _hold_until(close > upper, close < lower)
    if strategy == "AI Model":
        proba = params["proba"].reindex(df.index)
        pos = _hold_until(proba >= params["buy_level"], proba <= params["sell_level"], start=1.0)
        return pos.where(proba.notna().cummax(), 0.0)  # flat before the first out-of-sample day
    raise ValueError(strategy)


def simulate(df: pd.DataFrame, desired: pd.Series, fee: float = 0.001,
             stop_pct: float | None = None, take_pct: float | None = None, trail_pct: float | None = None,
             cooldown: int = 5):
    """cooldown: days to wait after a protective exit before re-entering if the strategy is still long
    (0 = wait until the strategy itself exits and re-signals)."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    want = desired.to_numpy(float)
    n = len(c)
    ret, held = np.zeros(n), np.zeros(n)
    trades, round_trips = [], []
    in_pos, blocked_until = False, -1
    entry = peak = 0.0
    idx = df.index

    def close_trip(t, price, reason):
        trades.append({"date": idx[t].date(), "action": "SELL", "price": price, "reason": reason})
        round_trips.append(price / entry * (1 - fee) ** 2 - 1)

    for t in range(1, n):
        signal = want[t - 1]
        if signal == 0:
            blocked_until = -1  # strategy reset: free to take its next entry
        if not in_pos and signal == 1 and t > blocked_until:
            in_pos, entry, peak = True, c[t - 1], c[t - 1]
            ret[t] -= fee
            trades.append({"date": idx[t - 1].date(), "action": "BUY", "price": c[t - 1], "reason": "Signal"})
        elif in_pos and signal == 0:
            in_pos = False
            ret[t] -= fee
            close_trip(t - 1, c[t - 1], "Signal")

        if not in_pos:
            continue
        held[t] = 1
        fixed = entry * (1 - stop_pct / 100) if stop_pct else None
        trailing = peak * (1 - trail_pct / 100) if trail_pct else None
        stop = max([x for x in (fixed, trailing) if x is not None], default=None)
        target = entry * (1 + take_pct / 100) if take_pct else None

        exit_px, reason = None, ""
        if stop is not None and l[t] <= stop:
            exit_px = min(o[t], stop)
            reason = "Trailing stop" if trailing is not None and stop == trailing else "Stop-loss"
        elif target is not None and h[t] >= target:
            exit_px, reason = max(o[t], target), "Take-profit"

        if exit_px is not None:
            ret[t] += exit_px / c[t - 1] - 1 - fee
            in_pos = False
            blocked_until = t + cooldown if cooldown > 0 else n
            close_trip(t, exit_px, reason)
        else:
            ret[t] += c[t] / c[t - 1] - 1
            peak = max(peak, c[t])

    if in_pos:
        round_trips.append(c[-1] / entry * (1 - fee) - 1)  # mark open trade to market
    return pd.Series(ret, index=idx), pd.Series(held, index=idx), pd.DataFrame(trades), np.array(round_trips)


def metrics(equity: pd.Series, returns: pd.Series, held: pd.Series, trips: np.ndarray | None = None) -> dict:
    years = max(len(equity), 1) / 252
    final = float(equity.iloc[-1])
    vol = returns.std() * np.sqrt(252)
    downside = returns[returns < 0].std() * np.sqrt(252)
    drawdown = float((equity / equity.cummax() - 1).min())
    cagr = final ** (1 / years) - 1 if years > 0 and final > 0 else 0.0
    out = {
        "Total Return": final - 1, "CAGR": cagr,
        "Sharpe": returns.mean() * 252 / vol if vol > 0 else 0.0,
        "Sortino": returns.mean() * 252 / downside if downside > 0 else 0.0,
        "Calmar": cagr / abs(drawdown) if drawdown < 0 else 0.0,
        "Max Drawdown": drawdown, "Volatility": vol, "Exposure": float(held.mean()),
        "Trades": 0, "Win Rate": None, "Profit Factor": None, "Avg Win": None, "Avg Loss": None,
    }
    if trips is not None and len(trips):
        wins, losses = trips[trips > 0], trips[trips <= 0]
        out.update({
            "Trades": len(trips), "Win Rate": len(wins) / len(trips),
            "Profit Factor": wins.sum() / abs(losses.sum()) if losses.sum() < 0 else float("inf"),
            "Avg Win": wins.mean() if len(wins) else 0.0, "Avg Loss": losses.mean() if len(losses) else 0.0,
        })
    return out


def run_backtest(hist: pd.DataFrame, strategy: str, params: dict, fee: float = 0.001,
                 stop_pct: float | None = None, take_pct: float | None = None, trail_pct: float | None = None,
                 cooldown: int = 5) -> dict:
    df = add_indicators(hist).dropna(subset=["Close"])
    daily = df["Close"].pct_change().fillna(0.0)
    desired = signals(df, strategy, params).fillna(0.0)

    strat_ret, held, trades, trips = simulate(df, desired, fee, stop_pct, take_pct, trail_pct, cooldown)
    equity = (1 + strat_ret).cumprod()
    hold_equity = (1 + daily).cumprod()
    return {
        "df": df,
        "equity": equity,
        "hold_equity": hold_equity,
        "strategy": metrics(equity, strat_ret, held, trips),
        "hold": metrics(hold_equity, daily, pd.Series(1.0, index=daily.index)),
        "trades": trades,
        "trips": trips,
    }
