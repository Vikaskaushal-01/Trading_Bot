"""Technical indicators and a rule-based technical rating."""

import pandas as pd
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with indicator columns added (NaN warm-up rows kept)."""
    df = df.copy()
    if len(df) < 2:
        return df
    close, high, low = df["Close"], df["High"], df["Low"]

    df["EMA20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["SMA200"] = ta.trend.SMAIndicator(close, window=200).sma_indicator()
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_pct"] = bb.bollinger_pband()

    df["ATR"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    return df


def technical_rating(df: pd.DataFrame) -> dict:
    """Score the latest bar from -100 (strong sell) to +100 (strong buy) with the reasons."""
    last = df.iloc[-1]
    checks = []

    def vote(cond_buy, cond_sell, label):
        if pd.isna(cond_buy) or pd.isna(cond_sell):
            return
        checks.append((1 if cond_buy else -1 if cond_sell else 0, label))

    close = last["Close"]
    vote(close > last.get("EMA20"), close < last.get("EMA20"), "Price vs EMA20")
    vote(close > last.get("EMA50"), close < last.get("EMA50"), "Price vs EMA50")
    if not pd.isna(last.get("SMA200")):
        vote(close > last["SMA200"], close < last["SMA200"], "Price vs SMA200")
    vote(last.get("EMA20") > last.get("EMA50"), last.get("EMA20") < last.get("EMA50"), "EMA20 vs EMA50")
    vote(last.get("MACD") > last.get("MACD_signal"), last.get("MACD") < last.get("MACD_signal"), "MACD crossover")
    rsi = last.get("RSI")
    vote(rsi < 30, rsi > 70, "RSI overbought/oversold")
    bbp = last.get("BB_pct")
    vote(bbp < 0.05, bbp > 0.95, "Bollinger band position")

    if not checks:
        return {"score": 0, "label": "Neutral", "checks": []}
    score = round(sum(v for v, _ in checks) / len(checks) * 100)
    if score >= 50:
        label = "Strong Buy"
    elif score >= 15:
        label = "Buy"
    elif score <= -50:
        label = "Strong Sell"
    elif score <= -15:
        label = "Sell"
    else:
        label = "Neutral"
    return {"score": score, "label": label, "checks": checks}
