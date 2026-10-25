"""Market data access via yfinance, cached so reruns don't hammer the API."""

import pandas as pd
import streamlit as st
import yfinance as yf

from utils.config import UNIVERSE, CURRENCY_SYMBOLS

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[[c for c in OHLCV if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def get_history(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """OHLCV history for one symbol (empty frame on failure)."""
    try:
        return _clean(yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True))
    except Exception:
        return pd.DataFrame(columns=OHLCV)


@st.cache_data(ttl=60, show_spinner=False)
def get_quotes(symbols: tuple) -> dict:
    """Latest price, daily change and a 1-month sparkline for many symbols in one request."""
    quotes = {}
    if not symbols:
        return quotes
    try:
        raw = yf.download(
            list(symbols), period="1mo", interval="1d", group_by="ticker",
            auto_adjust=True, threads=True, progress=False,
        )
    except Exception:
        raw = pd.DataFrame()

    for sym in symbols:
        try:
            df = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
            closes = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if len(closes) < 2:
                continue
            price, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
            vols = pd.to_numeric(df["Volume"], errors="coerce").dropna()
            volume = float(vols.iloc[-1]) if len(vols) else 0.0
            quotes[sym] = {
                "symbol": sym,
                "name": UNIVERSE.get(sym, (sym,))[0],
                "price": price,
                "change": price - prev,
                "change_pct": (price - prev) / prev * 100,
                "volume": volume,
                "spark": closes.round(4).tolist(),
                "currency": get_currency(sym),
            }
        except Exception:
            continue
    return quotes


@st.cache_data(ttl=15, show_spinner=False)
def get_live_price(symbol: str) -> float | None:
    """Most recent traded price (native currency); used for order execution."""
    try:
        price = yf.Ticker(symbol).fast_info["lastPrice"]
        if price and price > 0:
            return float(price)
    except Exception:
        pass
    hist = get_history(symbol, period="5d")
    return float(hist["Close"].iloc[-1]) if not hist.empty else None


@st.cache_data(ttl=86400, show_spinner=False)
def get_currency(symbol: str) -> str:
    if symbol in UNIVERSE:
        return UNIVERSE[symbol][2]
    if symbol.endswith((".NS", ".BO")):
        return "INR"
    try:
        return (yf.Ticker(symbol).fast_info.get("currency") or "USD").upper()
    except Exception:
        return "USD"


@st.cache_data(ttl=900, show_spinner=False)
def fx_to_usd(currency: str) -> float:
    """Multiplier converting one unit of `currency` into USD."""
    if currency == "USD":
        return 1.0
    hist = get_history(f"{currency}USD=X", period="5d")
    if not hist.empty:
        return float(hist["Close"].iloc[-1])
    inverse = get_history(f"USD{currency}=X", period="5d")
    if not inverse.empty:
        return 1.0 / float(inverse["Close"].iloc[-1])
    return 1.0


def usd_price(symbol: str, native_price: float) -> float:
    return native_price * fx_to_usd(get_currency(symbol))


def fmt_price(value: float, currency: str = "USD") -> str:
    sym = CURRENCY_SYMBOLS.get(currency, currency + " ")
    decimals = 2 if abs(value) >= 1 else 4
    return f"{sym}{value:,.{decimals}f}"


@st.cache_data(ttl=3600, show_spinner=False)
def validate_symbol(symbol: str) -> bool:
    return not get_history(symbol, period="5d").empty
