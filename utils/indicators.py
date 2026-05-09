import ta
import pandas as pd

def add_indicators(df):

    close_series = pd.Series(df["Close"]).squeeze()

    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(
        close=close_series
    ).rsi()

    # EMA
    df["EMA"] = ta.trend.EMAIndicator(
        close=close_series
    ).ema_indicator()

    # MACD
    df["MACD"] = ta.trend.MACD(
        close=close_series
    ).macd()

    df.dropna(inplace=True)

    return df