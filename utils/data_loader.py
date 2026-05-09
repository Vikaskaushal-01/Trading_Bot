import yfinance as yf
import pandas as pd

def load_data(symbol="AAPL", period="1y"):

    df = yf.download(
        symbol,
        period=period,
        auto_adjust=True
    )

    df.reset_index(inplace=True)

    # Fix MultiIndex Columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Convert Close to 1D Series
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Remove NaN
    df.dropna(inplace=True)

    return df