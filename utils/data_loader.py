import yfinance as yf

def load_data(symbol="AAPL",period="1y"):
    df=yf.download(symbol,period=period)
    df.reset_index(inplace=True)
    return df