import ta

def add_indicators(df):
    df['RSI']=ta.momentum.RSIIndicator(df['Close']).rsi()
    df['EMA']=ta.trend.EMAIndicator(df['Close']).ema_indicator()
    df['MACD']=ta.trend.MACD(df['Close']).macd()
    return df