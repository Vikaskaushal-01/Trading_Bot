"""Feature engineering for the prediction models.

Every feature is scale-free (ratios, percentages, oscillators) so one model can
learn from a $20 stock and an $80,000 coin alike, and every feature only uses
information available at that day's close (no look-ahead).
"""

import numpy as np
import pandas as pd
import ta

# name -> (group, plain-English meaning)
FEATURE_INFO = {
    # Momentum: is price accelerating up or down?
    "ret_1": ("Momentum", "1-day return"),
    "ret_3": ("Momentum", "3-day return"),
    "ret_5": ("Momentum", "1-week return"),
    "ret_10": ("Momentum", "2-week return"),
    "ret_20": ("Momentum", "1-month return"),
    "ret_60": ("Momentum", "3-month return"),
    "ret_120": ("Momentum", "6-month return"),
    "rsi_7": ("Momentum", "Fast RSI (7) — short-term overbought/oversold"),
    "rsi_14": ("Momentum", "RSI (14) — classic overbought >70 / oversold <30"),
    "stoch_k": ("Momentum", "Stochastic %K — close vs 14-day range"),
    "stoch_d": ("Momentum", "Stochastic %D — smoothed %K"),
    "williams_r": ("Momentum", "Williams %R — distance from 14-day high"),
    "cci_20": ("Momentum", "Commodity Channel Index — deviation from typical price"),
    "macd_n": ("Momentum", "MACD line as % of price"),
    "macd_hist_n": ("Momentum", "MACD histogram as % of price"),
    "macd_hist_chg": ("Momentum", "3-day change of MACD histogram (momentum turning)"),
    # Trend: which way and how strongly is price trending?
    "dist_ema20": ("Trend", "Distance from 20-day EMA"),
    "dist_ema50": ("Trend", "Distance from 50-day EMA"),
    "dist_sma200": ("Trend", "Distance from 200-day SMA (long-term trend)"),
    "ema20_50": ("Trend", "EMA20 vs EMA50 spread (golden/death cross state)"),
    "ema20_slope": ("Trend", "5-day slope of the 20-day EMA"),
    "adx": ("Trend", "ADX — trend strength (>25 = trending)"),
    "di_diff": ("Trend", "+DI minus −DI — trend direction"),
    "trend_days": ("Trend", "Share of last 20 days closing above EMA20"),
    # Volatility: how wild is the price action?
    "atr_pct": ("Volatility", "Average True Range as % of price"),
    "vol_10": ("Volatility", "10-day annualised volatility"),
    "vol_20": ("Volatility", "20-day annualised volatility"),
    "vol_regime": ("Volatility", "10-day vs 60-day volatility (expanding or calming)"),
    "bb_pct": ("Volatility", "Bollinger %B — position inside the bands"),
    "bb_width": ("Volatility", "Bollinger band width (squeeze detector)"),
    "range_pct": ("Volatility", "Today's high-low range as % of price"),
    "gap": ("Volatility", "Overnight gap (open vs previous close)"),
    # Volume: is money flowing in or out?
    "vol_z": ("Volume", "Volume z-score vs 20-day average"),
    "obv_slope": ("Volume", "On-Balance Volume 10-day slope (accumulation)"),
    "mfi_14": ("Volume", "Money Flow Index — volume-weighted RSI"),
    "cmf_20": ("Volume", "Chaikin Money Flow — buying vs selling pressure"),
    "up_vol_ratio": ("Volume", "Share of 20-day volume on up days"),
    # Price structure: where are we relative to key levels?
    "dist_52w_high": ("Price structure", "Distance below the 52-week high"),
    "dist_52w_low": ("Price structure", "Distance above the 52-week low"),
    "donchian_pos": ("Price structure", "Position in the 20-day high/low channel"),
    "body": ("Price structure", "Candle body (close−open) vs range"),
    "upper_wick": ("Price structure", "Upper wick vs range (selling into strength)"),
    "lower_wick": ("Price structure", "Lower wick vs range (buying the dip)"),
    "up_days_10": ("Price structure", "Share of up days in the last 10"),
    # Market context: what is the broader market doing?
    "mkt_ret_1": ("Market context", "Benchmark 1-day return"),
    "mkt_ret_5": ("Market context", "Benchmark 1-week return"),
    "mkt_ret_20": ("Market context", "Benchmark 1-month return"),
    "mkt_trend": ("Market context", "Benchmark vs its 50-day average (risk-on/off)"),
    "rel_strength_20": ("Market context", "1-month return minus benchmark's (relative strength)"),
    "beta_60": ("Market context", "60-day beta to the benchmark"),
    "corr_60": ("Market context", "60-day correlation with the benchmark"),
    "vix": ("Market context", "VIX fear index level"),
    "vix_chg_5": ("Market context", "1-week change in VIX"),
    # Calendar
    "dow": ("Calendar", "Day of week"),
    "month": ("Calendar", "Month of year"),
}

FEATURE_GROUPS = list(dict.fromkeys(g for g, _ in FEATURE_INFO.values()))
DEFAULT_GROUPS = [g for g in FEATURE_GROUPS if g != "Calendar"]


def benchmark_for(symbol: str, currency: str) -> str:
    if symbol.endswith("-USD"):
        return "^IXIC" if symbol == "BTC-USD" else "BTC-USD"
    if currency == "INR" or symbol.endswith((".NS", ".BO")):
        return "^NSEI"
    return "^GSPC"


def _align(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Reindex onto the asset's calendar using only past values (forward fill)."""
    s = series[~series.index.duplicated()].sort_index()
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def compute_features(df: pd.DataFrame, market: pd.DataFrame | None = None,
                     vix: pd.DataFrame | None = None) -> pd.DataFrame:
    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"].astype(float)
    r = c.pct_change()
    f = pd.DataFrame(index=df.index)

    # ---- momentum ---- #
    for n in (1, 3, 5, 10, 20, 60, 120):
        f[f"ret_{n}"] = c.pct_change(n)
    f["rsi_7"] = ta.momentum.RSIIndicator(c, 7).rsi()
    f["rsi_14"] = ta.momentum.RSIIndicator(c, 14).rsi()
    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    f["stoch_k"], f["stoch_d"] = stoch.stoch(), stoch.stoch_signal()
    f["williams_r"] = ta.momentum.WilliamsRIndicator(h, l, c, lbp=14).williams_r()
    f["cci_20"] = ta.trend.CCIIndicator(h, l, c, window=20).cci()
    macd = ta.trend.MACD(c)
    f["macd_n"] = macd.macd() / c
    f["macd_hist_n"] = macd.macd_diff() / c
    f["macd_hist_chg"] = f["macd_hist_n"].diff(3)

    # ---- trend ---- #
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    sma200 = c.rolling(200).mean()
    f["dist_ema20"] = c / ema20 - 1
    f["dist_ema50"] = c / ema50 - 1
    f["dist_sma200"] = c / sma200 - 1
    f["ema20_50"] = ema20 / ema50 - 1
    f["ema20_slope"] = ema20.pct_change(5)
    adx = ta.trend.ADXIndicator(h, l, c, window=14)
    f["adx"] = adx.adx()
    f["di_diff"] = adx.adx_pos() - adx.adx_neg()
    f["trend_days"] = (c > ema20).astype(float).rolling(20).mean()

    # ---- volatility ---- #
    atr = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    f["atr_pct"] = atr / c
    f["vol_10"] = r.rolling(10).std() * np.sqrt(252)
    f["vol_20"] = r.rolling(20).std() * np.sqrt(252)
    f["vol_regime"] = r.rolling(10).std() / r.rolling(60).std()
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    f["bb_pct"] = bb.bollinger_pband()
    f["bb_width"] = bb.bollinger_wband() / 100
    f["range_pct"] = (h - l) / c
    f["gap"] = o / c.shift(1) - 1

    # ---- volume (skipped automatically when an asset reports no volume) ---- #
    vol_mean, vol_std = v.rolling(20).mean(), v.rolling(20).std()
    f["vol_z"] = (v - vol_mean) / vol_std.replace(0, np.nan)
    obv = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    f["obv_slope"] = (obv - obv.shift(10)) / (vol_mean * 10).replace(0, np.nan)
    f["mfi_14"] = ta.volume.MFIIndicator(h, l, c, v, window=14).money_flow_index()
    f["cmf_20"] = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v, window=20).chaikin_money_flow()
    up_vol = v.where(r > 0, 0.0).rolling(20).sum()
    f["up_vol_ratio"] = up_vol / v.rolling(20).sum().replace(0, np.nan)

    # ---- price structure ---- #
    f["dist_52w_high"] = c / h.rolling(252, min_periods=120).max() - 1
    f["dist_52w_low"] = c / l.rolling(252, min_periods=120).min() - 1
    hi20, lo20 = h.rolling(20).max(), l.rolling(20).min()
    f["donchian_pos"] = (c - lo20) / (hi20 - lo20).replace(0, np.nan)
    rng = (h - l).replace(0, np.nan)
    f["body"] = (c - o) / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    f["up_days_10"] = (r > 0).astype(float).rolling(10).mean()

    # ---- market context ---- #
    if market is not None and len(market) > 60:
        mc = _align(market["Close"], df.index)
        mr = mc.pct_change()
        f["mkt_ret_1"], f["mkt_ret_5"], f["mkt_ret_20"] = mr, mc.pct_change(5), mc.pct_change(20)
        f["mkt_trend"] = mc / mc.rolling(50).mean() - 1
        f["rel_strength_20"] = f["ret_20"] - f["mkt_ret_20"]
        f["beta_60"] = r.rolling(60).cov(mr) / mr.rolling(60).var()
        f["corr_60"] = r.rolling(60).corr(mr)
    if vix is not None and len(vix) > 60:
        vc = _align(vix["Close"], df.index)
        f["vix"], f["vix_chg_5"] = vc, vc.pct_change(5)

    # ---- calendar ---- #
    f["dow"] = df.index.dayofweek
    f["month"] = df.index.month

    return f.replace([np.inf, -np.inf], np.nan)


def usable_columns(features: pd.DataFrame, groups: list[str], warmup: int = 260) -> list[str]:
    """Columns in the chosen groups with enough real data (drops e.g. volume features for FX)."""
    body = features.iloc[warmup:] if len(features) > warmup + 100 else features
    return [
        col for col, (group, _) in FEATURE_INFO.items()
        if group in groups and col in features and body[col].notna().mean() > 0.95
        and body[col].nunique() > 2
    ]
